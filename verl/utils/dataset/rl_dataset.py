# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional, Union, List, Dict, Any

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

        self.sample_states: Dict[str, Dict[str, Any]] = {}
        self.min_buffer_length = self.config.get("min_buffer_length", 8)
        self.reward_buffer_length = self.config.get("reward_buffer_length", 16)
        self.epoch_estimation = self.config.get("epoch_estimation", True)
        self.robust_epoch_estimation = self.config.get("robust_epoch_estimation", True)

    def update_sample_state(
        self, 
        index: Union[str, int], 
        rewards: Union[List[float], np.ndarray]
    ) -> None:
        """
        Update the sampling state for a given index with new rewards.
        
        Maintains an exponential moving average (EMA) of success rates and
        tracks historical reward information for adaptive sampling.
        
        Args:
            index: Unique identifier for the sample (will be converted to string)
            rewards: Array of reward values (must be -1, 0, or 1)
        """
        # Normalize rewards: convert -1 to 0, keep 0 and 1 as is
        normalized_rewards = []
        for reward in rewards:
            assert reward in {-1, 0, 1}, f"Reward must be -1, 0, or 1, got {reward}"
            if reward == -1:
                normalized_rewards.append(0)
            else:
                normalized_rewards.append(reward)
    
        # Ensure index is a string
        if not isinstance(index, str):
            index = str(index)
    
        # Initialize state if this is a new index
        if index not in self.sample_states:
            self.sample_states[index] = {
                'rewards': [],
                'ema': 0.0,
                'count': 0,
                'success_rate': 0.0,
                'last_count': 0,
                'last_success_rate': 0.0,
                'last_ema': 0.0,
            }
        
        # Save previous state
        self.sample_states[index]['last_count'] = self.sample_states[index]['count']
        self.sample_states[index]['last_success_rate'] = self.sample_states[index]['success_rate']
        self.sample_states[index]['last_ema'] = self.sample_states[index]['ema']

        # Update with new rewards
        self.sample_states[index]['count'] += len(normalized_rewards)
        self.sample_states[index]['rewards'].append(normalized_rewards)
        self.sample_states[index]['success_rate'] = np.mean(normalized_rewards)

        # Calculate EMA based on configuration
        if self.epoch_estimation:
            # Use current epoch's success rate
            epoch_success_rate = np.mean(normalized_rewards)
            new_ema = epoch_success_rate

            # If current batch is small and robust estimation is enabled,
            # use historical data to stabilize the estimate
            if len(normalized_rewards) < self.reward_buffer_length and self.robust_epoch_estimation:
                flattened_rewards = []
                for reward_list in reversed(self.sample_states[index]['rewards']):
                    flattened_rewards.extend(reversed(reward_list))
                    if len(flattened_rewards) >= self.reward_buffer_length:
                        break

                new_ema = np.mean(flattened_rewards[:self.reward_buffer_length])
        else:
            # Use simple moving average over recent rewards
            flattened_rewards = []
            for reward_list in reversed(self.sample_states[index]['rewards']):
                flattened_rewards.extend(reversed(reward_list))
                if len(flattened_rewards) >= self.reward_buffer_length:
                    break

            simple_ema = np.mean(flattened_rewards[:self.reward_buffer_length])
            new_ema = simple_ema
                        
        # Update EMA
        self.sample_states[index]['ema'] = new_ema

    def get_sample_state(self, index: Union[str, int]) -> List[Union[int, float]]:
        """
        Get the current sampling state for a given index.
        
        Args:
            index: Unique identifier for the sample
        
        Returns:
            List containing [is_buffer_full, ema, total_count]
            - is_buffer_full: 1 if min buffer length reached, 0 otherwise
            - ema: Exponential moving average of success rate
            - total_count: Total number of samples processed
        """
        # Ensure index is a string
        if isinstance(index, int):
            index = str(index)
        if not isinstance(index, str):
            print(f"[Warning] Index should be string type, got {type(index)}")
    
        # Return default state if index not found
        if index not in self.sample_states:
            is_full = 1 if self.min_buffer_length == 0 else 0
            ema = 0.0
            return [is_full, ema, 0]
        
        state = self.sample_states[index]
        
        # Check if we have enough samples in the buffer
        is_buffer_full = 1 if state['count'] >= self.min_buffer_length else 0

        ema = state['ema']
        return [is_buffer_full, ema, state['count']]

    def resume_sample_states(self, sample_states: Dict[str, Dict[str, Any]]) -> None:
        """
        Resume training by restoring sample states from a checkpoint.
        
        Replays all historical rewards to rebuild the internal state,
        ensuring consistency with the saved checkpoint.
        
        Args:
            sample_states: Dictionary mapping indices to their saved states
        """
        for index in list(sample_states.keys()):
            all_reward_lists = sample_states[index]['rewards']

            # Replay all reward updates to rebuild state
            for reward_list in all_reward_lists:
                self.update_sample_state(index, reward_list)


    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        max_size = self.config.get("max_size", None)
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            if max_size is not None and len(dataframe) > max_size:
                dataframe = dataframe.shuffle(seed=42).select(range(max_size))
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        state_dict = self.get_sample_state(index)
        row_dict["reward_states"] = state_dict
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def on_batch_end(self, batch: Any) -> None:
        """
        Callback executed at the end of each batch to update sample states.
        
        Extracts rewards from the batch and updates the corresponding
        sample states for budget allocation in future epochs.
        
        Args:
            batch: Batch object containing indices and reward information
        """
        index_to_rewards = defaultdict(list)
        
        reward_tensor = batch.batch["token_level_scores"]
        # Group rewards by index
        for i in range(len(batch.non_tensor_batch['index'])):
            reward_sum = reward_tensor[i].sum().item()
            index_to_rewards[batch.non_tensor_batch['index'][i]].append(reward_sum)
        
        # Update states for all indices in this batch
        for index, reward_list in index_to_rewards.items():
            self.update_sample_state(index, reward_list)