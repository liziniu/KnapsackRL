# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val


def compute_coverage_metric(
    batch: DataProto,
) -> Dict[str, float]:
    """
    Compute coverage and effectiveness metrics for a training batch.
    
    This function analyzes how well the training covers different prompts and
    calculates the effectiveness of the gradient signals based on reward diversity.
    
    Coverage metrics measure:
    - How many prompts have at least one positive/negative sample
    - What proportion of prompts have only positive or only negative samples
    - How many prompts have diverse rewards (effective for learning)
    
    Difficulty metrics categorize prompts by success rate:
    - Extremely easy: success_rate == 1.0
    - Easy: 0.8 <= success_rate < 1.0
    - Medium: 0.2 < success_rate < 0.8
    - Hard: 0.0 < success_rate <= 0.2
    - Extremely hard: success_rate == 0.0
    
    Args:
        batch: DataProto object containing batch data with token_level_rewards and uids
    
    Returns:
        Dictionary containing all computed metrics with float values
    """
    # Extract rewards and compute total scores per sample
    scores = batch.batch["token_level_rewards"].sum(dim=-1)
    batch_size = scores.shape[0]

    # Map unique identifiers to indices
    uids = batch.non_tensor_batch["uid"]
    uid_to_index = {}
    num_unique_prompts = 0
    for uid in uids:
        if uid not in uid_to_index:
            uid_to_index[uid] = num_unique_prompts
            num_unique_prompts += 1
    
    # Group scores by prompt
    index_to_scores = defaultdict(list)
    for i in range(batch_size):
        prompt_idx = uid_to_index[uids[i]]
        index_to_scores[prompt_idx].append(scores[i])

    # Count positive and negative samples for each prompt
    positive_counts = torch.zeros(num_unique_prompts, dtype=torch.int32)
    negative_counts = torch.zeros(num_unique_prompts, dtype=torch.int32)
    
    for prompt_idx in range(num_unique_prompts):
        for score in index_to_scores[prompt_idx]:
            if score > 0:
                positive_counts[prompt_idx] += 1
            else:
                negative_counts[prompt_idx] += 1
    
    # Calculate coverage metrics
    has_positive = positive_counts > 0
    has_negative = negative_counts > 0
    
    positive_coverage_ratio = has_positive.float().sum() / num_unique_prompts
    negative_coverage_ratio = has_negative.float().sum() / num_unique_prompts
    
    # Prompts with only positive or only negative samples
    all_positive_ratio = (has_positive & (negative_counts == 0)).float().mean()
    all_negative_ratio = (has_negative & (positive_counts == 0)).float().mean()
    
    # Effective prompts have both positive and negative samples
    num_effective_prompts = (has_positive & has_negative).float().sum()
    effective_prompt_ratio = num_effective_prompts / num_unique_prompts
    
    # Effective samples come from prompts with diverse rewards (std > 0)
    num_effective_samples = 0
    for scores_list in index_to_scores.values():
        # Convert to numpy for std calculation
        scores_array = torch.stack(scores_list).cpu().numpy()
        if np.std(scores_array) > 0:
            num_effective_samples += len(scores_list)
    
    effective_gradient_ratio = num_effective_samples / batch_size

    extremely_easy_count = 0
    easy_count = 0
    medium_count = 0
    hard_count = 0
    extremely_hard_count = 0
    
    success_rate_lst = (positive_counts / (positive_counts + negative_counts)).numpy()
    for success_rate in success_rate_lst:
        if success_rate == 0.0:
            extremely_hard_count += 1
        elif success_rate <= 0.2:
            hard_count += 1
        elif success_rate < 0.8:
            medium_count += 1
        elif success_rate < 1.0:
            easy_count += 1
        else:  # success_rate == 1.0
            extremely_easy_count += 1

    difficulty_metrics = {
        "prompt/extremely_easy_count": extremely_easy_count,
        "prompt/extremely_easy_ratio": extremely_easy_count / num_unique_prompts,
        "prompt/easy_count": easy_count,
        "prompt/easy_ratio": easy_count / num_unique_prompts,
        "prompt/medium_count": medium_count,
        "prompt/medium_ratio": medium_count / num_unique_prompts,
        "prompt/hard_count": hard_count,
        "prompt/hard_ratio": hard_count / num_unique_prompts,
        "prompt/extremely_hard_count": extremely_hard_count,
        "prompt/extremely_hard_ratio": extremely_hard_count / num_unique_prompts,
    }

    # Compile all metrics
    metrics = {
        # Coverage metrics
        "prompt/positive_coverage_ratio": positive_coverage_ratio.item(),
        "prompt/negative_coverage_ratio": negative_coverage_ratio.item(),
        "prompt/all_positive_ratio": all_positive_ratio.item(),
        "prompt/all_negative_ratio": all_negative_ratio.item(),
        
        # Training effectiveness metrics
        "training/effective_prompt_ratio": effective_prompt_ratio.item(),
        "training/effective_gradient_ratio": effective_gradient_ratio,
        "training/num_effective_prompts": num_effective_prompts.item(),
        "training/num_effective_samples": num_effective_samples,
    }

    # Add difficulty distribution if available
    metrics.update(difficulty_metrics)
    
    return metrics


def bootstrap_metric_vectorized(
    data: np.ndarray,
    subset_sizes: List[int],
    reduce_fns: List[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    is_dict_data: bool = False
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Perform vectorized bootstrap resampling for multiple subset sizes.
    
    This function efficiently computes bootstrap statistics by generating all
    random indices at once and reusing them across different subset sizes.
    
    Args:
        data: Input data array to bootstrap from.
        subset_sizes: List of subset sizes to compute bootstrap statistics for.
        reduce_fns: List of reduction functions to apply to each bootstrap sample.
            Each function should take an array and return a scalar.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.
        is_dict_data: Whether the data contains dictionary elements (e.g., for
            majority voting). If True, uses list comprehension; if False, uses
            vectorized operations. Defaults to False.
    
    Returns:
        Dictionary mapping each subset_size to a list of (mean, std) tuples,
        one for each reduction function.
    
    Example:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> result = bootstrap_metric_vectorized(
        ...     data=data,
        ...     subset_sizes=[2, 3],
        ...     reduce_fns=[np.max, np.min],
        ...     n_bootstrap=100
        ... )
        >>> # Returns: {2: [(mean_max, std_max), (mean_min, std_min)],
        >>> #           3: [(mean_max, std_max), (mean_min, std_min)]}
    
    Optimization details:
        - Generates all bootstrap indices in one call: O(n_bootstrap * max_size)
        - Reuses random samples for all subset sizes
        - Uses vectorized numpy operations for numeric data
    """
    rng = np.random.RandomState(seed)
    n_data = len(data)
    max_subset_size = max(subset_sizes)
    
    # Generate all bootstrap indices at once: (n_bootstrap, max_subset_size)
    all_bootstrap_idxs = rng.choice(
        n_data, 
        size=(n_bootstrap, max_subset_size), 
        replace=True
    )
    
    results = {}
    for subset_size in subset_sizes:
        # Use only the first subset_size columns
        bootstrap_idxs = all_bootstrap_idxs[:, :subset_size]
        
        metric_results = []
        for reduce_fn in reduce_fns:
            if is_dict_data:
                # Handle dictionary-type data (e.g., for majority voting)
                # Cannot be fully vectorized due to complex data structure
                bootstrap_metrics = []
                for idxs in bootstrap_idxs:
                    bootstrap_sample = [data[i] for i in idxs]
                    bootstrap_metrics.append(reduce_fn(bootstrap_sample))
                bootstrap_metrics = np.array(bootstrap_metrics)
            else:
                # Numeric data: use vectorized operations
                bootstrap_samples = data[bootstrap_idxs]  # (n_bootstrap, subset_size)
                bootstrap_metrics = np.apply_along_axis(reduce_fn, 1, bootstrap_samples)
            
            metric_results.append((
                np.mean(bootstrap_metrics), 
                np.std(bootstrap_metrics)
            ))
        
        results[subset_size] = metric_results
    
    return results


def process_single_sample_metrics(
    var_vals_dict: Dict[str, List],
    seed: int,
    n_bootstrap: int = 1000,
    calc_maj_val_fn: Optional[Callable] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for a single sample across all variables.
    
    For each variable, computes:
    - Basic statistics: mean and standard deviation
    - Bootstrap estimates for best/worst values at various sample sizes
    - Majority voting statistics (if prediction data is available)
    
    Args:
        var_vals_dict: Dictionary mapping variable names to lists of values.
            Special key "pred" is used for majority voting if present.
        seed: Random seed for bootstrap sampling.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        calc_maj_val_fn: Optional function for calculating majority voting values.
            Should have signature: calc_maj_val(data, vote_key, val_key) -> float.
    
    Returns:
        Dictionary mapping variable names to metric dictionaries.
        Each metric dictionary contains keys like:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of best values in bootstrap samples
        - "best@N/std": Std of best values in bootstrap samples
        - "worst@N/mean": Mean of worst values in bootstrap samples
        - "worst@N/std": Std of worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting (if predictions available)
        - "maj@N/std": Std of majority voting (if predictions available)
    
    Example:
        >>> var_vals = {
        ...     "score": [0.8, 0.9, 0.85, 0.88],
        ...     "pred": ["A", "A", "B", "A"]
        ... }
        >>> metrics = process_single_sample_metrics(var_vals, seed=42)
    """
    result = {}
    
    for var_name, var_vals in var_vals_dict.items():
        # Skip string-type variables (except for 'pred' used in majority voting)
        if isinstance(var_vals[0], str):
            continue
        
        var_vals_array = np.array(var_vals)
        metric = {}
        n_resps = len(var_vals_array)
        
        # Basic statistics
        metric[f"mean@{n_resps}"] = float(var_vals_array.mean())
        
        if n_resps > 1:
            metric[f"std@{n_resps}"] = float(var_vals_array.std())
            
            # Calculate required subset sizes (powers of 2 up to n_resps)
            ns = []
            n = 2
            while n < n_resps:
                ns.append(n)
                n *= 2
            ns.append(n_resps)
            
            # Compute bootstrap statistics for all subset sizes at once
            bootstrap_results = bootstrap_metric_vectorized(
                data=var_vals_array,
                subset_sizes=ns,
                reduce_fns=[np.max, np.min],
                n_bootstrap=n_bootstrap,
                seed=seed,
                is_dict_data=False
            )
            
            for n in ns:
                (bon_mean, bon_std), (won_mean, won_std) = bootstrap_results[n]
                metric[f"best@{n}/mean"] = float(bon_mean)
                metric[f"best@{n}/std"] = float(bon_std)
                metric[f"worst@{n}/mean"] = float(won_mean)
                metric[f"worst@{n}/std"] = float(won_std)
            
            # Process majority voting if prediction data is available
            if "pred" in var_vals_dict and var_vals_dict["pred"] is not None:
                if calc_maj_val_fn is None:
                    raise ValueError(
                        "calc_maj_val_fn must be provided when 'pred' is in var_vals_dict"
                    )
                
                vote_data = np.array([
                    {"val": val, "pred": pred} 
                    for val, pred in zip(var_vals, var_vals_dict["pred"], strict=True)
                ], dtype=object)
                
                maj_bootstrap_results = bootstrap_metric_vectorized(
                    data=vote_data,
                    subset_sizes=ns,
                    reduce_fns=[partial(calc_maj_val_fn, vote_key="pred", val_key="val")],
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    is_dict_data=True
                )
                
                for n in ns:
                    (maj_n_mean, maj_n_std) = maj_bootstrap_results[n][0]
                    metric[f"maj@{n}/mean"] = float(maj_n_mean)
                    metric[f"maj@{n}/std"] = float(maj_n_std)
        
        result[var_name] = metric
    
    return result


def process_datasource_parallel(
    args: Tuple[str, str, Dict[str, List], int, int, Optional[Callable]]
) -> Tuple[str, str, Dict[str, Dict[str, float]]]:
    """
    Worker function for parallel processing of a single (data_source, uid) pair.
    
    This function is designed to be called by a ProcessPoolExecutor for
    parallel computation across multiple samples.
    
    Args:
        args: Tuple containing:
            - data_source: Data source identifier
            - uid: Sample unique identifier
            - var_vals_dict: Variable values dictionary
            - seed: Random seed
            - n_bootstrap: Number of bootstrap iterations
            - calc_maj_val_fn: Optional majority voting function
    
    Returns:
        Tuple of (data_source, uid, metrics_dict)
    """
    data_source, uid, var_vals_dict, seed, n_bootstrap, calc_maj_val_fn = args
    metrics = process_single_sample_metrics(
        var_vals_dict, seed, n_bootstrap, calc_maj_val_fn
    )
    return data_source, uid, metrics


def process_validation_metrics_optimized(
    data_sources: List[str],
    sample_uids: List[str],
    infos_dict: Dict[str, List[Any]],
    seed: int = 42,
    n_bootstrap: int = 1000,
    n_workers: Optional[int] = 8,
    use_parallel: bool = True,
    calc_maj_val_fn: Optional[Callable] = calc_maj_val,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Process validation metrics with optimized bootstrap resampling.
    
    This function organizes validation metrics by data source and sample UID,
    computes various statistical measures, and performs bootstrap sampling
    to estimate statistics for different sample sizes.
    
    Main optimizations:
    1. Vectorized bootstrap sampling: All random indices generated at once
    2. Batch processing: Multiple subset sizes computed simultaneously
    3. Parallel processing: Samples processed across multiple CPU cores
    4. Reduced type conversions: Direct use of numpy arrays
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample UIDs corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values.
            Each list should have the same length as data_sources.
        seed: Random seed for bootstrap sampling. Defaults to 42.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
            Can be reduced (e.g., to 500) for faster computation with slightly
            lower precision.
        n_workers: Number of parallel workers. If None, uses CPU count.
            Defaults to None.
        use_parallel: Whether to use parallel processing. Set to False for
            small datasets or debugging. Defaults to True.
        calc_maj_val_fn: Optional function for calculating majority voting values.
            Required if 'pred' key is present in infos_dict.
            Should have signature: calc_maj_val(data, vote_key, val_key) -> float.
    
    Returns:
        Nested dictionary with structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Metric names include:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of best values in bootstrap samples of size N
        - "best@N/std": Std of best values in bootstrap samples
        - "worst@N/mean": Mean of worst values in bootstrap samples
        - "worst@N/std": Std of worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting (if predictions available)
        - "maj@N/std": Std of majority voting (if predictions available)
    
    Raises:
        ValueError: If calc_maj_val_fn is None but 'pred' is in infos_dict.
    
    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7]}
        >>> result = process_validation_metrics_optimized(
        ...     data_sources=data_sources,
        ...     sample_uids=sample_uids,
        ...     infos_dict=infos_dict,
        ...     seed=42,
        ...     n_bootstrap=500,
        ...     use_parallel=True
        ... )
    """
    # Step 1: Group data by data_source and uid
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])
    
    # Step 2: Prepare tasks for parallel processing
    tasks = []
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            tasks.append((
                data_source, 
                uid, 
                dict(var2vals), 
                seed, 
                n_bootstrap,
                calc_maj_val_fn
            ))
    
    # Step 3: Process tasks in parallel or serial mode
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(dict))
    
    if use_parallel and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_datasource_parallel, task) 
                for task in tasks
            ]
            
            for future in as_completed(futures):
                data_source, uid, metrics = future.result()
                data_src2uid2var2metric[data_source][uid] = metrics
    else:
        # Serial processing for small datasets or debugging
        for task in tasks:
            data_source, uid, metrics = process_datasource_parallel(task)
            data_src2uid2var2metric[data_source][uid] = metrics
    
    # Step 4: Aggregate metrics across UIDs
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(
                        metric_val
                    )
    
    # Step 5: Compute final averages across all UIDs
    data_src2var2metric2val = defaultdict(lambda: defaultdict(dict))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = float(
                    np.mean(uid_vals)
                )
    
    return dict(data_src2var2metric2val)