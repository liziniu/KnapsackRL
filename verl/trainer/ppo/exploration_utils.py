"""
Knapsack-based Sampling Budget Allocation

This module implements dynamic programming algorithms for allocating sampling budgets
across multiple states based on their success rates and costs.
"""

import numpy as np
import time
from typing import List, Tuple, Dict

# Try to import numba for optimization if available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def sampling_budget_by_knapsack(
    reward_states: np.ndarray,
    n: int,
    min_n: int = 2,
    max_n: int = None,
    alpha: float = 0.9,
    solver: str = "dp",
    verbose: bool = False,
    rank: int = 0,
    allow_fallback: bool = True,
    negative_n: int = None,
    positive_n: int = None,
    total_budget: int = None,
    cost_estimation: str = "probability",
) -> Tuple[np.ndarray, Dict]:
    """
    Allocate sampling budget across states using knapsack optimization.
    
    Args:
        reward_states: Array of shape (batch_size, 3) containing [validity, success_rate, count]
        n: Default number of samples per state
        min_n: Minimum samples for states with 0 < success_rate < 1
        max_n: Maximum samples per state (defaults to n if not specified)
        alpha: Confidence level for probability-based cost estimation
        solver: Optimization solver method ("dp" for dynamic programming)
        verbose: Whether to print detailed results
        rank: Process rank for distributed training (only rank 0 prints)
        allow_fallback: Whether to redistribute excess budget
        negative_n: Samples for states with success_rate = 0
        positive_n: Samples for states with success_rate = 1
        total_budget: Total sampling budget (defaults to batch_size * n)
        cost_estimation: Cost calculation method ("probability" or "expectation")
    
    Returns:
        Tuple of (sampling_budget_array, info_dict)
    """
    batch_size = len(reward_states)
    
    # Set default values
    if total_budget is None:
        total_budget = batch_size * n
    if negative_n is None:
        negative_n = min_n
    if positive_n is None:
        positive_n = min_n
    
    # Initialize arrays
    sampling_budget = np.zeros(batch_size, dtype=np.int32)
    costs = np.zeros(batch_size, dtype=np.float32)
    values = np.zeros(batch_size, dtype=np.float32)
    success_rates = np.zeros(batch_size, dtype=np.float32)
    counts = np.zeros(batch_size, dtype=np.int32)
    effective_indices = []
    
    # Process each state and calculate costs/values
    for i in range(batch_size):
        if reward_states[i][0] == 1:
            success_rate = reward_states[i][1]
            count = reward_states[i][2]

            # Calculate cost based on success rate
            if 0 < success_rate < 1:
                if cost_estimation == "probability":
                    cost = np.log(1 - alpha) / np.log(max(success_rate, 1 - success_rate))
                elif cost_estimation == "expectation":
                    cost = 1 / (success_rate * (1 - success_rate)) - 1
                else:
                    raise ValueError(f"Unknown cost estimation method: {cost_estimation}")

                sampling_budget[i] += min_n
                effective_indices.append(i)
            elif success_rate == 0:
                cost = float('inf')
                sampling_budget[i] += int(negative_n)
            else:  # success_rate == 1
                cost = 0
                sampling_budget[i] += int(positive_n)
         
            costs[i] = cost
            values[i] = (1 - success_rate) * success_rate * ((1 - success_rate) ** 2)
            success_rates[i] = success_rate
        else:
            sampling_budget[i] += n
        
        counts[i] = reward_states[i][2]
    
    # Calculate remaining budget after initial allocation
    total_budget_backup = total_budget
    total_budget -= sum(sampling_budget)

    fallback = False
    fallback_effective_ratio = 1.0
    
    if total_budget > 0 and len(effective_indices) == 0:
        start_time = time.time()
        # Prioritize allocation to states with success_rate = 0
        zero_success_indices = np.where(success_rates == 0)[0]
        
        if len(zero_success_indices) > 0:
            # Distribute budget among zero-success states
            budget_per_state = total_budget // len(zero_success_indices)
            remainder = total_budget % len(zero_success_indices)
            
            for idx, i in enumerate(zero_success_indices):
                extra_budget = budget_per_state + (1 if idx < remainder else 0)
                sampling_budget[i] += extra_budget
                total_budget -= extra_budget

            print(f"[Info] Effective indices are empty with {total_budget} budget left. "
                  f"Assigned {budget_per_state} budget to {len(zero_success_indices)} states with success_rate = 0")
        else:
            # If no zero-success states, distribute uniformly
            sampling_budget = np.zeros(batch_size, dtype=np.int32)
            sampling_budget[:] = n
            print(f"[Warning] Effective indices are empty. Assigned {total_budget} budget to all states")
        
        end_time = time.time()
    else:
        # Estimate total cost for effective indices
        estimated_cost = 0
        for idx in effective_indices:
            estimated_cost += int(min(max_n, costs[idx]))

        # If estimated cost is less than budget, redistribute excess
        if estimated_cost < total_budget and allow_fallback:
            fallback = True
            fallback_effective_ratio = estimated_cost / total_budget

            zero_success_indices = np.where(success_rates == 0)[0]
            abundant_budget = total_budget - estimated_cost

            if len(zero_success_indices) > 0:
                candidate_indices = zero_success_indices
            else:
                candidate_indices = [i for i in range(batch_size) if i not in effective_indices]

            budget_per_candidate = abundant_budget // len(candidate_indices)
            remainder = abundant_budget % len(candidate_indices)
            
            for idx, i in enumerate(candidate_indices):
                extra_budget = budget_per_candidate + (1 if idx < remainder else 0)
                sampling_budget[i] += extra_budget

            print(f"[Info] Effective cost {estimated_cost} (method: {cost_estimation}) is less than "
                  f"total budget {total_budget}. Assigned additional budget {abundant_budget} to "
                  f"{len(candidate_indices)} states with success_rate = 0 or 1")
            
            total_budget = total_budget - abundant_budget

        start_time = time.time()
        main_sampling_budget = solve_by_dp(
            success_rates,
            effective_indices,
            min_n,
            max_n - min_n,
            total_budget,
            solver=solver,
        )
        end_time = time.time()

        sampling_budget = sampling_budget + main_sampling_budget

    # Verify budget allocation
    assert np.sum(sampling_budget) == total_budget_backup, (
        f"Budget mismatch: expected {total_budget_backup}, got {np.sum(sampling_budget)}"
    )

    if verbose and rank == 0:
        print(f"Knapsack computation time: {end_time - start_time:.1f} s")
        print_results(success_rates, counts, costs, values, sampling_budget)

    info = {
        "fallback": fallback,
        "fallback_effective_ratio": fallback_effective_ratio,
    }

    return sampling_budget, info


if NUMBA_AVAILABLE:
    @njit
    def dp_core_numba(success_rates, n_items, total_budget, max_n, min_n, solver):
        """
        Numba-accelerated core dynamic programming computation.
        
        Args:
            success_rates: Array of success rates for effective items
            n_items: Number of items to allocate budget to
            total_budget: Total available budget
            max_n: Maximum allocation per item
            min_n: Minimum allocation already assigned
            solver: Solver method name
        
        Returns:
            Tuple of (dp_table, choice_table)
        """
        dp = np.full((n_items + 1, total_budget + 1), -np.inf)
        dp[0, 0] = 0.0
        
        choice = np.full((n_items + 1, total_budget + 1, 2), -1, dtype=np.int32)
        
        for i in range(1, n_items + 1):
            success_rate = success_rates[i - 1]
            
            for w in range(total_budget + 1):
                # Option 1: Don't allocate to this item
                if dp[i - 1, w] > dp[i, w]:
                    dp[i, w] = dp[i - 1, w]
                    choice[i, w, 0] = 0
                    choice[i, w, 1] = 0
                
                # Option 2: Allocate k units to this item
                for k in range(1, min(max_n + 1, w + 1)):
                    if dp[i - 1, w - k] == -np.inf:
                        continue
                    
                    if success_rate == 0 or success_rate == 1:
                        value_gain = 0
                    else:
                        if solver == "dp":
                            value_gain = (
                                (1 - (success_rate ** (k + min_n)) - ((1 - success_rate) ** (k + min_n)))
                                * success_rate * ((1 - success_rate) ** 2)
                            )
                        else:
                            raise ValueError(f"Unknown solver: {solver}")
                    
                    new_value = dp[i - 1, w - k] + value_gain
                    
                    if new_value > dp[i, w]:
                        dp[i, w] = new_value
                        choice[i, w, 0] = i - 1
                        choice[i, w, 1] = k
        
        return dp, choice


def solve_by_dp(
    success_rates: np.ndarray,
    effective_indices: List[int],
    min_n: int,
    max_n: int,
    total_budget: int,
    solver: str = "dp"
) -> np.ndarray:
    """
    Solve budget allocation using dynamic programming.
    
    Automatically selects Numba-optimized version for large problems.
    
    Args:
        success_rates: Success rate for each state
        effective_indices: Indices of states eligible for optimization
        min_n: Minimum allocation already assigned
        max_n: Maximum additional allocation per state
        total_budget: Remaining budget to allocate
        solver: Optimization method
    
    Returns:
        Array of additional budget allocations per state
    """
    batch_size = len(success_rates)
    sampling_budget = np.zeros(batch_size, dtype=np.int32)
    
    if total_budget <= 0:
        print(f"[Warning] Total budget = {total_budget} is non-positive, returning zero allocation")
        return sampling_budget
    
    if len(effective_indices) == 0:
        print("[Warning] Effective indices list is empty, using uniform allocation")
        base_allocation = total_budget // batch_size
        remainder = total_budget % batch_size
        
        for i in range(batch_size):
            sampling_budget[i] += base_allocation
        
        for i in range(remainder):
            sampling_budget[i] += 1
        
        return sampling_budget
    
    n_items = len(effective_indices)
    
    # Use Numba-optimized version for large problems
    use_numba = NUMBA_AVAILABLE and (n_items * total_budget * max_n > 50000)
    
    if use_numba:
        # Extract success rates for effective indices
        effective_success_rates = np.array([success_rates[i] for i in effective_indices])
        
        # Run Numba-accelerated computation
        dp, choice = dp_core_numba(effective_success_rates, n_items, total_budget, max_n, min_n, solver)
        
        # Find maximum usable budget if full budget is not feasible
        if dp[n_items, total_budget] == -np.inf:
            max_usable_budget = total_budget
            while max_usable_budget > 0 and dp[n_items, max_usable_budget] == -np.inf:
                max_usable_budget -= 1
            
            if max_usable_budget > 0:
                # Backtrack solution
                w = max_usable_budget
                for i in range(n_items, 0, -1):
                    item_idx, allocation = choice[i, w, 0], choice[i, w, 1]
                    if allocation > 0:
                        actual_idx = effective_indices[item_idx]
                        sampling_budget[actual_idx] += allocation
                        w -= allocation
                
                remaining_budget = total_budget - max_usable_budget
            else:
                remaining_budget = total_budget
        else:
            # Backtrack solution with full budget
            w = total_budget
            for i in range(n_items, 0, -1):
                item_idx, allocation = choice[i, w, 0], choice[i, w, 1]
                if allocation > 0:
                    actual_idx = effective_indices[item_idx]
                    sampling_budget[actual_idx] += allocation
                    w -= allocation
            remaining_budget = 0
    
    else:
        # Standard DP implementation for smaller problems
        dp = np.full((n_items + 1, total_budget + 1), -np.inf)
        dp[0][0] = 0.0
        
        choice = np.full((n_items + 1, total_budget + 1, 2), -1, dtype=int)
        
        for i in range(1, n_items + 1):
            actual_idx = effective_indices[i - 1]
            success_rate = success_rates[actual_idx]
            
            for w in range(total_budget + 1):
                # Option 1: Don't allocate to this item
                if dp[i - 1][w] > dp[i][w]:
                    dp[i][w] = dp[i - 1][w]
                    choice[i][w] = [0, 0]
                
                # Option 2: Allocate k units to this item
                for k in range(1, min(max_n + 1, w + 1)):
                    if dp[i - 1][w - k] == -np.inf:
                        continue
                    
                    if success_rate == 0 or success_rate == 1:
                        value_gain = 0
                    else:
                        # Note: min_n is already assigned to these states
                        if solver == "dp":
                            value_gain = (
                                (1 - (success_rate ** (k + min_n)) - ((1 - success_rate) ** (k + min_n)))
                                * success_rate * ((1 - success_rate) ** 2)
                            )
                        else:
                            raise ValueError(f"Unknown solver: {solver}")
                    
                    new_value = dp[i - 1][w - k] + value_gain
                    
                    if new_value > dp[i][w]:
                        dp[i][w] = new_value
                        choice[i][w] = [i - 1, k]
        
        # Find maximum usable budget
        max_usable_budget = total_budget
        while max_usable_budget > 0 and dp[n_items][max_usable_budget] == -np.inf:
            max_usable_budget -= 1
        
        if max_usable_budget > 0:
            # Backtrack solution
            w = max_usable_budget
            for i in range(n_items, 0, -1):
                item_idx, allocation = choice[i][w]
                if allocation > 0:
                    actual_idx = effective_indices[item_idx]
                    sampling_budget[actual_idx] += allocation
                    w -= allocation
            
            remaining_budget = total_budget - max_usable_budget
        else:
            remaining_budget = total_budget
    
    # Distribute any remaining budget uniformly across effective indices
    if remaining_budget > 0 and effective_indices:
        base_allocation = remaining_budget // len(effective_indices)
        remainder = remaining_budget % len(effective_indices)
        
        for idx in effective_indices:
            sampling_budget[idx] += base_allocation
        
        for i in range(remainder):
            idx = effective_indices[i]
            sampling_budget[idx] += 1
    
    return sampling_budget


def print_results(
    success_rates: np.ndarray,
    counts: np.ndarray,
    costs: np.ndarray,
    values: np.ndarray,
    sampling_budget: np.ndarray,
    max_items_to_print: int = 16
) -> None:
    """
    Print formatted results table.
    
    Args:
        success_rates: Success rate for each state
        counts: Sample count for each state
        costs: Estimated cost for each state
        values: Computed value for each state
        sampling_budget: Allocated budget for each state
        max_items_to_print: Maximum number of rows to display
    """
    try:
        from tabulate import tabulate
        table_available = True
    except ImportError:
        table_available = False
        print("[Warning] tabulate not available, showing simple output")

    batch_size = len(success_rates)
    
    if table_available:
        table_data = []
        for i in range(min(batch_size, max_items_to_print)):
            cost_str = "INF" if np.isinf(costs[i]) else f"{costs[i]:.1f}"
            value_str = f"{values[i]:.2f}"
            
            # Calculate value-to-cost ratio for display
            if costs[i] == 0:
                ratio = "INF"
            elif np.isinf(costs[i]):
                ratio = "0.00"
            else:
                ratio = f"{values[i] / costs[i]:.2f}"
            
            table_data.append([
                i,
                f"{success_rates[i]:.2f}",
                counts[i],
                cost_str,
                value_str,
                ratio,
                sampling_budget[i]
            ])

        print(tabulate(
            table_data,
            headers=["Index", "Success Rate", "Count", "Cost", "Value", "Ratio", "Assignment"],
            tablefmt="grid",
            floatfmt=".2f"
        ))
    else:
        print("Index | Success Rate | Count | Cost  | Value | Assignment")
        print("-" * 60)
        for i in range(min(batch_size, max_items_to_print)):
            cost_str = "INF" if np.isinf(costs[i]) else f"{costs[i]:.1f}"
            value_str = f"{values[i]:.2f}"
            print(f"{i:5d} | {success_rates[i]:11.2f} | {counts[i]:5d} | "
                  f"{cost_str:>4s} | {value_str:>5s} | {sampling_budget[i]:10d}")