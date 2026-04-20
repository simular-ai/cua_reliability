#!/usr/bin/env python3
"""
Compute reliability metrics from experiment results.

Supports two modes:
  1. Single setting  -- compute Pass^1, Pass^3 for one set of N=3 runs
  2. Compare settings -- compute metrics for both settings PLUS McNemar and
     Wilcoxon paired tests

Examples
--------
Single setting:
    python analysis/compute_metrics.py \
      --run-dirs path/to/run1 path/to/run2 path/to/run3 \
      --example-json evaluation_examples/test_nogdrive.json

Compare two settings:
    python analysis/compute_metrics.py \
      --run-dirs path/to/baseline/run1 path/to/baseline/run2 path/to/baseline/run3 \
      --compare-dirs path/to/new/run1 path/to/new/run2 path/to/new/run3 \
      --example-json evaluation_examples/test_nogdrive.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, OrderedDict
from itertools import combinations

import numpy as np
from scipy.stats import chi2, t, wilcoxon as scipy_wilcoxon


# ---------------------------------------------------------------------------
# Cell 1 -- helpers
# ---------------------------------------------------------------------------

def load_trajectory_from_jsonl(filepath):
    """Load trajectory from a JSONL file.

    Args:
        filepath (str): Path to the JSONL file containing the trajectory.

    Returns:
        list: A list of dictionaries representing the trajectory.
    """
    with open(filepath, 'r') as f:
        trajectory = [json.loads(line) for line in f]
    return trajectory


def filter_by_keys(run_results, keys):
    """Filter the run results by a set of keys.

    Args:
        run_results (dict): The run results to filter.
        keys (set): The set of keys to keep in the filtered results.

    Returns:
        dict: The filtered run results.
    """
    filtered_results = {}
    for domain, examples in run_results.items():
        filtered_examples = {}
        for example_id, stats in examples.items():
            filtered_stats = {k: v for k, v in stats.items() if k in keys}
            if filtered_stats:
                filtered_examples[example_id] = filtered_stats
        if filtered_examples:
            filtered_results[domain] = filtered_examples
    return filtered_results


# ---------------------------------------------------------------------------
# Cell 2 -- calculate_steps, get_run_results, _load_results_from_directory
# ---------------------------------------------------------------------------

def calculate_steps(traj):
    """Calculate the number of steps in a trajectory.

    This isn't trivially len(traj) because judge steps aren't counted as OSWorld steps

    Args:
        traj (list): The trajectory to analyze.

    Returns:
        int: The number of steps in the trajectory.
    """
    num_steps = 0
    for step in traj:
        if "step_num" in step and type(step["step_num"]) is not int:
            continue  # Likely a judge step; ignore these as we only care about agent steps
        num_steps += 1
    return num_steps


def get_run_results(target_dir, example_json=None, examples_dir="evaluation_examples/examples", statistics=set(["success"]), missing_as_failure=False, override_target_dir=None, override_example_json=None, quiet=False):
    """
    Load run results from target directory, with optional override from another directory.

    Args:
        target_dir (str): Primary target directory path
        example_json (str): Primary example set JSON file
        examples_dir (str): Directory containing example JSON files
        statistics (set): Set of statistics to collect
        missing_as_failure (bool): If True, treat missing result files as failures
        override_target_dir (str): Optional override directory path
        override_example_json (str): Optional override example set JSON file

    Returns:
        dict: Results with overrides applied where applicable
    """
    return _load_results_from_directory(target_dir, example_json, examples_dir, statistics, missing_as_failure, override_target_dir, override_example_json, quiet=quiet)


def _load_results_from_directory(target_dir, example_json=None, examples_dir="evaluation_examples/examples", statistics=set(["success"]), missing_as_failure=False, override_target_dir=None, override_example_json=None, quiet=False):
    """Load results from a single directory."""
    _print = (lambda *a, **kw: None) if quiet else print
    if not os.path.exists(target_dir):
        _print("Target directory does not exist")
        return None

    success_result = {}
    all_result_for_analysis = {}

    if example_json:
        with open(example_json, "r") as f:
            example_data = json.load(f)
        example_id_filter = set(sum([l for l in example_data.values()], []))
        all_result_for_analysis = {k: {v: {} for v in v_list} for k, v_list in example_data.items()}
        _print("Number of examples to run:", len(example_id_filter))

    for domain in sorted(os.listdir(target_dir)):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            # Skip if the domain path is not a directory
            continue
        # all_result_for_analysis[domain] = {}
        for example_id in os.listdir(domain_path):
            if example_json and example_id_filter and example_id not in example_id_filter:
                continue # Only analyze examples in the filter, if any
            if override_target_dir and os.path.exists(os.path.join(override_target_dir, domain, example_id)):
                example_path = os.path.join(override_target_dir, domain, example_id)
                _print(f"Using override for {domain}/{example_id}")
            else:
                example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue # Skip if the example path is not a directory
            all_result_for_analysis[domain][example_id] = all_result_for_analysis[domain].get(example_id, {})
            traj_json = os.path.join(example_path, "traj.jsonl")
            if not os.path.exists(traj_json):
                _print(f"Trajectory JSON file does not exist for {domain}/{example_id}. Skipping...")
                continue
            for statistic in statistics:
                if statistic == "success":
                    if "result.txt" not in os.listdir(example_path):
                        if missing_as_failure:
                            _print(f"Result file does not exist for {domain}/{example_id}. Treating as failure (0.0 success)...")
                            result_float = 0.0
                            # Still need trajectory for step counting if available
                            if os.path.exists(traj_json):
                                traj = load_trajectory_from_jsonl(traj_json)
                            else:
                                traj = []
                        else:
                            _print(f"Result file does not exist for {domain}/{example_id}. Skipping...")
                            continue
                    else:
                        with open(os.path.join(example_path, "result.txt"), "r") as f:
                            result = f.read().strip()
                        try:
                            result_float = float(result)
                        except:
                            _print(f"Result file is not a float for {domain}/{example_id}.")
                            if missing_as_failure:
                                _print(f"Treating as failure (0.0 success)...")
                                result_float = 0.0
                                traj = load_trajectory_from_jsonl(traj_json)
                            else:
                                continue
                        result_float = max(0, min(1, result_float)) # Clamp to [0, 1]
                        traj = load_trajectory_from_jsonl(traj_json)

                    for max_steps in [15, 50, 100]:
                        result = result_float if calculate_steps(traj) <= max_steps else 0.0
                        success_result[max_steps] = success_result.get(max_steps, {})
                        success_result[max_steps][domain] = success_result[max_steps].get(domain, []) + [result]
                        all_result_for_analysis[domain][example_id]["success"] = all_result_for_analysis[domain][example_id].get("success", {15: [], 50: [], 100: []})
                        all_result_for_analysis[domain][example_id]["success"][max_steps] = success_result[max_steps][domain][-1]
                elif statistic == "error":
                    if "error.txt" not in os.listdir(example_path):
                        continue
                    _print(f"Error file exists for {domain}/{example_id}. Reading...")
                    with open(os.path.join(example_path, "error.txt"), "r") as f:
                        error_msg = f.read().strip()
                    all_result_for_analysis[domain][example_id]["error"] = error_msg
                elif statistic == "steps":
                    traj = load_trajectory_from_jsonl(traj_json)
                    num_steps = calculate_steps(traj)
                    all_result_for_analysis[domain][example_id]["steps"] = num_steps
                else:
                    # Assume statistic is a property in the trajectory
                    traj = load_trajectory_from_jsonl(traj_json)
                    for i, step in enumerate(traj):
                        if statistic not in step:
                            continue
                        all_result_for_analysis[domain][example_id][statistic] = all_result_for_analysis[domain][example_id].get(statistic, []) + [step[statistic]]

    for max_steps in [15, 50, 100]:
        if max_steps not in success_result:
            continue
        for domain in success_result[max_steps]:
            _print(f"{'Domain:':<12} {domain:<20} "
                  f"{'Ran:':<10} {len(success_result[max_steps][domain]):<6} "
                  f"{'Success Rate:':<15} {sum(success_result[max_steps][domain]) / len(success_result[max_steps][domain]) * 100:6.2f}% "
                  f"{'Examples:':<10} {sum(success_result[max_steps][domain])}")
        success_len = len(sum(success_result[max_steps].values(), []))
        success_sum = sum(num for sublist in success_result[max_steps].values() for num in sublist)
        _print(f"[{max_steps} {'steps]':<30}"
              f"{'Ran:':<10} {success_len:<6} "
              f"{'Success Rate:':<15} {success_sum / success_len * 100:6.2f}% "
              f"{'Examples:':<10} {success_sum}")
    return all_result_for_analysis


# ---------------------------------------------------------------------------
# Cell 3 -- calculate_pass_hat_k_for_k
# ---------------------------------------------------------------------------

def calculate_pass_hat_k_for_k(
    target_dirs,
    example_json,
    k,
    max_steps=100,
    success_threshold=0.9,
    missing_as_failure=False,
    override_target_dirs=[],
    quiet=False,
    override_example_json=None,
):
    """
    Calculate pass^k metric for a specific k: probability that ALL k randomly
    sampled attempts succeed.

    Returns a generic metric-shaped result object so paired comparisons can be
    reused across different metrics.
    """

    if not example_json or not os.path.exists(example_json):
        _print("Example JSON file does not exist or not provided")
        return None

    with open(example_json, "r") as f:
        example_data = json.load(f)

    all_results = {}
    _print = (lambda *a, **kw: None) if quiet else print
    for i, target_dir in enumerate(target_dirs):
        _print(f"Loading results from: {target_dir}")
        if override_target_dirs and len(override_target_dirs) == len(target_dirs):
            override_target_dir = override_target_dirs[i]
        else:
            override_target_dir = None

        results = get_run_results(
            target_dir,
            example_json,
            statistics=set(["success"]),
            missing_as_failure=missing_as_failure,
            override_target_dir=override_target_dir,
            override_example_json=override_example_json,
            quiet=quiet,
        )
        if results:
            all_results[target_dir] = results

    if not all_results:
        _print("No valid results found in any target directory")
        return None

    if k > len(all_results):
        raise ValueError(
            f"k={k} is greater than number of available runs ({len(all_results)})"
        )

    metric_results = {
        "metric_name": f"pass_hat_{k}",
        "overall": {"total_problems": 0, "sum_metric": 0.0, "metric": 0.0},
        "by_domain": {},
        "per_task_metric": [],
        "per_task_metric_by_domain": {},
        "per_task_metadata": [],
    }

    for domain, example_ids in example_data.items():
        domain_total = 0
        domain_sum_metric = 0.0
        domain_per_task_metric = []

        for example_id in example_ids:
            trajectory_results = []

            for target_dir, results in all_results.items():
                if (
                    domain in results
                    and example_id in results[domain]
                    and "success" in results[domain][example_id]
                ):
                    success_score = results[domain][example_id]["success"][max_steps]
                    trajectory_results.append(
                        1 if success_score >= success_threshold else 0
                    )
                elif missing_as_failure:
                    trajectory_results.append(0)

            if len(trajectory_results) < k:
                continue

            domain_total += 1

            if len(trajectory_results) == k:
                task_metric = 1.0 if sum(trajectory_results) == k else 0.0
            else:
                num_all_success_combinations = 0
                total_combinations = 0

                for combination in combinations(range(len(trajectory_results)), k):
                    selected_results = [trajectory_results[i] for i in combination]
                    if sum(selected_results) == k:
                        num_all_success_combinations += 1
                    total_combinations += 1

                task_metric = num_all_success_combinations / total_combinations

            domain_sum_metric += task_metric
            domain_per_task_metric.append(task_metric)
            metric_results["per_task_metric"].append(task_metric)
            metric_results["per_task_metadata"].append(
                {
                    "domain": domain,
                    "example_id": example_id,
                    "num_runs": len(trajectory_results),
                    "num_successes": sum(trajectory_results),
                }
            )

        domain_metric = (domain_sum_metric / domain_total * 100) if domain_total > 0 else 0.0

        metric_results["by_domain"][domain] = {
            "total_problems": domain_total,
            "sum_metric": domain_sum_metric,
            "metric": domain_metric,
        }
        metric_results["per_task_metric_by_domain"][domain] = domain_per_task_metric

        _print(
            f"Domain {domain:20} - Total: {domain_total:3d}, "
            f"Pass^{k}: {domain_metric:6.2f}%"
        )

    total_problems = sum(
        stats["total_problems"] for stats in metric_results["by_domain"].values()
    )
    total_sum_metric = sum(
        stats["sum_metric"] for stats in metric_results["by_domain"].values()
    )
    overall_metric = (total_sum_metric / total_problems * 100) if total_problems > 0 else 0.0

    metric_results["overall"] = {
        "total_problems": total_problems,
        "sum_metric": total_sum_metric,
        "metric": overall_metric,
    }

    _print(f"\n{'='*60}")
    _print(f"PASS^{k} RESULTS (Max Steps: {max_steps})")
    if missing_as_failure:
        _print("(Missing result files treated as failures)")
    _print(f"{'='*60}")
    _print(f"Total Problems: {total_problems}")
    _print(f"Pass^{k}: {overall_metric:.2f}%")
    _print(f"{'='*60}")

    return metric_results


# ---------------------------------------------------------------------------
# Cell 4 -- calculate_bernoulli_variance
# ---------------------------------------------------------------------------

def calculate_bernoulli_variance(
    target_dirs,
    example_json,
    max_steps=100,
    success_threshold=0.9,
    missing_as_failure=False,
    override_target_dirs=[],
    override_example_json=None,
    quiet=False,
):

    _print = (lambda *a, **kw: None) if quiet else print
    if not example_json or not os.path.exists(example_json):
        _print("Example JSON file does not exist or not provided")
        return None

    with open(example_json, "r") as f:
        example_data = json.load(f)

    all_results = {}
    for i, target_dir in enumerate(target_dirs):
        _print(f"Loading results from: {target_dir}")

        override_target_dir = (
            override_target_dirs[i]
            if override_target_dirs and len(override_target_dirs) == len(target_dirs)
            else None
        )

        results = get_run_results(
            target_dir,
            example_json,
            statistics=set(["success"]),
            missing_as_failure=missing_as_failure,
            override_target_dir=override_target_dir,
            override_example_json=override_example_json,
            quiet=quiet,
        )

        if results:
            all_results[target_dir] = results

    if not all_results:
        _print("No valid results found")
        return None

    metric_results = {
        "metric_name": "bernoulli_variance",
        "overall": {
            "total_problems": 0,
            "sum_metric": 0.0,
            "metric": 0.0,
        },
        "by_domain": {},
        "per_task_metric": [],
        "per_task_metric_by_domain": {},
        "per_task_metadata": [],
    }

    for domain, example_ids in example_data.items():
        domain_total = 0
        domain_sum_metric = 0.0
        domain_per_task_metric = []

        for example_id in example_ids:
            trajectory_results = []

            for target_dir, results in all_results.items():
                if (
                    domain in results
                    and example_id in results[domain]
                    and "success" in results[domain][example_id]
                ):
                    success_score = results[domain][example_id]["success"][max_steps]
                    trajectory_results.append(
                        1 if success_score >= success_threshold else 0
                    )
                elif missing_as_failure:
                    trajectory_results.append(0)

            n_i = len(trajectory_results)
            if n_i == 0:
                continue

            c_i = sum(trajectory_results)
            p_hat_i = c_i / n_i
            task_metric = p_hat_i * (1 - p_hat_i)

            metric_results["per_task_metric"].append(task_metric)
            domain_per_task_metric.append(task_metric)
            metric_results["per_task_metadata"].append(
                {
                    "domain": domain,
                    "example_id": example_id,
                    "num_runs": n_i,
                    "num_successes": c_i,
                    "p_hat": p_hat_i,
                }
            )

            domain_total += 1
            domain_sum_metric += task_metric

        domain_metric = domain_sum_metric / domain_total if domain_total > 0 else 0.0

        metric_results["by_domain"][domain] = {
            "total_problems": domain_total,
            "sum_metric": domain_sum_metric,
            "metric": domain_metric,
        }
        metric_results["per_task_metric_by_domain"][domain] = domain_per_task_metric

        _print(
            f"Domain {domain:20} - Total: {domain_total:3d}, "
            f"Variance: {domain_metric:.6f}"
        )

    total_problems = sum(
        stats["total_problems"] for stats in metric_results["by_domain"].values()
    )
    total_sum_metric = sum(
        stats["sum_metric"] for stats in metric_results["by_domain"].values()
    )
    overall_metric = total_sum_metric / total_problems if total_problems > 0 else 0.0

    metric_results["overall"] = {
        "total_problems": total_problems,
        "sum_metric": total_sum_metric,
        "metric": overall_metric,
    }

    _print(f"\n{'='*60}")
    _print("BERNOULLI VARIANCE (Plug-in)")
    if missing_as_failure:
        _print("(Missing results treated as failures)")
    _print(f"{'='*60}")
    _print(f"Total Problems: {total_problems}")
    _print(f"Variance: {overall_metric:.6f}")
    _print(f"{'='*60}")

    return metric_results


# ---------------------------------------------------------------------------
# Cell 5 -- stats_calculation, compare_metric_results
# ---------------------------------------------------------------------------

def stats_calculation(trial1, trial2, alpha=0.05):
    trial1 = np.array(trial1, dtype=float)
    trial2 = np.array(trial2, dtype=float)

    if len(trial1) != len(trial2):
        raise ValueError("Length mismatch")

    d = trial1 - trial2
    n = len(d)

    if n == 0:
        return None

    mean_diff = d.mean()

    if n == 1:
        return {
            "mean_diff": mean_diff,
            "se": 0.0,
            "n": n,
            "significant": False,
        }

    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)

    if se > 0:
        t_stat = mean_diff / se
    else:
        t_stat = 0.0

    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    significant = abs(t_stat) > t_crit

    return {
        "mean_diff": mean_diff,
        "se": se,
        "n": n,
        "significant": significant,
    }


def compare_metric_results(trial0_results, trial1_results, alpha=0.05):
    output = {
        "metric_name": trial0_results.get("metric_name", "metric"),
        "overall": None,
        "by_domain": {},
    }

    output["overall"] = stats_calculation(
        trial0_results["per_task_metric"],
        trial1_results["per_task_metric"],
        alpha=alpha,
    )

    domains0 = trial0_results.get("per_task_metric_by_domain", {})
    domains1 = trial1_results.get("per_task_metric_by_domain", {})

    for domain in domains0:
        if domain not in domains1:
            continue

        m0 = domains0[domain]
        m1 = domains1[domain]

        if len(m0) != len(m1) or len(m0) == 0:
            continue

        output["by_domain"][domain] = stats_calculation(m0, m1, alpha=alpha)

    return output


# ---------------------------------------------------------------------------
# Cell 6 -- difficulty breakdown, McNemar, Wilcoxon
# ---------------------------------------------------------------------------

def calculate_difficulty_breakdown(
    target_dirs,
    example_json,
    max_steps=100,
    success_threshold=0.9,
    missing_as_failure=True,
    override_target_dirs=[],
    override_example_json=None,
    quiet=False,
):
    """
    Calculate easy/medium/hard breakdown based on consistency across runs.

    Easy: All runs pass (3/3 success)
    Medium: Mixed results (1/3 or 2/3 success)
    Hard: All runs fail (0/3 success)

    Args:
        target_dirs: List of 3 target directory paths
        example_json: Path to example JSON file
        max_steps: Maximum steps to consider for success
        success_threshold: Threshold for considering a run successful
        missing_as_failure: Treat missing results as failures
        override_target_dirs: Optional override directories
        override_example_json: Optional override example JSON

    Returns:
        dict: Difficulty breakdown with counts per domain and overall
    """

    if len(target_dirs) != 3:
        raise ValueError(f"Expected exactly 3 target directories, got {len(target_dirs)}")

    _print = (lambda *a, **kw: None) if quiet else print
    if not example_json or not os.path.exists(example_json):
        _print("Example JSON file does not exist or not provided")
        return None

    with open(example_json, "r") as f:
        example_data = json.load(f)

    # Load results from all directories
    all_results = {}
    for i, target_dir in enumerate(target_dirs):
        _print(f"Loading results from: {target_dir}")

        override_target_dir = (
            override_target_dirs[i]
            if override_target_dirs and len(override_target_dirs) == len(target_dirs)
            else None
        )

        results = get_run_results(
            target_dir,
            example_json,
            statistics=set(["success"]),
            missing_as_failure=missing_as_failure,
            override_target_dir=override_target_dir,
            override_example_json=override_example_json,
            quiet=quiet,
        )

        if results:
            all_results[target_dir] = results

    if len(all_results) != 3:
        _print(f"Warning: Expected 3 result sets, got {len(all_results)}")
        return None

    difficulty_results = {
        "breakdown": {
            "overall": {"easy": 0, "medium": 0, "hard": 0},
            "by_domain": {}
        },
        "per_task_difficulty": [],
        "per_task_difficulty_by_domain": {},
        "target_dirs": target_dirs
    }

    # Process each domain
    for domain, example_ids in example_data.items():
        domain_counts = {"easy": 0, "medium": 0, "hard": 0}
        domain_per_task = []

        for example_id in example_ids:
            successes = []

            # Get success for this task across all 3 runs
            for target_dir, results in all_results.items():
                if (
                    domain in results
                    and example_id in results[domain]
                    and "success" in results[domain][example_id]
                ):
                    success_score = results[domain][example_id]["success"][max_steps]
                    successes.append(1 if success_score >= success_threshold else 0)
                elif missing_as_failure:
                    successes.append(0)

            # Skip if we don't have exactly 3 results
            if len(successes) != 3:
                continue

            success_count = sum(successes)

            # Categorize difficulty
            if success_count == 3:
                difficulty = "easy"
            elif success_count == 0:
                difficulty = "hard"
            else:
                difficulty = "medium"

            domain_counts[difficulty] += 1
            domain_per_task.append(difficulty)

            difficulty_results["per_task_difficulty"].append({
                "domain": domain,
                "example_id": example_id,
                "difficulty": difficulty,
                "successes": successes,
                "success_count": success_count
            })

        difficulty_results["breakdown"]["by_domain"][domain] = domain_counts
        difficulty_results["per_task_difficulty_by_domain"][domain] = domain_per_task

        total_domain_tasks = sum(domain_counts.values())
        if total_domain_tasks > 0:
            _print(f"Domain {domain:20} - Total: {total_domain_tasks:3d}, "
                  f"Easy: {domain_counts['easy']:3d} ({domain_counts['easy']/total_domain_tasks*100:5.1f}%), "
                  f"Medium: {domain_counts['medium']:3d} ({domain_counts['medium']/total_domain_tasks*100:5.1f}%), "
                  f"Hard: {domain_counts['hard']:3d} ({domain_counts['hard']/total_domain_tasks*100:5.1f}%)")

    # Calculate overall totals
    overall_counts = {"easy": 0, "medium": 0, "hard": 0}
    for domain_counts in difficulty_results["breakdown"]["by_domain"].values():
        for difficulty, count in domain_counts.items():
            overall_counts[difficulty] += count

    difficulty_results["breakdown"]["overall"] = overall_counts

    total_tasks = sum(overall_counts.values())
    if total_tasks > 0:
        _print(f"\n{'='*60}")
        _print("DIFFICULTY BREAKDOWN")
        _print(f"{'='*60}")
        _print(f"Total Tasks: {total_tasks}")
        _print(f"Easy: {overall_counts['easy']} ({overall_counts['easy']/total_tasks*100:.1f}%)")
        _print(f"Medium: {overall_counts['medium']} ({overall_counts['medium']/total_tasks*100:.1f}%)")
        _print(f"Hard: {overall_counts['hard']} ({overall_counts['hard']/total_tasks*100:.1f}%)")
        _print(f"{'='*60}")

    return difficulty_results


def compare_difficulty_breakdowns(
    difficulty_breakdown_1,
    difficulty_breakdown_2,
    label_1="Experiment 1",
    label_2="Experiment 2"
):
    """
    Compare two difficulty breakdowns and analyze transitions.

    Args:
        difficulty_breakdown_1: First difficulty breakdown from calculate_difficulty_breakdown
        difficulty_breakdown_2: Second difficulty breakdown from calculate_difficulty_breakdown
        label_1: Label for first experiment
        label_2: Label for second experiment

    Returns:
        dict: Transition analysis including raw counts and statistics
    """

    # Create lookup dictionaries for easy access
    tasks_1 = {(task["domain"], task["example_id"]): task["difficulty"]
              for task in difficulty_breakdown_1["per_task_difficulty"]}
    tasks_2 = {(task["domain"], task["example_id"]): task["difficulty"]
              for task in difficulty_breakdown_2["per_task_difficulty"]}

    # Find common tasks
    common_tasks = set(tasks_1.keys()) & set(tasks_2.keys())

    if len(common_tasks) == 0:
        print("No common tasks found between the two experiments")
        return None

    print(f"Analyzing transitions for {len(common_tasks)} common tasks")

    # Initialize transition matrix
    difficulties = ["easy", "medium", "hard"]
    transitions = defaultdict(lambda: defaultdict(int))
    transitions_by_domain = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Count transitions
    for domain, example_id in common_tasks:
        diff_1 = tasks_1[(domain, example_id)]
        diff_2 = tasks_2[(domain, example_id)]

        transitions[diff_1][diff_2] += 1
        transitions_by_domain[domain][diff_1][diff_2] += 1

    # Create results structure
    results = {
        "experiments": [label_1, label_2],
        "common_tasks": len(common_tasks),
        "transitions": {
            "overall": dict(transitions),
            "by_domain": {domain: dict(domain_trans) for domain, domain_trans in transitions_by_domain.items()}
        },
        "breakdown_comparison": {
            label_1: difficulty_breakdown_1["breakdown"],
            label_2: difficulty_breakdown_2["breakdown"]
        },
        "original_breakdowns": {
            "breakdown_1": difficulty_breakdown_1,
            "breakdown_2": difficulty_breakdown_2
        }
    }

    # Print transition matrix
    print(f"\n{'='*60}")
    print(f"DIFFICULTY TRANSITIONS: {label_1} -> {label_2}")
    print(f"{'='*60}")

    # Overall transition matrix
    print("Overall Transition Matrix:")
    print(f"{'From/To':<10} {'Easy':<8} {'Medium':<8} {'Hard':<8} {'Total':<8}")
    print("-" * 50)

    for from_diff in difficulties:
        row_total = sum(transitions[from_diff][to_diff] for to_diff in difficulties)
        row_str = f"{from_diff.capitalize():<10}"
        for to_diff in difficulties:
            count = transitions[from_diff][to_diff]
            pct = f"({count/row_total*100:.1f}%)" if row_total > 0 else "(0%)"
            row_str += f"{count:<3}{pct:<5} "
        row_str += f"{row_total:<8}"
        print(row_str)

    # Calculate key metrics
    stable_count = sum(transitions[diff][diff] for diff in difficulties)
    improved_count = transitions["medium"]["easy"] + transitions["hard"]["easy"] + transitions["hard"]["medium"]
    degraded_count = transitions["easy"]["medium"] + transitions["easy"]["hard"] + transitions["medium"]["hard"]

    print(f"\nSummary:")
    print(f"Stable (no change): {stable_count}/{len(common_tasks)} ({stable_count/len(common_tasks)*100:.1f}%)")
    print(f"Improved: {improved_count}/{len(common_tasks)} ({improved_count/len(common_tasks)*100:.1f}%)")
    print(f"Degraded: {degraded_count}/{len(common_tasks)} ({degraded_count/len(common_tasks)*100:.1f}%)")

    return results


def mcnemar_test(difficulty_breakdown_1, difficulty_breakdown_2, alpha=0.05):
    """
    McNemar's test for reliability transitions (binary: reliable vs not reliable).

    Args:
        difficulty_breakdown_1: First experiment breakdown
        difficulty_breakdown_2: Second experiment breakdown
        alpha: Significance level

    Returns:
        dict: McNemar test results
    """

    # Create lookup dictionaries for success counts
    tasks_1 = {(task["domain"], task["example_id"]): task["success_count"]
              for task in difficulty_breakdown_1["per_task_difficulty"]}
    tasks_2 = {(task["domain"], task["example_id"]): task["success_count"]
              for task in difficulty_breakdown_2["per_task_difficulty"]}

    # Find common tasks
    common_tasks = set(tasks_1.keys()) & set(tasks_2.keys())

    if len(common_tasks) == 0:
        return {'error': 'No common tasks found'}

    # Define reliability: reliable = success_count == 3, not reliable = success_count < 3
    reliable_1 = {task: (tasks_1[task] == 3) for task in common_tasks}
    reliable_2 = {task: (tasks_2[task] == 3) for task in common_tasks}

    # Count transitions
    b = 0  # reliable -> not reliable
    c = 0  # not reliable -> reliable

    for task in common_tasks:
        if reliable_1[task] and not reliable_2[task]:
            b += 1  # reliable -> not reliable
        elif not reliable_1[task] and reliable_2[task]:
            c += 1  # not reliable -> reliable

    # McNemar's test
    if b + c == 0:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False,
                'b': b, 'c': c, 'n_changed': 0}

    # Test statistic: chi2 = (b - c)^2 / (b + c)
    chi2_stat = (b - c) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, 1)
    significant = p_value < alpha

    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'significant': significant,
        'b': b,  # reliable -> not reliable
        'c': c,  # not reliable -> reliable
        'n_changed': b + c
    }


def wilcoxon_signed_rank_test(difficulty_breakdown_1, difficulty_breakdown_2, alpha=0.05):
    """
    Wilcoxon signed-rank test on success count changes.

    Args:
        difficulty_breakdown_1: First experiment breakdown
        difficulty_breakdown_2: Second experiment breakdown
        alpha: Significance level

    Returns:
        dict: Wilcoxon test results
    """

    # Create lookup dictionaries for success counts
    tasks_1 = {(task["domain"], task["example_id"]): task["success_count"]
              for task in difficulty_breakdown_1["per_task_difficulty"]}
    tasks_2 = {(task["domain"], task["example_id"]): task["success_count"]
              for task in difficulty_breakdown_2["per_task_difficulty"]}

    # Find common tasks
    common_tasks = set(tasks_1.keys()) & set(tasks_2.keys())

    if len(common_tasks) == 0:
        return {'error': 'No common tasks found'}

    # Calculate differences: d_i = c_i(after) - c_i(before)
    differences = []
    for task in common_tasks:
        d_i = tasks_2[task] - tasks_1[task]  # after - before
        differences.append(d_i)

    differences = np.array(differences)

    # Remove zero differences (ties)
    non_zero_diffs = differences[differences != 0]

    if len(non_zero_diffs) == 0:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False,
                'n_pairs': len(differences), 'n_ties': len(differences)}

    # Perform Wilcoxon signed-rank test
    try:
        statistic, p_value = scipy_wilcoxon(non_zero_diffs, alternative='two-sided')
        significant = p_value < alpha
    except ValueError as e:
        return {'error': str(e), 'n_pairs': len(differences)}

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'n_pairs': len(differences),
        'n_ties': len(differences) - len(non_zero_diffs),
        'mean_difference': np.mean(differences),
        'raw_differences': differences
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute reliability metrics from experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Paths to the run directories (typically 3 runs for one setting).",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs="+",
        default=None,
        help="Optional second set of run directories to compare against --run-dirs.",
    )
    parser.add_argument(
        "--example-json",
        required=True,
        help="Path to the example-set JSON file (e.g. evaluation_examples/test_nogdrive.json).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum trajectory steps before treating a run as failure (default: 100).",
    )
    parser.add_argument(
        "--missing-as-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat missing result files as failures (default: True). Use --no-missing-as-failure to disable.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.9,
        help="Threshold for binary success (default: 0.9).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed per-run loading output (default: quiet).",
    )

    args = parser.parse_args()
    quiet = not args.verbose

    # ---- Baseline / first setting ----------------------------------------
    baseline_pass1 = calculate_pass_hat_k_for_k(
        target_dirs=args.run_dirs,
        example_json=args.example_json,
        k=1,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        missing_as_failure=args.missing_as_failure,
        quiet=quiet,
    )

    baseline_pass3 = calculate_pass_hat_k_for_k(
        target_dirs=args.run_dirs,
        example_json=args.example_json,
        k=3,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        missing_as_failure=args.missing_as_failure,
        quiet=quiet,
    )

    # ---- Comparison setting (optional) ------------------------------------
    new_pass1 = None
    new_pass3 = None

    if args.compare_dirs:
        new_pass1 = calculate_pass_hat_k_for_k(
            target_dirs=args.compare_dirs,
            example_json=args.example_json,
            k=1,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
            missing_as_failure=args.missing_as_failure,
            quiet=quiet,
        )

        new_pass3 = calculate_pass_hat_k_for_k(
            target_dirs=args.compare_dirs,
            example_json=args.example_json,
            k=3,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
            missing_as_failure=args.missing_as_failure,
            quiet=quiet,
        )

    # ---- Summary table ----------------------------------------------------
    print()
    print(f"{'Setting':<12}| {'Pass^1':<8}| {'Pass^3':<8}| {'b-c':<8}| {'Δcx':<8}")
    print("-" * 12 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 8)

    _print_single_row("Baseline", baseline_pass1, baseline_pass3)

    if args.compare_dirs and new_pass1 and new_pass3:
        # Difficulty breakdowns needed for McNemar and Wilcoxon
        baseline_diff = calculate_difficulty_breakdown(
            target_dirs=args.run_dirs,
            example_json=args.example_json,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
            missing_as_failure=args.missing_as_failure,
            quiet=True,
        )

        new_diff = calculate_difficulty_breakdown(
            target_dirs=args.compare_dirs,
            example_json=args.example_json,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
            missing_as_failure=args.missing_as_failure,
            quiet=True,
        )

        bc_str = ""
        dcx_str = ""
        if baseline_diff and new_diff:
            mcnemar_res = mcnemar_test(baseline_diff, new_diff)
            b_val = mcnemar_res.get("b", 0)
            c_val = mcnemar_res.get("c", 0)
            # Paper convention: b-c positive = more tasks became reliable
            # Code: b=reliable->not, c=not->reliable, so flip to c-b
            bc_diff = c_val - b_val if isinstance(b_val, int) and isinstance(c_val, int) else "?"
            mc_sig = "*" if mcnemar_res.get("significant", False) else ""
            bc_str = f"{bc_diff}{mc_sig}"

            wilcox_res = wilcoxon_signed_rank_test(baseline_diff, new_diff)
            w_mean = wilcox_res.get("mean_difference", float("nan"))
            w_sig = "*" if wilcox_res.get("significant", False) else ""
            dcx_str = f"{w_mean:.3f}{w_sig}"

        _print_single_row("New", new_pass1, new_pass3, bc_str, dcx_str)

    print()


def _print_single_row(label, pass1_results, pass3_results, bc_str="", dcx_str=""):
    p1 = pass1_results["overall"]["metric"] / 100.0 if pass1_results else float("nan")
    p3 = pass3_results["overall"]["metric"] / 100.0 if pass3_results else float("nan")
    print(f"{label:<12}| {p1:<8.3f}| {p3:<8.3f}| {bc_str:<8}| {dcx_str:<8}")


if __name__ == "__main__":
    main()
