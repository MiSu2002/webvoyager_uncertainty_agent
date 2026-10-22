import numpy as np
from typing import List, Dict, Any


def compute_task_success_rate(results: List[Dict[str, Any]]) -> float:
    """
    Computes percentage of successfully completed tasks.
    """
    if not results:
        return 0.0
    successes = sum(1 for r in results if r.get("success", False))
    return successes / len(results)


def compute_trajectory_checkpoint_ece(
    predictions: List[float], 
    outcomes: List[int], 
    num_bins: int = 10
) -> float:
    """
    Computes Trajectory-Checkpoint Expected Calibration Error (TC-ECE).
    
    Args:
        predictions: List of estimated risk scores R_t at checkpoints.
        outcomes: Binary indicators of actual failure (1 = failure, 0 = success).
        num_bins: Number of confidence/risk bins.
    """
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n_samples = len(predictions)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            bin_accuracy = np.mean(outcomes[in_bin])  # Empirical failure rate
            bin_confidence = np.mean(predictions[in_bin])  # Estimated risk R_t
            ece += (bin_count / n_samples) * np.abs(bin_confidence - bin_accuracy)

    return float(ece)