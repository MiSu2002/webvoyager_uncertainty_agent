import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from typing import Dict, List, Any
from core.uncertainty_scorer import TrajectoryUncertaintyScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AblationExperimentHarness:
    def __init__(self, task_dataset: List[Dict[str, Any]]):
        self.task_dataset = task_dataset

    def run_variant(self, variant_name: str, weights: Dict[str, float]) -> Dict[str, Any]:
        logging.info(f"Running Ablation Variant: {variant_name} with weights: {weights}")
        
        scorer = TrajectoryUncertaintyScorer(weights=weights)
        success_count = 0
        total_tasks = len(self.task_dataset)

        for task in self.task_dataset:
            history = task.get("history", [])
            rollouts = task.get("rollouts", [])
            subgoal_id = task.get("subgoal_id", None)

            r_t = scorer.compute_trajectory_uncertainty(
                trajectory_history=history,
                candidate_rollouts=rollouts,
                active_subgoal_id=subgoal_id
            )
            
            # If risk is below threshold, consider it a successfully managed/completed task
            if r_t < 0.35:
                success_count += 1

        success_rate = (success_count / total_tasks) * 100.0 if total_tasks > 0 else 0.0
        result = {"Variant": variant_name, "Success Rate (%)": round(success_rate, 2)}
        logging.info(result)
        return result


if __name__ == "__main__":
    # Expanded dataset with nuanced failure conditions
    complex_mock_tasks = [
        {
            "task_id": "task_1_clean",
            "history": [{"observation": {"http_status_code": 200, "dom_text": "success"}}],
            "rollouts": ["click [a]", "click [a]"],
            "subgoal_id": "step_1"
        },
        {
            "task_id": "task_2_silent_error",
            "history": [{"observation": {"http_status_code": 200, "dom_text": "no results found for query"}}],
            "rollouts": ["click [b]", "click [c]", "type random"], # High epistemic entropy
            "subgoal_id": "step_2"
        },
        {
            "task_id": "task_3_cascading_failure",
            "history": [{"observation": {"http_status_code": 500, "dom_text": "internal server error"}}],
            "rollouts": ["click [d]", "click [d]"],
            "subgoal_id": "step_3"
        }
    ]

    harness = AblationExperimentHarness(complex_mock_tasks)

    print("\n================ DETAILED ABLATION RESULTS ================")
    harness.run_variant("Full Model (M_full)", {"relational": 0.4, "epistemic": 0.3, "env": 0.3})
    harness.run_variant("Ablation: No Relational Graph", {"relational": 0.0, "epistemic": 0.5, "env": 0.5})
    harness.run_variant("Ablation: No Epistemic Entropy", {"relational": 0.6, "epistemic": 0.0, "env": 0.4})
    print("===========================================================\n")