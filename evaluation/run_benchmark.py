import json
import logging
from typing import List, Dict, Any
from agents.uncertainty_agent import UncertaintyCalibratedAgent
from agents.base_react import StandardReActAgent
from evaluation.metrics import compute_task_success_rate, compute_trajectory_checkpoint_ece

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WebVoyagerBenchmarkHarness:
    """
    Evaluation harness to run and benchmark long-horizon web navigation agents.
    """
    def __init__(self, task_dataset_path: str, agent_type: str = "uncertainty_calibrated"):
        self.tasks = self._load_tasks(task_dataset_path)
        self.agent_type = agent_type
        self.results: List[Dict[str, Any]] = []

    def _load_tasks(self, path: str) -> List[Dict[str, Any]]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("Task dataset not found. Using mock task for demonstration.")
            return [{"task_id": "mock_001", "goal": "Find the cheapest flight to Tokyo", "start_url": "https://example.com"}]

    def run_evaluation(self, max_steps: int = 15) -> Dict[str, float]:
        logging.info(f"Starting evaluation using agent type: {self.agent_type}")
        
        # Initialize agent based on configuration
        thresholds = {'low': 0.3, 'high': 0.7, 'critical': 0.9}
        if self.agent_type == "uncertainty_calibrated":
            agent = UncertaintyCalibratedAgent(llm_model=None, thresholds=thresholds)
        else:
            agent = StandardReActAgent(llm_model=None)

        all_predictions = []
        all_outcomes = []

        for task in self.tasks:
            task_id = task.get("task_id")
            goal = task.get("goal")
            logging.info(f"Running Task [{task_id}]: {goal}")

            step_count = 0
            task_success = False
            current_obs = {"dom_text": "Welcome to home page", "http_status_code": 200}

            while step_count < max_steps:
                step_count += 1
                
                # Execute agent step
                if self.agent_type == "uncertainty_calibrated":
                    # Captures risk score and action decision
                    action = agent.step(current_obs)
                    # Simulated checkpoint risk tracking for ECE calculation
                    all_predictions.append(0.2 if step_count < 3 else 0.8)
                else:
                    action = agent.step(current_obs)
                    all_predictions.append(0.5)

                # Simulate environment feedback (In real testbed, use Playwright browser driver)
                if step_count >= 5:
                    task_success = True  # Mock completion criterion
                    all_outcomes.append(0) # 0 = Success
                    break
                else:
                    all_outcomes.append(1) # 1 = Ongoing / Failure risk point

            self.results.append({
                "task_id": task_id,
                "success": task_success,
                "steps_taken": step_count
            })

        # Compute Final Benchmark Metrics
        success_rate = compute_task_success_rate(self.results)
        tc_ece = compute_trajectory_checkpoint_ece(all_predictions, all_outcomes)

        metrics_summary = {
            "Task Success Rate": success_rate,
            "Trajectory-Checkpoint ECE": tc_ece
        }
        logging.info(f"Evaluation Complete. Summary Metrics: {metrics_summary}")
        return metrics_summary


if __name__ == "__main__":
    harness = WebVoyagerBenchmarkHarness(task_dataset_path="webvoyager_tasks.json", agent_type="uncertainty_calibrated")
    harness.run_evaluation()