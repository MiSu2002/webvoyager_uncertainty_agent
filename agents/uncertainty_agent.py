from typing import Dict, List, Any
from core.uncertainty_scorer import TrajectoryUncertaintyScorer


class UncertaintyCalibratedAgent:
    def __init__(self, llm_model, thresholds: Dict[str, float]):
        self.llm = llm_model
        self.theta_low = thresholds.get('low', 0.3)
        self.theta_high = thresholds.get('high', 0.7)
        self.theta_crit = thresholds.get('critical', 0.9)
        
        self.uncertainty_scorer = TrajectoryUncertaintyScorer()
        self.history: List[Dict[str, Any]] = []

    def step(self, current_observation: Dict[str, Any], candidate_rollouts: List[str] = None) -> Any:
        self.history.append({"observation": current_observation})
        
        # Compute history-conditioned trajectory risk score R_t
        r_t = self.uncertainty_scorer.compute_trajectory_uncertainty(
            trajectory_history=self.history,
            candidate_rollouts=candidate_rollouts
        )
        
        # Adaptive Gating Policy Φ(R_t)
        if r_t < self.theta_low:
            action = self.llm.generate_action(self.history)
            return self.execute_action(action)
        elif self.theta_low <= r_t < self.theta_high:
            verified_obs = self.trigger_dom_reparse(current_observation)
            self.history[-1]["observation"] = verified_obs
            action = self.llm.generate_action(self.history)
            return self.execute_action(action)
        elif self.theta_high <= r_t < self.theta_crit:
            self.history = self.trigger_adaptive_replan(self.history)
            action = self.llm.generate_action(self.history)
            return self.execute_action(action)
        else:
            return self.trigger_human_or_system_escalation()

    def execute_action(self, action): return action
    def trigger_dom_reparse(self, obs): return obs
    def trigger_adaptive_replan(self, h): return h
    def trigger_human_or_system_escalation(self): return "ESCALATE"