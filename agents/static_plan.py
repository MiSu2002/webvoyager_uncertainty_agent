from typing import Dict, List, Any


class StaticPlanAndActAgent:
    """
    Baseline agent that uses a rigid upfront plan or triggers blind replanning 
    at fixed step intervals (e.g., every K steps) regardless of trajectory risk.
    """
    def __init__(self, llm_model: Any, replan_interval: int = 5):
        self.llm = llm_model
        self.replan_interval = replan_interval
        self.step_counter = 0
        self.current_plan = []
        self.history: List[Dict[str, Any]] = []

    def step(self, current_observation: Dict[str, Any]) -> str:
        self.history.append({"observation": current_observation})
        self.step_counter += 1

        # Blind fixed-interval replanning (ignoring actual environmental feedback)
        if self.step_counter % self.replan_interval == 0:
            self.current_plan = self.trigger_blind_replan(self.history)

        action = self.execute_next_plan_step(self.current_plan)
        return action

    def trigger_blind_replan(self, history: List[Dict[str, Any]]) -> List[str]:
        # Placeholder for blind high-level plan generation
        return ["step_1", "step_2", "step_3"]

    def execute_next_plan_step(self, plan: List[str]) -> str:
        return "EXECUTE_FIXED_PLAN_ACTION"