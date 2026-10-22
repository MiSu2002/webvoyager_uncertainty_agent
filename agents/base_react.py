from typing import Dict, List, Any


class StandardReActAgent:
    """
    Standard ReAct Agent baseline that executes actions greedily without uncertainty tracking.
    """
    def __init__(self, llm_model: Any):
        self.llm = llm_model
        self.history: List[Dict[str, Any]] = []

    def step(self, current_observation: Dict[str, Any]) -> str:
        """
        Appends observation to history and generates next action without gating.
        """
        self.history.append({"observation": current_observation})
        
        # Greedy generation without risk management
        action = self.generate_greedy_action(self.history)
        return action

    def generate_greedy_action(self, history: List[Dict[str, Any]]) -> str:
        # Placeholder for standard LLM prompt execution
        return "CLICK [element_id]"