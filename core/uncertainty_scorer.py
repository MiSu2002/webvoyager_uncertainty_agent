import math
from typing import Dict, List, Any, Optional
import numpy as np


class RelationalGraph:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[tuple] = []

    def add_subgoal(self, node_id: str, description: str, prerequisites: List[str] = None):
        self.nodes[node_id] = {"description": description, "status": "pending", "risk": 0.0}
        if prerequisites:
            for p_id in prerequisites:
                if p_id in self.nodes:
                    self.edges.append((p_id, node_id))

    def update_node_status(self, node_id: str, status: str, risk_score: float):
        if node_id in self.nodes:
            self.nodes[node_id]["status"] = status
            self.nodes[node_id]["risk"] = risk_score
            self._propagate_risk(node_id)

    def _propagate_risk(self, updated_node_id: str):
        updated_risk = self.nodes[updated_node_id]["risk"]
        for parent, child in self.edges:
            if parent == updated_node_id:
                self.nodes[child]["risk"] = max(self.nodes[child]["risk"], updated_risk * 0.85)

    def compute_relational_discrepancy(self) -> float:
        if not self.nodes:
            return 0.0
        total_risk = sum(node["risk"] for node in self.nodes.values())
        return min(1.0, total_risk / len(self.nodes))


class TrajectoryUncertaintyScorer:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {"relational": 0.4, "epistemic": 0.3, "env": 0.3}
        self.relational_graph = RelationalGraph()

    def _compute_env_risk(self, current_observation: Dict[str, Any]) -> float:
        dom_text = current_observation.get("dom_text", "").lower()
        error_logs = current_observation.get("error_logs", [])
        http_status = current_observation.get("http_status_code", 200)

        risk_score = 0.0
        if http_status >= 400:
            risk_score += 0.8
        
        error_keywords = ["404 not found", "no results found", "access denied", "item unavailable"]
        for kw in error_keywords:
            if kw in dom_text:
                risk_score += 0.4
                break

        if error_logs:
            risk_score += 0.5

        return min(1.0, risk_score)

    def _compute_epistemic_uncertainty(self, candidate_actions: List[str]) -> float:
        if not candidate_actions or len(candidate_actions) < 2:
            return 0.0
        unique_ratio = len(set(candidate_actions)) / len(candidate_actions)
        return float(max(0.0, min(1.0, unique_ratio - (1.0 / len(candidate_actions)))))

    def compute_trajectory_uncertainty(
        self,
        trajectory_history: List[Dict[str, Any]],
        candidate_rollouts: Optional[List[str]] = None,
        active_subgoal_id: Optional[str] = None
    ) -> float:
        if not trajectory_history:
            return 0.0

        current_obs = trajectory_history[-1].get("observation", {})
        u_env = self._compute_env_risk(current_obs)

        if active_subgoal_id:
            status = "failed" if u_env > 0.6 else "satisfied"
            self.relational_graph.update_node_status(active_subgoal_id, status, risk_score=u_env)
        
        u_relational = self.relational_graph.compute_relational_discrepancy()
        u_epistemic = self._compute_epistemic_uncertainty(candidate_rollouts or [])

        r_t = (
            self.weights["relational"] * u_relational +
            self.weights["epistemic"] * u_epistemic +
            self.weights["env"] * u_env
        )
        return float(max(0.0, min(1.0, r_t)))