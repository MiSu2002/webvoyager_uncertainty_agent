from enum import Enum
from typing import Dict, Optional, Any


class AgentInterventionMode(Enum):
    DIRECT_EXECUTION = 1
    SELF_VERIFICATION = 2
    ADAPTIVE_REPLANNING = 3
    ASSISTANCE_ESCALATION = 4


class AdaptiveGatingPolicy:
    """
    Governs long-horizon agent control flow based on calibrated trajectory risk R_t.
    """
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.theta_low = thresholds.get('low', 0.3) if thresholds else 0.3
        self.theta_high = thresholds.get('high', 0.7) if thresholds else 0.7
        self.theta_crit = thresholds.get('critical', 0.9) if thresholds else 0.9

    def evaluate_gate(self, r_t: float) -> AgentInterventionMode:
        """
        Maps risk score R_t to an operational intervention mode.
        """
        if r_t < self.theta_low:
            return AgentInterventionMode.DIRECT_EXECUTION
        elif self.theta_low <= r_t < self.theta_high:
            return AgentInterventionMode.SELF_VERIFICATION
        elif self.theta_high <= r_t < self.theta_crit:
            return AgentInterventionMode.ADAPTIVE_REPLANNING
        else:
            return AgentInterventionMode.ASSISTANCE_ESCALATION