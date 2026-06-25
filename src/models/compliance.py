# C:\Users\skfot\PycharmProjects\sports_gm_forecaster\models\compliance.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass(frozen=True)
class PlayerAsset:
    player_id: str
    name: str
    current_salary: int
    contract_years_remaining: int

@dataclass
class TradeProposal:
    team_id: str
    players_outgoing: List[Dict[str, Any]]
    players_incoming: List[Dict[str, Any]]
    draft_assets_involved: List[str] = field(default_factory=list)

@dataclass
class ComplianceResult:
    is_valid: bool
    score: float  # NEW ADDITION: 1.0 = Perfect, 0.5 = High-Risk, 0.0 = Illegal
    violation_reasons: List[str] = field(default_factory=list)
    salary_cap_delta: int = 0
    luxury_tax_triggered: bool = False
    suggested_adjustments: Optional[str] = None