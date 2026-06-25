# src/graph/state.py

from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from src.models.compliance import TradeProposal

class AgentState(TypedDict):
    # ==========================================
    # 1. LIVE TEAM TELEMETRY (Data Layer)
    # ==========================================
    target_team: str
    team_name: str
    team_payroll: int
    team_metrics: dict

    # ==========================================
    # 2. CBA CONTEXT (RAG Layer)
    # ==========================================
    cba_context: str
    cba_hard_cap: int
    cba_first_apron: int

    # ==========================================
    # 3. GRAPH MEMORY & FLOW CONTROL
    # ==========================================
    messages: Annotated[List[BaseMessage], add_messages]
    current_depth: int
    loop_count: int
    parse_error: bool

    # ==========================================
    # 4. TREE OF THOUGHTS STATE (Generation Layer)
    # ==========================================
    proposed_trades: List[TradeProposal]

    # ==========================================
    # 5. COMPLIANCE & OUTPUT (Execution Layer)
    # ==========================================
    is_compliant: bool
    proposed_trade: Optional[TradeProposal]
    proposed_salary: Optional[int]
    compliance_violations: List[str]