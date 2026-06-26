# 1. Standard Library
import os
import json
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional
import random

# 2. Third-Party Packages
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, END

# 3. Local Application Modules
from src.models.compliance import TradeProposal
from src.engine.cba_validator import CBAValidator
from src.data_layer.nba_extractor import NBAMVPExtractor

# Initialize Environment Variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# =====================================================================
# 1. MINIMAL RAG & EMBEDDING SETUP (In-Memory for Speed)
# =====================================================================
# Hardcoded mini-chunks of the CBA text to satisfy chunking/RAG requirements
cba_chunks = [
    "CBA Rule Section 1: The Salary Cap maximum for the 2026 season is $260 million.",
    "CBA Rule Section 2: The Luxury Tax First Apron is triggered at $280 million.",
    "CBA Rule Section 3: Teams over the First Apron cannot acquire players via sign-and-trade."
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Embeddings model loaded: text-embedding-3-small")
print(embeddings)
vector_store = InMemoryVectorStore(embeddings)
print("Vector store initialized with embeddings model: text-embedding-3-small")
print(vector_store)

# Bulk seed the vector database instantly
vector_store.add_texts(cba_chunks)
print("Vector store seeded with CBA chunks.")
print(vector_store)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
print("Retriever created from vector store with search_kwargs: {'k': 1}")
print(retriever)


# =====================================================================
# 2. STATE DEFINITION & MEMORY REQUIREMENT
# =====================================================================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    target_team: str                # Added
    team_name: str                  # Added
    team_payroll: int               # Added
    team_metrics: dict              # Added
    proposed_trades: List[TradeProposal]
    proposed_trade: Optional[TradeProposal]
    proposed_salary: Optional[int]
    current_depth: int
    cba_context: str
    cba_hard_cap: int               # Added to track parsed RAG limits
    cba_first_apron: int            # Added to track parsed RAG limits
    is_compliant: bool
    compliance_violations: List[str]
    loop_count: int
    parse_error: bool               # Added for the routing conditional edge

class CBAMacros(BaseModel):
    hard_cap: int = Field(description="The maximum salary cap allowed in dollars as an integer. Example: 260000000")
    first_apron: int = Field(description="The luxury tax first apron threshold in dollars as an integer.")



class PlayerAssetModel(BaseModel):
    player_id: str
    name: str
    current_salary: int
    contract_years_remaining: int

class TradeProposalModel(BaseModel):
    team_id: str
    players_outgoing: List[PlayerAssetModel]
    players_incoming: List[PlayerAssetModel]
    draft_assets_involved: List[str] = Field(default_factory=list)

class TradeScenarios(BaseModel):
    proposals: List[TradeProposalModel] = Field(
        description="An array containing exactly 3 distinct, creative trade proposals.",
        min_length=3,
        max_length=3
    )

class ExecutiveDecision(BaseModel):
    selected_index: int = Field(
        description="The zero-based index of the chosen trade option (e.g., 0, 1, or 2)."
    )
    justification: str = Field(
        description="A brief, 2-sentence executive summary justifying the strategic choice."
    )
# =====================================================================
# 3. GRAPH NODES (Reasoning, Action, Evaluation)
# =====================================================================
def team_profile_node(state: AgentState):
    """Node: Ingests live team financial and roster telemetry into the graph state."""
    # Defaulting to NYK for the simulation if not dynamically extracted from the prompt
    target_team = state.get("target_team", "NYK")

    try:
        extractor = NBAMVPExtractor()
        team_data = extractor.fetch_basic_team_profile(target_team)

        print(f"[Data Pipeline] Successfully locked live state for {team_data['team_name']}. Payroll: ${team_data['metrics']['total_salary']:,}")

        # Sync the live data into the LangGraph state
        return {
            "team_name": team_data["team_name"],
            "team_payroll": team_data["metrics"]["total_salary"],
            "team_metrics": team_data["metrics"]
        }
    except Exception as e:
        print(f"[Data Pipeline Error] API connection failed: {e}. Defaulting to safe mock data.")
        return {
            "team_name": f"Mock {target_team}",
            "team_payroll": 172000000, # Safe fallback
            "team_metrics": {}
        }




def gm_recommender(state: AgentState):
    """Node 1 (ToT Generator): Spins up 3 parallel trade paths based on context."""
    messages = state.get("messages", [])
    context = state.get("cba_context", "No context retrieved yet.")
    current_depth = state.get("current_depth", 0)

    # Extract live data layer fields injected by team_profile_node
    team_name = state.get("team_name", "Target Team")
    team_payroll = state.get("team_payroll", 172000000)
    team_metrics = state.get("team_metrics", {})

    # Use the message history to gracefully peek at what the data layer found
    roster_hint = f"Current Payroll: ${team_payroll:,}. Strategy Baseline: Off-Rating {team_metrics.get('off_rating', 'N/A')}"

    structured_llm = llm.with_structured_output(TradeScenarios)

    # Enhanced Contextual Prompt
    system_prompt = (
        f"You are an expert Sports GM Assistant engineering roster modifications for the {team_name}.\n"
        f"Team Financial State: {roster_hint}\n"
        f"Analyze this CBA constraint context: {context}\n"
        f"Generate exactly 3 distinct, highly creative trade proposals designed to improve this roster "
        f"while remaining within the strict salary matching parameters."
    )

    try:
        response_payload = structured_llm.invoke([HumanMessage(content=system_prompt)] + messages)

        proposed_trades = []

        # Unpack the Pydantic models back into your original dataclass format for the physics engine
        for prop in response_payload.proposals:
            proposal = TradeProposal(
                team_id=prop.team_id,
                # .model_dump() converts the Pydantic models back to the Dicts your validator expects
                players_outgoing=[p.model_dump() for p in prop.players_outgoing],
                players_incoming=[p.model_dump() for p in prop.players_incoming],
                draft_assets_involved=prop.draft_assets_involved
            )
            proposed_trades.append(proposal)

        primary_proposal = proposed_trades[0]
        primary_salary = sum(p.get("current_salary", 0) for p in primary_proposal.players_incoming)

        # We must manually create an AIMessage for the state history since we used structured_output
        success_msg = AIMessage(content=f"Generated {len(proposed_trades)} compliant trade structures.")

        # Success State
        return {
            "messages": [success_msg],
            "proposed_trades": proposed_trades,
            "proposed_trade": primary_proposal,
            "proposed_salary": primary_salary,
            "current_depth": current_depth + 1,
            "loop_count": state.get("loop_count", 0) + 1,
            "parse_error": False  # CLEAR THE ERROR FLAG
        }

    except Exception as e:
        # If it still fails (e.g., LLM refused to yield 3 items), the circuit breaker handles it
        error_msg = f"SYSTEM ERROR: Failed to match required schema. Exception: {str(e)}"

        return {
            "messages": [AIMessage(content=error_msg)],
            "parse_error": True,  # SET THE ERROR FLAG
            "loop_count": state.get("loop_count", 0) + 1
        }


def rag_tool_node(state: AgentState):
    """Node: Retrieves CBA rules and extracts dynamic constraints."""
    last_message = state["messages"][-1].content

    # Retrieve the raw text from the vector store
    docs = retriever.invoke(last_message)
    retrieved_text = docs[0].page_content if docs else ""

    # Failsafe values in case of total retrieval failure
    extracted_cap = 260000000
    extracted_apron = 280000000

    if retrieved_text:
        try:
            # Bind the Pydantic schema to the LLM for strict structured output
            structured_llm = llm.with_structured_output(CBAMacros)

            prompt = f"Extract the financial constraints from this CBA rule text:\n{retrieved_text}"
            extracted_data = structured_llm.invoke(prompt)

            extracted_cap = extracted_data.hard_cap
            extracted_apron = extracted_data.first_apron
            print(f"[RAG Extractor] Parsed Cap: ${extracted_cap:,} | Apron: ${extracted_apron:,}")

        except Exception as e:
            print(f"[RAG Extractor] Failed to parse integers. Defaulting to failsafe. Error: {e}")

    # Return the text AND the exact integers to the state
    return {
        "cba_context": retrieved_text,
        "cba_hard_cap": extracted_cap,
        "cba_first_apron": extracted_apron
    }

def load_cba_limits() -> tuple[int, int]:
    """Load CBA financial limits from config/cba_rules.json."""
    rules_path = Path(__file__).resolve().parent / "config" / "cba_rules.json"

    with rules_path.open("r", encoding="utf-8") as file:
        rules = json.load(file)

    first_apron = int(rules["first_apron"])
    hard_cap = int(rules.get("hard_cap", rules["second_apron"]))

    return hard_cap, first_apron


def bfs_evaluator(state: AgentState):
    """Node 2: Evaluates parallel trades via BFS and prunes illegal paths."""
    proposed_trades = state.get("proposed_trades", [])

    dynamic_cap = state.get("cba_hard_cap", 260000000)
    dynamic_apron = state.get("cba_first_apron", 280000000)

    # Grabs the $172,000,000 payroll pulled live via NBAMVPExtractor!
    current_payroll = state.get("team_payroll", 172000000)

    if not proposed_trades:
        error_msg = AIMessage(content="SYSTEM ERROR: No trades passed to evaluator.")
        return {"is_compliant": False, "messages": [error_msg]}

    validator = CBAValidator(hard_cap=dynamic_cap, first_apron=dynamic_apron)
    # REMOVED THE MOCK OVERRIDE LINE: current_payroll = 200000000

    surviving_trades = []
    evaluation_logs = []

    # Iterate through the parallel tier of trades
    for idx, trade in enumerate(proposed_trades):
        result = validator.evaluate_trade(trade, current_team_payroll=current_payroll)

        if result.score == 0.0:
            evaluation_logs.append(f"Path {idx + 1} KILLED (Score 0.0): {result.violation_reasons[0]}")
        else:
            status = "APPROVED (1.0)" if result.score == 1.0 else "HIGH-RISK (0.5)"
            surviving_trades.append(trade)
            evaluation_logs.append(f"Path {idx + 1} SURVIVED {status}.")

    summary_text = "BFS Tier Evaluation:\n" + "\n".join(evaluation_logs)
    is_compliant = len(surviving_trades) > 0

    return {
        "proposed_trades": surviving_trades,
        "messages": [AIMessage(content=summary_text)],
        "is_compliant": is_compliant,
        "compliance_violations": [log for log in evaluation_logs if "KILLED" in log]
    }


def executive_gm_agent(state: AgentState):
    """Node 3 (The Decision): Executive GM reviews surviving ToT paths and selects the winner programmatically."""
    surviving_trades = state.get("proposed_trades", [])
    messages = state.get("messages", [])

    # NEW: Pull team metrics to inform strategic decision-making
    team_name = state.get("team_name", "Target Team")
    team_metrics = state.get("team_metrics", {})
    off_rating = team_metrics.get("off_rating", 110.0)

    # Failsafe
    if not surviving_trades:
        return {"messages": [
            AIMessage(content="Executive GM: All parallel trade paths failed compliance. No viable moves available.")]}

    # Format options with deeply contextual team impact
    options_text = "SURVIVING COMPLIANT TRADES FOR REVIEW:\n"
    for idx, trade in enumerate(surviving_trades):
        incoming_names = [p.get("name", "Unknown Player") for p in trade.players_incoming]
        outgoing_names = [p.get("name", "Unknown Player") for p in trade.players_outgoing]
        incoming_salary = sum(p.get("current_salary", 0) for p in trade.players_incoming)

        options_text += (
            f"Index [{idx}]:\n"
            f"  - Outgoing: {', '.join(outgoing_names)}\n"
            f"  - Incoming: {', '.join(incoming_names)}\n"
            f"  - Financial Footprint: Net Incoming Salary ${incoming_salary:,}\n"
        )

    # Contextualized Executive Prompt
    system_prompt = (
        f"You are the franchise's Executive General Manager for the {team_name}.\n"
        f"Your current baseline offensive efficiency rating is {off_rating}. You want to maximize value.\n"
        f"Review these legally validated, CBA-compliant trade options:\n\n{options_text}\n"
        f"Select the absolute best strategic path that balances financial compliance with roster improvement. "
        f"You must return the exact index number of your choice and a brief, 2-sentence executive summary justifying the strategic choice."
    )

    structured_llm = llm.with_structured_output(ExecutiveDecision)

    try:
        decision = structured_llm.invoke([HumanMessage(content=system_prompt)] + messages)
        safe_index = decision.selected_index if 0 <= decision.selected_index < len(surviving_trades) else 0
        winning_trade = surviving_trades[safe_index]

        final_message = AIMessage(
            content=f"Executive Decision (Selected Index {safe_index}): {decision.justification}"
        )

        return {
            "messages": [final_message],
            "proposed_trades": [winning_trade]
        }

    except Exception as e:
        error_msg = f"SYSTEM ERROR: Executive GM failed to return a valid schema. Defaulting to first safe trade. Error: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "proposed_trades": [surviving_trades[0]]
        }

# =====================================================================
# 3.5 HUMAN ESCALATION NODE (Failsafe)
# =====================================================================

def human_escalation_node(state: AgentState):
    """Node: Failsafe triggered. Halts execution for human recertification."""
    messages = state.get("messages", [])

    # In a real UI, this would pause the graph and wait for a user interrupt.
    # For the script, we log the catastrophic failure and cleanly exit.
    alert = "CRITICAL: Agent exceeded maximum retry loops. State corrupted or LLM caught in failure loop. Requires human recertification."

    return {
        "messages": [AIMessage(content=alert)],
        "is_compliant": False
    }


# =====================================================================
# 4. CONDITIONAL ROUTING EDGE
# =====================================================================
def route_compliance(state: AgentState):
    """Evaluates the state payload to choose the next step."""
    if state["is_compliant"]:
        return "approved"
    if state["loop_count"] >= 3:
        return "force_terminate"  # Prevents runaway LLM spending credits
    return "recalculate"

def route_after_generation(state: AgentState):
    """Evaluates if the generation yielded valid JSON before hitting the physics engine."""
    if state.get("parse_error", False):
        if state.get("loop_count", 0) >= 3:
            return "escalate"  # Circuit breaker tripped
        return "retry"         # Send back to gm_recommender to fix the JSON
    return "validate"          # JSON is pristine, send to BFS

# =====================================================================
# 5. COMPILING THE STATE MACHINE
# =====================================================================
# --- ROUTING & COMPILATION ---
workflow = StateGraph(AgentState)

workflow.add_node("rag_tool", rag_tool_node)
workflow.add_node("team_profile", team_profile_node) # NEW: Register the data node
workflow.add_node("gm_recommender", gm_recommender)
workflow.add_node("bfs_evaluator", bfs_evaluator)
workflow.add_node("executive_gm", executive_gm_agent)
workflow.add_node("human_escalation", human_escalation_node)

# START -> RAG
workflow.set_entry_point("rag_tool")

# RAG -> DATA LAYER
workflow.add_edge("rag_tool", "team_profile")

# DATA LAYER -> GENERATOR
workflow.add_edge("team_profile", "gm_recommender")

# THE NEW SWITCHBOARD: Intercept the flow before it hits the BFS Evaluator
workflow.add_conditional_edges(
    "gm_recommender",
    route_after_generation,
    {
        "validate": "bfs_evaluator",
        "retry": "gm_recommender",
        "escalate": "human_escalation"
    }
)

# Route from the evaluator
workflow.add_conditional_edges(
    "bfs_evaluator",
    route_compliance,
    {
        "approved": "executive_gm",
        "force_terminate": "human_escalation", # Route loop-outs here instead of END
        "recalculate": "gm_recommender"
    }
)

workflow.add_edge("executive_gm", END)
workflow.add_edge("human_escalation", END) # Cleanly terminate after escalation

app = workflow.compile()

# --- EXECUTION SCRIPT ---
if __name__ == "__main__":
    # 1. List of exciting target teams to explore dynamically
    nba_teams = ["NYK", "BOS", "LAL", "GSW", "MIA", "PHX", "DEN", "MIL"]
    selected_team = random.choice(nba_teams)

    # 2. List of distinct target scenarios to vary the LLM's prompt context
    scenarios = [
        f"We need to clear cap space and find a trading partner for {selected_team}.",
        f"The {selected_team} front office wants to acquire an elite floor-spacing wing player.",
        f"Build a multi-player depth trade structure for {selected_team} focusing on frontcourt physical upgrades.",
        f"Construct a block-buster salary-matching framework to bring an all-star to {selected_team}."
    ]
    selected_prompt = random.choice(scenarios)

    # 3. Initialize the state with the dynamically randomized target team and prompt
    initial_input = {
        "target_team": selected_team,  # Triggers your NBAMVPExtractor for this specific team!
        "messages": [HumanMessage(content=selected_prompt)]
    }

    print(f"\n🎲 Launching Random Simulation Tier...")
    print(f"🎯 Target Team: {selected_team}")
    print(f"📝 Prompt Context: {selected_prompt}\n")

    # Invoke the LangGraph State Machine
    final_state = app.invoke(initial_input)

    print("\n--- FINAL INTEGRATION: EXECUTION COMPLETE ---")
    print(f"Executive Decision: {final_state['messages'][-1].content}")
    print(f"\nLocked State Trade Count: {len(final_state.get('proposed_trades', []))}")