# src/engine/cba_validator.py

from src.models.compliance import TradeProposal, ComplianceResult

class CBAValidator:
    def __init__(self, hard_cap: int, first_apron: int):
        self.hard_cap = hard_cap
        self.first_apron = first_apron

    def evaluate_trade(self, proposal: TradeProposal, current_team_payroll: int) -> ComplianceResult:
        violations = []
        score = 1.0  # Start with a perfect score baseline

        # Calculate aggregate outgoing and incoming salaries
        outgoing_salary = sum(player.get("current_salary", 0) for player in proposal.players_outgoing)
        incoming_salary = sum(player.get("current_salary", 0) for player in proposal.players_incoming)

        # Calculate new payroll state
        net_delta = incoming_salary - outgoing_salary
        new_payroll = current_team_payroll + net_delta
        luxury_tax_triggered = new_payroll >= self.first_apron

        # Rule 1: Hard Cap Check (Fatal Violation - Score 0.0)
        if new_payroll > self.hard_cap:
            violations.append(f"Trade pushes payroll to ${new_payroll:,}, exceeding the hard cap of ${self.hard_cap:,}.")
            score = 0.0

        # Rule 2: Basic Salary Matching (Fatal Violation - Score 0.0)
        max_incoming_allowed = (outgoing_salary * 1.25) + 250000
        if incoming_salary > max_incoming_allowed and outgoing_salary > 0:
            violations.append(
                f"Incoming salary (${incoming_salary:,}) exceeds the maximum allowed matching salary (${max_incoming_allowed:,.0f})."
            )
            score = 0.0

        # Rule 3: The 0.5 "High-Risk" Score (Premature Pruning Prevention)
        # If the trade is legal but triggers the luxury apron, we score it 0.5.
        # This allows the BFS to carry it forward as a transitional state.
        if score > 0.0 and luxury_tax_triggered:
            score = 0.5
            violations.append(f"Warning: Trade pushes payroll to ${new_payroll:,}, triggering the First Apron. High-risk transitional state.")

        # The state is only perfectly valid if no violations/warnings were triggered
        is_valid = len(violations) == 0

        return ComplianceResult(
            is_valid=is_valid,
            score=score,
            violation_reasons=violations,
            salary_cap_delta=net_delta,
            luxury_tax_triggered=luxury_tax_triggered
        )