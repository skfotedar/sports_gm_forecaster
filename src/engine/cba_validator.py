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

        # Determine apron status BEFORE the trade to apply proper matching rules
        already_above_apron = current_team_payroll >= self.first_apron
        luxury_tax_triggered = new_payroll >= self.first_apron

        # Rule 1: Hard Cap Check (Fatal Violation - Score 0.0)
        if new_payroll > self.hard_cap:
            violations.append(
                f"Trade pushes payroll to ${new_payroll:,}, exceeding the hard cap of ${self.hard_cap:,}.")
            score = 0.0

        # Rule 2: Dynamic Salary Matching based on Apron Tier status
        if outgoing_salary > 0:
            if already_above_apron:
                # First Apron restrictions: Cannot take back more salary than you send out ($0 cushion)
                max_incoming_allowed = outgoing_salary
                rule_text = "First Apron Restriction (100% max incoming)"
            else:
                # Standard tax-paying/non-tax team matching rule
                max_incoming_allowed = (outgoing_salary * 1.25) + 250000
                rule_text = "Standard Cushion (125% + $250k)"

            if incoming_salary > max_incoming_allowed:
                violations.append(
                    f"Illegal Salary Matching under {rule_text}: Incoming salary (${incoming_salary:,}) "
                    f"exceeds maximum allowed (${max_incoming_allowed:,.0f}) for outgoing salary of (${outgoing_salary:,})."
                )
                score = 0.0

        # Rule 3: The 0.5 "High-Risk" Score (Premature Pruning Prevention)
        if score > 0.0 and luxury_tax_triggered:
            score = 0.5
            violations.append(
                f"Warning: Trade pushes payroll to ${new_payroll:,}, triggering the First Apron. High-risk transitional state.")

        is_valid = len(violations) == 0

        return ComplianceResult(
            is_valid=is_valid,
            score=score,
            violation_reasons=violations,
            salary_cap_delta=net_delta,
            luxury_tax_triggered=luxury_tax_triggered
        )