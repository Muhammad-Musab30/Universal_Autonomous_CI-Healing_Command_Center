"""Repair Confidence Scorer & Risk-Tiered Autonomy Engine.

Calculates a transparent 0-to-100 Repair Confidence Score with explicit point breakdowns:
  +25 pts: Original failure deterministically reproduced before patch.
  +30 pts: Exact failing test AND full test suite pass after patch (with counterfactual revert verification).
  +20 pts: Minimal-Diff Enforcement (edits restricted to implicated files, diff < 50 lines, explicit justification).
  +15 pts: Static safety checks pass (ast.parse valid, zero deleted asserts, zero broad excepts, no secrets).
  +10 pts: Matches a known pattern or team convention in memory.json.
  -60 pts: Immediate penalty if assertions are deleted, unrelated files are modified, or tests fail.

Risk-Tiered Autonomy:
  Score 85–100: HIGH_CONFIDENCE_AUTO_READY (Ready for 1-click Human-Approved PR creation)
  Score 50–84:  MEDIUM_CONFIDENCE_REVIEW_REQUIRED (Diagnosis + suggested patch with caution warnings)
  Score 0–49:   LOW_CONFIDENCE_ESCALATE_ONLY (Refuses merge; provides root-cause investigation only)
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ConfidenceBreakdown(BaseModel):
    deterministic_repro: int = 0          # max 25
    tests_and_counterfactual: int = 0      # max 30
    minimal_diff_enforced: int = 0         # max 20
    static_safety_passed: int = 0          # max 15
    memory_convention_matched: int = 0     # max 10
    penalties: int = 0                     # -60 if violations
    total_score: float = 0.0               # 0.0 to 100.0
    risk_tier: str = "LOW_CONFIDENCE_ESCALATE_ONLY"
    reasoning: List[str] = Field(default_factory=list)

    # Aliases ensuring frontend & API consistency
    reproduction_pts: int = 0
    test_suite_pts: int = 0
    minimal_diff_pts: int = 0
    static_safety_pts: int = 0
    memory_match_pts: int = 0
    penalties_pts: int = 0
    justification: str = ""


class ConfidenceScorer:
    """
    Transparent confidence scoring engine mapping evidence, test results,
    patch quality, and team conventions into Risk-Tiered Autonomy.
    """

    @classmethod
    def calculate_score(
        cls,
        initial_failure_reproduced: bool = True,
        tests_passed: bool = True,
        counterfactual_verified: bool = True,
        quality_gate_passed: bool = True,
        quality_gate_details: Optional[Dict[str, Any]] = None,
        diff_line_count: int = 0,
        target_file_is_implicated: bool = True,
        has_justification: bool = True,
        memory_matched: bool = False,
        deleted_assertions: bool = False,
        unrelated_files_modified: bool = False
    ) -> ConfidenceBreakdown:
        reasoning: List[str] = []
        q_details = quality_gate_details or {}
        checklist = q_details.get("checklist", {})

        # 1. Deterministic Failure Reproduction (+25 pts)
        if initial_failure_reproduced:
            repro_pts = 25
            reasoning.append("+25 pts: Original failure deterministically reproduced in isolated sandbox.")
        else:
            repro_pts = 0
            reasoning.append("0 pts: Initial test run did not deterministically reproduce failure.")

        # 2. Test Suite & Counterfactual Proof (+30 pts)
        if tests_passed and counterfactual_verified:
            test_pts = 30
            reasoning.append("+30 pts: Tests passed with full 4-step counterfactual causal proof (fail->pass->fail on revert).")
        elif tests_passed:
            test_pts = 15
            reasoning.append("+15 pts: Tests passed in sandbox without full counterfactual revert check.")
        else:
            test_pts = 0
            reasoning.append("0 pts: Tests failed to pass after patch application.")

        # 3. Minimal-Diff Enforcement (+20 pts)
        # Criteria: diff < 50 lines, target file is implicated, and explicit justification provided
        is_minimal_diff = diff_line_count < 50 or checklist.get("minimal_diff", False)
        is_implicated = target_file_is_implicated and checklist.get("only_implicated_files", True)
        has_just = has_justification or checklist.get("has_justification", True)

        if is_minimal_diff and is_implicated and has_just:
            diff_pts = 20
            reasoning.append(f"+20 pts: Minimal-Diff Enforced ({diff_line_count} lines < 50, implicated files only, with justification).")
        elif is_minimal_diff:
            diff_pts = 10
            reasoning.append(f"+10 pts: Diff size is small ({diff_line_count} lines), but partial constraint satisfied.")
        else:
            diff_pts = 0
            reasoning.append(f"0 pts: Diff exceeded minimal budget ({diff_line_count} lines).")

        # 4. Static AST Safety Checks (+15 pts)
        # ast_valid, no_deleted_assertions, no_broad_exceptions, no_todos_or_secrets
        ast_ok = checklist.get("ast_valid", True)
        no_del_assert = checklist.get("no_deleted_assertions", True) and not deleted_assertions
        no_broad = checklist.get("no_broad_exceptions", True)
        no_secrets = checklist.get("no_todos_or_secrets", True)

        if quality_gate_passed and ast_ok and no_del_assert and no_broad and no_secrets:
            safety_pts = 15
            reasoning.append("+15 pts: Static AST safety checks passed (zero syntax errors, zero broad catches, zero secrets).")
        else:
            safety_pts = 0
            reasoning.append("0 pts: Static safety checks raised violations.")

        # 5. Team Memory / Pattern Match (+10 pts)
        if memory_matched:
            memory_pts = 10
            reasoning.append("+10 pts: Patch matches known team convention or historical pattern from memory.json.")
        else:
            memory_pts = 0
            reasoning.append("0 pts: No historical memory convention matched (novel bug).")

        # 6. Immediate Penalties (-60 pts)
        penalty_pts = 0
        if not no_del_assert or deleted_assertions:
            penalty_pts += 60
            reasoning.append("-60 pts PENALTY: Test assertion deletion detected!")
        if not is_implicated or unrelated_files_modified:
            penalty_pts += 60
            reasoning.append("-60 pts PENALTY: Unrelated files modified outside the causal graph!")
        if not tests_passed:
            penalty_pts += 60
            reasoning.append("-60 pts PENALTY: Test suite failing after repair attempt!")

        # Total Calculation
        raw_score = repro_pts + test_pts + diff_pts + safety_pts + memory_pts - penalty_pts
        total_score = max(0.0, min(100.0, float(raw_score)))

        # Risk-Tiered Classification
        if total_score >= 85.0:
            risk_tier = "HIGH_CONFIDENCE_AUTO_READY"
        elif total_score >= 50.0:
            risk_tier = "MEDIUM_CONFIDENCE_REVIEW_REQUIRED"
        else:
            risk_tier = "LOW_CONFIDENCE_ESCALATE_ONLY"

        return ConfidenceBreakdown(
            deterministic_repro=repro_pts,
            tests_and_counterfactual=test_pts,
            minimal_diff_enforced=diff_pts,
            static_safety_passed=safety_pts,
            memory_convention_matched=memory_pts,
            penalties=penalty_pts,
            reproduction_pts=repro_pts,
            test_suite_pts=test_pts,
            minimal_diff_pts=diff_pts,
            static_safety_pts=safety_pts,
            memory_match_pts=memory_pts,
            penalties_pts=-penalty_pts if penalty_pts > 0 else 0,
            total_score=total_score,
            risk_tier=risk_tier,
            reasoning=reasoning,
            justification=" | ".join(reasoning)
        )
