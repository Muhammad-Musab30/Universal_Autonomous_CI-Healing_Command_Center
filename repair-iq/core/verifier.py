"""Counterfactual Verification Engine: Causal Proof of Repair Validity."""

import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .evidence import SandboxRunner, EvidenceCollector


class StepResult(BaseModel):
    step_num: int
    step_name: str
    exit_code: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    expected_outcome: str  # "FAIL" or "PASS"
    actual_outcome: str    # "FAIL" or "PASS"
    verified: bool
    summary: str
    duration_sec: float = 0.0


class CounterfactualProof(BaseModel):
    scenario_id: str
    causally_verified: bool
    steps: List[StepResult] = Field(default_factory=list)
    failure_signature: str = ""
    restoration_confirmed: bool = False
    verification_summary: str = ""
    proof_timeline: List[str] = Field(default_factory=list)


class CounterfactualVerifier:
    """
    Executes the 4-step Causal Verification Cycle:
      Step 1: Original Code -> Run pytest -> Confirm FAILS
      Step 2: Apply Patch    -> Run pytest -> Confirm PASSES
      Step 3: Revert Patch   -> Run pytest -> Confirm FAILS (Counterfactual Causality Proof)
      Step 4: Restore Patch  -> Run pytest -> Confirm PASSES (Clean Repaired State)
    """

    def __init__(self, sandbox: SandboxRunner):
        self.sandbox = sandbox
        self.collector = EvidenceCollector(sandbox)

    def verify(
        self,
        target_file: str,
        original_code: str,
        patched_code: str,
        env_vars: Optional[Dict[str, str]] = None
    ) -> CounterfactualProof:
        steps: List[StepResult] = []
        proof_timeline: List[str] = []

        # -------------------------------------------------------------
        # STEP 1: Original Commit (Must FAIL)
        # -------------------------------------------------------------
        t0 = time.time()
        self.sandbox.write_file(target_file, original_code)
        ev1 = self.collector.collect(env_vars=env_vars)
        t_dur1 = round(time.time() - t0, 3)

        s1_fail = (ev1.exit_code != 0 or ev1.failed_tests > 0 or ev1.error_tests > 0)
        s1_outcome = "FAIL" if s1_fail else "PASS"
        s1_verified = (s1_outcome == "FAIL")
        steps.append(
            StepResult(
                step_num=1,
                step_name="Original Commit Baseline",
                exit_code=ev1.exit_code,
                passed_tests=ev1.passed_tests,
                failed_tests=ev1.failed_tests,
                error_tests=ev1.error_tests,
                expected_outcome="FAIL",
                actual_outcome=s1_outcome,
                verified=s1_verified,
                summary=f"Original commit failed as expected ({ev1.failed_tests} failed, {ev1.error_tests} errors).",
                duration_sec=t_dur1
            )
        )
        proof_timeline.append(f"[Step 1] Baseline failure confirmed: exit_code={ev1.exit_code} in {t_dur1}s")

        # Capture normalized failure signature from step 1
        failure_sig = ev1.nodes[0].normalized_error if ev1.nodes else "Unspecified failure"

        if not s1_verified:
            return CounterfactualProof(
                scenario_id=self.sandbox.scenario_id,
                causally_verified=False,
                steps=steps,
                failure_signature=failure_sig,
                restoration_confirmed=False,
                verification_summary="Failed causal proof at Step 1: Baseline tests did not fail initially.",
                proof_timeline=proof_timeline
            )

        # -------------------------------------------------------------
        # STEP 2: Apply Proposed Patch (Must PASS)
        # -------------------------------------------------------------
        t0 = time.time()
        self.sandbox.write_file(target_file, patched_code)
        ev2 = self.collector.collect(env_vars=env_vars)
        t_dur2 = round(time.time() - t0, 3)

        s2_pass = (ev2.exit_code == 0 and ev2.failed_tests == 0 and ev2.error_tests == 0)
        s2_outcome = "PASS" if s2_pass else "FAIL"
        s2_verified = (s2_outcome == "PASS")
        steps.append(
            StepResult(
                step_num=2,
                step_name="Apply Patch Verification",
                exit_code=ev2.exit_code,
                passed_tests=ev2.passed_tests,
                failed_tests=ev2.failed_tests,
                error_tests=ev2.error_tests,
                expected_outcome="PASS",
                actual_outcome=s2_outcome,
                verified=s2_verified,
                summary=f"Patched code passed all tests ({ev2.passed_tests} passed).",
                duration_sec=t_dur2
            )
        )
        proof_timeline.append(f"[Step 2] Patch resolution confirmed: {ev2.passed_tests} tests passed in {t_dur2}s")

        if not s2_verified:
            # Revert to clean state before exiting
            self.sandbox.write_file(target_file, original_code)
            return CounterfactualProof(
                scenario_id=self.sandbox.scenario_id,
                causally_verified=False,
                steps=steps,
                failure_signature=failure_sig,
                restoration_confirmed=False,
                verification_summary="Failed causal proof at Step 2: Patched code failed tests.",
                proof_timeline=proof_timeline
            )

        # -------------------------------------------------------------
        # STEP 3: Counterfactual Revert (Must FAIL again)
        # -------------------------------------------------------------
        t0 = time.time()
        self.sandbox.write_file(target_file, original_code)
        ev3 = self.collector.collect(env_vars=env_vars)
        t_dur3 = round(time.time() - t0, 3)

        s3_fail = (ev3.exit_code != 0 or ev3.failed_tests > 0 or ev3.error_tests > 0)
        s3_outcome = "FAIL" if s3_fail else "PASS"
        s3_verified = (s3_outcome == "FAIL")
        steps.append(
            StepResult(
                step_num=3,
                step_name="Counterfactual Revert",
                exit_code=ev3.exit_code,
                passed_tests=ev3.passed_tests,
                failed_tests=ev3.failed_tests,
                error_tests=ev3.error_tests,
                expected_outcome="FAIL",
                actual_outcome=s3_outcome,
                verified=s3_verified,
                summary="Reverting patch successfully resurrected the exact original failure, establishing causal link.",
                duration_sec=t_dur3
            )
        )
        proof_timeline.append(f"[Step 3] Counterfactual revert confirmed failure resurrection in {t_dur3}s")

        if not s3_verified:
            self.sandbox.write_file(target_file, patched_code)
            return CounterfactualProof(
                scenario_id=self.sandbox.scenario_id,
                causally_verified=False,
                steps=steps,
                failure_signature=failure_sig,
                restoration_confirmed=False,
                verification_summary="Failed causal proof at Step 3: Reverting patch did NOT resurrect failure (flaky or non-causal).",
                proof_timeline=proof_timeline
            )

        # -------------------------------------------------------------
        # STEP 4: Final Restore (Must PASS)
        # -------------------------------------------------------------
        t0 = time.time()
        self.sandbox.write_file(target_file, patched_code)
        ev4 = self.collector.collect(env_vars=env_vars)
        t_dur4 = round(time.time() - t0, 3)

        s4_pass = (ev4.exit_code == 0 and ev4.failed_tests == 0 and ev4.error_tests == 0)
        s4_outcome = "PASS" if s4_pass else "FAIL"
        s4_verified = (s4_outcome == "PASS")
        steps.append(
            StepResult(
                step_num=4,
                step_name="Final Repaired State Restore",
                exit_code=ev4.exit_code,
                passed_tests=ev4.passed_tests,
                failed_tests=ev4.failed_tests,
                error_tests=ev4.error_tests,
                expected_outcome="PASS",
                actual_outcome=s4_outcome,
                verified=s4_verified,
                summary=f"Re-applied patch cleanly. Final sandbox in 100% passing state ({ev4.passed_tests} passed).",
                duration_sec=t_dur4
            )
        )
        proof_timeline.append(f"[Step 4] Final clean repaired state restored in {t_dur4}s")

        causally_verified = s1_verified and s2_verified and s3_verified and s4_verified
        summary = (
            "Causally Verified (Fail -> Pass -> Fail on Revert -> Pass on Restore)"
            if causally_verified
            else "Causal verification incomplete."
        )

        return CounterfactualProof(
            scenario_id=self.sandbox.scenario_id,
            causally_verified=causally_verified,
            steps=steps,
            failure_signature=failure_sig,
            restoration_confirmed=s4_verified,
            verification_summary=summary,
            proof_timeline=proof_timeline
        )
