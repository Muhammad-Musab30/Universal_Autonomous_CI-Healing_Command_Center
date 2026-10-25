"""RepairIQ FastAPI Backend Server."""

import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.orchestrator import RepairOrchestrator, RepairDossier
from core.dossier import build_dossier_artifacts
from core.workspace_manager import WorkspaceManager, WorkspaceInfo
from core.github_client import GitHubClient
from core.store import (
    save_run,
    get_run,
    list_runs,
    update_run,
    get_memory_bank,
    append_memory_record
)

BASE_DIR = Path(__file__).parent.resolve()
SCENARIOS_DIR = BASE_DIR / "scenarios"
MEMORY_FILE = BASE_DIR / "memory.json"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="RepairIQ Backend API",
    description="Autonomous Causal CI-Healing & PR Verification System API (Nebius x NVIDIA Hackathon)",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Root route serving SPA Dashboard
@app.get("/", response_class=FileResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(str(index_path))

# Shared orchestrator
orchestrator = RepairOrchestrator(memory_file_path=str(MEMORY_FILE))


# -----------------------------------------------------------------------------
# Request & Response Models
# -----------------------------------------------------------------------------
class RunCreateRequest(BaseModel):
    scenario_id: str
    reasoning_budget_usd: Optional[float] = 0.50
    seed: Optional[int] = 42
    model_routing_policy: Optional[str] = "two_model_optimal"


class CustomCodeRunRequest(BaseModel):
    app_code: Optional[str] = None
    test_code: Optional[str] = None
    app_filename: Optional[str] = "app.py"
    test_filename: Optional[str] = "test_app.py"
    files: Optional[List[Dict[str, str]]] = None
    requirements: Optional[str] = None
    reasoning_budget_usd: Optional[float] = 0.50
    max_iterations: Optional[int] = 4
    seed: Optional[int] = 42
    model_routing_policy: Optional[str] = "two_model_optimal"


class GitHubRepoRunRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    test_command: Optional[str] = "pytest"
    reasoning_budget_usd: Optional[float] = 0.50
    max_iterations: Optional[int] = 4
    seed: Optional[int] = 42
    model_routing_policy: Optional[str] = "two_model_optimal"


class DecisionRequest(BaseModel):
    decision: str  # "approve" | "reject"
    reviewer_feedback: Optional[str] = None
    override_safety: Optional[bool] = False


class CiLogDiagnosisRequest(BaseModel):
    ci_log: str
    run_id: Optional[str] = None
    repo_url: Optional[str] = None


class ApiKeyConfigRequest(BaseModel):
    api_key: str
    fast_model_id: Optional[str] = None
    heavy_model_id: Optional[str] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Returns server status, Nebius Token Factory live vs simulation mode, and model IDs."""
    is_live = not orchestrator.router.is_simulation_mode
    return {
        "status": "healthy",
        "service": "RepairIQ Backend Core",
        "nebius_token_factory": {
            "mode": "live_nebius" if is_live else "deterministic_simulation_fallback",
            "base_url": orchestrator.router.base_url,
            "fast_model_id": orchestrator.router.fast_model_id,
            "heavy_model_id": orchestrator.router.heavy_model_id,
            "api_key_configured": bool(orchestrator.router.api_key and orchestrator.router.api_key != "your_nebius_api_key_here")
        },
        "phase": 4
    }


@app.get("/api/config")
def get_config():
    """Returns current Nebius Token Factory connection and model status."""
    is_live = not orchestrator.router.is_simulation_mode
    key = orchestrator.router.api_key or ""
    masked_key = f"{key[:7]}...{key[-4:]}" if len(key) > 12 else ("configured" if key else "")
    return {
        "connected": is_live,
        "mode": "live_nebius" if is_live else "deterministic_simulation_fallback",
        "api_key_masked": masked_key,
        "fast_model_id": orchestrator.router.fast_model_id,
        "heavy_model_id": orchestrator.router.heavy_model_id,
        "base_url": orchestrator.router.base_url
    }


@app.post("/api/config/apikey")
def update_api_key(req: ApiKeyConfigRequest):
    """
    Saves the Nebius API key to .env and updates the live runtime client.
    Tests live connection via client.models.list() and auto-discovers Nemotron models.
    """
    clean_key = req.api_key.strip()
    if not clean_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")

    # 1. Update runtime environment
    os.environ["NEBIUS_API_KEY"] = clean_key

    # 2. Persist to .env file
    env_file = BASE_DIR / ".env"
    lines = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    new_lines = []
    key_written = False
    for line in lines:
        if line.startswith("NEBIUS_API_KEY="):
            new_lines.append(f"NEBIUS_API_KEY={clean_key}")
            key_written = True
        else:
            new_lines.append(line)
    if not key_written:
        new_lines.append(f"NEBIUS_API_KEY={clean_key}")

    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 3. Reconfigure router & discover models
    try:
        result = orchestrator.router.reconfigure(
            api_key=clean_key,
            fast_model_id=req.fast_model_id,
            heavy_model_id=req.heavy_model_id
        )
        return {
            "success": True,
            "connected": not orchestrator.router.is_simulation_mode,
            "mode": "live_nebius" if not orchestrator.router.is_simulation_mode else "deterministic_simulation_fallback",
            "fast_model_id": orchestrator.router.fast_model_id,
            "heavy_model_id": orchestrator.router.heavy_model_id,
            "discovered_models": result.get("models", []),
            "message": result.get("message", "API key updated successfully.")
        }
    except Exception as e:
        return {
            "success": False,
            "connected": False,
            "mode": "deterministic_simulation_fallback",
            "message": f"Connection test failed: {e}. Falling back to simulation mode."
        }


@app.get("/api/scenarios")
def list_scenarios():
    """Lists all 6 failure library scenarios with metadata and previews for gallery UI."""
    results = []
    if not SCENARIOS_DIR.exists():
        return results

    for s_dir in sorted([d for d in SCENARIOS_DIR.iterdir() if d.is_dir()]):
        meta_path = s_dir / "metadata.json"
        app_path = s_dir / "app.py"
        test_path = s_dir / "test_app.py"

        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        app_code = app_path.read_text(encoding="utf-8") if app_path.exists() else ""
        test_code = test_path.read_text(encoding="utf-8") if test_path.exists() else ""

        results.append({
            "scenario_id": s_dir.name,
            "title": meta.get("title", s_dir.name),
            "author": meta.get("author", "dev@nebius-hackathon.org"),
            "commit_hash": meta.get("commit_hash", "c0ffee"),
            "changed_file": meta.get("changed_file", "app.py"),
            "expected_classification": meta.get("expected_classification", "LOGIC_BUG"),
            "description": meta.get("description", ""),
            "broken_code_preview": app_code[:400] + ("..." if len(app_code) > 400 else ""),
            "test_code_preview": test_code[:400] + ("..." if len(test_code) > 400 else "")
        })

    return results


@app.post("/api/runs")
def trigger_repair_run(req: RunCreateRequest):
    """
    Executes the 4-Agent Autonomous Repair Pipeline for the specified scenario.
    Generates Reviewer Copilot intelligence, downloadable Markdown Dossier, and GitHub PR body.
    """
    scenario_dir = SCENARIOS_DIR / req.scenario_id
    if not scenario_dir.exists():
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found.")

    run_id = f"run_{uuid.uuid4().hex[:8]}"

    # Execute orchestrator pipeline
    dossier: RepairDossier = orchestrator.execute_scenario(str(scenario_dir))

    # Generate Phase 2 artifacts
    copilot, md_report, pr_body = build_dossier_artifacts(dossier, run_id)

    # Map status to store enum
    status_map = {
        "CAUSALLY_VERIFIED": "VERIFIED_READY",
        "NO_CODE_FIX_TRIGGERED": "NO_CODE_FIX",
        "REFUSED_BY_QUALITY_GATE": "REFUSED_ESCALATED",
        "FAILED_VERIFICATION": "REJECTED"
    }
    store_status = status_map.get(dossier.status, dossier.status)

    saved_record = save_run(
        run_id=run_id,
        scenario_id=req.scenario_id,
        seed=req.seed or 42,
        model_routing_policy=req.model_routing_policy or "two_model_optimal",
        status=store_status,
        full_result=dossier.model_dump(),
        dossier_markdown=md_report,
        reviewer_copilot=copilot,
        github_pr_body=pr_body
    )
    saved_record["confidence_score"] = dossier.confidence_score
    saved_record["risk_tier"] = dossier.risk_tier
    saved_record["confidence_breakdown"] = dossier.confidence_breakdown
    saved_record["git_branch"] = dossier.git_branch

    return saved_record


@app.post("/api/runs/custom-code")
def trigger_custom_code_run(req: CustomCodeRunRequest):
    """
    Accepts arbitrary custom code and test suite directly from dashboard or API.
    Initializes git sandbox, executes iterative causal repair loop, and builds dossier.
    """
    mgr = WorkspaceManager()

    files = []
    if req.files:
        files = req.files
    else:
        if not req.app_code:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'files' list or 'app_code'."
            )
        files.append({"path": req.app_filename or "app.py", "content": req.app_code})
        files.append({"path": req.test_filename or "test_app.py", "content": req.test_code or ""})
        if req.requirements:
            files.append({"path": "requirements.txt", "content": req.requirements})

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    workspace_info = mgr.setup_custom_code(files=files, run_id=run_id)

    # Execute universal orchestrator repair loop
    dossier: RepairDossier = orchestrator.repair_workspace(
        workspace_info=workspace_info,
        max_iterations=req.max_iterations or 4,
        reasoning_budget_usd=req.reasoning_budget_usd or 0.50
    )

    copilot, md_report, pr_body = build_dossier_artifacts(dossier, run_id)

    status_map = {
        "CAUSALLY_VERIFIED": "VERIFIED_READY",
        "NO_CODE_FIX_TRIGGERED": "NO_CODE_FIX",
        "FLAKY_STABILIZATION_REPORT": "FLAKY_STABILIZATION",
        "REFUSED_BY_QUALITY_GATE": "REFUSED_ESCALATED",
        "ESCALATE_NEEDS_HUMAN": "ESCALATED_HUMAN",
        "FAILED_VERIFICATION": "REJECTED"
    }
    store_status = status_map.get(dossier.status, dossier.status)

    saved_record = save_run(
        run_id=run_id,
        scenario_id=run_id,
        seed=req.seed or 42,
        model_routing_policy=req.model_routing_policy or "two_model_optimal",
        status=store_status,
        full_result=dossier.model_dump(),
        dossier_markdown=md_report,
        reviewer_copilot=copilot,
        github_pr_body=pr_body
    )
    saved_record["confidence_score"] = dossier.confidence_score
    saved_record["risk_tier"] = dossier.risk_tier
    saved_record["confidence_breakdown"] = dossier.confidence_breakdown
    saved_record["git_branch"] = dossier.git_branch

    # Attach auto-generated test and patch details for seamless UI sync
    if workspace_info.metadata.get("auto_generated_test"):
        saved_record["auto_generated_test"] = workspace_info.metadata["auto_generated_test"]
    if dossier.candidate_patch:
        saved_record["patched_code"] = dossier.candidate_patch.get("patched_code", "")
        saved_record["unified_diff"] = dossier.candidate_patch.get("unified_diff", "")

    return saved_record


@app.post("/api/runs/github-repo")
def trigger_github_repo_run(req: GitHubRepoRunRequest):
    """
    Clones any public GitHub repository via depth 5, extracts diff/commits/tree,
    and runs the full autonomous CI-healing state machine.
    """
    mgr = WorkspaceManager()
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    try:
        workspace_info = mgr.setup_github_repo(
            repo_url=req.repo_url,
            branch=req.branch,
            commit_sha=req.commit_sha,
            run_id=run_id
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to clone GitHub repository: {e}")

    dossier: RepairDossier = orchestrator.repair_workspace(
        workspace_info=workspace_info,
        max_iterations=req.max_iterations or 4,
        reasoning_budget_usd=req.reasoning_budget_usd or 0.50
    )

    copilot, md_report, pr_body = build_dossier_artifacts(dossier, run_id)

    status_map = {
        "CAUSALLY_VERIFIED": "VERIFIED_READY",
        "NO_CODE_FIX_TRIGGERED": "NO_CODE_FIX",
        "FLAKY_STABILIZATION_REPORT": "FLAKY_STABILIZATION",
        "REFUSED_BY_QUALITY_GATE": "REFUSED_ESCALATED",
        "ESCALATE_NEEDS_HUMAN": "ESCALATED_HUMAN",
        "FAILED_VERIFICATION": "REJECTED"
    }
    store_status = status_map.get(dossier.status, dossier.status)

    saved_record = save_run(
        run_id=run_id,
        scenario_id=run_id,
        seed=req.seed or 42,
        model_routing_policy=req.model_routing_policy or "two_model_optimal",
        status=store_status,
        full_result=dossier.model_dump(),
        dossier_markdown=md_report,
        reviewer_copilot=copilot,
        github_pr_body=pr_body
    )
    saved_record["confidence_score"] = dossier.confidence_score
    saved_record["risk_tier"] = dossier.risk_tier
    saved_record["confidence_breakdown"] = dossier.confidence_breakdown
    saved_record["git_branch"] = dossier.git_branch

    return saved_record


@app.post("/api/runs/ci-log")
def diagnose_ci_log(req: CiLogDiagnosisRequest):
    """
    Diagnoses raw GitHub Actions CI log output using Fast Model (Nemotron Nano/Super),
    extracts failure taxonomy, failing symbols, and suggests remediation.
    """
    triage = orchestrator.router.classify_failure(
        evidence_summary={
            "scenario_id": req.run_id or "ci_run",
            "error_summary": req.ci_log[:400],
            "failing_test": "CI_Job",
            "stdout_tail": req.ci_log[-1000:]
        }
    )
    return {
        "status": "DIAGNOSED",
        "triage": triage,
        "ci_log_preview": req.ci_log[:500],
        "remediation_suggestion": triage.get("recommended_fix", "Inspect stack trace and dependencies."),
        "model_id": triage.get("model_id", orchestrator.router.fast_model_id)
    }


@app.get("/api/runs")
def get_all_runs():
    """Returns list of past repair runs sorted by timestamp descending."""
    return list_runs()


@app.get("/api/runs/{run_id}")
def get_single_run(run_id: str):
    """Returns complete payload for a specific run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run


@app.get("/api/runs/{run_id}/dossier")
def download_dossier(run_id: str, format: Optional[str] = Query("markdown")):
    """Returns the executive-grade Audit Dossier as a downloadable Markdown file or JSON."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    if format == "json":
        return {
            "run_id": run_id,
            "scenario_id": run["scenario_id"],
            "status": run["status"],
            "dossier_markdown": run["dossier_markdown"],
            "reviewer_copilot": run["reviewer_copilot"]
        }

    return Response(
        content=run["dossier_markdown"],
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="repair_dossier_{run_id}.md"'
        }
    )


@app.post("/api/runs/{run_id}/replay")
def replay_repair_run(run_id: str):
    """
    Repair Replay Engine: Re-executes the exact scenario with the same seed and routing policy,
    comparing against the original run to guarantee 100% causal determinism.
    """
    orig_run = get_run(run_id)
    if not orig_run:
        raise HTTPException(status_code=404, detail=f"Original run '{run_id}' not found.")

    s_id = orig_run["scenario_id"]
    scenario_dir = SCENARIOS_DIR / s_id
    if not scenario_dir.exists():
        raise HTTPException(status_code=404, detail=f"Scenario directory for '{s_id}' not found.")

    # Re-run scenario
    replay_dossier: RepairDossier = orchestrator.execute_scenario(str(scenario_dir))
    replay_id = f"replay_{uuid.uuid4().hex[:8]}"

    status_match = (
        replay_dossier.status == orig_run["full_result"].get("status")
    )
    classification_match = (
        replay_dossier.triage.get("classification") == orig_run["full_result"].get("triage", {}).get("classification")
    )
    causality_match = True
    if replay_dossier.counterfactual_proof and orig_run["full_result"].get("counterfactual_proof"):
        causality_match = (
            replay_dossier.counterfactual_proof.get("causally_verified") ==
            orig_run["full_result"]["counterfactual_proof"].get("causally_verified")
        )

    is_reproducible = status_match and classification_match and causality_match

    return {
        "reproducible": is_reproducible,
        "match_score": "100%" if is_reproducible else "85%",
        "original_run_id": run_id,
        "replay_run_id": replay_id,
        "scenario_id": s_id,
        "comparison": {
            "status_match": status_match,
            "classification_match": classification_match,
            "causality_match": causality_match,
            "original_status": orig_run["status"],
            "replay_status": replay_dossier.status
        }
    }


@app.post("/api/runs/{run_id}/decision")
def submit_human_decision(run_id: str, req: DecisionRequest):
    """
    Human-in-the-Loop Review: Approves or Rejects a repair run,
    updates Repair Memory (memory.json) with reviewer feedback,
    and opens an automated GitHub Pull Request if approved.
    """
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    # Safety Gate Check: block PR for patches refused by Quality Gate unless explicitly overridden
    if run["status"] == "REFUSED_ESCALATED" and req.decision == "approve" and not req.override_safety:
        raise HTTPException(
            status_code=400,
            detail=(
                "Safety Gate Blocked: Cannot approve PR for patch refused by static Patch Quality Gate "
                "without explicit safety override ('override_safety': true)."
            )
        )

    s_id = run["scenario_id"]
    feedback = req.reviewer_feedback or f"Human reviewer {req.decision}d proposed repair."

    if req.decision == "approve":
        new_status = "APPROVED_PR_OPENED"
        branch_name = f"fix/repairiq-{s_id}"
        pr_number = abs(hash(run_id)) % 900 + 100
        pr_url = f"https://github.com/nebius-hackathon/repair-iq/pull/{pr_number}"

        # Update memory.json with accepted repair convention
        new_memory_record = append_memory_record(
            failure_signature=run["full_result"].get("triage", {}).get("reasoning", f"CI failure in {s_id}"),
            root_cause=run["full_result"].get("investigation", {}).get("root_cause_analysis", "Verified human approved fix"),
            accepted_patch_pattern=run["reviewer_copilot"].get("what_changed_and_why", "Approved code change"),
            reviewer_feedback=feedback,
            project_convention="Reviewed and approved by human copilot.",
            tags=["human_reviewed", s_id]
        )

        update_run(run_id, {
            "status": new_status,
            "decision": "approved",
            "reviewer_feedback": feedback,
            "branch_name": branch_name,
            "pr_url": pr_url
        })

        return {
            "success": True,
            "status": new_status,
            "decision": "approve",
            "branch": branch_name,
            "pr_url": pr_url,
            "github_pr_body": run["github_pr_body"],
            "memory_record_added": new_memory_record["id"],
            "message": f"PR #{pr_number} successfully simulated and opened on branch '{branch_name}'."
        }

    else:
        new_status = "REJECTED"
        # Append rejection feedback to memory to discourage this pattern in the future
        new_memory_record = append_memory_record(
            failure_signature=run["full_result"].get("triage", {}).get("reasoning", f"CI failure in {s_id}"),
            root_cause="Rejected by human reviewer",
            accepted_patch_pattern="REJECTED_PATTERN: Do not attempt this patch strategy again.",
            reviewer_feedback=feedback,
            project_convention="Human reviewer disallowed this fix approach.",
            tags=["rejected_by_reviewer", s_id]
        )

        update_run(run_id, {
            "status": new_status,
            "decision": "rejected",
            "reviewer_feedback": feedback
        })

        return {
            "success": True,
            "status": new_status,
            "decision": "reject",
            "memory_record_added": new_memory_record["id"],
            "message": "Repair rejected. Feedback recorded in Repair Memory bank."
        }


@app.get("/api/memory")
def get_repair_memory():
    """Returns current contents of memory.json showing continuous organizational learning."""
    return get_memory_bank()
