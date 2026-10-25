"""RepairIQ Core Engine: Autonomous Causal CI-Healing & PR Verification System."""

from .router import ModelRouter, TokenCostTracker, ModelTier, FailureTaxonomy
from .evidence import (
    EvidenceCollector,
    SandboxRunner,
    EvidenceGraph,
    FailureNode,
    FlakyDetectionResult,
    SecurityError,
    build_evidence_graph
)
from .quality_gate import PatchQualityGate, QualityCheckResult
from .verifier import CounterfactualVerifier, CounterfactualProof
from .orchestrator import RepairOrchestrator, RepairDossier
from .workspace_manager import WorkspaceManager, WorkspaceInfo
from .confidence import ConfidenceScorer, ConfidenceBreakdown
from .github_client import GitHubClient

__all__ = [
    "ModelRouter",
    "TokenCostTracker",
    "ModelTier",
    "FailureTaxonomy",
    "EvidenceCollector",
    "SandboxRunner",
    "EvidenceGraph",
    "FailureNode",
    "FlakyDetectionResult",
    "SecurityError",
    "build_evidence_graph",
    "PatchQualityGate",
    "QualityCheckResult",
    "CounterfactualVerifier",
    "CounterfactualProof",
    "RepairOrchestrator",
    "RepairDossier",
    "WorkspaceManager",
    "WorkspaceInfo",
    "ConfidenceScorer",
    "ConfidenceBreakdown",
    "GitHubClient",
]
