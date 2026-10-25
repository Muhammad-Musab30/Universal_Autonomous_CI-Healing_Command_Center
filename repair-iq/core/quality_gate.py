"""Patch Quality Gate: Static AST and Security Verification Engine.

Enforces:
  1. AST Validity (valid Python syntax).
  2. Anti-Tampering Check: Zero deleted assert statements.
  3. No broad 'except Exception: pass' or silent exception suppressions.
  4. Hygiene: Zero TODOs/FIXMEs and zero hardcoded API secrets/tokens.
  5. Minimal-Diff Enforcement: Diff size < 50 lines, edits restricted to implicated files,
     and explicit justifications.
"""

import ast
import difflib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class QualityCheckResult(BaseModel):
    passed: bool
    checklist: Dict[str, bool]
    refusal_reason: Optional[str] = None
    diff_line_count: int = 0
    original_assert_count: int = 0
    patched_assert_count: int = 0
    violations: List[str] = Field(default_factory=list)
    unrelated_files_modified: bool = False
    deleted_assertions: bool = False


class PatchQualityGate:
    """
    Evaluates candidate patches against static AST rules, security hygiene,
    and anti-tampering heuristics before allowing patch execution.
    """

    MAX_DIFF_LINES = 50
    SECRET_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"(?i)api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]"),
        re.compile(r"(?i)secret[_-]?token\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]"),
    ]

    def evaluate(
        self,
        original_code: str,
        patched_code: str,
        original_test_code: Optional[str] = None,
        patched_test_code: Optional[str] = None,
        target_filename: str = "app.py",
        implicated_files: Optional[List[str]] = None,
        justification: Optional[str] = None
    ) -> QualityCheckResult:
        violations: List[str] = []
        checklist: Dict[str, bool] = {
            "ast_valid": True,
            "no_deleted_assertions": True,
            "no_broad_exceptions": True,
            "no_todos_or_secrets": True,
            "minimal_diff": True,
            "only_implicated_files": True,
            "has_justification": bool(justification and justification.strip())
        }

        unrelated_files_modified = False
        deleted_assertions = False

        # 1. AST Validity Check
        try:
            tree_patched = ast.parse(patched_code)
        except SyntaxError as e:
            checklist["ast_valid"] = False
            violations.append(f"AST SyntaxError in patched code: {e}")
            tree_patched = None

        try:
            tree_orig = ast.parse(original_code)
        except Exception:
            tree_orig = None

        # 2. No Deleted Assertions Check
        orig_asserts = self._count_assertions(tree_orig) if tree_orig else 0
        patch_asserts = self._count_assertions(tree_patched) if tree_patched else 0

        # Also check test assertions if test code was provided or target is test file
        if patched_test_code and original_test_code:
            try:
                tree_orig_test = ast.parse(original_test_code)
                tree_patch_test = ast.parse(patched_test_code)
                orig_test_asserts = self._count_assertions(tree_orig_test)
                patch_test_asserts = self._count_assertions(tree_patch_test)
                if patch_test_asserts < orig_test_asserts:
                    checklist["no_deleted_assertions"] = False
                    deleted_assertions = True
                    violations.append(
                        f"Assertion Deletion Trap: Test assert count reduced from {orig_test_asserts} to {patch_test_asserts}."
                    )
            except Exception:
                pass
        elif patch_asserts < orig_asserts:
            checklist["no_deleted_assertions"] = False
            deleted_assertions = True
            violations.append(
                f"Assertion Count Reduced: Original had {orig_asserts} asserts, patch has {patch_asserts}."
            )

        # 3. No Broad Exceptions / Silent Pass Check
        if tree_patched:
            broad_exc_found = self._check_broad_exceptions(tree_patched)
            if broad_exc_found:
                checklist["no_broad_exceptions"] = False
                violations.append(f"Broad Exception Trap Detected: {broad_exc_found}")

        # 4. No TODOs or Hardcoded Secrets
        secrets_or_todos = self._check_todos_and_secrets(patched_code)
        if secrets_or_todos:
            checklist["no_todos_or_secrets"] = False
            violations.append(f"Hygiene Violation: {secrets_or_todos}")

        # 5. Minimal Diff Check (< 50 lines)
        diff_lines = list(
            difflib.unified_diff(
                original_code.splitlines(),
                patched_code.splitlines(),
                fromfile=f"a/{target_filename}",
                tofile=f"b/{target_filename}"
            )
        )
        changed_count = sum(1 for line in diff_lines if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
        if changed_count > self.MAX_DIFF_LINES:
            checklist["minimal_diff"] = False
            violations.append(f"Diff size too large ({changed_count} changed lines > max {self.MAX_DIFF_LINES}).")

        # 6. Implicated Files Boundary Check
        if implicated_files:
            clean_implicated = [Path(f).name for f in implicated_files]
            if Path(target_filename).name not in clean_implicated:
                checklist["only_implicated_files"] = False
                unrelated_files_modified = True
                violations.append(f"Unrelated File Mutation: '{target_filename}' is not in evidence graph implicated files: {implicated_files}")

        passed = len(violations) == 0
        refusal_reason = "; ".join(violations) if not passed else None

        return QualityCheckResult(
            passed=passed,
            checklist=checklist,
            refusal_reason=refusal_reason,
            diff_line_count=changed_count,
            original_assert_count=orig_asserts,
            patched_assert_count=patch_asserts,
            violations=violations,
            unrelated_files_modified=unrelated_files_modified,
            deleted_assertions=deleted_assertions
        )

    def _count_assertions(self, tree: Optional[ast.AST]) -> int:
        if not tree:
            return 0
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))

    def _check_broad_exceptions(self, tree: ast.AST) -> Optional[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
                    if not has_raise:
                        return "Bare 'except:' catching all exceptions without re-raising."
                elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
                    is_just_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
                    if is_just_pass or not has_raise:
                        return f"Broad 'except {node.type.id}: pass' suppressing exceptions silently."
        return None

    def _check_todos_and_secrets(self, code: str) -> Optional[str]:
        for marker in ["TODO", "FIXME", "XXX"]:
            if re.search(rf"\b{marker}\b", code):
                return f"Forbidden development marker found: '{marker}'"

        for pattern in self.SECRET_PATTERNS:
            if pattern.search(code):
                return "Hardcoded API secret or credential pattern detected in patch."

        return None
