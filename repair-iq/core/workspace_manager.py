"""Workspace Manager: Dynamic Ingestion Engine for RepairIQ.

Supports 3 dynamic input modes:
  Mode 1: Arbitrary GitHub Repository URL (clone, branch/commit checkout, diff & tree extraction)
  Mode 2: Live Custom Code & Test Input (arbitrary files -> local git init -> commit failing state)
  Mode 3: Seeded Reference Repositories (missing_env_repo, flaky_network_repo, logic_regression_repo)
"""

import os
import sys
import ast
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger("RepairIQ.Workspace")


class WorkspaceInfo(BaseModel):
    """Encapsulates all metadata and state for an isolated sandbox workspace."""
    run_id: str
    workspace_dir: str
    mode: str  # "github" | "custom_code" | "seeded"
    commit_hash: str = ""
    recent_commits: List[Dict[str, Any]] = Field(default_factory=list)
    git_diff: str = ""
    file_tree: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.workspace_dir).resolve()


class AutoTestGenerator:
    """
    Intelligently inspects user-provided Python source code and generates
    a comprehensive, idiomatic pytest test suite if test_app.py is
    empty, missing, or mismatched against defined symbols.
    """

    @classmethod
    def extract_defined_symbols(cls, code: str) -> Dict[str, Any]:
        """Extracts top-level functions, classes, and parameters defined in code."""
        symbols = {
            "functions": [],      # list of dicts: {"name": str, "args": list[str]}
            "classes": [],        # list of dicts: {"name": str, "methods": list[str]}
            "has_main": False
        }
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arg_names = [a.arg for a in node.args.args if a.arg != "self"]
                    symbols["functions"].append({"name": node.name, "args": arg_names})
                elif isinstance(node, ast.ClassDef):
                    methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    symbols["classes"].append({"name": node.name, "methods": methods})
                elif isinstance(node, ast.If):
                    if isinstance(node.test, ast.Compare):
                        left = getattr(node.test.left, "id", None)
                        if left == "__name__":
                            symbols["has_main"] = True
        except Exception:
            pass
        return symbols

    @classmethod
    def is_test_mismatched(cls, app_code: str, test_code: Optional[str]) -> tuple[bool, str]:
        """
        Returns (True, reason) if test_code is empty, or imports symbols from app
        that do not exist in app_code, or fails to reference any defined symbol.
        """
        if not test_code or not test_code.strip():
            return True, "test_app.py is empty"

        symbols = cls.extract_defined_symbols(app_code)
        defined_names = set([f["name"] for f in symbols["functions"]] + [c["name"] for c in symbols["classes"]])

        if not defined_names:
            return False, "no symbols defined in app"

        try:
            test_tree = ast.parse(test_code)
        except Exception:
            return True, "test_app.py has syntax errors"

        # Check for imports from app (e.g. from app import allocate_stock)
        imported_from_app = set()
        for node in ast.walk(test_tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in ["app", ".app", "app.py"]:
                    for alias in node.names:
                        if alias.name != "*":
                            imported_from_app.add(alias.name)

        # If test imports names from app that are not defined in app.py:
        if imported_from_app:
            mismatched = imported_from_app - defined_names
            if mismatched:
                return True, f"test_app.py imports non-existent symbol(s) {', '.join(sorted(mismatched))} not found in app.py"

        # Also check if any defined symbol is referenced in test_code
        any_symbol_tested = False
        for name in defined_names:
            if name in test_code:
                any_symbol_tested = True
                break
        if not any_symbol_tested:
            return True, "test_app.py does not test any function or class defined in app.py"

        return False, "test matches app"

    @classmethod
    def generate_test_suite(cls, app_code: str, app_module: str = "app") -> str:
        """
        Generates a robust, production-grade pytest test suite testing edge cases,
        typical inputs, boundary conditions, and __main__ block for app_code.
        """
        symbols = cls.extract_defined_symbols(app_code)
        functions = symbols["functions"]
        classes = symbols["classes"]

        lines = [
            f'"""Auto-generated Comprehensive Test Suite for {app_module}.py."""',
            "import pytest",
        ]

        all_import_names = [f["name"] for f in functions] + [c["name"] for c in classes]
        if all_import_names:
            lines.append(f"from {app_module} import {', '.join(all_import_names)}")
        lines.append("")

        for func in functions:
            fname = func["name"]
            fargs = func["args"]

            # Case A: Sorting algorithms (e.g. bubble_sort, quick_sort, merge_sort, sort_*, *_sort)
            if "sort" in fname.lower():
                lines.extend([
                    f"def test_{fname}_unsorted_list():",
                    f"    sample = [64, 34, 25, 12, 22, 11, 90]",
                    f"    expected = sorted(sample.copy())",
                    f"    result = {fname}(sample.copy())",
                    f"    assert result == expected, f'Expected {{expected}}, got {{result}}'",
                    "",
                    f"def test_{fname}_already_sorted():",
                    f"    sample = [1, 2, 3, 4, 5]",
                    f"    result = {fname}(sample.copy())",
                    f"    assert result == [1, 2, 3, 4, 5]",
                    "",
                    f"def test_{fname}_edge_cases():",
                    f"    assert {fname}([]) == []",
                    f"    assert {fname}([42]) == [42]",
                    f"    assert {fname}([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]",
                    ""
                ])
            # Case B: Search algorithms
            elif "search" in fname.lower():
                lines.extend([
                    f"def test_{fname}_found():",
                    f"    arr = [1, 3, 5, 7, 9, 11]",
                    f"    res = {fname}(arr, 7) if len({fargs}) >= 2 else {fname}(arr)",
                    f"    assert res is not None",
                    "",
                    f"def test_{fname}_not_found():",
                    f"    arr = [1, 3, 5, 7, 9, 11]",
                    f"    res = {fname}(arr, 99) if len({fargs}) >= 2 else {fname}(arr)",
                    f"    assert res in [-1, None, False]",
                    ""
                ])
            # Case C: Math / Calculation / Number functions
            elif any(k in fname.lower() for k in ["calc", "sum", "add", "sub", "multiply", "divide", "fact", "fib", "prime", "discount", "tax", "total"]):
                arg_vals = ["10" if i == 0 else "2" for i in range(len(fargs))]
                call_str = f"{fname}({', '.join(arg_vals)})"
                lines.extend([
                    f"def test_{fname}_standard_input():",
                    f"    result = {call_str}",
                    f"    assert result is not None",
                    "",
                    f"def test_{fname}_edge_zero():",
                    f"    try:",
                    f"        res = {fname}({', '.join(['0' for _ in fargs])})",
                    f"        assert res is not None",
                    f"    except (ValueError, ZeroDivisionError):",
                    f"        pass",
                    ""
                ])
            # Case D: General function
            else:
                arg_vals = []
                for a in fargs:
                    a_low = a.lower()
                    if any(k in a_low for k in ["arr", "list", "items", "nums", "data", "elements"]):
                        arg_vals.append("[1, 2, 3]")
                    elif any(k in a_low for k in ["str", "text", "s", "msg", "line", "name"]):
                        arg_vals.append('"hello"')
                    elif any(k in a_low for k in ["n", "num", "count", "val", "x", "y", "a", "b", "limit", "id"]):
                        arg_vals.append("5")
                    elif any(k in a_low for k in ["flag", "is_", "has_", "enable"]):
                        arg_vals.append("True")
                    elif any(k in a_low for k in ["dict", "map", "config", "payload"]):
                        arg_vals.append('{"key": "val"}')
                    else:
                        arg_vals.append("None")

                call_str = f"{fname}({', '.join(arg_vals)})"
                lines.extend([
                    f"def test_{fname}_execution():",
                    f"    result = {call_str}",
                    f"    assert result is not None",
                    ""
                ])

        for cls_info in classes:
            cname = cls_info["name"]
            lines.extend([
                f"def test_{cname}_instantiation():",
                f"    obj = {cname}()",
                f"    assert obj is not None",
                ""
            ])

        if symbols["has_main"] or "if __name__" in app_code:
            lines.extend([
                "def test_app_main_execution():",
                f"    import importlib",
                f"    mod = importlib.import_module('{app_module}')",
                f"    assert mod is not None",
                ""
            ])

        return "\n".join(lines).strip() + "\n"


class WorkspaceManager:
    """
    Manages isolated working directories inside .sandbox/run_<uuid>/.
    Handles repository cloning, custom code initialization, and seeded templates.
    """

    def __init__(self, sandbox_base_dir: Optional[Union[str, Path]] = None):
        if sandbox_base_dir:
            self.sandbox_base_dir = Path(sandbox_base_dir).resolve()
        else:
            # Default to repair-iq/.sandbox
            self.sandbox_base_dir = (Path(__file__).resolve().parent.parent / ".sandbox").resolve()
        self.sandbox_base_dir.mkdir(parents=True, exist_ok=True)

    def _generate_run_id(self) -> str:
        return f"run_{uuid.uuid4().hex[:12]}"

    def _build_file_tree(self, root_dir: Path) -> List[str]:
        """Recursively lists all repository files ignoring VCS, caches, and virtualenvs."""
        ignored_names = {
            ".git", "node_modules", ".venv", "venv", "__pycache__",
            ".pytest_cache", ".ruff_cache", ".mypy_cache", ".coverage",
            ".idea", ".vscode", ".DS_Store"
        }
        file_list: List[str] = []
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in ignored_names]
            for f in files:
                if f in ignored_names or f.endswith((".pyc", ".pyo")):
                    continue
                full_path = Path(root) / f
                rel_path = full_path.relative_to(root_dir)
                file_list.append(str(rel_path).replace("\\", "/"))
        return sorted(file_list)

    def _extract_git_metadata(self, workspace_path: Path) -> tuple[str, List[Dict[str, Any]], str]:
        """Extracts commit hash, recent commits, and git diff safely."""
        commit_hash = "unknown"
        recent_commits: List[Dict[str, Any]] = []
        git_diff = ""

        # 1. Commit Hash
        try:
            res_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10
            )
            if res_hash.returncode == 0:
                commit_hash = res_hash.stdout.strip()
        except Exception:
            pass

        # 2. Recent Commits (up to 5)
        try:
            res_log = subprocess.run(
                ["git", "log", "-n", "5", "--pretty=format:%H|%an|%ae|%s|%cr"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10
            )
            if res_log.returncode == 0 and res_log.stdout.strip():
                for line in res_log.stdout.strip().splitlines():
                    parts = line.split("|")
                    if len(parts) >= 5:
                        recent_commits.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "email": parts[2],
                            "message": parts[3],
                            "relative_time": parts[4]
                        })
                    else:
                        recent_commits.append({"raw": line})
        except Exception:
            pass

        # 3. Git Diff (HEAD~1 or HEAD)
        try:
            res_diff = subprocess.run(
                ["git", "diff", "HEAD~1"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10
            )
            if res_diff.returncode == 0 and res_diff.stdout.strip():
                git_diff = res_diff.stdout.strip()
            else:
                # If only 1 commit exists or HEAD~1 fails, get diff of HEAD commit
                res_show = subprocess.run(
                    ["git", "show", "--stat", "--patch", "HEAD"],
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10
                )
                if res_show.returncode == 0:
                    git_diff = res_show.stdout.strip()
        except Exception:
            git_diff = ""

        return commit_hash, recent_commits, git_diff

    # =========================================================================
    # MODE 1: Arbitrary GitHub Repository URL
    # =========================================================================
    @staticmethod
    def discover_repository_source_files(workspace_path: Path) -> Dict[str, Any]:
        """
        Scans a cloned repository and discovers:
          - primary_file: most important application entrypoint
          - python_files: list of all non-test python files
          - test_files: list of all test files
          - has_tests: boolean
          - other_source_files: cpp, c, java, js, ts, go, rust, etc.
        """
        ignored_dirs = {
            ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
            ".mypy_cache", ".coverage", ".idea", ".vscode", ".DS_Store", "node_modules",
            "dist", "build", ".next", ".nuxt", "site-packages", "target", "vendor"
        }

        source_exts = {
            ".py", ".cpp", ".c", ".cc", ".cxx", ".h", ".hpp",
            ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs"
        }

        py_files: List[str] = []
        test_files: List[str] = []
        other_files: List[str] = []

        for root, dirs, files in os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            for f in files:
                if f.startswith(".") or f.endswith((".pyc", ".pyo", ".min.js", ".min.css")):
                    continue
                full_p = Path(root) / f
                try:
                    rel_p = str(full_p.relative_to(workspace_path)).replace("\\", "/")
                except Exception:
                    continue

                ext = full_p.suffix.lower()
                if ext not in source_exts:
                    continue

                is_test = (
                    f.startswith("test_") or f.endswith("_test.py") or
                    ".test." in f or ".spec." in f or
                    "/tests/" in rel_p or "/test/" in rel_p or "\\tests\\" in str(full_p)
                )

                if is_test:
                    test_files.append(rel_p)
                elif ext == ".py":
                    py_files.append(rel_p)
                else:
                    other_files.append(rel_p)

        # Prioritize key entrypoints across Python, C/C++, Java, JS/TS, Go, Rust
        priority_names = [
            "app.py", "main.py", "server.py", "interview_engine.py", "evaluator.py",
            "index.py", "cli.py", "router.py", "core.py", "api.py", "model.py", "utils.py",
            "main.cpp", "quick_sort.cpp", "main.c", "app.cpp", "server.cpp",
            "Main.java", "App.java", "Application.java",
            "index.js", "server.js", "app.js", "index.ts", "server.ts", "app.ts",
            "main.go", "main.rs", "lib.rs"
        ]

        primary_file = None
        for name in priority_names:
            matched_py = next((p for p in py_files if Path(p).name.lower() == name.lower()), None)
            if matched_py:
                primary_file = matched_py
                break
            matched_other = next((p for p in other_files if Path(p).name.lower() == name.lower()), None)
            if matched_other:
                primary_file = matched_other
                break

        if not primary_file:
            if py_files:
                # Prefer files at root directory, then by size
                sorted_py = sorted(py_files, key=lambda p: (0 if "/" not in p else 1, -(workspace_path / p).stat().st_size if (workspace_path / p).exists() else 0))
                primary_file = sorted_py[0]
            elif other_files:
                sorted_other = sorted(other_files, key=lambda p: (0 if "/" not in p else 1, -(workspace_path / p).stat().st_size if (workspace_path / p).exists() else 0))
                primary_file = sorted_other[0]
            elif test_files:
                primary_file = test_files[0]
            else:
                primary_file = "app.py"

        return {
            "primary_file": primary_file,
            "python_files": py_files,
            "test_files": test_files,
            "has_tests": len(test_files) > 0,
            "other_source_files": other_files
        }

    def setup_github_repo(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        commit_sha: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> WorkspaceInfo:
        """
        Clones any public GitHub repository into an isolated sandbox using depth 5.
        Universal branch-agnostic cloning:
          1. Tries user-specified branch with --branch.
          2. On failure, cleans up destination and retries without --branch to clone default branch.
          3. Wraps all subprocess calls in try/except blocks to prevent crashes.
        """
        # Normalize repository URL
        clean_url = repo_url.strip()
        if clean_url.startswith("git@github.com:"):
            clean_url = "https://github.com/" + clean_url[len("git@github.com:"):]
        elif not clean_url.startswith("http://") and not clean_url.startswith("https://"):
            clean_url = "https://" + clean_url
        if clean_url.endswith(".git"):
            clean_url = clean_url[:-4]

        r_id = run_id or self._generate_run_id()
        workspace_path = (self.sandbox_base_dir / r_id).resolve()
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)
        self.sandbox_base_dir.mkdir(parents=True, exist_ok=True)

        target_branch = branch.strip() if branch and branch.strip() and branch.strip().lower() not in ["default", "head"] else None
        clone_success = False
        last_error = ""

        # Attempt 1: If branch is specified, try cloning that branch directly
        if target_branch:
            try:
                cmd = ["git", "clone", "--depth", "5", "--branch", target_branch, clean_url, str(workspace_path)]
                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                if res.returncode == 0 and workspace_path.exists() and any(workspace_path.iterdir()):
                    clone_success = True
                else:
                    last_error = res.stderr or res.stdout or f"exit code {res.returncode}"
            except Exception as e:
                last_error = str(e)

        # Attempt 2: If Attempt 1 failed or no branch was specified, clean directory and clone repository default branch
        if not clone_success:
            if workspace_path.exists():
                shutil.rmtree(workspace_path, ignore_errors=True)
            try:
                cmd = ["git", "clone", "--depth", "5", clean_url, str(workspace_path)]
                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                if res.returncode == 0 and workspace_path.exists() and any(workspace_path.iterdir()):
                    clone_success = True
                else:
                    err_msg = res.stderr or res.stdout or last_error or "Unknown git clone failure"
                    raise RuntimeError(f"Failed to clone GitHub repository '{clean_url}': {err_msg}")
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                raise RuntimeError(f"Git clone error for '{clean_url}': {e}")

        # Optional commit checkout
        if commit_sha:
            try:
                res_co = subprocess.run(
                    ["git", "checkout", commit_sha],
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20
                )
                if res_co.returncode != 0:
                    logger.warning(f"Could not checkout commit {commit_sha}: {res_co.stderr}")
            except Exception as e:
                logger.warning(f"Error checking out commit {commit_sha}: {e}")

        # Determine active branch
        active_branch = "default"
        try:
            res_br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(workspace_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            if res_br.returncode == 0 and res_br.stdout.strip():
                active_branch = res_br.stdout.strip()
        except Exception:
            pass

        commit_hash, recent_commits, git_diff = self._extract_git_metadata(workspace_path)
        file_tree = self._build_file_tree(workspace_path)
        discovered = self.discover_repository_source_files(workspace_path)

        metadata = {
            "scenario_id": r_id,
            "mode": "github",
            "repo_url": clean_url,
            "branch": active_branch,
            "requested_branch": target_branch or "default",
            "commit_sha": commit_sha or commit_hash,
            "commit_hash": commit_hash,
            "author": recent_commits[0].get("author", "GitHub User") if recent_commits else "GitHub User",
            "primary_file": discovered["primary_file"],
            "changed_file": discovered["primary_file"],
            "has_tests": discovered["has_tests"],
            "test_files": discovered["test_files"],
            "python_files": discovered["python_files"]
        }

        return WorkspaceInfo(
            run_id=r_id,
            workspace_dir=str(workspace_path),
            mode="github",
            commit_hash=commit_hash,
            recent_commits=recent_commits,
            git_diff=git_diff,
            file_tree=file_tree,
            metadata=metadata
        )

    # =========================================================================
    # MODE 2: Live Custom Code & Test Input
    # =========================================================================
    def setup_custom_code(
        self,
        files: List[Dict[str, str]],
        run_id: Optional[str] = None,
        commit_message: str = "Initial failing state",
        author_name: str = "RepairIQ Agent",
        author_email: str = "ci@repairiq.dev",
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkspaceInfo:
        """
        Accepts arbitrary files: list[dict] (e.g. [{"path": "app.py", "content": "..."}]).
        Initializes a real local git repository inside .sandbox/run_<uuid>/ and commits the
        initial broken state so git diff, git status, and branch creation work identically to a cloned repo.
        """
        if not files:
            raise ValueError("Files list cannot be empty for custom code setup.")

        meta = metadata or {}

        # Check for app.py and test_app.py to detect empty or mismatched tests
        app_file_entry = next((f for f in files if f.get("path") in ["app.py", "main.py"] or (f.get("path", "").endswith(".py") and not Path(f.get("path", "")).name.startswith("test_"))), None)
        test_file_entry = next((f for f in files if f.get("path") in ["test_app.py", "tests/test_app.py"] or Path(f.get("path", "")).name.startswith("test_")), None)

        if app_file_entry:
            app_code = app_file_entry.get("content", "")
            test_code = test_file_entry.get("content", "") if test_file_entry else ""
            is_mismatched, reason = AutoTestGenerator.is_test_mismatched(app_code, test_code)
            if is_mismatched:
                logger.info(f"Auto-generating test suite for custom code: {reason}")
                gen_suite = AutoTestGenerator.generate_test_suite(app_code, app_module="app")
                if test_file_entry:
                    test_file_entry["content"] = gen_suite
                else:
                    test_file_entry = {"path": "test_app.py", "content": gen_suite}
                    files.append(test_file_entry)
                meta["auto_generated_test"] = gen_suite
                meta["test_generator_reason"] = reason

        r_id = run_id or self._generate_run_id()
        workspace_path = (self.sandbox_base_dir / r_id).resolve()
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)
        workspace_path.mkdir(parents=True, exist_ok=True)

        # 1. Write all files to disk
        for file_entry in files:
            rel_path = file_entry.get("path")
            content = file_entry.get("content", "")
            if not rel_path:
                continue
            target_file = workspace_path / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding="utf-8")

        # 2. Initialize real git repository
        subprocess.run(["git", "init"], cwd=str(workspace_path), capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        subprocess.run(["git", "config", "user.name", author_name], cwd=str(workspace_path), capture_output=True, text=True, encoding="utf-8", errors="replace")
        subprocess.run(["git", "config", "user.email", author_email], cwd=str(workspace_path), capture_output=True, text=True, encoding="utf-8", errors="replace")

        # 3. Stage and commit
        subprocess.run(["git", "add", "."], cwd=str(workspace_path), capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=str(workspace_path), capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

        commit_hash, recent_commits, git_diff = self._extract_git_metadata(workspace_path)
        file_tree = self._build_file_tree(workspace_path)

        meta.update({
            "scenario_id": r_id,
            "mode": meta.get("mode", "custom_code"),
            "commit_hash": commit_hash,
            "author": author_name,
            "commit_message": commit_message
        })

        return WorkspaceInfo(
            run_id=r_id,
            workspace_dir=str(workspace_path),
            mode=meta.get("mode", "custom_code"),
            commit_hash=commit_hash,
            recent_commits=recent_commits,
            git_diff=git_diff,
            file_tree=file_tree,
            metadata=meta
        )

    # =========================================================================
    # MODE 3: Seeded Reference Repositories
    # =========================================================================
    @staticmethod
    def get_seeded_template(template_name: str) -> List[Dict[str, str]]:
        """
        Returns clean, editable file lists for the 3 reference templates:
          - 'missing_env_repo'
          - 'flaky_network_repo'
          - 'logic_regression_repo'
        """
        templates: Dict[str, List[Dict[str, str]]] = {
            "missing_env_repo": [
                {
                    "path": "app.py",
                    "content": (
                        '"""Service module requiring runtime environment configuration."""\n'
                        'import os\n\n'
                        'def get_payment_gateway_url() -> str:\n'
                        '    # Requires PAYMENT_GATEWAY_URL env var; raises KeyError if missing\n'
                        '    return os.environ["PAYMENT_GATEWAY_URL"]\n\n'
                        'def process_transaction(amount: float) -> dict:\n'
                        '    url = get_payment_gateway_url()\n'
                        '    return {"gateway": url, "amount": amount, "status": "APPROVED"}\n'
                    )
                },
                {
                    "path": "test_app.py",
                    "content": (
                        '"""Test suite for payment transaction gateway."""\n'
                        'import pytest\n'
                        'from app import process_transaction\n\n'
                        'def test_process_transaction():\n'
                        '    res = process_transaction(150.0)\n'
                        '    assert res["status"] == "APPROVED"\n'
                        '    assert "gateway" in res\n'
                    )
                },
                {
                    "path": ".env.example",
                    "content": "PAYMENT_GATEWAY_URL=https://api.gateway.sandbox.internal/v1\n"
                },
                {
                    "path": "README.md",
                    "content": (
                        "# Payment Gateway Microservice\n"
                        "Requires `PAYMENT_GATEWAY_URL` configured in environment.\n"
                    )
                }
            ],
            "flaky_network_repo": [
                {
                    "path": "app.py",
                    "content": (
                        '"""Simulated microservice client with intermittent network failures."""\n'
                        'import os\n\n'
                        'def fetch_remote_metrics(endpoint_id: str) -> dict:\n'
                        '    state_file = os.path.join(os.path.dirname(__file__), ".request_counter")\n'
                        '    count = 0\n'
                        '    if os.path.exists(state_file):\n'
                        '        try:\n'
                        '            with open(state_file, "r") as f:\n'
                        '                count = int(f.read().strip())\n'
                        '        except Exception:\n'
                        '            count = 0\n'
                        '    count += 1\n'
                        '    with open(state_file, "w") as f:\n'
                        '        f.write(str(count))\n\n'
                        '    # Nondeterministic intermittent failure: fails on odd calls, passes on even calls\n'
                        '    if count % 2 == 1:\n'
                        '        raise ConnectionResetError(f"Connection reset by peer while contacting endpoint {endpoint_id}")\n\n'
                        '    return {"endpoint_id": endpoint_id, "healthy": True, "latency_ms": 14.2}\n'
                    )
                },
                {
                    "path": "test_app.py",
                    "content": (
                        '"""Tests for remote metrics collector."""\n'
                        'import pytest\n'
                        'from app import fetch_remote_metrics\n\n'
                        'def test_fetch_remote_metrics():\n'
                        '    metrics = fetch_remote_metrics("ep-core-cluster")\n'
                        '    assert metrics["healthy"] is True\n'
                    )
                },
                {
                    "path": "README.md",
                    "content": (
                        "# Remote Metrics Service\n"
                        "Demonstrates intermittent network socket flakiness.\n"
                    )
                }
            ],
            "logic_regression_repo": [
                {
                    "path": "app.py",
                    "content": (
                        '"""Billing calculation engine with recent logic regression."""\n\n'
                        'def calculate_discounted_total(price: float, quantity: int, discount_code: str = None) -> float:\n'
                        '    subtotal = price * quantity\n'
                        '    # Logic Regression: Applied 50% discount instead of 15% for SAVE15\n'
                        '    discount_rate = 0.50 if discount_code == "SAVE15" else 0.0\n'
                        '    discount_amount = subtotal * discount_rate\n'
                        '    tax = (subtotal - discount_amount) * 0.08\n'
                        '    return round(subtotal - discount_amount + tax, 2)\n'
                    )
                },
                {
                    "path": "test_app.py",
                    "content": (
                        '"""Tests for billing calculations."""\n'
                        'import pytest\n'
                        'from app import calculate_discounted_total\n\n'
                        'def test_save15_discount():\n'
                        '    # 100 * 1 = 100. 15% discount -> 85.00. 8% tax (6.80) -> 91.80\n'
                        '    total = calculate_discounted_total(100.0, 1, discount_code="SAVE15")\n'
                        '    assert total == 91.80, f"Expected total 91.80, got {total}"\n'
                    )
                },
                {
                    "path": "README.md",
                    "content": (
                        "# Billing Calculation Engine\n"
                        "Contains a logic calculation regression on discount rates.\n"
                    )
                }
            ]
        }

        if template_name not in templates:
            raise KeyError(f"Unknown template '{template_name}'. Available: {list(templates.keys())}")
        return templates[template_name]

    def setup_seeded_repo(self, template_name: str, run_id: Optional[str] = None) -> WorkspaceInfo:
        """Loads a seeded template into Mode 2 as an isolated, fully editable git repository."""
        files = self.get_seeded_template(template_name)
        return self.setup_custom_code(
            files=files,
            run_id=run_id,
            commit_message=f"Seeded reference repository: {template_name}",
            metadata={"template_name": template_name, "mode": "seeded"}
        )

    # =========================================================================
    # Workspace Maintenance
    # =========================================================================
    def cleanup_workspace(self, run_id: str) -> bool:
        """Removes the isolated sandbox run directory."""
        target = self.sandbox_base_dir / run_id
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            return True
        return False
