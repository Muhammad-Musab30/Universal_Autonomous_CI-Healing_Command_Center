"""GitHub API Client and Local Git PR Automation Flow.

Features:
  1. Creates real local branches: `fix/repairiq-<run_id>` inside sandbox repositories.
  2. Commits minimal patches and generates unified `git format-patch` payloads.
  3. Integrates with GitHub REST API (`POST /repos/{owner}/{repo}/pulls`) when `GITHUB_TOKEN` is present.
  4. Provides 1-click local PR export bundle when running on custom pasted code or offline.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union

try:
    import httpx
except ImportError:
    httpx = None


class GitHubClient:
    """
    Manages local git branch/commit creation and remote GitHub Pull Request submission.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN", "")

    def create_repair_branch_and_commit(
        self,
        workspace_dir: Union[str, Path],
        run_id: str,
        commit_message: Optional[str] = None,
        author_name: str = "RepairIQ Agent",
        author_email: str = "ci@repairiq.dev"
    ) -> Dict[str, Any]:
        """
        Creates an isolated repair branch 'fix/repairiq-<run_id>' in the sandbox repo,
        commits all modified files, and generates a portable git format-patch.
        """
        w_path = Path(workspace_dir).resolve()
        branch_name = f"fix/repairiq-{run_id}"
        c_msg = commit_message or f"fix(repairiq): autonomous causal repair for {run_id}"

        # 1. Configure user if not set
        subprocess.run(["git", "config", "user.name", author_name], cwd=str(w_path), capture_output=True, text=True, encoding="utf-8", errors="replace")
        subprocess.run(["git", "config", "user.email", author_email], cwd=str(w_path), capture_output=True, text=True, encoding="utf-8", errors="replace")

        # 2. Checkout new branch
        res_branch = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=str(w_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if res_branch.returncode != 0:
            # Branch might already exist; switch to it
            subprocess.run(["git", "checkout", branch_name], cwd=str(w_path), capture_output=True, text=True, encoding="utf-8", errors="replace")

        # 3. Stage changes
        subprocess.run(["git", "add", "."], cwd=str(w_path), capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

        # 4. Commit changes
        res_commit = subprocess.run(
            ["git", "commit", "-m", c_msg],
            cwd=str(w_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        # 5. Get commit SHA
        res_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(w_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        commit_sha = res_sha.stdout.strip() if res_sha.returncode == 0 else "HEAD"

        # 6. Generate format-patch payload
        res_patch = subprocess.run(
            ["git", "format-patch", "-1", "HEAD", "--stdout"],
            cwd=str(w_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        format_patch = res_patch.stdout if res_patch.returncode == 0 else ""

        return {
            "branch": branch_name,
            "commit_sha": commit_sha,
            "commit_message": c_msg,
            "format_patch": format_patch,
            "is_committed": res_commit.returncode == 0 or "nothing to commit" in res_commit.stdout
        }

    def open_pull_request(
        self,
        repo_url: Optional[str],
        branch_name: str,
        title: str,
        pr_body: str,
        base_branch: str = "main",
        format_patch: str = ""
    ) -> Dict[str, Any]:
        """
        Attempts to open a real GitHub PR via REST API if GITHUB_TOKEN and a remote repo URL exist.
        Otherwise returns a complete Local PR export payload.
        """
        # Parse owner and repo name from GitHub URL
        owner_repo = self._parse_github_url(repo_url) if repo_url else None

        if self.token and owner_repo and httpx is not None:
            owner, repo = owner_repo
            api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "RepairIQ-Autonomous-Agent"
            }
            payload = {
                "title": title,
                "body": pr_body,
                "head": branch_name,
                "base": base_branch
            }

            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(api_url, json=payload, headers=headers)
                    if resp.status_code in [200, 201]:
                        data = resp.json()
                        return {
                            "status": "PR_OPENED_ON_GITHUB",
                            "pr_url": data.get("html_url"),
                            "pr_number": data.get("number"),
                            "branch": branch_name,
                            "github_api_success": True
                        }
                    else:
                        return {
                            "status": "LOCAL_PR_READY",
                            "branch": branch_name,
                            "github_api_status": resp.status_code,
                            "github_api_error": resp.text[:300],
                            "pr_body": pr_body,
                            "format_patch": format_patch,
                            "message": f"GitHub API responded with HTTP {resp.status_code}. Local PR bundle ready."
                        }
            except Exception as e:
                return {
                    "status": "LOCAL_PR_READY",
                    "branch": branch_name,
                    "github_api_error": str(e),
                    "pr_body": pr_body,
                    "format_patch": format_patch,
                    "message": f"GitHub API connection failed ({e}). Local PR bundle generated."
                }

        # Local PR / Custom Code Flow
        return {
            "status": "LOCAL_PR_READY",
            "branch": branch_name,
            "pr_title": title,
            "pr_body": pr_body,
            "format_patch": format_patch,
            "is_exported": True,
            "message": f"Created local branch '{branch_name}' with verified patch committed. Format-patch ready for 1-click export."
        }

    def _parse_github_url(self, repo_url: str) -> Optional[tuple[str, str]]:
        match = re.search(r"github\.com[/:]([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+?)(?:\.git)?$", repo_url)
        if match:
            return match.group(1), match.group(2)
        return None
