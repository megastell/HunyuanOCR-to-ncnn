from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]


REQUIRED_DOCS = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "docs/install_linux.md",
    "docs/install_windows.md",
    "docs/phase4l_dual_platform_release_acceptance.json",
    "docs/phase4l_dual_platform_release_acceptance_milestone.md",
    "docs/phase4m_release_notes.md",
    "docs/github_discussion_hunyuanocr_ncnn_draft.md",
)

REQUIRED_LICENSES = (
    "third_party/licenses/ncnn-LICENSE.txt",
    "third_party/licenses/stb_image-LICENSE.txt",
    "third_party/licenses/Tencent-HunyuanOCR-LICENSE.txt",
)

REQUIRED_TAGS = (
    "milestone-open-source-release-prep-fp32",
    "milestone-reproducible-release-acceptance-fp32",
    "milestone-dual-platform-reproduced-release-acceptance-fp32",
)

PHASE4M_ALLOWED_CHANGES = {
    "README.md",
    "docs/phase4m_release_notes.md",
    "docs/github_discussion_hunyuanocr_ncnn_draft.md",
    "docs/phase4m_open_source_release_audit.json",
    "docs/phase4m_open_source_release_audit_milestone.md",
    "tools/release/audit_phase4m_release_readiness.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit final open-source release readiness for Phase 4M."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "docs/phase4m_open_source_release_audit.json",
    )
    parser.add_argument(
        "--milestone",
        type=Path,
        default=PROJECT_DIR / "docs/phase4m_open_source_release_audit_milestone.md",
    )
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{completed.stdout}")
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(relative: str) -> dict[str, object]:
    path = PROJECT_DIR / relative
    record: dict[str, object] = {
        "path": relative,
        "exists": path.is_file(),
    }
    if path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    return record


def load_json(relative: str) -> dict[str, object]:
    path = PROJECT_DIR / relative
    if not path.is_file():
        raise RuntimeError(f"Missing JSON report: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def windows_path_to_wsl(path_text: str) -> Path | None:
    if len(path_text) < 3 or path_text[1:3] != ":\\":
        return None
    drive = path_text[0].lower()
    rest = path_text[3:].replace("\\", "/")
    return Path("/mnt") / drive / rest


def external_file_record(path_text: str) -> dict[str, object]:
    candidate = Path(path_text)
    if not candidate.is_file():
        converted = windows_path_to_wsl(path_text)
        if converted is not None:
            candidate = converted
    record: dict[str, object] = {
        "path": path_text,
        "checked_path": str(candidate),
        "exists": candidate.is_file(),
    }
    if candidate.is_file():
        record["bytes"] = candidate.stat().st_size
        record["sha256"] = sha256_file(candidate)
    return record


def normalize_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def audit_reports() -> dict[str, object]:
    phase4k = load_json("docs/phase4k_reproducible_release_acceptance.json")
    phase4l = load_json("docs/phase4l_dual_platform_release_acceptance.json")
    linux = load_json("docs/linux_phase4l_release_validation.json")
    windows = load_json("docs/windows_phase4l_reproduced_artifact_acceptance.json")
    reports = {
        "phase4k": {
            "status": phase4k.get("status"),
            "manifest_file_count": phase4k.get("manifest", {}).get("file_count"),
            "manifest_total_gib": phase4k.get("manifest", {}).get("total_gib"),
            "artifacts_reference_unchanged": phase4k.get(
                "existing_artifacts_reference_unchanged"
            ),
        },
        "phase4l": {
            "status": phase4l.get("status"),
            "linux_suites": phase4l.get("linux_release_acceptance", {}).get(
                "ctest_suites", []
            ),
            "windows_suites": phase4l.get(
                "windows_reproduced_artifact_acceptance", {}
            ).get("ctest_suites", []),
            "offline_release_acceptance": phase4l.get(
                "offline_release_acceptance", {}
            ),
        },
        "linux": {
            "status": linux.get("status"),
            "packages": linux.get("packages", []),
        },
        "windows": {
            "status": windows.get("status"),
            "packages": normalize_list(windows.get("packages", [])),
            "model_copy_file_count": windows.get("model_copy", {}).get("file_count"),
        },
    }
    for name, report in (
        ("phase4k", phase4k),
        ("phase4l", phase4l),
        ("linux", linux),
        ("windows", windows),
    ):
        if report.get("status") != "passed":
            raise RuntimeError(f"{name} report did not pass: {report.get('status')}")
    return reports


def audit_artifacts(reports: dict[str, object]) -> dict[str, object]:
    linux_packages = reports["linux"]["packages"]  # type: ignore[index]
    windows_packages = reports["windows"]["packages"]  # type: ignore[index]
    artifact_records = []
    for package in normalize_list(linux_packages):
        if isinstance(package, dict) and "path" in package:
            artifact_records.append(external_file_record(str(package["path"])))
    for package in normalize_list(windows_packages):
        if isinstance(package, dict) and "path" in package:
            artifact_records.append(external_file_record(str(package["path"])))
    missing = [record for record in artifact_records if not record["exists"]]
    if missing:
        raise RuntimeError(f"Release package files are missing: {missing}")
    return {
        "release_packages": artifact_records,
        "package_count": len(artifact_records),
    }


def audit_git() -> dict[str, object]:
    tags = run_git(["tag", "--list", "milestone-*", "--sort=creatordate"]).splitlines()
    remotes = run_git(["remote", "-v"]).splitlines()
    status = run_git(["status", "--short"])
    branch = run_git(["branch", "--show-current"])
    head = run_git(["log", "-1", "--oneline", "--decorate"])
    branch_verbose = run_git(["branch", "-vv"]).splitlines()
    missing_tags = [tag for tag in REQUIRED_TAGS if tag not in tags]
    if missing_tags:
        raise RuntimeError(f"Missing milestone tags: {missing_tags}")
    status_paths = []
    for line in status.splitlines():
        if not line:
            continue
        path = line[2:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        status_paths.append(path)
    unexpected_paths = [
        path for path in status_paths if path not in PHASE4M_ALLOWED_CHANGES
    ]
    if unexpected_paths:
        raise RuntimeError(f"Unexpected non-Phase-4M worktree changes: {unexpected_paths}")
    return {
        "branch": branch,
        "head": head,
        "status_short": status,
        "status_paths": status_paths,
        "only_phase4m_changes_present": not unexpected_paths,
        "remote_configured": bool(remotes),
        "remotes": remotes,
        "milestone_tag_count": len(tags),
        "required_tags_present": True,
        "branch_verbose": branch_verbose,
        "release_policy": {
            "remote_push_performed": False,
            "github_discussion_published": False,
            "upstream_pr_created": False,
        },
    }


def write_milestone(path: Path, report: dict[str, object]) -> None:
    git = report["git"]
    artifact = report["artifacts"]
    phase4l = report["reports"]["phase4l"]
    lines = [
        "# Phase 4M Open Source Release Audit",
        "",
        "## Status",
        "",
        "Phase 4M is complete. The repository has a final pre-publication audit,",
        "release notes, and a GitHub Discussion draft, without pushing a remote,",
        "publishing the Discussion, or creating an upstream PR.",
        "",
        "## Verified Evidence",
        "",
        f"- Current branch: `{git['branch']}`",
        f"- HEAD: `{git['head']}`",
        f"- Only Phase 4M changes present at audit time: {str(git['only_phase4m_changes_present']).lower()}",
        f"- Remote configured: {str(git['remote_configured']).lower()}",
        f"- Milestone tags found: {git['milestone_tag_count']}",
        f"- Release packages checked: {artifact['package_count']}",
        f"- Linux suites: {', '.join(phase4l['linux_suites'])}",
        f"- Windows suites: {', '.join(phase4l['windows_suites'])}",
        "",
        "## Added Documents",
        "",
        "- `docs/phase4m_release_notes.md`",
        "- `docs/github_discussion_hunyuanocr_ncnn_draft.md`",
        "- `docs/phase4m_open_source_release_audit.json`",
        "",
        "## Remaining Release Actions",
        "",
        "- Add a real GitHub remote URL before public release.",
        "- Push the final branch and milestone tags when ready.",
        "- Publish the GitHub Discussion draft manually after reviewing model",
        "  license wording and repository URL.",
        "- Optionally prepare an upstream ncnn_llm PR after the public repository",
        "  is available.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    docs = [file_record(relative) for relative in REQUIRED_DOCS]
    licenses = [file_record(relative) for relative in REQUIRED_LICENSES]
    missing_docs = [record["path"] for record in docs if not record["exists"]]
    missing_licenses = [record["path"] for record in licenses if not record["exists"]]
    if missing_docs or missing_licenses:
        raise RuntimeError(
            f"Missing docs={missing_docs}, missing licenses={missing_licenses}"
        )
    reports = audit_reports()
    artifacts = audit_artifacts(reports)
    git = audit_git()
    report = {
        "phase": "4M",
        "status": "passed",
        "started_at_unix": started,
        "elapsed_seconds": time.time() - started,
        "project_dir": str(PROJECT_DIR),
        "docs": docs,
        "licenses": licenses,
        "reports": reports,
        "artifacts": artifacts,
        "git": git,
        "remaining_risks": [
            "No Git remote is configured in the local repository yet.",
            "Converted model files are external and governed by the Tencent Hunyuan Community License Agreement, not Apache-2.0.",
            "Final public release should rerun Phase 4L from the final clean commit and archive packages externally.",
            "GitHub Discussion and any ncnn_llm pull request remain manual follow-up actions.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_milestone(args.milestone, report)
    print(args.report)
    print(args.milestone)


if __name__ == "__main__":
    main()
