r"""Clean Harbor workspace artifacts while preserving templates and data.

This utility deletes generated outputs under a workspace root:
- jobs/
- trials/
- out/
- logs/

It does NOT touch template folders, PDFs, rubric files, or README content.
By default it targets `examples/harbor-workspace` in the repo root.

Usage:
  uv run python examples/harbor-workspace/clean_harbor_workspace.py
  uv run python examples/harbor-workspace/clean_harbor_workspace.py --workspace /path/to/workspace
  uv run python examples/harbor-workspace/clean_harbor_workspace.py --dry-run
  uv run python examples/harbor-workspace/clean_harbor_workspace.py --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


KEEP_FILES = {".gitkeep"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_workspace_root() -> Path:
    return _repo_root() / "examples" / "harbor-workspace"


def _looks_like_workspace(path: Path) -> bool:
    markers = [
        path / "README.md",
        path / "rubric.csv",
        path / "ground-template",
        path / "ground-template-easy",
        path / "data",
    ]
    return any(marker.exists() for marker in markers)


def _clean_dir(path: Path) -> list[Path]:
    removed: list[Path] = []
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return removed

    for child in path.iterdir():
        if child.is_file() and child.name in KEEP_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(child)

    return removed


def main() -> int:
    """Parse arguments and clean workspace output directories."""
    parser = argparse.ArgumentParser(
        description="Clean Harbor workspace outputs (jobs/trials/out/logs)."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root (default: examples/harbor-workspace).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip workspace sanity checks.",
    )
    args = parser.parse_args()

    workspace = (args.workspace or _default_workspace_root()).resolve()
    if not workspace.exists():
        raise SystemExit(f"Workspace not found: {workspace}")
    if not workspace.is_dir():
        raise SystemExit(f"--workspace must be a directory: {workspace}")
    if not args.force and not _looks_like_workspace(workspace):
        raise SystemExit(
            f"{workspace} does not look like a Harbor workspace. "
            "Pass --force to clean anyway."
        )

    targets = [
        workspace / "jobs",
        workspace / "trials",
        workspace / "out",
        workspace / "logs",
    ]
    pending = [path for path in targets if path.exists() and any(path.iterdir())]
    if args.dry_run:
        if pending:
            print("Would clean:")
            for path in pending:
                print(f"  - {path}")
        else:
            print("Workspace already clean.")
        return 0

    removed_total: list[Path] = []
    for target in targets:
        removed_total.extend(_clean_dir(target))

    if removed_total:
        print(f"Cleaned {len(removed_total)} paths under {workspace}")
    else:
        print("Workspace already clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
