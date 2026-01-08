r"""Run Harbor with the native Modal environment integration.

This wrapper simply forwards args to `run_harbor.py` while forcing `--env modal`,
so Harbor uses Modal sandboxes natively (no custom upload/sandbox management).

Example:
    uv run python src/pbench_containerized_eval/run_harbor_modal.py \\
      jobs start -c out/harbor/supercon-mini-v2/ground-template/job.yaml \\
      -a gemini-cli -m gemini/gemini-2.5-flash

"""

from __future__ import annotations

import sys

from pbench_containerized_eval import run_harbor


def main(argv: list[str]) -> int:
    """Invoke Harbor with Modal enforced on the environment."""
    if not argv:
        print(
            "Usage: run_harbor_modal.py <harbor args...>\n"
            "Example: run_harbor_modal.py jobs start -c "
            "out/harbor/supercon-mini-v2/ground-template/job.yaml -a oracle",
            file=sys.stderr,
        )
        return 2

    if "--env" not in argv and "--environment-type" not in argv:
        argv = [*argv, "--env", "modal"]

    if _should_force_delete(argv):
        argv = [*argv, "--delete"]

    sys.argv = [sys.argv[0], *argv]
    return run_harbor.main()


def _should_force_delete(argv: list[str]) -> bool:
    if "--delete" in argv or "--no-delete" in argv:
        return False
    if "start" not in argv:
        return False
    return "jobs" in argv or "trials" in argv


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
