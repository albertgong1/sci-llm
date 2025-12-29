"""Harbor CLI entrypoint with pbench runtime patches."""

from __future__ import annotations

import os
import sys

from pbench_containerized_eval.harbor_patches import patch_daytona_environment


def main() -> None:
    """Run Harbor CLI with pbench-specific runtime patches."""
    if os.environ.get("PBENCH_PATCH_HARBOR_DAYTONA", "1").lower() in {"1", "true"}:
        patch_daytona_environment()

    from harbor.cli.main import app

    sys.argv = ["harbor", *sys.argv[1:]]
    app()


if __name__ == "__main__":
    main()
