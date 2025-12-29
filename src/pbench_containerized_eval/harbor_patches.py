"""Runtime patches for Harbor integrations."""

from __future__ import annotations

import os


def _normalize_allow_list(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned or None


def patch_daytona_environment() -> None:
    """Patch Harbor's Daytona environment to accept a network allowlist."""
    from harbor.environments.daytona import DaytonaClientManager, DaytonaEnvironment
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.paths import EnvironmentPaths
    from harbor.utils.logger import logger as harbor_logger

    from daytona import (
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        Image,
        Resources,
    )
    from daytona._async.snapshot import SnapshotState

    if getattr(EnvironmentFactory, "_PBENCH_DAYTONA_PATCHED", False):
        return

    class PatchedDaytonaEnvironment(DaytonaEnvironment):
        """Daytona environment with optional network allowlist support."""

        def __init__(
            self,
            *args: object,
            network_allow_list: str | None = None,
            **kwargs: object,
        ):
            super().__init__(*args, **kwargs)
            env_allow = _normalize_allow_list(
                os.environ.get("PBENCH_DAYTONA_NETWORK_ALLOW_LIST")
            )
            self._network_allow_list = (
                _normalize_allow_list(network_allow_list) or env_allow
            )

        async def start(self, force_build: bool) -> None:
            resources = Resources(
                cpu=self.task_env_config.cpus,
                memory=self.task_env_config.memory_mb // 1024,
                disk=self.task_env_config.storage_mb // 1024,
            )

            self._client_manager = await DaytonaClientManager.get_instance()
            daytona = await self._client_manager.get_client()

            snapshot_name: str | None = None
            snapshot_exists = False

            if self._snapshot_template_name:
                snapshot_name = self._snapshot_template_name.format(
                    name=self.environment_name
                )

                try:
                    snapshot = await daytona.snapshot.get(snapshot_name)
                    if snapshot.state == SnapshotState.ACTIVE:
                        snapshot_exists = True
                except Exception:
                    snapshot_exists = False

            if snapshot_exists and force_build:
                self.logger.warning(
                    "Snapshot template specified but force_build is True. "
                    "Snapshot will be used instead of building from scratch."
                )

            if snapshot_exists and snapshot_name:
                params = CreateSandboxFromSnapshotParams(
                    auto_delete_interval=0,
                    snapshot=snapshot_name,
                    network_block_all=self._network_block_all,
                    network_allow_list=self._network_allow_list,
                )
            elif force_build or not self.task_env_config.docker_image:
                self.logger.debug(
                    f"Building environment from Dockerfile {self._environment_definition_path}"
                )
                image = Image.from_dockerfile(self._environment_definition_path)
                params = CreateSandboxFromImageParams(
                    image=image,
                    auto_delete_interval=0,
                    resources=resources,
                    network_block_all=self._network_block_all,
                    network_allow_list=self._network_allow_list,
                )
            else:
                self.logger.debug(
                    f"Using prebuilt image: {self.task_env_config.docker_image}"
                )
                image = Image.base(self.task_env_config.docker_image)
                params = CreateSandboxFromImageParams(
                    image=image,
                    auto_delete_interval=0,
                    resources=resources,
                    network_block_all=self._network_block_all,
                    network_allow_list=self._network_allow_list,
                )

            await self._create_sandbox(params=params)

            await self.exec(
                f"mkdir -p {str(EnvironmentPaths.agent_dir)} {str(EnvironmentPaths.verifier_dir)}"
            )

    EnvironmentFactory._ENVIRONMENT_MAP[EnvironmentType.DAYTONA] = (
        PatchedDaytonaEnvironment
    )
    setattr(EnvironmentFactory, "_PBENCH_DAYTONA_PATCHED", True)
    harbor_logger.debug("Patched Daytona environment with network allowlist support.")
