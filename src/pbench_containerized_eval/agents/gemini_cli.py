"""Patched Gemini CLI agent with extra diagnostics for Daytona runs."""

from __future__ import annotations

import os

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_CA_CERT_PATH = "/etc/ssl/certs/ca-certificates.crt"


def _normalize_auth_env(env: dict[str, str]) -> None:
    gemini_key = env.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    google_key = env.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if gemini_key:
        env["GEMINI_API_KEY"] = gemini_key
        env["GOOGLE_API_KEY"] = gemini_key
    elif google_key:
        env["GEMINI_API_KEY"] = google_key
        env["GOOGLE_API_KEY"] = google_key


def _inject_node_tls_env(env: dict[str, str]) -> None:
    node_opts = env.get("NODE_OPTIONS", "")
    extra_opts = ["--dns-result-order=ipv4first"]
    for opt in extra_opts:
        if opt not in node_opts:
            node_opts = f"{node_opts} {opt}".strip()

    if node_opts:
        env["NODE_OPTIONS"] = node_opts

    env.setdefault("NODE_EXTRA_CA_CERTS", _CA_CERT_PATH)
    env.setdefault("SSL_CERT_FILE", _CA_CERT_PATH)


def _preflight_command() -> str:
    return (
        "if command -v curl >/dev/null 2>&1; then "
        "echo '[gemini-cli preflight] curl https://generativelanguage.googleapis.com' "
        "> /logs/agent/gemini-cli-network.txt; "
        "curl -I -sS --max-time 10 https://generativelanguage.googleapis.com "
        ">> /logs/agent/gemini-cli-network.txt 2>&1; "
        "else echo '[gemini-cli preflight] curl missing' "
        "> /logs/agent/gemini-cli-network.txt; fi"
    )


class PatchedGeminiCli(GeminiCli):
    """Gemini CLI with extra env fixes + debug artifacts for network failures."""

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """Build Gemini CLI commands with optional network preflight."""
        commands = super().create_run_agent_commands(instruction)
        if not commands:
            return commands

        env = dict(commands[0].env or {})
        _normalize_auth_env(env)
        _inject_node_tls_env(env)
        commands[0].env = env

        if os.environ.get("PBENCH_GEMINI_PREFLIGHT", "1").lower() not in {"0", "false"}:
            return [ExecInput(command=_preflight_command(), env=env), *commands]

        return commands

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run Gemini CLI and capture any error report artifacts."""
        try:
            await super().run(instruction, environment, context)
        finally:
            await self._copy_error_report(environment)

    async def _copy_error_report(self, environment: BaseEnvironment) -> None:
        copy_command = (
            "find /tmp -maxdepth 1 -type f -name 'gemini-client-error-*.json' "
            "2>/dev/null | head -n 1 | "
            "xargs -r -I{} cp {} /logs/agent/gemini-cli.error.json"
        )
        try:
            await environment.exec(command=copy_command)
        except Exception:
            pass
