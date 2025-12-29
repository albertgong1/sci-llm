"""Host-side Gemini agent that uploads predictions into the container."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import types

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


def _strip_at_paper(instruction: str) -> str:
    lines = instruction.splitlines()
    if lines and lines[0].lstrip().startswith("@"):
        return "\n".join(lines[1:]).lstrip()
    return instruction


def _extract_json(text: str) -> object:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Gemini returned empty output.")

    fenced = re.search(r"```json\\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_candidates = [cleaned.find("{"), cleaned.find("[")]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        raise ValueError("No JSON object found in Gemini output.")

    start_idx = min(start_candidates)
    end_idx = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end_idx <= start_idx:
        raise ValueError("Malformed JSON in Gemini output.")

    snippet = cleaned[start_idx : end_idx + 1]
    return json.loads(snippet)


class HostGeminiCli(BaseAgent):
    """Run Gemini from the host and upload predictions into the sandbox."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        """Return the agent name for Harbor registration."""
        return "gemini-cli"

    def version(self) -> str | None:
        """Return the host agent version marker."""
        return "host"

    async def setup(self, environment: BaseEnvironment) -> None:
        """No setup required for host-side execution."""
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run Gemini on the host and upload predictions into the sandbox."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required.")

        model = (
            self.model_name.split("/")[-1] if self.model_name else "gemini-2.5-flash"
        )
        paper_path = Path(environment.environment_dir) / "paper.pdf"
        if not paper_path.exists():
            raise FileNotFoundError(f"Paper PDF not found: {paper_path}")

        client = genai.Client(api_key=api_key)
        uploaded = client.files.upload(
            file=str(paper_path), config={"mime_type": "application/pdf"}
        )

        prompt = _strip_at_paper(instruction)
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_text(text=prompt), uploaded],
        )

        raw_text = response.text or ""
        output_path = self.logs_dir / "gemini-host.txt"
        output_path.write_text(raw_text)

        predictions = _extract_json(raw_text)
        predictions_path = self.logs_dir / "predictions.json"
        predictions_path.write_text(json.dumps(predictions, indent=2))

        await environment.exec(command="mkdir -p /app/output")
        await environment.upload_file(
            source_path=predictions_path,
            target_path="/app/output/predictions.json",
        )
