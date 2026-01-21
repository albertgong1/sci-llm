"""Score Harbor trial predictions after running tasks without a verifier.

This helper rebuilds expected.json from a ground-truth dataset and then calls the
template's `tests/check_prediction.py` to compute reward/details per trial. The
results are written under each trial's `verifier/` folder so downstream tools
like `collect_harbor_results.py` can consume them.

If a trial is missing `predictions.json`, the script attempts to recover a JSON
payload from agent logs (e.g., `agent/gemini-cli.txt`) and writes a synthetic
`verifier/app_output/predictions.json` before scoring.

Example:
  uv run python examples/harbor-workspace/score_harbor_results.py \
    --gt-hf-repo kilian-group/supercon-mini-v2 --gt-hf-split full

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from json import JSONDecoder


_TASK_PROPERTY_FILTERS: dict[str, set[str]] = {
    "tc": {"Tc (of this sample) recommended"},
}


def default_workspace_root() -> Path:
    """Return the workspace root (directory containing this script)."""
    return Path(__file__).resolve().parent


def _load_rubric_mapping(rubric_path: Path) -> dict[str, str]:
    """Load property_name -> rubric mapping from a rubric CSV."""
    mapping: dict[str, str] = {}
    with rubric_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prop = row.get("property_name")
            rubric = row.get("rubric")
            if prop and rubric:
                mapping[prop] = rubric
    return mapping


def _resolve_property_filter(task: str | None) -> set[str] | None:
    """Return the property filter for a task alias (or None for all)."""
    if task is None:
        return None
    return _TASK_PROPERTY_FILTERS.get(task.strip().lower())


def _flatten_dataset(
    dataset: Iterable[dict[str, Any]],
    *,
    rubric_mapping: dict[str, str],
    property_filter: set[str] | None,
) -> dict[str, list[dict[str, str]]]:
    """Flatten HF rows (refno + properties list) into expected.json-style rows."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in dataset:
        refno = str(row.get("refno") or "").strip()
        if not refno:
            continue
        props = row.get("properties") or []
        if not isinstance(props, list):
            continue

        for prop in props:
            if not isinstance(prop, dict):
                continue
            prop_name = str(prop.get("property_name") or "").strip()
            if not prop_name:
                continue
            if property_filter and prop_name not in property_filter:
                continue

            grouped.setdefault(refno.lower(), []).append(
                {
                    "material": str(prop.get("material_or_system") or ""),
                    "property_name": prop_name,
                    "property_value": str(prop.get("value_string") or ""),
                    "property_unit": "",
                    "rubric": rubric_mapping.get(prop_name, "categorical"),
                }
            )
    return grouped


def _resolve_refno(trial_dir: Path, workspace: Path) -> str:
    """Resolve the refno for a trial from config.json or trial name."""
    config_path = trial_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            task_path = config.get("task", {}).get("path")
            if task_path:
                task_dir = Path(task_path)
                if not task_dir.is_absolute():
                    task_dir = workspace / task_dir
                meta_path = task_dir / "environment" / "task_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    refno = str(meta.get("refno") or "").strip()
                    if refno:
                        return refno
        except Exception:
            pass
    return trial_dir.name.split("__", 1)[0]


def _find_predictions(trial_dir: Path) -> Path | None:
    """Locate predictions.json within a trial directory."""
    candidates = [
        trial_dir / "verifier" / "app_output" / "predictions.json",
        trial_dir / "agent" / "app_output" / "predictions.json",
        trial_dir / "agent" / "output" / "predictions.json",
        trial_dir / "agent" / "predictions.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in trial_dir.rglob("predictions.json"):
        return candidate
    return None


def _extract_first_json_array(text: str) -> list[dict[str, Any]] | None:
    """Extract the first JSON array from mixed text (handles fenced blocks)."""

    def looks_like_predictions_array(obj: Any) -> bool:
        if not isinstance(obj, list) or not obj:
            return False
        first = obj[0]
        if not isinstance(first, dict):
            return False
        return "property_name" in first and (
            "material" in first or "material_or_system" in first
        )

    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if looks_like_predictions_array(obj):
                return obj
        except Exception:
            pass

    decoder = JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except Exception:
            continue
        if looks_like_predictions_array(obj):
            return obj
    return None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from mixed text (handles fenced blocks)."""

    def looks_like_properties_object(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        props = obj.get("properties")
        if not isinstance(props, list) or not props:
            return False
        first = props[0]
        if not isinstance(first, dict):
            return False
        return "property_name" in first and (
            "material_or_system" in first or "material" in first
        )

    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if looks_like_properties_object(obj):
                return obj
        except Exception:
            pass

    decoder = JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except Exception:
            continue
        if looks_like_properties_object(obj):
            return obj
    return None


def _extract_text_from_jsonlines_log(text: str) -> str | None:
    """Decode assistant output embedded in JSONL agent logs (e.g., Claude Code)."""
    decoded_parts: list[str] = []
    any_json = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            return None
        try:
            obj = json.loads(line)
        except Exception:
            return None
        any_json = True

        if isinstance(obj, dict):
            message = obj.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    decoded_parts.append(content)
            elif isinstance(message, str):
                decoded_parts.append(message)

            if "content" in obj and isinstance(obj.get("content"), str):
                decoded_parts.append(obj["content"])

    if not any_json:
        return None
    return "\n".join(decoded_parts)


def _extract_predictions_from_text(text: str) -> Any | None:
    """Try to extract a predictions payload from a log text."""
    decoded = _extract_text_from_jsonlines_log(text)
    text = decoded or text

    extracted_obj = _extract_first_json_object(text)
    if extracted_obj is not None:
        return extracted_obj

    extracted_arr = _extract_first_json_array(text)
    if extracted_arr is not None:
        return extracted_arr
    return None


def _collect_agent_texts(trial_dir: Path) -> list[str]:
    """Collect agent log texts and trajectory messages for parsing."""
    texts: list[str] = []
    agent_dir = trial_dir / "agent"
    if not agent_dir.exists():
        return texts

    for txt_path in sorted(agent_dir.glob("*.txt")):
        try:
            texts.append(txt_path.read_text())
        except Exception:
            continue

    for cmd_txt in sorted(agent_dir.glob("command-*/stdout.txt")):
        try:
            texts.append(cmd_txt.read_text())
        except Exception:
            continue

    for json_path in sorted(agent_dir.glob("*.trajectory.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        messages = data.get("messages") if isinstance(data, dict) else None
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and isinstance(
                    message.get("content"), str
                ):
                    texts.append(message["content"])

    traj_path = agent_dir / "trajectory.json"
    if traj_path.exists():
        try:
            data = json.loads(traj_path.read_text())
        except Exception:
            data = None
        if isinstance(data, dict):
            steps = data.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if isinstance(step, dict) and isinstance(step.get("message"), str):
                        texts.append(step["message"])

    return texts


def _find_or_rebuild_predictions(trial_dir: Path, *, workspace: Path) -> Path | None:
    """Locate predictions.json or rebuild it from agent logs."""
    existing = _find_predictions(trial_dir)
    if existing is not None:
        return existing

    payload: Any | None = None
    for text in _collect_agent_texts(trial_dir):
        payload = _extract_predictions_from_text(text)
        if payload is not None:
            break
    if payload is None:
        return None

    verifier_dir = trial_dir / "verifier"
    app_output = verifier_dir / "app_output"
    app_output.mkdir(parents=True, exist_ok=True)
    predictions_path = app_output / "predictions.json"
    predictions_path.write_text(json.dumps(payload, indent=2))
    return predictions_path


def _run_check(
    *,
    scorer_path: Path,
    expected_path: Path,
    predictions_path: Path,
    reward_path: Path,
    details_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the template's check_prediction.py against a single trial."""
    cmd = [
        sys.executable,
        str(scorer_path),
        "--expected",
        str(expected_path),
        "--predictions",
        str(predictions_path),
        "--reward",
        str(reward_path),
        "--details",
        str(details_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def main() -> int:
    """Score Harbor trials by rebuilding expected.json from a dataset."""
    parser = argparse.ArgumentParser(
        description="Score Harbor trials after no-score runs."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root (default: this script's directory).",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help="Trials directory (default: <workspace>/trials).",
    )
    parser.add_argument(
        "--template",
        default="ground-template",
        help="Template folder containing tests/check_prediction.py.",
    )
    parser.add_argument(
        "--gt-hf-repo",
        default="kilian-group/supercon-mini-v2",
        help="Ground-truth dataset repo (default: kilian-group/supercon-mini-v2).",
    )
    parser.add_argument(
        "--gt-hf-split",
        default="test",
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--gt-hf-revision",
        default="main",
        help="Dataset revision (default: main).",
    )
    parser.add_argument(
        "--rubric-csv",
        type=Path,
        default=None,
        help="Rubric CSV path (default: <workspace>/rubric.csv).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task alias filter (e.g., tc). If omitted, score all properties.",
    )
    parser.add_argument(
        "--trial",
        action="append",
        default=None,
        help="Only score specific trial dir names (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of trials to score.",
    )
    args = parser.parse_args()

    workspace = (args.workspace or default_workspace_root()).resolve()
    trials_dir = (args.trials_dir or workspace / "trials").resolve()
    if not trials_dir.exists():
        raise SystemExit(f"Trials directory not found: {trials_dir}")

    template_root = (workspace / args.template).resolve()
    scorer_path = template_root / "tests" / "check_prediction.py"
    if not scorer_path.exists():
        raise SystemExit(f"Missing scorer at: {scorer_path}")

    rubric_path = (args.rubric_csv or (workspace / "rubric.csv")).resolve()
    if not rubric_path.exists():
        raise SystemExit(f"Rubric CSV not found: {rubric_path}")

    rubric_mapping = _load_rubric_mapping(rubric_path)
    property_filter = _resolve_property_filter(args.task)
    dataset = load_dataset(
        args.gt_hf_repo,
        split=args.gt_hf_split,
        revision=args.gt_hf_revision,
    )
    grouped = _flatten_dataset(
        dataset, rubric_mapping=rubric_mapping, property_filter=property_filter
    )

    trial_dirs = [p for p in trials_dir.iterdir() if p.is_dir()]
    trial_dirs.sort()
    if args.trial:
        requested = {name.strip() for name in args.trial if name and name.strip()}
        trial_dirs = [p for p in trial_dirs if p.name in requested]
    if args.limit is not None:
        trial_dirs = trial_dirs[: args.limit]

    scored = 0
    skipped = 0
    for trial_dir in trial_dirs:
        refno = _resolve_refno(trial_dir, workspace)
        truth_rows = grouped.get(refno.lower())
        if not truth_rows:
            print(f"Skipping {trial_dir.name}: no ground truth for refno {refno}")
            skipped += 1
            continue

        predictions_path = _find_or_rebuild_predictions(trial_dir, workspace=workspace)
        if not predictions_path:
            print(
                f"Skipping {trial_dir.name}: predictions.json not found "
                "and no JSON payload parsed from agent logs."
            )
            skipped += 1
            continue

        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)

        expected = {
            "task": args.task or "all",
            "refno": refno,
            "ground_truth": truth_rows,
        }
        expected_path = verifier_dir / "expected.json"
        expected_path.write_text(json.dumps(expected, indent=2))

        result = _run_check(
            scorer_path=scorer_path,
            expected_path=expected_path,
            predictions_path=predictions_path,
            reward_path=verifier_dir / "reward.txt",
            details_path=verifier_dir / "details.json",
        )
        if result.returncode != 0:
            print(f"{trial_dir.name}: scored with errors (exit {result.returncode}).")
            if result.stderr:
                print(result.stderr.strip())
        scored += 1

    print(
        f"Scored {scored} trial(s). Skipped {skipped}. Results stored under each trial's verifier/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
