"""Build MinerU-backed bundles for scientific PDFs.

Usage: call `ensure_mineru_bundle(...)` from task-preparation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Literal


MinerUBackend = Literal[
    "pipeline",
    "hybrid-auto-engine",
    "hybrid-http-client",
    "vlm-auto-engine",
    "vlm-http-client",
]
MinerUMethod = Literal["auto", "txt", "ocr"]
MinerUModelSource = Literal["huggingface", "modelscope", "local"]


@dataclass(slots=True)
class MinerUConfig:
    """Configuration for running the MinerU CLI."""

    binary: str = "mineru"
    backend: MinerUBackend = "hybrid-auto-engine"
    method: MinerUMethod = "auto"
    formula: bool = True
    table: bool = True
    lang: str | None = None
    source: MinerUModelSource | None = None
    device: str | None = None
    extra_args: list[str] = field(default_factory=list)
    force: bool = False


@dataclass(slots=True)
class MinerUBundlePaths:
    """Normalized MinerU artifact paths for one source PDF."""

    source_pdf_path: Path
    bundle_dir: Path
    primary_markdown_path: Path
    clean_markdown_path: Path | None
    outline_path: Path | None
    table_index_path: Path | None
    tables_path: Path | None
    captions_path: Path | None
    readme_path: Path
    manifest_path: Path
    raw_markdown_path: Path | None
    content_list_path: Path | None
    middle_json_path: Path | None
    model_json_path: Path | None
    layout_pdf_path: Path | None
    span_pdf_path: Path | None
    images_dir: Path | None


def ensure_mineru_bundle(
    pdf_path: Path,
    bundle_root: Path,
    config: MinerUConfig,
) -> MinerUBundlePaths:
    """Return a normalized MinerU bundle, generating it if needed."""
    pdf_path = pdf_path.resolve()
    bundle_dir = bundle_root.resolve() / pdf_path.stem

    if not config.force and _bundle_exists(bundle_dir):
        _write_derived_bundle_files(
            pdf_path=pdf_path,
            bundle_dir=bundle_dir,
            config=config,
        )
        return _bundle_paths(pdf_path, bundle_dir)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    _clear_bundle_dir(bundle_dir)

    mineru_binary = shutil.which(config.binary)
    if mineru_binary is None:
        raise FileNotFoundError(
            f"MinerU CLI '{config.binary}' was not found on PATH. "
            "Install MinerU and ensure the `mineru` command is available."
        )

    with tempfile.TemporaryDirectory(prefix="mineru-raw-") as tmp_dir_str:
        raw_output_dir = Path(tmp_dir_str)
        result = _run_mineru(
            pdf_path=pdf_path, output_dir=raw_output_dir, config=config
        )
        try:
            result_root = _find_result_root(raw_output_dir, pdf_path.stem)
        except FileNotFoundError as exc:
            raise RuntimeError(_format_mineru_failure(exc, result)) from exc
        _materialize_bundle(
            pdf_path=pdf_path,
            result_root=result_root,
            bundle_dir=bundle_dir,
            config=config,
        )

    return _bundle_paths(pdf_path, bundle_dir)


def _bundle_exists(bundle_dir: Path) -> bool:
    """Return True when a normalized MinerU bundle already exists."""
    return (bundle_dir / "primary.md").exists() and (
        bundle_dir / "manifest.json"
    ).exists()


def _clear_bundle_dir(bundle_dir: Path) -> None:
    """Delete previous normalized MinerU outputs before rebuilding."""
    for child in bundle_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _bundle_paths(source_pdf_path: Path, bundle_dir: Path) -> MinerUBundlePaths:
    """Construct normalized MinerU bundle paths."""
    return MinerUBundlePaths(
        source_pdf_path=source_pdf_path,
        bundle_dir=bundle_dir,
        primary_markdown_path=bundle_dir / "primary.md",
        clean_markdown_path=_optional_path(bundle_dir / "clean.md"),
        outline_path=_optional_path(bundle_dir / "outline.md"),
        table_index_path=_optional_path(bundle_dir / "table_index.md"),
        tables_path=_optional_path(bundle_dir / "tables.md"),
        captions_path=_optional_path(bundle_dir / "captions.md"),
        readme_path=bundle_dir / "README.md",
        manifest_path=bundle_dir / "manifest.json",
        raw_markdown_path=_optional_path(bundle_dir / "raw.md"),
        content_list_path=_optional_path(bundle_dir / "content_list.json"),
        middle_json_path=_optional_path(bundle_dir / "middle.json"),
        model_json_path=_optional_path(bundle_dir / "model.json"),
        layout_pdf_path=_optional_path(bundle_dir / "layout.pdf"),
        span_pdf_path=_optional_path(bundle_dir / "span.pdf"),
        images_dir=_optional_path(bundle_dir / "images"),
    )


def _optional_path(path: Path) -> Path | None:
    """Return the path when it exists."""
    return path if path.exists() else None


def _run_mineru(
    pdf_path: Path, output_dir: Path, config: MinerUConfig
) -> subprocess.CompletedProcess[str]:
    """Invoke MinerU for a single PDF."""
    cmd: list[str] = [
        config.binary,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-m",
        config.method,
        "-b",
        config.backend,
        "-f",
        _bool_arg(config.formula),
        "-t",
        _bool_arg(config.table),
    ]
    if config.lang:
        cmd.extend(["-l", config.lang])
    if config.source:
        cmd.extend(["--source", config.source])
    if config.device:
        cmd.extend(["-d", config.device])
    if config.extra_args:
        cmd.extend(config.extra_args)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(_format_mineru_failure(None, result))
    return result


def _bool_arg(value: bool) -> str:
    """Return MinerU's lowercase boolean representation."""
    return "true" if value else "false"


def _find_result_root(output_dir: Path, pdf_stem: str) -> Path:
    """Locate the directory containing MinerU's output files."""
    candidate_dirs: set[Path] = set()
    marker_patterns = (
        "*_content_list.json",
        "*_middle.json",
        "*_model.json",
        "*.md",
    )
    for pattern in marker_patterns:
        for match in output_dir.rglob(pattern):
            if match.is_file():
                candidate_dirs.add(match.parent)

    if not candidate_dirs:
        raise FileNotFoundError(
            f"MinerU did not produce any recognizable output files under {output_dir}."
        )

    pdf_stem_lower = pdf_stem.lower()

    def rank(path: Path) -> tuple[int, int, int, str]:
        score = 0
        joined = "/".join(part.lower() for part in path.parts)
        if pdf_stem_lower in path.name.lower():
            score += 4
        if f"/{pdf_stem_lower}/" in f"/{joined}/":
            score += 6
        depth = len(path.relative_to(output_dir).parts)
        return (-score, depth, len(path.as_posix()), path.as_posix())

    return sorted(candidate_dirs, key=rank)[0]


def _materialize_bundle(
    pdf_path: Path,
    result_root: Path,
    bundle_dir: Path,
    config: MinerUConfig,
) -> None:
    """Copy MinerU outputs and emit normalized companion files."""
    copied_paths = _copy_raw_outputs(result_root, bundle_dir)
    raw_markdown_path = _select_primary_markdown(bundle_dir, copied_paths)

    raw_markdown_text = ""
    if raw_markdown_path is not None:
        raw_markdown_text = raw_markdown_path.read_text(errors="replace")
        (bundle_dir / "raw.md").write_text(raw_markdown_text)

    content_list_path = _select_json(
        bundle_dir, copied_paths, suffix="_content_list.json"
    )
    _select_json(bundle_dir, copied_paths, suffix="_middle.json")
    _select_json(bundle_dir, copied_paths, suffix="_model.json")

    content_items = _load_json_list(content_list_path)
    derived_context = _build_context(
        raw_markdown_text=raw_markdown_text,
        content_items=content_items,
    )

    primary_text = _render_primary_markdown(
        pdf_path=pdf_path,
        raw_markdown_text=raw_markdown_text,
        context=derived_context,
    )
    (bundle_dir / "primary.md").write_text(primary_text)
    _write_derived_bundle_files(
        pdf_path=pdf_path,
        bundle_dir=bundle_dir,
        config=config,
    )


def _copy_raw_outputs(result_root: Path, bundle_dir: Path) -> list[Path]:
    """Copy MinerU outputs into the normalized bundle."""
    copied_paths: list[Path] = []
    for source_path in sorted(result_root.iterdir()):
        destination = bundle_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination)
        else:
            shutil.copy2(source_path, destination)
        copied_paths.append(destination)
    return copied_paths


def _select_primary_markdown(bundle_dir: Path, copied_paths: list[Path]) -> Path | None:
    """Return the main markdown file from MinerU's raw outputs."""
    markdown_candidates = [
        path for path in copied_paths if path.is_file() and path.suffix.lower() == ".md"
    ]
    if not markdown_candidates:
        return None
    markdown_candidates.sort(key=lambda path: (path.name != "content.md", path.name))
    return markdown_candidates[0]


def _select_json(
    bundle_dir: Path, copied_paths: list[Path], suffix: str
) -> Path | None:
    """Return a copied json file matching the expected MinerU suffix."""
    candidates = [
        path for path in copied_paths if path.is_file() and path.name.endswith(suffix)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.name)
    destination = bundle_dir / suffix.removeprefix("_")
    if candidates[0] != destination:
        shutil.copy2(candidates[0], destination)
    return destination


def _load_json_list(json_path: Path | None) -> list[dict[str, Any]]:
    """Load a JSON array when present."""
    if json_path is None or not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(errors="replace"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _build_context(
    raw_markdown_text: str, content_items: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build normalized content views used by the task bundle."""
    blocks = _normalize_blocks(
        raw_markdown_text=raw_markdown_text, content_items=content_items
    )
    sections = _segment_sections(blocks)
    tables = [block for block in blocks if block["kind"] == "table"]
    captions = [
        block
        for block in blocks
        if block["kind"] in {"figure_caption", "table_caption"}
    ]

    return {
        "blocks": blocks,
        "sections": sections,
        "tables": tables,
        "captions": captions,
    }


def _normalize_blocks(
    raw_markdown_text: str, content_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Normalize raw markdown and MinerU content items into structured blocks."""
    if content_items:
        blocks = _blocks_from_content_items(content_items)
    else:
        blocks = _blocks_from_markdown(raw_markdown_text)

    cleaned_blocks: list[dict[str, Any]] = []
    for block in blocks:
        text = _normalize_visible_text(str(block.get("text", "")))
        if not text.strip():
            continue
        if _should_drop_block(text):
            continue
        cleaned = dict(block)
        cleaned["text"] = text
        cleaned_blocks.append(cleaned)
    return cleaned_blocks


def _blocks_from_content_items(
    content_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert MinerU content list items into normalized text blocks."""
    blocks: list[dict[str, Any]] = []
    for index, item in enumerate(content_items, start=1):
        item_type = str(item.get("type") or "").strip().lower()
        page_idx = _safe_int(item.get("page_idx"))
        text = _content_item_text(item)
        kind = _map_content_item_kind(item_type=item_type, text=text)
        blocks.append(
            {
                "index": index,
                "page": None if page_idx is None else page_idx + 1,
                "kind": kind,
                "text": text,
            }
        )
    return blocks


def _content_item_text(item: dict[str, Any]) -> str:
    """Extract the visible text from one MinerU content-list item."""
    for key in ("text", "md", "content", "latex"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value

    lines = item.get("lines")
    if isinstance(lines, list):
        parts: list[str] = []
        for line in lines:
            if isinstance(line, dict):
                span_text = line.get("text")
                if isinstance(span_text, str):
                    parts.append(span_text)
        if parts:
            return "\n".join(parts)
    return ""


def _map_content_item_kind(item_type: str, text: str) -> str:
    """Map MinerU item types into the bundle's simplified block kinds."""
    if "table" in item_type and "caption" in item_type:
        return "table_caption"
    if "figure" in item_type and "caption" in item_type:
        return "figure_caption"
    if "table" in item_type:
        return "table"
    if "title" in item_type:
        return "title"
    if "section" in item_type or "heading" in item_type:
        return "heading"
    if "equation" in item_type:
        return "equation"
    if re.match(r"^\s*(table|fig(?:ure)?)\s+\d+", text, flags=re.IGNORECASE):
        if text.lower().startswith("table"):
            return "table_caption"
        return "figure_caption"
    return "paragraph"


def _blocks_from_markdown(raw_markdown_text: str) -> list[dict[str, Any]]:
    """Split raw markdown into paragraph-like blocks."""
    lines = raw_markdown_text.splitlines()
    blocks: list[dict[str, Any]] = []
    current: list[str] = []
    block_index = 1

    def flush() -> None:
        nonlocal block_index
        text = "\n".join(current).strip()
        if text:
            blocks.append(
                {
                    "index": block_index,
                    "page": None,
                    "kind": _infer_markdown_block_kind(text),
                    "text": text,
                }
            )
            block_index += 1
        current.clear()

    for line in lines:
        if line.strip():
            current.append(line)
        else:
            flush()
    flush()
    return blocks


def _infer_markdown_block_kind(text: str) -> str:
    """Infer a coarse block type from markdown text."""
    if text.startswith("#"):
        return "heading"
    if re.match(r"^\s*(table|fig(?:ure)?)\s+\d+", text, flags=re.IGNORECASE):
        if text.lower().startswith("table"):
            return "table_caption"
        return "figure_caption"
    if "\t" in text or "|" in text:
        return "table"
    return "paragraph"


def _normalize_visible_text(text: str) -> str:
    """Apply lightweight normalization to OCR- and markdown-heavy text."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = html.unescape(normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    normalized = re.sub(
        r"\b([A-Z][a-z]?)\s+([A-Z][a-z]?(?:\s+\d+)+)\b",
        lambda match: match.group(1) + re.sub(r"\s+", "", match.group(2)),
        normalized,
    )
    normalized = re.sub(
        r"\b([A-Z][a-z]?(?:\d+(?:\.\d+)?)?)\s+([A-Z][a-z]?(?:\d+(?:\.\d+)?)?)\b",
        lambda match: _join_formula_tokens(match.group(1), match.group(2)),
        normalized,
    )
    normalized = re.sub(
        r"\b([IPFCRAB])\s+(\d)\s*/\s*([a-zA-Z])\s*([a-zA-Z])\s*([a-zA-Z])\b",
        r"\1\2/\3\4\5",
        normalized,
    )
    normalized = re.sub(
        r"\b([A-Za-z])\s+(\d)\s*/\s*([A-Za-z]+)\b", r"\1\2/\3", normalized
    )
    return normalized.strip()


def _join_formula_tokens(left: str, right: str) -> str:
    """Join split chemical formula tokens while leaving prose intact."""
    if any(char.isdigit() for char in left + right):
        return f"{left}{right}"
    return f"{left} {right}"


def _should_drop_block(text: str) -> bool:
    """Return True for obvious MinerU boilerplate/noise blocks."""
    compact = " ".join(text.lower().split())
    return any(
        marker in compact
        for marker in (
            "discarded",
            "end of content",
            "generated by mineru",
        )
    )


def _segment_sections(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a lightweight outline from normalized blocks."""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for block in blocks:
        if block["kind"] in {"title", "heading"}:
            current = {
                "title": block["text"].lstrip("# ").strip(),
                "page": block.get("page"),
                "block_index": block["index"],
            }
            sections.append(current)
    return sections


def _render_primary_markdown(
    pdf_path: Path,
    raw_markdown_text: str,
    context: dict[str, Any],
) -> str:
    """Render the main agent-facing MinerU markdown file."""
    lines: list[str] = [
        f"# MinerU Extraction for {pdf_path.name}",
        "",
    ]

    sections = context["sections"]
    if sections:
        lines.extend(["## Outline", ""])
        for section in sections:
            location = ""
            if section.get("page") is not None:
                location = f" (page {section['page']})"
            lines.append(f"- {section['title']}{location}")
        lines.append("")

    table_index = _render_table_index(context["tables"])
    if table_index:
        lines.extend(["## Tables", "", *table_index, ""])

    body = _render_clean_blocks(context["blocks"])
    if body:
        lines.extend(["## Extracted Content", "", *body, ""])
    elif raw_markdown_text.strip():
        lines.extend(["## Extracted Content", "", raw_markdown_text.strip(), ""])

    return "\n".join(lines).strip() + "\n"


def _render_table_index(tables: list[dict[str, Any]]) -> list[str]:
    """Render a short list of tables for the primary markdown."""
    lines: list[str] = []
    for index, table in enumerate(tables, start=1):
        preview = " ".join(table["text"].split())
        preview = preview[:180] + ("..." if len(preview) > 180 else "")
        page = table.get("page")
        location = f"page {page}" if page is not None else "page ?"
        lines.append(f"- Table {index} ({location}): {preview}")
    return lines


def _render_clean_blocks(blocks: list[dict[str, Any]]) -> list[str]:
    """Render cleaned blocks with section-like spacing."""
    lines: list[str] = []
    figure_count = 0
    table_count = 0

    for block in blocks:
        kind = block["kind"]
        text = block["text"].strip()
        if not text:
            continue

        if kind in {"title", "heading"}:
            heading = text.lstrip("# ").strip()
            lines.extend([f"### {heading}", ""])
            continue

        if kind == "table":
            table_count += 1
            lines.extend(
                [f"#### Table {table_count}", "", *_table_block_to_lines(text), ""]
            )
            continue

        if kind == "figure_caption":
            figure_count += 1
            lines.extend([f"#### Figure {figure_count} Caption", "", text, ""])
            continue

        if kind == "table_caption":
            lines.extend(["#### Table Caption", "", text, ""])
            continue

        lines.extend([text, ""])

    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _table_block_to_lines(text: str) -> list[str]:
    """Render a raw table block into TSV-like markdown lines."""
    if "|" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rendered: list[str] = []
        for line in lines:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not cells:
                continue
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            rendered.append("\t".join(cells))
        if rendered:
            return ["```tsv", *rendered, "```"]

    if "\t" in text:
        return ["```tsv", *[line for line in text.splitlines() if line.strip()], "```"]

    return [text]


def _write_derived_bundle_files(
    pdf_path: Path,
    bundle_dir: Path,
    config: MinerUConfig,
) -> None:
    """Write derived helper files from the normalized primary/raw bundle."""
    primary_text = (bundle_dir / "primary.md").read_text(errors="replace")
    raw_text = ""
    raw_path = bundle_dir / "raw.md"
    if raw_path.exists():
        raw_text = raw_path.read_text(errors="replace")
    content_items = _load_json_list(_optional_path(bundle_dir / "content_list.json"))
    context = _build_context(
        raw_markdown_text=raw_text or primary_text, content_items=content_items
    )

    clean_text = _render_clean_markdown(context)
    outline_text = _render_outline_markdown(pdf_path=pdf_path, context=context)
    table_index_text = _render_table_index_markdown(context["tables"])
    tables_text = _render_tables_markdown(context["tables"])
    captions_text = _render_captions_markdown(context["captions"])
    readme_text = _render_readme_markdown(
        pdf_path=pdf_path, bundle_dir=bundle_dir, config=config
    )
    manifest_text = _render_manifest_json(
        pdf_path=pdf_path, bundle_dir=bundle_dir, context=context
    )

    (bundle_dir / "clean.md").write_text(clean_text)
    (bundle_dir / "outline.md").write_text(outline_text)
    (bundle_dir / "table_index.md").write_text(table_index_text)
    (bundle_dir / "tables.md").write_text(tables_text)
    (bundle_dir / "captions.md").write_text(captions_text)
    (bundle_dir / "README.md").write_text(readme_text)
    (bundle_dir / "manifest.json").write_text(manifest_text)


def _render_clean_markdown(context: dict[str, Any]) -> str:
    """Render a cleaned text-first markdown view."""
    lines = ["# Clean MinerU Text", ""]
    lines.extend(_render_clean_blocks(context["blocks"]))
    return "\n".join(lines).strip() + "\n"


def _render_outline_markdown(pdf_path: Path, context: dict[str, Any]) -> str:
    """Render a compact outline markdown file."""
    lines = [f"# Outline for {pdf_path.name}", ""]
    sections = context["sections"]
    if not sections:
        lines.append("- No explicit section headings detected.")
    else:
        for section in sections:
            location = ""
            if section.get("page") is not None:
                location = f" (page {section['page']})"
            lines.append(f"- {section['title']}{location}")
    return "\n".join(lines).strip() + "\n"


def _render_table_index_markdown(tables: list[dict[str, Any]]) -> str:
    """Render the table index helper file."""
    lines = ["# Table Index", ""]
    entries = _render_table_index(tables)
    if entries:
        lines.extend(entries)
    else:
        lines.append("- No tables detected.")
    return "\n".join(lines).strip() + "\n"


def _render_tables_markdown(tables: list[dict[str, Any]]) -> str:
    """Render all detected tables into a separate helper file."""
    lines = ["# Tables", ""]
    if not tables:
        lines.append("No tables detected.")
    else:
        for index, table in enumerate(tables, start=1):
            page = table.get("page")
            location = f"page {page}" if page is not None else "page ?"
            lines.extend(
                [
                    f"## Table {index} ({location})",
                    "",
                    *_table_block_to_lines(table["text"]),
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def _render_captions_markdown(captions: list[dict[str, Any]]) -> str:
    """Render figure and table captions into a separate helper file."""
    lines = ["# Captions", ""]
    if not captions:
        lines.append("No captions detected.")
    else:
        for index, caption in enumerate(captions, start=1):
            page = caption.get("page")
            location = f"page {page}" if page is not None else "page ?"
            label = (
                "Figure Caption"
                if caption["kind"] == "figure_caption"
                else "Table Caption"
            )
            lines.extend([f"## {label} {index} ({location})", "", caption["text"], ""])
    return "\n".join(lines).strip() + "\n"


def _render_readme_markdown(
    pdf_path: Path, bundle_dir: Path, config: MinerUConfig
) -> str:
    """Render the bundle README."""
    lines = [
        f"# MinerU Bundle for {pdf_path.name}",
        "",
        "This directory contains normalized MinerU outputs for one source PDF.",
        "",
        "## Files",
        "",
        "- `primary.md`: main agent-facing markdown view",
        "- `clean.md`: cleaned text-first view",
        "- `outline.md`: section outline",
        "- `table_index.md`: compact table preview list",
        "- `tables.md`: extracted tables rendered in TSV-like form",
        "- `captions.md`: extracted figure/table captions",
        "- `raw.md`: copied raw markdown from MinerU when available",
        "- `images/`: extracted image assets when MinerU emitted them",
        "- `manifest.json`: bundle metadata",
        "",
        "## MinerU Configuration",
        "",
        f"- backend: `{config.backend}`",
        f"- method: `{config.method}`",
        f"- formula: `{config.formula}`",
        f"- table: `{config.table}`",
    ]
    if config.lang:
        lines.append(f"- lang: `{config.lang}`")
    if config.source:
        lines.append(f"- source: `{config.source}`")
    if config.device:
        lines.append(f"- device: `{config.device}`")
    return "\n".join(lines).strip() + "\n"


def _render_manifest_json(
    pdf_path: Path, bundle_dir: Path, context: dict[str, Any]
) -> str:
    """Render bundle metadata as json."""
    payload: dict[str, Any] = {
        "source_pdf": pdf_path.name,
        "bundle_dir": bundle_dir.name,
        "n_blocks": len(context["blocks"]),
        "n_sections": len(context["sections"]),
        "n_tables": len(context["tables"]),
        "n_captions": len(context["captions"]),
        "has_images": (bundle_dir / "images").exists(),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _safe_int(value: Any) -> int | None:
    """Convert a value to int when possible."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_mineru_failure(
    error: Exception | None, result: subprocess.CompletedProcess[str]
) -> str:
    """Format a useful MinerU execution failure message."""
    pieces = ["MinerU failed while building a paper bundle."]
    if error is not None:
        pieces.append(str(error))
    if result.stdout.strip():
        pieces.append("stdout:\n" + result.stdout.strip())
    if result.stderr.strip():
        pieces.append("stderr:\n" + result.stderr.strip())
    return "\n\n".join(pieces)
