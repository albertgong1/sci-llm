# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

# Single column figures
TICK_FONT_SIZE = 6
MINOR_TICK_FONT_SIZE = 3
LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 7
MARKER_ALPHA = 1
MARKER_SIZE = 3
LINE_ALPHA = 0.75
OUTWARD = 4


def get_display_label(agent: str, model: str, multiline: bool = True) -> str:
    """Get a consistent display label for an agent+model combination.

    Uses LaTeX formatting for matplotlib labels:
    - zeroshot with GPT model -> "gpt-5.2" (or appropriate model)
    - zeroshot with Gemini model -> "gemini-3-pro" (or appropriate model)
    - codex -> "codex"
    - gemini-cli -> "gemini-cli"
    - terminus-2 with GPT model -> "T2-gpt"
    - terminus-2 with Gemini model -> "T2-gemini"
    - qwen -> "qwen-code"

    Args:
        agent: The agent name (e.g., "zeroshot", "codex", "terminus-2").
        model: The model name (e.g., "openai/gpt-4", "gemini/gemini-3-pro-preview").
        multiline: If True, split long labels across two lines (unused with new format).

    Returns:
        A formatted display label string.

    """
    model_short = model.split("/")[-1]
    is_gemini = "gemini" in model_short

    if agent == "zeroshot":
        # Return the model display name for zeroshot
        return get_model_display_name(model_short)
    elif agent == "terminus-2":
        return "T2-gemini" if is_gemini else "T2-gpt"
    elif agent == "codex":
        return "codex"
    elif agent == "gemini-cli":
        return "gemini-cli"
    elif agent == "qwen":
        return "qwen-code"
    return agent


def get_model_display_name(model: str) -> str:
    """Get a display name for a model.

    Args:
        model: The model name (e.g., "gpt-5.2-2025-08-07", "gemini-3-pro-preview").

    Returns:
        A formatted display name string.

    """
    model_short = model.split("/")[-1]

    # Map model names to display names
    if "gemini-3-pro" in model_short:
        return "gemini-3-pro"
    elif "gemini-3-flash" in model_short:
        return "gemini-3-flash"
    elif "gpt-5.2" in model_short:
        return "gpt-5.2"
    elif "gpt-5.1" in model_short:
        return "gpt-5.1"
    elif "gpt-5-mini" in model_short:
        return "gpt-5-mini"
    elif "qwen3-max" in model_short:
        return "qwen3-max"

    return model_short
