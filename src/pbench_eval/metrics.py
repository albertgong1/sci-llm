"""Implementation of metrics."""

import pandas as pd
import logging
from typing import Literal

from llm_utils import get_llm, InferenceGenerationConfig
import json

from pbench_eval.match import generate_embeddings, generate_property_name_matches
from pbench_eval.utils import scorer_pymatgen, score_value

logger = logging.getLogger(__name__)

SERVER = "gemini"
MODEL_NAME = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 4096


def get_conditions_for_property(
    property_name: str,
    rubric_df: pd.DataFrame,
    important_only: bool = True,
) -> list[str]:
    """Get the condition column names for a given property from the rubric.

    Args:
        property_name: The property name to look up
        rubric_df: DataFrame containing the rubric with condition_name column
        important_only: If True, only return conditions marked as "important"

    Returns:
        List of condition column names for this property

    """
    # Filter to rows for this property where condition_name is not empty
    property_conditions = rubric_df[
        (rubric_df["property_name"] == property_name)
        & (rubric_df["condition_name"].notna())
        & (rubric_df["condition_name"] != "")
    ]

    if important_only and "not_important" in rubric_df.columns:
        # Keep only rows where not_important != "not_important"
        property_conditions = property_conditions[
            property_conditions["not_important"] != "not_important"
        ]

    return property_conditions["condition_name"].tolist()


def check_conditions_match(
    row: pd.Series,
    condition_columns: list[str],
    gt_suffix: str = "_gt",
    pred_suffix: str = "_pred",
) -> bool:
    """Check if all condition columns match between ground truth and prediction.

    Args:
        row: DataFrame row containing both gt and pred condition values
        condition_columns: List of condition column names (without suffixes)
        gt_suffix: Suffix for ground truth columns
        pred_suffix: Suffix for prediction columns

    Returns:
        True if all conditions match (or are both missing), False otherwise

    """
    for cond in condition_columns:
        gt_col = f"{cond}{gt_suffix}"
        pred_col = f"{cond}{pred_suffix}"

        # If columns don't exist, skip this condition
        if gt_col not in row.index or pred_col not in row.index:
            continue

        gt_val = row.get(gt_col)
        pred_val = row.get(pred_col)

        # Both missing is OK
        if pd.isna(gt_val) and pd.isna(pred_val):
            continue

        # One missing, one not - not a match
        if pd.isna(gt_val) or pd.isna(pred_val):
            return False

        # Compare values (convert to string for comparison)
        if str(gt_val).strip().lower() != str(pred_val).strip().lower():
            return False

    return True


#
# Property extraction metrics
#
def construct_context(row: pd.Series) -> str:
    """Construct the context from the ground-truth property row.

    Args:
        row: Ground-truth property row

    Returns:
        Context string

    """
    return json.dumps({k: v for k, v in row.items() if k == "value_unit"})


def compute_recall_per_material_property(
    df: pd.DataFrame,
    conversion_df: pd.DataFrame | None = None,
    matching_mode: Literal["material", "conditions"] = "material",
    material_column: str = "material_or_system",
    rubric_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score recall for a dataframe of predicted-ground truth pairs.

    We use the following formula to score recall for each (material, property) pair:
    recall = max(scores) if scores else 0.0
    where scores is a list of scores for each matching row in the group and a score is computed using the score_value function, which applies the rubric to the predicted and ground truth values.

    Args:
        df: DataFrame with the following columns:
            - property_name_gt: Ground truth property name.
            - value_string_gt: Ground truth value string.
            - value_string_pred: Predicted value string.
            - rubric: Rubric for the property.
            - For material mode: material_or_system_gt, material_or_system_pred
            - For conditions mode: condition columns with _gt and _pred suffixes
        conversion_df: DataFrame with unit conversion factors. If None, skip unit conversion.
        matching_mode: "material" for material-based matching (supercon),
                       "conditions" for condition-based matching (biosurfactants)
        material_column: Column name for material matching (used when matching_mode="material")
        rubric_df: DataFrame containing the rubric with condition definitions
                   (required when matching_mode="conditions")

    Returns:
        DataFrame with recall score for each (material, property) pair.

    """
    if matching_mode == "conditions" and rubric_df is None:
        raise ValueError("rubric_df is required when matching_mode='conditions'")

    # Group by ground truth material AND property name to calculate recall scores
    grouped = df.groupby(["refno", "agent", "model", "id_gt"], dropna=False)
    logger.info(f"Processing {len(grouped)} unique (material, property) pairs...")
    results = []

    gt_material_col = f"{material_column}_gt"
    pred_material_col = f"{material_column}_pred"

    for (refno, agent, model, id_gt), group in grouped:
        matching_rows = []
        property_name = group["property_name_gt"].iloc[0]

        for idx, row in group.iterrows():
            # Property must match (using the "is_match" column)
            if not row["is_match"]:
                continue

            if matching_mode == "material":
                # Material-based matching (supercon)
                if pd.notna(row.get(gt_material_col)) and pd.notna(
                    row.get(pred_material_col)
                ):
                    if scorer_pymatgen(
                        str(row[gt_material_col]),
                        str(row[pred_material_col]),
                    ):
                        matching_rows.append(row)
            else:
                # Condition-based matching (biosurfactants)
                condition_columns = get_conditions_for_property(
                    property_name, rubric_df, important_only=True
                )
                if check_conditions_match(row, condition_columns, "_gt", "_pred"):
                    matching_rows.append(row)

        num_matches = len(matching_rows)

        if num_matches == 0:
            # No matches, score is 0
            recall_score = 0.0
        else:
            # At least one match, calculate scores and take max
            scores = []

            for row in matching_rows:
                # Skip if values are missing
                if (
                    pd.isna(row["value_string_pred"])
                    or pd.isna(row["value_string_gt"])
                    or pd.isna(row["rubric"])
                ):
                    continue

                # Calculate score
                score = score_value(
                    pred_value=row["value_string_pred"],
                    answer_value=row["value_string_gt"],
                    rubric=row["rubric"],
                    conversion_df=conversion_df,
                )
                scores.append(score)

            # Take maximum score
            recall_score = max(scores) if scores else 0.0

        result = {
            "refno": refno,
            "agent": agent,
            "model": model,
            "id_gt": id_gt,
            "property_name_gt": property_name,
            "value_string_gt": ", ".join(
                list(set([str(row["value_string_gt"]) for _, row in group.iterrows()]))
            ),
            "num_property_matches": len(group),
            "num_property_material_matches": num_matches,
            "recall_score": recall_score,
            "matches": ", ".join(
                [
                    f"{row['property_name_pred']}: {row['value_string_pred']}"
                    for row in matching_rows
                ]
            ),
            "answers": ", ".join(
                [
                    f"{row['value_string_pred']}"
                    for _, row in group.iterrows()
                    if row["is_match"]
                ]
            ),
        }

        # Add material columns if in material mode
        if matching_mode == "material" and gt_material_col in group.columns:
            result["material_or_system_gt"] = group[gt_material_col].iloc[0]
            result["material_or_system_pred"] = ", ".join(
                list(
                    set(
                        [
                            str(row[pred_material_col])
                            for _, row in group.iterrows()
                            if pred_material_col in row.index
                        ]
                    )
                )
            )

        results.append(result)

    df_results = pd.DataFrame(results)
    return df_results


def compute_precision_per_material_property(
    df: pd.DataFrame,
    conversion_df: pd.DataFrame | None = None,
    matching_mode: Literal["material", "conditions"] = "material",
    material_column: str = "material_or_system",
    rubric_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score precision for a dataframe of predicted-ground truth pairs.

    Args:
        df: DataFrame with the following columns:
            - property_name_pred: Predicted property name.
            - value_string_pred: Predicted value string.
            - value_string_gt: Ground truth value string.
            - rubric: Rubric for the property.
            - For material mode: material_or_system_gt, material_or_system_pred
            - For conditions mode: condition columns with _gt and _pred suffixes
        conversion_df: DataFrame with unit conversion factors. If None, skip unit conversion.
        matching_mode: "material" for material-based matching (supercon),
                       "conditions" for condition-based matching (biosurfactants)
        material_column: Column name for material matching (used when matching_mode="material")
        rubric_df: DataFrame containing the rubric with condition definitions
                   (required when matching_mode="conditions")

    Returns:
        DataFrame with precision score for each predicted material.

    """
    if matching_mode == "conditions" and rubric_df is None:
        raise ValueError("rubric_df is required when matching_mode='conditions'")

    # Group by predicted material and calculate precision scores
    grouped = df.groupby(["refno", "agent", "model", "id_pred"], dropna=False)
    logger.info(f"Processing {len(grouped)} unique predicted materials...")
    results = []

    gt_material_col = f"{material_column}_gt"
    pred_material_col = f"{material_column}_pred"

    for (refno, agent, model, id_pred), group in grouped:
        matching_rows = []
        property_name = group["property_name_pred"].iloc[0]

        for idx, row in group.iterrows():
            # Property must match (using the "is_match" column)
            if not row["is_match"]:
                continue

            if matching_mode == "material":
                # Material-based matching (supercon)
                if pd.notna(row.get(pred_material_col)) and pd.notna(
                    row.get(gt_material_col)
                ):
                    if scorer_pymatgen(
                        str(row[pred_material_col]),
                        str(row[gt_material_col]),
                    ):
                        matching_rows.append(row)
            else:
                # Condition-based matching (biosurfactants)
                condition_columns = get_conditions_for_property(
                    property_name, rubric_df, important_only=True
                )
                if check_conditions_match(row, condition_columns, "_gt", "_pred"):
                    matching_rows.append(row)

        num_matches = len(matching_rows)

        if num_matches == 0:
            # No matches, score is 0
            precision_score = 0.0
        else:
            # At least one match, calculate scores and take max
            scores = []

            for row in matching_rows:
                # Skip if values are missing
                if (
                    pd.isna(row["value_string_pred"])
                    or pd.isna(row["value_string_gt"])
                    or pd.isna(row["rubric"])
                ):
                    continue

                # Calculate score
                score = score_value(
                    pred_value=row["value_string_pred"],
                    answer_value=row["value_string_gt"],
                    rubric=row["rubric"],
                    conversion_df=conversion_df,
                )
                scores.append(score)

            # Take maximum score
            precision_score = max(scores) if scores else 0.0

        result = {
            "refno": refno,
            "agent": agent,
            "model": model,
            "id_pred": id_pred,
            "property_name_pred": property_name,
            "value_string_pred": ", ".join(
                list(
                    set([str(row["value_string_pred"]) for _, row in group.iterrows()])
                )
            ),
            "num_property_matches": len(group),
            "num_property_material_matches": num_matches,
            "precision_score": precision_score,
            "matches": ", ".join(
                [
                    f"{row['property_name_gt']}: {row['value_string_gt']}"
                    for row in matching_rows
                ]
            ),
            "answers": ", ".join(
                [
                    f"{row['value_string_gt']}"
                    for _, row in group.iterrows()
                    if row["is_match"]
                ]
            ),
        }

        # Add material columns if in material mode
        if matching_mode == "material" and pred_material_col in group.columns:
            result["material_or_system_pred"] = group[pred_material_col].iloc[0]
            result["material_or_system_gt"] = ", ".join(
                list(
                    set(
                        [
                            str(row[gt_material_col])
                            for _, row in group.iterrows()
                            if gt_material_col in row.index
                        ]
                    )
                )
            )

        results.append(result)

    df_results = pd.DataFrame(results)
    return df_results


def add_property_name_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Add embeddings and context to a dataframe of properties.

    NOTE: this queries the Gemini API and requires setting up GOOGLE_API_KEY environment variable.

    Args:
        df: DataFrame with properties.

    Returns:
        DataFrame with embeddings.

    """
    unique_property_names = list(set(df["property_name"].tolist()))
    embeddings = generate_embeddings(unique_property_names)
    df_embeddings = pd.DataFrame(
        {
            "property_name": unique_property_names,
            "embedding": embeddings,
        }
    )
    df = df.merge(df_embeddings, on="property_name", how="left")
    return df


async def compute_mean_recall_precision(
    df_pred: pd.DataFrame,
    df_gt: pd.DataFrame,
    property_matching_prompt_template: str,
    conversion_df: pd.DataFrame,
) -> tuple[float, float]:
    """Calculate mean recall and precision metrics for a single task.

    Args:
        df_pred: DataFrame with predicted properties.
        df_gt: DataFrame with ground truth properties.
            NOTE: should include a column "rubric"
        property_matching_prompt_template: Prompt template for property matching.
        conversion_df: DataFrame with unit conversion factors.

    Returns:
        tuple[float, float]: Mean recall and precision.

    """
    assert len(df_pred["refno"].unique()) == 1, "Expected only one refno per file"
    assert len(df_gt["refno"].unique()) == 1, "Expected only one refno per file"
    # assert that df_gt contains the necessary columns
    assert "rubric" in df_gt.columns, (
        "Ground truth dataframe must contain a 'rubric' column"
    )

    # Initialize LLM that will be used for all property matching
    llm = get_llm(SERVER, MODEL_NAME)

    # Create inference config
    inf_gen_config = InferenceGenerationConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        output_format="json",
    )

    # -- Generate embeddings for predicted properties --
    df_pred = add_property_name_embeddings(df_pred)
    df_gt = add_property_name_embeddings(df_gt)
    df_pred["context"] = df_pred["location.evidence"]
    df_gt["context"] = df_gt.apply(construct_context, axis=1)

    # -- Core functionality to compute recall --
    # Generate matches between predicted and ground truth properties
    # TODO: save these matches to file so that we can analyze the LLM-as-a-judge predictions in them later
    df_pred_matches, _ = await generate_property_name_matches(
        df_pred,
        df_gt,
        llm,
        inf_gen_config,
        property_matching_prompt_template,
        left_on=["property_name", "context"],
        right_on=["property_name", "context"],
        left_suffix="_pred",
        right_suffix="_gt",
    )
    df_recall = compute_recall_per_material_property(df_pred_matches, conversion_df)

    # -- Core functionality to compute precision --
    # Generate matches between predicted and ground truth properties
    # TODO: save these matches to file so that we can analyze the LLM-as-a-judge predictions in them later
    df_gt_matches, _ = await generate_property_name_matches(
        df_gt,
        df_pred,
        llm,
        inf_gen_config,
        property_matching_prompt_template,
        left_on=["property_name", "context"],
        right_on=["property_name", "context"],
        left_suffix="_gt",
        right_suffix="_pred",
    )
    df_precision = compute_precision_per_material_property(df_gt_matches, conversion_df)

    # Compute mean recall and precision
    mean_recall = df_recall["recall_score"].mean()
    mean_precision = df_precision["precision_score"].mean()

    return mean_recall, mean_precision
