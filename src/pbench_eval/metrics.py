"""Implementation of metrics."""

import pandas as pd
import logging
from llm_utils import get_llm, InferenceGenerationConfig
import json

from pbench_eval.match import generate_embeddings, generate_property_name_matches
from pbench_eval.utils import scorer_pymatgen, score_value

logger = logging.getLogger(__name__)

SERVER = "gemini"
MODEL_NAME = "gemini-3-flash-preview"
MAX_OUTPUT_TOKENS = 4096

# Initialize LLM that will be used for all property matching
llm = get_llm(SERVER, MODEL_NAME)

# Create inference config
inf_gen_config = InferenceGenerationConfig(
    max_output_tokens=MAX_OUTPUT_TOKENS,
    output_format="json",
)


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
    df: pd.DataFrame, conversion_df: pd.DataFrame
) -> pd.DataFrame:
    """Score recall for a dataframe of predicted-ground truth pairs.

    We use the following formula to score recall for each (material, property) pair:
    recall = max(scores) if scores else 0.0
    where scores is a list of scores for each matching row in the group and a score is computed using the score_value function, which applies the rubric to the predicted and ground truth values.

    Args:
        df: DataFrame with the following columns:
            - material_or_system_gt: Ground truth material or system.
            - property_name_gt: Ground truth property name.
            - value_string_gt: Ground truth value string.
            - material_or_system_pred: Predicted material or system.
            - value_string_pred: Predicted value string.
            - rubric: Rubric for the property.
        conversion_df: DataFrame with unit conversion factors.

    Returns:
        DataFrame with recall score for each (material, property) pair.

    """
    # Group by ground truth material AND property name to calculate recall scores
    grouped = df.groupby(
        ["refno", "model", "material_or_system_gt", "property_name_gt"]
    )
    logger.info(f"Processing {len(grouped)} unique (material, property) pairs...")
    # import pdb; pdb.set_trace()
    results = []

    for (refno, model, material_gt, property_gt), group in grouped:
        # Check which rows have matching materials using scorer_pymatgen
        matching_rows = []

        for idx, row in group.iterrows():
            # Check if materials match using pymatgen
            if pd.notna(material_gt) and pd.notna(row["material_or_system_pred"]):
                if scorer_pymatgen(
                    str(material_gt), str(row["material_or_system_pred"])
                ):
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

        results.append(
            {
                "refno": refno,
                "model": model,
                "material_or_system_gt": material_gt,
                "property_name_gt": property_gt,
                "value_string_gt": ", ".join(
                    list(
                        set(
                            [str(row["value_string_gt"]) for _, row in group.iterrows()]
                        )
                    )
                ),
                "num_property_matches": len(group),
                "num_property_material_matches": num_matches,
                "material_or_system_pred": ", ".join(
                    list(
                        set(
                            [
                                str(row["material_or_system_pred"])
                                for _, row in group.iterrows()
                            ]
                        )
                    )
                ),
                "recall_score": recall_score,
                "matches": ", ".join(
                    [
                        f"{row['property_name_pred']}: {row['value_string_pred']}"
                        for row in matching_rows
                    ]
                ),
            }
        )

    df_results = pd.DataFrame(results)
    return df_results


def compute_precision_per_material_property(
    df: pd.DataFrame, conversion_df: pd.DataFrame
) -> pd.DataFrame:
    """Score precision for a dataframe of predicted-ground truth pairs.

    Args:
        df: DataFrame with the following columns:
            - material_or_system_pred: Predicted material or system.
            - value_string_pred: Predicted value string.
            - value_string_gt: Ground truth value string.
            - rubric: Rubric for the property.
        conversion_df: DataFrame with unit conversion factors.

    Returns:
        DataFrame with precision score for each predicted material.

    """
    # Group by predicted material and calculate precision scores
    grouped = df.groupby(
        ["refno", "model", "material_or_system_pred", "property_name_pred"]
    )
    logger.info(f"Processing {len(grouped)} unique predicted materials...")
    results = []

    for (refno, model, material_pred, property_pred), group in grouped:
        # Check which rows have matching materials using scorer_pymatgen
        matching_rows = []

        for idx, row in group.iterrows():
            # Check if materials match using pymatgen
            if pd.notna(material_pred) and pd.notna(row["material_or_system_gt"]):
                if scorer_pymatgen(
                    str(material_pred), str(row["material_or_system_gt"])
                ):
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

        results.append(
            {
                "refno": refno,
                "model": model,
                "material_or_system_pred": material_pred,
                "property_name_pred": property_pred,
                "value_string_pred": ", ".join(
                    list(
                        set(
                            [
                                str(row["value_string_pred"])
                                for _, row in group.iterrows()
                            ]
                        )
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
            }
        )
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
