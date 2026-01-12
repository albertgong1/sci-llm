"""Functionality to match property names, materials, and conditions."""

# standard imports
import numpy as np
import logging
from tqdm import tqdm
from collections import OrderedDict
import asyncio
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# llm imports
from google import genai
from google.genai import types

# pbench imports
from llm_utils import (
    LLMChat,
    InferenceGenerationConfig,
    Conversation,
    Message,
    LLMChatResponse,
)

logger = logging.getLogger(__name__)


#
# Functionality for matching property names
#
BATCH_SIZE = 100
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
TOP_K = 3


def generate_embeddings(property_names: list[str]) -> list[np.ndarray]:
    """Generate embeddings for a list of property names.

    Args:
        property_names: List of property names to generate embeddings for.

    Returns:
        List of embeddings.

    """
    client = genai.Client()
    embeddings = []
    for i in range(0, len(property_names), BATCH_SIZE):
        batch = property_names[i : i + BATCH_SIZE]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=batch,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings.extend([emb.values for emb in result.embeddings])

    return embeddings


async def check_if_same_property(
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    name1: str,
    context1: str,
    name2: str,
    context2: str,
    prompt_template: str,
) -> dict:
    """Check if two property names are the same using an LLM.

    Args:
        llm: LLM instance
        inf_gen_config: Inference generation configuration
        name1: name of the first property
        context1: context of the first property
        name2: name of the second property
        context2: context of the second property
        prompt_template: prompt template to use

    Returns:
        dict: Dictionary containing the result of the check

    """
    # Build conversation
    prompt = prompt_template.format(
        name1=name1,
        context1=context1,  # use the evidence field as the context
        name2=name2,
        context2=context2,
    )
    conv = Conversation(messages=[Message(role="user", content=[prompt])])

    # Generate response
    response: LLMChatResponse = await llm.generate_response_async(conv, inf_gen_config)
    if response.pred:
        is_match = response.pred.get("is_match", False)
        reason = response.pred.get("reason", "No reason provided")
        confidence = response.pred.get("confidence")
        matched_via = response.pred.get("matched_via")
    else:
        is_match = False
        reason = "Empty response from LLM"
        confidence = None
        matched_via = None

    result = {
        "is_match": is_match,
        "reason": reason,
        "confidence": confidence,
        "matched_via": matched_via,
        "model": llm.model_name,
        "prompt": prompt,
    }

    return result


async def generate_property_name_matches(
    df_pred: pd.DataFrame,
    df_gt_embeddings: pd.DataFrame,
    refno: str,
    df_gt_refno: pd.DataFrame,
    pred_matches_path: Path,
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
) -> pd.DataFrame:
    """Get top-k matches for each predicted property name (to compute precision)

    We should take in df_pred and df_gt.
    Then take the unique property names from each.
    Then generate embeddings for each of the unique pred and gt property names.
    Then join on the unique property names.
    Compute the similarity matrix between the embeddings.

    Args:
        df_pred: DataFrame of predicted properties.
        df_gt_embeddings: DataFrame of ground truth embeddings.
        refno: Reference number of the properties.
        df_gt_refno: DataFrame of ground truth properties.
        pred_matches_path: Path to save the predicted matches.
        llm: LLM to use for matching.
        inf_gen_config: Inference generation configuration.

    Returns:
        DataFrame of predicted matches.

    """
    # Compute the similarity matrix between the predicted and ground truth properties
    similarity_matrix = cosine_similarity(
        np.vstack(df_pred["embedding"].values),
        np.vstack(df_gt_embeddings["embedding"].values),
    )
    top_k_matches_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :TOP_K]
    pred_matches = []
    for i in tqdm(range(len(df_pred)), desc=f"Processing refno {refno}"):
        pred = df_pred.iloc[i].to_dict()
        # get the top-k matches from the ground truth embeddings
        top_k_matches = df_gt_embeddings.iloc[top_k_matches_indices[i]][
            "property_name"
        ].tolist()
        df_gt_top_k = df_gt_refno[df_gt_refno["property_name"].isin(top_k_matches)]
        # Create async tasks for all unique matches
        tasks = OrderedDict()
        idx_to_task_id = {}
        for idx, gt in df_gt_top_k.iterrows():
            name1 = pred["property_name"]
            context1 = pred["location.evidence"]
            name2 = gt["property_name"]
            context2 = json.dumps(
                {
                    k: v
                    for k, v in gt.items()
                    if k.startswith("conditions.") or k == "value_unit"
                }
            )
            task_id = (name1, context1, name2, context2)
            idx_to_task_id[idx] = task_id
            if task_id not in tasks:
                task = check_if_same_property(
                    llm, inf_gen_config, name1, context1, name2, context2
                )
                tasks[task_id] = task
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks.values())
        results = {task_id: result for task_id, result in zip(tasks.keys(), results)}
        # Combine the results with the predicted and ground truth properties
        for idx, gt in df_gt_top_k.iterrows():
            result = results[idx_to_task_id[idx]]
            pred_matches.append(
                {
                    **pred,
                    **result,
                    "gt_id": idx,  # use this to join the results with the ground truth properties
                }
            )
    # save pred_matches to csv
    df_pred_matches = pd.DataFrame(pred_matches)
    df_pred_matches = df_pred_matches.merge(
        df_gt_refno,
        left_on="gt_id",
        right_index=True,
        how="left",
        suffixes=("_pred", "_gt"),
    )
    # df_pred_matches.to_csv(pred_matches_path, index=False)
    # logging.info(f"Saved pred matches to {pred_matches_path}")
    return df_pred_matches


#
# Functionality for matching material names
#
def is_material_name_same(material1: str, material2: str) -> bool:
    """Check if two material names are the same.

    Args:
        material1: First material name.
        material2: Second material name.

    Returns:
        True if the material names are the same, False otherwise.

    """
    return material1 == material2
