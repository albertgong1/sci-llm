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
from joblib import Memory

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


# Initialize joblib Memory for persistent caching
_cache_dir = Path(".cache/property_matching")
_cache_dir.mkdir(parents=True, exist_ok=True)
memory = Memory(_cache_dir, verbose=0)


@memory.cache
def _store_property_check_result(
    model_name: str,
    prompt: str,
    config_json: str,
    result: dict,
) -> dict:
    """Store and retrieve cached property check results.

    This is a simple pass-through function that joblib caches.
    When called with the same inputs, it returns the cached result.

    Args:
        model_name: Name of the LLM model
        prompt: Input prompt
        config_json: JSON string of inference config
        result: Result dictionary to cache

    Returns:
        The result dictionary

    """
    return result


async def check_if_same_property(
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    prompt: str,
    use_cache: bool = True,
) -> dict:
    """Check if two property names are the same using an LLM.

    Args:
        llm: LLM instance
        inf_gen_config: Inference generation configuration
        prompt: input to the LLM
        use_cache: Whether to use caching (default: True)

    Returns:
        dict: Dictionary containing the result of the check

    """
    # Create cache key from hashable parameters
    config_json = json.dumps(inf_gen_config.model_dump())

    # Try to get from cache - we pass a dummy result to check if it's cached
    if use_cache:
        try:
            # Use call_and_shelve to check cache without executing
            cached = _store_property_check_result.call_and_shelve(
                llm.model_name,
                prompt,
                config_json,
                {},  # dummy result
            )
            result = cached.get()
            # If we got a non-empty cached result, return it
            if result and result.get("model"):
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return result
        except Exception as e:
            logger.debug(f"Cache miss or error: {e}")

    # Build conversation
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

    # Store in cache by calling the cached function with the real result
    if use_cache:
        try:
            _store_property_check_result(
                llm.model_name,
                prompt,
                config_json,
                result,
            )
            logger.debug(f"Cached result for prompt: {prompt[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    return result


async def generate_property_name_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    prompt_template: str,
    top_k: int = TOP_K,
    left_on: list[str] = ["property_name", "context"],
    right_on: list[str] = ["property_name", "context"],
    left_suffix: str = "_x",
    right_suffix: str = "_y",
) -> pd.DataFrame:
    """For each row in df1, find the top-k matches in df2 based on property name and context

    NOTE: this queries the Gemini API and requires setting up GOOGLE_API_KEY environment variable.

    Args:
        df1: DataFrame of properties 1 with columns "embedding" and those in `left_on`.
        df2: DataFrame of properties 2 with columns "embedding" and those in `right_on`.
        llm: LLM to use for matching.
        inf_gen_config: Inference generation configuration.
        prompt_template: Prompt template to use for matching.
        top_k: Number of top matches to return.
        left_on: Columns to join on for df1.
        right_on: Columns to join on for df2.
        left_suffix: Suffix for columns in df1.
        right_suffix: Suffix for columns in df2.

    Returns:
        DataFrame containing top_k * len(df1) rows with columns from df1 and df2.

    """
    # import pdb; pdb.set_trace()
    # TODO: group df1 on left_on columns, so that we can skip some LLM calls,
    # then expand the groups to get the full df1 before returning the result
    # initial match on property name only using embedding similarity
    Y = df2.drop_duplicates(subset=["property_name"])
    # Compute the similarity matrix between property names from df1 and df2
    similarity_matrix = cosine_similarity(
        np.vstack(df1["embedding"].values),
        np.vstack(Y["embedding"].values),
    )
    top_k_matches_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :top_k]

    # further match on additional context using LLM
    matches = []
    for i in tqdm(range(len(df1)), desc="Processing df1"):
        # if i < 74:
        #     continue
        # import pdb; pdb.set_trace()
        x = df1.iloc[i].to_dict()
        # Step 1. Find the rows in Y whose property name is in the top_k matches for x
        # NOTE: this may yield more than k matches since some rows share property name, but not context
        top_k_matches = Y.iloc[top_k_matches_indices[i]]["property_name"].tolist()
        df2_top_k = df2[df2["property_name"].isin(top_k_matches)]
        logger.info(f"Found {len(df2_top_k)} matches for {x['property_name']}")
        # Step 2. Construct async tasks, reusing the same task for rows with the same property name and context
        tasks = OrderedDict()
        idx_to_task_id = {}
        for idx, y in df2_top_k.iterrows():
            # NOTE: rename the variables to avoid conflicts when substituting them into the prompt template
            x_variables = {k + "_1": x[k] for k in left_on}
            y_variables = {k + "_2": y[k] for k in right_on}
            task_id = (json.dumps(x_variables), json.dumps(y_variables))
            idx_to_task_id[idx] = task_id
            if task_id not in tasks:
                prompt = prompt_template.format(
                    **x_variables,
                    **y_variables,
                )
                task = check_if_same_property(llm, inf_gen_config, prompt)
                tasks[task_id] = task
        # Execute all tasks concurrently
        # import pdb; pdb.set_trace()
        results = await asyncio.gather(*tasks.values())
        results = {task_id: result for task_id, result in zip(tasks.keys(), results)}
        # Step 3. Combine the results with the rows in df2_top_k
        for idx, y in df2_top_k.iterrows():
            result = results[idx_to_task_id[idx]]
            matches.append(
                {
                    **x,
                    **result,
                    "y_id": idx,  # later use this to join the results with df2 to get the remaining columns in df2
                }
            )
    # Step 4. Merge the results with df2 to get the remaining columns in df2
    df_matches = pd.DataFrame(matches)
    df_matches = df_matches.merge(
        df2,
        left_on="y_id",
        right_index=True,
        how="left",
        suffixes=(left_suffix, right_suffix),
    )
    return df_matches


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
