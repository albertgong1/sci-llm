"""Functionality to match property names, materials, and conditions."""

# standard imports
import asyncio
from collections import OrderedDict
import logging
import json
import os
import time

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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
EMBEDDING_BATCH_SIZE = max(1, int(os.environ.get("PBENCH_EMBED_BATCH_SIZE", "25")))
EMBEDDING_TIMEOUT_MS = max(
    1000, int(os.environ.get("PBENCH_EMBED_TIMEOUT_MS", "300000"))
)
EMBEDDING_MAX_RETRIES = max(1, int(os.environ.get("PBENCH_EMBED_MAX_RETRIES", "6")))
EMBEDDING_RETRY_BASE_SEC = float(os.environ.get("PBENCH_EMBED_RETRY_BASE_SEC", "2.0"))
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
TOP_K = 3


def _is_retryable_embedding_error(exc: Exception) -> bool:
    """Return whether an embedding failure looks transient and worth retrying."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.WriteTimeout,
        ),
    ):
        return True

    message = str(exc).lower()
    retryable_markers = (
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "deadline exceeded",
        "dns",
        "failed to lookup address information",
        "gateway timeout",
        "internal server error",
        "operation timed out",
        "rate limit",
        "read timeout",
        "remoteprotocolerror",
        "service unavailable",
        "stream disconnected",
        "timed out",
        "try again",
        "unavailable",
    )
    return any(marker in message for marker in retryable_markers)


def _embed_batch_once(
    client: genai.Client, property_names: list[str]
) -> list[np.ndarray]:
    """Generate one batch of embeddings in order."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL_NAME,
        contents=property_names,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return [np.asarray(emb.values) for emb in result.embeddings]


def _embed_batch_with_retry(
    client: genai.Client,
    property_names: list[str],
    *,
    max_retries: int = EMBEDDING_MAX_RETRIES,
) -> list[np.ndarray]:
    """Retry transient embedding failures and split batches if needed."""
    for attempt in range(1, max_retries + 1):
        try:
            return _embed_batch_once(client, property_names)
        except Exception as exc:
            if not _is_retryable_embedding_error(exc) or attempt == max_retries:
                if len(property_names) > 1 and _is_retryable_embedding_error(exc):
                    midpoint = len(property_names) // 2
                    logger.warning(
                        "Embedding batch of %d items still timed out after %d attempt(s); splitting the batch.",
                        len(property_names),
                        attempt,
                    )
                    left_embeddings = _embed_batch_with_retry(
                        client,
                        property_names[:midpoint],
                        max_retries=max_retries,
                    )
                    right_embeddings = _embed_batch_with_retry(
                        client,
                        property_names[midpoint:],
                        max_retries=max_retries,
                    )
                    return [*left_embeddings, *right_embeddings]
                raise

            sleep_sec = EMBEDDING_RETRY_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "Transient embedding failure on attempt %d/%d for batch size %d: %s. Retrying in %.1fs.",
                attempt,
                max_retries,
                len(property_names),
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    raise RuntimeError("Embedding retry loop exhausted unexpectedly.")


def generate_embeddings(property_names: list[str]) -> list[np.ndarray]:
    """Generate embeddings for a list of property names.

    Args:
        property_names: List of property names to generate embeddings for.

    Returns:
        List of embeddings.

    """
    client = genai.Client(http_options=types.HttpOptions(timeout=EMBEDDING_TIMEOUT_MS))
    embeddings: list[np.ndarray] = []
    for i in range(0, len(property_names), EMBEDDING_BATCH_SIZE):
        batch = property_names[i : i + EMBEDDING_BATCH_SIZE]
        embeddings.extend(_embed_batch_with_retry(client, batch))

    return embeddings


async def check_if_same_property(
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    prompt: str,
    property_name_1: str,
    property_name_2: str,
) -> tuple[dict, dict]:
    """Check if two property names are the same using an LLM.

    Args:
        llm: LLM instance
        inf_gen_config: Inference generation configuration
        prompt: input to the LLM
        property_name_1: First property name
        property_name_2: Second property name

    Returns:
        dict: Dictionary containing the result of the check
        dict: Dictionary containing the raw response from the LLM

    """
    if property_name_1.strip() == property_name_2.strip():
        # import pdb; pdb.set_trace()
        # Shortcut: if property names are identical, return match
        result = {
            "is_match": True,
            "reason": "Property names are identical",
            "confidence": "high",
            "matched_via": "exact",
            "judge": llm.model_name,
            "prompt": None,
        }
        return result, {}

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
        "judge": llm.model_name,
        "prompt": prompt,
    }

    return result, {**response.model_dump(), "judge": llm.model_name}


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

    # -- further match on additional context using LLM --
    matches = []
    # construct all tasks first to run them concurrently
    tasks = OrderedDict()
    idx_to_task_id = {}
    for i in tqdm(range(len(df1)), desc="Processing df1"):
        x = df1.iloc[i].to_dict()
        # Find the rows in Y whose property name is in the top_k matches for x
        # NOTE: this may yield more than k matches since some rows share property name, but not context
        top_k_matches = Y.iloc[top_k_matches_indices[i]]["property_name"].tolist()
        df2_top_k = df2[df2["property_name"].isin(top_k_matches)]
        logger.debug(f"Found {len(df2_top_k)} matches for {x['property_name']}")
        # Construct async tasks, reusing the same task for rows with the same property name and context
        for idx, y in df2_top_k.iterrows():
            # NOTE: rename the variables to avoid conflicts when substituting them into the prompt template
            x_variables = {k + "_1": x[k] for k in left_on}
            y_variables = {k + "_2": y[k] for k in right_on}
            task_id = (json.dumps(x_variables), json.dumps(y_variables))
            idx_to_task_id[(i, idx)] = task_id
            if task_id not in tasks:
                prompt = prompt_template.format(
                    **x_variables,
                    **y_variables,
                )
                task = check_if_same_property(
                    llm, inf_gen_config, prompt, x["property_name"], y["property_name"]
                )
                tasks[task_id] = task
    # Execute all tasks concurrently
    if False:
        BATCH_SIZE = 100
    else:
        BATCH_SIZE = len(tasks)  # run all at once
    results_data = []
    for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc="Calling LLM API in batches"):
        batch_tasks = {k: tasks[k] for k in list(tasks.keys())[i : i + BATCH_SIZE]}
        batch_results = await asyncio.gather(*batch_tasks.values())
        results_data.extend(batch_results)
        # await asyncio.sleep(1)  # brief pause to avoid overwhelming the API
    results = {
        task_id: result for task_id, (result, _) in zip(tasks.keys(), results_data)
    }

    # Combine the results with the rows in df2_top_k
    for i in tqdm(range(len(df1)), desc="Processing df1"):
        x = df1.iloc[i].to_dict()
        top_k_matches = Y.iloc[top_k_matches_indices[i]]["property_name"].tolist()
        df2_top_k = df2[df2["property_name"].isin(top_k_matches)]
        for idx, y in df2_top_k.iterrows():
            result = results[idx_to_task_id[(i, idx)]]
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

    # Return LLM responses along with match results
    responses = [response for _, response in results_data]
    # df_responses = pd.DataFrame({"responses" : responses})
    # expand the properties dict into separate columns
    df_responses = pd.json_normalize(responses)

    return df_matches, df_responses


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
