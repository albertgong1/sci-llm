"""Async script to extract all properties from a paper."""

# 1. Load the paper

# 2. Load the properties

# 3. Form prompt from properties and paper

# 4. Call the model

# 5. Save the results

import argparse
import asyncio
import json
import logging
import math
from copy import deepcopy
from pathlib import Path

import pandas as pd

from phantom_eval import get_parser as get_base_parser
from phantom_eval._types import Conversation, LLMChatResponse
from phantom_eval.agents import get_agent
from phantom_eval.agents.common import Agent
from phantom_eval.llm import InferenceGenerationConfig, LLMChat, get_llm

from prompts import (
    COT_EXAMPLES,
    LLMPrompt,
    get_llm_prompt,
)
from .utils import load_data, setup_logging

logger = logging.getLogger(__name__)


def get_model_kwargs(args: argparse.Namespace) -> dict:
    """Get the model kwargs for the given arguments.

    Args:
        args: The arguments namespace.

    Returns:
        A dictionary of model kwargs.

    """
    match args.server:
        case "vllm":
            model_kwargs = dict(
                max_model_len=args.inf_vllm_max_model_len,
                tensor_parallel_size=args.inf_vllm_tensor_parallel_size,
                use_api=not args.inf_vllm_offline
                or args.method in ["react", "act", "react->cot-sc", "cot-sc->react"],
                lora_path=args.inf_vllm_lora_path,
                port=args.inf_vllm_port,
            )
        case _:
            model_kwargs = dict(
                usage_tier=args.inf_usage_tier,
                enforce_rate_limits=not args.inf_relax_rate_limits,
                llms_rpm_tpm_config_fpath=args.inf_llms_rpm_tpm_config_fpath,
            )
    return model_kwargs


def get_agent_kwargs(args: argparse.Namespace) -> dict:
    """Get the agent kwargs for the given arguments.

    Args:
        args: The arguments namespace.

    Returns:
        A dictionary of agent kwargs.

    """
    match args.method:
        case "zeroshot":
            agent_kwargs = dict()
        case "cot":
            agent_kwargs = dict(
                cot_examples=COT_EXAMPLES,
            )
        case _:
            agent_kwargs = dict()
    return agent_kwargs


async def main(args: argparse.Namespace) -> None:
    """Main function to run the property extraction.

    Args:
        args: The arguments namespace.

    Returns:
        None.

    """
    logger.info(f"Loading LLM='{args.model_name}'")
    model_kwargs = get_model_kwargs(args)
    llm_chat: LLMChat = get_llm(args.server, args.model_name, model_kwargs=model_kwargs)
    llm_prompt: LLMPrompt = get_llm_prompt(args.method, args.model_name)
    default_inf_gen_config = InferenceGenerationConfig(
        max_tokens=args.inf_max_tokens,
        temperature=args.inf_temperature,
        top_k=args.inf_top_k,
        top_p=args.inf_top_p,
        repetition_penalty=args.inf_repetition_penalty,
        max_retries=args.inf_max_retries,
        wait_seconds=args.inf_wait_seconds,
    )

    for seed in args.inf_seed_list:
        logger.info(f"Running inference for method='{args.method}' with {seed=}")
        for split in args.split_list:
            dataset = load_data(
                args.dataset,
                split,
                from_local=args.from_local,
                exclude_aggregation_questions=args.exclude_aggregation_questions,
            )
            logger.info(f"Loading dataset='{args.dataset}' :: {split=}")
            df_qa_pairs = pd.DataFrame(dataset["qa_pairs"])
            df_text = pd.DataFrame(dataset["text"])

            # Construct agent for the data split
            agent_kwargs = get_agent_kwargs(args)
            agent: Agent = get_agent(
                args.method,
                text_corpus=df_text,
                llm_prompt=llm_prompt,
                agent_kwargs=agent_kwargs,
            )

            num_df_qa_pairs = len(df_qa_pairs)
            if args.inf_vllm_offline and args.method not in [
                "react",
                "act",
                "react->cot-sc",
                "cot-sc->react",
            ]:
                batch_size = num_df_qa_pairs
            else:
                if args.batch_number is not None:
                    assert args.batch_number >= 1, "Batch number must be >= 1"
                    assert args.batch_number <= math.ceil(
                        num_df_qa_pairs / args.batch_size
                    ), "Batch number must be <= ceil(num_df_qa_pairs / batch_size)"
                batch_size = args.batch_size

            for batch_number in range(1, math.ceil(num_df_qa_pairs / batch_size) + 1):
                lora_run_name = (
                    f"__lora_path={args.inf_vllm_lora_path.replace('/', '--')}"
                    if args.inf_vllm_lora_path
                    else ""
                )
                run_name = (
                    f"split={split}"
                    + f"__model_name={args.model_name.replace('/', '--')}"
                    + lora_run_name
                    + f"__bs={batch_size}"
                    + f"__bn={batch_number}"
                    + f"__seed={seed}"
                )
                pred_path = (
                    Path(args.output_dir) / "preds" / args.method / f"{run_name}.json"
                )

                # Skip if the batch number is not the one specified
                if (args.batch_number is not None) and (
                    batch_number != args.batch_number
                ):
                    continue
                # Skip if the output file already exists and --force is not set
                if pred_path.exists() and not args.force:
                    logger.info(
                        f"Skipping {pred_path} as it already exists. Use --force to overwrite."
                    )
                    continue

                # Get batch
                batch_start_idx = (batch_number - 1) * batch_size
                batch_end_idx = batch_start_idx + batch_size
                logger.info(
                    f"Getting predictions for questions [{batch_start_idx}, {batch_end_idx}) "
                    f"out of {num_df_qa_pairs}"
                )
                batch_df_qa_pairs = df_qa_pairs.iloc[batch_start_idx:batch_end_idx]

                # Run the method and get final responses for the batch
                # In zeroshot, fewshot, the LLM responds with the final answer in 1 turn only,
                # so they support batch async inference
                agent_interactions = None
                methods_with_batch_run = [
                    "zeroshot",
                    "zeroshot-sc",
                    "zeroshot-rag",
                    "fewshot",
                    "fewshot-sc",
                    "fewshot-rag",
                    "cot",
                    "cot-sc",
                    "cot-rag",
                ]
                match args.method:
                    case method if method in methods_with_batch_run:
                        questions: list[str] = batch_df_qa_pairs["question"].tolist()
                        inf_gen_config = default_inf_gen_config.model_copy(
                            update=dict(seed=seed), deep=True
                        )
                        responses: list[LLMChatResponse] = await agent.batch_run(
                            llm_chat,
                            questions,
                            inf_gen_config,
                        )
                        # NOTE: the agent interactions are just single Conversation objects containing the
                        # prompt for the self-consistency methods, we save the Conversation object from the
                        # last iteration
                        agent_interactions: list[Conversation] = (
                            agent.agent_interactions
                        )
                    case "react" | "act" | "react->cot-sc" | "cot-sc->react":
                        # Run all agents in parallel using asyncio.gather
                        responses: list[LLMChatResponse] = []
                        inf_gen_config = default_inf_gen_config.model_copy(
                            update=dict(seed=seed), deep=True
                        )
                        agents = [deepcopy(agent) for _ in range(batch_size)]
                        responses = await asyncio.gather(
                            *[
                                agent.run(
                                    llm_chat,
                                    qa_sample.question,
                                    inf_gen_config,
                                )
                                for agent, qa_sample in zip(
                                    agents, batch_df_qa_pairs.itertuples()
                                )
                            ]
                        )
                        agent_interactions: list[Conversation] = [
                            agent.agent_interactions for agent in agents
                        ]

                # Log the final answers for the batch
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving predictions to {pred_path}")

                # Save after each batch run
                unsaveable_agent_kwargs: list[str] = [
                    "cot_llm_prompt",
                    "react_llm_prompt",
                ]
                agent_kwargs_to_save = agent_kwargs.copy()
                for kw in unsaveable_agent_kwargs:
                    agent_kwargs_to_save.pop(kw, None)

                save_preds(
                    pred_path,
                    split,
                    inf_gen_config,
                    model_kwargs,
                    agent_kwargs_to_save,
                    args,
                    batch_number,
                    batch_df_qa_pairs,
                    responses,
                    interactions=agent_interactions
                    if not args.ignore_agent_interactions
                    else [],
                )


def save_preds(
    pred_path: Path,
    split: str,
    inf_gen_config: InferenceGenerationConfig,
    model_kwargs: dict,
    agent_kwargs: dict,
    args: argparse.Namespace,
    batch_number: int,
    batch_df_qa_pairs: pd.DataFrame,
    responses: list[LLMChatResponse],
    interactions: list[Conversation] | None = None,
) -> None:
    """Save the predictions to a JSON file.

    Args:
        pred_path: The path to the predictions file.
        split: The split of the data.
        inf_gen_config: The inference generation configuration.
        model_kwargs: The model kwargs.
        agent_kwargs: The agent kwargs.
        args: The arguments namespace.
        batch_number: The batch number.
        batch_df_qa_pairs: The batch of data.
        responses: The responses.
        interactions: The interactions.

    Returns:
        None.

    """
    preds = {}
    batch_size = len(batch_df_qa_pairs)

    for i, qa_sample in enumerate(batch_df_qa_pairs.itertuples()):
        uid = qa_sample.id

        # Get the appropriate prediction value and query info
        pred_value = responses[i].pred

        preds[uid] = {
            "true": qa_sample.answer,
            "pred": pred_value,
            "error": responses[i].error,
            "interaction": interactions[i].model_dump() if interactions else [],
            "metadata": {
                "model": args.model_name,
                "dataset": args.dataset,
                "split": split,
                "batch_size": batch_size,
                "batch_number": batch_number,
                "type": int(qa_sample.type),
                "difficulty": int(qa_sample.difficulty),
            },
            "inference_params": inf_gen_config.model_dump(),
            "model_kwargs": model_kwargs,
            "agent_kwargs": agent_kwargs,
            "usage": responses[i].usage,
        }

    with open(pred_path, "w") as f:
        json.dump(preds, f, indent=4)
        f.flush()


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser for property extraction with default dataset set to sci-llm-mini.

    Returns
    -------
        ArgumentParser with modified default dataset

    """
    parser = argparse.ArgumentParser(
        parents=[get_base_parser()], conflict_handler="resolve"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kilian-group/sci-llm-mini",
        help="Dataset to use",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    if args.method in ["zeroshot-rag", "fewshot-rag", "cot-rag"]:
        if args.retrieval_method in ["bm25", "dense"]:
            assert args.index_path is not None, (
                "index_path must be specified when retrieval_method is bm25 or dense"
            )
            assert args.corpus_path is not None, (
                "corpus_path must be specified when retrieval_method is bm25 or dense"
            )
            assert len(args.split_list) == 1, (
                "When retrieval_method is bm25 or dense, we can only evaluate one split at a time"
            )

    # NOTE: asyncio.run should only be called once in a single Python instance.
    # Thus, any high-level function containing awaits in its implementation
    # must be marked with the `async` keyword in the function definition.
    # See also: https://proxiesapi.com/articles/how-many-times-should-asyncio-run-be-called-python
    asyncio.run(main(args))
