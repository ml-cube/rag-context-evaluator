"""Script to evaluate Context Relevancy Guard on "wiki_qa-train" benchmark dataset.
* https://huggingface.co/datasets/microsoft/wiki_qa

"""

import os
import time
from typing import List, Tuple

import openai
from sklearn.metrics import classification_report

from guardrails import Guard
from validator.prompts.prompts import (
    Ml3RagContextEvalBasePrompt,
    RagContextUsefulnessPrompt,
)


from sklearn.utils import shuffle
import polars as pl
from typing import cast
from validator.models import RagRatingResponse
from main import LlmRagContextEvaluator
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
RANDOM_STATE = 119
MAX_TOKENS = 1024
TEMPERATURE = 0.0
MODELS = ["gpt-4o-mini"]
N_EVAL_SAMPLE_SIZE = 200
RELEVANCE_RESULTS_PATH = "context_relevance_guard_results.csv"
USEFULNESS_RESULTS_PATH = "context_usefulness_guard_results.csv"

EVALUATION_DATASET_PATH = os.path.join("evaluation_dataset", "wiki_qa_train.parquet")
RELEVANT_COLUMN_STR = "relevant_str"
GUARD_PASSED_COLUMN_STR = "guard_passed_{model}"
PASS_THRESHOLD = 1


def evaluate_guard_on_dataset(
    test_dataset: pl.DataFrame, guard: Guard, model_name: str
) -> Tuple[List[float], List[bool]]:
    """Evaluate guard on benchmark dataset.

    Args:
        test_dataset (Dataframe): Dataframe of test examples.
        guard (Guard): Guard we want to evaluate.
        model_name (str): the name of the llm model

    Returns:
        Tuple[List[float], List[bool]]: Tuple where the first lists contains latency, and the second list contains a boolean indicating whether the guard passed.
    """

    latency_measurements = []
    guard_passed = []
    for row in tqdm(test_dataset.iter_rows(named=True), "Scanning dataset"):
        start_time = time.perf_counter()
        response = guard(
            llm_api=openai.chat.completions.create,
            prompt=row["query_text"],
            model=model_name,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            metadata={
                "user_input": row["query_text"],
                "retrieved_context": row["document_text"],
            },
        )
        latency_measurements.append(time.perf_counter() - start_time)
        guard_passed.append(response.validation_passed)
    return latency_measurements, guard_passed


def get_eval_dataset(path: str) -> pl.DataFrame:
    """Get evaluation dataset. The dataset contains the folloqing columns:
        - query_id: The id of the user query.
        - query_text: The user query.
        - document_title: The title of the retrieved context.
        - document_text: The retrieved context.
        - document_text_with_emphasis: The retrieved context with emphasis on the relevant part.
        - relevant: Boolean indicating whether the context is relevant to the user query.

    Args:
        path (str): Path to dataset.

    Returns:
        pl.DataFrame: Evaluation dataset.
    """
    test_dataset = pl.read_parquet(path)
    test_dataset = cast(pl.DataFrame, shuffle(test_dataset, random_state=RANDOM_STATE))
    test_dataset = test_dataset[:N_EVAL_SAMPLE_SIZE]
    test_dataset = test_dataset.with_columns(
        [
            pl.when(pl.col("relevant"))
            .then(pl.lit("relevant"))
            .otherwise(pl.lit("unrelated"))
            .alias(RELEVANT_COLUMN_STR),
        ]
    )
    return test_dataset


def llm_rag_context_evaluation(
    prompt_generator: type[Ml3RagContextEvalBasePrompt],
    save_results_path: str | None = None,
) -> None:
    """Evaluate the Context Relevancy Guard on the `wiki_qa-train` benchmark dataset.

    Args:
        prompt_generator (type[Ml3RagContextEvalBasePrompt]): The prompt generator for the guard.
        save_results_path (str | None, optional): Path to save the results. Defaults to None.
    """

    # 1. Load the evaluation dataset
    test_dataset = get_eval_dataset(EVALUATION_DATASET_PATH)

    # 2. Pre-compute the ground truth
    y_true = test_dataset.select(pl.col(RELEVANT_COLUMN_STR)).to_series().to_list()

    # 3. Evaluate the guard on the dataset for each model
    for model in MODELS:
        guard = Guard().use(
            LlmRagContextEvaluator(
                rag_context_eval_prompt_generator=prompt_generator(
                    prompt_name="context_llm_judge"
                ),
                pass_threshold=PASS_THRESHOLD,
                model_name=model,
                response_format=RagRatingResponse,
                on_fail="noop",
                on="prompt",
            )
        )

        # Evaluate the guard on the dataset
        latency_measurements, guard_passed = evaluate_guard_on_dataset(
            test_dataset=test_dataset, guard=guard, model_name=model
        )

        guard_passed_column = f"guard_passed_{model}"
        latency_column = f"guard_latency_{model}"

        test_dataset = test_dataset.with_columns(
            pl.Series(guard_passed_column, guard_passed),
            pl.Series(latency_column, latency_measurements),
        )

        logger.info(f"\nModel: {model}")
        logger.info("Guard Results")

        guard_passed_column_str = GUARD_PASSED_COLUMN_STR.format(model=model)

        # Add a column to the dataset indicating whether the guard passed
        test_dataset = test_dataset.with_columns(
            [
                pl.when(pl.col(guard_passed_column))
                .then(pl.lit("relevant"))
                .otherwise(pl.lit("unrelated"))
                .alias(guard_passed_column_str),
            ]
        )

        # Get the predictions
        y_pred = (
            test_dataset.select(pl.col(guard_passed_column_str)).to_series().to_list()
        )

        # Compute the classification report

        logger.info(classification_report(y_true, y_pred))
        logger.info("Latency")
        logger.info(test_dataset[f"guard_latency_{model}"].describe())
        logger.info("median latency")
        logger.info(test_dataset[f"guard_latency_{model}"].median())

    if save_results_path:
        test_dataset.write_csv(save_results_path)


if __name__ == "__main__":
    llm_rag_context_evaluation(RagContextUsefulnessPrompt, USEFULNESS_RESULTS_PATH)
