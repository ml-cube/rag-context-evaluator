"""Script to evaluate Context Relevancy Guard on "wiki_qa-train" benchmark dataset.
* https://huggingface.co/datasets/microsoft/wiki_qa

Model: gpt-4o-mini
Guard Results
              precision    recall  f1-score   support

    relevant       0.70      0.86      0.77        93
   unrelated       0.85      0.68      0.76       107

    accuracy                           0.77       200
   macro avg       0.78      0.77      0.76       200
weighted avg       0.78      0.77      0.76       200

Latency
count    200.000000
mean       2.812122
std        1.753805
min        1.067620
25%        1.708051
50%        2.248962
75%        3.321251
max       14.102804
Name: guard_latency_gpt-4o-mini, dtype: float64
median latency
2.2489616039965767

Model: gpt-4-turbo
Guard Results
              precision    recall  f1-score   support

    relevant       0.64      0.90      0.75        93
   unrelated       0.87      0.56      0.68       107

    accuracy                           0.72       200
   macro avg       0.76      0.73      0.72       200
weighted avg       0.76      0.72      0.71       200

Latency
count    200.000000
mean       8.561413
std        6.425799
min        1.624563
25%        3.957226
50%        5.979291
75%       11.579224
max       34.342637
Name: guard_latency_gpt-4-turbo, dtype: float64
median latency
5.979290812509134
"""

import os
import time
from getpass import getpass
from typing import List, Tuple

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import openai
import pandas as pd
from sklearn.metrics import classification_report

from guardrails import Guard
from main import RagContextRelevancePrompt, LlmRagRelevanceEvaluator

from sklearn.utils import shuffle
import pandas as pd
import polars as pl
from phoenix.evals import download_benchmark_dataset
from typing import cast
from validator.models import RagRelevanceResponse
from tqdm import tqdm

RANDOM_STATE = 119
MODELS = ["gpt-4o-mini"]
N_EVAL_SAMPLE_SIZE = 5
SAVE_RESULTS_PATH = "context_relevance_guard_results.csv"


def evaluate_guard_on_dataset(
    test_dataset: pl.DataFrame, guard: Guard, model: str
) -> Tuple[List[float], List[bool]]:
    """Evaluate guard on benchmark dataset.

    :param test_dataset: Dataframe of test examples.
    :param guard: Guard we want to evaluate.

    :return: Tuple where the first lists contains latency, and the second list contains a boolean indicating whether the guard passed.
    """
    latency_measurements = []
    guard_passed = []
    for row in tqdm(test_dataset.iter_rows(named=True), "Scanning dataset"):
        start_time = time.perf_counter()
        response = guard(
            llm_api=openai.chat.completions.create,
            prompt=row["query_text"],
            model=model,
            max_tokens=1024,
            temperature=0.5,
            metadata={
                "user_input": row["query_text"],
                "retrieved_context": row["document_text"],
            },
        )
        latency_measurements.append(time.perf_counter() - start_time)
        guard_passed.append(response.validation_passed)
    return latency_measurements, guard_passed


if __name__ == "__main__":
    if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
        openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Columns: Index(['query_id', 'query_text', 'document_title', 'document_text', 'document_text_with_emphasis', 'relevant']

    test_dataset = download_benchmark_dataset(
        task="binary-relevance-classification", dataset_name="wiki_qa-train"
    )

    test_dataset = shuffle(test_dataset, random_state=RANDOM_STATE)

    test_dataset = test_dataset[:N_EVAL_SAMPLE_SIZE]
    test_dataset = pl.DataFrame(test_dataset)

    relevant_str = "relevant_str"

    test_dataset = test_dataset.with_columns(
        [
            pl.when(pl.col("relevant"))
            .then(pl.lit("relevant"))
            .otherwise(pl.lit("unrelated"))
            .alias(relevant_str),
        ]
    )
    y_true = test_dataset.select(pl.col(relevant_str)).to_series().to_list()

    for model in MODELS:
        guard = Guard().use(
            LlmRagRelevanceEvaluator(
                eval_llm_prompt_generator=RagContextRelevancePrompt(
                    prompt_name="context_relevance_judge"
                ),
                pass_threshold=1,
                llm_callable=model,
                response_format=RagRelevanceResponse,
                on_fail="noop",
                on="prompt",
            )
        )

        latency_measurements, guard_passed = evaluate_guard_on_dataset(
            test_dataset=test_dataset, guard=guard, model=model
        )
        guard_passed_column = f"guard_passed_{model}"
        latency_column = f"guard_latency_{model}"

        test_dataset = test_dataset.with_columns(
            pl.Series(guard_passed_column, guard_passed),
            pl.Series(latency_column, guard_passed),
        )

        print(f"\nModel: {model}")
        print("Guard Results")
        # Calculate precision, recall and f1-score for when the Guard fails (e.g. flags an irrelevant answer)

        guard_passed_column_str = f"guard_passed_{model}"

        test_dataset = test_dataset.with_columns(
            [
                pl.when(pl.col(guard_passed_column))
                .then(pl.lit("relevant"))
                .otherwise(pl.lit("unrelated"))
                .alias(guard_passed_column_str),
            ]
        )

        y_pred = (
            test_dataset.select(pl.col(guard_passed_column_str)).to_series().to_list()
        )

        print(classification_report(y_true, y_pred))
        print("Latency")
        print(test_dataset[f"guard_latency_{model}"].describe())
        print("median latency")
        print(test_dataset[f"guard_latency_{model}"].median())

    if SAVE_RESULTS_PATH:
        test_dataset.write_csv(SAVE_RESULTS_PATH)
