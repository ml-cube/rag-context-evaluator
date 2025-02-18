from typing import Any, Callable, Dict, Optional
from guardrails import Guard
import openai
import logging
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from pydantic import BaseModel
import litellm
from validator.prompts.prompts import (
    Ml3RagContextEvalBasePrompt,
)
from validator.prompts.prompt_hub import RAG_CONTEXT_RELEVANCE_PROMPT
from validator.models import RagRatingResponse
from validator.prompts.prompts import RagContextRelevancePrompt
from langchain import chat_models
from typing import cast

logger = logging.getLogger(__name__)


@register_validator(name="mlcube/rag_context_evaluator", data_type="string")
class LlmRagContextEvaluator(Validator):
    def __init__(
        self,
        rag_context_eval_prompt_generator: Ml3RagContextEvalBasePrompt,
        pass_threshold: int,
        model_name: str,
        response_format: type[BaseModel] = RagRatingResponse,
        on_fail: Optional[Callable] = "noop",  # type: ignore
        default_min: int = 0,
        default_max: int = 1,
        **kwargs,
    ):
        super().__init__(
            on_fail,
            eval_llm_prompt_generator=rag_context_eval_prompt_generator,
            llm_callable=model_name,
            **kwargs,
        )
        self._llm_evaluator_prompt_generator = rag_context_eval_prompt_generator
        self._model_name = model_name
        self._response_format = response_format
        self._pass_threshold = pass_threshold
        self._default_min = default_min
        self._default_max = default_max

    def get_llm_response(self, prompt: str) -> RagRatingResponse:
        """Gets the response from the LLM.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        # 1. Create messages
        messages = [{"content": prompt, "role": "user"}]

        # 2. Get the model provider given the model name
        _model, model_provider, *_rest = litellm.get_llm_provider(self._model_name)  # type: ignore

        # 3. Inizialize the chat model with the
        model = chat_models.init_chat_model(
            model=_model,
            model_provider=model_provider,
        ).with_structured_output(self._response_format)

        # 4. Get LLM response
        try:
            response = model.invoke(messages)
        except Exception:
            logger.exception("Failed to get response from LLM.")
            response = self._response_format.model_validate(
                {
                    "rating": self._default_min,
                    "explanation": "The model failed to generate a response.",
                }
            )

        # 3. Return the response
        return cast(RagRatingResponse, response)

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        """
        Validates is based on the relevance of the reference text to the original question.

        **Key Properties**
        | Property                      | Description                       |
        | ----------------------------- | --------------------------------- |
        | Name for `format` attribute   | `mlcube/rag_context_evaluator`    |
        | Supported data types          | `string`                          |
        | Programmatic fix              | N/A                               |


        Args:
            value (Any): The value to validate. It must contain 'original_prompt' and 'reference_text' keys.
            metadata (Dict): The metadata for the validation.
                user_input: Required key. User query passed into RAG LLM.
                retrieved_context: Required key. Context used by RAG LLM.
                llm_response: Optional key. By default, the gaurded LLM will make the RAG LLM call, which corresponds
                    to the `value`. If the user calls the guard with on="prompt", then the original RAG LLM response
                    needs to be passed into the guard as metadata for the LLM judge to evaluate.
                min_range_value: Optional key. The minimum value for the rating. Default is 1.
                max_range_value: Optional key. The maximum value for the rating. Default is 5.

        Returns:
            ValidationResult: The result of the validation. It can be a PassResult if the reference
                              text is relevant to the original question, or a FailResult otherwise.
        """

        # 1. Get the question and arg from the value
        user_input = metadata.get("user_input", None)
        if user_input is None:
            raise RuntimeError(
                "user_input missing from value. Please provide the original prompt."
            )

        retrieved_context = metadata.get("retrieved_context", None)
        if retrieved_context is None:
            raise RuntimeError(
                "retreived_context missing from value. Please provide the retreived_context."
            )

        min_range_value = int(metadata.get("min_range_value", self._default_min))
        max_range_value = int(metadata.get("max_range_value", self._default_max))

        # 2. Setup the prompt
        prompt = self._llm_evaluator_prompt_generator.generate_prompt(
            user_input=user_input,
            retrieved_context=retrieved_context,
            min_range_value=min_range_value,
            max_range_value=max_range_value,
        )
        logging.debug(f"evaluator prompt: {prompt}")

        # 3. Get the LLM response
        llm_response = self.get_llm_response(prompt)
        logging.debug(f"llm evaluator response: {llm_response}")

        # 4. Check the LLM response and return the result
        if llm_response.rating < self._pass_threshold:
            return FailResult(
                error_message=f"""Validation failed. The LLM Judge assigned an evaluation score: {llm_response.rating} below the provided threshold: {self._pass_threshold}". \nEvaluator prompt: {prompt}"""
            )

        return PassResult()


if __name__ == "__main__":
    from validator.prompts.prompt_hub import RAG_CONTEXT_RELEVANCE_PROMPT

    print(RAG_CONTEXT_RELEVANCE_PROMPT)
    guard = Guard().use(
        LlmRagContextEvaluator(
            rag_context_eval_prompt_generator=RagContextRelevancePrompt(
                prompt_name="context_relevance_judge"
            ),
            pass_threshold=1,
            model_name="gpt-4o-mini",
            response_format=RagRatingResponse,
            on_fail="noop",  # type: ignore
            on="prompt",
        )
    )
    response = guard(
        llm_api=openai.chat.completions.create,
        prompt="Che tempo fa oggi a Milano?",
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0,
        metadata={
            "user_input": "What's the weather in Milan, today?",
            "retrieved_context": "Milan, what a beautiful day. Sunny and warm.",
        },
    )
    print(response)
