RAG_CONTEXT_RELEVANCE_PROMT = (
    "You are an expert evaluator for RAG systems. Your task is to assess whether the context "
    "retrieved by the RAG system is relevant to the user question.\n"
    "A context is defined as relevant if it is related to the question even if it does not directly answer the question.\n\n"
    "[BEGIN DATA]\n"
    "************\n"
    "[Question]: {user_input}"
    "\n************\n"
    "[Retrieved Context]: {retreived_context}"
    "\n************\n"
    "[END DATA]\n\n"
    "Please rate the relevance with a range from {min_range_value} to {max_range_value} "
    "and provide and explanation in {language} for your rating.\n\n"
    "Return a structured format json string that follows this schema:\n"
    '{{rating": rate", "explanation": "explanation"}}'
)
