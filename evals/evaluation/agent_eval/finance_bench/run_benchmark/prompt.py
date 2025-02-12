LONG_CONTEXT_PROMPT_TEMPLATE="""\
You are an expert in financial anaylsis.
Answer the user question based on the following document:
{document}

User question: {question}
Now take a deep breath and think step by step to answer the question.
"""

RAG_PROMPT_TEMPLATE="""\
You are an expert in financial anaylsis.
Answer the user question based on the following document:
{document}

User question: {question}
"""