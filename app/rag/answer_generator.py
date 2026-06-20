from app.llm.llm_client import LLMClient
from app.rag.context_builder import build_context, extract_sources
from app.rag.prompt_builder import build_rag_prompt


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    llm_client: LLMClient,
    max_context_chars: int = 4000,
) -> dict:
    """
    Generate a source-grounded answer from a question and retrieved chunks.
    """
    if question is None or not question.strip():
        raise ValueError("question must be a non-empty string")

    if llm_client is None:
        raise ValueError("llm_client cannot be None")

    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than 0")

    normalized_question = question.strip()
    context = build_context(retrieved_chunks, max_chars=max_context_chars)
    sources = extract_sources(retrieved_chunks)
    prompt = build_rag_prompt(normalized_question, context)
    answer = llm_client.generate(prompt)

    return {
        "question": normalized_question,
        "answer": answer,
        "sources": sources,
        "context": context,
        "prompt": prompt,
    }