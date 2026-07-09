import re


FALLBACK_ANSWER = "Bu bilgi verilen dok\u00fcman i\u00e7eri\u011finde bulunamad\u0131."


def clean_answer(answer: str | None) -> str:
    """
    Clean raw LLM answer text for API and CLI output.
    """
    if answer is None:
        return FALLBACK_ANSWER

    cleaned_answer = answer.strip()
    cleaned_answer = re.sub(
        r"^(cevap:|answer:)\s*",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    )
    cleaned_answer = cleaned_answer.strip()
    cleaned_answer = cleaned_answer.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_answer = re.sub(r"\n\s*\n+", "\n\n", cleaned_answer)

    if not cleaned_answer:
        return FALLBACK_ANSWER

    return cleaned_answer
