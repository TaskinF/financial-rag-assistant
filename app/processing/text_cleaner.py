import re


def clean_text(text: str | None) -> str:
    """
    Clean raw document text by normalizing whitespace while preserving
    financially meaningful characters such as numbers, symbols, and dates.

    Args:
        text: Raw text input.

    Returns:
        Cleaned text as a string. Returns an empty string if input is None.
    """
    if text is None:
        return ""

    cleaned_text = text.replace("\t", " ")
    cleaned_text = re.sub(r"\s*\n\s*", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)
    cleaned_text = re.sub(r"[ ]{2,}", " ", cleaned_text)

    return cleaned_text.strip()