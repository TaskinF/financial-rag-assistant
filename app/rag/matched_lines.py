import re


STOPWORDS = {
    "nedir",
    "ne",
    "kaç",
    "fonun",
    "ve",
    "ile",
    "mi",
    "mı",
    "mu",
    "mü",
    "olarak",
}

FINANCIAL_KEYWORDS = {
    "ücret",
    "yönetim",
    "portföy",
    "toplam",
    "değer",
    "oran",
    "hisse",
    "fon",
    "ucret",
    "yonetim",
    "portfoy",
    "deger",
}


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace to single spaces.
    """
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase alphanumeric words.
    """
    return re.findall(r"\b[\wçğıöşüÇĞİÖŞÜ]+\b", text.lower())


def _meaningful_question_tokens(question: str) -> set[str]:
    """
    Extract non-stopword question tokens.
    """
    return {
        token
        for token in _tokenize(question)
        if token not in STOPWORDS and len(token) > 1
    }


def _split_candidate_lines(text: str) -> list[str]:
    """
    Split chunk text into candidate lines using newlines and punctuation hints.
    """
    raw_parts = re.split(r"[\r\n]+", text)
    candidates: list[str] = []

    for part in raw_parts:
        normalized_part = part.strip()
        if not normalized_part:
            continue

        candidates.append(normalized_part)

        # Table-heavy PDFs may collapse rows into long single lines. Also consider
        # punctuation-based fragments as extra candidates.
        if len(normalized_part) > 120:
            fragments = re.split(r"(?<=[.;:])\s+", normalized_part)
            for fragment in fragments:
                fragment = fragment.strip()
                if fragment:
                    candidates.append(fragment)

    return candidates


def _has_financial_value(line: str) -> bool:
    """
    Detect numeric or percentage-like patterns in a line.
    """
    return bool(
        re.search(r"\d", line)
        or re.search(r"%", line)
        or re.search(r"\d[\d\.,]*", line)
    )


def _score_line(question_tokens: set[str], line: str) -> float:
    """
    Compute a simple lexical match score for a candidate line.
    """
    normalized_line = _normalize_text(line)
    line_tokens = set(_tokenize(normalized_line))

    score = 0.0

    token_matches = question_tokens.intersection(line_tokens)
    score += float(len(token_matches)) * 2.0

    financial_matches = question_tokens.intersection(FINANCIAL_KEYWORDS).intersection(
        line_tokens
    )
    score += float(len(financial_matches)) * 3.0

    if line_tokens.intersection(FINANCIAL_KEYWORDS):
        score += 1.5

    if _has_financial_value(normalized_line):
        score += 1.0

    if token_matches and _has_financial_value(normalized_line):
        score += 1.0

    return score


def extract_matched_lines(
    question: str,
    retrieved_chunks: list[dict],
    max_lines: int = 5,
) -> list[dict]:
    """
    Extract the most relevant financial lines from retrieved chunks for a question.
    """
    if question is None or not question.strip():
        raise ValueError("question must be a non-empty string")

    if max_lines <= 0:
        raise ValueError("max_lines must be greater than 0")

    if not retrieved_chunks:
        return []

    question_tokens = _meaningful_question_tokens(question)
    matches: list[dict] = []
    seen_lines: set[str] = set()

    for chunk in retrieved_chunks:
        chunk_text = str(chunk.get("text", ""))
        source_file = chunk.get("source_file")
        page_number = chunk.get("page_number")
        chunk_id = chunk.get("chunk_id")
        score = chunk.get("score")

        for candidate in _split_candidate_lines(chunk_text):
            normalized_line = _normalize_text(candidate)
            if not normalized_line:
                continue

            dedupe_key = normalized_line.lower()
            if dedupe_key in seen_lines:
                continue

            match_score = _score_line(question_tokens, normalized_line)
            if match_score <= 0:
                continue

            seen_lines.add(dedupe_key)
            matches.append(
                {
                    "line": normalized_line,
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "score": score,
                    "match_score": match_score,
                }
            )

    matches.sort(key=lambda item: item["match_score"], reverse=True)
    return matches[:max_lines]
