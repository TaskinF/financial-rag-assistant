import math


def is_relevant_result(
    result: dict,
    expected_page: int,
    expected_keywords: list[str],
) -> bool:
    """
    Determine whether a retrieved result is relevant for the evaluation target.
    """
    page_number = result.get("page_number")
    if page_number == expected_page:
        return True

    text = str(result.get("text") or result.get("content") or "")
    normalized_text = text.lower()

    for keyword in expected_keywords:
        if keyword.lower() in normalized_text:
            return True

    return False


def compute_relevance_by_rank(
    results: list[dict],
    expected_page: int,
    expected_keywords: list[str],
) -> list[int]:
    """
    Convert ranked retrieval results into a binary relevance list.
    """
    return [
        1 if is_relevant_result(result, expected_page, expected_keywords) else 0
        for result in results
    ]


def hit_at_k(relevance: list[int], k: int) -> float:
    """
    Return 1.0 if at least one relevant item exists in the top-k results.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    return 1.0 if any(relevance[:k]) else 0.0


def precision_at_k(relevance: list[int], k: int) -> float:
    """
    Compute precision at k for binary relevance.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    return sum(relevance[:k]) / float(k)


def recall_at_k(relevance: list[int], k: int) -> float:
    """
    Compute recall at k assuming a single expected relevant target.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    return 1.0 if any(relevance[:k]) else 0.0


def mrr_at_k(relevance: list[int], k: int) -> float:
    """
    Compute mean reciprocal rank at k for a single query.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    for index, value in enumerate(relevance[:k], start=1):
        if value:
            return 1.0 / float(index)

    return 0.0


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """
    Compute nDCG at k using binary relevance.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    dcg = 0.0
    for index, value in enumerate(relevance[:k], start=1):
        if value:
            dcg += 1.0 / math.log2(index + 1)

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for index, value in enumerate(ideal_relevance[:k], start=1):
        if value:
            idcg += 1.0 / math.log2(index + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval_results(
    results: list[dict],
    expected_page: int,
    expected_keywords: list[str],
    k: int,
) -> dict:
    """
    Evaluate ranked retrieval results with standard ranking metrics.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevance = compute_relevance_by_rank(results, expected_page, expected_keywords)
    page_hit = any(result.get("page_number") == expected_page for result in results[:k])
    keyword_hit = any(
        is_relevant_result(
            {"text": result.get("text") or result.get("content") or ""},
            expected_page=-1,
            expected_keywords=expected_keywords,
        )
        for result in results[:k]
    )
    hit = hit_at_k(relevance, k)

    return {
        "relevance_by_rank": relevance,
        "hit_at_k": hit,
        "precision_at_k": precision_at_k(relevance, k),
        "recall_at_k": recall_at_k(relevance, k),
        "mrr_at_k": mrr_at_k(relevance, k),
        "ndcg_at_k": ndcg_at_k(relevance, k),
        "page_hit": page_hit,
        "keyword_hit": keyword_hit,
        "status": "PASS" if hit == 1.0 else "FAIL",
    }
