import pytest

from app.rag.retrieval_evaluation import (
    compute_relevance_by_rank,
    evaluate_retrieval_results,
    hit_at_k,
    is_relevant_result,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_is_relevant_result_returns_true_for_expected_page_match():
    result = {"page_number": 3, "text": "irrelevant text"}

    assert is_relevant_result(result, expected_page=3, expected_keywords=["missing"])


def test_is_relevant_result_returns_true_for_expected_keyword_match():
    result = {"page_number": 1, "text": "Fon Yönetim Ücreti 7.125.856,54 0,1836 %"}

    assert is_relevant_result(
        result,
        expected_page=3,
        expected_keywords=["fon yönetim ücreti"],
    )


def test_is_relevant_result_returns_false_when_no_page_or_keyword_match():
    result = {"page_number": 1, "text": "Nakde donusum orani"}

    assert not is_relevant_result(
        result,
        expected_page=3,
        expected_keywords=["fon yönetim ücreti"],
    )


def test_compute_relevance_by_rank_returns_expected_binary_list():
    results = [
        {"page_number": 1, "text": "irrelevant"},
        {"page_number": 3, "text": "relevant by page"},
        {"page_number": 2, "text": "Fon Yönetim Ücreti"},
    ]

    relevance = compute_relevance_by_rank(
        results,
        expected_page=3,
        expected_keywords=["Fon Yönetim Ücreti"],
    )

    assert relevance == [0, 1, 1]


def test_hit_at_k_computes_correctly():
    assert hit_at_k([0, 0, 1], k=2) == 0.0
    assert hit_at_k([0, 1, 0], k=2) == 1.0


def test_precision_at_k_computes_correctly():
    assert precision_at_k([1, 0, 1], k=2) == 0.5


def test_recall_at_k_computes_correctly():
    assert recall_at_k([0, 0, 0], k=3) == 0.0
    assert recall_at_k([0, 1, 0], k=2) == 1.0


def test_mrr_at_k_computes_correctly():
    assert mrr_at_k([0, 1, 0], k=3) == 0.5
    assert mrr_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_at_k_is_between_zero_and_one_and_higher_for_better_ranking():
    better = ndcg_at_k([1, 0, 0], k=3)
    worse = ndcg_at_k([0, 0, 1], k=3)

    assert 0.0 <= better <= 1.0
    assert 0.0 <= worse <= 1.0
    assert better > worse


def test_metric_functions_raise_value_error_for_non_positive_k():
    with pytest.raises(ValueError):
        hit_at_k([1], k=0)

    with pytest.raises(ValueError):
        precision_at_k([1], k=0)

    with pytest.raises(ValueError):
        recall_at_k([1], k=0)

    with pytest.raises(ValueError):
        mrr_at_k([1], k=0)

    with pytest.raises(ValueError):
        ndcg_at_k([1], k=0)

    with pytest.raises(ValueError):
        evaluate_retrieval_results([], expected_page=1, expected_keywords=[], k=0)


def test_evaluate_retrieval_results_returns_pass_or_fail_status():
    passing_results = [
        {"page_number": 3, "text": "irrelevant"},
        {"page_number": 1, "text": "other"},
    ]
    failing_results = [
        {"page_number": 1, "text": "irrelevant"},
        {"page_number": 2, "text": "other"},
    ]

    passing_eval = evaluate_retrieval_results(
        passing_results,
        expected_page=3,
        expected_keywords=["missing"],
        k=2,
    )
    failing_eval = evaluate_retrieval_results(
        failing_results,
        expected_page=3,
        expected_keywords=["missing"],
        k=2,
    )

    assert passing_eval["status"] == "PASS"
    assert failing_eval["status"] == "FAIL"
