from app.text_cleaner import clean_text


def test_clean_text_returns_empty_string_for_none_input():
    assert clean_text(None) == ""


def test_clean_text_strips_leading_and_trailing_spaces():
    text = "   Revenue increased in Q1.   "

    assert clean_text(text) == "Revenue increased in Q1."


def test_clean_text_reduces_multiple_spaces_to_single_space():
    text = "Net   income    rose significantly."

    assert clean_text(text) == "Net income rose significantly."


def test_clean_text_normalizes_tabs_and_newlines():
    text = "Revenue\t\tGrowth\n\n  Margin"

    assert clean_text(text) == "Revenue Growth\nMargin"


def test_clean_text_preserves_financial_symbols_numbers_and_percentages():
    text = "Revenue was $1,200.50 on 2024-12-31 with 12.5% growth."

    assert clean_text(text) == "Revenue was $1,200.50 on 2024-12-31 with 12.5% growth."