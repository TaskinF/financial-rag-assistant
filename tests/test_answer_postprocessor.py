from app.rag.answer_postprocessor import clean_answer


def test_clean_answer_returns_fallback_for_none_input():
    assert clean_answer(None) == "Bu bilgi verilen dok\u00fcman i\u00e7eri\u011finde bulunamad\u0131."


def test_clean_answer_removes_cevap_prefix():
    assert clean_answer("Cevap: Fonun yonetim ucreti vardir.") == "Fonun yonetim ucreti vardir."


def test_clean_answer_removes_answer_prefix():
    assert clean_answer("Answer: Fund value increased.") == "Fund value increased."


def test_clean_answer_removes_prefix_case_insensitively():
    assert clean_answer("cEvAp: Oran %12,5.") == "Oran %12,5."


def test_clean_answer_trims_extra_whitespace():
    assert clean_answer("   Cevap:   Net kar artti.   ") == "Net kar artti."


def test_clean_answer_normalizes_multiple_blank_lines():
    answer = "Cevap:\nIlk satir.\n\n\nIkinci satir."

    assert clean_answer(answer) == "Ilk satir.\n\nIkinci satir."


def test_clean_answer_returns_normal_answer_without_changes():
    assert clean_answer("Fonun toplam degeri 100 TL.") == "Fonun toplam degeri 100 TL."
