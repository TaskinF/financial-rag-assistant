import pytest

from app.rag.prompt_builder import build_rag_prompt


def test_build_rag_prompt_raises_value_error_for_empty_question():
    with pytest.raises(ValueError):
        build_rag_prompt("", "sample context")


def test_build_rag_prompt_marks_empty_context_explicitly():
    prompt = build_rag_prompt("Fonun dagilimi nedir?", "")

    assert "[Context bos. Kullanilabilir dokuman icerigi bulunamadi.]" in prompt


def test_build_rag_prompt_includes_missing_information_response_rule():
    prompt = build_rag_prompt("Fonun dagilimi nedir?", "sample context")

    assert "Bu bilgi verilen dokuman iceriginde bulunamadi." in prompt


def test_build_rag_prompt_instructs_model_not_to_generate_sources():
    prompt = build_rag_prompt("Fonun dagilimi nedir?", "sample context")

    assert "Kaynaklar bolumu uretme." in prompt
    assert "Kaynaklar:" not in prompt


def test_build_rag_prompt_includes_financial_table_extraction_guidance():
    prompt = build_rag_prompt("Fonun dagilimi nedir?", "sample context")

    assert "tablo satirlari tek satira donusmus olabilir" in prompt
    assert "<kalem adi> <tutar> <oran>" in prompt


def test_build_rag_prompt_uses_answer_only_output_format():
    prompt = build_rag_prompt("Fonun dagilimi nedir?", "sample context")

    assert "[Cevap Formati]" in prompt
    assert "Cevap:" in prompt
