def build_rag_prompt(question: str, context: str) -> str:
    """
    Build a source-grounded RAG prompt from a user question and retrieved context.
    """
    if question is None or not question.strip():
        raise ValueError("question must be a non-empty string")

    normalized_question = question.strip()
    normalized_context = context.strip() if context is not None else ""

    if not normalized_context:
        normalized_context = "[Context is empty]"

    return f"""[Sistem / Rol]
Sen finansal dokümanları analiz eden dikkatli bir asistansın.
Yalnızca verilen context'i kullanarak Türkçe cevap üret.

[Context]
{normalized_context}

[Kullanıcı Sorusu]
{normalized_question}

[Cevap Talimatları]
- Sadece verilen context'i kullan.
- Context'te cevap yoksa aynen şu cümleyi yaz:
  "Bu bilgi verilen doküman içeriğinde bulunamadı."
- Finansal değerleri, oranları ve tarihleri değiştirme.
- Tahmin yapma.
- Cevabını açık, kısa ve profesyonel Türkçe ile yaz.
- Cevabın sonunda kullanılan kaynakları belirt.
- Kaynak formatını aynen şöyle kullan:
  Kaynak: <file>, Sayfa <page>
"""