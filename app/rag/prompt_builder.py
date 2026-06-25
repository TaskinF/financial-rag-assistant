def build_rag_prompt(question: str, context: str) -> str:
    """
    Build a stricter source-grounded RAG prompt for financial question answering.
    """
    if question is None or not question.strip():
        raise ValueError("question must be a non-empty string")

    normalized_question = question.strip()
    normalized_context = context.strip() if context is not None else ""

    if not normalized_context:
        normalized_context = "[Context bos. Kullanilabilir dokuman icerigi bulunamadi.]"

    return f"""[Sistem / Rol]
Sen finansal dokumanlari analiz eden dikkatli bir asistansin.
Yalnizca verilen context'e dayanarak Turkce cevap uret.

[Context]
{normalized_context}

[Kullanici Sorusu]
{normalized_question}

[Cevap Talimatlari]
- Sadece verilen context'i kullan.
- Context disinda hicbir bilgi kullanma.
- Dis bilgi ekleme.
- Tahmin yapma.
- Yatirim tavsiyesi verme.
- Finansal degerleri, oranlari ve tarihleri context'te gectigi gibi koru.
- Dosya, sayfa veya kaynak bilgisi uydurma ya da uretme.
- PDF extraction nedeniyle tablo satirlari tek satira donusmus olabilir.
- Bir finansal satir su yapida gorunebilir: <kalem adi> <tutar> <oran>
- Eger soru bir finansal kalemle ilgiliyse ve context icinde ayni veya cok benzer kalem adi geciyorsa, yanindaki tutar ve orani cevapta kullan.
- Context icinde cevap acikca bulunuyorsa kesinlikle "Bu bilgi verilen dokuman iceriginde bulunamadi." deme.
- Eger cevap gercekten context icinde yoksa tam olarak su ifadeyi dondur:
  "Bu bilgi verilen dokuman iceriginde bulunamadi."
- Kaynaklar bolumu uretme.
- Cevabi kisa, dogrudan ve profesyonel Turkce ile yaz.

[Cevap Formati]
Cevap:
<kisa ve dogrudan cevap>
"""
