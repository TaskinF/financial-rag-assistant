import os

import requests
import streamlit as st

if __package__:
    from ui.api_client import RAGAPIClient
else:
    from api_client import RAGAPIClient


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_START_PAGE = 1
DEFAULT_END_PAGE = 3


def build_document_lookup(documents: list[dict]) -> dict[str, dict]:
    """Build a document_id lookup while preserving document records."""
    return {
        document["document_id"]: document
        for document in documents
    }


def format_document_label(document: dict) -> str:
    """Format a document for the selection widget."""
    filename = document.get("filename") or "unknown_file"
    document_id = document.get("document_id") or "unknown_document"
    return f"{filename} ({document_id})"


def format_source_score(score: float | int | None) -> str:
    """Format a retrieval score for display."""
    return f"{float(score):.4f}" if score is not None else "N/A"


def show_api_error(error: Exception) -> None:
    """Display an API error without exposing a traceback."""
    if isinstance(error, requests.exceptions.ConnectionError):
        st.error(
            "FastAPI backend'e ulaşılamıyor. Önce "
            "`uvicorn app.main:app --reload` komutunu çalıştırın."
        )
        return

    if isinstance(error, requests.exceptions.Timeout):
        st.error("API isteği zaman aşımına uğradı. Lütfen tekrar deneyin.")
        return

    if isinstance(error, requests.exceptions.HTTPError):
        detail = "API isteği başarısız oldu."
        if error.response is not None:
            try:
                detail = error.response.json().get("detail", detail)
            except ValueError:
                pass
        st.error(detail)
        return

    if isinstance(error, requests.exceptions.RequestException):
        st.error("FastAPI backend ile iletişim kurulurken bir hata oluştu.")
        return

    st.error("Beklenmeyen bir hata oluştu. Lütfen tekrar deneyin.")


def check_backend_health(client: RAGAPIClient) -> bool:
    """Return whether the FastAPI backend is reachable and healthy."""
    try:
        response = client.health_check()
        return response.get("status") == "ok"
    except requests.exceptions.RequestException as error:
        show_api_error(error)
        return False


def render_upload_section(
    client: RAGAPIClient,
    backend_available: bool,
) -> None:
    """Render PDF upload and indexing controls."""
    st.header("1. PDF Upload")

    upload_result = st.session_state.pop("upload_success", None)
    if upload_result:
        st.success(
            "Doküman başarıyla indexlendi: "
            f"{upload_result['filename']} "
            f"({upload_result['chunk_count']} chunk, "
            f"document_id={upload_result['document_id']})"
        )

    uploaded_file = st.file_uploader(
        "Financial PDF",
        type=["pdf"],
        disabled=not backend_available,
    )
    document_id = st.text_input(
        "Document ID (optional)",
        help="Boş bırakılırsa backend deterministik bir ID üretir.",
        disabled=not backend_available,
    )

    with st.expander("Advanced upload settings"):
        chunk_size = int(
            st.number_input(
                "Chunk size",
                min_value=1,
                value=1000,
                step=100,
                disabled=not backend_available,
            )
        )
        chunk_overlap = int(
            st.number_input(
                "Chunk overlap",
                min_value=0,
                value=200,
                step=50,
                disabled=not backend_available,
            )
        )
        limit_page_range = st.checkbox(
            "Limit indexing to page range",
            value=True,
            help="Varsayılan olarak yalnızca ilk 3 sayfa indexlenir.",
            disabled=not backend_available,
        )

        start_page: int | None = None
        end_page: int | None = None
        if limit_page_range:
            page_columns = st.columns(2)
            with page_columns[0]:
                start_page = int(
                    st.number_input(
                        "Start page",
                        min_value=1,
                        value=DEFAULT_START_PAGE,
                        step=1,
                        disabled=not backend_available,
                    )
                )
            with page_columns[1]:
                end_page = int(
                    st.number_input(
                        "End page",
                        min_value=1,
                        value=DEFAULT_END_PAGE,
                        step=1,
                        disabled=not backend_available,
                    )
                )

    if st.button(
        "Upload and Index",
        type="primary",
        disabled=not backend_available,
    ):
        if uploaded_file is None:
            st.warning("Lütfen bir PDF dosyası seçin.")
            return

        if chunk_size <= 0:
            st.warning("Chunk size sıfırdan büyük olmalıdır.")
            return

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            st.warning(
                "Chunk overlap sıfır veya daha büyük ve chunk size'dan küçük olmalıdır."
            )
            return

        if (
            start_page is not None
            and end_page is not None
            and start_page > end_page
        ):
            st.warning("Start page, end page değerinden büyük olamaz.")
            return

        try:
            with st.spinner("PDF yükleniyor ve indexleniyor..."):
                result = client.upload_document(
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    document_id=document_id.strip() or None,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    start_page=start_page,
                    end_page=end_page,
                )

            st.session_state["upload_success"] = result
            st.session_state["documents_refresh_needed"] = True
            st.session_state["preferred_document_id"] = result["document_id"]
            st.rerun()
        except ValueError as error:
            st.warning(str(error))
        except requests.exceptions.RequestException as error:
            show_api_error(error)


def render_document_selector(
    client: RAGAPIClient,
    backend_available: bool,
) -> dict | None:
    """Render indexed document selection and return the selected record."""
    st.header("2. Document Selection")

    if not backend_available:
        st.info("Backend bağlantısı kurulduğunda dokümanlar burada listelenecek.")
        return None

    try:
        response = client.list_documents()
        documents = response.get("documents", [])
        st.session_state["documents_refresh_needed"] = False
    except requests.exceptions.RequestException as error:
        show_api_error(error)
        return None

    if not documents:
        st.info("Henüz indexlenmiş bir doküman bulunmuyor.")
        return None

    documents_by_id = build_document_lookup(documents)
    document_ids = list(documents_by_id)
    preferred_document_id = st.session_state.pop("preferred_document_id", None)
    current_document_id = st.session_state.get("selected_document_id")

    if preferred_document_id in documents_by_id:
        st.session_state["selected_document_id"] = preferred_document_id
    elif current_document_id not in documents_by_id:
        st.session_state["selected_document_id"] = document_ids[0]

    selected_document_id = st.selectbox(
        "Indexed document",
        options=document_ids,
        format_func=lambda item: format_document_label(documents_by_id[item]),
        key="selected_document_id",
    )
    selected_document = documents_by_id[selected_document_id]

    metadata_columns = st.columns(4)
    metadata_columns[0].metric(
        "Chunks",
        selected_document.get("chunk_count", 0),
    )
    metadata_columns[1].metric(
        "Status",
        selected_document.get("status", "unknown"),
    )
    metadata_columns[2].markdown(
        f"**Document ID**  \n{selected_document_id}"
    )
    metadata_columns[3].markdown(
        f"**Indexed at**  \n{selected_document.get('indexed_at', '-')}"
    )
    st.caption(f"File: {selected_document.get('filename', '-')}")

    return selected_document


def render_question_section(
    client: RAGAPIClient,
    selected_document: dict | None,
    backend_available: bool,
) -> None:
    """Render document question and answer controls."""
    st.header("3. Ask a Question")

    question = st.text_area(
        "Question",
        value="Fonun yönetim ücreti nedir?",
        disabled=not backend_available,
    )

    with st.expander("Advanced answer settings"):
        top_k = int(
            st.number_input(
                "Top K",
                min_value=1,
                value=3,
                step=1,
                disabled=not backend_available,
            )
        )
        answer_top_k = int(
            st.number_input(
                "Answer Top K",
                min_value=1,
                value=2,
                step=1,
                disabled=not backend_available,
            )
        )
        llm_provider = st.selectbox(
            "LLM provider",
            options=["ollama", "fake"],
            disabled=not backend_available,
        )
        llm_model = st.text_input(
            "LLM model",
            value="gemma3:4b",
            disabled=not backend_available,
        )
        max_context_chars = int(
            st.number_input(
                "Max context characters",
                min_value=1,
                value=4000,
                step=500,
                disabled=not backend_available,
            )
        )

    if st.button(
        "Ask",
        type="primary",
        disabled=not backend_available,
    ):
        if selected_document is None:
            st.warning("Lütfen önce bir doküman seçin.")
            return

        if not question.strip():
            st.warning("Lütfen bir soru girin.")
            return

        if answer_top_k > top_k:
            st.warning("Answer Top K, Top K değerinden büyük olamaz.")
            return

        try:
            with st.spinner("Dokümanda yanıt aranıyor..."):
                result = client.ask_document(
                    document_id=selected_document["document_id"],
                    question=question,
                    top_k=top_k,
                    answer_top_k=answer_top_k,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    max_context_chars=max_context_chars,
                )
        except ValueError as error:
            st.warning(str(error))
            return
        except requests.exceptions.RequestException as error:
            show_api_error(error)
            return

        st.subheader("Answer")
        st.success(result.get("answer", "Yanıt alınamadı."))

        st.subheader("Sources")
        sources = result.get("sources", [])
        if not sources:
            st.info("Yanıt için kaynak bulunamadı.")

        for index, source in enumerate(sources, start=1):
            source_file = source.get("source_file") or "unknown_source"
            page_number = source.get("page_number")
            with st.expander(
                f"Source {index}: {source_file}, page {page_number}"
            ):
                st.write(f"source_file: {source_file}")
                st.write(f"page_number: {page_number}")
                st.write(f"chunk_id: {source.get('chunk_id')}")
                st.write(f"score: {format_source_score(source.get('score'))}")

        st.subheader("Retrieval Metadata")
        metadata_columns = st.columns(4)
        metadata_columns[0].metric(
            "Retrieved",
            result.get("retrieved_count", 0),
        )
        metadata_columns[1].metric(
            "Answer Context",
            result.get("answer_context_count", 0),
        )
        metadata_columns[2].metric(
            "Provider",
            result.get("llm_provider", llm_provider),
        )
        metadata_columns[3].metric(
            "Model",
            result.get("llm_model", llm_model),
        )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Financial Document RAG Assistant",
        page_icon="📄",
        layout="wide",
    )

    st.title("Financial Document RAG Assistant")
    st.write(
        "Finansal PDF raporlarını yükleme, seçme ve kaynaklı soru-cevap."
    )

    api_base_url = os.getenv("RAG_API_BASE_URL", DEFAULT_API_BASE_URL)
    st.sidebar.subheader("Backend")
    st.sidebar.code(api_base_url)

    try:
        client = RAGAPIClient(base_url=api_base_url)
    except ValueError as error:
        st.error(f"API URL ayarı geçersiz: {error}")
        return

    backend_available = check_backend_health(client)
    if backend_available:
        st.sidebar.success("FastAPI connected")
    else:
        st.sidebar.error("FastAPI unavailable")

    render_upload_section(client, backend_available)
    st.divider()
    selected_document = render_document_selector(client, backend_available)
    st.divider()
    render_question_section(
        client,
        selected_document,
        backend_available,
    )


if __name__ == "__main__":
    main()
