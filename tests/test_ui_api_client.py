from unittest.mock import Mock

import pytest
import requests

from ui.api_client import RAGAPIClient


def mock_response(payload: dict | None = None) -> Mock:
    response = Mock()
    response.json.return_value = payload or {}
    return response


def test_constructor_removes_trailing_slash_from_base_url():
    client = RAGAPIClient(base_url="http://127.0.0.1:8000/")

    assert client.base_url == "http://127.0.0.1:8000"


def test_constructor_rejects_empty_base_url():
    with pytest.raises(ValueError):
        RAGAPIClient(base_url="   ")


def test_health_check_calls_correct_endpoint():
    client = RAGAPIClient()
    response = mock_response({"status": "ok"})
    client.session.get = Mock(return_value=response)

    result = client.health_check()

    client.session.get.assert_called_once_with(
        "http://127.0.0.1:8000/health",
        timeout=300,
    )
    response.raise_for_status.assert_called_once_with()
    assert result == {"status": "ok"}


def test_list_documents_calls_correct_endpoint():
    client = RAGAPIClient()
    response = mock_response({"documents": [], "total": 0})
    client.session.get = Mock(return_value=response)

    result = client.list_documents()

    client.session.get.assert_called_once_with(
        "http://127.0.0.1:8000/documents",
        timeout=300,
    )
    response.raise_for_status.assert_called_once_with()
    assert result == {"documents": [], "total": 0}


def test_upload_document_sends_multipart_file_and_form_data():
    client = RAGAPIClient()
    response = mock_response({"document_id": "doc_a"})
    client.session.post = Mock(return_value=response)
    pdf_bytes = b"%PDF-1.4 test"

    result = client.upload_document(
        filename="report.pdf",
        file_bytes=pdf_bytes,
        document_id="doc_a",
        chunk_size=800,
        chunk_overlap=100,
        start_page=2,
        end_page=5,
    )

    client.session.post.assert_called_once_with(
        "http://127.0.0.1:8000/documents/upload",
        files={
            "file": (
                "report.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
        data={
            "document_id": "doc_a",
            "chunk_size": "800",
            "chunk_overlap": "100",
            "start_page": "2",
            "end_page": "5",
        },
        timeout=300,
    )
    response.raise_for_status.assert_called_once_with()
    assert result == {"document_id": "doc_a"}


def test_upload_document_rejects_empty_pdf_bytes():
    client = RAGAPIClient()

    with pytest.raises(ValueError):
        client.upload_document("report.pdf", b"")


def test_upload_document_rejects_non_pdf_filename():
    client = RAGAPIClient()

    with pytest.raises(ValueError):
        client.upload_document("report.txt", b"content")


def test_ask_document_sends_correct_url_and_json_body():
    client = RAGAPIClient()
    response = mock_response({"answer": "Fake answer"})
    client.session.post = Mock(return_value=response)

    result = client.ask_document(
        document_id="doc_a",
        question="What is the management fee?",
        top_k=5,
        answer_top_k=2,
        llm_provider="fake",
        llm_model="fake",
        max_context_chars=3000,
    )

    client.session.post.assert_called_once_with(
        "http://127.0.0.1:8000/documents/doc_a/ask",
        json={
            "question": "What is the management fee?",
            "top_k": 5,
            "answer_top_k": 2,
            "llm_provider": "fake",
            "llm_model": "fake",
            "max_context_chars": 3000,
        },
        timeout=300,
    )
    response.raise_for_status.assert_called_once_with()
    assert result == {"answer": "Fake answer"}


def test_ask_document_rejects_empty_document_id():
    client = RAGAPIClient()

    with pytest.raises(ValueError):
        client.ask_document("", "What is the management fee?")


def test_ask_document_rejects_empty_question():
    client = RAGAPIClient()

    with pytest.raises(ValueError):
        client.ask_document("doc_a", "   ")


def test_ask_document_rejects_answer_top_k_greater_than_top_k():
    client = RAGAPIClient()

    with pytest.raises(ValueError):
        client.ask_document(
            "doc_a",
            "What is the management fee?",
            top_k=2,
            answer_top_k=3,
        )


def test_http_error_response_is_raised():
    client = RAGAPIClient()
    response = mock_response()
    response.raise_for_status.side_effect = requests.HTTPError("server error")
    client.session.get = Mock(return_value=response)

    with pytest.raises(requests.HTTPError):
        client.health_check()
