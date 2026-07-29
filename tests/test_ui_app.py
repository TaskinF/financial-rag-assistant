import importlib

from ui import app as ui_app


class HealthyAPIClient:
    def health_check(self) -> dict:
        return {"status": "ok"}


def test_ui_app_module_can_be_imported():
    module = importlib.import_module("ui.app")

    assert callable(module.main)


def test_default_api_base_url_is_correct():
    assert ui_app.DEFAULT_API_BASE_URL == "http://127.0.0.1:8000"


def test_default_upload_page_range_is_first_three_pages():
    assert ui_app.DEFAULT_START_PAGE == 1
    assert ui_app.DEFAULT_END_PAGE == 3


def test_backend_health_check_works_with_mock_client():
    assert ui_app.check_backend_health(HealthyAPIClient()) is True


def test_document_label_contains_filename_and_document_id():
    label = ui_app.format_document_label(
        {
            "filename": "fund_report.pdf",
            "document_id": "fund_report_1234",
        }
    )

    assert label == "fund_report.pdf (fund_report_1234)"


def test_source_score_is_formatted_to_four_decimal_places():
    assert ui_app.format_source_score(0.912345) == "0.9123"


def test_empty_document_lookup_returns_empty_dict():
    assert ui_app.build_document_lookup([]) == {}
