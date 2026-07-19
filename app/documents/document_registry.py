from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def create_document_id(filename: str) -> str:
    """
    Create a deterministic document identifier from a filename.
    """
    if filename is None or not filename.strip():
        raise ValueError("filename cannot be empty")

    normalized_filename = filename.strip().lower()
    stem = Path(normalized_filename).stem

    translation_table = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
        }
    )
    stem = stem.translate(translation_table)
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")

    if not stem:
        stem = "document"

    suffix = hashlib.sha256(normalized_filename.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{suffix}"


class DocumentRegistry:
    """
    JSON-backed registry for indexed PDF documents.
    """

    def __init__(
        self,
        registry_path: str = "artifacts/document_registry.json",
    ) -> None:
        """
        Initialize the document registry and ensure the registry file exists.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._save_registry({"documents": []})

    def list_documents(self) -> list[dict]:
        """
        Return all registered documents as shallow copies.
        """
        registry = self._load_registry()
        return [dict(document) for document in registry["documents"]]

    def get_document(self, document_id: str) -> dict | None:
        """
        Return a registered document by document_id.
        """
        normalized_document_id = self._validate_document_id(document_id)

        for document in self._load_registry()["documents"]:
            if document.get("document_id") == normalized_document_id:
                return dict(document)

        return None

    def register_document(self, document: dict) -> dict:
        """
        Insert or update a document record in the registry.
        """
        if not document:
            raise ValueError("document cannot be empty")

        document_id = self._validate_document_id(document.get("document_id"))
        document_record = dict(document)
        document_record["document_id"] = document_id

        registry = self._load_registry()
        documents = registry["documents"]

        for index, existing_document in enumerate(documents):
            if existing_document.get("document_id") == document_id:
                documents[index] = document_record
                self._save_registry(registry)
                return dict(document_record)

        documents.append(document_record)
        self._save_registry(registry)
        return dict(document_record)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document record from the registry.
        """
        normalized_document_id = self._validate_document_id(document_id)
        registry = self._load_registry()
        documents = registry["documents"]

        for index, document in enumerate(documents):
            if document.get("document_id") == normalized_document_id:
                del documents[index]
                self._save_registry(registry)
                return True

        return False

    def _load_registry(self) -> dict:
        """
        Load the registry JSON file.
        """
        with self.registry_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if "documents" not in data or not isinstance(data["documents"], list):
            data = {"documents": []}

        return data

    def _save_registry(self, data: dict) -> None:
        """
        Save registry JSON content as UTF-8.
        """
        with self.registry_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _validate_document_id(self, document_id: str | None) -> str:
        """
        Validate and normalize a document_id value.
        """
        if document_id is None or not str(document_id).strip():
            raise ValueError("document_id cannot be empty")

        return str(document_id).strip()
