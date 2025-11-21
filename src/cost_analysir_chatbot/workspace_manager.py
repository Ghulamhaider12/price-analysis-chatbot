from __future__ import annotations

import json
import re
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class WorkspaceManager:
    """Creates and tracks workspaces along with their documents."""

    METADATA_FILE = "workspace.json"
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".xml",
        ".xls",
        ".xlsx",
        ".xlsm",
        ".txt",
        ".md",
        ".rtf",
        ".docx",
    }

    def __init__(self, base_path: str | Path = "workspaces") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _slugify(self, name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", name.strip().lower())
        slug = slug.strip("-")
        return slug or "workspace"

    def _workspace_dir(self, slug: str) -> Path:
        return self.base_path / slug

    def workspace_dir(self, slug: str) -> Path:
        return self._workspace_dir(slug)

    def _metadata_path(self, slug: str) -> Path:
        return self._workspace_dir(slug) / self.METADATA_FILE

    def documents_dir(self, slug: str) -> Path:
        return self._workspace_dir(slug) / "documents"

    def document_path(self, slug: str, stored_name: str) -> Path:
        return self.documents_dir(slug) / stored_name

    def create_workspace(self, name: str, description: str = "") -> Dict[str, str]:
        slug = self._slugify(name)
        ws_dir = self._workspace_dir(slug)
        if ws_dir.exists():
            raise ValueError(f"Workspace '{slug}' already exists.")
        ws_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir(slug).mkdir(parents=True, exist_ok=True)
        (ws_dir / "index").mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": name,
            "slug": slug,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": [],
        }
        self._write_metadata(slug, metadata)
        return metadata

    def list_workspaces(self) -> List[Dict[str, str]]:
        workspaces: List[Dict[str, str]] = []
        for item in sorted(self.base_path.iterdir()):
            if not item.is_dir():
                continue
            metadata_path = item / self.METADATA_FILE
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                workspaces.append(data)
        return workspaces

    def get_workspace(self, slug: str) -> Dict[str, str]:
        metadata_path = self._metadata_path(slug)
        if not metadata_path.exists():
            raise ValueError(f"Workspace '{slug}' does not exist.")
        with metadata_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def add_document_from_path(self, slug: str, source_path: str | Path, display_name: str | None = None) -> Path:
        workspace = self.get_workspace(slug)
        docs_dir = self.documents_dir(slug)
        docs_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(source_path)
        self._validate_extension(source_path.suffix)
        if display_name:
            filename = display_name
        else:
            filename = source_path.name
        dest_path = self._unique_destination(docs_dir, filename)
        shutil.copy2(source_path, dest_path)
        self._record_document(workspace, slug, dest_path.name, source_path.name)
        return dest_path

    def add_document_from_bytes(self, slug: str, filename: str, data: bytes) -> Path:
        workspace = self.get_workspace(slug)
        docs_dir = self.documents_dir(slug)
        docs_dir.mkdir(parents=True, exist_ok=True)
        self._validate_extension(Path(filename).suffix)
        dest_path = self._unique_destination(docs_dir, filename)
        with dest_path.open("wb") as fh:
            fh.write(data)
        self._record_document(workspace, slug, dest_path.name, filename)
        return dest_path

    def workspace_document_paths(self, slug: str) -> List[Path]:
        docs_dir = self.documents_dir(slug)
        if not docs_dir.exists():
            return []
        files: List[Path] = []
        for path in docs_dir.iterdir():
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(path)
        return sorted(files)

    def _unique_destination(self, docs_dir: Path, filename: str) -> Path:
        base = Path(filename).stem
        ext = Path(filename).suffix or ".pdf"
        counter = 1
        candidate = docs_dir / f"{base}{ext}"
        while candidate.exists():
            candidate = docs_dir / f"{base}-{counter}{ext}"
            counter += 1
        return candidate

    def _record_document(self, workspace: Dict[str, str], slug: str, stored_name: str, original_name: str) -> None:
        stored_path = self.document_path(slug, stored_name)
        digest = self._file_sha256(stored_path)
        documents = workspace.setdefault("documents", [])
        documents.append(
            {
                "stored_name": stored_name,
                "original_name": original_name,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "sha256": digest,
                "indexed_at": None,
                "chunk_count": 0,
            }
        )
        self._write_metadata(slug, workspace)

    def update_document_record(self, slug: str, stored_name: str, **updates: object) -> None:
        workspace = self.get_workspace(slug)
        updated = False
        for record in workspace.get("documents", []):
            if record.get("stored_name") == stored_name:
                record.update(updates)
                updated = True
                break
        if updated:
            self._write_metadata(slug, workspace)

    def _write_metadata(self, slug: str, metadata: Dict[str, str]) -> None:
        metadata_path = self._metadata_path(slug)
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _validate_extension(self, suffix: str) -> None:
        if suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document type '{suffix}'.")

    def file_sha256(self, path: Path) -> str:
        return self._file_sha256(path)

    def _file_sha256(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
