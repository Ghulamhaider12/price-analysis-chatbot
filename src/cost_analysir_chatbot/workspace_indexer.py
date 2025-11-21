from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .workspace_manager import WorkspaceManager


class WorkspaceIndexer:
    """Persists embeddings per workspace to avoid reprocessing documents."""

    def __init__(self, manager: WorkspaceManager) -> None:
        self.manager = manager

    def _index_dir(self, slug: str) -> Path:
        directory = self.manager.workspace_dir(slug) / "index"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _index_path(self, slug: str, stored_name: str) -> Path:
        return self._index_dir(slug) / f"{stored_name}.json"

    def populate_pipeline(self, slug: str, pipeline) -> None:
        workspace = self.manager.get_workspace(slug)
        documents = workspace.get("documents", [])
        for record in documents:
            stored_name = record["stored_name"]
            doc_path = self.manager.document_path(slug, stored_name)
            if not doc_path.exists():
                continue
            file_hash = self.manager.file_sha256(doc_path)
            index_path = self._index_path(slug, stored_name)
            needs_reindex = True
            if index_path.exists() and record.get("sha256") == file_hash:
                needs_reindex = False

            if needs_reindex:
                processed_chunks = pipeline.ingest_documents([doc_path])
                payload = {
                    "stored_name": stored_name,
                    "sha256": file_hash,
                    "chunks": processed_chunks,
                }
                self._write_index(index_path, payload)
                self.manager.update_document_record(
                    slug,
                    stored_name,
                    sha256=file_hash,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                    chunk_count=len(processed_chunks),
                )
            else:
                payload = self._read_index(index_path)
                for chunk in payload.get("chunks", []):
                    pipeline.vector_store.add(
                        chunk["embedding"],
                        text=chunk["text"],
                        doc_id=chunk["metadata"].get("doc_id"),
                        metadata=chunk["metadata"],
                    )

    def _write_index(self, path: Path, payload: Dict[str, object]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def _read_index(self, path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
