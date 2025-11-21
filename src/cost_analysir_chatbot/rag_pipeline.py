from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, List

from .cost_tracker import CostTracker
from .document_processor import DocumentProcessor, ExtractionResult
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .text_utils import chunk_text, count_tokens
from .vector_store import VectorStore, VectorDocument

ANTHROPIC_MODEL_ALIASES = {
    "claude-3-5-sonnet-latest":"claude-sonnet-4-5-20250929",
    "claude-3-opus-latest": "claude-3-opus-20240229",
    "claude-3-haiku-latest": "claude-3-haiku-20240307",
}

PRICING = {
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    # Claude 4.5 models (latest generation)
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # Claude 4.1 models
    "claude-opus-4-1": {"input": 15.0, "output": 75.0},
    "claude-opus-4-1-20250805": {"input": 15.0, "output": 75.0},
    # Claude 3.5 models (legacy)
    "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    # Claude 3 models (legacy)
    "claude-3-opus-latest": {"input": 15.0, "output": 75.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-haiku-latest": {"input": 0.25, "output": 1.25},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "tesseract-ocr": {"input": 0.0, "output": 0.0},
}


@dataclass
class SearchResult:
    document: VectorDocument
    score: float


class RAGPipeline:
    """Coordinates ingestion, vector search, analysis, and cost tracking."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-large",
        chat_model: str = "gpt-4o-mini",
        chat_provider: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        ocr_cost_per_page: float = 0.0,
        ocr_provider: str = "tesseract-ocr",
        max_context_tokens: int = 6000,
    ) -> None:
        self.openai_api_key = openai_api_key or api_key
        if not self.openai_api_key:
            raise ValueError("An OpenAI API key is required for embeddings.")
        self.anthropic_api_key = anthropic_api_key
        self.embedding_model = embedding_model
        self.chat_provider = chat_provider or self._infer_provider(chat_model)
        self.chat_model = self._normalize_chat_model(chat_model, self.chat_provider)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cost_tracker = CostTracker(PRICING)
        self.ocr_cost_per_page = ocr_cost_per_page
        self.ocr_provider = self._normalize_ocr_provider(ocr_provider)
        self.max_context_tokens = max_context_tokens
        self.vector_store = VectorStore()
        self.openai = OpenAIClient(api_key=self.openai_api_key, cost_tracker=self.cost_tracker)
        self.anthropic = (
            AnthropicClient(api_key=self.anthropic_api_key, cost_tracker=self.cost_tracker)
            if self.anthropic_api_key
            else None
        )
        self.document_processor = DocumentProcessor(
            ocr_provider=self.ocr_provider,
            anthropic_client=self.anthropic,
        )

    def _infer_provider(self, model: str) -> str:
        return "anthropic" if model.lower().startswith("claude") else "openai"

    def set_chat_model(self, model: str, provider: str | None = None) -> None:
        self.chat_provider = provider or self._infer_provider(model)
        self.chat_model = self._normalize_chat_model(model, self.chat_provider)

    def ingest_documents(self, paths: Iterable[str | Path]) -> List[dict]:
        processed_chunks: List[dict] = []
        for path in paths:
            path = Path(path)
            extraction = self.document_processor.extract_text(path)
            text = extraction.text
            if not text.strip():
                continue
            if extraction.ocr_pages > 0:
                self._log_ocr_usage(path, extraction)
            chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            embeddings = self.openai.embed_texts(chunks, model=self.embedding_model)
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {"source": str(path), "chunk": str(idx)}
                doc_id = f"{path.stem}-{idx}"
                self.vector_store.add(embedding, text=chunk, doc_id=doc_id, metadata=metadata)
                processed_chunks.append(
                    {
                        "text": chunk,
                        "embedding": list(embedding),
                        "metadata": {**metadata, "doc_id": doc_id},
                    }
                )
        return processed_chunks

    def _log_ocr_usage(self, path: Path, extraction: ExtractionResult) -> None:
        model = extraction.ocr_provider or self.ocr_provider
        metadata = {
            "pages": str(extraction.ocr_pages),
            "document": str(path),
            "ocr_provider": model,
        }
        if extraction.ocr_prompt_tokens or extraction.ocr_completion_tokens:
            metadata["pricing"] = "token"
            self.cost_tracker.log(
                operation="ocr",
                model=model,
                prompt_tokens=extraction.ocr_prompt_tokens,
                completion_tokens=extraction.ocr_completion_tokens,
                metadata=metadata,
            )
            return
        metadata["pricing"] = "per_page"
        metadata["unit_cost_usd"] = f"{self.ocr_cost_per_page:.6f}"
        total_cost = extraction.ocr_pages * self.ocr_cost_per_page
        self.cost_tracker.log(
            operation="ocr",
            model=model,
            prompt_tokens=extraction.ocr_pages,
            completion_tokens=0,
            metadata=metadata,
            cost_override=round(total_cost, 6),
        )

    def _normalize_chat_model(self, model: str, provider: str) -> str:
        if provider != "anthropic":
            return model
        return self._normalize_anthropic_model(model)

    def _normalize_ocr_provider(self, provider: str) -> str:
        if provider.lower().startswith("claude"):
            return self._normalize_anthropic_model(provider)
        return provider

    def _normalize_anthropic_model(self, model: str) -> str:
        normalized = ANTHROPIC_MODEL_ALIASES.get(model)
        if normalized:
            return normalized
        dated_pattern = re.compile(r"^(claude-[\w-]+?)-(20\d{6,})$")
        match = dated_pattern.match(model)
        if match:
            return model
        suffix_model = f"{model}-latest"
        return ANTHROPIC_MODEL_ALIASES.get(suffix_model, model)

    def search_context(self, query: str, *, top_k: int = 5) -> List[SearchResult]:
        query_embedding = self.openai.embed_texts([query], model=self.embedding_model)[0]
        matches = self.vector_store.search(query_embedding, top_k=top_k)
        return [SearchResult(document=doc, score=score) for doc, score in matches]

    def analyze(self, question: str, *, top_k: int = 5) -> dict:
        context_results = self.search_context(question, top_k=top_k)
        context_text = self._build_context_text(context_results)
        system_prompt = (
            "You are an investment analysis assistant. Evaluate financial health, risk exposure, "
            "portfolio alignment, and notable red flags using only the provided context."
        )
        user_prompt = (
            "Context:\n"
            f"{context_text}\n\n"
            f"Question: {question}\n"
            "Respond with a concise investment memo that cites the sources you used."
        )
        if self.chat_provider == "anthropic":
            if not self.anthropic:
                raise EnvironmentError("ANTHROPIC_API_KEY must be set to use Claude models.")
            response = self.anthropic.chat_completion(
                model=self.chat_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            response = self.openai.chat_completion(
                model=self.chat_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        return {
            "answer": response,
            "context": context_results,
            "costs": self.cost_tracker.summary(),
        }

    def cost_report(self) -> dict:
        return self.cost_tracker.summary()

    def _build_context_text(self, context_results: List[SearchResult]) -> str:
        if not context_results:
            return "No relevant context found."
        blocks: List[str] = []
        tokens_used = 0
        for result in context_results:
            block = (
                f"Source: {result.document.metadata.get('source')}\n"
                f"Relevance: {result.score:.4f}\n"
                f"{result.document.text}"
            )
            block_tokens = count_tokens(self.chat_model, block)
            if tokens_used + block_tokens > self.max_context_tokens:
                break
            blocks.append(block)
            tokens_used += block_tokens
        if not blocks:
            return "No relevant context found."
        return "\n\n".join(blocks)
