from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from cost_analysir_chatbot.rag_pipeline import RAGPipeline

# Load environment variables from .env file
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the investment analysis RAG chatbot.")
    parser.add_argument(
        "--documents",
        nargs="+",
        required=True,
        help="Document paths (PDF, images, XML, Excel) to ingest.",
    )
    parser.add_argument("--question", required=True, help="Investment analysis question to ask.")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model identifier.",
    )
    parser.add_argument(
        "--chat-model",
        default="gpt-4o-mini",
        help="Chat/completion model identifier (GPT or Claude).",
    )
    parser.add_argument(
        "--chat-provider",
        choices=["openai", "anthropic"],
        help="Override chat provider (defaults based on model prefix).",
    )
    parser.add_argument(
        "--ocr-cost-per-page",
        type=float,
        default=0.0,
        help="USD cost per OCR page (set if using a paid OCR API).",
    )
    parser.add_argument(
        "--ocr-provider",
        default="tesseract-ocr",
        help="OCR engine identifier (e.g., tesseract-ocr, azure-computer-vision, claude-3-5-sonnet-latest).",
    )
    parser.add_argument(
        "--use-claude-ocr",
        action="store_true",
        help="Shortcut to run OCR with Claude (requires ANTHROPIC_API_KEY). Overrides --ocr-provider.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to retrieve.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable before running.")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    ocr_provider = args.ocr_provider
    if args.use_claude_ocr:
        if not anthropic_key:
            raise EnvironmentError("Set ANTHROPIC_API_KEY to enable Claude OCR.")
        ocr_provider = "claude-3-5-sonnet-latest"
    pipeline = RAGPipeline(
        api_key=api_key,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        chat_provider=args.chat_provider,
        anthropic_api_key=anthropic_key,
        ocr_cost_per_page=args.ocr_cost_per_page,
        ocr_provider=ocr_provider,
    )
    for path in args.documents:
        if not Path(path).exists():
            raise FileNotFoundError(f"Document not found: {path}")
    try:
        pipeline.ingest_documents(args.documents)
        analysis = pipeline.analyze(args.question, top_k=args.top_k)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
    print("\n=== Investment Analysis ===\n")
    print(analysis["answer"])
    print("\n=== Retrieved Context ===\n")
    for ctx in analysis["context"]:
        print(f"Source: {ctx.document.metadata.get('source')} | Score: {ctx.score:.4f}")
    print("\n=== Cost Breakdown (USD) ===\n")
    print(json.dumps(analysis["costs"], indent=2))


if __name__ == "__main__":
    main()
