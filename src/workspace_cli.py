from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from cost_analysir_chatbot.rag_pipeline import RAGPipeline
from cost_analysir_chatbot.workspace_manager import WorkspaceManager
from cost_analysir_chatbot.workspace_indexer import WorkspaceIndexer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workspace manager for Cost AnalysIR.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new workspace.")
    create_parser.add_argument("name", help="Workspace name.")
    create_parser.add_argument("--description", default="", help="Workspace description.")

    subparsers.add_parser("list", help="List existing workspaces.")

    add_parser = subparsers.add_parser("add-docs", help="Add PDF documents to a workspace.")
    add_parser.add_argument("workspace", help="Workspace slug.")
    add_parser.add_argument("files", nargs="+", help="Paths to PDF files.")

    query_parser = subparsers.add_parser("query", help="Ask a question across workspace documents.")
    _add_pipeline_args(query_parser)
    query_parser.add_argument("workspace", help="Workspace slug.")
    query_parser.add_argument("question", help="Natural language question.")

    report_parser = subparsers.add_parser("report", help="Generate a multi-question report.")
    _add_pipeline_args(report_parser)
    report_parser.add_argument("workspace", help="Workspace slug.")
    report_parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to include in the report. Use multiple times for multiple sections.",
    )
    report_parser.add_argument(
        "--questions-file",
        help="Path to a text file. Each non-empty line is treated as a report question.",
    )
    return parser


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument(
        "--chat-provider",
        choices=["openai", "anthropic"],
        help="Override chat provider (defaults based on model prefix).",
    )


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY before running workspace commands.")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    return RAGPipeline(
        api_key=api_key,
        embedding_model=getattr(args, "embedding_model", "text-embedding-3-large"),
        chat_model=getattr(args, "chat_model", "gpt-4o-mini"),
        chat_provider=getattr(args, "chat_provider", None),
        anthropic_api_key=anthropic_key,
    )



def cmd_create(args: argparse.Namespace, manager: WorkspaceManager) -> None:
    workspace = manager.create_workspace(args.name, description=args.description)
    print(json.dumps(workspace, indent=2))


def cmd_list(_: argparse.Namespace, manager: WorkspaceManager) -> None:
    workspaces = manager.list_workspaces()
    print(json.dumps(workspaces, indent=2))


def cmd_add_docs(args: argparse.Namespace, manager: WorkspaceManager) -> None:
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        dest = manager.add_document_from_path(args.workspace, path)
        print(f"Added {path} to workspace '{args.workspace}' as {dest.name}")


def cmd_query(args: argparse.Namespace, manager: WorkspaceManager) -> None:
    pipeline = build_pipeline(args)
    indexer = WorkspaceIndexer(manager)
    indexer.populate_pipeline(args.workspace, pipeline)
    result = pipeline.analyze(args.question)
    print(result["answer"])
    print("\nContext Sources:")
    for ctx in result["context"]:
        print(f"- {ctx.document.metadata.get('source')} (score {ctx.score:.4f})")
    print("\nCosts:")
    print(json.dumps(result["costs"], indent=2))


def load_report_questions(args: argparse.Namespace) -> List[str]:
    questions: List[str] = []
    if args.questions:
        questions.extend(args.questions)
    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as fh:
            questions.extend([line.strip() for line in fh if line.strip()])
    if not questions:
        raise ValueError("Provide at least one question via --question or --questions-file.")
    return questions


def cmd_report(args: argparse.Namespace, manager: WorkspaceManager) -> None:
    questions = load_report_questions(args)
    pipeline = build_pipeline(args)
    indexer = WorkspaceIndexer(manager)
    indexer.populate_pipeline(args.workspace, pipeline)
    sections = []
    for q in questions:
        result = pipeline.analyze(q)
        sections.append({"question": q, "answer": result["answer"]})
    report_lines = [f"# Workspace Report: {args.workspace}", ""]
    for section in sections:
        report_lines.append(f"## Question: {section['question']}")
        report_lines.append(section["answer"])
        report_lines.append("")
    print("\n".join(report_lines))


COMMAND_HANDLERS = {
    "create": cmd_create,
    "list": cmd_list,
    "add-docs": cmd_add_docs,
    "query": cmd_query,
    "report": cmd_report,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manager = WorkspaceManager()
    handler = COMMAND_HANDLERS[args.command]
    handler(args, manager)


if __name__ == "__main__":
    main()
