from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from cost_analysir_chatbot.rag_pipeline import RAGPipeline
from cost_analysir_chatbot.workspace_manager import WorkspaceManager
from cost_analysir_chatbot.workspace_indexer import WorkspaceIndexer

SUPPORTED_EXTS = [
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
    "xml",
    "xls",
    "xlsx",
    "xlsm",
    "txt",
    "md",
    "rtf",
    "docx",
]

CHAT_MODEL_OPTIONS = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo-0125",
    ],
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1",
        "claude-opus-4-1-20250805",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-latest",
        "claude-3-opus-20240229",
        "claude-3-haiku-latest",
        "claude-3-haiku-20240307",
    ],
}


def get_workspace_manager() -> WorkspaceManager:
    base_dir = Path(os.getenv("WORKSPACES_DIR", "workspaces"))
    return WorkspaceManager(base_path=base_dir)


def ensure_api_keys() -> tuple[str, Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Set the OPENAI_API_KEY environment variable before using the app.")
        st.stop()
    return api_key, anthropic_key


def sidebar_models(anthropic_key: Optional[str]) -> dict:
    st.header("Model Settings")
    embedding_model = st.selectbox(
        "Embedding Model",
        options=[
            "text-embedding-3-large",
            "text-embedding-3-small",
        ],
        index=0,
    )
    chat_provider_label = st.selectbox(
        "Chat Provider",
        options=[
            ("openai", "OpenAI (GPT-4o family)"),
            ("anthropic", "Anthropic Claude"),
        ],
        format_func=lambda option: option[1],
        index=0,
    )
    chat_provider = chat_provider_label[0]
    chat_model = st.selectbox(
        "Chat Model",
        options=CHAT_MODEL_OPTIONS[chat_provider],
        index=0,
    )
    top_k = st.slider("Context Passages", min_value=3, max_value=10, value=5)
    use_claude_env = os.getenv("USE_CLAUDE_OCR", "false").lower() in {"1", "true", "yes"}
    use_claude_ocr = st.checkbox(
        "Use Claude OCR (vision)",
        value=use_claude_env,
        help="Send each PDF page to Claude Vision (requires ANTHROPIC_API_KEY).",
    )
    if use_claude_ocr and not anthropic_key:
        st.warning("Claude OCR is enabled but ANTHROPIC_API_KEY is not set.")
    ocr_provider = st.text_input(
        "OCR provider label",
        value="tesseract-ocr",
        disabled=use_claude_ocr,
    )
    ocr_cost_per_page = st.number_input(
        "OCR cost per page (USD)",
        min_value=0.0,
        value=0.0,
        step=0.001,
        format="%.4f",
        disabled=use_claude_ocr,
    )
    return {
        "embedding_model": embedding_model,
        "chat_provider": chat_provider,
        "chat_model": chat_model,
        "top_k": top_k,
        "use_claude_ocr": use_claude_ocr,
        "ocr_provider": ocr_provider,
        "ocr_cost_per_page": ocr_cost_per_page,
    }


def sidebar_workspace(manager: WorkspaceManager) -> Optional[dict]:
    st.header("Workspaces")
    workspaces = manager.list_workspaces()
    if not workspaces:
        st.info("Create a workspace to begin.")
    options = {f"{ws['name']} ({ws['slug']})": ws for ws in workspaces}
    selected = None
    if options:
        choice = st.selectbox("Choose workspace", list(options.keys()))
        selected = options[choice]
    st.subheader("Create Workspace")
    name = st.text_input("Workspace name")
    description = st.text_input("Description")
    if st.button("Create workspace"):
        if not name.strip():
            st.error("Workspace name cannot be empty.")
        else:
            try:
                workspace = manager.create_workspace(name, description)
                st.success(f"Workspace '{workspace['slug']}' created.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    return selected


def display_workspace_documents(manager: WorkspaceManager, slug: str) -> None:
    workspace = manager.get_workspace(slug)
    docs = workspace.get("documents", [])
    if docs:
        st.subheader("Workspace Documents")
        st.dataframe(pd.DataFrame(docs))
    else:
        st.info("No PDFs uploaded to this workspace yet.")


def upload_documents_ui(manager: WorkspaceManager, slug: str) -> None:
    uploads = st.file_uploader(
        "Upload PDFs to this workspace",
        type=SUPPORTED_EXTS,
        accept_multiple_files=True,
    )
    if uploads and st.button("Save PDFs to workspace"):
        saved = 0
        for upload in uploads:
            try:
                manager.add_document_from_bytes(slug, upload.name, upload.getbuffer())
                saved += 1
            except ValueError as exc:
                st.error(str(exc))
        if saved:
            st.success(f"Stored {saved} PDF(s) in workspace '{slug}'.")
            st.rerun()


def build_pipeline(settings: dict, api_key: str, anthropic_key: Optional[str]) -> RAGPipeline:
    ocr_provider = settings["ocr_provider"]
    ocr_cost = settings["ocr_cost_per_page"]
    if settings["use_claude_ocr"]:
        if not anthropic_key:
            raise EnvironmentError("Set ANTHROPIC_API_KEY to use Claude OCR.")
        ocr_provider = "claude-3-5-sonnet-latest"
        ocr_cost = 0.0
    return RAGPipeline(
        api_key=api_key,
        embedding_model=settings["embedding_model"],
        chat_model=settings["chat_model"],
        chat_provider=settings["chat_provider"],
        anthropic_api_key=anthropic_key,
        ocr_cost_per_page=ocr_cost,
        ocr_provider=ocr_provider,
    )


def run_query(manager: WorkspaceManager, indexer: WorkspaceIndexer, slug: str, question: str, settings: dict, api_key: str, anthropic_key: Optional[str]) -> Optional[dict]:
    docs = manager.workspace_document_paths(slug)
    if not docs:
        st.error("Workspace has no PDFs. Upload documents first.")
        return None
    pipeline = build_pipeline(settings, api_key, anthropic_key)
    indexer.populate_pipeline(slug, pipeline)
    return pipeline.analyze(question, top_k=settings["top_k"])


def render_costs(cost_summary: dict) -> None:
    st.subheader("LLM & OCR Cost Breakdown (USD)")
    rows = []
    for op in cost_summary["operations"]:
        entry = {
            "Operation": op["operation"],
            "Model": op["model"],
            "Prompt Tokens": op["prompt_tokens"],
            "Completion Tokens": op["completion_tokens"],
            "Cost (USD)": op["cost_usd"],
        }
        metadata = op.get("metadata") or {}
        for key, value in metadata.items():
            entry[f"meta:{key}"] = value
        rows.append(entry)
    if rows:
        st.dataframe(pd.DataFrame(rows))
    ocr_rows = [r for r in cost_summary["operations"] if r["operation"] == "ocr"]
    ocr_cost = sum(r["cost_usd"] for r in ocr_rows)
    llm_cost = cost_summary["total_cost_usd"] - ocr_cost
    col1, col2 = st.columns(2)
    col1.metric("OCR Cost (USD)", f"${ocr_cost:.6f}")
    col2.metric("LLM Cost (USD)", f"${llm_cost:.6f}")
    if ocr_rows:
        st.markdown("**OCR Details**")
        details = [
            {
                "Provider": r["model"],
                "Document": (r.get("metadata") or {}).get("document"),
                "Pages": (r.get("metadata") or {}).get("pages"),
                "Pricing": (r.get("metadata") or {}).get("pricing"),
                "Prompt Tokens": r["prompt_tokens"],
                "Completion Tokens": r["completion_tokens"],
                "Cost (USD)": r["cost_usd"],
            }
            for r in ocr_rows
        ]
        st.table(pd.DataFrame(details))
    st.metric("Total Cost", f"${cost_summary['total_cost_usd']:.6f}")


def main() -> None:
    st.set_page_config(page_title="Cost AnalysIR Workspaces", layout="wide")
    st.title("Cost AnalysIR Workspaces")
    st.write(
        "Organize PDFs into workspaces, run natural-language queries, and generate reports with full cost transparency."
    )
    api_key, anthropic_key = ensure_api_keys()
    manager = get_workspace_manager()
    indexer = WorkspaceIndexer(manager)

    with st.sidebar:
        settings = sidebar_models(anthropic_key)
        selected_workspace = sidebar_workspace(manager)

    if not selected_workspace:
        st.stop()

    slug = selected_workspace["slug"]
    st.subheader(f"Workspace: {selected_workspace['name']} ({slug})")
    st.write(selected_workspace.get("description", ""))

    upload_documents_ui(manager, slug)
    display_workspace_documents(manager, slug)

    st.subheader("Ask a Question across Workspace PDFs")
    question = st.text_area("Enter question", placeholder="e.g., Summarize liquidity risk exposure.")
    if st.button("Run Query", type="primary"):
        if not question.strip():
            st.warning("Enter a question to analyze.")
        else:
            with st.spinner("Running workspace analysis..."):
                result = run_query(manager, indexer, slug, question, settings, api_key, anthropic_key)
            if result:
                st.markdown("### Investment Memo")
                st.write(result["answer"])
                st.markdown("### Retrieved Context")
                if result["context"]:
                    context_rows = [
                        {
                            "Source": ctx.document.metadata.get("source"),
                            "Chunk": ctx.document.metadata.get("chunk"),
                            "Score": round(ctx.score, 4),
                            "Preview": ctx.document.text[:200] + "...",
                        }
                        for ctx in result["context"]
                    ]
                    st.dataframe(pd.DataFrame(context_rows))
                else:
                    st.info("No relevant context retrieved.")
                render_costs(result["costs"])

    st.subheader("Generate Workspace Report")
    multi_questions = st.text_area(
        "Enter one question per line to build a multi-section report.",
        placeholder="What is the revenue outlook?\nList key contractual obligations.",
    )
    if st.button("Generate Report"):
        questions = [line.strip() for line in multi_questions.splitlines() if line.strip()]
        if not questions:
            st.warning("Provide at least one question.")
        else:
            sections: List[dict] = []
            pipeline = None
            with st.spinner("Generating report..."):
                docs = manager.workspace_document_paths(slug)
                if not docs:
                    st.error("Workspace has no PDFs.")
                else:
                    pipeline = build_pipeline(settings, api_key, anthropic_key)
                    indexer.populate_pipeline(slug, pipeline)
                    for q in questions:
                        sections.append({"question": q, "result": pipeline.analyze(q, top_k=settings["top_k"])})
            if sections and pipeline:
                st.markdown("### Workspace Report")
                for section in sections:
                    st.write(f"#### Question: {section['question']}")
                    st.write(section["result"]["answer"])
                render_costs(pipeline.cost_tracker.summary())


if __name__ == "__main__":
    main()
