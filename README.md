# Cost AnalysIR Chatbot

This project ingests investment documents, extracts text with OCR, generates OpenAI embeddings, stores vectors for retrieval, and runs RAG-based analysis while tracking every API dollar.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env file and add your actual API keys
```

## Usage

```bash
python -m src.main \
  --documents reports/q1.pdf portfolio.xlsx filings/10k.xml \
  --question "Assess revenue concentration risk and CapEx outlook." \
  --ocr-provider azure-computer-vision \
  --ocr-cost-per-page 0.002  # set if using a paid OCR API
```

To send OCR through Claude Vision (and automatically track token spend):

```bash
python -m src.main \
  --documents scans/contract.png \
  --question "Highlight major liabilities." \
  --use-claude-ocr
```

The CLI prints the investment memo, retrieved context metadata, and a JSON cost report that lists every embeddings/completions call and the total USD amount.

## Workspace CLI

Organize PDFs by project/client using persistent workspaces:

```bash
# Create a workspace
python -m src.workspace_cli create "Q2 Investor Decks" --description "All Q2 collateral"

# Add PDFs
python -m src.workspace_cli add-docs q2-investor-decks docs/*.pdf

# Ask a question across the workspace
python -m src.workspace_cli query q2-investor-decks \
  --question "Summarize capital allocation priorities for FY25."

# Generate a multi-question report
python -m src.workspace_cli report q2-investor-decks \
  --question "What are the main growth drivers?" \
  --question "List notable risks with citations."
```

Workspaces are stored under `workspaces/` by default (override with `WORKSPACES_DIR=/path/to/storage`).
Accepted file types per workspace: PDF, PNG/JPG/TIFF images, XML, Excel (`.xls/.xlsx/.xlsm`), plain text/Markdown/RTF, and Word `.docx`.

Embeddings persist inside each workspace's `index/` directory. When you upload a document it is processed once, its chunks/embeddings are cached, and future queries simply load the cached vectors without re-embedding. If a file changes (hash mismatch) it is automatically reprocessed.

### Streamlit UI

To run the browser UI for drag-and-drop uploads:

```bash
streamlit run src/streamlit_app.py
```

Upload PDFs, Excel workbooks, XML filings, images (PNG/JPEG/TIFF), plain-text/Markdown/RTF, or Word (`.docx`) files. The dashboard ingests everything stored in the selected workspace, runs retrieval, and breaks down the spend.
Flip the **Use Claude OCR (vision)** toggle in the sidebar (with `ANTHROPIC_API_KEY` set) to enable Claude-based OCR; otherwise it defaults to local Tesseract. You can also preload the toggle by setting `USE_CLAUDE_OCR=true` in the environment when launching Streamlit.

The Streamlit dashboard now exposes workspace management:

1. Create a workspace (or pick an existing one) from the sidebar.
2. Upload PDFs into that workspace; the app stores them under `workspaces/<slug>/documents/`.
3. Ask questions or enter multiple prompts to generate a structured report. All answers cite the retrieved context and the cost ledger breaks down embeddings, Claude/GPT calls, and OCR spend per workspace run.

### OCR Provider & Costing

- OCR is performed with Tesseract by default (on-device, zero marginal cost) and surfaces as `tesseract-ocr` in the ledger.
- To use Claude for OCR, either pass `--use-claude-ocr` (CLI) or enable the sidebar toggle. Each page is sent through Claude Vision; the ledger records actual prompt/completion tokens plus the page count so you can see precise OCR spend.
- If you proxy OCR through a different paid API (e.g., Azure Computer Vision or AWS Textract), set both `--ocr-provider` and `--ocr-cost-per-page` (or use the Streamlit sidebar fields). The cost tracker logs each OCR run with the provider label, document path, page count, and per-page rate so the summary shows LLM + OCR spend side by side.

### Components

- `DocumentProcessor` OCRs PDFs, scans images, parses XML, and unwraps Excel sheets.
- `VectorStore` keeps normalized embeddings for cosine similarity search.
- `OpenAIClient` wraps OpenAI calls and logs usage to the `CostTracker`.
- `RAGPipeline` orchestrates ingestion, retrieval, analysis, and cost summaries.

You can swap models with `--embedding-model` or `--chat-model` to reflect pricing differences captured in `PRICING` inside `rag_pipeline.py`.

### Using Claude (Anthropic)

- Add `ANTHROPIC_API_KEY=sk-ant-...` to your `.env` file.
- Supported models include the latest Claude 4.5 generation (`claude-sonnet-4-5`, `claude-haiku-4-5`), Claude 4.1 (`claude-opus-4-1`), and legacy 3.x models (`claude-3-5-sonnet-latest`, `claude-3-opus-latest`, `claude-3-haiku-latest`). Both alias and dated formats are available—pick whichever your account recognizes.
- In the CLI, either pass `--chat-provider anthropic` or select a `claude-*` model (provider auto-detected). If a dated string returns 404, switch to the matching `-latest` alias.
- In the Streamlit UI choose "Anthropic Claude 3.x" from the Chat Provider dropdown and pick the desired Claude model. Both alias and dated IDs are listed to avoid the not-found error.
