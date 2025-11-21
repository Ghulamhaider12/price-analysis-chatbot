# Price Analysis Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot for analyzing investment documents and financial reports.

## Features

- **Document Processing**: Upload PDFs, Excel files, images, and more
- **Natural Language Queries**: Ask questions about your documents in plain English
- **Cost Tracking**: Full transparency of API usage and costs
- **Multiple AI Providers**: Support for OpenAI and Anthropic models
- **Workspace Management**: Organize documents into workspaces
- **OCR Support**: Extract text from images and scanned documents

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/Ghulamhaider12/price-analysis-chatbot.git
cd price-analysis-chatbot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your actual API keys

# Run locally
streamlit run src/streamlit_app.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file: `src/streamlit_app.py`
5. Add secrets in Streamlit Cloud dashboard:
   ```toml
   OPENAI_API_KEY = "your-openai-key"
   ANTHROPIC_API_KEY = "your-anthropic-key"
   USE_CLAUDE_OCR = "false"
   WORKSPACES_DIR = "workspaces"
   ```

## Environment Variables

- `OPENAI_API_KEY`: Required for embeddings and chat
- `ANTHROPIC_API_KEY`: Optional, for Claude models
- `USE_CLAUDE_OCR`: Enable Claude Vision for OCR (default: false)
- `WORKSPACES_DIR`: Directory for workspace storage (default: workspaces)

## Usage

### Streamlit Interface

1. Upload documents to a workspace
2. Ask questions about your documents
3. View analysis results with cost breakdown
4. Export reports and manage workspaces

### Command Line Interface

```bash
# Analyze documents directly
python -m src.main \
  --documents reports/q1.pdf portfolio.xlsx \
  --question "What are the key financial risks?" \
  --ocr-provider tesseract-ocr

# Workspace management
python -m src.workspace_cli create "My Workspace" --description "Q1 Analysis"
python -m src.workspace_cli add-docs "My Workspace" document1.pdf document2.xlsx
python -m src.workspace_cli query "My Workspace" "What is the revenue growth?"
```

## Supported File Types

- **PDFs**: Financial reports, presentations
- **Excel/CSV**: Spreadsheets, data files
- **Images**: Screenshots, charts (PNG, JPG, TIFF)
- **Word Documents**: Reports, memos
- **XML**: Structured data files

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with OpenAI/Anthropic APIs
- **Vector Store**: In-memory FAISS for document embeddings
- **Document Processing**: Multi-format parsing with OCR support
- **Cost Tracking**: Real-time API usage monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
