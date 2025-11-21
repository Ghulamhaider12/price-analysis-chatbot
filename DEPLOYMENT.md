# Deployment Guide

This guide covers how to deploy your price analysis chatbot to various platforms with proper environment variable configuration.

## 🚀 Streamlit Community Cloud (Recommended)

### 1. Repository Setup
- Push your code to GitHub (use the `clean-main` branch if main has issues)
- Ensure `.env` and `.streamlit/secrets.toml` are in `.gitignore`

### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository: `Ghulamhaider12/price-analysis-chatbot`
4. Set main file path: `src/streamlit_app.py`
5. Choose branch: `clean-main` (or `main` if available)

### 3. Configure Secrets
In your Streamlit Cloud app settings, add these secrets:

```toml
[secrets]
OPENAI_API_KEY = "sk-proj-your-actual-openai-key"
ANTHROPIC_API_KEY = "sk-ant-your-actual-anthropic-key"
USE_CLAUDE_OCR = "false"
WORKSPACES_DIR = "workspaces"
```

### 4. Deploy
Click "Deploy" and your app will be live at `https://your-app-name.streamlit.app`

---

## 🔧 Other Deployment Platforms

### Heroku

1. **Install Heroku CLI and login**
```bash
heroku login
```

2. **Create Heroku app**
```bash
heroku create your-app-name
```

3. **Set environment variables**
```bash
heroku config:set OPENAI_API_KEY=sk-proj-your-key
heroku config:set ANTHROPIC_API_KEY=sk-ant-your-key
heroku config:set USE_CLAUDE_OCR=false
heroku config:set WORKSPACES_DIR=workspaces
```

4. **Deploy**
```bash
git push heroku main
```

### Railway

1. Connect your GitHub repository
2. In project settings → Variables, add:
   - `OPENAI_API_KEY`: your OpenAI key
   - `ANTHROPIC_API_KEY`: your Anthropic key
   - `USE_CLAUDE_OCR`: false
   - `WORKSPACES_DIR`: workspaces

### Render

1. Connect your GitHub repository
2. In Environment section, add the same variables as above
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run src/streamlit_app.py --server.port $PORT`

### Docker

1. **Create Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""
ENV USE_CLAUDE_OCR="false"
ENV WORKSPACES_DIR="workspaces"

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and run**
```bash
docker build -t price-analysis-chatbot .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  price-analysis-chatbot
```

---

## 🔐 Security Best Practices

### Never Commit Secrets
- ✅ Use `.env` for local development
- ✅ Use platform-specific secret management for deployment
- ❌ Never commit `.env` or `secrets.toml` with real keys
- ❌ Never hardcode API keys in source code

### Environment Variable Priority
The app checks for API keys in this order:
1. Streamlit secrets (`st.secrets`)
2. Environment variables (`os.getenv`)
3. `.env` file (via `python-dotenv`)

### Rotate Keys Regularly
- Generate new API keys periodically
- Update them in your deployment platform
- Revoke old keys

---

## 🐛 Troubleshooting

### "Set the OPENAI_API_KEY environment variable"
- **Local**: Check your `.env` file exists and has the correct key
- **Streamlit Cloud**: Verify secrets are set in app settings
- **Other platforms**: Check environment variables in platform dashboard

### "ModuleNotFoundError: No module named 'dotenv'"
- Run: `pip install python-dotenv`
- Or deploy using `requirements.txt` which includes it

### Git Push Rejected (Secrets Detected)
- Remove `.env` from Git: `git rm --cached .env`
- Add `.env` to `.gitignore`
- Commit and push again

### App Won't Start on Platform
- Check logs for specific error messages
- Verify all required environment variables are set
- Ensure `requirements.txt` includes all dependencies

---

## 📝 Quick Setup Commands

### Local Development
```bash
# Clone and setup
git clone https://github.com/Ghulamhaider12/price-analysis-chatbot.git
cd price-analysis-chatbot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your actual API keys

# Run locally
streamlit run src/streamlit_app.py
```

### Streamlit Cloud Deployment
1. Fork/clone the repository
2. Push to your GitHub
3. Go to share.streamlit.io
4. Connect repository
5. Set secrets in app settings
6. Deploy

That's it! Your price analysis chatbot should now be running in the cloud. 🎉
