#!/usr/bin/env python3
"""
Helper script to set up .env file for the price analysis chatbot.
"""

import os
from pathlib import Path


def main():
    """Create .env file from template and current environment variables."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    env_example = project_root / "env.example"
    
    if env_file.exists():
        print(f"✅ .env file already exists at {env_file}")
        return
    
    if not env_example.exists():
        print(f"❌ Template file {env_example} not found")
        return
    
    # Read template
    template_content = env_example.read_text()
    
    # Check for existing environment variables
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    use_claude_ocr = os.getenv("USE_CLAUDE_OCR", "false")
    workspaces_dir = os.getenv("WORKSPACES_DIR", "workspaces")
    
    # Replace placeholders with actual values if available
    content = template_content
    if openai_key:
        content = content.replace("OPENAI_API_KEY=sk-your-openai-key-here", f"OPENAI_API_KEY={openai_key}")
        print("✅ Found existing OPENAI_API_KEY in environment")
    
    if anthropic_key:
        content = content.replace("ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here", f"ANTHROPIC_API_KEY={anthropic_key}")
        print("✅ Found existing ANTHROPIC_API_KEY in environment")
    
    content = content.replace("USE_CLAUDE_OCR=false", f"USE_CLAUDE_OCR={use_claude_ocr}")
    content = content.replace("WORKSPACES_DIR=workspaces", f"WORKSPACES_DIR={workspaces_dir}")
    
    # Write .env file
    env_file.write_text(content)
    print(f"✅ Created .env file at {env_file}")
    
    if not openai_key:
        print("⚠️  Please edit .env and add your OPENAI_API_KEY")
    if not anthropic_key:
        print("⚠️  Please edit .env and add your ANTHROPIC_API_KEY (optional)")
    
    print("\n🎉 Setup complete! You can now run the application without export commands.")
    print("   The .env file will be automatically loaded when you run the scripts.")


if __name__ == "__main__":
    main()
