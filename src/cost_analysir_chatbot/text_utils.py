from __future__ import annotations

from typing import List

import tiktoken


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200, max_tokens: int = 6000) -> List[str]:
    """
    Chunk text into smaller pieces, respecting both word count and token limits.
    
    Args:
        text: Text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
        max_tokens: Maximum tokens per chunk (for embedding model compatibility)
    """
    words = text.split()
    if not words:
        return []
    
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    
    while start < len(words):
        # Start with target chunk size
        end = min(len(words), start + chunk_size)
        chunk_text = " ".join(words[start:end])
        
        # Check if chunk exceeds token limit
        chunk_tokens = count_tokens("text-embedding-3-large", chunk_text)
        
        # If too many tokens, reduce chunk size
        while chunk_tokens > max_tokens and end > start + 1:
            end = start + int((end - start) * 0.8)  # Reduce by 20%
            chunk_text = " ".join(words[start:end])
            chunk_tokens = count_tokens("text-embedding-3-large", chunk_text)
        
        # If still too large with just one word, truncate the text
        if chunk_tokens > max_tokens:
            # Rough approximation: 4 characters per token
            max_chars = max_tokens * 4
            chunk_text = chunk_text[:max_chars]
        
        chunks.append(chunk_text)
        
        if end == len(words):
            break
        start += step
    
    return chunks


def count_tokens(model: str, text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
