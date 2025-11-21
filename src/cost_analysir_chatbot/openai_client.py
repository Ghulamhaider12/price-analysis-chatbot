from __future__ import annotations

from typing import Iterable, List, Sequence

from openai import OpenAI

from .cost_tracker import CostTracker
from .text_utils import count_tokens


class OpenAIClient:
    """Wraps OpenAI calls and records usage in the CostTracker."""

    def __init__(self, api_key: str, cost_tracker: CostTracker) -> None:
        self.client = OpenAI(api_key=api_key)
        self.cost_tracker = cost_tracker

    def embed_texts(self, texts: Sequence[str], *, model: str) -> List[List[float]]:
        # OpenAI embedding models have token limits (e.g., text-embedding-3-large: 8192 tokens)
        # We need to batch requests to stay within limits
        max_tokens_per_batch = 6000  # Conservative limit to account for encoding overhead
        max_tokens_per_text = 7500   # Maximum tokens for a single text chunk
        
        all_embeddings = []
        current_batch = []
        current_tokens = 0
        total_prompt_tokens = 0
        batch_count = 0
        
        for text in texts:
            text_tokens = count_tokens(model, text)
            
            # If a single text is too large, truncate it
            if text_tokens > max_tokens_per_text:
                # Truncate the text to fit within limits (rough approximation: 4 chars per token)
                max_chars = max_tokens_per_text * 4
                text = text[:max_chars]
                text_tokens = count_tokens(model, text)
            
            # If adding this text would exceed the batch limit, process current batch
            if current_batch and (current_tokens + text_tokens > max_tokens_per_batch):
                batch_embeddings, batch_tokens = self._embed_batch(current_batch, model)
                all_embeddings.extend(batch_embeddings)
                total_prompt_tokens += batch_tokens
                current_batch = []
                current_tokens = 0
                batch_count += 1
            
            current_batch.append(text)
            current_tokens += text_tokens
        
        # Process remaining batch
        if current_batch:
            batch_embeddings, batch_tokens = self._embed_batch(current_batch, model)
            all_embeddings.extend(batch_embeddings)
            total_prompt_tokens += batch_tokens
            batch_count += 1
        
        # Log total usage
        self.cost_tracker.log(
            operation="embeddings",
            model=model,
            prompt_tokens=total_prompt_tokens,
            metadata={"texts_count": str(len(texts)), "batches": str(batch_count)},
        )
        
        return all_embeddings
    
    def _embed_batch(self, texts: List[str], model: str) -> tuple[List[List[float]], int]:
        """Embed a single batch of texts and return embeddings + token count."""
        response = self.client.embeddings.create(model=model, input=texts)
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = usage.prompt_tokens
        else:
            prompt_tokens = sum(count_tokens(model, text) for text in texts)
        
        return [item.embedding for item in response.data], prompt_tokens

    def chat_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        else:
            prompt_tokens = count_tokens(model, system_prompt + user_prompt)
            completion_tokens = count_tokens(model, response.choices[0].message.content)
        self.cost_tracker.log(
            operation="chat_completion",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return response.choices[0].message.content

