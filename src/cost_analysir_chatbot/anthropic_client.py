from __future__ import annotations

import base64
from typing import Tuple

from anthropic import Anthropic, NotFoundError

from .cost_tracker import CostTracker
from .text_utils import count_tokens


class AnthropicClient:
    """Wraps Anthropic Claude API calls and logs usage/costs."""

    def __init__(self, api_key: str, cost_tracker: CostTracker) -> None:
        self.client = Anthropic(api_key=api_key)
        self.cost_tracker = cost_tracker

    def chat_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> str:
        try:
            response = self.client.messages.create(
                model=model,
                system=system_prompt,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except NotFoundError as exc:
            raise ValueError(
                f"Anthropic model '{model}' was not found. Select a published Claude model (e.g., claude-3-5-sonnet-latest)."
            ) from exc
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens
        else:
            prompt_tokens = count_tokens(model, system_prompt + user_prompt)
            completion_tokens = count_tokens(
                model,
                "".join(block.text for block in response.content if getattr(block, "type", "") == "text"),
            )
        self.cost_tracker.log(
            operation="chat_completion",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        content_parts = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                content_parts.append(block.text)
        return "\n\n".join(content_parts).strip()

    def ocr_image(
        self,
        *,
        model: str,
        image_bytes: bytes,
        media_type: str = "image/png",
        prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> Tuple[str, int, int]:
        if prompt is None:
            prompt = "Extract the textual content of this document image. Respond with raw text only."
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except NotFoundError as exc:
            raise ValueError(
                f"Anthropic model '{model}' was not found. Select a published Claude model (e.g., claude-3-5-sonnet-latest)."
            ) from exc
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens
        else:
            prompt_tokens = count_tokens(model, prompt)
            completion_tokens = count_tokens(
                model,
                "".join(block.text for block in response.content if getattr(block, "type", "") == "text"),
            )
        content_parts = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                content_parts.append(block.text)
        return "\n\n".join(content_parts).strip(), prompt_tokens, completion_tokens
