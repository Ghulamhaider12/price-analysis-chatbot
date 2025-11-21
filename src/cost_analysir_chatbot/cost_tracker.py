from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

MILLION = 1_000_000


@dataclass
class CostRecord:
    operation: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    metadata: Dict[str, str] = field(default_factory=dict)


class CostTracker:
    """Aggregates per-call pricing for OpenAI usage."""

    def __init__(self, pricing: Dict[str, Dict[str, float]]) -> None:
        self.pricing = pricing
        self.records: List[CostRecord] = []

    def log(
        self,
        *,
        operation: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        metadata: Dict[str, str] | None = None,
        cost_override: float | None = None,
    ) -> CostRecord:
        price_info = self.pricing.get(model, {})
        input_rate = price_info.get("input", 0.0)
        output_rate = price_info.get("output", 0.0)
        if cost_override is not None:
            cost = cost_override
        else:
            cost = (prompt_tokens / MILLION) * input_rate
            cost += (completion_tokens / MILLION) * output_rate
        record = CostRecord(
            operation=operation,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6),
            metadata=metadata or {},
        )
        self.records.append(record)
        return record

    def total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 6)

    def summary(self) -> Dict[str, object]:
        return {
            "total_cost_usd": self.total_cost(),
            "operations": [
                {
                    "operation": r.operation,
                    "model": r.model,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "cost_usd": r.cost_usd,
                    "metadata": r.metadata,
                }
                for r in self.records
            ],
        }
