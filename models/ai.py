"""AI service models for structured LLM interaction."""

from typing import Any
from pydantic import BaseModel, Field


class AIAction(BaseModel):
    """A single action recommended by the AI planner."""

    operation: str
    target: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class AIPlan(BaseModel):
    """Structured plan returned by AI planner.

    Only stores concise reasoning/evidence — no chain-of-thought.
    """

    decision: str = ""
    confidence: float = 0.0
    reasoning_summary: str = ""
    actions: list[AIAction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
