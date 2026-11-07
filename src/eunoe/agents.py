"""The default agent: instructions and tools, published and swappable.

The instruction text is part of an arm's definition, because a model told to look things up will.
So it is equal within a comparison, and a robustness run swaps it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_INSTRUCTIONS = (
    "{brief} Earlier parts of this session may be shown in shortened form. "
    "When a question asks for a value, reply on one line as ANSWER: <value>, or ANSWER: UNKNOWN if you cannot tell. "
    "Use your tools when you need to."
)
RECALL_NOTE = " The recall tool searches everything that happened earlier in this session, verbatim."


@dataclass
class DefaultAgent:
    """Implements `Agent`. Set `instructions` or subclass it to swap the prompt."""

    brief: str = "You are a helpful assistant working through a long session."
    tool_specs: list = field(default_factory=list)
    instructions: str = ""

    def __post_init__(self) -> None:
        if not self.instructions:
            self.instructions = DEFAULT_INSTRUCTIONS.format(brief=self.brief)

    def tools(self) -> list:
        return list(self.tool_specs)
