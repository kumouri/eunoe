"""Blocks to a provider's request shape. A pure function, so a request's bytes are reproducible."""
from __future__ import annotations

import json

from eunoe.interfaces import Block, RenderedContext


def canonical(block: Block) -> str:
    """One block's stable bytes: what a prefix cache compares."""
    return json.dumps([block.role, block.type, block.text, block.name, block.input, block.tool_id, block.signature],
                      sort_keys=True, ensure_ascii=False)


def to_anthropic(ctx: RenderedContext, thinking_replay: bool = True, breakpoints=(), cache_control=None) -> dict:
    """`{system, messages, tools}` for the Messages API. Contiguous same-role blocks merge into one turn.

    `breakpoints` names block indexes that get `cache_control`, on top of the request's own. A lookup only checks 20
    positions back, so a second breakpoint is what keeps a read reachable after a large gap or an edit past it.
    """
    messages: list = []
    marked = set(breakpoints)
    for index, block in enumerate(ctx.blocks):
        content = _anthropic_block(block, thinking_replay)
        if content is None:
            continue
        if index in marked and cache_control:
            content["cache_control"] = dict(cache_control)
        if messages and messages[-1]["role"] == block.role:
            messages[-1]["content"].append(content)
        else:
            messages.append({"role": block.role, "content": [content]})
    for message in messages:
        if message["role"] == "user":
            # A tool_result has to open the turn that follows its tool_use.
            message["content"].sort(key=lambda c: c["type"] != "tool_result")
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "(session start)"}]})
    out = {"messages": messages}
    if ctx.system:
        out["system"] = ctx.system
    if ctx.tools:
        out["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in ctx.tools]
    return out


def _anthropic_block(block: Block, thinking_replay: bool):
    if block.type == "text":
        return {"type": "text", "text": block.text or " "}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.tool_id, "name": block.name, "input": block.input or {}}
    if block.type == "tool_result":
        return {"type": "tool_result", "tool_use_id": block.tool_id, "content": block.text or " "}
    if block.type == "thinking":
        if not thinking_replay:
            return None
        return {"type": "thinking", "thinking": block.text, "signature": block.signature}
    raise ValueError(f"unknown block type {block.type!r}")


def to_openai_input(ctx: RenderedContext) -> dict:
    """`{instructions, input, tools}` for the Responses API, sent whole: no `previous_response_id`."""
    items: list = []
    for block in ctx.blocks:
        if block.type == "text":
            kind = "output_text" if block.role == "assistant" else "input_text"
            items.append({"type": "message", "role": block.role, "content": [{"type": kind, "text": block.text or " "}]})
        elif block.type == "tool_use":
            items.append({"type": "function_call", "call_id": block.tool_id, "name": block.name,
                          "arguments": json.dumps(block.input or {}, sort_keys=True)})
        elif block.type == "tool_result":
            items.append({"type": "function_call_output", "call_id": block.tool_id, "output": block.text or " "})
    out = {"input": items}
    if ctx.system:
        out["instructions"] = ctx.system
    if ctx.tools:
        out["tools"] = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.schema}
                        for t in ctx.tools]
    return out


def to_gemini(ctx: RenderedContext) -> dict:
    """`{systemInstruction, contents, tools}` for generateContent. Stateless, so edits simply send."""
    contents: list = []
    for block in ctx.blocks:
        role = "model" if block.role == "assistant" else "user"
        if block.type == "text":
            part = {"text": block.text or " "}
        elif block.type == "tool_use":
            part = {"functionCall": {"name": block.name, "args": block.input or {}}}
        elif block.type == "tool_result":
            part = {"functionResponse": {"name": _call_name(ctx, block.tool_id), "response": {"result": block.text}}}
        else:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": role, "parts": [part]})
    out = {"contents": contents}
    if ctx.system:
        out["systemInstruction"] = {"parts": [{"text": ctx.system}]}
    if ctx.tools:
        out["tools"] = [{"functionDeclarations": [
            {"name": t.name, "description": t.description, "parameters": t.schema} for t in ctx.tools]}]
    return out


def _call_name(ctx: RenderedContext, tool_id: str) -> str:
    for block in ctx.blocks:
        if block.type == "tool_use" and block.tool_id == tool_id:
            return block.name
    return "tool"


def to_chat_messages(ctx: RenderedContext) -> list:
    """Plain role and content messages, for a local model or a harness with no tool blocks."""
    messages: list = []
    if ctx.system:
        messages.append({"role": "system", "content": ctx.system})
    for block in ctx.blocks:
        if block.type == "thinking":
            continue
        if block.type == "tool_use":
            text = f"[called {block.name} {json.dumps(block.input or {}, sort_keys=True)}]"
        elif block.type == "tool_result":
            text = f"[result]\n{block.text}"
        else:
            text = block.text
        if messages and messages[-1]["role"] == block.role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": block.role, "content": text})
    return messages


def to_text(ctx: RenderedContext) -> str:
    """Everything a model could read, as text. The scripted reader answers from this and nothing else."""
    return "\n".join(block.text if block.type != "tool_use" else f"{block.name} {json.dumps(block.input or {})}"
                     for block in ctx.blocks if block.type != "thinking")
