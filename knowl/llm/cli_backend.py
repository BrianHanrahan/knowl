"""CLI backend — routes LLM requests through Claude Agent SDK (uses Claude Max subscription).

Same async generator interface as claude.stream_message_with_tools(),
yielding StreamEvent instances (TextChunkEvent, ToolCallEvent, DoneEvent).
"""

from __future__ import annotations

import os
import sys
from typing import AsyncIterator

from knowl.llm.claude import (
    TextChunkEvent,
    ToolCallEvent,
    DoneEvent,
    StreamEvent,
    format_system_prompt,
)
from knowl.log import get_logger

logger = get_logger(__name__)


def _get_mcp_server_config(project: str | None) -> dict:
    """Build the MCP server config dict for claude_agent_sdk."""
    args = ["-m", "knowl.mcp.server"]
    if project:
        args.extend(["--project", project])

    config = {
        "command": sys.executable,
        "args": args,
    }

    # Pass KNOWL_DIR if set
    knowl_dir = os.environ.get("KNOWL_DIR")
    if knowl_dir:
        config["env"] = {"KNOWL_DIR": knowl_dir}

    return config


async def stream_message_with_tools(
    user_message: str,
    tool_executor=None,  # Ignored — MCP server handles tool execution
    context: list[dict[str, str]] | None = None,
    history: list[dict[str, str]] | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    project: str | None = None,
    attachments: list[dict] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream a Claude response via the Agent SDK with Knowl MCP tools.

    Same interface as claude.stream_message_with_tools() — yields
    TextChunkEvent, ToolCallEvent, and DoneEvent.

    Key difference: Claude Agent SDK controls the tool loop. We observe
    tool calls for UI rendering but don't execute them — the MCP server
    handles execution when Claude calls it.
    """
    try:
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            ResultMessage,
            AssistantMessage,
            TextBlock,
            ToolUseBlock,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The 'claude-agent-sdk' package is not installed. "
            "Install it with: pip install claude-agent-sdk"
        ) from exc

    system_prompt = format_system_prompt(context or [])

    # Build the prompt — include history context if present
    prompt_parts = []
    if history:
        prompt_parts.append("Previous conversation:")
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(f"  {role}: {content}")
        prompt_parts.append("")
    prompt_parts.append(user_message)

    # Add attachment descriptions (CLI mode supports text attachments initially)
    if attachments:
        for att in attachments:
            if att.get("type") == "text":
                prompt_parts.append(att.get("text", ""))

    full_prompt = "\n".join(prompt_parts)

    mcp_config = _get_mcp_server_config(project)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt if system_prompt else None,
        model=model,
        mcp_servers={"knowl": mcp_config},
        allowed_tools=["mcp__knowl__*"],
        max_turns=20,
    )

    full_text_parts: list[str] = []

    try:
        async for message in query(prompt=full_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_text_parts.append(block.text)
                        yield TextChunkEvent(text=block.text)
                    elif isinstance(block, ToolUseBlock):
                        yield ToolCallEvent(name=block.name, input=block.input)
            elif isinstance(message, ResultMessage):
                if message.result and message.result not in full_text_parts:
                    full_text_parts.append(message.result)
                    yield TextChunkEvent(text=message.result)
                yield DoneEvent(full_text="".join(full_text_parts))
                return

        # If we exit the loop without a ResultMessage, still emit DoneEvent
        yield DoneEvent(full_text="".join(full_text_parts))

    except Exception as exc:
        logger.error("CLI backend error: %s", exc)
        raise RuntimeError(f"Claude Agent SDK error: {exc}") from exc


async def format_with_cli(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """One-shot prompt via Agent SDK for formatting (no tools needed)."""
    try:
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            ResultMessage,
            AssistantMessage,
            TextBlock,
        )
    except ImportError as exc:
        raise RuntimeError("claude-agent-sdk not installed") from exc

    options = ClaudeAgentOptions(model=model, max_turns=1, allowed_tools=[])
    parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
        elif isinstance(message, ResultMessage):
            if message.result:
                parts.append(message.result)
    return "".join(parts)
