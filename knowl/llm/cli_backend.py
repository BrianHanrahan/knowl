"""CLI backend — routes LLM requests through Claude Agent SDK (uses Claude Max subscription).

Same async generator interface as claude.stream_message_with_tools(),
yielding StreamEvent instances (TextChunkEvent, ToolCallEvent, DoneEvent).

The Agent SDK uses anyio task groups internally. FastAPI's StreamingResponse
consumes async generators from a different asyncio Task, which causes anyio's
cancel-scope checks to fail. We work around this by running the SDK query in
a separate thread with its own event loop, bridging events through a
thread-safe queue.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
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

_SENTINEL = object()


def _get_mcp_server_config(project: str | None) -> dict:
    """Build the MCP server config dict for claude_agent_sdk."""
    args = ["-m", "knowl.mcp.server"]
    if project:
        args.extend(["--project", project])

    config = {
        "command": sys.executable,
        "args": args,
    }

    knowl_dir = os.environ.get("KNOWL_DIR")
    if knowl_dir:
        config["env"] = {"KNOWL_DIR": knowl_dir}

    return config


def _sdk_env() -> dict[str, str]:
    """Build env overrides for the Agent SDK subprocess.

    The SDK merges os.environ with options.env (options wins), so we
    explicitly blank out vars we want removed:
    - CLAUDECODE / CLAUDE_CODE_ENTRYPOINT: prevents nested session error
    - ANTHROPIC_API_KEY: forces Claude Code to use the subscription token
      set via `claude setup-token` instead of the API key
    """
    return {
        "CLAUDECODE": "",
        "CLAUDE_CODE_ENTRYPOINT": "",
        "ANTHROPIC_API_KEY": "",
    }


def _run_sdk_in_thread(
    full_prompt: str,
    options,
    q: queue.Queue,
):
    """Run the Agent SDK query in a dedicated thread with its own event loop.

    This isolates anyio's cancel scopes from FastAPI's task scheduling.
    """
    async def _run():
        from claude_agent_sdk import (
            query,
            ResultMessage,
            AssistantMessage,
            TextBlock,
            ToolUseBlock,
        )

        full_text_parts: list[str] = []
        try:
            async for message in query(prompt=full_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            full_text_parts.append(block.text)
                            q.put(TextChunkEvent(text=block.text))
                        elif isinstance(block, ToolUseBlock):
                            q.put(ToolCallEvent(name=block.name, input=block.input))
                elif isinstance(message, ResultMessage):
                    if message.result and message.result not in full_text_parts:
                        full_text_parts.append(message.result)
                        q.put(TextChunkEvent(text=message.result))
                    q.put(DoneEvent(full_text="".join(full_text_parts)))
                    return

            q.put(DoneEvent(full_text="".join(full_text_parts)))
        except Exception as exc:
            logger.error("CLI backend error: %s", exc)
            q.put(exc)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.error("CLI backend thread error: %s\n%s", exc, tb)
        q.put(exc)
    finally:
        loop.close()
        logger.info("CLI backend thread finished")
        q.put(_SENTINEL)


async def stream_message_with_tools(
    user_message: str,
    tool_executor=None,
    context: list[dict[str, str]] | None = None,
    history: list[dict[str, str]] | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    project: str | None = None,
    attachments: list[dict] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream a Claude response via the Agent SDK with Knowl MCP tools."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:
        raise RuntimeError(
            "The 'claude-agent-sdk' package is not installed. "
            "Install it with: pip install claude-agent-sdk"
        ) from exc

    system_prompt = format_system_prompt(context or [])

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

    # Handle attachments — CLI backend can't send image content blocks via
    # the Agent SDK's string prompt, so save images/PDFs to temp files and
    # tell the agent to use the Read tool to view them.
    if attachments:
        import base64 as _b64
        import tempfile

        for att in attachments:
            if att.get("type") == "text":
                prompt_parts.append(att.get("text", ""))
            elif att.get("type") == "image":
                source = att.get("source", {})
                if source.get("type") == "base64":
                    img_data = _b64.b64decode(source["data"])
                    ext = {
                        "image/png": ".png", "image/jpeg": ".jpg",
                        "image/gif": ".gif", "image/webp": ".webp",
                    }.get(source.get("media_type", ""), ".png")
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=ext, prefix="knowl_upload_", delete=False
                    )
                    tmp.write(img_data)
                    tmp.close()
                    prompt_parts.append(
                        f"\n[Attached image saved at: {tmp.name} — "
                        f"use the Read tool to view it]"
                    )
            elif att.get("type") == "document":
                source = att.get("source", {})
                if source.get("type") == "base64":
                    doc_data = _b64.b64decode(source["data"])
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".pdf", prefix="knowl_upload_", delete=False
                    )
                    tmp.write(doc_data)
                    tmp.close()
                    prompt_parts.append(
                        f"\n[Attached PDF saved at: {tmp.name} — "
                        f"use the Read tool to view it]"
                    )

    full_prompt = "\n".join(prompt_parts)
    mcp_config = _get_mcp_server_config(project)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt if system_prompt else None,
        model=model,
        mcp_servers={"knowl": mcp_config},
        allowed_tools=["mcp__knowl__*", "Read", "Write", "Edit"],
        permission_mode="acceptEdits",
        max_turns=20,
        env=_sdk_env(),
    )

    # Thread-safe queue bridges the SDK thread and FastAPI's async generator
    q: queue.Queue = queue.Queue()

    thread = threading.Thread(
        target=_run_sdk_in_thread,
        args=(full_prompt, options, q),
        daemon=True,
    )
    thread.start()

    loop = asyncio.get_event_loop()
    try:
        while True:
            # Poll the thread-safe queue without blocking the event loop
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise RuntimeError(f"Claude Agent SDK error: {item}") from item
            yield item
    finally:
        # Thread is daemon — will be cleaned up when process exits
        pass


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

    options = ClaudeAgentOptions(model=model, max_turns=1, allowed_tools=[], env=_sdk_env())
    parts: list[str] = []
    # format_with_cli is called directly (not from StreamingResponse), so
    # the anyio scope issue doesn't apply here.
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
        elif isinstance(message, ResultMessage):
            if message.result:
                parts.append(message.result)
    return "".join(parts)
