"""Claude integration — send messages to Anthropic Claude with assembled context."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Awaitable, Union

from knowl.log import get_logger

logger = get_logger(__name__)


# ── Stream event types for tool-use loop ─────────────────────────────

@dataclass
class TextChunkEvent:
    text: str

@dataclass
class ToolCallEvent:
    name: str
    input: dict

@dataclass
class DoneEvent:
    full_text: str

StreamEvent = Union[TextChunkEvent, ToolCallEvent, DoneEvent]


# ── Tool definitions ─────────────────────────────────────────────────

TOOLS = [
    # Web tools
    {
        "name": "web_search",
        "description": (
            "Search the web using Google. Use this when the user asks about "
            "current events, recent information, or anything that requires "
            "up-to-date knowledge beyond your training data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on Google.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch and read the content of a web page. Use this to get "
            "detailed information from a specific URL, such as a search "
            "result you want to read in full."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to fetch.",
                },
            },
            "required": ["url"],
        },
    },
    # Context tools
    {
        "name": "list_context_files",
        "description": (
            "List all context files available in the Knowl system. "
            "Returns global files and project-specific files with their names, "
            "token counts, and paths. Use this to see what context exists before "
            "reading or modifying files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Project name to list files for. If omitted, lists global files only.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_context_file",
        "description": (
            "Read the full content of a context file by its path. "
            "Use list_context_files first to discover available file paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The full file path as returned by list_context_files.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_context_file",
        "description": (
            "Create or update a context file. If the path exists, the file is overwritten. "
            "If creating a new file, provide a filename and scope. "
            "Use this when the user asks you to save, update, or remember information "
            "in their context files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Full path of an existing file to overwrite. "
                        "Omit this if creating a new file — use filename and scope instead."
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": "Filename for a new file (e.g. 'notes.md'). Used with scope when creating.",
                },
                "scope": {
                    "type": "string",
                    "description": "Where to create the file: 'global' or a project name. Used with filename when creating.",
                },
                "content": {
                    "type": "string",
                    "description": "The markdown content to write to the file.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "delete_context_file",
        "description": (
            "Delete a context file by its path. Use with caution — this is permanent. "
            "Only delete when the user explicitly asks to remove a file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The full file path to delete.",
                },
            },
            "required": ["path"],
        },
    },
]


def get_api_key() -> str:
    """Read ANTHROPIC_API_KEY from environment. Raises if missing."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it to your Anthropic API key to use Claude."
        )
    return key


def get_client() -> Any:
    """Return an Anthropic client instance."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The 'anthropic' package is not installed. "
            "Install it with: pip install anthropic"
        ) from exc
    return anthropic.Anthropic(api_key=get_api_key())


def format_system_prompt(context: list[dict[str, str]]) -> str:
    """Turn assembled context pieces into a single system prompt.

    Each context piece has keys: role, content, source.
    They are concatenated with section headers.
    """
    if not context:
        return ""
    sections: list[str] = []
    for piece in context:
        source = piece.get("source", "unknown")
        content = piece.get("content", "").strip()
        if content:
            sections.append(f"## {source}\n\n{content}")
    return "\n\n---\n\n".join(sections)


def send_message(
    user_message: str,
    context: list[dict[str, str]] | None = None,
    history: list[dict[str, str]] | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> str:
    """Send a message to Claude with assembled context.

    Args:
        user_message: The user's message text.
        context: List of context pieces from store.assemble_context().
        history: Optional conversation history as list of {"role": ..., "content": ...}.
        model: Claude model to use.
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's response text.
    """
    import anthropic

    client = get_client()
    system_prompt = format_system_prompt(context or [])

    messages: list[dict[str, str]] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        return response.content[0].text
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(f"Authentication failed — check your ANTHROPIC_API_KEY: {exc}") from exc
    except anthropic.APIError as exc:
        raise RuntimeError(f"Claude API error: {exc}") from exc


def stream_message(
    user_message: str,
    context: list[dict[str, str]] | None = None,
    history: list[dict[str, str]] | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
):
    """Stream a message response from Claude, yielding text chunks.

    Same interface as send_message but yields str chunks instead of
    returning a single string.
    """
    import anthropic

    client = get_client()
    system_prompt = format_system_prompt(context or [])

    messages: list[dict[str, str]] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    try:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(f"Authentication failed — check your ANTHROPIC_API_KEY: {exc}") from exc
    except anthropic.APIError as exc:
        raise RuntimeError(f"Claude API error: {exc}") from exc


# ── Async tool-use support ───────────────────────────────────────────

def get_async_client() -> Any:
    """Return an async Anthropic client instance."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The 'anthropic' package is not installed. "
            "Install it with: pip install anthropic"
        ) from exc
    return anthropic.AsyncAnthropic(api_key=get_api_key())


ToolExecutor = Callable[[str, dict], Awaitable[str]]


async def stream_message_with_tools(
    user_message: str,
    tool_executor: ToolExecutor,
    context: list[dict[str, str]] | None = None,
    history: list[dict[str, str]] | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> AsyncIterator[StreamEvent]:
    """Async generator that streams Claude responses with tool-use loop.

    Yields StreamEvent instances: TextChunkEvent, ToolCallEvent, DoneEvent.
    The tool_executor callback is called when Claude invokes a tool.
    """
    import anthropic

    client = get_async_client()
    system_prompt = format_system_prompt(context or [])

    # Tell Claude exactly what tools it has so it doesn't hallucinate others
    tool_note = (
        "\n\n---\n\n## Available Tools\n\n"
        "You have access to these tools — use ONLY these, no others:\n\n"
        "**Web tools:**\n"
        "- **web_search**: Search Google for current information.\n"
        "- **fetch_page**: Read the full content of a web page URL.\n\n"
        "**Context tools** (for managing the user's Knowl context files):\n"
        "- **list_context_files**: List available context files (global and/or project).\n"
        "- **read_context_file**: Read a context file's content by path.\n"
        "- **write_context_file**: Create or update a context file.\n"
        "- **delete_context_file**: Delete a context file by path.\n\n"
        "When the user asks you to save, remember, update, or modify information in their "
        "context, use the context tools. Always list files first to discover paths before "
        "reading or writing. When creating a new file, use filename + scope (not path)."
    )
    if system_prompt:
        system_prompt += tool_note
    else:
        system_prompt = tool_note.strip()

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": TOOLS,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    try:
        while True:
            response = await client.messages.create(**kwargs)

            # Collect text and tool-use blocks from the response
            text_parts = []
            tool_uses = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    yield TextChunkEvent(text=block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason == "tool_use" and tool_uses:
                # Append assistant message with full content
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool and collect results
                tool_results = []
                for tool_use in tool_uses:
                    yield ToolCallEvent(name=tool_use.name, input=tool_use.input)
                    result_text = await tool_executor(tool_use.name, tool_use.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results})
                kwargs["messages"] = messages
                # Loop to let Claude process tool results
                continue
            else:
                # end_turn — done
                full_text = "".join(text_parts)
                yield DoneEvent(full_text=full_text)
                break

    except anthropic.AuthenticationError as exc:
        raise RuntimeError(f"Authentication failed — check your ANTHROPIC_API_KEY: {exc}") from exc
    except anthropic.APIError as exc:
        raise RuntimeError(f"Claude API error: {exc}") from exc
