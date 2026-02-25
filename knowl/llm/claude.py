"""Claude integration — send messages to Anthropic Claude with assembled context."""

from __future__ import annotations

import os
from typing import Any

from knowl.log import get_logger

logger = get_logger(__name__)


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
