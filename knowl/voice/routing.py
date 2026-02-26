"""Intent routing — classify transcribed text as chat, capture, or command."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from knowl.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """Classified user intent from transcribed speech."""
    kind: str          # "chat", "capture", or "command"
    text: str          # The cleaned-up text (prefix stripped)
    command: str = ""  # For "command" kind: the parsed command name
    args: dict[str, str] | None = None  # For "command" kind: parsed args


# ---------------------------------------------------------------------------
# Prefix detection
# ---------------------------------------------------------------------------

# Capture prefixes — trigger context-capture mode.
# Spoken as "note <text>", "remember <text>", "save <text>".
CAPTURE_PREFIXES = [
    r"^note\b[:\s]*",
    r"^remember\b[:\s]*",
    r"^save\b[:\s]*",
    r"^capture\b[:\s]*",
    r"^add to context\b[:\s]*",
]

# Command patterns — fixed vocabulary for voice-driven management.
COMMAND_PATTERNS: list[tuple[str, re.Pattern[str], list[str]]] = [
    # (command_name, regex, group_names)
    ("switch_project", re.compile(
        r"^(?:switch\s+(?:to\s+)?|use\s+)(?:project\s+)?(?P<name>.+)$", re.IGNORECASE
    ), ["name"]),
    ("list_projects", re.compile(
        r"^(?:list|show)\s+(?:all\s+)?projects$", re.IGNORECASE
    ), []),
    ("list_context", re.compile(
        r"^(?:list|show)\s+(?:all\s+)?context(?:\s+files)?$", re.IGNORECASE
    ), []),
    ("show_status", re.compile(
        r"^(?:show\s+)?status$", re.IGNORECASE
    ), []),
    ("create_project", re.compile(
        r"^(?:create|new)\s+project\s+(?P<name>.+)$", re.IGNORECASE
    ), ["name"]),
    ("promote", re.compile(
        r"^promote\s+(?P<file>\S+)(?:\s+from\s+(?P<project>\S+))?$", re.IGNORECASE
    ), ["file", "project"]),
    ("clear_history", re.compile(
        r"^clear\s+(?:chat\s+)?history$", re.IGNORECASE
    ), []),
    ("inspect_context", re.compile(
        r"^(?:inspect|preview)\s+context$", re.IGNORECASE
    ), []),
]


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def classify(text: str) -> Intent:
    """Classify transcribed text into an intent.

    Priority:
    1. Command — matches a fixed voice command pattern.
    2. Capture — starts with a capture prefix ("note:", "remember:", etc.).
    3. Chat — everything else goes to the conversation.
    """
    text = text.strip()
    if not text:
        return Intent(kind="chat", text="")

    # 1. Try command patterns
    for cmd_name, pattern, group_names in COMMAND_PATTERNS:
        m = pattern.match(text)
        if m:
            args = {name: m.group(name).strip() for name in group_names if m.group(name)}
            return Intent(kind="command", text=text, command=cmd_name, args=args)

    # 2. Try capture prefixes
    for prefix_re in CAPTURE_PREFIXES:
        m = re.match(prefix_re, text, re.IGNORECASE)
        if m:
            captured_text = text[m.end():].strip()
            if captured_text:
                return Intent(kind="capture", text=captured_text)

    # 3. Default to chat
    return Intent(kind="chat", text=text)
