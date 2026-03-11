"""Register the Knowl MCP server in ~/.claude.json for Claude Code integration.

Usage: python -m knowl.mcp.setup [--project <name>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


CLAUDE_CONFIG = Path.home() / ".claude.json"


def setup(project: str | None = None) -> None:
    """Register Knowl MCP server in Claude Code's config."""
    # Load existing config
    config: dict = {}
    if CLAUDE_CONFIG.exists():
        try:
            config = json.loads(CLAUDE_CONFIG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Warning: {CLAUDE_CONFIG} is not valid JSON, creating fresh config")
            config = {}

    # Build MCP server entry
    args = ["-m", "knowl.mcp.server"]
    if project:
        args.extend(["--project", project])

    server_config = {
        "command": sys.executable,
        "args": args,
    }

    # Add to mcpServers
    config.setdefault("mcpServers", {})
    config["mcpServers"]["knowl"] = server_config

    # Write back
    CLAUDE_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Registered Knowl MCP server in {CLAUDE_CONFIG}")
    print(f"  Command: {sys.executable} {' '.join(args)}")
    if project:
        print(f"  Project: {project}")
    print("\nClaude Code will now have access to Knowl tools.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Register Knowl MCP server with Claude Code")
    parser.add_argument("--project", "-p", default=None, help="Default project name")
    args = parser.parse_args()
    setup(args.project)
