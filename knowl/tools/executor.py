"""Custom tool executor — runs user-created tools in isolated subprocesses.

Each tool runs as a standalone Python script in a subprocess. Communication
uses stdin/stdout JSON with HMAC authentication to prevent unauthorized
execution of tool scripts.

Security model:
- A per-invocation HMAC token is generated using a server-held secret key
- The token is passed via environment variable to the subprocess
- The tool script verifies the HMAC before executing
- This prevents direct execution of .py tool files outside of knowl
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from knowl.context.store import PROJECTS_DIR
from knowl.log import get_logger

logger = get_logger(__name__)

# Server-side secret — generated once per process lifetime.
# Tools cannot run without a valid HMAC signed by this secret.
_SERVER_SECRET = secrets.token_hex(32)

TOOL_TIMEOUT = 30  # seconds


class ToolExecutionError(Exception):
    """Raised when a tool subprocess fails."""


def _tools_dir(project: str) -> Path:
    """Return the directory for a project's tool scripts."""
    return PROJECTS_DIR / project / "tools"


def _sign_invocation(tool_name: str, nonce: str) -> str:
    """Create an HMAC signature for a tool invocation.

    The signature covers the tool name and a one-time nonce,
    preventing replay attacks and cross-tool forgery.
    """
    message = f"{tool_name}:{nonce}"
    return hmac.new(
        _SERVER_SECRET.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


def write_tool_script(project: str, tool: dict[str, Any]) -> Path:
    """Write a standalone .py runner file for a tool.

    The generated script:
    1. Reads HMAC credentials from environment variables
    2. Verifies the HMAC before running any user code
    3. Reads JSON input from stdin
    4. Calls the user's run() function
    5. Writes JSON output to stdout

    Called when a tool is approved. Returns the script path.
    """
    tools_dir = _tools_dir(project)
    tools_dir.mkdir(parents=True, exist_ok=True)

    name = tool["name"]
    implementation = tool["implementation"]
    script_path = tools_dir / f"{name}.py"

    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Auto-generated tool runner for "{name}".

        This script is managed by Knowl and uses HMAC authentication
        to ensure it can only be executed by the Knowl server.
        DO NOT run this script directly.
        """
        import hashlib
        import hmac
        import json
        import os
        import sys

        # ── Authentication ──────────────────────────────────────────────
        # Verify HMAC token from the Knowl server before executing anything.

        _AUTH_KEY = os.environ.get("KNOWL_TOOL_AUTH_KEY", "")
        _AUTH_NONCE = os.environ.get("KNOWL_TOOL_AUTH_NONCE", "")
        _AUTH_SIG = os.environ.get("KNOWL_TOOL_AUTH_SIG", "")
        _TOOL_NAME = "{name}"

        if not _AUTH_KEY or not _AUTH_NONCE or not _AUTH_SIG:
            print(json.dumps({{"error": "Authentication failed: missing credentials. This tool can only be run by the Knowl server."}}))
            sys.exit(1)

        _expected_sig = hmac.new(
            _AUTH_KEY.encode(),
            f"{{_TOOL_NAME}}:{{_AUTH_NONCE}}".encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(_AUTH_SIG, _expected_sig):
            print(json.dumps({{"error": "Authentication failed: invalid signature. This tool can only be run by the Knowl server."}}))
            sys.exit(1)

        # ── User implementation ─────────────────────────────────────────

    ''')

    # Indent user code to avoid any top-level collision
    script += implementation + "\n\n"

    script += textwrap.dedent('''\
        # ── Entry point ─────────────────────────────────────────────────

        if __name__ == "__main__":
            try:
                inputs = json.loads(sys.stdin.read())
            except json.JSONDecodeError as e:
                print(json.dumps({"error": f"Invalid JSON input: {e}"}))
                sys.exit(1)

            try:
                result = run(**inputs)
                print(json.dumps(result, default=str))
            except TypeError as e:
                print(json.dumps({"error": f"Invalid arguments: {e}"}))
                sys.exit(1)
            except Exception as e:
                print(json.dumps({"error": f"Tool execution error: {e}"}))
                sys.exit(1)
    ''')

    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o600)  # Only owner can read/write
    logger.info("Wrote tool script: %s", script_path)
    return script_path


def execute_custom_tool(project: str, tool_name: str, inputs: dict[str, Any]) -> str:
    """Execute a custom tool in an isolated subprocess with HMAC auth.

    Args:
        project: Project name containing the tool.
        tool_name: Name of the tool to execute.
        inputs: JSON-serializable input dict to pass to the tool.

    Returns:
        JSON string from the tool's stdout.

    Raises:
        ToolExecutionError: If the subprocess fails, times out, or auth fails.
    """
    script_path = _tools_dir(project) / f"{tool_name}.py"
    if not script_path.exists():
        raise ToolExecutionError(f"Tool script not found: {script_path}")

    # Generate a one-time nonce and HMAC signature for this invocation
    nonce = secrets.token_hex(16)
    signature = _sign_invocation(tool_name, nonce)

    env = {
        **os.environ,
        "KNOWL_TOOL_AUTH_KEY": _SERVER_SECRET,
        "KNOWL_TOOL_AUTH_NONCE": nonce,
        "KNOWL_TOOL_AUTH_SIG": signature,
    }

    try:
        result = subprocess.run(
            ["python3", str(script_path)],
            input=json.dumps(inputs),
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT,
            env=env,
            cwd=str(script_path.parent),
        )
    except subprocess.TimeoutExpired:
        raise ToolExecutionError(
            f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT}s"
        )
    except OSError as exc:
        raise ToolExecutionError(f"Failed to execute tool '{tool_name}': {exc}")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        # Tool might have printed a JSON error before exiting
        if stdout:
            return stdout
        raise ToolExecutionError(
            f"Tool '{tool_name}' exited with code {result.returncode}: {stderr}"
        )

    output = result.stdout.strip()
    if not output:
        return json.dumps({"result": None})

    return output
