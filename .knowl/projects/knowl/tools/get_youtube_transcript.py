#!/usr/bin/env python3
"""Auto-generated tool runner for "get_youtube_transcript".

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
_TOOL_NAME = "get_youtube_transcript"

if not _AUTH_KEY or not _AUTH_NONCE or not _AUTH_SIG:
    print(json.dumps({"error": "Authentication failed: missing credentials. This tool can only be run by the Knowl server."}))
    sys.exit(1)

_expected_sig = hmac.new(
    _AUTH_KEY.encode(),
    f"{_TOOL_NAME}:{_AUTH_NONCE}".encode(),
    hashlib.sha256,
).hexdigest()

if not hmac.compare_digest(_AUTH_SIG, _expected_sig):
    print(json.dumps({"error": "Authentication failed: invalid signature. This tool can only be run by the Knowl server."}))
    sys.exit(1)

# ── User implementation ─────────────────────────────────────────


import re
import json
import urllib.request
import urllib.parse
import html
import xml.etree.ElementTree as ET

def run(video_url: str, include_timestamps: bool = False) -> dict:
    # Extract video ID
    video_id = video_url.strip()
    patterns = [
        r'(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})',
    ]
    for pattern in patterns:
        m = re.search(pattern, video_url)
        if m:
            video_id = m.group(1)
            break

    if len(video_id) != 11:
        return {"error": f"Could not extract video ID from: {video_url}"}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
    }

    # Use youtubei API to get captions
    api_url = "https://www.youtube.com/youtubei/v1/get_transcript"
    payload = json.dumps({
        "context": {
            "client": {
                "clientName": "WEB",
                "clientVersion": "2.20240101.00.00",
                "hl": "en",
                "gl": "US"
            }
        },
        "params": _encode_params(video_id)
    }).encode("utf-8")

    req = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"Failed to call youtubei API: {e}"}

    # Parse transcript from response
    try:
        segments = (
            data["actions"][0]["updateEngagementPanelAction"]["content"]
            ["transcriptRenderer"]["content"]["transcriptSearchPanelRenderer"]
            ["body"]["transcriptSegmentListRenderer"]["initialSegments"]
        )
    except (KeyError, IndexError, TypeError):
        return {"error": "Could not parse transcript from API response. Video may not have a transcript."}

    lines = []
    for seg in segments:
        renderer = seg.get("transcriptSegmentRenderer", {})
        text_runs = renderer.get("snippet", {}).get("runs", [])
        text = html.unescape("".join(r.get("text", "") for r in text_runs)).strip()
        if not text:
            continue
        if include_timestamps:
            start_ms = int(renderer.get("startMs", 0))
            secs = start_ms // 1000
            mins, s = divmod(secs, 60)
            hrs, m = divmod(mins, 60)
            if hrs:
                ts = f"[{hrs}:{m:02}:{s:02}]"
            else:
                ts = f"[{m}:{s:02}]"
            lines.append(f"{ts} {text}")
        else:
            lines.append(text)

    if not lines:
        return {"error": "Transcript was empty."}

    return {"transcript": "\n".join(lines), "video_id": video_id, "line_count": len(lines)}


def _encode_params(video_id: str) -> str:
    """Encode the params field required by the get_transcript API."""
    import base64
    # Proto encoding for: field 1 (string video_id), field 2 (string "en")
    def encode_string_field(field_num: int, value: str) -> bytes:
        encoded = value.encode("utf-8")
        tag = (field_num << 3) | 2  # wire type 2 = length-delimited
        return _varint(tag) + _varint(len(encoded)) + encoded

    def _varint(n: int) -> bytes:
        buf = []
        while True:
            towrite = n & 0x7F
            n >>= 7
            if n:
                buf.append(towrite | 0x80)
            else:
                buf.append(towrite)
                break
        return bytes(buf)

    inner = encode_string_field(1, video_id)
    outer = encode_string_field(1, base64.b64encode(inner).decode()) + encode_string_field(2, "en")
    return base64.b64encode(outer).decode()


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
