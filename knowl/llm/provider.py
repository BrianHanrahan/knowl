"""
LLM provider abstraction.

Supports multiple backends so users can bring their own model:
- Anthropic Claude
- OpenAI / GPT
- Google Gemini
- Ollama (local)
- Any OpenAI-compatible endpoint (LM Studio, Together, etc.)
"""

import json
from pathlib import Path


PROVIDER_CONFIGS = {
    "anthropic": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "google": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "default_model": "gemini-2.0-flash",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "ollama": {
        "name": "Ollama (local)",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3",
        "api_key_env": "",
    },
    "openai_compatible": {
        "name": "OpenAI-compatible endpoint",
        "base_url": "",
        "default_model": "",
        "api_key_env": "",
    },
}


def get_provider_config(provider_name):
    """Return the default config for a known provider."""
    return PROVIDER_CONFIGS.get(provider_name)


def list_providers():
    """Return list of supported provider names."""
    return list(PROVIDER_CONFIGS.keys())
