"""Tests for knowl.llm.claude."""

from unittest.mock import MagicMock, patch

import pytest

from knowl.llm import claude


class TestGetApiKey:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        assert claude.get_api_key() == "sk-test-123"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            claude.get_api_key()

    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            claude.get_api_key()


class TestFormatSystemPrompt:
    def test_empty_context(self):
        assert claude.format_system_prompt([]) == ""

    def test_single_context(self):
        context = [{"role": "system", "content": "I am a developer.", "source": "global/identity.md"}]
        result = claude.format_system_prompt(context)
        assert "## global/identity.md" in result
        assert "I am a developer." in result

    def test_multiple_context(self):
        context = [
            {"role": "system", "content": "Identity content", "source": "global/identity.md"},
            {"role": "system", "content": "Project content", "source": "project/architecture.md"},
        ]
        result = claude.format_system_prompt(context)
        assert "## global/identity.md" in result
        assert "## project/architecture.md" in result
        assert "---" in result  # separator


class TestSendMessage:
    @patch("knowl.llm.claude.get_client")
    def test_sends_correct_structure(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude!")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        context = [{"role": "system", "content": "I am a developer.", "source": "global/identity.md"}]
        result = claude.send_message("Hello", context=context)

        assert result == "Hello from Claude!"
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert "## global/identity.md" in call_kwargs["system"]

    @patch("knowl.llm.claude.get_client")
    def test_with_history(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        claude.send_message("Follow up", history=history)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert len(call_kwargs["messages"]) == 3
        assert call_kwargs["messages"][-1] == {"role": "user", "content": "Follow up"}

    @patch("knowl.llm.claude.get_client")
    def test_no_system_prompt_when_no_context(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        claude.send_message("Hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs
