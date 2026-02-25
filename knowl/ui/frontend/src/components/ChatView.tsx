import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import * as api from "../api";

interface Props {
  project: string | null;
  refreshKey: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatView({ project, refreshKey }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    api
      .getHistory(project || undefined)
      .then((data) => setMessages(data.history));
  }, [project, refreshKey]);

  useEffect(scrollToBottom, [messages, streamingText, scrollToBottom]);

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg || streaming) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: msg }]);
    setStreaming(true);
    setStreamingText("");

    abortRef.current = api.streamChat(
      msg,
      {
        onChunk: (text) => {
          setStreamingText((prev) => prev + text);
        },
        onDone: (fullText) => {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: fullText },
          ]);
          setStreamingText("");
          setStreaming(false);
        },
        onError: (error) => {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `Error: ${error}` },
          ]);
          setStreamingText("");
          setStreaming(false);
        },
      },
      project || undefined
    );
  };

  const handleClear = async () => {
    await api.clearHistory(project || undefined);
    setMessages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-view">
      <div className="chat-header">
        <span>Chat{project ? ` — ${project}` : ""}</span>
        <button className="btn btn-sm" onClick={handleClear}>
          Clear History
        </button>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && !streaming && (
          <div className="chat-empty">
            Send a message to start chatting with Claude.
            {project && ` Context from "${project}" will be included.`}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`chat-message chat-${m.role}`}>
            <div className="chat-role">{m.role === "user" ? "You" : "Claude"}</div>
            <div className="chat-content">
              {m.role === "assistant" ? (
                <ReactMarkdown>{m.content}</ReactMarkdown>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}

        {streaming && streamingText && (
          <div className="chat-message chat-assistant">
            <div className="chat-role">Claude</div>
            <div className="chat-content">
              <ReactMarkdown>{streamingText}</ReactMarkdown>
              <span className="streaming-cursor" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
          disabled={streaming}
          rows={2}
        />
        <button
          className="btn btn-primary"
          onClick={handleSend}
          disabled={streaming || !input.trim()}
        >
          {streaming ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
