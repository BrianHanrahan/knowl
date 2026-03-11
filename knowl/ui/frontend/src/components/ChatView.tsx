import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import * as api from "../api";
import VoiceMicButton from "./VoiceMicButton";

interface Props {
  project: string | null;
  refreshKey: number;
  onRefresh?: () => void;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ToolProposal {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  implementation: string;
  status: "pending" | "approved" | "rejected";
  showCode?: boolean;
}

export default function ChatView({ project, refreshKey, onRefresh }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [proposals, setProposals] = useState<ToolProposal[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    api
      .getHistory(project || undefined)
      .then((data) => setMessages(data.history));
  }, [project, refreshKey]);

  useEffect(scrollToBottom, [messages, streamingText, scrollToBottom]);

  const getFilePreview = useCallback((file: File): string | null => {
    if (file.type.startsWith("image/")) {
      return URL.createObjectURL(file);
    }
    return null;
  }, []);

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const addFiles = (newFiles: FileList | File[]) => {
    const arr = Array.from(newFiles);
    setFiles((prev) => {
      const combined = [...prev, ...arr];
      if (combined.length > 10) {
        alert("Maximum 10 files allowed");
        return combined.slice(0, 10);
      }
      return combined;
    });
  };

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg || streaming) return;

    const currentFiles = files;
    setInput("");
    setFiles([]);

    const displayContent = currentFiles.length > 0
      ? msg + "\n\nAttachments: " + currentFiles.map((f) => f.name).join(", ")
      : msg;
    setMessages((prev) => [...prev, { role: "user", content: displayContent }]);
    setStreaming(true);
    setStreamingText("");
    setToolStatus(null);

    const callbacks = {
      onChunk: (text: string) => {
        setToolStatus(null);
        setStreamingText((prev) => prev + text);
      },
      onToolCall: (name: string, input: Record<string, any>) => {
        const inp = input as Record<string, any>;
        if (name === "web_search") {
          setToolStatus(`Searching: "${inp.query}"...`);
        } else if (name === "fetch_page") {
          setToolStatus(`Reading: ${inp.url}...`);
        } else if (name === "list_context_files") {
          setToolStatus(`Listing context files...`);
        } else if (name === "read_context_file") {
          setToolStatus(`Reading: ${inp.path?.split("/").pop()}...`);
        } else if (name === "write_context_file") {
          const target = inp.path?.split("/").pop() || inp.filename || "file";
          setToolStatus(`Writing: ${target}...`);
        } else if (name === "delete_context_file") {
          setToolStatus(`Deleting: ${inp.path?.split("/").pop()}...`);
        } else {
          setToolStatus(`Running: ${name}...`);
        }
      },
      onToolProposal: (tool: Record<string, unknown>) => {
        setToolStatus(null);
        setProposals((prev) => [
          ...prev,
          { ...(tool as any), status: "pending", showCode: false },
        ]);
      },
      onDone: (fullText: string) => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: fullText },
        ]);
        setStreamingText("");
        setToolStatus(null);
        setStreaming(false);
      },
      onError: (error: string) => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${error}` },
        ]);
        setStreamingText("");
        setToolStatus(null);
        setStreaming(false);
      },
    };

    if (currentFiles.length > 0) {
      abortRef.current = api.streamChatWithFiles(
        msg,
        currentFiles,
        callbacks,
        project || undefined
      );
    } else {
      abortRef.current = api.streamChat(
        msg,
        callbacks,
        project || undefined
      );
    }
  };

  const handleProposalApprove = async (name: string) => {
    if (!project) return;
    try {
      await api.approveTool(project, name);
      setProposals((prev) =>
        prev.map((p) => (p.name === name ? { ...p, status: "approved" } : p))
      );
      onRefresh?.();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleProposalReject = async (name: string) => {
    if (!project) return;
    try {
      await api.rejectTool(project, name);
      setProposals((prev) =>
        prev.map((p) => (p.name === name ? { ...p, status: "rejected" } : p))
      );
      onRefresh?.();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const toggleProposalCode = (name: string) => {
    setProposals((prev) =>
      prev.map((p) =>
        p.name === name ? { ...p, showCode: !p.showCode } : p
      )
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

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  return (
    <div
      className={`chat-view ${dragOver ? "chat-dropzone" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
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
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}

        {proposals.map((p) => (
          <div key={p.name} className={`tool-proposal-card tool-proposal-${p.status}`}>
            <div className="tool-proposal-header">
              <span className="tool-proposal-icon">&#9881;</span>
              <strong>{p.name}</strong>
              {p.status !== "pending" && (
                <span className={`tool-proposal-badge badge-${p.status}`}>
                  {p.status}
                </span>
              )}
            </div>
            <div className="tool-proposal-desc">{p.description}</div>
            <button
              className="btn-icon tool-code-toggle"
              onClick={() => toggleProposalCode(p.name)}
            >
              {p.showCode ? "Hide Code" : "View Code"}
            </button>
            {p.showCode && (
              <pre className="tool-code">{p.implementation}</pre>
            )}
            {p.status === "pending" && (
              <div className="tool-proposal-actions">
                <button
                  className="btn btn-sm btn-primary"
                  onClick={() => handleProposalApprove(p.name)}
                >
                  Approve
                </button>
                <button
                  className="btn btn-sm"
                  onClick={() => handleProposalReject(p.name)}
                >
                  Reject
                </button>
              </div>
            )}
          </div>
        ))}

        {streaming && toolStatus && !streamingText && (
          <div className="chat-message chat-assistant">
            <div className="chat-role">Claude</div>
            <div className="chat-content tool-status">
              <span className="tool-spinner" />
              {toolStatus}
            </div>
          </div>
        )}

        {streaming && streamingText && (
          <div className="chat-message chat-assistant">
            <div className="chat-role">Claude</div>
            <div className="chat-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingText}</ReactMarkdown>
              <span className="streaming-cursor" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {files.length > 0 && (
        <div className="file-chips">
          {files.map((f, i) => {
            const preview = getFilePreview(f);
            return (
              <div key={`${f.name}-${i}`} className="file-chip">
                {preview ? (
                  <img src={preview} alt={f.name} className="file-chip-img" />
                ) : (
                  <span className="file-chip-icon">&#128196;</span>
                )}
                <span className="file-chip-name">{f.name}</span>
                <button
                  className="file-chip-remove"
                  onClick={() => removeFile(i)}
                  title="Remove file"
                >
                  &times;
                </button>
              </div>
            );
          })}
        </div>
      )}

      <div className="chat-input-area">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*,.txt,.md,.csv,.json,.pdf,.py,.xml,.js,.ts,.html,.css,.yaml,.yml,.toml,.sql,.go,.rs,.java,.c,.cpp,.rb,.sh"
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          className="file-attach-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={streaming}
          title="Attach files"
        >
          &#128206;
        </button>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
          disabled={streaming}
          rows={2}
        />
        <VoiceMicButton
          onTranscript={(text) => setInput((prev) => prev + text)}
          disabled={streaming}
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
