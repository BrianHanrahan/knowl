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
  attachments?: string[]; // filenames for display
}

interface ToolProposal {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  implementation: string;
  status: "pending" | "approved" | "rejected";
  showCode?: boolean;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ChatView({ project, refreshKey, onRefresh }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [proposals, setProposals] = useState<ToolProposal[]>([]);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
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

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg || streaming) return;

    const filesToSend = [...attachedFiles];
    const fileNames = filesToSend.map((f) => f.name);

    setInput("");
    setAttachedFiles([]);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: msg, attachments: fileNames.length ? fileNames : undefined },
    ]);
    setStreaming(true);
    setStreamingText("");
    setToolStatus(null);

    const callbacks = {
      onChunk: (text: string) => {
        setToolStatus(null);
        setStreamingText((prev) => prev + text);
      },
      onToolCall: (name: string, input: Record<string, unknown>) => {
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

    if (filesToSend.length > 0) {
      abortRef.current = api.streamChatWithFiles(
        msg,
        filesToSend,
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

  const handleAttachFiles = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    const newFiles = Array.from(files);
    setAttachedFiles((prev) => {
      const combined = [...prev, ...newFiles];
      if (combined.length > 10) {
        alert("Maximum 10 files allowed");
        return prev;
      }
      return combined;
    });
    // Reset input so same file can be re-selected
    e.target.value = "";
  };

  const removeAttachedFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
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
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              ) : (
                m.content
              )}
              {m.attachments && m.attachments.length > 0 && (
                <div className="chat-attachments">
                  {m.attachments.map((name, j) => (
                    <span key={j} className="attachment-badge">
                      <span className="attachment-icon">&#128206;</span>
                      {name}
                    </span>
                  ))}
                </div>
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

      <div className="chat-input-area">
        {attachedFiles.length > 0 && (
          <div className="attached-files-bar">
            {attachedFiles.map((f, i) => (
              <span key={i} className="attached-file-pill">
                <span className="attachment-icon">&#128206;</span>
                <span className="attached-file-name">{f.name}</span>
                <span className="attached-file-size">{formatFileSize(f.size)}</span>
                <button
                  className="attached-file-remove"
                  onClick={() => removeAttachedFile(i)}
                  title="Remove"
                >
                  &times;
                </button>
              </span>
            ))}
          </div>
        )}
        <div className="chat-input-row">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileChange}
            style={{ display: "none" }}
            accept="image/*,.pdf,.txt,.md,.csv,.json,.xml,.py,.js,.ts,.html,.css,.yaml,.yml,.toml,.sh,.go,.rs,.java,.c,.cpp,.sql,.rb,.swift,.kt,.lua"
          />
          <button
            className="btn btn-attach"
            onClick={handleAttachFiles}
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
    </div>
  );
}
