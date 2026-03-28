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
  const [preloadedContext, setPreloadedContext] = useState<{ source: string; tokens: number }[]>([]);
  const [loadedContext, setLoadedContext] = useState<{ path: string; tokens: number }[]>([]);
  const [contextExpanded, setContextExpanded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamingTextRef = useRef("");

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

  const handleStop = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    // Save partial response to history
    setStreamingText((partial) => {
      if (partial) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: partial + "\n\n*(interrupted)*" },
        ]);
      }
      return "";
    });
    setToolStatus(null);
    setStreaming(false);
  };

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg) return;

    if (msg === "/clear" || msg === "/reset") {
      setInput("");
      await handleClear();
      return;
    }

    // /btw interjection while streaming — abort and resend with added context
    if (streaming) {
      const partialText = streamingTextRef.current;
      handleStop();
      // Small delay to let abort settle
      await new Promise((r) => setTimeout(r, 50));
      // Now send the interjection as a new message with context
      const btw = msg.startsWith("/btw ") ? msg.slice(5) : msg;
      const fullMsg = partialText
        ? `(Continuing from partial response: "${partialText.slice(-200)}...")\n\n${btw}`
        : btw;
      setInput("");
      // Fall through to send fullMsg
      await doSend(fullMsg, []);
      return;
    }

    const currentFiles = files;
    setInput("");
    setFiles([]);
    await doSend(msg, currentFiles);
  };

  const doSend = async (msg: string, currentFiles: File[]) => {
    const displayContent = currentFiles.length > 0
      ? msg + "\n\nAttachments: " + currentFiles.map((f) => f.name).join(", ")
      : msg;
    setMessages((prev) => [...prev, { role: "user", content: displayContent }]);
    setStreaming(true);
    setStreamingText("");
    streamingTextRef.current = "";
    setToolStatus(null);

    const callbacks = {
      onChunk: (text: string) => {
        setToolStatus(null);
        setStreamingText((prev) => {
          const updated = prev + text;
          streamingTextRef.current = updated;
          return updated;
        });
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
      onContextSources: (sources: { source: string; tokens: number }[]) => {
        setPreloadedContext(sources);
        setLoadedContext([]);
      },
      onContextLoaded: (path: string, tokens: number) => {
        setLoadedContext((prev) => {
          if (prev.some((p) => p.path === path)) return prev;
          return [...prev, { path, tokens }];
        });
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
    setProposals([]);
    setPreloadedContext([]);
    setLoadedContext([]);
    setToolStatus(null);
    setStreamingText("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const autoGrow = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
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

      <div className="composer">
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

        {files.length > 0 && (
          <div className="composer-files">
            {files.map((f, i) => {
              const preview = getFilePreview(f);
              return (
                <div key={`${f.name}-${i}`} className="composer-file-chip">
                  {preview ? (
                    <img src={preview} alt={f.name} className="composer-file-thumb" />
                  ) : (
                    <svg className="composer-file-icon" width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                      <path d="M4 1h5.586a1 1 0 0 1 .707.293l2.414 2.414a1 1 0 0 1 .293.707V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1zm5 1H4v12h8V5h-2a1 1 0 0 1-1-1V2z"/>
                    </svg>
                  )}
                  <span className="composer-file-name">{f.name}</span>
                  <button
                    className="composer-file-remove"
                    onClick={() => removeFile(i)}
                    title="Remove"
                  >
                    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
                      <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        )}

        <div className="composer-row">
          <div className="composer-actions-left">
            <button
              className="composer-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={streaming}
              title="Attach files"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
              </svg>
            </button>
          </div>

          <textarea
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              autoGrow(e.target);
            }}
            onKeyDown={handleKeyDown}
            placeholder={streaming ? "Type to interject..." : "Message Knowl..."}
            rows={1}
          />

          <div className="composer-actions-right">
            <VoiceMicButton
              onTranscript={(text) => setInput((prev) => prev + text)}
              disabled={streaming}
            />
            {streaming && !input.trim() ? (
              <button
                className="composer-send stop"
                onClick={handleStop}
                title="Stop response"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              </button>
            ) : (
              <button
                className={`composer-send ${input.trim() ? "ready" : ""}`}
                onClick={handleSend}
                disabled={!input.trim()}
                title={streaming ? "Send interjection" : "Send message"}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M3.478 2.405a.75.75 0 0 0-.926.94l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.405z"/>
                </svg>
              </button>
            )}
          </div>
        </div>

        <div className="composer-hint">
          <span>Enter to send, Shift+Enter for newline</span>
        </div>
      </div>

      {(preloadedContext.length > 0 || loadedContext.length > 0) && (() => {
        const totalTokens =
          preloadedContext.reduce((s, c) => s + c.tokens, 0) +
          loadedContext.reduce((s, c) => s + c.tokens, 0);
        const totalFiles = preloadedContext.length + loadedContext.length;
        return (
          <div className="context-tracker">
            <div
              className="context-tracker-header"
              onClick={() => setContextExpanded((v) => !v)}
            >
              <span className={`toggle-arrow ${contextExpanded ? "expanded" : ""}`}>&#9656;</span>
              <span className="context-tracker-title">
                Context: {totalFiles} file{totalFiles !== 1 ? "s" : ""} &middot; {totalTokens}t
              </span>
              {streaming && loadedContext.length > 0 && (
                <span className="context-tracker-live">live</span>
              )}
            </div>
            {contextExpanded && (
              <div className="context-tracker-body">
                {preloadedContext.length > 0 && (
                  <div className="context-tracker-group">
                    <div className="context-tracker-group-label">Pre-loaded</div>
                    {preloadedContext.map((s) => (
                      <div key={s.source} className="context-tracker-item">
                        <span className="context-tracker-name">{s.source.split("/").pop()}</span>
                        <span className="context-tracker-tokens">{s.tokens}t</span>
                      </div>
                    ))}
                  </div>
                )}
                {loadedContext.length > 0 && (
                  <div className="context-tracker-group">
                    <div className="context-tracker-group-label">Loaded by Claude</div>
                    {loadedContext.map((s) => (
                      <div key={s.path} className="context-tracker-item">
                        <span className="context-tracker-name">{s.path.split("/").pop()}</span>
                        <span className="context-tracker-tokens">{s.tokens}t</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}
