import { useState, useEffect } from "react";
import * as api from "../api";

interface Props {
  project: string;
  refreshKey: number;
  onRefresh: () => void;
}

export default function ToolsPanel({ project, refreshKey, onRefresh }: Props) {
  const [tools, setTools] = useState<api.CustomTool[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api.getTools(project).then((data) => setTools(data.tools));
  }, [project, refreshKey]);

  const pending = tools.filter((t) => t.status === "pending");
  const approved = tools.filter((t) => t.status === "approved");
  const disabled = tools.filter((t) => t.status === "disabled");

  const handleApprove = async (name: string) => {
    await api.approveTool(project, name);
    onRefresh();
  };

  const handleReject = async (name: string) => {
    await api.rejectTool(project, name);
    onRefresh();
  };

  const handleDisable = async (name: string) => {
    await api.disableTool(project, name);
    onRefresh();
  };

  const handleEnable = async (name: string) => {
    await api.enableTool(project, name);
    onRefresh();
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete tool "${name}"?`)) return;
    await api.deleteTool(project, name);
    onRefresh();
  };

  if (tools.length === 0) return null;

  return (
    <div className="tools-panel">
      <h3>Custom Tools</h3>

      {pending.length > 0 && (
        <div className="tool-section">
          <h4>Pending Approval</h4>
          {pending.map((t) => (
            <div key={t.name} className="tool-item tool-pending">
              <div className="tool-header">
                <span className="tool-name">{t.name}</span>
                <div className="tool-actions">
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={() => handleApprove(t.name)}
                  >
                    Approve
                  </button>
                  <button
                    className="btn btn-sm"
                    onClick={() => handleReject(t.name)}
                  >
                    Reject
                  </button>
                </div>
              </div>
              <div className="tool-desc">{t.description}</div>
              <button
                className="btn-icon tool-code-toggle"
                onClick={() =>
                  setExpanded(expanded === t.name ? null : t.name)
                }
              >
                {expanded === t.name ? "Hide Code" : "View Code"}
              </button>
              {expanded === t.name && (
                <pre className="tool-code">{t.implementation}</pre>
              )}
            </div>
          ))}
        </div>
      )}

      {approved.length > 0 && (
        <div className="tool-section">
          <h4>Active</h4>
          {approved.map((t) => (
            <div key={t.name} className="tool-item">
              <div className="tool-header">
                <input
                  type="checkbox"
                  checked
                  onChange={() => handleDisable(t.name)}
                  title="Disable tool"
                />
                <span className="tool-name">{t.name}</span>
                <button
                  className="btn-icon"
                  onClick={() => handleDelete(t.name)}
                  title="Delete"
                >
                  x
                </button>
              </div>
              <div className="tool-desc">{t.description}</div>
            </div>
          ))}
        </div>
      )}

      {disabled.length > 0 && (
        <div className="tool-section">
          <h4>Disabled</h4>
          {disabled.map((t) => (
            <div key={t.name} className="tool-item tool-disabled">
              <div className="tool-header">
                <input
                  type="checkbox"
                  checked={false}
                  onChange={() => handleEnable(t.name)}
                  title="Re-enable tool"
                />
                <span className="tool-name">{t.name}</span>
                <button
                  className="btn-icon"
                  onClick={() => handleDelete(t.name)}
                  title="Delete"
                >
                  x
                </button>
              </div>
              <div className="tool-desc">{t.description}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
