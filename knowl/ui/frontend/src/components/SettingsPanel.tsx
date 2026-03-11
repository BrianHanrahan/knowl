import { useEffect, useState } from "react";
import * as api from "../api";

interface Props {
  onClose: () => void;
}

export default function SettingsPanel({ onClose }: Props) {
  const [backend, setBackendState] = useState<"api" | "cli">("api");
  const [hasApiKey, setHasApiKey] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getBackendStatus().then((status) => {
      setBackendState(status.backend as "api" | "cli");
      setHasApiKey(status.has_api_key);
      setLoading(false);
    });
  }, []);

  const handleToggle = async (mode: "api" | "cli") => {
    setBackendState(mode);
    await api.updateConfig({ backend: mode });
  };

  if (loading) return <div className="settings-panel">Loading...</div>;

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h3>Settings</h3>
        <button className="btn btn-sm" onClick={onClose}>
          &times;
        </button>
      </div>

      <div className="settings-section">
        <label className="settings-label">LLM Backend</label>
        <div className="backend-toggle">
          <button
            className={`btn btn-sm ${backend === "api" ? "btn-active" : ""}`}
            onClick={() => handleToggle("api")}
          >
            API Key
          </button>
          <button
            className={`btn btn-sm ${backend === "cli" ? "btn-active" : ""}`}
            onClick={() => handleToggle("cli")}
          >
            Claude Code
          </button>
        </div>
        <p className="settings-hint">
          {backend === "api"
            ? hasApiKey
              ? "Using ANTHROPIC_API_KEY from environment."
              : "Warning: ANTHROPIC_API_KEY not set. Chat will fail."
            : "Using Claude Max subscription via Claude Agent SDK."}
        </p>
      </div>
    </div>
  );
}
