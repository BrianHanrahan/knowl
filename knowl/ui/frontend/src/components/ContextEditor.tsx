import { useState, useEffect } from "react";
import * as api from "../api";

interface Props {
  filePath: string;
  onClose: () => void;
}

export default function ContextEditor({ filePath, onClose }: Props) {
  const [content, setContent] = useState("");
  const [original, setOriginal] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .readFile(filePath)
      .then((data) => {
        setContent(data.content);
        setOriginal(data.content);
      })
      .catch((err) => setError(err.message));
  }, [filePath]);

  const dirty = content !== original;

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await api.writeFile(filePath, content);
      setOriginal(content);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "s" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSave();
    }
  };

  const filename = filePath.split("/").pop() || filePath;

  return (
    <div className="context-editor">
      <div className="editor-header">
        <span className="editor-title">
          {filename}
          {dirty && <span className="dirty-marker"> (unsaved)</span>}
        </span>
        <div className="editor-actions">
          <button
            className="btn btn-primary btn-sm"
            onClick={handleSave}
            disabled={saving || !dirty}
          >
            {saving ? "Saving..." : "Save"}
          </button>
          <button className="btn btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      {error && <div className="editor-error">{error}</div>}

      <textarea
        className="editor-textarea"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        onKeyDown={handleKeyDown}
        spellCheck={false}
      />
    </div>
  );
}
