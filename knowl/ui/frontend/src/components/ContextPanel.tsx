import { useState, useEffect } from "react";
import * as api from "../api";
import VoiceMicButton from "./VoiceMicButton";

interface Props {
  project: string | null;
  refreshKey: number;
  onFileClick: (path: string) => void;
  onRefresh: () => void;
}

export default function ContextPanel({
  project,
  refreshKey,
  onFileClick,
  onRefresh,
}: Props) {
  const [globalFiles, setGlobalFiles] = useState<api.ContextFile[]>([]);
  const [projectFiles, setProjectFiles] = useState<api.ContextFile[]>([]);
  const [adding, setAdding] = useState(false);
  const [newFileName, setNewFileName] = useState("");
  const [newFileScope, setNewFileScope] = useState<"global" | "project">(
    "project"
  );
  const [renamingPath, setRenamingPath] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  useEffect(() => {
    api.getGlobalFiles().then(setGlobalFiles);
    if (project) {
      api.getProjectFiles(project).then(setProjectFiles);
    } else {
      setProjectFiles([]);
    }
  }, [project, refreshKey]);

  const handleToggle = async (file: api.ContextFile) => {
    if (!project) return;
    const currentActive = projectFiles
      .filter((f) => f.active)
      .map((f) => f.name);
    const newActive = file.active
      ? currentActive.filter((n) => n !== file.name)
      : [...currentActive, file.name];
    await api.setActiveFiles(project, newActive);
    onRefresh();
  };

  const handleAdd = async () => {
    const name = newFileName.trim();
    if (!name) return;
    const scope = newFileScope === "global" ? "global" : project!;
    await api.createFile(name.endsWith(".md") ? name : `${name}.md`, scope);
    setNewFileName("");
    setAdding(false);
    onRefresh();
  };

  const handleRenameStart = (file: api.ContextFile) => {
    setRenamingPath(file.path);
    setRenameValue(file.name);
  };

  const handleRenameSubmit = async () => {
    if (!renamingPath || !renameValue.trim()) {
      setRenamingPath(null);
      return;
    }
    try {
      await api.renameFile(renamingPath, renameValue.trim());
      setRenamingPath(null);
      onRefresh();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleDelete = async (path: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete this file?")) return;
    await api.deleteFile(path);
    onRefresh();
  };

  return (
    <div className="context-panel">
      <h3>Context Files</h3>

      {globalFiles.length > 0 && (
        <div className="file-section">
          <h4>Global</h4>
          {globalFiles.map((f) => (
            <div key={f.path} className="file-item">
              <span className="file-check active-always" title="Always active">
                *
              </span>
              {renamingPath === f.path ? (
                <input
                  className="rename-input"
                  autoFocus
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={handleRenameSubmit}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleRenameSubmit();
                    if (e.key === "Escape") setRenamingPath(null);
                  }}
                />
              ) : (
                <span
                  className="file-name"
                  onClick={() => onFileClick(f.path)}
                  onDoubleClick={(e) => { e.stopPropagation(); handleRenameStart(f); }}
                  title="Click to edit, double-click to rename"
                >
                  {f.name}
                </span>
              )}
              <span className="file-tokens">{f.tokens}t</span>
              <button
                className="btn-icon"
                onClick={(e) => handleDelete(f.path, e)}
                title="Delete"
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}

      {project && (
        <div className="file-section">
          <h4>{project}</h4>
          {projectFiles.length === 0 && (
            <div className="empty-hint">No context files yet.</div>
          )}
          {projectFiles.map((f) => (
            <div key={f.path} className="file-item">
              <input
                type="checkbox"
                checked={f.active || false}
                onChange={() => handleToggle(f)}
                title={f.active ? "Deactivate" : "Activate"}
              />
              {renamingPath === f.path ? (
                <input
                  className="rename-input"
                  autoFocus
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={handleRenameSubmit}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleRenameSubmit();
                    if (e.key === "Escape") setRenamingPath(null);
                  }}
                />
              ) : (
                <span
                  className="file-name"
                  onClick={() => onFileClick(f.path)}
                  onDoubleClick={(e) => { e.stopPropagation(); handleRenameStart(f); }}
                  title="Click to edit, double-click to rename"
                >
                  {f.name}
                </span>
              )}
              <span className="file-tokens">{f.tokens}t</span>
              <button
                className="btn-icon"
                onClick={(e) => handleDelete(f.path, e)}
                title="Delete"
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}

      {adding ? (
        <form
          className="add-file-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleAdd();
          }}
        >
          <input
            autoFocus
            placeholder="filename.md"
            value={newFileName}
            onChange={(e) => setNewFileName(e.target.value)}
          />
          <VoiceMicButton
            onTranscript={(text) => setNewFileName(text.trim())}
          />
          {project && (
            <select
              value={newFileScope}
              onChange={(e) =>
                setNewFileScope(e.target.value as "global" | "project")
              }
            >
              <option value="project">{project}</option>
              <option value="global">Global</option>
            </select>
          )}
          <button type="submit" className="btn btn-sm">
            Add
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setAdding(false)}
          >
            Cancel
          </button>
        </form>
      ) : (
        <button className="btn btn-sm" onClick={() => setAdding(true)}>
          + Add File
        </button>
      )}
    </div>
  );
}
