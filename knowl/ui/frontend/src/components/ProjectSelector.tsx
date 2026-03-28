import { useState } from "react";
import VoiceMicButton from "./VoiceMicButton";

interface Props {
  projects: string[];
  active: string | null;
  onSwitch: (name: string | null) => void;
  onCreate: (name: string) => void;
  onDelete: (name: string) => void;
}

export default function ProjectSelector({
  projects,
  active,
  onSwitch,
  onCreate,
  onDelete,
}: Props) {
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");

  const handleCreate = () => {
    const name = newName.trim();
    if (!name) return;
    onCreate(name);
    setNewName("");
    setCreating(false);
  };

  const handleDelete = () => {
    if (!active) return;
    if (window.confirm(`Delete project '${active}'? This cannot be undone.`)) {
      onDelete(active);
    }
  };

  return (
    <div className="project-selector">
      <select
        value={active || ""}
        onChange={(e) => onSwitch(e.target.value || null)}
      >
        <option value="">(no project)</option>
        {projects.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>

      {active && (
        <button
          className="btn btn-sm"
          onClick={handleDelete}
          title="Delete project"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            <path d="M10 11v6" />
            <path d="M14 11v6" />
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
          </svg>
        </button>
      )}

      {creating ? (
        <form
          className="create-project-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleCreate();
          }}
        >
          <input
            autoFocus
            placeholder="Project name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <VoiceMicButton
            onTranscript={(text) => setNewName(text.trim())}
          />
          <button type="submit" className="btn btn-sm">
            Create
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setCreating(false)}
          >
            Cancel
          </button>
        </form>
      ) : (
        <button className="btn btn-sm" onClick={() => setCreating(true)}>
          + New
        </button>
      )}
    </div>
  );
}
