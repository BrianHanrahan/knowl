import { useState } from "react";
import VoiceMicButton from "./VoiceMicButton";

interface Props {
  projects: string[];
  active: string | null;
  onSwitch: (name: string | null) => void;
  onCreate: (name: string) => void;
}

export default function ProjectSelector({
  projects,
  active,
  onSwitch,
  onCreate,
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
