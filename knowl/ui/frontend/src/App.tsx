import { useState, useEffect, useCallback } from "react";
import * as api from "./api";
import ProjectSelector from "./components/ProjectSelector";
import ContextPanel from "./components/ContextPanel";
// TokenBudget removed — context tracking moved to ChatView
import ChatView from "./components/ChatView";
import ToolsPanel from "./components/ToolsPanel";
import ContextEditor from "./components/ContextEditor";
import ContextInspect from "./components/ContextInspect";
import SettingsPanel from "./components/SettingsPanel";

type View = "chat" | "editor" | "inspect" | "settings";

export default function App() {
  const [projects, setProjects] = useState<string[]>([]);
  const [activeProject, setActiveProject] = useState<string | null>(null);
  const [model, setModel] = useState("claude-sonnet-4-6");
  const [view, setView] = useState<View>("chat");
  const [editingFile, setEditingFile] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  useEffect(() => {
    api.getProjects().then((data) => {
      setProjects(data.projects);
      setActiveProject(data.active_project);
    });
    api.getConfig().then((cfg) => {
      setModel(cfg.llm?.model || "claude-sonnet-4-6");
    });
  }, []);

  const handleSwitchProject = async (name: string | null) => {
    await api.updateConfig({ active_project: name });
    setActiveProject(name);
    setView("chat");
    refresh();
  };

  const handleCreateProject = async (name: string) => {
    await api.createProject(name);
    const data = await api.getProjects();
    setProjects(data.projects);
    await handleSwitchProject(name);
  };

  const handleDeleteProject = async (name: string) => {
    await api.deleteProject(name);
    const data = await api.getProjects();
    setProjects(data.projects);
    if (activeProject === name) {
      await handleSwitchProject(null);
    }
  };

  const handleFileClick = (path: string) => {
    setEditingFile(path);
    setView("editor");
  };

  const handleEditorClose = () => {
    setEditingFile(null);
    setView("chat");
    refresh();
  };

  // Escape key returns to chat from any non-chat view
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && view !== "chat") {
        // Don't override Escape if user is typing in an input/textarea
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        if (view === "editor") {
          handleEditorClose();
        } else {
          setView("chat");
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [view]);

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Knowl</h1>
        <ProjectSelector
          projects={projects}
          active={activeProject}
          onSwitch={handleSwitchProject}
          onCreate={handleCreateProject}
          onDelete={handleDeleteProject}
        />
        <span className="model-badge">{model}</span>
        <button
          className={`btn btn-sm btn-icon ${view === "settings" ? "btn-active" : ""}`}
          onClick={() => setView(view === "settings" ? "chat" : "settings")}
          title="Settings"
        >
          &#9881;
        </button>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <ContextPanel
            project={activeProject}
            refreshKey={refreshKey}
            onFileClick={handleFileClick}
            onRefresh={refresh}
          />

          {activeProject && (
            <ToolsPanel
              project={activeProject}
              refreshKey={refreshKey}
              onRefresh={refresh}
            />
          )}
        </aside>

        <main className="main-panel">
          <nav className="view-tabs">
            <button
              className={`view-tab ${view === "chat" ? "view-tab-active" : ""}`}
              onClick={() => setView("chat")}
            >
              Chat
            </button>
            {editingFile && (
              <button
                className={`view-tab ${view === "editor" ? "view-tab-active" : ""}`}
                onClick={() => setView("editor")}
              >
                <span className="view-tab-label">{editingFile.split("/").pop()}</span>
                <span
                  className="view-tab-close"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleEditorClose();
                  }}
                >
                  ×
                </span>
              </button>
            )}
            <button
              className={`view-tab ${view === "inspect" ? "view-tab-active" : ""}`}
              onClick={() => setView("inspect")}
            >
              Inspect
            </button>
          </nav>

          <div style={{ display: view === "chat" ? "contents" : "none" }}>
            <ChatView project={activeProject} refreshKey={refreshKey} onRefresh={refresh} />
          </div>
          {view === "editor" && editingFile && (
            <ContextEditor filePath={editingFile} onClose={handleEditorClose} />
          )}
          {view === "inspect" && (
            <ContextInspect project={activeProject} />
          )}
          {view === "settings" && (
            <SettingsPanel onClose={() => setView("chat")} />
          )}
        </main>
      </div>
    </div>
  );
}
