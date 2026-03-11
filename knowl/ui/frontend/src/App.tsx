import { useState, useEffect, useCallback } from "react";
import * as api from "./api";
import ProjectSelector from "./components/ProjectSelector";
import ContextPanel from "./components/ContextPanel";
import TokenBudget from "./components/TokenBudget";
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

  const handleFileClick = (path: string) => {
    setEditingFile(path);
    setView("editor");
  };

  const handleEditorClose = () => {
    setEditingFile(null);
    setView("chat");
    refresh();
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Knowl</h1>
        <ProjectSelector
          projects={projects}
          active={activeProject}
          onSwitch={handleSwitchProject}
          onCreate={handleCreateProject}
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

          {activeProject && (
            <TokenBudget project={activeProject} refreshKey={refreshKey} />
          )}

          <div className="sidebar-actions">
            <button
              className={`btn btn-sm ${view === "inspect" ? "btn-active" : ""}`}
              onClick={() => setView(view === "inspect" ? "chat" : "inspect")}
            >
              Inspect Context
            </button>
          </div>
        </aside>

        <main className="main-panel">
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
