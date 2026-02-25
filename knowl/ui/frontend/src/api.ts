/** API client for Knowl backend. */

const BASE = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// ── Projects ────────────────────────────────────────────────────────

export interface ProjectsResponse {
  projects: string[];
  active_project: string | null;
}

export const getProjects = () => request<ProjectsResponse>("/api/projects");

export const createProject = (name: string) =>
  request<{ name: string; path: string }>("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });

export const deleteProject = (name: string) =>
  request<{ deleted: string }>(`/api/projects/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });

// ── Config ──────────────────────────────────────────────────────────

export interface Config {
  active_project: string | null;
  llm?: { model?: string };
  voice?: Record<string, unknown>;
}

export const getConfig = () => request<Config>("/api/config");

export const updateConfig = (updates: {
  active_project?: string | null;
  model?: string;
}) =>
  request<Config>("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });

// ── Context files ───────────────────────────────────────────────────

export interface ContextFile {
  name: string;
  tokens: number;
  path: string;
  active?: boolean;
}

export const getGlobalFiles = () =>
  request<ContextFile[]>("/api/context/global");

export const getProjectFiles = (project: string) =>
  request<ContextFile[]>(`/api/context/project/${encodeURIComponent(project)}`);

export const readFile = (path: string) =>
  request<{ path: string; content: string }>(
    `/api/context/file?path=${encodeURIComponent(path)}`
  );

export const writeFile = (path: string, content: string) =>
  request<{ path: string; tokens: number }>("/api/context/file", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content }),
  });

export const createFile = (
  filename: string,
  scope: string,
  content?: string
) =>
  request<{ path: string; tokens: number }>("/api/context/file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, scope, content }),
  });

export const deleteFile = (path: string) =>
  request<{ deleted: string }>(
    `/api/context/file?path=${encodeURIComponent(path)}`,
    { method: "DELETE" }
  );

export const setActiveFiles = (project: string, files: string[]) =>
  request<{ project: string; active_files: string[] }>(
    `/api/context/active/${encodeURIComponent(project)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files }),
    }
  );

export const promoteFile = (project: string, filename: string) =>
  request<{ promoted: string; path: string }>("/api/context/promote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project, filename }),
  });

// ── Inspect ─────────────────────────────────────────────────────────

export interface InspectResponse {
  files: { source: string; tokens: number }[];
  total_tokens: number;
  budget: number;
  system_prompt: string;
}

export const inspectContext = (project?: string, budget?: number) => {
  const params = new URLSearchParams();
  if (project) params.set("project", project);
  if (budget) params.set("budget", String(budget));
  return request<InspectResponse>(`/api/inspect?${params}`);
};

export const getTokenBreakdown = (project: string) =>
  request<Record<string, number>>(
    `/api/tokens/${encodeURIComponent(project)}`
  );

// ── Chat ────────────────────────────────────────────────────────────

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export const getHistory = (project?: string) => {
  const params = project ? `?project=${encodeURIComponent(project)}` : "";
  return request<{ project: string | null; history: HistoryMessage[] }>(
    `/api/history${params}`
  );
};

export const clearHistory = (project?: string) => {
  const params = project ? `?project=${encodeURIComponent(project)}` : "";
  return request<{ cleared: boolean }>(`/api/history${params}`, {
    method: "DELETE",
  });
};

/** Send a chat message and receive SSE stream. Returns an AbortController. */
export function streamChat(
  message: string,
  callbacks: {
    onChunk: (text: string) => void;
    onDone: (fullText: string) => void;
    onError: (error: string) => void;
  },
  project?: string,
  model?: string
): AbortController {
  const controller = new AbortController();

  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, project, model }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(body.detail || res.statusText);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "chunk") {
              callbacks.onChunk(data.text);
            } else if (data.type === "done") {
              callbacks.onDone(data.full_text);
            } else if (data.type === "error") {
              callbacks.onError(data.error);
            }
          } catch {
            // skip malformed lines
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

// ── Promotions ──────────────────────────────────────────────────────

export interface PromotionSuggestion {
  filename: string;
  projects: string[];
  similarity: number;
  reason: string;
  conflict: boolean;
}

export const getPromotions = (threshold?: number) => {
  const params = threshold ? `?threshold=${threshold}` : "";
  return request<PromotionSuggestion[]>(`/api/promotions${params}`);
};

export const applyPromotion = (
  filename: string,
  sourceProject?: string,
  force?: boolean
) =>
  request<{ promoted: string; path: string }>("/api/promotions/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename,
      source_project: sourceProject,
      force: force || false,
    }),
  });
