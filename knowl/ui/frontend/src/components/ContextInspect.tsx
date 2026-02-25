import { useState, useEffect } from "react";
import * as api from "../api";

interface Props {
  project: string | null;
}

export default function ContextInspect({ project }: Props) {
  const [data, setData] = useState<api.InspectResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .inspectContext(project || undefined)
      .then(setData)
      .finally(() => setLoading(false));
  }, [project]);

  if (loading) return <div className="inspect-loading">Loading context...</div>;
  if (!data) return <div className="inspect-empty">No context assembled.</div>;

  return (
    <div className="context-inspect">
      <div className="inspect-header">
        <h3>Context Inspect</h3>
        <span className="inspect-meta">
          {data.files.length} files | {data.total_tokens} / {data.budget} tokens
        </span>
      </div>

      <div className="inspect-files">
        {data.files.map((f) => (
          <div key={f.source} className="inspect-file">
            <span className="inspect-source">{f.source}</span>
            <span className="inspect-tokens">{f.tokens}t</span>
          </div>
        ))}
      </div>

      <div className="inspect-prompt">
        <h4>System Prompt Preview</h4>
        <pre className="inspect-pre">{data.system_prompt}</pre>
      </div>
    </div>
  );
}
