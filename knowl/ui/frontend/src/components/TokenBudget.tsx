import { useState, useEffect } from "react";
import * as api from "../api";

interface Props {
  project: string;
  refreshKey: number;
}

export default function TokenBudget({ project, refreshKey }: Props) {
  const [tokens, setTokens] = useState<Record<string, number>>({});
  const budget = 8000;

  useEffect(() => {
    api.getTokenBreakdown(project).then(setTokens);
  }, [project, refreshKey]);

  const total = tokens.total || 0;
  const pct = Math.min(100, (total / budget) * 100);
  const entries = Object.entries(tokens).filter(([k]) => k !== "total");

  return (
    <div className="token-budget">
      <h4>Token Budget</h4>
      <div className="budget-bar">
        <div
          className="budget-fill"
          style={{
            width: `${pct}%`,
            backgroundColor: pct > 90 ? "#e74c3c" : pct > 70 ? "#f39c12" : "#2ecc71",
          }}
        />
      </div>
      <div className="budget-label">
        {total} / {budget} tokens ({pct.toFixed(0)}%)
      </div>
      {entries.length > 0 && (
        <div className="budget-breakdown">
          {entries.map(([name, count]) => (
            <div key={name} className="budget-entry">
              <span>{name}</span>
              <span>{count}t</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
