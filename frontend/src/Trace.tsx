import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { shortModelName } from "./agents";
import type { AgentRun } from "./db";
import { SUMMARIZER_ID } from "./stream";

interface TraceProps {
  agents: AgentRun[];
  task?: string;
}

// The synthesis chip is labelled "Chairman" in the UI; `provider` stays
// "summarizer" everywhere else (wire format, stored turns), so the rename is
// display-only and old turns pick it up too.
const chipName = (a: AgentRun) => (a.id === SUMMARIZER_ID ? "Chairman" : a.provider);

// The orchestration trace for one assistant turn: a strip of per-agent chips
// (status dot, provider, model, latency) above the synthesized answer. Chips
// pulse while queued, then stream their text in place as each branch produces
// it. Clicking any chip that has started expands that agent's raw response —
// including mid-stream, so a live branch can be watched as it writes.
export default function Trace({ agents, task }: TraceProps) {
  const [openId, setOpenId] = useState<string | null>(null);
  const open = agents.find((a) => a.id === openId && a.status !== "pending") ?? null;

  return (
    <div className="trace">
      <div className="trace__chips">
        {task && <span className="trace__task">{task.replace(/_/g, " ")}</span>}
        {agents.map((a) => (
          <button
            key={a.id}
            className={`chip chip--${a.status}${a.id === openId ? " chip--open" : ""}`}
            onClick={() => setOpenId((cur) => (cur === a.id ? null : a.id))}
            disabled={a.status === "pending"}
            aria-expanded={a.id === openId}
            title={
              a.status === "pending"
                ? `${chipName(a)} is running`
                : `View ${chipName(a)}'s raw response`
            }
          >
            <span className="chip__dot" />
            <span className="chip__name">{chipName(a)}</span>
            {a.model && (
              <span className="chip__model" title={a.model}>
                {shortModelName(a.model)}
              </span>
            )}
            {a.status === "pending" && <span className="chip__meta">running…</span>}
            {a.status === "streaming" && <span className="chip__meta">writing…</span>}
            {a.status === "stopped" && <span className="chip__meta">stopped</span>}
            {a.status === "error" && <span className="chip__meta">failed</span>}
            {a.status === "done" && a.latencyMs != null && (
              <span className="chip__meta">{(a.latencyMs / 1000).toFixed(1)}s</span>
            )}
          </button>
        ))}
      </div>

      {open && (
        <div className="trace__detail">
          {open.error ? (
            <div className="trace__error">{open.error}</div>
          ) : (
            <div className="markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {open.content || "(empty response)"}
              </ReactMarkdown>
            </div>
          )}
          <div className="trace__stats">
            {open.latencyMs != null && <span>{open.latencyMs} ms</span>}
            {open.promptTokens != null && <span>{open.promptTokens} tok in</span>}
            {open.completionTokens != null && <span>{open.completionTokens} tok out</span>}
          </div>
        </div>
      )}
    </div>
  );
}
