import { useEffect, useRef, useState } from "react";
import {
  MAX_AGENTS,
  activeAgents,
  newAgent,
  shortModelName,
} from "./agents";
import type { AgentConfig, FreeModel, Roster } from "./agents";

// Collapse + height of the panel, remembered across reloads (mirrors the
// sidebar-collapse preference pattern in App.tsx).
const COLLAPSE_KEY = "a1a.agentsCollapsed";
const HEIGHT_KEY = "a1a.agentsHeight";
const MIN_HEIGHT = 150;
const DEFAULT_HEIGHT = 320;

interface AgentsPanelProps {
  roster: Roster;
  // The free-model catalog, fetched once by App — empty while it loads, or for
  // good if the fetch failed, in which case the pickers degrade to text inputs.
  models: FreeModel[];
  onChange: (roster: Roster) => void;
}

// Lab id -> display name for the model picker's <optgroup> labels.
const LAB_LABELS: Record<string, string> = {
  openai: "OpenAI",
  google: "Google",
  nvidia: "NVIDIA",
};

function labLabel(lab: string): string {
  return LAB_LABELS[lab] ?? lab;
}

// "262144" -> "262k", "1000000" -> "1M". Context lengths are the one number
// worth showing next to a model id, and only at a glance.
function formatContext(n?: number | null): string {
  if (!n) return "";
  if (n >= 1_000_000) return `${Math.round(n / 1_000_000)}M`;
  return `${Math.round(n / 1000)}k`;
}

// Group the flat catalog by lab, preserving the server's ordering within a lab.
function byLab(models: FreeModel[]): [string, FreeModel[]][] {
  const groups = new Map<string, FreeModel[]>();
  for (const m of models) {
    const list = groups.get(m.lab) ?? [];
    list.push(m);
    groups.set(m.lab, list);
  }
  return [...groups.entries()];
}

// The model dropdown, or a plain text input when the catalog could not be
// fetched — an offline backend should not make the roster uneditable.
function ModelPicker({
  models,
  value,
  onChange,
  ariaLabel,
  disabled = false,
}: {
  models: FreeModel[];
  value: string;
  onChange: (id: string) => void;
  ariaLabel: string;
  disabled?: boolean;
}) {
  if (models.length === 0) {
    return (
      <input
        className="agentcard__model agentcard__model--raw"
        value={value}
        placeholder="model id, e.g. openai/gpt-oss-20b:free"
        aria-label={ariaLabel}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  // A roster saved before a model was retired would otherwise silently switch
  // to the first option; keep the stale id selectable and visibly flagged.
  const known = models.some((m) => m.id === value);

  return (
    <select
      className="agentcard__model"
      value={value}
      aria-label={ariaLabel}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
    >
      {!known && value && <option value={value}>{shortModelName(value)} (unavailable)</option>}
      {byLab(models).map(([lab, group]) => (
        <optgroup key={lab} label={labLabel(lab)}>
          {group.map((m) => (
            <option key={m.id} value={m.id}>
              {shortModelName(m.id)}
              {m.context_length ? ` · ${formatContext(m.context_length)}` : ""}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

export default function AgentsPanel({ roster, models, onChange }: AgentsPanelProps) {
  // Which agents have their system prompt expanded. Collapsed by default so a
  // roster of six stays scannable.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // The whole panel collapses to just its header, and its height is draggable
  // against the chat list above it. Both remembered across reloads.
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(COLLAPSE_KEY) === "true"
  );
  const [height, setHeight] = useState(() => {
    const stored = Number(localStorage.getItem(HEIGHT_KEY));
    return Number.isFinite(stored) && stored >= MIN_HEIGHT ? stored : DEFAULT_HEIGHT;
  });
  // While a drag is live: the pointer's start Y and the panel height then.
  const dragRef = useRef<{ startY: number; startHeight: number } | null>(null);

  useEffect(() => {
    localStorage.setItem(COLLAPSE_KEY, String(collapsed));
  }, [collapsed]);
  useEffect(() => {
    localStorage.setItem(HEIGHT_KEY, String(height));
  }, [height]);

  function onResizeDown(e: React.PointerEvent<HTMLDivElement>) {
    dragRef.current = { startY: e.clientY, startHeight: height };
    e.currentTarget.setPointerCapture(e.pointerId);
  }
  function onResizeMove(e: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    // Dragging up grows the panel; leave the chat list at least ~180px.
    const max = Math.max(MIN_HEIGHT, window.innerHeight - 240);
    const next = drag.startHeight + (drag.startY - e.clientY);
    setHeight(Math.min(max, Math.max(MIN_HEIGHT, next)));
  }
  function onResizeUp(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragRef.current) return;
    dragRef.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
  }

  const active = activeAgents(roster);
  const atCapacity = roster.agents.length >= MAX_AGENTS;

  function patchAgent(id: string, patch: Partial<AgentConfig>) {
    onChange({
      ...roster,
      agents: roster.agents.map((a) => (a.id === id ? { ...a, ...patch } : a)),
    });
  }

  function addAgent() {
    if (atCapacity) return;
    // Default to a model not already in the roster — two identical agents would
    // just pay twice for the same answer.
    const used = new Set(roster.agents.map((a) => a.model));
    const pick = models.find((m) => !used.has(m.id)) ?? models[0];
    onChange({ ...roster, agents: [...roster.agents, newAgent(pick?.id ?? "")] });
  }

  function removeAgent(id: string) {
    onChange({ ...roster, agents: roster.agents.filter((a) => a.id !== id) });
  }

  function toggleExpanded(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  return (
    <section
      className={`agents${roster.enabled ? " agents--on" : ""}${
        collapsed ? " agents--collapsed" : ""
      }`}
      style={collapsed ? undefined : { height }}
    >
      {!collapsed && (
        <div
          className="agents__resize"
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize agents panel"
          onPointerDown={onResizeDown}
          onPointerMove={onResizeMove}
          onPointerUp={onResizeUp}
        />
      )}
      <header className="agents__head">
        <button
          className="agents__collapse"
          aria-expanded={!collapsed}
          aria-label={collapsed ? "Expand agents" : "Collapse agents"}
          title={collapsed ? "Expand" : "Collapse"}
          onClick={() => setCollapsed((c) => !c)}
        >
          <span
            className={`agents__chevron${collapsed ? "" : " agents__chevron--open"}`}
            aria-hidden="true"
          >
            ›
          </span>
          <span className="agents__title">Agents</span>
        </button>
        <div className="agents__headmeta">
          {!collapsed && (
            <span className="agents__sub">
              {roster.enabled
                ? `${active.length} running per turn`
                : "One model per turn"}
            </span>
          )}
          <button
            className={`toggle${roster.enabled ? " toggle--on" : ""}`}
            role="switch"
            aria-checked={roster.enabled}
            aria-label="Use a multi-agent line-up"
            title={
              roster.enabled
                ? "Multi-agent line-up is active — the single-model picker is off"
                : "Turn on to answer each turn with several agents instead of one model"
            }
            onClick={() => onChange({ ...roster, enabled: !roster.enabled })}
          >
            <span className="toggle__knob" />
          </button>
        </div>
      </header>

      {!collapsed && (
      <div className="agents__body">
        {roster.agents.length === 0 && (
          <p className="agents__empty">
            No agents yet. Add one to choose its model and give it a role.
          </p>
        )}

        {roster.agents.map((agent) => {
          const open = expanded.has(agent.id);
          return (
            <div
              key={agent.id}
              className={`agentcard${agent.enabled ? "" : " agentcard--off"}`}
            >
              <div className="agentcard__top">
                <button
                  className={`agentcard__dot${agent.enabled ? " agentcard__dot--on" : ""}`}
                  role="switch"
                  aria-checked={agent.enabled}
                  aria-label={`${agent.enabled ? "Disable" : "Enable"} ${agent.name}`}
                  title={agent.enabled ? "Enabled — click to skip" : "Disabled"}
                  onClick={() => patchAgent(agent.id, { enabled: !agent.enabled })}
                />
                <input
                  className="agentcard__name"
                  value={agent.name}
                  aria-label="Agent name"
                  placeholder="agent name"
                  onChange={(e) => patchAgent(agent.id, { name: e.target.value })}
                />
                <button
                  className="agentcard__x"
                  aria-label={`Remove ${agent.name}`}
                  title="Remove agent"
                  onClick={() => removeAgent(agent.id)}
                >
                  ✕
                </button>
              </div>

              <ModelPicker
                models={models}
                value={agent.model}
                ariaLabel={`Model for ${agent.name}`}
                onChange={(model) => patchAgent(agent.id, { model })}
              />

              <button
                className="agentcard__prompt-toggle"
                aria-expanded={open}
                onClick={() => toggleExpanded(agent.id)}
              >
                <span className={`agentcard__chevron${open ? " agentcard__chevron--open" : ""}`}>
                  ›
                </span>
                <span className="agentcard__prompt-preview">
                  {agent.system.trim()
                    ? agent.system.trim().replace(/\s+/g, " ")
                    : "Add a system prompt"}
                </span>
              </button>

              {open && (
                <textarea
                  className="agentcard__prompt"
                  value={agent.system}
                  rows={4}
                  placeholder="e.g. You are a critical reviewer. Challenge assumptions and flag weak evidence."
                  aria-label={`System prompt for ${agent.name}`}
                  onChange={(e) => patchAgent(agent.id, { system: e.target.value })}
                />
              )}
            </div>
          );
        })}

        <button
          className="agents__add"
          onClick={addAgent}
          disabled={atCapacity}
          title={
            atCapacity
              ? `${MAX_AGENTS} agents is the maximum — each one is a request per turn`
              : "Add another agent to the line-up"
          }
        >
          + Add agent
        </button>

        <div className="agents__synth">
          <div className="agents__synth-head">
            <div className="agents__titles">
              <span className="agents__synth-title">Chairman</span>
              <span className="agents__sub">
                {active.length < 2
                  ? "Needs 2+ agents"
                  : "Merges answers into one"}
              </span>
            </div>
            <button
              className={`toggle toggle--sm${roster.summarizer.enabled ? " toggle--on" : ""}`}
              role="switch"
              aria-checked={roster.summarizer.enabled}
              aria-label="Synthesize the agents' answers"
              onClick={() =>
                onChange({
                  ...roster,
                  summarizer: {
                    ...roster.summarizer,
                    enabled: !roster.summarizer.enabled,
                    // First enable with no model chosen: fall back to the
                    // smallest catalog entry rather than a blank select.
                    model: roster.summarizer.model || models[models.length - 1]?.id || "",
                  },
                })
              }
            >
              <span className="toggle__knob" />
            </button>
          </div>

          {/* The summarizer's picker stays visible even when off, just dimmed,
              so the option is always discoverable. The prompt box only unfolds
              once it is enabled, to keep an inactive panel compact. */}
          <div
            className={`agents__synth-body${
              roster.summarizer.enabled ? "" : " agents__synth-body--off"
            }`}
          >
            <ModelPicker
              models={models}
              value={roster.summarizer.model}
              ariaLabel="Chairman model"
              disabled={!roster.summarizer.enabled}
              onChange={(model) =>
                onChange({ ...roster, summarizer: { ...roster.summarizer, model } })
              }
            />
            {roster.summarizer.enabled && (
              <textarea
                className="agentcard__prompt"
                value={roster.summarizer.system}
                rows={3}
                placeholder="Optional — leave blank for the default synthesis prompt."
                aria-label="Chairman system prompt"
                onChange={(e) =>
                  onChange({
                    ...roster,
                    summarizer: { ...roster.summarizer, system: e.target.value },
                  })
                }
              />
            )}
          </div>
        </div>
      </div>
      )}
    </section>
  );
}
