// The user's agent roster: which models answer each turn, and the system prompt
// each one runs under.
//
// This is a *client* setting. The backend still ships task presets in
// tasks.yaml, but when the roster is enabled the UI sends its own line-up on
// every request and that wins for the turn (see `_resolve_overrides` in
// app/api/server.py). Nothing here reaches the server unless `enabled` is true,
// so switching the master toggle off restores the preset behaviour exactly.
//
// Stored in localStorage rather than IndexedDB: it is one small object read
// synchronously at first paint, unlike the chat transcripts in db.ts.

const ROSTER_KEY = "a1a.roster";

// Every roster agent runs through the one provider that fronts the free models.
export const ROSTER_PROVIDER = "openrouter";

// Mirrors MAX_CUSTOM_AGENTS in app/config/settings.py. The server truncates
// past this anyway; the UI enforces it so the limit is visible, not surprising.
export const MAX_AGENTS = 6;

export interface FreeModel {
  id: string;
  name: string;
  lab: string;
  context_length?: number | null;
  max_completion_tokens?: number | null;
}

export interface AgentConfig {
  id: string; // stable across renders/reorders; not sent to the server
  name: string; // the agent's label in the trace; server falls back to the model id
  model: string;
  system: string;
  enabled: boolean;
}

export interface SummarizerConfig {
  enabled: boolean;
  model: string;
  system: string; // empty = use the backend's default synthesis prompt
}

export interface Roster {
  enabled: boolean;
  agents: AgentConfig[];
  summarizer: SummarizerConfig;
}

// What POST /api/chat{,/stream} expects in `agents` / `summarizer`.
export interface AgentOverride {
  name: string;
  provider: string;
  model: string;
  system?: string;
}

export function emptyRoster(): Roster {
  return {
    enabled: false,
    agents: [],
    summarizer: { enabled: false, model: "", system: "" },
  };
}

// A short, human label for a model id: "openai/gpt-oss-20b:free" -> "gpt-oss-20b".
export function shortModelName(modelId: string): string {
  const afterLab = modelId.includes("/") ? modelId.slice(modelId.indexOf("/") + 1) : modelId;
  return afterLab.replace(/:free$/, "");
}

// The strongest models to feature at the top of the composer's single-model
// picker, best first. Ids missing from the live catalog are simply skipped;
// the rest of the catalog follows so nothing is hidden. Curated for "free now" —
// when paid frontier lands, extend this list (and the backend catalog).
export const FEATURED_MODEL_IDS = [
  "nvidia/nemotron-3-ultra-550b-a55b:free",
  "nvidia/nemotron-3-super-120b-a12b:free",
  "google/gemma-4-31b-it:free",
  "openai/gpt-oss-20b:free",
];

// Reorder a flat catalog so the featured models lead, in FEATURED order, then
// everything else in the catalog's own order.
export function featuredModels(models: FreeModel[]): FreeModel[] {
  const byId = new Map(models.map((m) => [m.id, m]));
  const featured = FEATURED_MODEL_IDS.map((id) => byId.get(id)).filter(
    (m): m is FreeModel => Boolean(m)
  );
  const rest = models.filter((m) => !FEATURED_MODEL_IDS.includes(m.id));
  return [...featured, ...rest];
}

// A starter roster once the model catalog is known: the two models the backend's
// `default` task uses, so turning the toggle on changes nothing until edited.
export function defaultRoster(models: FreeModel[]): Roster {
  const pick = (id: string) => models.find((m) => m.id === id)?.id;
  const chosen = [
    pick("openai/gpt-oss-20b:free") ?? models[0]?.id,
    pick("nvidia/nemotron-3-super-120b-a12b:free") ?? models[1]?.id,
  ].filter((id): id is string => Boolean(id));

  return {
    enabled: false,
    agents: chosen.map((model) => newAgent(model)),
    summarizer: {
      enabled: false,
      model: pick("nvidia/nemotron-nano-9b-v2:free") ?? chosen[0] ?? "",
      system: "",
    },
  };
}

export function newAgent(model: string): AgentConfig {
  return {
    id: crypto.randomUUID(),
    name: shortModelName(model),
    model,
    system: "",
    enabled: true,
  };
}

// Read the stored roster. A corrupt or half-written value must not break the
// app on load, so anything unparseable resets to empty rather than throwing.
export function loadRoster(): Roster | null {
  const raw = localStorage.getItem(ROSTER_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<Roster>;
    if (!Array.isArray(parsed.agents)) return null;
    return {
      enabled: Boolean(parsed.enabled),
      agents: parsed.agents
        .filter((a): a is AgentConfig => Boolean(a && typeof a.model === "string"))
        .map((a) => ({
          id: a.id || crypto.randomUUID(),
          name: a.name || shortModelName(a.model),
          model: a.model,
          system: a.system ?? "",
          enabled: a.enabled !== false,
        }))
        .slice(0, MAX_AGENTS),
      summarizer: {
        enabled: Boolean(parsed.summarizer?.enabled),
        model: parsed.summarizer?.model ?? "",
        system: parsed.summarizer?.system ?? "",
      },
    };
  } catch {
    return null;
  }
}

export function saveRoster(roster: Roster): void {
  try {
    localStorage.setItem(ROSTER_KEY, JSON.stringify(roster));
  } catch (err) {
    // Quota or a disabled store: the roster still works for this session.
    console.error("saveRoster failed", err);
  }
}

// True when the roster should replace the task preset for a turn: the master
// switch is on and at least one enabled agent has a model. An enabled-but-empty
// roster would send nothing and silently fall back, so it does not count.
export function rosterActive(roster: Roster): boolean {
  return roster.enabled && activeAgents(roster).length > 0;
}

export function activeAgents(roster: Roster): AgentConfig[] {
  return roster.agents.filter((a) => a.enabled && a.model);
}

// The `agents` payload for a chat request. Names are deduped server-side too,
// but sending distinct ones keeps the trace chips readable.
export function toRequestAgents(roster: Roster): AgentOverride[] {
  return activeAgents(roster).map((a) => ({
    name: a.name.trim() || shortModelName(a.model),
    provider: ROSTER_PROVIDER,
    model: a.model,
    system: a.system.trim() || undefined,
  }));
}

// The `summarizer` payload, or null when synthesis is off. Synthesis only runs
// when two or more agents succeed, so it is pointless below two agents.
export function toRequestSummarizer(roster: Roster): AgentOverride | null {
  const { summarizer } = roster;
  if (!summarizer.enabled || !summarizer.model) return null;
  if (activeAgents(roster).length < 2) return null;
  return {
    name: "summarizer",
    provider: ROSTER_PROVIDER,
    model: summarizer.model,
    system: summarizer.system.trim() || undefined,
  };
}
