import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  MAX_AGENTS,
  activeAgents,
  defaultRoster,
  emptyRoster,
  loadRoster,
  newAgent,
  rosterActive,
  saveRoster,
  shortModelName,
  toRequestAgents,
  toRequestSummarizer,
} from "./agents";
import type { FreeModel, Roster } from "./agents";

const MODELS: FreeModel[] = [
  { id: "openai/gpt-oss-20b:free", name: "gpt-oss", lab: "openai", context_length: 131072 },
  {
    id: "nvidia/nemotron-3-super-120b-a12b:free",
    name: "nemotron super",
    lab: "nvidia",
    context_length: 262144,
  },
  { id: "nvidia/nemotron-nano-9b-v2:free", name: "nano", lab: "nvidia", context_length: 128000 },
];

// A roster with `n` enabled agents on distinct models.
function rosterOf(n: number, overrides: Partial<Roster> = {}): Roster {
  return {
    ...emptyRoster(),
    enabled: true,
    agents: Array.from({ length: n }, (_, i) => newAgent(`model-${i}`)),
    ...overrides,
  };
}

// jsdom is not configured for these tests, so stand in for the browser globals
// the module touches.
beforeEach(() => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
  });
  let n = 0;
  vi.stubGlobal("crypto", { randomUUID: () => `uuid-${n++}` });
});

describe("shortModelName", () => {
  it("strips the lab prefix and the :free suffix", () => {
    expect(shortModelName("openai/gpt-oss-20b:free")).toBe("gpt-oss-20b");
    expect(shortModelName("nvidia/nemotron-nano-9b-v2:free")).toBe("nemotron-nano-9b-v2");
  });

  it("passes through an id with neither", () => {
    expect(shortModelName("local-model")).toBe("local-model");
  });
});

describe("defaultRoster", () => {
  it("seeds the backend's default pair, switched off", () => {
    const roster = defaultRoster(MODELS);

    expect(roster.enabled).toBe(false);
    expect(roster.agents.map((a) => a.model)).toEqual([
      "openai/gpt-oss-20b:free",
      "nvidia/nemotron-3-super-120b-a12b:free",
    ]);
    // Seeding must never change what a turn does until the user opts in.
    expect(rosterActive(roster)).toBe(false);
  });

  it("falls back to whatever the catalog offers when the defaults are gone", () => {
    const other: FreeModel[] = [
      { id: "google/gemma-4-31b-it:free", name: "gemma", lab: "google" },
      { id: "google/gemma-4-26b-a4b-it:free", name: "gemma moe", lab: "google" },
    ];

    expect(defaultRoster(other).agents.map((a) => a.model)).toEqual([
      "google/gemma-4-31b-it:free",
      "google/gemma-4-26b-a4b-it:free",
    ]);
  });

  it("survives an empty catalog without inventing agents", () => {
    expect(defaultRoster([]).agents).toEqual([]);
  });
});

describe("rosterActive", () => {
  it("is false while the master switch is off", () => {
    expect(rosterActive({ ...rosterOf(2), enabled: false })).toBe(false);
  });

  it("is false when every agent is disabled, so nothing is sent", () => {
    const roster = rosterOf(2);
    roster.agents.forEach((a) => (a.enabled = false));

    expect(rosterActive(roster)).toBe(false);
  });

  it("is false when the only agent has no model chosen", () => {
    expect(rosterActive({ ...rosterOf(1), agents: [newAgent("")] })).toBe(false);
  });

  it("is true with the switch on and one usable agent", () => {
    expect(rosterActive(rosterOf(1))).toBe(true);
  });
});

describe("toRequestAgents", () => {
  it("sends only enabled agents, in roster order", () => {
    const roster = rosterOf(3);
    roster.agents[1].enabled = false;

    expect(toRequestAgents(roster).map((a) => a.model)).toEqual(["model-0", "model-2"]);
  });

  it("routes every agent through the openrouter provider", () => {
    expect(toRequestAgents(rosterOf(2)).every((a) => a.provider === "openrouter")).toBe(true);
  });

  it("trims the system prompt and omits it when blank", () => {
    const roster = rosterOf(2);
    roster.agents[0].system = "  Be concise.  ";
    roster.agents[1].system = "   ";

    const [first, second] = toRequestAgents(roster);
    expect(first.system).toBe("Be concise.");
    expect(second.system).toBeUndefined();
  });

  it("falls back to the model name when the label was cleared", () => {
    const roster = rosterOf(1);
    roster.agents[0].model = "openai/gpt-oss-20b:free";
    roster.agents[0].name = "   ";

    expect(toRequestAgents(roster)[0].name).toBe("gpt-oss-20b");
  });
});

describe("toRequestSummarizer", () => {
  const withSummarizer = (n: number, model = "model-s") =>
    rosterOf(n, { summarizer: { enabled: true, model, system: "" } });

  it("is null when synthesis is switched off", () => {
    expect(toRequestSummarizer(rosterOf(2))).toBeNull();
  });

  it("is null below two agents, since there is nothing to merge", () => {
    expect(toRequestSummarizer(withSummarizer(1))).toBeNull();
  });

  it("is null when no summarizer model was chosen", () => {
    expect(toRequestSummarizer(withSummarizer(2, ""))).toBeNull();
  });

  it("is sent once two agents will run", () => {
    expect(toRequestSummarizer(withSummarizer(2))).toEqual({
      name: "summarizer",
      provider: "openrouter",
      model: "model-s",
      system: undefined,
    });
  });
});

describe("loadRoster", () => {
  it("returns null before anything was saved, so the caller can seed", () => {
    expect(loadRoster()).toBeNull();
  });

  it("round-trips a saved roster", () => {
    const roster = rosterOf(2);
    saveRoster(roster);

    expect(loadRoster()).toEqual(roster);
  });

  it("returns null for unparseable JSON instead of throwing", () => {
    localStorage.setItem("a1a.roster", "{not json");

    expect(loadRoster()).toBeNull();
  });

  it("returns null when the stored shape is wrong", () => {
    localStorage.setItem("a1a.roster", JSON.stringify({ enabled: true }));

    expect(loadRoster()).toBeNull();
  });

  it("repairs partial agent entries and drops modelless ones", () => {
    localStorage.setItem(
      "a1a.roster",
      JSON.stringify({
        enabled: true,
        agents: [{ model: "model-a" }, { name: "broken" }, null],
      })
    );

    const loaded = loadRoster()!;
    expect(loaded.agents).toHaveLength(1);
    expect(loaded.agents[0]).toMatchObject({
      model: "model-a",
      name: "model-a",
      system: "",
      enabled: true,
    });
    expect(loaded.agents[0].id).toBeTruthy();
    expect(loaded.summarizer).toEqual({ enabled: false, model: "", system: "" });
  });

  it("clamps a stored roster to the agent cap", () => {
    saveRoster(rosterOf(MAX_AGENTS + 3));

    expect(loadRoster()!.agents).toHaveLength(MAX_AGENTS);
  });
});

describe("activeAgents", () => {
  it("ignores the master switch — it reports the roster's own contents", () => {
    const roster = { ...rosterOf(2), enabled: false };

    expect(activeAgents(roster)).toHaveLength(2);
  });
});
