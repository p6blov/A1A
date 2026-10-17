import { describe, it, expect } from "vitest";
import { splitSentences, textRankSummarize } from "./textrank";

describe("splitSentences", () => {
  it("splits on sentence punctuation and newlines", () => {
    expect(splitSentences("One. Two! Three?")).toEqual(["One.", "Two!", "Three?"]);
    expect(splitSentences("line a\nline b\n\nline c")).toEqual(["line a", "line b", "line c"]);
  });

  it("ignores empty/whitespace fragments", () => {
    expect(splitSentences("   \n\n  ")).toEqual([]);
    expect(splitSentences("")).toEqual([]);
  });
});

describe("textRankSummarize", () => {
  const paragraph = [
    "The orchestrator fans a prompt out to several providers in parallel.",
    "Each provider returns a candidate answer with token usage.",
    "The graph then finalizes a single reply for the user.",
    "Providers run concurrently to keep latency low.",
    "Token counts are recorded for telemetry and cost tracking.",
    "A failing provider is isolated so it does not break the others.",
    "The final answer is what the chat UI renders to the user.",
    "Latency is dominated by the slowest provider in the fan-out.",
  ].join(" ");

  it("returns at most maxSentences sentences in original order", () => {
    const out = textRankSummarize(paragraph, 3, 1000);
    expect(out.length).toBeGreaterThan(0);
    // Selected sentences keep their source order: each pick appears after the prior.
    const picks = out.split(/(?<=[.!?])\s+/);
    expect(picks.length).toBeLessThanOrEqual(3);
    const positions = picks.map((p) => paragraph.indexOf(p.trim()));
    const sorted = [...positions].sort((a, b) => a - b);
    expect(positions).toEqual(sorted);
  });

  it("respects the character cap", () => {
    const out = textRankSummarize(paragraph, 5, 40);
    expect(out.length).toBeLessThanOrEqual(40);
  });

  it("returns the (capped) text when there are too few sentences to rank", () => {
    expect(textRankSummarize("Only one sentence here.", 5, 1000)).toBe("Only one sentence here.");
  });

  it("handles empty input", () => {
    expect(textRankSummarize("", 5, 1000)).toBe("");
  });
});
