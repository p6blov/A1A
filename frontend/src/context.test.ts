import { describe, it, expect, vi } from "vitest";
import {
  estimateTokens,
  isInstruction,
  selectPinnedIndices,
  buildCompactedOutbound,
  type CompactionConfig,
} from "./context";
import type { Message } from "./db";

const CFG: CompactionConfig = {
  thresholdTokens: 60,
  recentBudgetTokens: 25,
  checkpointStride: 4,
  maxPinned: 6,
  summarySentences: 3,
  summaryMaxChars: 200,
};

// Uniform 16-char content ⇒ ceil(16/4)+4 = 8 tokens per message, so windowing math
// is deterministic. Content is non-instructional, so the only pin is the first user
// message.
function mk(n: number): Message[] {
  return Array.from({ length: n }, (_, i) => ({
    role: i % 2 === 0 ? "user" : "assistant",
    content: "x".repeat(16),
  })) as Message[];
}

describe("estimateTokens", () => {
  it("counts ~4 chars/token plus per-message overhead", () => {
    expect(estimateTokens([{ content: "abcd" }])).toBe(5); // ceil(4/4)+4
    expect(estimateTokens([{ content: "abcd" }, { content: "abcd" }])).toBe(10);
  });
});

describe("isInstruction", () => {
  it("flags directive/constraint phrasing", () => {
    expect(isInstruction("You must always use snake_case")).toBe(true);
    expect(isInstruction("Important: keep responses short")).toBe(true);
    expect(isInstruction("Note: avoid jargon")).toBe(true);
    expect(isInstruction("Respond in JSON only")).toBe(true);
    expect(isInstruction("- use tabs\n- avoid spaces")).toBe(true);
  });

  it("does not flag ordinary questions", () => {
    expect(isInstruction("What's the weather today?")).toBe(false);
    expect(isInstruction("Tell me a joke")).toBe(false);
  });
});

describe("selectPinnedIndices", () => {
  const msgs: Message[] = [
    { role: "user", content: "hi" },
    { role: "assistant", content: "yo" },
    { role: "user", content: "always use tabs" },
    { role: "assistant", content: "ok" },
    { role: "user", content: "thanks" },
  ];

  it("always pins the first user message plus instruction messages", () => {
    expect(selectPinnedIndices(msgs, 6)).toEqual([0, 2]);
  });

  it("respects the cap while keeping the first user message", () => {
    expect(selectPinnedIndices(msgs, 1)).toEqual([0]);
  });
});

describe("buildCompactedOutbound", () => {
  it("passes short chats through unchanged with no summarize call", async () => {
    const summarize = vi.fn(async () => "SUMMARY");
    const msgs = mk(5); // 5*8 = 40 tokens ≤ threshold
    const r = await buildCompactedOutbound(msgs, undefined, summarize, CFG);
    expect(r.outbound).toBe(msgs);
    expect(r.cache).toBeUndefined();
    expect(summarize).not.toHaveBeenCalled();
  });

  it("compacts long chats: pins first message, inserts summary, keeps recency", async () => {
    const summarize = vi.fn(async () => "SUMMARY");
    const msgs = mk(16); // 128 tokens > threshold
    const r = await buildCompactedOutbound(msgs, undefined, summarize, CFG);

    expect(summarize).toHaveBeenCalledTimes(1);
    expect(r.outbound.length).toBeLessThan(msgs.length);
    expect(r.outbound[0]).toBe(msgs[0]); // first user message pinned verbatim
    expect(r.outbound).toContain(msgs[msgs.length - 1]); // latest turn preserved
    expect(r.outbound[r.outbound.length - 1]).toBe(msgs[msgs.length - 1]);
    expect(r.outbound.some((m) => m.content.startsWith("Summary of earlier conversation:"))).toBe(true);
    expect(r.cache).toEqual({ checkpoint: 12, summary: "SUMMARY", summarizerId: "textrank-v1" });
  });

  it("freezes the summary: reuses cache within a stride, recomputes across one", async () => {
    const summarize = vi.fn(async () => "SUMMARY");

    const first = await buildCompactedOutbound(mk(16), undefined, summarize, CFG);
    expect(summarize).toHaveBeenCalledTimes(1);
    const cache = first.cache;

    // One more turn — checkpoint still snaps to 12 ⇒ reuse, no recompute.
    const reuse = await buildCompactedOutbound(mk(17), cache, summarize, CFG);
    expect(summarize).toHaveBeenCalledTimes(1);
    expect(reuse.cache).toBeUndefined();
    expect(reuse.outbound.some((m) => m.content.startsWith("Summary of earlier conversation:"))).toBe(true);

    // Enough turns to advance the checkpoint to 16 ⇒ recompute.
    const advance = await buildCompactedOutbound(mk(19), cache, summarize, CFG);
    expect(summarize).toHaveBeenCalledTimes(2);
    expect(advance.cache?.checkpoint).toBe(16);
  });
});
