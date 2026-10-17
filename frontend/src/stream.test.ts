import { describe, it, expect } from "vitest";
import { liveReply, SUMMARIZER_ID } from "./stream";
import type { AgentRun } from "./db";

const run = (p: Partial<AgentRun> & { id: string }): AgentRun => ({
  provider: p.id,
  status: "streaming",
  ...p,
});

describe("liveReply", () => {
  it("is empty before any text has streamed", () => {
    expect(liveReply([])).toBe("");
    expect(liveReply([run({ id: "a", status: "pending" })])).toBe("");
  });

  it("previews the first agent's streamed text", () => {
    expect(
      liveReply([run({ id: "a", content: "hello" }), run({ id: "b", content: "hi" })])
    ).toBe("hello");
  });

  it("skips a failed agent — its answer is not the one that wins", () => {
    expect(
      liveReply([
        run({ id: "a", status: "error", content: "half a th" }),
        run({ id: "b", content: "good answer" }),
      ])
    ).toBe("good answer");
  });

  it("switches to the summarizer once synthesis starts", () => {
    const agents = [
      run({ id: "a", status: "done", content: "agent answer" }),
      run({ id: SUMMARIZER_ID, content: "merged" }),
    ];
    expect(liveReply(agents)).toBe("merged");
  });

  it("shows nothing rather than stale agent text while the summarizer is pending", () => {
    // Swapping an agent's answer for the synthesis mid-turn would rewrite text
    // the user is already reading.
    const agents = [
      run({ id: "a", status: "done", content: "agent answer" }),
      run({ id: SUMMARIZER_ID, status: "pending" }),
    ];
    expect(liveReply(agents)).toBe("");
  });
});
