import { describe, expect, it } from "vitest";
import { createSSEParser } from "./sse";

describe("createSSEParser", () => {
  it("parses a complete frame", () => {
    const parse = createSSEParser();
    const events = parse('event: agent\ndata: {"provider":"researcher"}\n\n');
    expect(events).toEqual([{ event: "agent", data: { provider: "researcher" } }]);
  });

  it("buffers frames split across chunk boundaries", () => {
    const parse = createSSEParser();
    expect(parse("event: fin")).toEqual([]);
    expect(parse('al\ndata: {"reply":')).toEqual([]);
    const events = parse('"done"}\n\n');
    expect(events).toEqual([{ event: "final", data: { reply: "done" } }]);
  });

  it("returns multiple frames arriving in one chunk, in order", () => {
    const parse = createSSEParser();
    const events = parse(
      'event: agent\ndata: {"n":1}\n\nevent: agent\ndata: {"n":2}\n\n'
    );
    expect(events.map((e) => e.data.n)).toEqual([1, 2]);
  });

  it("defaults the event name to message", () => {
    const parse = createSSEParser();
    expect(parse('data: {"x":1}\n\n')).toEqual([
      { event: "message", data: { x: 1 } },
    ]);
  });

  it("skips comments, blank frames, and malformed JSON without dying", () => {
    const parse = createSSEParser();
    expect(parse(":keepalive\n\n")).toEqual([]);
    expect(parse("event: broken\ndata: {not json}\n\n")).toEqual([]);
    // The stream still works afterwards.
    expect(parse('event: ok\ndata: {"a":1}\n\n')).toHaveLength(1);
  });
});
