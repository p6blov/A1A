// Which streamed text the answer bubble shows while a turn is still running.
//
// Token deltas are appended straight onto the trace chips (`AgentRun.content`),
// so there is no separate buffer to keep in sync — the bubble is just a view
// over the chips. Once the turn's `final` event lands, the assistant message
// has real content and this preview is no longer consulted.

import type { AgentRun } from "./db";

export const SUMMARIZER_ID = "summarizer";

// The reply-in-progress: the summarizer's text once synthesis has started
// (its answer is the one that will win), otherwise the first agent that hasn't
// failed — the same ranking `finalize_node` uses on the server, since the chips
// are already in the task's configured agent order.
export function liveReply(agents: AgentRun[]): string {
  const summarizer = agents.find((a) => a.id === SUMMARIZER_ID);
  // Deliberately returns "" while the summarizer is still pending: swapping an
  // agent's answer for the synthesis mid-turn would rewrite text the user is
  // reading.
  if (summarizer) return summarizer.content ?? "";
  return agents.find((a) => a.status !== "error" && a.content)?.content ?? "";
}
