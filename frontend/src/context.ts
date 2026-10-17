// Client-side context compaction. Runs entirely in the browser before a chat is
// sent to /api/chat. The stored transcript is never altered — we only transform
// the OUTBOUND payload and cache a derived summary alongside the chat.
//
// Strategy:
//   • Budget-gated: below a token estimate, send the full history unchanged
//     (cheap chats stay lossless and byte-identical → cache-friendly).
//   • Pin, don't rank: always keep the first user message and any message flagged
//     as an instruction/constraint, verbatim.
//   • Freeze the summary: summarize messages[0..checkpoint) once and reuse it,
//     recomputing only when the checkpoint advances past a stride boundary, so the
//     sent prefix stays stable across turns.
//
// Outbound shape when compaction triggers:
//   [ ...pinnedOlderVerbatim, summaryMessage, ...recentWindow ]

import type { Chat, CompactionCache, Message } from "./db";
import { ACTIVE_SUMMARIZER } from "./summarizer";

export interface CompactionConfig {
  thresholdTokens: number; // skip compaction at/below this estimate
  recentBudgetTokens: number; // size of the verbatim recency window
  checkpointStride: number; // summary boundary snaps to multiples of this
  maxPinned: number; // cap on verbatim pinned messages
  summarySentences: number; // max sentences in the summary
  summaryMaxChars: number; // hard cap on summary length
}

export const DEFAULT_CONFIG: CompactionConfig = {
  thresholdTokens: 6000,
  recentBudgetTokens: 2500,
  checkpointStride: 8,
  maxPinned: 6,
  summarySentences: 6,
  summaryMaxChars: 1500,
};

// ~4 chars/token plus a small per-message role/overhead constant. Deliberately a
// cheap heuristic (no tokenizer dependency); keep thresholds conservative since it
// can under/over-count for code or CJK.
export function estimateTokens(messages: Pick<Message, "content">[]): number {
  let total = 0;
  for (const m of messages) total += Math.ceil(m.content.length / 4) + 4;
  return total;
}

// --- Pinning -------------------------------------------------------------------

const INSTRUCTION_PATTERNS: RegExp[] = [
  /\b(always|never|must(?: not)?|should(?:n't)?|do not|don't|avoid|ensure|make sure|required?|constraint|rule|important|remember|prefer)\b/i,
  /\bnote:/i,
  /\b(format|output|respond|reply|answer)\b[^.\n]{0,24}\b(as|in|using|with)\b/i,
  /\buse\b[^.\n]{0,30}\b(not|instead of)\b/i,
];

const DIRECTIVE_VERBS = new Set([
  "use", "make", "ensure", "avoid", "include", "exclude", "write", "return",
  "format", "respond", "keep", "follow", "add", "remove", "set", "do", "don't",
  "prefer", "limit", "wrap", "escape", "treat", "assume",
]);

// Structural cue: a list (≥2 bullet/numbered lines) whose items start with a
// directive verb reads as a constraints list.
function hasDirectiveList(content: string): boolean {
  let count = 0;
  for (const line of content.split(/\r?\n/)) {
    const m = line.match(/^\s*(?:[-*]|\d+[.)])\s+(\S+)/);
    if (m && DIRECTIVE_VERBS.has(m[1].toLowerCase().replace(/[^a-z']/g, ""))) count++;
  }
  return count >= 2;
}

export function isInstruction(content: string): boolean {
  return INSTRUCTION_PATTERNS.some((re) => re.test(content)) || hasDirectiveList(content);
}

// Indices (ascending) of messages to keep verbatim regardless of summary: the
// first user message (unconditional) plus instruction-flagged user messages,
// capped at maxPinned. Conservative by design — false negatives are caught by the
// first-message pin; the cap bounds prefix bloat.
export function selectPinnedIndices(messages: Message[], maxPinned: number): number[] {
  const firstUser = messages.findIndex((m) => m.role === "user");
  const picked = new Set<number>();
  if (firstUser >= 0) picked.add(firstUser);
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === "user" && isInstruction(messages[i].content)) picked.add(i);
  }
  // firstUser has the smallest index, so slicing the sorted list keeps it.
  return [...picked].sort((a, b) => a - b).slice(0, maxPinned);
}

// --- Compaction ----------------------------------------------------------------

export type SummarizeFn = (
  messages: Message[],
  maxSentences: number,
  maxChars: number
) => Promise<string>;

/**
 * Build the payload to send. Returns the (possibly compacted) message list and, on
 * a recompute turn, the new cache to persist. `cache` is undefined when nothing
 * changed (reuse) — callers should only write it when present.
 *
 * Pure aside from the injected `summarize` (the worker in production, a stub in
 * tests), so the whole policy is unit-testable without a Worker.
 */
export async function buildCompactedOutbound(
  messages: Message[],
  cache: CompactionCache | undefined,
  summarize: SummarizeFn,
  cfg: CompactionConfig = DEFAULT_CONFIG
): Promise<{ outbound: Message[]; cache?: CompactionCache }> {
  // 1. Budget gate — short chats pass through untouched.
  if (estimateTokens(messages) <= cfg.thresholdTokens) {
    return { outbound: messages };
  }

  // 2. Recency window: the largest suffix fitting recentBudgetTokens (≥ last msg).
  let recentStart = messages.length - 1;
  let acc = estimateTokens([messages[recentStart]]);
  while (recentStart > 0) {
    const t = estimateTokens([messages[recentStart - 1]]);
    if (acc + t > cfg.recentBudgetTokens) break;
    acc += t;
    recentStart -= 1;
  }

  // 3. Snap the boundary down to a stride so it only advances in discrete jumps —
  //    this is what keeps the summary (and the sent prefix) frozen between
  //    checkpoints. If it floors to 0 there's nothing older to summarize yet, so
  //    send the full history (compaction engages once the chat is comfortably long).
  const checkpoint = Math.floor(recentStart / cfg.checkpointStride) * cfg.checkpointStride;
  if (checkpoint <= 0) {
    return { outbound: messages };
  }

  // 4. Split older region into pinned (verbatim) and the rest (summarized).
  const pinned = new Set(selectPinnedIndices(messages, cfg.maxPinned).filter((i) => i < checkpoint));
  const pinnedOlder: Message[] = [];
  const toSummarize: Message[] = [];
  for (let i = 0; i < checkpoint; i++) {
    (pinned.has(i) ? pinnedOlder : toSummarize).push(messages[i]);
  }

  // 5. Frozen summary: reuse the cached one when the checkpoint and summarizer
  //    match; otherwise recompute and emit a fresh cache.
  let summary: string;
  let newCache: CompactionCache | undefined;
  if (cache && cache.summarizerId === ACTIVE_SUMMARIZER.id && cache.checkpoint === checkpoint) {
    summary = cache.summary;
  } else {
    summary = await summarize(toSummarize, cfg.summarySentences, cfg.summaryMaxChars);
    newCache = { checkpoint, summary, summarizerId: ACTIVE_SUMMARIZER.id };
  }

  // 6. Assemble: pinned verbatim, then the summary as a labeled user turn (no
  //    backend change needed — the flattener renders it naturally), then recency.
  const recent = messages.slice(checkpoint);
  const outbound = summary.trim()
    ? [...pinnedOlder, summaryMessage(summary), ...recent]
    : [...pinnedOlder, ...recent];
  return { outbound, cache: newCache };
}

function summaryMessage(summary: string): Message {
  return { role: "user", content: `Summary of earlier conversation:\n${summary}` };
}

// --- Worker plumbing -----------------------------------------------------------

let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, { resolve: (s: string) => void; reject: (e: unknown) => void }>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./compaction.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent) => {
      const { reqId, summary, error } = e.data as { reqId: number; summary?: string; error?: string };
      const p = pending.get(reqId);
      if (!p) return;
      pending.delete(reqId);
      if (error != null) p.reject(new Error(error));
      else p.resolve(summary ?? "");
    };
    worker.onerror = (e) => {
      for (const p of pending.values()) p.reject(e);
      pending.clear();
    };
  }
  return worker;
}

const summarizeViaWorker: SummarizeFn = (messages, maxSentences, maxChars) => {
  const w = getWorker();
  const reqId = ++seq;
  return new Promise<string>((resolve, reject) => {
    pending.set(reqId, { resolve, reject });
    w.postMessage({ reqId, summarizerId: ACTIVE_SUMMARIZER.id, messages, maxSentences, maxChars });
  });
};

/** Production entry point: compact `messages` for `chat`, summarizing in a Worker. */
export function prepareOutbound(
  chat: Chat,
  messages: Message[],
  cfg: CompactionConfig = DEFAULT_CONFIG
): Promise<{ outbound: Message[]; cache?: CompactionCache }> {
  return buildCompactedOutbound(messages, chat.compaction, summarizeViaWorker, cfg);
}
