// Swappable summarizer interface. TextRank is the on-device baseline; an
// abstractive (cheap-model) summarizer can drop in behind the same interface
// later. The `id` is a cache-invalidation key — switching summarizers (or bumping
// a version) makes any cached summary with a different id stale (see context.ts).

import type { Message } from "./db";
import { textRankSummarizer } from "./textrank";

export interface SummarizeInput {
  messages: Message[]; // the older region, with pinned messages already excluded
  maxSentences: number;
  maxChars: number;
  signal?: AbortSignal;
}

export interface Summarizer {
  id: string;
  summarize(input: SummarizeInput): Promise<string>;
}

// Registry keyed by id so the worker can resolve a summarizer from a plain string
// (workers receive structured-cloned data, not live object references).
export const SUMMARIZERS: Record<string, Summarizer> = {
  [textRankSummarizer.id]: textRankSummarizer,
};

// The summarizer used for new compactions. Its id is written into the cache.
export const ACTIVE_SUMMARIZER: Summarizer = textRankSummarizer;
