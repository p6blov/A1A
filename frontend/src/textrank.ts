// Extractive summarization via TextRank (Mihalcea & Tarau, 2004).
//
// Sentences are nodes; edges are weighted by lexical overlap; PageRank scores
// centrality; the top-k sentences (re-ordered to their original positions for
// readability) form the summary. Pure and dependency-free so it can run inside a
// Web Worker. NOTE: TextRank assumes prose — on chat transcripts (short turns,
// lists, code) quality is modest, which is why the summarizer is swappable.

import type { Message } from "./db";
import type { Summarizer, SummarizeInput } from "./summarizer";

// A small English stopword set — enough to stop function words from dominating the
// overlap metric without pulling in a dependency.
const STOPWORDS = new Set(
  (
    "a an and are as at be but by for from has have he her his i if in into is it its " +
    "me my no nor not of on or our so that the their them then there these they this to " +
    "was we were what when which who will with would you your about above after again all " +
    "am any been before being below between both did do does doing down during each few " +
    "more most other over own same some such than too under up very out off"
  ).split(/\s+/)
);

// Render a list of messages into a plain transcript the summarizer reads over.
function renderTranscript(messages: Message[]): string {
  return messages
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
    .join("\n");
}

// Split text into sentences. Breaks on .!? boundaries and on newlines (chat turns
// and list items are often newline-delimited rather than punctuated).
export function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

// Content words of a sentence: lowercased, punctuation-stripped, stopwords removed.
function contentWords(sentence: string): string[] {
  return sentence
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOPWORDS.has(w));
}

// TextRank similarity: shared words normalized by the combined log-length of the
// two sentences. Guards against single-word sentences (log(1) = 0 ⇒ div-by-zero).
function similarity(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const setB = new Set(b);
  let shared = 0;
  const seen = new Set<string>();
  for (const w of a) {
    if (seen.has(w)) continue;
    seen.add(w);
    if (setB.has(w)) shared++;
  }
  if (shared === 0) return 0;
  const norm = Math.log(a.length + 1) + Math.log(b.length + 1);
  return norm === 0 ? 0 : shared / norm;
}

// Weighted PageRank by power iteration (damping 0.85). Returns a score per node.
function pageRank(weights: number[][], damping = 0.85, iterations = 40, tol = 1e-5): number[] {
  const n = weights.length;
  if (n === 0) return [];
  const outSum = weights.map((row) => row.reduce((s, w) => s + w, 0));
  let scores = new Array(n).fill(1 / n);
  const base = (1 - damping) / n;

  for (let it = 0; it < iterations; it++) {
    const next = new Array(n).fill(base);
    for (let j = 0; j < n; j++) {
      if (outSum[j] === 0) continue;
      const share = (damping * scores[j]) / outSum[j];
      for (let i = 0; i < n; i++) {
        if (i !== j && weights[j][i] > 0) next[i] += share * weights[j][i];
      }
    }
    let delta = 0;
    for (let i = 0; i < n; i++) delta += Math.abs(next[i] - scores[i]);
    scores = next;
    if (delta < tol) break;
  }
  return scores;
}

/**
 * Extractive summary of `text`: at most `maxSentences` of the highest-ranked
 * sentences, returned in their original order and capped at `maxChars`.
 */
export function textRankSummarize(text: string, maxSentences: number, maxChars: number): string {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return "";
  // Too few sentences to bother ranking — just return the (capped) text.
  if (sentences.length <= maxSentences) return cap(sentences.join(" "), maxChars);

  const words = sentences.map(contentWords);
  const weights = sentences.map((_, i) =>
    sentences.map((_, j) => (i === j ? 0 : similarity(words[i], words[j])))
  );
  const scores = pageRank(weights);

  const topIdx = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSentences)
    .map((x) => x.idx)
    .sort((a, b) => a - b); // restore original order for readability

  return cap(topIdx.map((i) => sentences[i]).join(" "), maxChars);
}

function cap(s: string, maxChars: number): string {
  return s.length <= maxChars ? s : `${s.slice(0, maxChars - 1).trimEnd()}…`;
}

// Baseline summarizer implementation.
export const textRankSummarizer: Summarizer = {
  id: "textrank-v1",
  async summarize({ messages, maxSentences, maxChars }: SummarizeInput): Promise<string> {
    return textRankSummarize(renderTranscript(messages), maxSentences, maxChars);
  },
};
