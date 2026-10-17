// Web Worker that runs the (potentially O(n²)) summarization off the main thread
// so the UI never blocks. It resolves the summarizer by id from the registry and
// posts the summary back, keyed by the request id so the main thread can match it.

import { SUMMARIZERS, ACTIVE_SUMMARIZER } from "./summarizer";
import type { Message } from "./db";

interface WorkerRequest {
  reqId: number;
  summarizerId: string;
  messages: Message[];
  maxSentences: number;
  maxChars: number;
}

// `self` inside a module worker is the DedicatedWorkerGlobalScope. We avoid pulling
// in the "webworker" lib (it conflicts with the project's DOM lib) by treating it
// as a minimal message port — that's all we use here.
const ctx = self as unknown as {
  postMessage(message: unknown): void;
  addEventListener(type: "message", listener: (event: MessageEvent) => void): void;
};

ctx.addEventListener("message", async (event: MessageEvent) => {
  const { reqId, summarizerId, messages, maxSentences, maxChars } = event.data as WorkerRequest;
  try {
    const summarizer = SUMMARIZERS[summarizerId] ?? ACTIVE_SUMMARIZER;
    const summary = await summarizer.summarize({ messages, maxSentences, maxChars });
    ctx.postMessage({ reqId, summary });
  } catch (err) {
    ctx.postMessage({ reqId, error: String(err) });
  }
});
