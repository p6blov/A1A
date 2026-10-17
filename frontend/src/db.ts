// On-device chat persistence backed by IndexedDB.
//
// The backend is stateless — it takes the full message history on every request —
// so persisting chats locally is all that's needed to reopen and continue any past
// conversation. One database, one object store keyed by chat id.

export type Role = "user" | "assistant";

// One agent's contribution to an assistant turn — the orchestration trace.
// An agent goes "pending" (queued) → "streaming" (its text is arriving token by
// token) → "done" / "error", or "stopped" if the user cancelled the turn before
// it resolved.
export type AgentStatus = "pending" | "streaming" | "done" | "error" | "stopped";

export interface AgentRun {
  id: string; // stable within a turn (provider name + index)
  provider: string;
  model?: string;
  status: AgentStatus;
  // The agent's raw answer (may differ from the synthesized reply). While
  // streaming this grows delta by delta; the agent's own end event replaces it
  // with the provider's authoritative copy.
  content?: string;
  error?: string;
  latencyMs?: number;
  promptTokens?: number | null;
  completionTokens?: number | null;
}

export interface Message {
  role: Role;
  content: string;
  // Assistant turns only: which agents produced this reply, and under which
  // task. Absent on user turns and on chats saved before this field existed.
  agents?: AgentRun[];
  task?: string;
}

// Derived compaction cache for long chats. This is NOT part of the transcript —
// it lets us reuse a "frozen" summary of the older messages [0..checkpoint) across
// turns instead of recomputing it every send (see context.ts). `messages` always
// holds the full conversation; this only affects what gets sent to the backend.
export interface CompactionCache {
  checkpoint: number; // summary covers messages[0..checkpoint)
  summary: string;
  summarizerId: string; // id/version mismatch ⇒ cache invalid, recompute
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  // Optional, derived. Absent for short chats that never trigger compaction.
  compaction?: CompactionCache;
}

const DB_NAME = "a1a";
const DB_VERSION = 1;
const STORE = "chats";

let dbPromise: Promise<IDBDatabase> | null = null;

// Open (and lazily memoize) the connection. The upgrade handler creates the store
// on first run / version bumps.
function getDB(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE, { keyPath: "id" });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

// Read every chat, most-recently-updated first (drives sidebar ordering).
export async function loadChats(): Promise<Chat[]> {
  try {
    const db = await getDB();
    const chats = await new Promise<Chat[]>((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).getAll();
      req.onsuccess = () => resolve(req.result as Chat[]);
      req.onerror = () => reject(req.error);
    });
    return chats.sort((a, b) => b.updatedAt - a.updatedAt);
  } catch (err) {
    console.error("loadChats failed; starting empty", err);
    return [];
  }
}

// Upsert a chat. Failures are logged but non-fatal — the chat still works in memory.
export async function saveChat(chat: Chat): Promise<void> {
  try {
    const db = await getDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(chat);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.error("saveChat failed", err);
  }
}

export async function deleteChat(id: string): Promise<void> {
  try {
    const db = await getDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.error("deleteChat failed", err);
  }
}
