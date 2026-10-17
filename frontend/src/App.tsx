import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Sidebar from "./Sidebar";
import Trace from "./Trace";
import AgentsPanel from "./AgentsPanel";
import { loadChats, saveChat, deleteChat as deleteChatFromDB } from "./db";
import type { AgentRun, Chat, Message } from "./db";
import {
  defaultRoster,
  emptyRoster,
  featuredModels,
  loadRoster,
  ROSTER_PROVIDER,
  rosterActive,
  saveRoster,
  shortModelName,
  toRequestAgents,
  toRequestSummarizer,
} from "./agents";
import type { FreeModel, Roster } from "./agents";
import { prepareOutbound } from "./context";
import { readSSE } from "./sse";
import { liveReply, SUMMARIZER_ID } from "./stream";

const SIDEBAR_PREF_KEY = "a1a.sidebarCollapsed";
const THEME_PREF_KEY = "a1a.theme";
// Which single model answers a turn when the multi-agent line-up is off.
const MODEL_PREF_KEY = "a1a.model";

type Theme = "light" | "dark";

// One entry of the stream's `start` line-up: an agent's role name (falls back to
// the provider), its provider, and the resolved model. The same shape describes
// the turn's summarizer when synthesis runs.
interface TaskAgentInfo {
  name?: string;
  provider: string;
  model?: string | null;
}

// Per-agent result inside the POST /api/chat response.
interface ApiAgentResult {
  provider: string;
  model?: string | null;
  content?: string;
  error?: string | null;
  latency_ms?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
}

function newChatObject(): Chat {
  const now = Date.now();
  return { id: crypto.randomUUID(), title: "New chat", messages: [], createdAt: now, updatedAt: now };
}

// Derive a chat title from the first user message (~40 chars).
function titleFromMessage(text: string): string {
  const trimmed = text.trim().replace(/\s+/g, " ");
  return trimmed.length > 40 ? `${trimmed.slice(0, 40)}…` : trimmed;
}

export default function App() {
  // Chats are persisted on-device (IndexedDB). The active chat's messages render on
  // the right; the sidebar lists all chats on the left.
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  // True once the stream's `start` event appended the assistant message (its
  // own chips render from then on, so the placeholder strip hides).
  const [streamStarted, setStreamStarted] = useState(false);
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_PREF_KEY) === "true"
  );
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem(THEME_PREF_KEY) as Theme | null) ?? "dark"
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  // The single model that answers a turn when the multi-agent line-up is off.
  // Empty until the catalog loads and seeds it (or a stored choice restores it).
  const [model, setModel] = useState(() => localStorage.getItem(MODEL_PREF_KEY) ?? "");
  // The user's own agent line-up (sidebar panel). Read synchronously so the
  // panel paints with the stored roster; seeded from the catalog on first run.
  const [roster, setRoster] = useState<Roster>(() => loadRoster() ?? emptyRoster());
  const [rosterSeeded, setRosterSeeded] = useState(() => loadRoster() !== null);
  // The free-model catalog, fetched once here and handed to the panel. Both the
  // seeding below and the panel's picker need it, and it is one upstream call.
  const [models, setModels] = useState<FreeModel[]>([]);
  const endRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  // Lets the Stop button abort the in-flight stream. Cleared when a turn ends.
  const abortRef = useRef<AbortController | null>(null);
  // Re-entrancy guard for send(). The `loading` state also disables the
  // composer, but state flushes are async — two near-simultaneous submits
  // (Enter + click, programmatic double-fire) could both read loading=false
  // and start two streams patching the same in-flight message. A ref flips
  // synchronously, so the second attempt is dropped no matter the timing.
  const sendingRef = useRef(false);

  const activeChat = chats.find((c) => c.id === activeId) ?? null;
  const messages = activeChat?.messages ?? [];
  // When the multi-agent line-up is on and non-empty it answers the turn; the
  // single-model picker steps aside.
  const customLineup = rosterActive(roster);
  // The synthesizer that will run this turn, or null — only in multi-agent mode
  // with 2+ agents. Used to preview and pre-place the summarizer chip.
  const pendingSummarizer = customLineup ? toRequestSummarizer(roster) : null;

  // The line-up a turn is about to run, as `{name, model}` pairs — the roster
  // when it is active, otherwise the single selected model.
  const lineup = customLineup
    ? toRequestAgents(roster).map((a) => ({ name: a.name, model: a.model }))
    : model
      ? [{ name: shortModelName(model), model }]
      : [];

  // While a turn is in flight, show that line-up as pending chips. The stream's
  // `start` event replaces them with the server's authoritative list. When
  // synthesis will run, its chip sits pending beside the agents from the start.
  const pendingAgents: AgentRun[] = lineup.map((a, i) => ({
    id: `${a.name}-${i}`,
    provider: a.name,
    model: a.model ?? undefined,
    status: "pending",
  }));
  if (pendingSummarizer) {
    pendingAgents.push({
      id: SUMMARIZER_ID,
      provider: "summarizer",
      model: pendingSummarizer.model,
      status: "pending",
    });
  }

  // Load persisted chats on mount; select the most recent, or start a fresh one.
  useEffect(() => {
    loadChats().then((loaded) => {
      if (loaded.length > 0) {
        setChats(loaded);
        setActiveId(loaded[0].id);
      } else {
        const fresh = newChatObject();
        setChats([fresh]);
        setActiveId(fresh.id);
      }
    });
  }, []);

  // Remember the chosen single model across reloads.
  useEffect(() => {
    if (model) localStorage.setItem(MODEL_PREF_KEY, model);
  }, [model]);

  // The free-model catalog, fetched once for the whole app. It feeds the
  // composer's single-model picker and the panel's per-agent pickers, seeds the
  // composer model on first run, and — first run only — seeds the roster with
  // the backend's default pair (still switched off, since seeding must not
  // change what a turn does). A stored choice wins; a failed fetch leaves the
  // pickers to fall back to free-text model ids.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/models")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((loaded: FreeModel[]) => {
        if (cancelled) return;
        setModels(loaded);
        // Seed (or repair a stale) single-model choice from the featured list.
        setModel((cur) =>
          cur && loaded.some((m) => m.id === cur)
            ? cur
            : featuredModels(loaded)[0]?.id ?? cur
        );
        // `rosterSeeded` is the mount-time answer to "was one already stored?",
        // which is exactly the question here — only a first run gets seeded.
        if (!rosterSeeded && loaded.length > 0) setRoster(defaultRoster(loaded));
      })
      .catch((err) => console.error("failed to load the model catalog", err))
      .finally(() => !cancelled && setRosterSeeded(true));
    return () => {
      cancelled = true;
    };
    // Deliberately mount-only: re-running on `rosterSeeded` would re-fetch the
    // catalog the moment seeding completes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist the roster, but not the empty pre-seed state — writing that would
  // make `loadRoster()` return non-null and suppress seeding on the next load.
  useEffect(() => {
    if (rosterSeeded) saveRoster(roster);
  }, [roster, rosterSeeded]);

  // Follow the conversation as it grows — but only when the user is already at
  // the bottom. Streaming re-renders on every delta, so unconditionally
  // scrolling here would yank the view back down while they scroll up to read.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight > 120) return;
    // "auto" mid-stream: a smooth scroll can't keep up with delta-rate updates.
    endRef.current?.scrollIntoView({ behavior: loading ? "auto" : "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_PREF_KEY, String(collapsed));
  }, [collapsed]);

  // Apply the chosen theme to the document root (CSS reads [data-theme]) and
  // persist it so the choice survives reloads.
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_PREF_KEY, theme);
  }, [theme]);

  // Close the settings dropdown on an outside click or the Escape key.
  useEffect(() => {
    if (!settingsOpen) return;
    function onPointerDown(e: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setSettingsOpen(false);
      }
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setSettingsOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [settingsOpen]);

  // Apply an update to a single chat: bump updatedAt, re-sort, and persist.
  // `persist: false` keeps the update in React state only — token deltas arrive
  // many times a second and must not each trigger an IndexedDB write; the
  // events that follow them (agent / summarizer / final) save the real content.
  function updateChat(
    id: string,
    mutate: (chat: Chat) => Chat,
    { persist = true }: { persist?: boolean } = {}
  ) {
    setChats((prev) => {
      const idx = prev.findIndex((c) => c.id === id);
      if (idx === -1) return prev;
      const updated = { ...mutate(prev[idx]), updatedAt: Date.now() };
      if (persist) saveChat(updated);
      const rest = prev.filter((c) => c.id !== id);
      return [updated, ...rest].sort((a, b) => b.updatedAt - a.updatedAt);
    });
  }

  function newChat() {
    const fresh = newChatObject();
    setChats((prev) => [fresh, ...prev]);
    setActiveId(fresh.id);
    setInput("");
  }

  function selectChat(id: string) {
    setActiveId(id);
    setInput("");
  }

  function removeChat(id: string) {
    deleteChatFromDB(id);
    setChats((prev) => {
      const remaining = prev.filter((c) => c.id !== id);
      if (id === activeId) {
        if (remaining.length > 0) {
          setActiveId(remaining[0].id);
        } else {
          const fresh = newChatObject();
          setActiveId(fresh.id);
          return [fresh];
        }
      }
      return remaining;
    });
  }

  function renameChat(id: string, title: string) {
    updateChat(id, (chat) => ({ ...chat, title }));
  }

  async function send() {
    const text = input.trim();
    if (!text || loading || !activeId || sendingRef.current) return;
    sendingRef.current = true;
    const chatId = activeId;
    const baseChat = chats.find((c) => c.id === chatId);
    if (!baseChat) {
      // Bailing out here without clearing the guard would wedge the composer
      // for the rest of the session — every later send() would return early.
      sendingRef.current = false;
      return;
    }

    const next: Message[] = [...messages, { role: "user", content: text }];
    const isFirstMessage = messages.length === 0;
    updateChat(chatId, (chat) => ({
      ...chat,
      messages: next,
      // Auto-title from the first user message while still on the default title.
      title:
        isFirstMessage && chat.title === "New chat" ? titleFromMessage(text) : chat.title,
    }));
    setInput("");
    setLoading(true);

    // Compact the outbound payload on-device. The stored `next` keeps the full
    // history; this only bounds what is sent. Short chats pass through unchanged.
    // Any failure falls back to the full history so a send is never blocked by it.
    let outbound: Message[] = next;
    try {
      const r = await prepareOutbound(baseChat, next);
      outbound = r.outbound;
      const cache = r.cache;
      if (cache) updateChat(chatId, (chat) => ({ ...chat, compaction: cache }));
    } catch (err) {
      console.error("context compaction failed; sending full history", err);
    }

    // Map one streamed agent/summarizer result onto the trace-chip shape.
    const toRun = (a: ApiAgentResult, id: string, label?: string): AgentRun => ({
      id,
      provider: label ?? a.provider,
      model: a.model ?? undefined,
      status: a.error ? "error" : "done",
      content: a.content,
      error: a.error ?? undefined,
      latencyMs: a.latency_ms ?? undefined,
      promptTokens: a.prompt_tokens,
      completionTokens: a.completion_tokens,
    });

    // Update the in-flight assistant message (always the chat's last message).
    const patchAssistant = (
      mutate: (m: Message) => Message,
      opts?: { persist?: boolean }
    ) =>
      updateChat(
        chatId,
        (chat) => {
          const msgs = [...chat.messages];
          const last = msgs[msgs.length - 1];
          if (last?.role === "assistant") msgs[msgs.length - 1] = mutate(last);
          return { ...chat, messages: msgs };
        },
        opts
      );

    // Append streamed text to the first chip matching `isTarget` and mark it
    // live. Not persisted — the agent's own end event writes the authoritative
    // content moments later. The match runs inside the state updater so it sees
    // the chips this same loop just created, not a stale render closure.
    const appendDelta = (isTarget: (a: AgentRun) => boolean, text: string) =>
      patchAssistant(
        (m) => {
          const agents = [...(m.agents ?? [])];
          const i = agents.findIndex(isTarget);
          if (i === -1) return m;
          agents[i] = {
            ...agents[i],
            status: "streaming",
            content: (agents[i].content ?? "") + text,
          };
          return { ...m, agents };
        },
        { persist: false }
      );

    // The summarizer chip is created by `summarizer_start`, but its first
    // tokens can be processed first — either event may open it.
    const ensureSummarizerChip = (model?: string | null) =>
      patchAssistant(
        (m) => {
          const agents = m.agents ?? [];
          if (agents.some((a) => a.id === SUMMARIZER_ID)) return m;
          return {
            ...m,
            agents: [
              ...agents,
              {
                id: SUMMARIZER_ID,
                provider: "summarizer",
                model: model ?? undefined,
                status: "pending",
              },
            ],
          };
        },
        { persist: false }
      );

    // Patch the in-flight assistant message with an error, or append one if
    // the failure happened before the stream's `start` event created it.
    const setAssistantError = (text: string) =>
      updateChat(chatId, (chat) => {
        const msgs = [...chat.messages];
        const last = msgs[msgs.length - 1];
        if (last?.role === "assistant") {
          msgs[msgs.length - 1] = { ...last, content: text };
          return { ...chat, messages: msgs };
        }
        return { ...chat, messages: [...msgs, { role: "assistant", content: text }] };
      });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: outbound,
          // Multi-agent line-up wins; otherwise run the one chosen model as a
          // single agent. With no model resolved (catalog still loading / it
          // failed), send neither and let the server fall back to its default.
          ...(customLineup
            ? {
                agents: toRequestAgents(roster),
                summarizer: toRequestSummarizer(roster),
              }
            : model
              ? {
                  agents: [
                    { name: shortModelName(model), provider: ROSTER_PROVIDER, model },
                  ],
                }
              : {}),
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      let sawFinal = false;
      for await (const { event, data } of readSSE(res.body)) {
        if (event === "start") {
          // The turn's true line-up: append the assistant message now with one
          // pending chip per agent, and store the server-redacted user message.
          const pending: AgentRun[] = ((data.agents ?? []) as TaskAgentInfo[]).map(
            (a, i) => ({
              id: `${a.name ?? a.provider}-${i}`,
              provider: a.name ?? a.provider,
              model: a.model ?? undefined,
              status: "pending",
            })
          );
          // If the server will synthesize, seat the summarizer chip pending now
          // so it appears beside the agents as they write, not only at the end.
          const startSummarizer = data.summarizer as TaskAgentInfo | null | undefined;
          if (startSummarizer) {
            pending.push({
              id: SUMMARIZER_ID,
              provider: "summarizer",
              model: startSummarizer.model ?? undefined,
              status: "pending",
            });
          }
          setStreamStarted(true);
          updateChat(chatId, (chat) => {
            const msgs = [...chat.messages];
            if (
              data.redacted_user_message != null &&
              msgs.length > 0 &&
              msgs[msgs.length - 1].role === "user"
            ) {
              msgs[msgs.length - 1] = {
                role: "user",
                content: data.redacted_user_message,
              };
            }
            return {
              ...chat,
              messages: [
                ...msgs,
                { role: "assistant", content: "", agents: pending },
              ],
            };
          });
        } else if (event === "agent_delta") {
          // Tokens from one branch — into the first chip still running under
          // that agent name.
          appendDelta(
            (a) =>
              a.id !== SUMMARIZER_ID &&
              a.provider === data.provider &&
              (a.status === "pending" || a.status === "streaming"),
            data.text ?? ""
          );
        } else if (event === "summarizer_delta") {
          ensureSummarizerChip();
          appendDelta((a) => a.id === SUMMARIZER_ID, data.text ?? "");
        } else if (event === "agent") {
          // One branch resolved — flip its chip, leave the others pending.
          const run = data as ApiAgentResult;
          patchAssistant((m) => ({
            ...m,
            agents: (m.agents ?? []).map((a) =>
              a.provider === run.provider &&
              (a.status === "pending" || a.status === "streaming")
                ? toRun(run, a.id)
                : a
            ),
          }));
        } else if (event === "summarizer_start") {
          ensureSummarizerChip(data.model);
        } else if (event === "summarizer") {
          const run = toRun(data as ApiAgentResult, SUMMARIZER_ID, "summarizer");
          patchAssistant((m) => ({
            ...m,
            agents: (m.agents ?? []).map((a) => (a.id === SUMMARIZER_ID ? run : a)),
          }));
        } else if (event === "final") {
          sawFinal = true;
          const errors: string[] = data.errors ?? [];
          const reply =
            data.reply || (errors.length ? `⚠️ ${errors[0]}` : "(empty response)");
          patchAssistant((m) => ({ ...m, content: reply }));
        } else if (event === "error") {
          sawFinal = true;
          setAssistantError(`⚠️ ${data.detail ?? "stream error"}`);
        }
      }
      if (!sawFinal) throw new Error("stream ended before the final event");
    } catch (err) {
      if (controller.signal.aborted) {
        // Stopped by the user: keep whatever streamed in as the reply and
        // retire the chips that never resolved. Not a failure — no error text.
        patchAssistant((m) => ({
          ...m,
          content: m.content || liveReply(m.agents ?? []),
          agents: (m.agents ?? []).map((a) =>
            a.status === "pending" || a.status === "streaming"
              ? { ...a, status: "stopped" }
              : a
          ),
        }));
      } else {
        setAssistantError(`⚠️ Request failed: ${String(err)}`);
      }
    } finally {
      abortRef.current = null;
      sendingRef.current = false;
      setLoading(false);
      setStreamStarted(false);
    }
  }

  // Stop the in-flight turn. Aborting the fetch closes the response body, which
  // cancels the server's SSE generator and, through it, the graph's branches.
  function stop() {
    abortRef.current?.abort();
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="layout">
      <Sidebar
        chats={chats}
        activeId={activeId}
        collapsed={collapsed}
        onSelect={selectChat}
        onNew={newChat}
        onDelete={removeChat}
        onRename={renameChat}
        onToggle={() => setCollapsed((c) => !c)}
      >
        {!collapsed && (
          <AgentsPanel roster={roster} models={models} onChange={setRoster} />
        )}
      </Sidebar>

      <div className="chat">
        <header className="chat__header">
          <div className="chat__brand">
            <span className="chat__title">A1A</span>
          </div>

          <div className="settings" ref={settingsRef}>
            <button
              className="settings__cog"
              onClick={() => setSettingsOpen((o) => !o)}
              aria-label="Settings"
              aria-haspopup="menu"
              aria-expanded={settingsOpen}
              title="Settings"
            >
              ⚙
            </button>

            {settingsOpen && (
              <div className="settings__menu" role="menu">
                <div className="settings__row">
                  <span className="settings__label">Dark mode</span>
                  <button
                    className={`toggle${theme === "dark" ? " toggle--on" : ""}`}
                    role="switch"
                    aria-checked={theme === "dark"}
                    aria-label="Toggle dark mode"
                    onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
                  >
                    <span className="toggle__knob" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </header>

        <div className="chat__messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="chat__empty">
              <div>Ask anything to get started.</div>
              {lineup.length > 0 && (
                <div className="lineup">
                  <span className="lineup__lead">
                    {customLineup ? "Line-up" : "Model"}
                  </span>
                  <span className="lineup__arrow" aria-hidden="true">
                    →
                  </span>
                  <div className="lineup__pills">
                    {lineup.map((a, i) => (
                      <span className="lineup__pill" key={i}>
                        <span className="lineup__dot" />
                        <span className="lineup__name">{a.name}</span>
                        {a.model && shortModelName(a.model) !== a.name && (
                          <span className="lineup__model">
                            {shortModelName(a.model)}
                          </span>
                        )}
                      </span>
                    ))}
                    {pendingSummarizer && (
                      <span className="lineup__pill lineup__pill--synth">
                        <span className="lineup__dot" />
                        <span className="lineup__name">synthesize</span>
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
          {messages.map((m, i) => {
            // Until `final` lands, the answer is previewed from the chips'
            // streamed text; `m.content` then takes over for good.
            const preview = m.role === "assistant" && !m.content
              ? liveReply(m.agents ?? [])
              : "";
            return m.role === "assistant" ? (
              <div key={i} className="turn">
                {m.agents && m.agents.length > 0 && (
                  <Trace agents={m.agents} task={m.task} />
                )}
                <div
                  className={`bubble bubble--assistant${
                    preview ? " bubble--streaming" : ""
                  }`}
                >
                  <div className="markdown">
                    {/* react-markdown does not render raw HTML by default, so
                        assistant output is safe from HTML/script injection. Links
                        open in a new tab. */}
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: (props) => (
                          <a {...props} target="_blank" rel="noreferrer noopener" />
                        ),
                      }}
                    >
                      {m.content || preview}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            ) : (
              <div key={i} className="bubble bubble--user">
                {m.content}
              </div>
            );
          })}
          {loading && !streamStarted && (
            <div className="turn">
              {pendingAgents.length > 0 && <Trace agents={pendingAgents} />}
              <div className="bubble bubble--assistant bubble--typing">…</div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="chat__input">
          {models.length > 0 && (
            <select
              className="modelsel"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={customLineup}
              aria-label="Model"
              title={
                customLineup
                  ? "Multi-agent line-up is active — turn it off in the Agents panel to pick a single model"
                  : "Model — which AI answers this turn"
              }
            >
              {featuredModels(models).map((m) => (
                <option key={m.id} value={m.id}>
                  {shortModelName(m.id)}
                </option>
              ))}
            </select>
          )}
          <textarea
            value={input}
            placeholder="Type a message…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
          />
          {loading ? (
            <button
              className="composer__stop"
              onClick={stop}
              aria-label="Stop generating"
              title="Stop generating"
            >
              ■
            </button>
          ) : (
            <button onClick={send} disabled={!input.trim()} aria-label="Send">
              ↑
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
