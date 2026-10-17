import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Chat } from "./db";

interface SidebarProps {
  chats: Chat[];
  activeId: string | null;
  collapsed: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onToggle: () => void;
  // Rendered below the chat list (the agents panel). Hidden when collapsed.
  children?: ReactNode;
}

// Coarse relative-time label for the chat list (e.g. "just now", "3h", "2d").
function relativeTime(ts: number): string {
  const secs = Math.floor((Date.now() - ts) / 1000);
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  return `${days}d`;
}

export default function Sidebar({
  chats,
  activeId,
  collapsed,
  onSelect,
  onNew,
  onDelete,
  onRename,
  onToggle,
  children,
}: SidebarProps) {
  // Which chat is being inline-renamed, plus its working title text.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  // Which chat is showing the inline "delete?" confirmation.
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  // Which chat's "⋮" actions menu is open.
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId) inputRef.current?.focus();
  }, [editingId]);

  // Close the actions menu on an outside click or Escape (mirrors the settings
  // dropdown in App.tsx). A click inside any row's menu wrapper is left alone.
  useEffect(() => {
    if (!menuOpenId) return;
    function onDown(e: MouseEvent) {
      if (!(e.target as HTMLElement).closest(".chatrow__menu-wrap")) {
        setMenuOpenId(null);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setMenuOpenId(null);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpenId]);

  function startRename(chat: Chat) {
    setConfirmingId(null);
    setEditingId(chat.id);
    setDraft(chat.title);
  }

  function commitRename() {
    if (editingId) {
      const title = draft.trim();
      if (title) onRename(editingId, title);
    }
    setEditingId(null);
  }

  if (collapsed) {
    return (
      <aside className="sidebar sidebar--collapsed">
        <button className="sidebar__icon" onClick={onToggle} aria-label="Expand sidebar" title="Expand">
          ☰
        </button>
        <button className="sidebar__icon" onClick={onNew} aria-label="New chat" title="New chat">
          +
        </button>
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__top">
        <button className="sidebar__icon" onClick={onToggle} aria-label="Collapse sidebar" title="Collapse">
          ☰
        </button>
        <button className="sidebar__new" onClick={onNew}>
          + New chat
        </button>
      </div>

      {/* Every chat stays in the list; CSS caps the height at five rows and the
          rest scroll, so the agents panel below always has room. */}
      <div className="sidebar__list">
        {chats.length === 0 && <div className="sidebar__empty">No chats yet.</div>}
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={`chatrow${chat.id === activeId ? " chatrow--active" : ""}`}
            onClick={() =>
              editingId !== chat.id && confirmingId !== chat.id && onSelect(chat.id)
            }
          >
            {confirmingId === chat.id ? (
              <div className="chatrow__confirm" onClick={(e) => e.stopPropagation()}>
                <span className="chatrow__confirm-text">Confirm delete?</span>
                <div className="chatrow__confirm-actions">
                  <button
                    className="chatrow__confirm-btn chatrow__confirm-btn--danger"
                    onClick={() => {
                      onDelete(chat.id);
                      setConfirmingId(null);
                    }}
                  >
                    Delete
                  </button>
                  <button
                    className="chatrow__confirm-btn"
                    onClick={() => setConfirmingId(null)}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : editingId === chat.id ? (
              <input
                ref={inputRef}
                className="chatrow__edit"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={commitRename}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    commitRename();
                  } else if (e.key === "Escape") {
                    setEditingId(null);
                  }
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <>
                <div className="chatrow__main">
                  <span className="chatrow__title">{chat.title}</span>
                  <span className="chatrow__time">{relativeTime(chat.updatedAt)}</span>
                </div>
                <div className="chatrow__menu-wrap">
                  <button
                    className={`chatrow__menu-btn${
                      menuOpenId === chat.id ? " chatrow__menu-btn--active" : ""
                    }`}
                    aria-label="Chat actions"
                    aria-haspopup="menu"
                    aria-expanded={menuOpenId === chat.id}
                    title="More"
                    onClick={(e) => {
                      e.stopPropagation();
                      setMenuOpenId((cur) => (cur === chat.id ? null : chat.id));
                    }}
                  >
                    ⋯
                  </button>
                  {menuOpenId === chat.id && (
                    <div
                      className="chatrow__menu"
                      role="menu"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        className="chatrow__menu-item"
                        role="menuitem"
                        onClick={() => {
                          setMenuOpenId(null);
                          startRename(chat);
                        }}
                      >
                        Rename
                      </button>
                      <button
                        className="chatrow__menu-item chatrow__menu-item--danger"
                        role="menuitem"
                        onClick={() => {
                          setMenuOpenId(null);
                          setConfirmingId(chat.id);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {children}
    </aside>
  );
}
