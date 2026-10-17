// Minimal Server-Sent-Events parsing for fetch() responses.
//
// EventSource only supports GET, and /api/chat/stream needs a POST body, so we
// read the response stream ourselves. The incremental parser is a plain
// function over string chunks — chunk boundaries land anywhere (mid-line,
// mid-frame), so it buffers until a blank line completes a frame.

export interface SSEEvent {
  event: string;
  // Parsed JSON payload of the frame's data line(s).
  data: any;
}

// Returns a function that accepts one decoded chunk and returns the events
// completed by it. State (the partial-frame buffer) lives in the closure.
export function createSSEParser(): (chunk: string) => SSEEvent[] {
  let buf = "";
  return (chunk: string) => {
    buf += chunk;
    const events: SSEEvent[] = [];
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const evt = parseFrame(frame);
      if (evt) events.push(evt);
    }
    return events;
  };
}

// One frame = the lines between blank-line separators.
function parseFrame(frame: string): SSEEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    // Comments (":keepalive") and unknown fields are ignored per the SSE spec.
  }
  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null; // malformed frame — skip rather than kill the stream
  }
}

// Read a fetch() body as SSE events.
export async function* readSSE(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<SSEEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parse = createSSEParser();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const evt of parse(decoder.decode(value, { stream: true }))) {
        yield evt;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
