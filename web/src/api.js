// Thin client for LLM.Api - one function per endpoint, nothing else.
const API_BASE_URL = "https://localhost:7060";

export async function generateText({ prompt, maxTokens, temperature, topK }) {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, maxTokens, temperature, topK }),
  });

  if (!response.ok) {
    const problem = await response.json().catch(() => null);
    throw new Error(problem?.detail ?? problem?.title ?? `Request failed (${response.status})`);
  }

  const data = await response.json();
  return data.text;
}

export function checkHealth() {
  return fetch(`${API_BASE_URL}/health`).then((r) => r.json());
}

/**
 * Opens a Server-Sent Events connection to /train/stream.
 * - onLogLine fires for every human-readable log line (loading data, step N/M, errors, ...).
 * - onProgress fires every training step with (current, total) - much finer-grained than the
 *   log lines, which only appear once per eval interval, so the progress bar doesn't sit
 *   frozen at 0% for a whole eval interval before it first moves.
 * - onDone / onError report the end of the run. Returns the EventSource so the caller can
 *   close() it early (Stop button).
 */
export function streamTraining(params, { onLogLine, onProgress, onDone, onError }) {
  const query = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, value]) => value !== undefined && value !== ""))
  );
  const source = new EventSource(`${API_BASE_URL}/train/stream?${query.toString()}`);

  source.onmessage = (event) => {
    if (event.data === "[[DONE]]") {
      source.close();
      onDone();
      return;
    }
    if (event.data.startsWith("[[ERROR]]")) {
      source.close();
      onError(event.data.replace("[[ERROR]] ", ""));
      return;
    }
    if (event.data.startsWith("[[PROGRESS]]")) {
      const [current, total] = event.data.replace("[[PROGRESS]] ", "").split("/").map(Number);
      onProgress?.(current, total);
      return;
    }
    onLogLine(event.data);
  };

  source.onerror = () => {
    source.close();
    onError("Connection to the training stream was lost.");
  };

  return source;
}

// Tells the server to stop the in-flight training run. Closing the EventSource alone only
// makes the browser stop listening - the server needs this explicit signal to free up the
// "one job at a time" slot promptly (detecting a dropped connection can be slow/unreliable).
export function cancelTraining() {
  return fetch(`${API_BASE_URL}/train/cancel`, { method: "POST" });
}

// A browser can only hand the server file *content*, never a local filesystem path, so
// "browse" uploads the file's bytes and gets back the server-side path /train/stream needs.
export async function uploadDataFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/data/upload`, { method: "POST", body: formData });

  if (!response.ok) {
    const problem = await response.json().catch(() => null);
    throw new Error(problem?.detail ?? problem?.title ?? `Upload failed (${response.status})`);
  }

  const data = await response.json();
  return data.path;
}
