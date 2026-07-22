// status: "idle" | "running" | "stopping" | "done" | "error"
const LABELS = {
  idle: "idle",
  running: "running",
  stopping: "stopping",
  done: "done",
  error: "error",
};

export default function StatusPill({ status }) {
  return <span className={`status-pill status-${status}`}>{LABELS[status] ?? status}</span>;
}
