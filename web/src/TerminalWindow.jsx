import StatusPill from "./StatusPill";
import ProgressBar from "./ProgressBar";

// Gives a panel the "terminal window" chrome: traffic-light dots, a title, and
// an optional status pill. Purely visual - shared so both panels look identical.
// `flush` drops the body's padding - used for the .log windows so the console
// output fills the window edge-to-edge instead of floating in a gray margin.
// `progress` (0-100), when given while status is "running"/"stopping", replaces
// the status pill with a progress bar instead.
export default function TerminalWindow({ title, status, progress, flush = false, children }) {
  const showProgress = progress != null && (status === "running" || status === "stopping");

  return (
    <section className="panel">
      <header className="panel-title-bar">
        <span className="dots">
          <i className="dot red" />
          <i className="dot yellow" />
          <i className="dot green" />
        </span>
        <span className="panel-title">{title}</span>
        {showProgress ? <ProgressBar value={progress} /> : status && <StatusPill status={status} />}
      </header>
      <div className={flush ? "panel-body panel-body--flush" : "panel-body"}>{children}</div>
    </section>
  );
}
