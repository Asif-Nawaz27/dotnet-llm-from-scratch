// A compact inline progress bar for the terminal title bar - value is 0-100.
export default function ProgressBar({ value }) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div className="progress-bar" title={`${pct}%`}>
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="progress-bar-label">{pct}%</span>
    </div>
  );
}
