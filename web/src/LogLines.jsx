const TAG = {
  info: "INFO",
  warn: "WARN",
  error: "ERROR",
};

// Renders a terminal-style log: each entry is colored and tagged by severity so
// info/warning/error stand out at a glance. `output` is untagged plain text (i.e. process
// stdout, or the model's generated text) - tagging every stdout line would be more noise
// than signal, so only info/warn/error system messages get a bracketed tag.
export default function LogLines({ lines, placeholder, busy, bottomRef }) {
  return (
    <pre className="console-output">
      {lines.length === 0 ? (
        placeholder
      ) : (
        lines.map((entry, i) => {
          const level = entry.level ?? "output";
          const tag = TAG[level];
          return (
            <div key={i} className={`log-line log-${level}`}>
              {tag && <span className="log-tag">[{tag}]</span>}
              {entry.text}
            </div>
          );
        })
      )}
      {busy && <span className="cursor" />}
      {bottomRef && <div ref={bottomRef} />}
    </pre>
  );
}
