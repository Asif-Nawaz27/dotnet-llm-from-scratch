import { useState } from "react";
import { generateText } from "./api";
import SliderField from "./SliderField";
import TerminalWindow from "./TerminalWindow";
import LogLines from "./LogLines";

export default function GeneratePanel() {
  const [prompt, setPrompt] = useState("The old mill");
  const [maxTokens, setMaxTokens] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [log, setLog] = useState([]);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const loading = status === "running";

  function appendLine(text, level = "info") {
    setLog((prev) => [...prev, { text, level }]);
  }

  async function handleGenerate() {
    setLog([]);
    setStatus("running");
    appendLine(`Generating (max ${maxTokens} tokens, temperature ${temperature}, top-k ${topK})…`);
    try {
      const text = await generateText({ prompt, maxTokens, temperature, topK });
      appendLine(`Done — ${text.length} characters generated.`);
      appendLine(text, "output");
      setStatus("done");
    } catch (err) {
      appendLine(err.message, "error");
      setStatus("error");
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !loading) handleGenerate();
  }

  return (
    <div className="panel-stack">
      <TerminalWindow title="generate.sh" status={status}>
        <label className="field">
          <span className="field-label">prompt</span>
          <textarea
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Seed text to continue from…"
          />
        </label>

        <div className="row">
          <SliderField label="max tokens" value={maxTokens} onChange={setMaxTokens} min={10} max={1000} step={10} />
          <SliderField label="temperature" value={temperature} onChange={setTemperature} min={0.1} max={2} step={0.1} />
          <SliderField label="top-k" value={topK} onChange={setTopK} min={0} max={100} step={5} />
        </div>

        <div className="actions">
          <button className="btn btn-primary" onClick={handleGenerate} disabled={loading}>
            {loading ? "generating…" : "▶ generate"}
          </button>
          <span className="hint">ctrl/cmd + enter</span>
        </div>
      </TerminalWindow>

      <TerminalWindow title="generate.log" flush>
        <LogLines lines={log} placeholder="// output will appear here" busy={loading} />
      </TerminalWindow>
    </div>
  );
}
