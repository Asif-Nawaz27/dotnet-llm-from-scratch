import { useEffect, useRef, useState } from "react";
import { cancelTraining, streamTraining, uploadDataFile } from "./api";
import SliderField from "./SliderField";
import TerminalWindow from "./TerminalWindow";
import LogLines from "./LogLines";

// Shown as a placeholder only, not a default value - the field is required, so Start
// stays disabled until the user actually browses for or types a real data file. Paths are
// resolved by LLM.Api, not by this app, so they're relative to that project's folder - one
// level up lands back at the repo root (matches TrainingService's CheckpointDir, "../checkpoints").
const DATA_PATH_PLACEHOLDER = "../data/sample.txt";

export default function TrainPanel() {
  const [dataPath, setDataPath] = useState("");
  const [steps, setSteps] = useState(3000);
  const [evalInterval, setEvalInterval] = useState(100);
  const [log, setLog] = useState([]);
  const [status, setStatus] = useState("idle"); // idle | running | stopping | done | error
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0); // 0-100, driven by the server's per-step [[PROGRESS]] messages

  const eventSourceRef = useRef(null);
  const logEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "end" });
  }, [log]);

  useEffect(() => () => eventSourceRef.current?.close(), []); // close on unmount

  function appendLine(text, level = "info") {
    setLog((prev) => [...prev, { text, level }]);
  }

  async function handleFileChosen(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // so choosing the same file again still fires onChange
    if (!file) return;

    setUploading(true);
    try {
      const path = await uploadDataFile(file);
      setDataPath(path);
      appendLine(`uploaded ${file.name}`);
    } catch (err) {
      appendLine(`upload failed: ${err.message}`, "error");
    } finally {
      setUploading(false);
    }
  }

  function startTraining() {
    setLog([]);
    setProgress(0);
    setStatus("running");
    eventSourceRef.current = streamTraining(
      { dataPath, steps, evalInterval },
      {
        onLogLine: (line) => appendLine(line),
        onProgress: (current, total) => setProgress((current / total) * 100),
        onDone: () => {
          setProgress(100);
          appendLine("training finished");
          setStatus("done");
        },
        onError: (message) => {
          appendLine(message, "error");
          setStatus("error");
        },
      }
    );
  }

  async function stopTraining() {
    setStatus("stopping"); // both buttons disabled until the server confirms the cancel
    try {
      await cancelTraining();
    } finally {
      eventSourceRef.current?.close();
      setStatus("idle");
      appendLine("stopped", "warn");
    }
  }

  const running = status === "running";
  const busy = status === "running" || status === "stopping" || uploading;
  const hasDataPath = dataPath.trim().length > 0;

  return (
    <div className="panel-stack">
      <TerminalWindow title="train.sh" status={status}>
        <label className="field">
          <span className="field-label">
            data file <span className="required">*</span>
          </span>
          <div className="file-row">
            <input
              value={dataPath}
              onChange={(e) => setDataPath(e.target.value)}
              disabled={busy}
              placeholder={DATA_PATH_PLACEHOLDER}
              required
            />
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => fileInputRef.current?.click()}
              disabled={busy}
            >
              {uploading ? "uploading…" : "browse"}
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".txt"
              onChange={handleFileChosen}
              style={{ display: "none" }}
            />
          </div>
          {!hasDataPath && <span className="field-hint">choose or enter a data file to continue</span>}
        </label>

        <div className="row">
          <SliderField label="steps" value={steps} onChange={setSteps} min={5} max={5000} step={5} disabled={busy} />
          <SliderField
            label="eval interval"
            value={evalInterval}
            onChange={setEvalInterval}
            min={5}
            max={200}
            step={5}
            disabled={busy}
          />
        </div><div></div>

        <div className="actions">
          <button className="btn btn-primary" onClick={startTraining} disabled={busy || !hasDataPath}>
            {running ? "training…" : status === "stopping" ? "stopping…" : "▶ start"}
          </button>
          <button className="btn btn-danger" onClick={stopTraining} disabled={!running}>
            ■ stop
          </button>
        </div>
      </TerminalWindow>

      <TerminalWindow title="train.log" status={status} progress={progress} flush>
        <LogLines
          lines={log}
          placeholder="// log output will stream here while training runs"
          busy={busy}
          bottomRef={logEndRef}
        />
      </TerminalWindow>
    </div>
  );
}
