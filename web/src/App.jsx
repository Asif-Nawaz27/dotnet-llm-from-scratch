import "./App.css";
import TrainPanel from "./TrainPanel";
import GeneratePanel from "./GeneratePanel";
import ThemeToggle from "./ThemeToggle";

export default function App() {
  return (
    <main className="app">
      <header className="app-header">
        <div>
          <h1>
            <span className="prompt">$</span> llm-console
          </h1>
          <p className="subtitle">from-scratch GPT model — train and generate, live</p>
        </div>
        <ThemeToggle />
      </header>
      <div className="panels">
        <TrainPanel />
        <GeneratePanel />
      </div>
    </main>
  );
}
