# dotnet-llm-from-scratch

A GPT-style, decoder-only language model built **entirely from scratch in C#/.NET** —
its own reverse-mode automatic differentiation engine, its own tensor library, its own
transformer blocks, optimizer, training loop, tokenizer, checkpointing, CLI, REST API,
and React web UI. No PyTorch, no TensorFlow, no ML.NET, no ONNX Runtime, no bindings to
a native tensor library of any kind. Every matrix multiply, every softmax, every
backward pass is plain C# you can step through in a debugger.

## Purpose

Most "build an LLM from scratch" resources (nanoGPT, micrograd, karpathy's lectures,
countless blog posts) are written in Python, because that's where the ML tooling and
audience live. That leaves a gap for .NET developers who want the same ground-up
understanding — how autograd actually works, how attention is actually computed, how a
training loop actually updates weights — without switching stacks or hiding the
mechanics behind a framework.

This project closes that gap. It is:

- **A from-scratch reference implementation.** Every piece of the pipeline (tensor ops,
  autodiff, embeddings, multi-head causal self-attention, LayerNorm, GELU MLPs, Adam,
  gradient clipping, cross-entropy loss, top-k/temperature sampling) is implemented
  directly in C#, in isolation, so each concept can be read and understood on its own
  rather than reverse-engineered out of a framework's abstractions.
- **A working end-to-end system**, not just a training script: a CLI for scripted
  training/generation, a REST API for programmatic access, a React web UI for
  interactive use, and Docker support for running it anywhere — all sharing the same
  underlying model and checkpoint format.
- **Small and CPU-only on purpose.** Default hyperparameters (4 layers, 128-dim
  embeddings, 128-token context) train in minutes on a laptop CPU, so the whole loop —
  edit a tensor op, retrain, see the effect — stays fast enough to actually iterate on.

## Who this is for / benefit to the developer community

- **.NET developers learning how LLMs work internally** — you can set a breakpoint in
  `Tensor.Backward()` or `CausalSelfAttention.Forward()` and watch gradients flow, with
  no C extension or CUDA kernel standing between you and the math.
- **Anyone who wants a minimal, hackable base** to try an idea — a new attention
  variant, a different optimizer, a BPE tokenizer instead of character-level — without
  wading through a production framework's configuration surface first.
- **Teams evaluating "do we need Python for this"** for a small, self-contained
  inference/training service — this shows what a fully C# training + serving stack
  looks like, checkpoint format and all.

## What this solution actually predicts

The model is a **character-level autoregressive language model**: given a sequence of
characters, it predicts a probability distribution over the *next single character*,
and generation repeatedly samples from that distribution (with temperature and top-k
filtering) to produce new text one character at a time.

Concretely, in this repo's default setup: train it on a text corpus (a short story is
included at [sample.txt](sample.txt)), and it learns the corpus's character-level
statistics — common letter sequences, spacing, punctuation patterns, and (given enough
data and training steps) recognizable words and short phrases in the style of what it
saw.

**Be realistic about scale.** This is an educational, CPU-sized model, not a production
LLM:
- With a small corpus (tens of KB) and the default ~820K-parameter model, expect
  output that captures local character statistics but not fluent, grammatical prose —
  that requires either a much larger corpus, many more training steps, or both.
  Garbled output from an undertrained model is expected behavior, not a bug.
- There's no KV-cache, no batching optimizations, no GPU support — generation re-runs
  the full forward pass over the current context on every new token, which is fine at
  this scale and not intended to scale further.

## Architecture

```
LLM.Core        Tensor + autodiff engine (the only "framework" this project has).
                Dense row-major float tensors, reverse-mode autodiff via per-tensor
                backward closures. Every op (Add, MatMul, Softmax, CrossEntropy,
                EmbeddingLookup, Permute/Transpose/Reshape, ...) lives in TensorOps.cs.

LLM.Model       The GPT architecture, built purely out of LLM.Core tensors:
                Embedding (token + positional), LayerNorm, Linear, CausalSelfAttention
                (multi-head, causal-masked), FeedForward (GELU MLP), Dropout,
                TransformerBlock (pre-norm residual block), and GptModel tying it all
                together with a config-driven number of layers/heads/embedding width.

LLM.Tokenizer   CharTokenizer: the simplest possible tokenizer - one token per distinct
                Unicode character seen in the training corpus. Deterministic, sorted
                vocab; trivial to swap for a BPE tokenizer later.

LLM.Training    Trainer (the training loop), AdamOptimizer, GradClip (global-norm
                clipping), DataLoader (random block sampling), TextGenerator
                (temperature + top-k autoregressive sampling), and Checkpoint
                (save/load a model + tokenizer as config.json + vocab.json +
                weights.bin).

LLM.Api         ASP.NET Core Web API: TrainingService runs one training job at a time
                on a background thread and streams progress over SSE; LlmInferenceService
                lazily loads the checkpoint and serves /generate. See "API" below.

LLM.Cli         A console app exposing `train` and `generate` commands for scripted /
                headless use, independent of the API or web UI.

web/            A React (Vite) single-page app: a Train panel (kick off training,
                watch live logs/progress over SSE, stop mid-run) and a Generate panel
                (prompt + sampling controls), both rendered as retro terminal windows.

tests/          xUnit tests for the core engine: gradient-check tests (numerical vs.
                autodiff gradients) and model sanity tests.
```

Data flow, end to end:

```
corpus (.txt)
   -> CharTokenizer.BuildFromCorpus   (build the char vocabulary)
   -> DataLoader                      (random fixed-length blocks -> input/target pairs)
   -> GptModel.ForwardWithLoss        (forward pass + cross-entropy loss)
   -> Tensor.Backward()               (autodiff computes all gradients)
   -> AdamOptimizer.Step()            (update weights)
   -> Checkpoint.Save()               (config.json + vocab.json + weights.bin)
   -> Checkpoint.Load()               (CLI `generate`, or LlmInferenceService)
   -> TextGenerator.Generate()        (autoregressive sampling -> text)
```

## Getting started

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download)
- [Node.js](https://nodejs.org/) (for the web UI)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (optional, only for containerized API)

### Train and generate from the CLI

```powershell
# Train a model on a text file, checkpointing to ./checkpoints/sample
dotnet run --project LLM.Cli -- train --data sample.txt --out checkpoints/sample --steps 3000

# Generate from the trained checkpoint
dotnet run --project LLM.Cli -- generate --model checkpoints/sample --prompt "The old mill" --tokens 500
```

Run `dotnet run --project LLM.Cli` with no arguments for the full list of `train`/
`generate` options (steps, batch size, block size, embedding width, heads, layers,
dropout, learning rate, eval interval, seed, temperature, top-k, ...).

### Run the API + web UI

```powershell
# Terminal 1 - API (serves https://localhost:7060 by default)
dotnet run --project LLM.Api

# Terminal 2 - web UI (Vite dev server on http://localhost:5173)
cd web
npm install
npm run dev
```

Open the printed `localhost:5173` URL. The Train panel uploads a data file (or accepts
a server-side path), streams training logs and a live progress bar over
Server-Sent Events, and can be stopped mid-run. The Generate panel sends a prompt and
sampling parameters to `/generate` and shows the result in a terminal-style log window.

**API endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Liveness/info message |
| `/health` | GET | `{ modelLoaded: bool }` |
| `/generate` | POST | `{ prompt, maxTokens, temperature, topK }` -> `{ text }` |
| `/train/stream` | GET (SSE) | Starts a training run; streams log lines and `[[PROGRESS]] step/total` events |
| `/train/cancel` | POST | Stops the in-flight training run |
| `/data/upload` | POST (multipart) | Uploads a text file for training; returns its server-side path |

### Run the API in Docker

A `Dockerfile` is provided at [LLM.Api/Dockerfile](LLM.Api/Dockerfile). Build context
must be the **repo root** (project references cross into sibling folders):

```powershell
docker build -t llm-api -f LLM.Api/Dockerfile .

docker run -d --name llm-api `
  -p 7060:7060 `
  -v "${PWD}\checkpoints:/app/checkpoints" `
  -v "${PWD}\data:/app/data" `
  -v "${env:USERPROFILE}\.aspnet\https:/https:ro" `
  -e ASPNETCORE_URLS="https://+:7060" `
  -e ASPNETCORE_Kestrel__Certificates__Default__Path="/https/aspnetcore-dev-cert.pfx" `
  -e ASPNETCORE_Kestrel__Certificates__Default__Password="<your-dev-cert-password>" `
  llm-api
```

The HTTPS env vars point Kestrel at an exported copy of your local ASP.NET Core dev
certificate (`dotnet dev-certs https -ep <path> -p <password>`), so the container can
keep serving HTTPS on the same port the web UI already expects. The two volume mounts
keep checkpoints and uploaded training data on your host filesystem, persisted across
container restarts, exactly mirroring how `CheckpointDir`/upload paths resolve when run
locally.

### Run the tests

```powershell
dotnet test tests/LLM.Core.Tests
```

Covers the autodiff engine itself: numerical-vs-analytic gradient checks and basic
model sanity checks (shapes, loss finiteness, etc.).

## Checkpoint format

`Checkpoint.Save`/`Checkpoint.Load` (in `LLM.Training`) write/read a directory of three
files:

- `config.json` — the `GptConfig` (vocab size, block size, embedding width, heads,
  layers, dropout) needed to reconstruct the exact same architecture before loading
  weights into it.
- `vocab.json` — the `CharTokenizer`'s character vocabulary, so encode/decode stay
  consistent with how the model was trained.
- `weights.bin` — every parameter tensor's raw `float` data, written/read in the
  deterministic order `GptModel.Parameters()` enumerates them.

`Load` validates all three files exist and that `weights.bin`'s size matches what the
config implies before reading it, so a checkpoint left corrupted by an interrupted save
fails with a clear, actionable message rather than a raw stream-read exception.

## Configuration reference

`GptConfig` (architecture, fixed at training time and stored in the checkpoint):

| Field | Default | Meaning |
|---|---|---|
| `VocabSize` | (from corpus) | Distinct characters in the training corpus |
| `BlockSize` | 128 | Max context length (tokens) |
| `NEmbd` | 128 | Embedding / residual stream width |
| `NHead` | 4 | Attention heads |
| `NLayer` | 4 | Transformer blocks |
| `DropoutProb` | 0.1 | Dropout probability |

`TrainingConfig` (training run, not stored in the checkpoint):

| Field | Default | Meaning |
|---|---|---|
| `BatchSize` | 32 | Sequences per step |
| `MaxSteps` | 3000 | Total optimizer steps |
| `EvalInterval` | 100 | Steps between val-loss evals + checkpoint saves |
| `EvalIters` | 20 | Batches averaged per eval |
| `LearningRate` | 3e-4 | Adam learning rate |
| `Beta1` / `Beta2` | 0.9 / 0.95 | Adam moment decay rates |
| `WeightDecay` | 0.01 | Adam weight decay |
| `GradClipNorm` | 1.0 | Global gradient-norm clipping threshold |

## Known limitations

- **Character-level tokenizer, closed vocabulary.** A trained model can only encode
  characters it saw during training — prompting it with an unseen character returns a
  clear 400 error (`/generate`) rather than crashing, but it still can't be generated
  from. Retrain on a corpus covering the characters you need.
- **No GPU support.** Everything runs on CPU with plain C# loops - fine at the sizes
  this project targets, not intended to scale to larger models.
- **No KV-cache.** Generation reprocesses the whole context window on every sampled
  token; simplicity was prioritized over throughput.
- **Small corpus in, small fluency out.** See "What this solution actually predicts"
  above - this is an intentional, honest constraint of the demo, not a defect.

## Project status

This is an educational / hobby project, actively iterated on rather than a finished
product. Expect the default corpus, hyperparameters, and even the checkpoint format to
keep evolving as the project grows.
