using LLM.Model;
using LLM.Tokenizer;
using LLM.Training;

namespace LLM.Api;

/// <summary>Request shape for a training run - the same knobs LLM.Cli's `train` command takes,
/// minus the checkpoint directory: that's server config (<see cref="TrainingService"/>), the
/// same one LlmInferenceService reads generated text from, so training and generation always
/// agree on where the model lives without the client having to keep the two in sync.</summary>
public sealed record TrainRequest(
    string DataPath,
    int Steps = 2000,
    int BatchSize = 32,
    int BlockSize = 128,
    int NEmbd = 128,
    int NHead = 4,
    int NLayer = 4,
    float Dropout = 0.1f,
    float Lr = 3e-4f,
    int EvalInterval = 20,
    int EvalIters = 10,
    int Seed = 1337);

/// <summary>
/// Runs one training job at a time on a background thread and reports progress through a
/// plain callback, so the API layer can forward each line straight into an SSE stream.
///
/// Stopping relies on an explicit <see cref="RequestStop"/> call (wired to a dedicated
/// /train/cancel endpoint) rather than only on the SSE connection dropping - detecting a
/// closed HTTP connection can be slow or unreliable, which left the "one job at a time"
/// flag stuck if a client reconnected quickly after hitting Stop.
/// </summary>
public sealed class TrainingService
{
    private readonly string _checkpointDir;
    private readonly LlmInferenceService _inferenceService;
    private int _isTraining;
    private CancellationTokenSource? _cts;

    public TrainingService(IConfiguration config, LlmInferenceService inferenceService)
    {
        _checkpointDir = config["CheckpointDir"]
            ?? throw new InvalidOperationException("Missing config value 'CheckpointDir'.");
        _inferenceService = inferenceService;
    }

    /// <summary>Reserves the single training slot. On success, <paramref name="token"/> fires
    /// when either the caller's request is aborted or <see cref="RequestStop"/> is called.</summary>
    public bool TryStart(CancellationToken requestAborted, out CancellationToken token)
    {
        if (Interlocked.CompareExchange(ref _isTraining, 1, 0) != 0)
        {
            token = default;
            return false;
        }

        _cts = CancellationTokenSource.CreateLinkedTokenSource(requestAborted);
        token = _cts.Token;
        return true;
    }

    /// <summary>Signals the in-flight run (if any) to stop at its next per-step check.</summary>
    public void RequestStop() => _cts?.Cancel();

    /// <summary>onProgress fires every training step (cheap - no eval-loss pass), independent
    /// of onLog which only fires at eval intervals; the web UI uses it to drive a smooth
    /// progress bar instead of one that only moves once per eval interval.</summary>
    public Task RunAsync(
        TrainRequest request,
        Action<string> onLog,
        CancellationToken cancellationToken,
        Action<int, int>? onProgress = null)
    {
        // Deliberately not passed as Task.Run's own token: that would let Task.Run skip
        // invoking the delegate entirely if the token were already cancelled, which would
        // skip the `finally` below and leave the training slot stuck forever.
        return Task.Run(() =>
        {
            try
            {
                onLog("Starting training run...");

                onLog($"Loading data file '{request.DataPath}'...");
                var text = File.ReadAllText(request.DataPath);
                onLog($"Loaded corpus: {text.Length} characters.");

                onLog("Building tokenizer from corpus...");
                var tokenizer = CharTokenizer.BuildFromCorpus(text);
                onLog($"Vocabulary size: {tokenizer.VocabSize} unique characters.");

                var allIds = tokenizer.Encode(text);
                int splitPoint = (int)(allIds.Length * 0.9);
                var trainIds = allIds[..splitPoint];
                var valIds = allIds[splitPoint..];
                onLog($"Train tokens: {trainIds.Length}, Val tokens: {valIds.Length}");

                var config = new GptConfig
                {
                    VocabSize = tokenizer.VocabSize,
                    BlockSize = request.BlockSize,
                    NEmbd = request.NEmbd,
                    NHead = request.NHead,
                    NLayer = request.NLayer,
                    DropoutProb = request.Dropout,
                };

                onLog("Building model...");
                var rng = new Random(request.Seed);
                var model = new GptModel(config, rng);
                onLog($"Model parameters: {model.Parameters().Sum(p => p.Size):N0}");

                var trainLoader = new DataLoader(trainIds, request.BlockSize, new Random(request.Seed + 1));
                var valLoader = new DataLoader(valIds, request.BlockSize, new Random(request.Seed + 2));

                var trainingConfig = new TrainingConfig
                {
                    BatchSize = request.BatchSize,
                    MaxSteps = request.Steps,
                    EvalInterval = request.EvalInterval,
                    EvalIters = request.EvalIters,
                    LearningRate = request.Lr,
                    CheckpointDir = _checkpointDir,
                };

                onLog($"Training for {request.Steps} steps...");
                Trainer.Train(model, trainLoader, valLoader, trainingConfig, tokenizer, onLog, cancellationToken, onProgress);
                onLog($"Done. Checkpoint saved to '{_checkpointDir}'.");

                _inferenceService.Invalidate(); // /generate picks up the freshly trained weights next call
            }
            finally
            {
                _cts?.Dispose();
                _cts = null;
                Interlocked.Exchange(ref _isTraining, 0);
            }
        });
    }
}
