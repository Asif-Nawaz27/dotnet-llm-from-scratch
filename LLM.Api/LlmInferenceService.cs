using LLM.Model;
using LLM.Tokenizer;
using LLM.Training;

namespace LLM.Api;

/// <summary>
/// The one thing the API needs from the model layer: turn a prompt into generated text.
/// Loads the checkpoint from disk once, the first time it's needed, and reuses it for
/// every request after that.
/// </summary>
public sealed class LlmInferenceService
{
    private readonly string _checkpointDir;
    private readonly object _loadLock = new();
    private GptModel? _model;
    private CharTokenizer? _tokenizer;

    public LlmInferenceService(IConfiguration config)
    {
        _checkpointDir = config["CheckpointDir"]
            ?? throw new InvalidOperationException("Missing config value 'CheckpointDir'.");
    }

    public bool IsModelLoaded => _model is not null;

    public string Generate(string prompt, int maxTokens = 200, float temperature = 0.8f, int topK = 40)
    {
        EnsureModelLoaded();
        return TextGenerator.Generate(_model!, _tokenizer!, prompt, maxTokens, temperature, topK, new Random());
    }

    /// <summary>Drops the cached model so the next Generate() call re-reads the checkpoint
    /// from disk - called after a training run finishes, since it writes to this same path.</summary>
    public void Invalidate()
    {
        lock (_loadLock)
        {
            _model = null;
            _tokenizer = null;
        }
    }

    private void EnsureModelLoaded()
    {
        if (_model is not null) return;

        lock (_loadLock)
        {
            if (_model is not null) return; // another request may have loaded it while we waited

            if (!Directory.Exists(_checkpointDir))
                throw new InvalidOperationException(
                    $"No checkpoint found at '{_checkpointDir}'. Train a model first with `LLM.Cli train`.");

            (_model, _tokenizer) = Checkpoint.Load(_checkpointDir);
        }
    }
}
