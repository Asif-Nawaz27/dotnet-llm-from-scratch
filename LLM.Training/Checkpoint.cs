using System.Text.Json;
using LLM.Model;
using LLM.Tokenizer;

namespace LLM.Training;

/// <summary>Saves/loads a checkpoint as three files in a directory:
/// config.json (GptConfig), vocab.json (tokenizer vocab), weights.bin (raw parameter data).
/// Weights are written/read in exactly the order <see cref="GptModel.Parameters"/> enumerates
/// them, which is deterministic given the same config - so the config must be saved/loaded
/// alongside the weights, never assumed.</summary>
public static class Checkpoint
{
    private const string ConfigFileName = "config.json";
    private const string VocabFileName = "vocab.json";
    private const string WeightsFileName = "weights.bin";

    public static void Save(string directory, GptModel model, CharTokenizer tokenizer)
    {
        Directory.CreateDirectory(directory);

        File.WriteAllText(Path.Combine(directory, ConfigFileName), JsonSerializer.Serialize(model.Config));
        tokenizer.SaveVocab(Path.Combine(directory, VocabFileName));

        using var stream = File.Create(Path.Combine(directory, WeightsFileName));
        using var writer = new BinaryWriter(stream);
        foreach (var p in model.Parameters())
        {
            foreach (var f in p.Data) writer.Write(f);
        }
    }

    public static (GptModel Model, CharTokenizer Tokenizer) Load(string directory)
    {
        var configPath = Path.Combine(directory, ConfigFileName);
        var vocabPath = Path.Combine(directory, VocabFileName);
        var weightsPath = Path.Combine(directory, WeightsFileName);

        foreach (var path in new[] { configPath, vocabPath, weightsPath })
        {
            if (!File.Exists(path))
                throw new InvalidOperationException(
                    $"Checkpoint at '{directory}' is missing '{Path.GetFileName(path)}'. " +
                    "The checkpoint is incomplete or was never fully saved - retrain to regenerate it.");
        }

        GptConfig config;
        try
        {
            config = JsonSerializer.Deserialize<GptConfig>(File.ReadAllText(configPath))
                     ?? throw new InvalidDataException($"'{configPath}' deserialized to null.");
        }
        catch (Exception ex) when (ex is JsonException or InvalidDataException)
        {
            throw new InvalidOperationException(
                $"Checkpoint config at '{configPath}' is corrupted or not valid JSON: {ex.Message}. " +
                "Retrain to regenerate the checkpoint.", ex);
        }

        CharTokenizer tokenizer;
        try
        {
            tokenizer = CharTokenizer.LoadVocab(vocabPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Checkpoint vocab at '{vocabPath}' is corrupted: {ex.Message}. " +
                "Retrain to regenerate the checkpoint.", ex);
        }

        // rng only seeds initial values, which are about to be overwritten by the checkpoint.
        var model = new GptModel(config, new Random(0));

        long expectedFloats = model.Parameters().Sum(p => (long)p.Data.Length);
        long expectedBytes = expectedFloats * sizeof(float);
        long actualBytes = new FileInfo(weightsPath).Length;
        if (actualBytes != expectedBytes)
        {
            throw new InvalidOperationException(
                $"Checkpoint weights at '{weightsPath}' are corrupted: expected {expectedBytes:N0} bytes " +
                $"({expectedFloats:N0} floats) for config ({config.NLayer} layers, {config.NEmbd} embd, " +
                $"vocab {config.VocabSize}), but found {actualBytes:N0} bytes. This usually means the save " +
                "was interrupted partway through (e.g. the process was stopped or crashed while writing " +
                "weights.bin) - retrain to regenerate a complete checkpoint.");
        }

        try
        {
            using var stream = File.OpenRead(weightsPath);
            using var reader = new BinaryReader(stream);
            foreach (var p in model.Parameters())
            {
                for (int i = 0; i < p.Data.Length; i++) p.Data[i] = reader.ReadSingle();
            }
        }
        catch (EndOfStreamException ex)
        {
            throw new InvalidOperationException(
                $"Checkpoint weights at '{weightsPath}' ended unexpectedly while reading - the file is " +
                "corrupted despite matching the expected size. Retrain to regenerate the checkpoint.", ex);
        }

        return (model, tokenizer);
    }
}
