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
        var configJson = File.ReadAllText(Path.Combine(directory, ConfigFileName));
        var config = JsonSerializer.Deserialize<GptConfig>(configJson)
                     ?? throw new InvalidDataException("Could not parse config.json");

        var tokenizer = CharTokenizer.LoadVocab(Path.Combine(directory, VocabFileName));

        // rng only seeds initial values, which are about to be overwritten by the checkpoint.
        var model = new GptModel(config, new Random(0));

        using var stream = File.OpenRead(Path.Combine(directory, WeightsFileName));
        using var reader = new BinaryReader(stream);
        foreach (var p in model.Parameters())
        {
            for (int i = 0; i < p.Data.Length; i++) p.Data[i] = reader.ReadSingle();
        }

        return (model, tokenizer);
    }
}
