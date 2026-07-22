using LLM.Model;
using LLM.Tokenizer;
using LLM.Training;

if (args.Length == 0)
{
    PrintUsage();
    return 1;
}

var command = args[0];
var rest = args.Skip(1).ToArray();

try
{
    switch (command)
    {
        case "train":
            return RunTrain(rest);
        case "generate":
            return RunGenerate(rest);
        default:
            Console.Error.WriteLine($"Unknown command '{command}'.");
            PrintUsage();
            return 1;
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 1;
}

static void PrintUsage()
{
    Console.WriteLine("""
        LLM.Cli - train and sample from a from-scratch GPT-style model.

        Usage:
          LLM.Cli train --data <path> --out <dir> [options]
          LLM.Cli generate --model <dir> --prompt "<text>" [options]

        train options:
          --data <path>          Path to a UTF-8 text corpus (required)
          --out <dir>            Checkpoint output directory (required)
          --steps <n>            Training steps (default 2000)
          --batch-size <n>       (default 32)
          --block-size <n>       Context length (default 128)
          --n-embd <n>           Embedding width (default 128)
          --n-head <n>           Attention heads (default 4)
          --n-layer <n>          Transformer blocks (default 4)
          --dropout <f>          Dropout probability (default 0.1)
          --lr <f>               Learning rate (default 3e-4)
          --eval-interval <n>    Steps between evals/checkpoints (default 100)
          --eval-iters <n>       Batches averaged per eval (default 20)
          --seed <n>             RNG seed (default 1337)

        generate options:
          --model <dir>          Checkpoint directory produced by `train` (required)
          --prompt <text>        Seed text (default: empty -> random start char)
          --tokens <n>           Number of new tokens to sample (default 500)
          --temperature <f>      Sampling temperature (default 0.8)
          --top-k <n>            Top-k filtering, 0 disables it (default 40)
          --seed <n>             RNG seed (default: time-based)
        """);
}

static Dictionary<string, string> ParseOptions(string[] args)
{
    var options = new Dictionary<string, string>();
    for (int i = 0; i < args.Length; i++)
    {
        if (!args[i].StartsWith("--")) continue;
        var key = args[i][2..];
        string value = i + 1 < args.Length && !args[i + 1].StartsWith("--") ? args[++i] : "true";
        options[key] = value;
    }
    return options;
}

static int RunTrain(string[] args)
{
    var opts = ParseOptions(args);

    if (!opts.TryGetValue("data", out var dataPath))
    {
        Console.Error.WriteLine("--data <path> is required.");
        return 1;
    }
    if (!opts.TryGetValue("out", out var outDir))
    {
        Console.Error.WriteLine("--out <dir> is required.");
        return 1;
    }

    int steps = int.Parse(opts.GetValueOrDefault("steps", "2000"));
    int batchSize = int.Parse(opts.GetValueOrDefault("batch-size", "32"));
    int blockSize = int.Parse(opts.GetValueOrDefault("block-size", "128"));
    int nEmbd = int.Parse(opts.GetValueOrDefault("n-embd", "128"));
    int nHead = int.Parse(opts.GetValueOrDefault("n-head", "4"));
    int nLayer = int.Parse(opts.GetValueOrDefault("n-layer", "4"));
    float dropout = float.Parse(opts.GetValueOrDefault("dropout", "0.1"));
    float lr = float.Parse(opts.GetValueOrDefault("lr", "0.0003"));
    int evalInterval = int.Parse(opts.GetValueOrDefault("eval-interval", "100"));
    int evalIters = int.Parse(opts.GetValueOrDefault("eval-iters", "20"));
    int seed = int.Parse(opts.GetValueOrDefault("seed", "1337"));

    var text = File.ReadAllText(dataPath);
    Console.WriteLine($"Loaded corpus: {text.Length} characters.");

    var tokenizer = CharTokenizer.BuildFromCorpus(text);
    Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize} unique characters.");

    var allIds = tokenizer.Encode(text);
    int splitPoint = (int)(allIds.Length * 0.9);
    var trainIds = allIds[..splitPoint];
    var valIds = allIds[splitPoint..];
    Console.WriteLine($"Train tokens: {trainIds.Length}, Val tokens: {valIds.Length}");

    var config = new GptConfig
    {
        VocabSize = tokenizer.VocabSize,
        BlockSize = blockSize,
        NEmbd = nEmbd,
        NHead = nHead,
        NLayer = nLayer,
        DropoutProb = dropout,
    };

    var rng = new Random(seed);
    var model = new GptModel(config, rng);
    int paramCount = model.Parameters().Sum(p => p.Size);
    Console.WriteLine($"Model parameters: {paramCount:N0}");

    var trainLoader = new DataLoader(trainIds, blockSize, new Random(seed + 1));
    var valLoader = new DataLoader(valIds, blockSize, new Random(seed + 2));

    var trainingConfig = new TrainingConfig
    {
        BatchSize = batchSize,
        MaxSteps = steps,
        EvalInterval = evalInterval,
        EvalIters = evalIters,
        LearningRate = lr,
        CheckpointDir = outDir,
    };

    Trainer.Train(model, trainLoader, valLoader, trainingConfig, tokenizer);

    Console.WriteLine($"Done. Checkpoint saved to '{outDir}'.");
    return 0;
}

static int RunGenerate(string[] args)
{
    var opts = ParseOptions(args);

    if (!opts.TryGetValue("model", out var modelDir))
    {
        Console.Error.WriteLine("--model <dir> is required.");
        return 1;
    }

    string prompt = opts.GetValueOrDefault("prompt", "");
    int maxNewTokens = int.Parse(opts.GetValueOrDefault("tokens", "500"));
    float temperature = float.Parse(opts.GetValueOrDefault("temperature", "0.8"));
    int topK = int.Parse(opts.GetValueOrDefault("top-k", "40"));
    var rng = opts.TryGetValue("seed", out var seedStr) ? new Random(int.Parse(seedStr)) : new Random();

    var (model, tokenizer) = Checkpoint.Load(modelDir);
    Console.WriteLine($"Loaded model: {model.Parameters().Sum(p => p.Size):N0} parameters, vocab={tokenizer.VocabSize}.");

    var generated = TextGenerator.Generate(model, tokenizer, prompt, maxNewTokens, temperature, topK, rng);
    Console.WriteLine("----- generated text -----");
    Console.WriteLine(generated);
    return 0;
}
