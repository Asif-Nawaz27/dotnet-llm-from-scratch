using LLM.Core;

namespace LLM.Model;

/// <summary>A small decoder-only GPT: token + positional embeddings, N pre-norm
/// transformer blocks, a final LayerNorm, and a linear head projecting back to vocab logits.</summary>
public sealed class GptModel
{
    public readonly GptConfig Config;
    private readonly Embedding _tokenEmbedding;
    private readonly Embedding _positionEmbedding;
    private readonly List<TransformerBlock> _blocks;
    private readonly LayerNorm _lnFinal;
    private readonly Linear _lmHead;
    private readonly Dropout _embedDropout;

    public GptModel(GptConfig config, Random rng)
    {
        Config = config;
        _tokenEmbedding = new Embedding(config.VocabSize, config.NEmbd, rng);
        _positionEmbedding = new Embedding(config.BlockSize, config.NEmbd, rng);
        _blocks = Enumerable.Range(0, config.NLayer).Select(_ => new TransformerBlock(config, rng)).ToList();
        _lnFinal = new LayerNorm(config.NEmbd);
        _lmHead = new Linear(config.NEmbd, config.VocabSize, rng);
        _embedDropout = new Dropout(config.DropoutProb, rng);
    }

    /// <summary>flatTokenIds is (batchSize * seqLen) row-major, i.e. row b occupies
    /// [b*seqLen, (b+1)*seqLen). Returns logits of shape (batchSize, seqLen, vocabSize).</summary>
    public Tensor Forward(int[] flatTokenIds, int batchSize, int seqLen, bool training)
    {
        if (seqLen > Config.BlockSize)
            throw new ArgumentException($"seqLen {seqLen} exceeds BlockSize {Config.BlockSize}.");

        var tok = _tokenEmbedding.Forward(flatTokenIds, batchSize, seqLen);
        var positions = Enumerable.Range(0, seqLen).ToArray();
        var pos = TensorOps.EmbeddingLookup(_positionEmbedding.Weight, positions, new[] { seqLen });

        var x = tok + pos; // (T,C) broadcasts over the batch dim of (B,T,C)
        x = _embedDropout.Forward(x, training);

        foreach (var block in _blocks) x = block.Forward(x, training);

        x = _lnFinal.Forward(x);
        return _lmHead.Forward(x); // (B, T, vocab)
    }

    /// <summary>Runs the forward pass and computes mean cross-entropy loss in one call.
    /// targets is flat (batchSize * seqLen), aligned with flatTokenIds shifted by one.</summary>
    public (Tensor Logits, Tensor Loss) ForwardWithLoss(int[] flatTokenIds, int[] targets, int batchSize, int seqLen)
    {
        var logits = Forward(flatTokenIds, batchSize, seqLen, training: true);
        var flatLogits = logits.Reshape(batchSize * seqLen, Config.VocabSize);
        var loss = TensorOps.CrossEntropy(flatLogits, targets);
        return (logits, loss);
    }

    public IEnumerable<Tensor> Parameters()
    {
        var all = _tokenEmbedding.Parameters()
            .Concat(_positionEmbedding.Parameters())
            .Concat(_lnFinal.Parameters())
            .Concat(_lmHead.Parameters());
        foreach (var block in _blocks) all = all.Concat(block.Parameters());
        return all;
    }
}
