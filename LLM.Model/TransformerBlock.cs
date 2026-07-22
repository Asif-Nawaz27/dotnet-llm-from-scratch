using LLM.Core;

namespace LLM.Model;

/// <summary>One pre-norm GPT block: x + Attn(LN(x)), then x + MLP(LN(x)).</summary>
public sealed class TransformerBlock
{
    private readonly LayerNorm _ln1;
    private readonly LayerNorm _ln2;
    private readonly CausalSelfAttention _attn;
    private readonly FeedForward _mlp;

    public TransformerBlock(GptConfig cfg, Random rng)
    {
        _ln1 = new LayerNorm(cfg.NEmbd);
        _ln2 = new LayerNorm(cfg.NEmbd);
        _attn = new CausalSelfAttention(cfg, rng);
        _mlp = new FeedForward(cfg, rng);
    }

    public Tensor Forward(Tensor x, bool training)
    {
        x = x + _attn.Forward(_ln1.Forward(x), training);
        x = x + _mlp.Forward(_ln2.Forward(x), training);
        return x;
    }

    public IEnumerable<Tensor> Parameters()
        => _ln1.Parameters().Concat(_ln2.Parameters()).Concat(_attn.Parameters()).Concat(_mlp.Parameters());
}
