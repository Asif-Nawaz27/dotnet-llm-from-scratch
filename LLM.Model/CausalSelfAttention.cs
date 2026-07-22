using LLM.Core;

namespace LLM.Model;

/// <summary>Multi-head causal self-attention. Q/K/V use separate Linear projections
/// (rather than one fused QKV matrix) purely to avoid needing a tensor-split op in the
/// engine - functionally identical to the fused version.</summary>
public sealed class CausalSelfAttention
{
    private readonly int _nHead;
    private readonly int _headDim;
    private readonly Linear _query, _key, _value, _outProj;
    private readonly Dropout _attnDropout, _residDropout;
    private readonly Dictionary<int, Tensor> _maskCache = new();

    public CausalSelfAttention(GptConfig cfg, Random rng)
    {
        if (cfg.NEmbd % cfg.NHead != 0)
            throw new ArgumentException("NEmbd must be divisible by NHead.");
        _nHead = cfg.NHead;
        _headDim = cfg.HeadDim;

        _query = new Linear(cfg.NEmbd, cfg.NEmbd, rng);
        _key = new Linear(cfg.NEmbd, cfg.NEmbd, rng);
        _value = new Linear(cfg.NEmbd, cfg.NEmbd, rng);
        _outProj = new Linear(cfg.NEmbd, cfg.NEmbd, rng);
        _attnDropout = new Dropout(cfg.DropoutProb, rng);
        _residDropout = new Dropout(cfg.DropoutProb, rng);
    }

    public Tensor Forward(Tensor x, bool training)
    {
        int b = x.Shape[0], t = x.Shape[1], c = x.Shape[2];

        var q = SplitHeads(_query.Forward(x), b, t, c);
        var k = SplitHeads(_key.Forward(x), b, t, c);
        var v = SplitHeads(_value.Forward(x), b, t, c);

        var kT = k.Transpose(2, 3); // (B, nHead, headDim, T)
        var scores = q.MatMul(kT) * (1f / MathF.Sqrt(_headDim)); // (B, nHead, T, T)

        var mask = GetCausalMask(t);
        var masked = scores + mask;
        var probs = masked.Softmax();
        probs = _attnDropout.Forward(probs, training);

        var attnOut = probs.MatMul(v); // (B, nHead, T, headDim)
        var merged = MergeHeads(attnOut, b, t, c);

        var output = _outProj.Forward(merged);
        return _residDropout.Forward(output, training);
    }

    private Tensor SplitHeads(Tensor x, int b, int t, int c)
        => x.Reshape(b, t, _nHead, _headDim).Permute(0, 2, 1, 3);

    private Tensor MergeHeads(Tensor x, int b, int t, int c)
        => x.Permute(0, 2, 1, 3).Reshape(b, t, c);

    private Tensor GetCausalMask(int t)
    {
        if (!_maskCache.TryGetValue(t, out var mask))
        {
            mask = TensorOps.CausalMask(t);
            _maskCache[t] = mask;
        }
        return mask;
    }

    public IEnumerable<Tensor> Parameters()
        => _query.Parameters().Concat(_key.Parameters()).Concat(_value.Parameters()).Concat(_outProj.Parameters());
}
