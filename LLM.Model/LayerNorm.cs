using LLM.Core;

namespace LLM.Model;

/// <summary>Standard LayerNorm over the last dimension, built purely by composing
/// primitive Tensor ops (Mean, Sub, Mul, Pow) so its gradient falls out of the engine's
/// chain rule instead of a hand-derived fused backward.</summary>
public sealed class LayerNorm
{
    public readonly Tensor Gamma; // (dim,)
    public readonly Tensor Beta;  // (dim,)
    private readonly float _eps;

    public LayerNorm(int dim, float eps = 1e-5f)
    {
        Gamma = Tensor.Full(1f, new[] { dim });
        Gamma.RequiresGrad = true;
        Beta = Tensor.Zeros(dim);
        Beta.RequiresGrad = true;
        _eps = eps;
    }

    public Tensor Forward(Tensor x)
    {
        int axis = x.NDim - 1;
        var mean = x.Mean(axis, keepdim: true);
        var centered = x - mean;
        var variance = (centered * centered).Mean(axis, keepdim: true);
        var std = (variance + _eps).Pow(0.5f);
        var normalized = centered * std.Pow(-1f);
        return normalized * Gamma + Beta;
    }

    public IEnumerable<Tensor> Parameters()
    {
        yield return Gamma;
        yield return Beta;
    }
}
