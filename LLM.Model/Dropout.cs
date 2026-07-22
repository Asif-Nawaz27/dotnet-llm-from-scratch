using LLM.Core;

namespace LLM.Model;

/// <summary>Inverted dropout: at train time, zeroes elements with probability p and rescales
/// survivors by 1/(1-p); at eval time it's the identity. Implemented as a plain elementwise
/// multiply by a constant (non-trainable) random mask, so its backward is just Mul's.</summary>
public sealed class Dropout
{
    private readonly float _p;
    private readonly Random _rng;

    public Dropout(float p, Random rng)
    {
        _p = p;
        _rng = rng;
    }

    public Tensor Forward(Tensor x, bool training)
    {
        if (!training || _p <= 0f) return x;

        var mask = new Tensor((int[])x.Shape.Clone(), requiresGrad: false);
        float scale = 1f / (1f - _p);
        for (int i = 0; i < mask.Data.Length; i++)
            mask.Data[i] = _rng.NextDouble() >= _p ? scale : 0f;

        return x * mask;
    }
}
