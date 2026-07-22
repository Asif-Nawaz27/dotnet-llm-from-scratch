using LLM.Core;

namespace LLM.Training;

/// <summary>Adam (Kingma &amp; Ba, 2014). Standard biased-moment-estimate-with-bias-correction
/// update, applied per-parameter-tensor over its flat Data/Grad arrays.</summary>
public sealed class AdamOptimizer
{
    private readonly List<Tensor> _parameters;
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private readonly float[][] _m;
    private readonly float[][] _v;
    private int _t;

    public AdamOptimizer(
        IEnumerable<Tensor> parameters,
        float lr = 3e-4f,
        float beta1 = 0.9f,
        float beta2 = 0.95f,
        float eps = 1e-8f,
        float weightDecay = 0f)
    {
        _parameters = parameters.ToList();
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;
        _m = _parameters.Select(p => new float[p.Data.Length]).ToArray();
        _v = _parameters.Select(p => new float[p.Data.Length]).ToArray();
        _t = 0;
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters) p.ZeroGrad();
    }

    public void Step()
    {
        _t++;
        float biasCorrection1 = 1f - MathF.Pow(_beta1, _t);
        float biasCorrection2 = 1f - MathF.Pow(_beta2, _t);

        for (int pi = 0; pi < _parameters.Count; pi++)
        {
            var p = _parameters[pi];
            if (p.Grad is null) continue; // this parameter wasn't used in the last forward pass
            var m = _m[pi];
            var v = _v[pi];
            var grad = p.Grad;
            var data = p.Data;

            for (int i = 0; i < data.Length; i++)
            {
                float g = grad[i];
                if (_weightDecay > 0f) g += _weightDecay * data[i];

                m[i] = _beta1 * m[i] + (1f - _beta1) * g;
                v[i] = _beta2 * v[i] + (1f - _beta2) * g * g;

                float mHat = m[i] / biasCorrection1;
                float vHat = v[i] / biasCorrection2;

                data[i] -= _lr * mHat / (MathF.Sqrt(vHat) + _eps);
            }
        }
    }
}
