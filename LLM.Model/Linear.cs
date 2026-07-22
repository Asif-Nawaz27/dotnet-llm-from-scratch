using LLM.Core;

namespace LLM.Model;

/// <summary>y = x @ W + b. Weight is stored as (in, out) - i.e. already "transposed" relative
/// to the PyTorch convention - so no Transpose op is needed on the forward path.
/// Works directly on both (N, in) and (B, T, in) inputs: <see cref="TensorOps.MatMul"/>
/// broadcasts a rank-2 weight across any leading batch dims of x.</summary>
public sealed class Linear
{
    public readonly Tensor Weight; // (in, out)
    public readonly Tensor? Bias;  // (out,)

    public Linear(int inFeatures, int outFeatures, Random rng, bool bias = true)
    {
        float std = 1f / MathF.Sqrt(inFeatures);
        Weight = Tensor.Randn(new[] { inFeatures, outFeatures }, std, rng);
        Bias = bias ? Tensor.Zeros(outFeatures) : null;
        if (Bias is not null) Bias.RequiresGrad = true;
    }

    public Tensor Forward(Tensor x)
    {
        var y = x.MatMul(Weight);
        return Bias is null ? y : y + Bias;
    }

    public IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
        if (Bias is not null) yield return Bias;
    }
}
