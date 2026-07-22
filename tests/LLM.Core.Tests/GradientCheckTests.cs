using LLM.Core;
using Xunit;

namespace LLM.Core.Tests;

/// <summary>
/// Numerical gradient checking: for every primitive op, perturb each input element by
/// +/- epsilon, measure the change in a scalar loss, and compare against the analytic
/// gradient produced by Backward(). This is the standard way to validate an autograd
/// engine and catches sign errors / wrong Jacobians immediately.
/// </summary>
public class GradientCheckTests
{
    private const float Eps = 1e-3f;
    private const float Tol = 5e-2f; // finite-diff on float32 is noisy; loose but effective tolerance

    private static Random Rng(int seed) => new(seed);

    private static Tensor RandomTensor(int[] shape, Random rng, bool requiresGrad = true)
    {
        var t = new Tensor(shape, requiresGrad);
        for (int i = 0; i < t.Data.Length; i++) t.Data[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>Builds the graph fresh each time (numerical diff needs independent forward passes).</summary>
    private static void CheckGradient(Func<Tensor[], Tensor> buildLoss, Tensor[] inputs)
    {
        // Analytic gradient.
        foreach (var t in inputs) t.Grad = null;
        var loss = buildLoss(inputs);
        loss.Backward();
        var analytic = inputs.Select(t => (float[])t.Grad!.Clone()).ToArray();

        // Numerical gradient via central differences.
        for (int ti = 0; ti < inputs.Length; ti++)
        {
            var t = inputs[ti];
            for (int i = 0; i < t.Data.Length; i++)
            {
                float orig = t.Data[i];

                t.Data[i] = orig + Eps;
                float lossPlus = buildLoss(inputs).Data[0];

                t.Data[i] = orig - Eps;
                float lossMinus = buildLoss(inputs).Data[0];

                t.Data[i] = orig;

                float numGrad = (lossPlus - lossMinus) / (2 * Eps);
                float analyticGrad = analytic[ti][i];

                Assert.True(Math.Abs(numGrad - analyticGrad) < Tol,
                    $"Gradient mismatch for input {ti}, element {i}: numeric={numGrad}, analytic={analyticGrad}");
            }
        }
    }

    private static Tensor SumAll(Tensor t)
    {
        // Flatten to 1D then reduce the only axis - collapses any shape to a true scalar
        // (shape []) using only ops already defined on TensorOps.
        var flat = TensorOps.Reshape(t, t.Size);
        return TensorOps.Sum(flat, axis: 0, keepdim: false);
    }

    private static Tensor ScalarLoss(Tensor t)
    {
        // Sum of squares collapses any shape to a scalar without introducing new op types.
        return SumAll(TensorOps.Mul(t, t));
    }

    [Fact]
    public void Add_Broadcast_GradientMatchesNumeric()
    {
        var rng = Rng(1);
        var a = RandomTensor(new[] { 3, 4 }, rng);
        var b = RandomTensor(new[] { 4 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.Add(inputs[0], inputs[1])), new[] { a, b });
    }

    [Fact]
    public void Mul_Broadcast_GradientMatchesNumeric()
    {
        var rng = Rng(2);
        var a = RandomTensor(new[] { 2, 3 }, rng);
        var b = RandomTensor(new[] { 3 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.Mul(inputs[0], inputs[1])), new[] { a, b });
    }

    [Fact]
    public void MatMul_GradientMatchesNumeric()
    {
        var rng = Rng(3);
        var a = RandomTensor(new[] { 2, 3 }, rng);
        var b = RandomTensor(new[] { 3, 4 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.MatMul(inputs[0], inputs[1])), new[] { a, b });
    }

    [Fact]
    public void MatMul_Batched_GradientMatchesNumeric()
    {
        var rng = Rng(4);
        var a = RandomTensor(new[] { 2, 3, 4 }, rng);
        var b = RandomTensor(new[] { 2, 4, 5 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.MatMul(inputs[0], inputs[1])), new[] { a, b });
    }

    [Fact]
    public void Pow_GradientMatchesNumeric()
    {
        var rng = Rng(5);
        var a = RandomTensor(new[] { 5 }, rng);
        for (int i = 0; i < a.Data.Length; i++) a.Data[i] = MathF.Abs(a.Data[i]) + 0.5f; // keep base positive
        CheckGradient(inputs => ScalarLoss(TensorOps.PowScalar(inputs[0], 1.5f)), new[] { a });
    }

    [Fact]
    public void Exp_GradientMatchesNumeric()
    {
        var rng = Rng(6);
        var a = RandomTensor(new[] { 5 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.Exp(inputs[0])), new[] { a });
    }

    [Fact]
    public void Tanh_GradientMatchesNumeric()
    {
        var rng = Rng(7);
        var a = RandomTensor(new[] { 5 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.Tanh(inputs[0])), new[] { a });
    }

    [Fact]
    public void Sum_Axis_GradientMatchesNumeric()
    {
        var rng = Rng(8);
        var a = RandomTensor(new[] { 3, 4 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.Sum(inputs[0], axis: 1)), new[] { a });
    }

    [Fact]
    public void Reshape_GradientMatchesNumeric()
    {
        var rng = Rng(9);
        var a = RandomTensor(new[] { 2, 6 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.Reshape(inputs[0], 3, 4)), new[] { a });
    }

    [Fact]
    public void Permute_GradientMatchesNumeric()
    {
        var rng = Rng(10);
        var a = RandomTensor(new[] { 2, 3, 4 }, rng);
        CheckGradient(inputs => ScalarLoss(TensorOps.Permute(inputs[0], 2, 0, 1)), new[] { a });
    }

    [Fact]
    public void Softmax_GradientMatchesNumeric()
    {
        var rng = Rng(11);
        var a = RandomTensor(new[] { 3, 5 }, rng);
        // Weight softmax output by random targets so the loss isn't degenerate (sum(softmax)==1 always).
        var weights = RandomTensor(new[] { 3, 5 }, rng, requiresGrad: false);
        CheckGradient(inputs => ScalarLoss(TensorOps.Mul(TensorOps.Softmax(inputs[0]), weights)), new[] { a });
    }

    [Fact]
    public void CrossEntropy_GradientMatchesNumeric()
    {
        var rng = Rng(12);
        var logits = RandomTensor(new[] { 4, 6 }, rng);
        var targets = new[] { 0, 5, 2, 3 };
        CheckGradient(inputs => TensorOps.CrossEntropy(inputs[0], targets), new[] { logits });
    }

    [Fact]
    public void EmbeddingLookup_GradientMatchesNumeric()
    {
        var rng = Rng(13);
        var weight = RandomTensor(new[] { 6, 4 }, rng);
        var indices = new[] { 0, 2, 2, 5 };
        CheckGradient(inputs => ScalarLoss(TensorOps.EmbeddingLookup(inputs[0], indices, new[] { 4 })), new[] { weight });
    }

    [Fact]
    public void ComposedLayerNormLikeExpression_GradientMatchesNumeric()
    {
        // mean/var/normalize composition, exactly as used by the model's LayerNorm layer.
        var rng = Rng(14);
        var x = RandomTensor(new[] { 3, 8 }, rng);
        var gamma = RandomTensor(new[] { 8 }, rng);
        var beta = RandomTensor(new[] { 8 }, rng);

        Tensor BuildNorm(Tensor[] inputs)
        {
            var xi = inputs[0];
            var g = inputs[1];
            var b = inputs[2];
            var mu = TensorOps.Mean(xi, axis: 1, keepdim: true);
            var xc = TensorOps.Sub(xi, mu);
            var variance = TensorOps.Mean(TensorOps.Mul(xc, xc), axis: 1, keepdim: true);
            var std = TensorOps.PowScalar(TensorOps.AddScalar(variance, 1e-5f), 0.5f);
            var norm = TensorOps.Mul(xc, TensorOps.PowScalar(std, -1f));
            var scaled = TensorOps.Add(TensorOps.Mul(norm, g), b);
            return ScalarLoss(scaled);
        }

        CheckGradient(BuildNorm, new[] { x, gamma, beta });
    }
}
