using LLM.Core;
using LLM.Model;
using Xunit;

namespace LLM.Core.Tests;

/// <summary>End-to-end sanity checks on the full model: does a forward+backward pass run
/// without shape errors, do gradients reach every parameter, and does loss actually go
/// down when we step a few times on a tiny fixed batch (the cheapest possible "does the
/// wiring work at all" signal, distinct from the primitive-level GradientCheckTests).</summary>
public class GptModelSanityTests
{
    private static GptConfig SmallConfig() => new()
    {
        VocabSize = 13,
        BlockSize = 8,
        NEmbd = 16,
        NHead = 4,
        NLayer = 2,
        DropoutProb = 0f, // deterministic for this test
    };

    [Fact]
    public void ForwardBackward_ProducesGradientsForEveryParameter()
    {
        var cfg = SmallConfig();
        var rng = new Random(42);
        var model = new GptModel(cfg, rng);

        int batchSize = 2, seqLen = 5;
        var tokenRng = new Random(1);
        var flatTokens = Enumerable.Range(0, batchSize * seqLen).Select(_ => tokenRng.Next(cfg.VocabSize)).ToArray();
        var targets = Enumerable.Range(0, batchSize * seqLen).Select(_ => tokenRng.Next(cfg.VocabSize)).ToArray();

        var (logits, loss) = model.ForwardWithLoss(flatTokens, targets, batchSize, seqLen);

        Assert.Equal(new[] { batchSize, seqLen, cfg.VocabSize }, logits.Shape);
        Assert.Equal(1, loss.Size);
        Assert.False(float.IsNaN(loss.Data[0]));

        loss.Backward();

        var parameters = model.Parameters().ToList();
        Assert.True(parameters.Count > 0);
        foreach (var p in parameters)
        {
            Assert.NotNull(p.Grad);
            Assert.True(p.Grad!.Any(g => g != 0f), "Expected at least one non-zero gradient per parameter tensor.");
        }
    }

    [Fact]
    public void TrainingSteps_ReduceLossOnFixedBatch()
    {
        var cfg = SmallConfig();
        var rng = new Random(7);
        var model = new GptModel(cfg, rng);
        var parameters = model.Parameters().ToList();

        int batchSize = 4, seqLen = 6;
        var tokenRng = new Random(2);
        var flatTokens = Enumerable.Range(0, batchSize * seqLen).Select(_ => tokenRng.Next(cfg.VocabSize)).ToArray();
        var targets = Enumerable.Range(0, batchSize * seqLen).Select(_ => tokenRng.Next(cfg.VocabSize)).ToArray();

        float lr = 0.05f;
        float firstLoss = 0f, lastLoss = 0f;

        for (int step = 0; step < 30; step++)
        {
            foreach (var p in parameters) p.ZeroGrad();
            var (_, loss) = model.ForwardWithLoss(flatTokens, targets, batchSize, seqLen);
            loss.Backward();
            foreach (var p in parameters)
                for (int i = 0; i < p.Data.Length; i++)
                    p.Data[i] -= lr * p.Grad![i];

            if (step == 0) firstLoss = loss.Data[0];
            lastLoss = loss.Data[0];
        }

        Assert.True(lastLoss < firstLoss * 0.5f,
            $"Expected loss to drop substantially by overfitting a fixed tiny batch: first={firstLoss}, last={lastLoss}");
    }
}
