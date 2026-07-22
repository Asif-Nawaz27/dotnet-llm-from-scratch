using LLM.Core;
using LLM.Model;
using LLM.Tokenizer;

namespace LLM.Training;

public static class Trainer
{
    public static void Train(
        GptModel model,
        DataLoader trainLoader,
        DataLoader valLoader,
        TrainingConfig cfg,
        CharTokenizer tokenizer,
        Action<string>? log = null,
        CancellationToken cancellationToken = default,
        Action<int, int>? onStep = null)
    {
        log ??= Console.WriteLine;
        var parameters = model.Parameters().ToList();
        var optimizer = new AdamOptimizer(parameters, cfg.LearningRate, cfg.Beta1, cfg.Beta2, weightDecay: cfg.WeightDecay);

        log($"Training process started...");

        for (int step = 1; step <= cfg.MaxSteps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            optimizer.ZeroGrad();

            log($"training step  {step}/{cfg.MaxSteps}");
            var (inputs, targets) = trainLoader.GetBatch(cfg.BatchSize);
            var (_, loss) = model.ForwardWithLoss(inputs, targets, cfg.BatchSize, model.Config.BlockSize);
            loss.Backward();

            GradClip.ClipGlobalNorm(parameters, cfg.GradClipNorm);
            optimizer.Step();

            // Fires every step, independent of EvalInterval - cheap (no eval-loss pass), so a
            // caller (the web UI's progress bar) can track real progress at fine granularity
            // instead of only updating whenever a full log line is emitted.
            onStep?.Invoke(step, cfg.MaxSteps);

            if (step % cfg.EvalInterval == 0 || step == cfg.MaxSteps)
            {
                log($"calculateing estimate loss...");

                float valLoss = EstimateLoss(model, valLoader, cfg.EvalIters, cfg.BatchSize);
                log($"calculated loss value...");

                log($"step {step,6}/{cfg.MaxSteps}  train_loss={loss.Data[0]:F4}  val_loss={valLoss:F4}");

                if (cfg.CheckpointDir is not null)
                {
                    Checkpoint.Save(cfg.CheckpointDir, model, tokenizer);
                    log($"Checkpoint saved to '{cfg.CheckpointDir}' (step {step}).");
                }
            }
        }
    }

    private static float EstimateLoss(GptModel model, DataLoader loader, int iters, int batchSize)
    {
        float sum = 0f;
        for (int i = 0; i < iters; i++)
        {
            var (inputs, targets) = loader.GetBatch(batchSize);
            var logits = model.Forward(inputs, batchSize, model.Config.BlockSize, training: false);
            var flatLogits = logits.Reshape(batchSize * model.Config.BlockSize, model.Config.VocabSize);
            var loss = TensorOps.CrossEntropy(flatLogits, targets);
            sum += loss.Data[0];
        }
        return sum / iters;
    }
}
