using LLM.Model;
using LLM.Tokenizer;

namespace LLM.Training;

/// <summary>Autoregressive sampling from a trained GptModel: repeatedly feeds the whole
/// context back in (no KV-cache - simplicity over speed, fine at this model scale),
/// takes the last position's logits, and samples with temperature + top-k filtering.</summary>
public static class TextGenerator
{
    public static string Generate(
        GptModel model,
        CharTokenizer tokenizer,
        string prompt,
        int maxNewTokens,
        float temperature,
        int topK,
        Random rng)
    {
        var ids = prompt.Length > 0 ? tokenizer.Encode(prompt).ToList() : new List<int> { rng.Next(tokenizer.VocabSize) };
        int vocab = model.Config.VocabSize;

        for (int step = 0; step < maxNewTokens; step++)
        {
            var context = ids.Count <= model.Config.BlockSize
                ? ids.ToArray()
                : ids.Skip(ids.Count - model.Config.BlockSize).ToArray();
            int t = context.Length;

            var logits = model.Forward(context, batchSize: 1, seqLen: t, training: false);

            var lastLogits = new float[vocab];
            int baseIdx = (t - 1) * vocab;
            Array.Copy(logits.Data, baseIdx, lastLogits, 0, vocab);

            for (int i = 0; i < vocab; i++) lastLogits[i] /= temperature;

            if (topK > 0 && topK < vocab)
            {
                var threshold = lastLogits.OrderByDescending(x => x).Skip(topK - 1).First();
                for (int i = 0; i < vocab; i++)
                    if (lastLogits[i] < threshold) lastLogits[i] = float.NegativeInfinity;
            }

            float max = lastLogits.Max();
            var expVals = new float[vocab];
            float sum = 0f;
            for (int i = 0; i < vocab; i++)
            {
                expVals[i] = MathF.Exp(lastLogits[i] - max);
                sum += expVals[i];
            }

            double sample = rng.NextDouble() * sum;
            double cumulative = 0;
            int chosen = vocab - 1;
            for (int i = 0; i < vocab; i++)
            {
                cumulative += expVals[i];
                if (sample <= cumulative) { chosen = i; break; }
            }

            ids.Add(chosen);
        }

        return tokenizer.Decode(ids);
    }
}
