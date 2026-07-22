using LLM.Core;

namespace LLM.Training;

/// <summary>Global-norm gradient clipping, applied across all parameters together (the
/// standard approach for transformer training stability).</summary>
public static class GradClip
{
    public static void ClipGlobalNorm(IEnumerable<Tensor> parameters, float maxNorm)
    {
        double sumSquares = 0;
        var list = parameters as IList<Tensor> ?? parameters.ToList();
        foreach (var p in list)
        {
            if (p.Grad is null) continue;
            foreach (var g in p.Grad) sumSquares += (double)g * g;
        }

        double norm = Math.Sqrt(sumSquares);
        if (norm <= maxNorm || norm == 0) return;

        float scale = (float)(maxNorm / norm);
        foreach (var p in list)
        {
            if (p.Grad is null) continue;
            for (int i = 0; i < p.Grad.Length; i++) p.Grad[i] *= scale;
        }
    }
}
