using LLM.Core;

namespace LLM.Model;

/// <summary>A lookup table of <see cref="NumEmbeddings"/> rows of width <see cref="Dim"/>.
/// Used for both token embeddings and learned positional embeddings.</summary>
public sealed class Embedding
{
    public readonly Tensor Weight; // (numEmbeddings, dim)
    public int Dim { get; }

    public Embedding(int numEmbeddings, int dim, Random rng)
    {
        Dim = dim;
        Weight = Tensor.Randn(new[] { numEmbeddings, dim }, std: 0.02f, rng);
    }

    /// <summary>Looks up rows for a batch of sequences, indices shape (B, T) flattened row-major.</summary>
    public Tensor Forward(int[] flatIndices, int batchSize, int seqLen)
        => TensorOps.EmbeddingLookup(Weight, flatIndices, new[] { batchSize, seqLen });

    public IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
    }
}
