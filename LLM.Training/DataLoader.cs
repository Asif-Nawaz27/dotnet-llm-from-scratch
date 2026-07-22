namespace LLM.Training;

/// <summary>Samples random contiguous windows from a tokenized corpus for
/// next-token-prediction training: target[i] = input[i+1].</summary>
public sealed class DataLoader
{
    private readonly int[] _data;
    private readonly int _blockSize;
    private readonly Random _rng;

    public DataLoader(int[] data, int blockSize, Random rng)
    {
        if (data.Length <= blockSize)
            throw new ArgumentException(
                $"Dataset of {data.Length} tokens is too small for block size {blockSize}; need at least {blockSize + 1} tokens.");
        _data = data;
        _blockSize = blockSize;
        _rng = rng;
    }

    /// <summary>Returns (inputs, targets), each flat arrays of length batchSize * blockSize,
    /// row b occupying [b*blockSize, (b+1)*blockSize).</summary>
    public (int[] Inputs, int[] Targets) GetBatch(int batchSize)
    {
        var inputs = new int[batchSize * _blockSize];
        var targets = new int[batchSize * _blockSize];

        for (int b = 0; b < batchSize; b++)
        {
            int start = _rng.Next(0, _data.Length - _blockSize - 1);
            int rowOffset = b * _blockSize;
            for (int t = 0; t < _blockSize; t++)
            {
                inputs[rowOffset + t] = _data[start + t];
                targets[rowOffset + t] = _data[start + t + 1];
            }
        }

        return (inputs, targets);
    }
}
