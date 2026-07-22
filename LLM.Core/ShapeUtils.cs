namespace LLM.Core;

/// <summary>
/// Pure helper functions for row-major shape/stride math and numpy-style broadcasting.
/// No autograd here - just index arithmetic shared by the Tensor ops.
/// </summary>
public static class ShapeUtils
{
    public static int ElementCount(int[] shape)
    {
        int n = 1;
        foreach (var s in shape) n *= s;
        return n;
    }

    /// <summary>Row-major contiguous strides for a given shape.</summary>
    public static int[] ContiguousStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    public static int[] UnravelIndex(int flat, int[] shape)
    {
        var idx = new int[shape.Length];
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            idx[i] = flat % shape[i];
            flat /= shape[i];
        }
        return idx;
    }

    public static int RavelIndex(int[] idx, int[] strides)
    {
        int flat = 0;
        for (int i = 0; i < idx.Length; i++) flat += idx[i] * strides[i];
        return flat;
    }

    /// <summary>Numpy-style broadcast of two shapes (right-aligned, size-1 dims stretch).</summary>
    public static int[] BroadcastShapes(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int ai = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            int bi = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            if (ai != bi && ai != 1 && bi != 1)
                throw new InvalidOperationException(
                    $"Cannot broadcast shapes [{string.Join(",", a)}] and [{string.Join(",", b)}]");
            result[i] = Math.Max(ai, bi);
        }
        return result;
    }

    /// <summary>
    /// Strides to use when reading a tensor of shape <paramref name="shape"/> as if it had
    /// shape <paramref name="targetShape"/> (broadcasting): size-1 / missing dims get stride 0.
    /// </summary>
    public static int[] BroadcastStrides(int[] shape, int[] targetShape)
    {
        int rank = targetShape.Length;
        var ownStrides = ContiguousStrides(shape);
        var result = new int[rank];
        int offset = rank - shape.Length;
        for (int i = 0; i < rank; i++)
        {
            if (i < offset)
            {
                result[i] = 0;
            }
            else
            {
                int dimSize = shape[i - offset];
                result[i] = dimSize == 1 ? 0 : ownStrides[i - offset];
            }
        }
        return result;
    }
}
