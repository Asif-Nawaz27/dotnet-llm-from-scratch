namespace LLM.Core;

/// <summary>
/// Primitive differentiable operations on <see cref="Tensor"/>. Everything higher up
/// the stack (LayerNorm, GELU, attention, ...) is built by composing these, so each
/// gradient formula here only needs to be correct once. Softmax and CrossEntropy are
/// the two exceptions: they are implemented as fused ops with their well-known
/// closed-form gradients, mainly for numerical stability.
/// </summary>
public static class TensorOps
{
    // ---------------------------------------------------------------- Add / Sub / Mul

    public static Tensor Add(Tensor a, Tensor b)
    {
        var outShape = ShapeUtils.BroadcastShapes(a.Shape, b.Shape);
        var aStrides = ShapeUtils.BroadcastStrides(a.Shape, outShape);
        var bStrides = ShapeUtils.BroadcastStrides(b.Shape, outShape);
        int n = ShapeUtils.ElementCount(outShape);
        var outData = new float[n];

        for (int flat = 0; flat < n; flat++)
        {
            var idx = ShapeUtils.UnravelIndex(flat, outShape);
            int aFlat = ShapeUtils.RavelIndex(idx, aStrides);
            int bFlat = ShapeUtils.RavelIndex(idx, bStrides);
            outData[flat] = a.Data[aFlat] + b.Data[bFlat];
        }

        var result = new Tensor(outData, outShape, a.RequiresGrad || b.RequiresGrad, "add");
        result.Prev.Add(a);
        result.Prev.Add(b);
        result.BackwardOp = () =>
        {
            var outGrad = result.Grad!;
            if (a.RequiresGrad)
            {
                var aGrad = a.GetOrInitGrad();
                for (int flat = 0; flat < n; flat++)
                {
                    var idx = ShapeUtils.UnravelIndex(flat, outShape);
                    int aFlat = ShapeUtils.RavelIndex(idx, aStrides);
                    aGrad[aFlat] += outGrad[flat];
                }
            }
            if (b.RequiresGrad)
            {
                var bGrad = b.GetOrInitGrad();
                for (int flat = 0; flat < n; flat++)
                {
                    var idx = ShapeUtils.UnravelIndex(flat, outShape);
                    int bFlat = ShapeUtils.RavelIndex(idx, bStrides);
                    bGrad[bFlat] += outGrad[flat];
                }
            }
        };
        return result;
    }

    public static Tensor Mul(Tensor a, Tensor b)
    {
        var outShape = ShapeUtils.BroadcastShapes(a.Shape, b.Shape);
        var aStrides = ShapeUtils.BroadcastStrides(a.Shape, outShape);
        var bStrides = ShapeUtils.BroadcastStrides(b.Shape, outShape);
        int n = ShapeUtils.ElementCount(outShape);
        var outData = new float[n];

        for (int flat = 0; flat < n; flat++)
        {
            var idx = ShapeUtils.UnravelIndex(flat, outShape);
            int aFlat = ShapeUtils.RavelIndex(idx, aStrides);
            int bFlat = ShapeUtils.RavelIndex(idx, bStrides);
            outData[flat] = a.Data[aFlat] * b.Data[bFlat];
        }

        var result = new Tensor(outData, outShape, a.RequiresGrad || b.RequiresGrad, "mul");
        result.Prev.Add(a);
        result.Prev.Add(b);
        result.BackwardOp = () =>
        {
            var outGrad = result.Grad!;
            if (a.RequiresGrad)
            {
                var aGrad = a.GetOrInitGrad();
                for (int flat = 0; flat < n; flat++)
                {
                    var idx = ShapeUtils.UnravelIndex(flat, outShape);
                    int aFlat = ShapeUtils.RavelIndex(idx, aStrides);
                    int bFlat = ShapeUtils.RavelIndex(idx, bStrides);
                    aGrad[aFlat] += outGrad[flat] * b.Data[bFlat];
                }
            }
            if (b.RequiresGrad)
            {
                var bGrad = b.GetOrInitGrad();
                for (int flat = 0; flat < n; flat++)
                {
                    var idx = ShapeUtils.UnravelIndex(flat, outShape);
                    int aFlat = ShapeUtils.RavelIndex(idx, aStrides);
                    int bFlat = ShapeUtils.RavelIndex(idx, bStrides);
                    bGrad[bFlat] += outGrad[flat] * a.Data[aFlat];
                }
            }
        };
        return result;
    }

    public static Tensor Neg(Tensor a) => MulScalar(a, -1f);

    public static Tensor Sub(Tensor a, Tensor b) => Add(a, Neg(b));

    // ---------------------------------------------------------------- Scalar ops

    public static Tensor AddScalar(Tensor a, float scalar) => UnaryOp(a, x => x + scalar, (_, _) => 1f, "add_scalar");

    public static Tensor MulScalar(Tensor a, float scalar) => UnaryOp(a, x => x * scalar, (_, _) => scalar, "mul_scalar");

    public static Tensor PowScalar(Tensor a, float p) =>
        UnaryOp(a, x => MathF.Pow(x, p), (x, _) => p * MathF.Pow(x, p - 1f), "pow_scalar");

    public static Tensor Exp(Tensor a) => UnaryOp(a, MathF.Exp, (_, y) => y, "exp");

    public static Tensor Log(Tensor a) => UnaryOp(a, MathF.Log, (x, _) => 1f / x, "log");

    public static Tensor Tanh(Tensor a) => UnaryOp(a, MathF.Tanh, (_, y) => 1f - y * y, "tanh");

    public static Tensor Relu(Tensor a) => UnaryOp(a, x => MathF.Max(0f, x), (x, _) => x > 0f ? 1f : 0f, "relu");

    private static Tensor UnaryOp(Tensor a, Func<float, float> fwd, Func<float, float, float> gradOfXY, string op)
    {
        var outData = new float[a.Size];
        for (int i = 0; i < outData.Length; i++) outData[i] = fwd(a.Data[i]);

        var result = new Tensor(outData, (int[])a.Shape.Clone(), a.RequiresGrad, op);
        result.Prev.Add(a);
        result.BackwardOp = () =>
        {
            if (!a.RequiresGrad) return;
            var aGrad = a.GetOrInitGrad();
            var outGrad = result.Grad!;
            for (int i = 0; i < outGrad.Length; i++)
                aGrad[i] += outGrad[i] * gradOfXY(a.Data[i], outData[i]);
        };
        return result;
    }

    // ---------------------------------------------------------------- MatMul (batched)

    /// <summary>
    /// Batched matrix multiply on the trailing two dims: (...,M,K) x (...,K,N) -> (...,M,N).
    /// Either operand may omit the batch dims entirely (rank 2), in which case it is
    /// shared/broadcast across every batch of the other operand (this covers the common
    /// "Linear layer applied to a batch of sequences" case). Otherwise batch dims must match.
    /// </summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        int aRank = a.NDim, bRank = b.NDim;
        if (aRank < 2 || bRank < 2)
            throw new ArgumentException("MatMul requires tensors of rank >= 2.");

        int M = a.Shape[aRank - 2], K = a.Shape[aRank - 1];
        int K2 = b.Shape[bRank - 2], N = b.Shape[bRank - 1];
        if (K != K2)
            throw new ArgumentException($"MatMul inner dims mismatch: {K} vs {K2}");

        int[] batchShape;
        bool aShared = aRank == 2 && bRank > 2;
        bool bShared = bRank == 2 && aRank > 2;

        if (aShared) batchShape = b.Shape[..(bRank - 2)];
        else if (bShared) batchShape = a.Shape[..(aRank - 2)];
        else
        {
            var batchA = a.Shape[..(aRank - 2)];
            var batchB = b.Shape[..(bRank - 2)];
            if (!batchA.SequenceEqual(batchB))
                throw new ArgumentException(
                    $"MatMul batch dims mismatch: [{string.Join(",", batchA)}] vs [{string.Join(",", batchB)}]");
            batchShape = batchA;
        }

        int batchSize = ShapeUtils.ElementCount(batchShape);
        var outShape = batchShape.Concat(new[] { M, N }).ToArray();
        var outData = new float[batchSize * M * N];

        for (int bi = 0; bi < batchSize; bi++)
        {
            int offA = aShared ? 0 : bi * M * K;
            int offB = bShared ? 0 : bi * K * N;
            int offOut = bi * M * N;
            MatMul2D(a.Data, offA, b.Data, offB, outData, offOut, M, K, N);
        }

        var result = new Tensor(outData, outShape, a.RequiresGrad || b.RequiresGrad, "matmul");
        result.Prev.Add(a);
        result.Prev.Add(b);
        result.BackwardOp = () =>
        {
            var outGrad = result.Grad!;
            float[]? aGrad = a.RequiresGrad ? a.GetOrInitGrad() : null;
            float[]? bGrad = b.RequiresGrad ? b.GetOrInitGrad() : null;

            for (int bi = 0; bi < batchSize; bi++)
            {
                int offA = aShared ? 0 : bi * M * K;
                int offB = bShared ? 0 : bi * K * N;
                int offOut = bi * M * N;

                if (aGrad is not null)
                {
                    // dA[m,k] += sum_n outGrad[m,n] * B[k,n]
                    for (int m = 0; m < M; m++)
                    for (int k = 0; k < K; k++)
                    {
                        float acc = 0f;
                        for (int n = 0; n < N; n++)
                            acc += outGrad[offOut + m * N + n] * b.Data[offB + k * N + n];
                        aGrad[offA + m * K + k] += acc;
                    }
                }

                if (bGrad is not null)
                {
                    // dB[k,n] += sum_m outGrad[m,n] * A[m,k]
                    for (int k = 0; k < K; k++)
                    for (int n = 0; n < N; n++)
                    {
                        float acc = 0f;
                        for (int m = 0; m < M; m++)
                            acc += outGrad[offOut + m * N + n] * a.Data[offA + m * K + k];
                        bGrad[offB + k * N + n] += acc;
                    }
                }
            }
        };
        return result;
    }

    private static void MatMul2D(float[] a, int offA, float[] b, int offB, float[] outData, int offOut, int M, int K, int N)
    {
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++)
                    acc += a[offA + m * K + k] * b[offB + k * N + n];
                outData[offOut + m * N + n] = acc;
            }
        }
    }

    // ---------------------------------------------------------------- Reductions

    public static Tensor Sum(Tensor a, int axis, bool keepdim = false)
    {
        if (axis < 0) axis += a.NDim;
        var inShape = a.Shape;
        var outShapeKeep = (int[])inShape.Clone();
        outShapeKeep[axis] = 1;
        var outStridesKeep = ShapeUtils.ContiguousStrides(outShapeKeep);
        int outSize = ShapeUtils.ElementCount(outShapeKeep);
        var outData = new float[outSize];
        int totalIn = a.Size;

        for (int flat = 0; flat < totalIn; flat++)
        {
            var idx = ShapeUtils.UnravelIndex(flat, inShape);
            idx[axis] = 0;
            int outFlat = ShapeUtils.RavelIndex(idx, outStridesKeep);
            outData[outFlat] += a.Data[flat];
        }

        int[] finalShape = keepdim ? outShapeKeep : RemoveAxis(outShapeKeep, axis);
        var result = new Tensor(outData, finalShape, a.RequiresGrad, "sum");
        result.Prev.Add(a);
        result.BackwardOp = () =>
        {
            if (!a.RequiresGrad) return;
            var aGrad = a.GetOrInitGrad();
            var outGrad = result.Grad!;
            for (int flat = 0; flat < totalIn; flat++)
            {
                var idx = ShapeUtils.UnravelIndex(flat, inShape);
                idx[axis] = 0;
                int outFlat = ShapeUtils.RavelIndex(idx, outStridesKeep);
                aGrad[flat] += outGrad[outFlat];
            }
        };
        return result;
    }

    public static Tensor Mean(Tensor a, int axis, bool keepdim = false)
    {
        if (axis < 0) axis += a.NDim;
        int count = a.Shape[axis];
        return MulScalar(Sum(a, axis, keepdim), 1f / count);
    }

    private static int[] RemoveAxis(int[] shape, int axis)
    {
        var result = new int[shape.Length - 1];
        int j = 0;
        for (int i = 0; i < shape.Length; i++)
        {
            if (i == axis) continue;
            result[j++] = shape[i];
        }
        return result;
    }

    // ---------------------------------------------------------------- Shape ops

    public static Tensor Reshape(Tensor a, params int[] newShape)
    {
        int n = ShapeUtils.ElementCount(newShape);
        if (n != a.Size)
            throw new ArgumentException(
                $"Reshape size mismatch: {a.Size} elements vs new shape [{string.Join(",", newShape)}]");

        var outData = (float[])a.Data.Clone();
        var result = new Tensor(outData, newShape, a.RequiresGrad, "reshape");
        result.Prev.Add(a);
        result.BackwardOp = () =>
        {
            if (!a.RequiresGrad) return;
            var aGrad = a.GetOrInitGrad();
            var outGrad = result.Grad!;
            for (int i = 0; i < outGrad.Length; i++) aGrad[i] += outGrad[i];
        };
        return result;
    }

    public static Tensor Permute(Tensor a, params int[] axes)
    {
        int rank = a.NDim;
        if (axes.Length != rank) throw new ArgumentException("Permute axes length must match tensor rank.");

        var inShape = a.Shape;
        var outShape = new int[rank];
        for (int i = 0; i < rank; i++) outShape[i] = inShape[axes[i]];

        var inStrides = ShapeUtils.ContiguousStrides(inShape);
        int n = a.Size;
        var outData = new float[n];

        for (int flatOut = 0; flatOut < n; flatOut++)
        {
            var outIdx = ShapeUtils.UnravelIndex(flatOut, outShape);
            var inIdx = new int[rank];
            for (int i = 0; i < rank; i++) inIdx[axes[i]] = outIdx[i];
            int flatIn = ShapeUtils.RavelIndex(inIdx, inStrides);
            outData[flatOut] = a.Data[flatIn];
        }

        var result = new Tensor(outData, outShape, a.RequiresGrad, "permute");
        result.Prev.Add(a);
        result.BackwardOp = () =>
        {
            if (!a.RequiresGrad) return;
            var aGrad = a.GetOrInitGrad();
            var outGrad = result.Grad!;
            for (int flatOut = 0; flatOut < n; flatOut++)
            {
                var outIdx = ShapeUtils.UnravelIndex(flatOut, outShape);
                var inIdx = new int[rank];
                for (int i = 0; i < rank; i++) inIdx[axes[i]] = outIdx[i];
                int flatIn = ShapeUtils.RavelIndex(inIdx, inStrides);
                aGrad[flatIn] += outGrad[flatOut];
            }
        };
        return result;
    }

    public static Tensor Transpose(Tensor a, int dim0, int dim1)
    {
        var axes = Enumerable.Range(0, a.NDim).ToArray();
        (axes[dim0], axes[dim1]) = (axes[dim1], axes[dim0]);
        return Permute(a, axes);
    }

    // ---------------------------------------------------------------- Softmax (fused)

    /// <summary>Softmax over the last dimension, with a numerically-stable forward pass
    /// and the standard closed-form Jacobian-vector product for backward.</summary>
    public static Tensor Softmax(Tensor a)
    {
        int lastDim = a.Shape[^1];
        int outerSize = a.Size / lastDim;
        var outData = new float[a.Size];

        for (int o = 0; o < outerSize; o++)
        {
            int b = o * lastDim;
            float max = float.NegativeInfinity;
            for (int i = 0; i < lastDim; i++) max = MathF.Max(max, a.Data[b + i]);
            float sum = 0f;
            for (int i = 0; i < lastDim; i++)
            {
                float e = MathF.Exp(a.Data[b + i] - max);
                outData[b + i] = e;
                sum += e;
            }
            for (int i = 0; i < lastDim; i++) outData[b + i] /= sum;
        }

        var result = new Tensor(outData, (int[])a.Shape.Clone(), a.RequiresGrad, "softmax");
        result.Prev.Add(a);
        result.BackwardOp = () =>
        {
            if (!a.RequiresGrad) return;
            var aGrad = a.GetOrInitGrad();
            var outGrad = result.Grad!;
            for (int o = 0; o < outerSize; o++)
            {
                int b = o * lastDim;
                float dot = 0f;
                for (int i = 0; i < lastDim; i++) dot += outGrad[b + i] * outData[b + i];
                for (int i = 0; i < lastDim; i++)
                    aGrad[b + i] += outData[b + i] * (outGrad[b + i] - dot);
            }
        };
        return result;
    }

    // ---------------------------------------------------------------- Cross entropy (fused)

    /// <summary>
    /// Mean cross-entropy loss over rows of <paramref name="logits"/> (shape [N, V]) against
    /// integer class labels. Implemented as fused log-softmax + NLL for numerical stability;
    /// backward is the classic (softmax(logits) - one_hot(target)) / N.
    /// </summary>
    public static Tensor CrossEntropy(Tensor logits, int[] targets)
    {
        int N = logits.Shape[0], V = logits.Shape[1];
        if (targets.Length != N)
            throw new ArgumentException("targets length must match number of rows in logits.");

        var softmaxData = new float[N * V];
        float lossSum = 0f;

        for (int r = 0; r < N; r++)
        {
            int b = r * V;
            float max = float.NegativeInfinity;
            for (int c = 0; c < V; c++) max = MathF.Max(max, logits.Data[b + c]);
            float sum = 0f;
            for (int c = 0; c < V; c++)
            {
                float e = MathF.Exp(logits.Data[b + c] - max);
                softmaxData[b + c] = e;
                sum += e;
            }
            for (int c = 0; c < V; c++) softmaxData[b + c] /= sum;

            int t = targets[r];
            lossSum += -MathF.Log(MathF.Max(softmaxData[b + t], 1e-12f));
        }

        float loss = lossSum / N;
        var result = new Tensor(new[] { loss }, new[] { 1 }, logits.RequiresGrad, "cross_entropy");
        result.Prev.Add(logits);
        result.BackwardOp = () =>
        {
            if (!logits.RequiresGrad) return;
            var g = logits.GetOrInitGrad();
            float upstream = result.Grad![0];
            for (int r = 0; r < N; r++)
            {
                int b = r * V;
                int t = targets[r];
                for (int c = 0; c < V; c++)
                {
                    float indicator = c == t ? 1f : 0f;
                    g[b + c] += upstream * (softmaxData[b + c] - indicator) / N;
                }
            }
        };
        return result;
    }

    // ---------------------------------------------------------------- Embedding gather

    /// <summary>Row-gather from an embedding table, e.g. weight[vocab, dim] with a flat
    /// list of row indices reshaped to <paramref name="indicesShape"/>.</summary>
    public static Tensor EmbeddingLookup(Tensor weight, int[] indices, int[] indicesShape)
    {
        int dim = weight.Shape[1];
        int L = ShapeUtils.ElementCount(indicesShape);
        if (indices.Length != L)
            throw new ArgumentException("indices length must match product of indicesShape.");

        var outShape = indicesShape.Concat(new[] { dim }).ToArray();
        var outData = new float[L * dim];
        for (int i = 0; i < L; i++)
            Array.Copy(weight.Data, indices[i] * dim, outData, i * dim, dim);

        var result = new Tensor(outData, outShape, weight.RequiresGrad, "embedding");
        result.Prev.Add(weight);
        result.BackwardOp = () =>
        {
            if (!weight.RequiresGrad) return;
            var wGrad = weight.GetOrInitGrad();
            var outGrad = result.Grad!;
            for (int i = 0; i < L; i++)
            {
                int wBase = indices[i] * dim, oBase = i * dim;
                for (int d = 0; d < dim; d++) wGrad[wBase + d] += outGrad[oBase + d];
            }
        };
        return result;
    }

    // ---------------------------------------------------------------- Causal mask helper

    /// <summary>A constant (non-trainable) [T, T] additive mask: 0 on/below the diagonal,
    /// -1e9 above it. Broadcast-added to attention scores to enforce causality.</summary>
    public static Tensor CausalMask(int t)
    {
        var data = new float[t * t];
        for (int i = 0; i < t; i++)
            for (int j = 0; j < t; j++)
                data[i * t + j] = j <= i ? 0f : -1e9f;
        return new Tensor(data, new[] { t, t }, requiresGrad: false, op: "causal_mask");
    }
}
