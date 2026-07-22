namespace LLM.Core;

/// <summary>
/// A dense, row-major, CPU-only N-dimensional array with reverse-mode automatic
/// differentiation. This is the entire "engine" the model is built on: every
/// higher-level op (Linear, LayerNorm, Attention, ...) is composed from the
/// primitive ops defined on this type in TensorOps.cs.
///
/// Design: each Tensor produced by an op keeps a reference to its parent tensors
/// and a closure ("_backward") that knows how to push gradients from this tensor's
/// Grad into its parents' Grad. Backward() builds a topological order of the graph
/// and invokes those closures in reverse order (standard "tape"-free autograd, the
/// same approach as micrograd/tinygrad).
/// </summary>
public sealed class Tensor
{
    public float[] Data;
    public float[]? Grad;
    public int[] Shape;
    public bool RequiresGrad;

    internal readonly List<Tensor> Prev;
    internal Action? BackwardOp;
    private readonly string _op;

    public int NDim => Shape.Length;
    public int Size => ShapeUtils.ElementCount(Shape);

    public Tensor(int[] shape, bool requiresGrad = false, string op = "leaf")
    {
        Shape = shape;
        Data = new float[ShapeUtils.ElementCount(shape)];
        RequiresGrad = requiresGrad;
        Prev = new List<Tensor>();
        _op = op;
    }

    public Tensor(float[] data, int[] shape, bool requiresGrad = false, string op = "leaf")
    {
        if (data.Length != ShapeUtils.ElementCount(shape))
            throw new ArgumentException(
                $"Data length {data.Length} does not match shape [{string.Join(",", shape)}]");
        Data = data;
        Shape = shape;
        RequiresGrad = requiresGrad;
        Prev = new List<Tensor>();
        _op = op;
    }

    public static Tensor Zeros(params int[] shape) => new(shape);

    public static Tensor Full(float value, int[] shape)
    {
        var t = new Tensor(shape);
        Array.Fill(t.Data, value);
        return t;
    }

    public static Tensor FromArray(float[] data, params int[] shape) => new((float[])data.Clone(), shape);

    /// <summary>Gaussian init scaled by <paramref name="std"/>, used for weight matrices.</summary>
    public static Tensor Randn(int[] shape, float std, Random rng, bool requiresGrad = true)
    {
        var t = new Tensor(shape, requiresGrad);
        for (int i = 0; i < t.Data.Length; i++)
        {
            // Box-Muller
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            t.Data[i] = (float)(z * std);
        }
        return t;
    }

    public void ZeroGrad()
    {
        if (Grad is null) Grad = new float[Data.Length];
        else Array.Clear(Grad, 0, Grad.Length);
    }

    private void EnsureGrad()
    {
        Grad ??= new float[Data.Length];
    }

    /// <summary>
    /// Runs backprop from this tensor (must be scalar) through the whole graph that
    /// produced it, accumulating into every ancestor's Grad.
    /// </summary>
    public void Backward()
    {
        if (Size != 1)
            throw new InvalidOperationException("Backward() can only be called on a scalar tensor.");

        var topo = new List<Tensor>();
        var visited = new HashSet<Tensor>();

        void Visit(Tensor t)
        {
            if (!visited.Add(t)) return;
            foreach (var p in t.Prev) Visit(p);
            topo.Add(t);
        }
        Visit(this);

        EnsureGrad();
        Grad![0] = 1.0f;

        for (int i = topo.Count - 1; i >= 0; i--)
        {
            topo[i].BackwardOp?.Invoke();
        }
    }

    internal float[] GetOrInitGrad()
    {
        EnsureGrad();
        return Grad!;
    }

    public override string ToString() => $"Tensor(op={_op}, shape=[{string.Join(",", Shape)}], requiresGrad={RequiresGrad})";

    // Thin operator sugar over TensorOps, kept here so call sites read naturally (a + b, a * w).
    public static Tensor operator +(Tensor a, Tensor b) => TensorOps.Add(a, b);
    public static Tensor operator -(Tensor a, Tensor b) => TensorOps.Sub(a, b);
    public static Tensor operator *(Tensor a, Tensor b) => TensorOps.Mul(a, b);
    public static Tensor operator +(Tensor a, float s) => TensorOps.AddScalar(a, s);
    public static Tensor operator *(Tensor a, float s) => TensorOps.MulScalar(a, s);
    public static Tensor operator -(Tensor a) => TensorOps.Neg(a);

    public Tensor MatMul(Tensor other) => TensorOps.MatMul(this, other);
    public Tensor Reshape(params int[] shape) => TensorOps.Reshape(this, shape);
    public Tensor Transpose(int dim0, int dim1) => TensorOps.Transpose(this, dim0, dim1);
    public Tensor Permute(params int[] axes) => TensorOps.Permute(this, axes);
    public Tensor Sum(int axis, bool keepdim = false) => TensorOps.Sum(this, axis, keepdim);
    public Tensor Mean(int axis, bool keepdim = false) => TensorOps.Mean(this, axis, keepdim);
    public Tensor Softmax() => TensorOps.Softmax(this);
    public Tensor Tanh() => TensorOps.Tanh(this);
    public Tensor Relu() => TensorOps.Relu(this);
    public Tensor Exp() => TensorOps.Exp(this);
    public Tensor Log() => TensorOps.Log(this);
    public Tensor Pow(float p) => TensorOps.PowScalar(this, p);
}
