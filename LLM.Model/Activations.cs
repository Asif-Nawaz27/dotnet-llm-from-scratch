using LLM.Core;

namespace LLM.Model;

public static class Activations
{
    private const float SqrtTwoOverPi = 0.7978845608f;

    /// <summary>Tanh approximation of GELU, built from primitive ops (Tanh/Pow/Mul/Add) so
    /// its gradient is derived automatically by the autograd engine.</summary>
    public static Tensor Gelu(Tensor x)
    {
        var xCubed = x.Pow(3f);
        var inner = (x + xCubed * 0.044715f) * SqrtTwoOverPi;
        var tanhPart = inner.Tanh() + 1f;
        return x * tanhPart * 0.5f;
    }
}
