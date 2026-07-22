using LLM.Core;

namespace LLM.Model;

/// <summary>The transformer block's MLP: Linear -> GELU -> Linear, with the standard
/// 4x hidden-dimension expansion.</summary>
public sealed class FeedForward
{
    private readonly Linear _fcIn;
    private readonly Linear _fcOut;
    private readonly Dropout _dropout;

    public FeedForward(GptConfig cfg, Random rng)
    {
        _fcIn = new Linear(cfg.NEmbd, 4 * cfg.NEmbd, rng);
        _fcOut = new Linear(4 * cfg.NEmbd, cfg.NEmbd, rng);
        _dropout = new Dropout(cfg.DropoutProb, rng);
    }

    public Tensor Forward(Tensor x, bool training)
    {
        var h = Activations.Gelu(_fcIn.Forward(x));
        var y = _fcOut.Forward(h);
        return _dropout.Forward(y, training);
    }

    public IEnumerable<Tensor> Parameters() => _fcIn.Parameters().Concat(_fcOut.Parameters());
}
