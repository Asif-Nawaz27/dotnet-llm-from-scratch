namespace LLM.Model;

/// <summary>Architecture hyperparameters for the GPT model. Kept small by default so the
/// whole thing trains on CPU in a reasonable time.</summary>
public sealed class GptConfig
{
    public int VocabSize { get; set; }
    public int BlockSize { get; set; } = 128;   // max context length (sequence length)
    public int NEmbd { get; set; } = 128;       // embedding / residual stream width
    public int NHead { get; set; } = 4;
    public int NLayer { get; set; } = 4;
    public float DropoutProb { get; set; } = 0.1f;

    public int HeadDim => NEmbd / NHead;
}
