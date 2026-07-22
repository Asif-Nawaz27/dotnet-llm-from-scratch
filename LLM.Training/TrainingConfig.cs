namespace LLM.Training;

public sealed class TrainingConfig
{
    public int BatchSize { get; set; } = 32;
    public int MaxSteps { get; set; } = 2000;
    public int EvalInterval { get; set; } = 100;
    public int EvalIters { get; set; } = 20;
    public float LearningRate { get; set; } = 3e-4f;
    public float Beta1 { get; set; } = 0.9f;
    public float Beta2 { get; set; } = 0.95f;
    public float WeightDecay { get; set; } = 0.01f;
    public float GradClipNorm { get; set; } = 1.0f;
    public string? CheckpointDir { get; set; }
}
