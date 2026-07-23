using Microsoft.AspNetCore.Mvc;

namespace LLM.Api.Controllers;

[ApiController]
public sealed class GenerateController : ControllerBase
{
    private readonly LlmInferenceService _inferenceService;

    public GenerateController(LlmInferenceService inferenceService)
    {
        _inferenceService = inferenceService;
    }

    [HttpPost("/generate")]
    public IActionResult Generate([FromBody] GenerateRequest request)
    {
        try
        {
            var text = _inferenceService.Generate(
                prompt: request.Prompt ?? "",
                maxTokens: request.MaxTokens ?? 200,
                temperature: request.Temperature ?? 0.8f,
                topK: request.TopK ?? 40);

            return Ok(new GenerateResponse(text));
        }
        catch (InvalidOperationException ex)
        {
            // Model not trained/loaded yet - a client error worth reporting, not a 500.
            return Problem(ex.Message, statusCode: StatusCodes.Status503ServiceUnavailable);
        }
        catch (ArgumentException ex)
        {
            // Prompt contains a character the model never saw during training (CharTokenizer.Encode) -
            // an expected user-input case, not a server fault.
            return Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
        }
    }
}

/// <summary>prompt: seed text; maxTokens: how many characters to generate; temperature:
/// higher = more random; topK: only sample from the K most likely next characters.</summary>
public sealed record GenerateRequest(string? Prompt, int? MaxTokens, float? Temperature, int? TopK);

public sealed record GenerateResponse(string Text);
