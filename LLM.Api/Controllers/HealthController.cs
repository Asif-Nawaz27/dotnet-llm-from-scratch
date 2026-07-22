using Microsoft.AspNetCore.Mvc;

namespace LLM.Api.Controllers;

[ApiController]
public sealed class HealthController : ControllerBase
{
    private readonly LlmInferenceService _inferenceService;

    public HealthController(LlmInferenceService inferenceService)
    {
        _inferenceService = inferenceService;
    }

    [HttpGet("/")]
    public IActionResult Index()
        => Ok("LLM.Api is running. POST /generate to sample text, GET /train/stream to train with live progress.");

    [HttpGet("/health")]
    public IActionResult Health() => Ok(new { modelLoaded = _inferenceService.IsModelLoaded });
}
