using Microsoft.AspNetCore.Mvc;

namespace LLM.Api.Controllers;

[ApiController]
public sealed class TrainController : ControllerBase
{
    private readonly TrainingService _trainingService;

    public TrainController(TrainingService trainingService)
    {
        _trainingService = trainingService;
    }

    // GET (not POST) because the browser's EventSource API - the standard way to consume a
    // Server-Sent Events stream - can only issue GET requests, so every parameter is a query param.
    [HttpGet("/train/stream")]
    public async Task StreamAsync(
        [FromQuery] string dataPath,
        [FromQuery] int steps = 3000,
        [FromQuery] int batchSize = 32,
        [FromQuery] int blockSize = 128,
        [FromQuery] int nEmbd = 128,
        [FromQuery] int nHead = 4,
        [FromQuery] int nLayer = 4,
        [FromQuery] float dropout = 0.1f,
        [FromQuery] float lr = 3e-4f,
        [FromQuery] int evalInterval = 100,
        [FromQuery] int evalIters = 20,
        [FromQuery] int seed = 1337)
    {
        var request = new TrainRequest(dataPath, steps, batchSize, blockSize, nEmbd, nHead, nLayer, dropout, lr, evalInterval, evalIters, seed);

        if (!_trainingService.TryStart(HttpContext.RequestAborted, out var token))
        {
            Response.StatusCode = StatusCodes.Status409Conflict;
            await Response.WriteAsync("A training run is already in progress.");
            return;
        }

        Response.Headers.ContentType = "text/event-stream";
        Response.Headers.CacheControl = "no-cache";

        async Task SendAsync(string line)
        {
            // SSE wire format: "data: <text>\n\n". Newlines inside a line would break the
            // framing, so collapse them - log lines here are always single-line anyway.
            try
            {
                await Response.WriteAsync($"data: {line.Replace('\n', ' ')}\n\n", HttpContext.RequestAborted);
                await Response.Body.FlushAsync(HttpContext.RequestAborted);
            }
            catch
            {
                // The client already disconnected (e.g. page closed/reloaded) - nothing more we
                // can do, and nothing left to report. Swallowing here matters: if this throws
                // from inside the catch block below, it would otherwise escape as an unhandled
                // exception instead of ending quietly, which is exactly what we don't want for
                // errors that are supposed to reach the user only via the train.log stream.
            }
        }

        // Separate from the log lines above: fires every training step so the UI's progress
        // bar moves continuously instead of only jumping once per eval interval.
        void OnProgress(int step, int maxSteps) => SendAsync($"[[PROGRESS]] {step}/{maxSteps}").GetAwaiter().GetResult();

        try
        {
            await _trainingService.RunAsync(request, line => SendAsync(line).GetAwaiter().GetResult(), token, OnProgress);
            await SendAsync("[[DONE]]");
        }
        catch (OperationCanceledException)
        {
            // Stopped via /train/cancel, or the client navigated away - nothing left to report.
        }
        catch (Exception ex)
        {
            // Any failure from here on (bad data file, dataset too small for the chosen block
            // size, etc.) is reported to the client as a log line, never as a raw server error.
            await SendAsync($"[[ERROR]] {ex.Message}");
        }
    }

    // Explicit stop signal, since detecting that the SSE connection itself closed can be slow
    // or unreliable - without this, clicking Stop and Start again quickly could find the
    // training slot still marked busy.
    [HttpPost("/train/cancel")]
    public IActionResult Cancel()
    {
        _trainingService.RequestStop();
        return Ok();
    }
}
