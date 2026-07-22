using Microsoft.AspNetCore.Mvc;

namespace LLM.Api.Controllers;

[ApiController]
public sealed class DataController : ControllerBase
{
    // Backs the web UI's "browse" button: a browser can't hand the server a local filesystem
    // path (only file content), so this saves the uploaded bytes under data/uploads and hands
    // back a server-side path that /train/stream's dataPath can then point at.
    [HttpPost("/data/upload")]
    public async Task<IActionResult> Upload(IFormFile file)
    {
        if (file.Length == 0)
            return BadRequest("No file provided.");

        var uploadsDir = Path.Combine("..", "data", "uploads");
        Directory.CreateDirectory(uploadsDir);

        var safeName = Path.GetFileName(file.FileName); // strip any path components from the original name
        var destPath = Path.Combine(uploadsDir, safeName);

        await using (var stream = System.IO.File.Create(destPath))
        {
            await file.CopyToAsync(stream);
        }

        return Ok(new { path = destPath.Replace('\\', '/') });
    }
}
