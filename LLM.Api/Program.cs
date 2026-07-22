using LLM.Api;

const string CorsPolicy = "WebApp";

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton<LlmInferenceService>();
builder.Services.AddSingleton<TrainingService>();
builder.Services.AddControllers();
builder.Services.AddCors(options =>
{
    // The React dev server (Vite defaults to 5173) needs cross-origin access to call this API.
    options.AddPolicy(CorsPolicy, policy => policy
        .WithOrigins("http://localhost:5173", "https://localhost:5173")
        .AllowAnyHeader()
        .AllowAnyMethod());
});

var app = builder.Build();
app.UseCors(CorsPolicy);
app.MapControllers();

app.Run();
