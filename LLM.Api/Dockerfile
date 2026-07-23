# Build context is the repo root (see the run instructions) since ProjectReference paths
# climb out of LLM.Api into sibling project folders.

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Restore first, with only the csproj files, so this layer is cached across code-only changes.
COPY LLM.Core/LLM.Core.csproj LLM.Core/
COPY LLM.Model/LLM.Model.csproj LLM.Model/
COPY LLM.Tokenizer/LLM.Tokenizer.csproj LLM.Tokenizer/
COPY LLM.Training/LLM.Training.csproj LLM.Training/
COPY LLM.Api/LLM.Api.csproj LLM.Api/
RUN dotnet restore LLM.Api/LLM.Api.csproj

COPY LLM.Core/ LLM.Core/
COPY LLM.Model/ LLM.Model/
COPY LLM.Tokenizer/ LLM.Tokenizer/
COPY LLM.Training/ LLM.Training/
COPY LLM.Api/ LLM.Api/

RUN dotnet publish LLM.Api/LLM.Api.csproj -c Release --no-restore -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
# WORKDIR is one level "inside" the repo layout on purpose: LlmInferenceService/TrainingService
# resolve CheckpointDir ("../checkpoints/...") and DataController resolves uploads
# ("../data/uploads") relative to the process's working directory, exactly as they do when run
# locally with `dotnet run` from the LLM.Api folder. Keeping that same one-level-up relationship
# here means no path handling had to change to run in a container.
WORKDIR /app/LLM.Api
COPY --from=build /app/publish .

EXPOSE 7060
ENTRYPOINT ["dotnet", "LLM.Api.dll"]
