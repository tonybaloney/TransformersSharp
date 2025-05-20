using TransformersSharp.MEAI;
using TransformersSharp.Pipelines;
using static TransformersSharp.Pipelines.ObjectDetectionPipeline;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire client integrations.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddProblemDetails();

// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
app.UseExceptionHandler();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

var objectDetectionPipeline = ObjectDetectionPipeline.FromModel("facebook/detr-resnet-50");
var speechToTextClient = SpeechToTextClient.FromModel("openai/whisper-small");

app.MapPost("/detect", (DetectRequest r) =>
{
    var result = objectDetectionPipeline.Detect(r.Url);
    return result;

}).Accepts<DetectRequest>("application/json")
    .Produces<DetectionResult>(StatusCodes.Status200OK)
    .WithName("Detect");

app.MapPost("/transcribe", async (HttpRequest request) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest("File upload required");

    var form = await request.ReadFormAsync();

    string output = string.Empty;
    foreach (var file in form.Files)
    {
        using var stream = file.OpenReadStream();
        output += await speechToTextClient.GetTextAsync(stream);
    }

    return Results.Ok(output);

}).Accepts<IFormFile>("multipart/form-data")
    .Produces<string>(StatusCodes.Status200OK)
    .WithName("Transcribe");

app.MapDefaultEndpoints();

app.Run();

record DetectRequest(string Url)
{
}
