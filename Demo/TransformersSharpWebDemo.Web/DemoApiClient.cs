using Microsoft.AspNetCore.Components.Forms;
using static TransformersSharp.Pipelines.ObjectDetectionPipeline;

namespace TransformersSharpWebDemo.Web;

public class DemoApiClient(HttpClient httpClient)
{
    public async Task<DetectResponse> GetObjectDetectionAsync(string imageUrl, CancellationToken cancellationToken = default)
    {
        List<DetectionResult>? detectedObjects = [];
        var url = imageUrl;
        DetectRequest detectRequest = new(url); // Replace with actual URL
        // Extend timeout because this can take a while
        httpClient.Timeout = TimeSpan.FromMinutes(5);
        var response = await httpClient.PostAsJsonAsync("/detect", detectRequest, cancellationToken);

        foreach (var detectionResult in await response.Content.ReadFromJsonAsync<DetectionResult[]>(cancellationToken))
        {
            detectedObjects.Add(detectionResult);
        }

        return new(url, detectedObjects?.ToArray() ?? []);
    }

    public async Task<string> GetTranscribeAsync(IBrowserFile selectedFile)
    {
        var content = new MultipartFormDataContent();
        var stream = selectedFile.OpenReadStream();
        var fileContent = new StreamContent(stream);
        fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("audio/flac");
        content.Add(fileContent, "file", selectedFile.Name);

        // Adjust the API URL as needed for your environment
        var response = await httpClient.PostAsync("/transcribe", content);

        if (response.IsSuccessStatusCode)
        {
            return await response.Content.ReadAsStringAsync();
        }
        else
        {
            throw new Exception($"Transcription failed: {response.ReasonPhrase}");
        }
    }
}

public record DetectRequest(string Url)
{
}

public record DetectResponse(string Url, DetectionResult[] DetectionResults)
{
}
