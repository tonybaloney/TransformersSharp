using static TransformersSharp.Pipelines.ObjectDetectionPipeline;

namespace TransformersSharpWebDemo.Web;

public class DetectionApiClient(HttpClient httpClient)
{
    public async Task<DetectResponse> GetObjectDetectionAsync(string imageUrl, CancellationToken cancellationToken = default)
    {
        List<DetectionResult>? detectedObjects = [];
        var url = imageUrl;
        DetectRequest detectRequest = new(url); // Replace with actual URL
        var response = await httpClient.PostAsJsonAsync("/detect", detectRequest, cancellationToken);

        foreach (var detectionResult in await response.Content.ReadFromJsonAsync<DetectionResult[]>(cancellationToken))
        {
            detectedObjects.Add(detectionResult);
        }

        return new(url, detectedObjects?.ToArray() ?? []);
    }
}

public record DetectRequest(string Url)
{
}

public record DetectResponse(string Url, DetectionResult[] DetectionResults)
{
}
