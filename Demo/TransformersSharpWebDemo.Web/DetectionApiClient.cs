using static TransformersSharp.Pipelines.ObjectDetectionPipeline;

namespace TransformersSharpWebDemo.Web;

public class DetectionApiClient(HttpClient httpClient)
{
    public async Task<DetectionResult[]> GetObjectDetectionAsync(CancellationToken cancellationToken = default)
    {
        List<DetectionResult>? detectedObjects = [];
        DetectRequest detectRequest = new("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"); // Replace with actual URL
        var response = await httpClient.PostAsJsonAsync("/detect", detectRequest, cancellationToken);

        foreach (var detectionResult in await response.Content.ReadFromJsonAsync<DetectionResult[]>(cancellationToken))
        {
            detectedObjects.Add(detectionResult);
        }

        return detectedObjects?.ToArray() ?? [];
    }
}

public record DetectRequest(string Url)
{
}
