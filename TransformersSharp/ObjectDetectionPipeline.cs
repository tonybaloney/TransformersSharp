using CSnakes.Runtime.Python;

namespace TransformersSharp;

public class ObjectDetectionPipeline: Pipeline
{
    public readonly record struct DetectionBox(int XMin, int YMin, int XMax, int YMax);
    public readonly record struct DetectionResult(string Label, double Score, DetectionBox Box);

    internal ObjectDetectionPipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }
    public static ObjectDetectionPipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new ObjectDetectionPipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "object-detection",
            model,
            null,
            torchDtype?.ToString(),
            device,
            trustRemoteCode));
    }
    public IEnumerable<DetectionResult> Detect(string path, double threshold = 0.5, double? timeout = null)
    {
        IEnumerable<(string Label, double Score, (long XMin, long YMin, long XMax, long YMax) Box)> results =
            TransformerEnvironment.TransformersWrapper.InvokeObjectDetectionPipeline(PipelineObject, path, threshold, timeout);

        return
            from e in results
            select new DetectionResult
            {
                Label = e.Label,
                Score = e.Score,
                Box = checked(new()
                {
                    XMin = (int)e.Box.XMin,
                    YMin = (int)e.Box.YMin,
                    XMax = (int)e.Box.XMax,
                    YMax = (int)e.Box.YMax,
                })
            };
    }
}
