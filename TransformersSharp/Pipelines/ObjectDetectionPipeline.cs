using CSnakes.Runtime.Python;

namespace TransformersSharp.Pipelines;

public class ObjectDetectionPipeline: Pipeline
{
    public struct DetectionBox
    {
        public int XMin { get; set; }
        public int YMin { get; set; }
        public int XMax { get; set; }
        public int YMax { get; set; }
    }

    public struct DetectionResult
    {
        public string Label { get; set; }
        public float Score { get; set; }
        public DetectionBox Box { get; set; }
    }
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
        var results = TransformerEnvironment.TransformersWrapper.InvokeObjectDetectionPipeline(PipelineObject, path, threshold, timeout);
        return results.Select(r => new DetectionResult
        {
            Label = r["label"].As<string>(),
            Score = r["score"].As<float>(),
            Box = r["box"].As<IReadOnlyDictionary<string, int>>().Aggregate(new DetectionBox(), (acc, kvp) =>
            {
                switch (kvp.Key)
                {
                    case "xmin":
                        acc.XMin = kvp.Value;
                        break;
                    case "ymin":
                        acc.YMin = kvp.Value;
                        break;
                    case "xmax":
                        acc.XMax = kvp.Value;
                        break;
                    case "ymax":
                        acc.YMax = kvp.Value;
                        break;
                }
                return acc;
            })
        });
    }
}
