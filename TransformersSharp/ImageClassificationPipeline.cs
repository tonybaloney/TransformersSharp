using CSnakes.Runtime.Python;

namespace TransformersSharp;

public class ImageClassificationPipeline : Pipeline
{
    public struct ClassificationResult
    {
        public string Label { get; set; }
        public float Score { get; set; }
    }

    internal ImageClassificationPipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }
    public static ImageClassificationPipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new ImageClassificationPipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "image-classification",
            model,
            null,
            torchDtype?.ToString(),
            device,
            trustRemoteCode));
    }

    /// <summary>
    /// Classifies an image using the image classification pipeline.
    /// </summary>
    /// <param name="path">URL to an image or the local path to an image file</param>
    /// <param name="functionToApply">Optional. Can be "sigmoid", "softmax" or "none"</param>
    /// <param name="topk">Number of top results to return</param>
    /// <param name="timeout">Optional timeout for the classification process</param>
    /// <returns>Enumerable of classification results</returns>
    public IEnumerable<ClassificationResult> Classify(string path, string? functionToApply = null, long? topk = null, double? timeout = null)
    {
        var results = TransformerEnvironment.TransformersWrapper.InvokeImageClassificationPipeline(PipelineObject, path, functionToApply, topk, timeout);

        return results.Select(r => new ClassificationResult { Label = r["label"].As<string>(), Score = r["score"].As<float>() });
    }
}

