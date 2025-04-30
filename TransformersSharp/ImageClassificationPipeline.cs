using CSnakes.Runtime.Python;
using TransformersSharp.Models;

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

    /// <summary>
    /// Classifies an image using the image classification pipeline.
    /// </summary>
    /// <param name="image">Image data</param>
    /// <param name="functionToApply">Optional. Can be "sigmoid", "softmax" or "none"</param>
    /// <param name="topk">Number of top results to return</param>
    /// <param name="timeout">Optional timeout for the classification process</param>
    /// <returns>Enumerable of classification results</returns>
    public IEnumerable<ClassificationResult> Classify(ImageData image, string? functionToApply = null, int topk = 5, double? timeout = null)
    {
        byte[] imageBytes = image.ImageBytes;
        string pixelMode = image.PixelMode switch
        {
            ImagePixelMode.RGB => "RGB",
            ImagePixelMode.Greyscale => "L",
            _ => throw new ArgumentOutOfRangeException(nameof(image.PixelMode), "Invalid pixel mode.")
        };
        int width = image.Width;
        int height = image.Height;
        var results = TransformerEnvironment.TransformersWrapper.InvokeImageClassificationFromBytes(PipelineObject, imageBytes, width, height, pixelMode, functionToApply, topk, timeout);
        return results.Select(r => new ClassificationResult { Label = r["label"].As<string>(), Score = r["score"].As<float>() });
    }
}

