using CSnakes.Runtime.Python;

namespace TransformersSharp;

public class TextClassificationPipeline : Pipeline
{
    internal TextClassificationPipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }

    public static TextClassificationPipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null)
    {
        return new TextClassificationPipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "text-classification",
            model,
            null,
            torchDtype?.ToString(),
            device));
    }

    public IReadOnlyList<(string Label, double Score)> Classify(string input)
    {
        return RunPipeline(input).Select(result => (result["label"].As<string>(), result["score"].As<double>())).ToList();
    }

    public IReadOnlyList<(string Label, double Score)> ClassifyBatch(IReadOnlyList<string> inputs)
    {
        return RunPipeline(inputs).Select(result => (result["label"].As<string>(), result["score"].As<double>())).ToList();
    }
}
