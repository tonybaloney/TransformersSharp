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
        var results = RunPipeline(input);
        return results.Select(result => (result["label"].As<string>(), result["score"].As<double>())).ToList();
    }

    public IReadOnlyList<IReadOnlyList<(string Label, double Score)>> ClassifyBatch(IReadOnlyList<string> inputs)
    {
        var results = RunPipeline(inputs);
        return results.Select(resultList =>
            resultList.Select(result => (result["label"].As<string>(), result["score"].As<double>())).ToList()).ToList();
    }
}
