using CSnakes.Runtime.Python;

namespace TransformersSharp;

public class TextGenerationPipeline : Pipeline
{
    internal TextGenerationPipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }

    public static TextGenerationPipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null)
    {
        return new TextGenerationPipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "text-generation",
            model,
            null,
            torchDtype?.ToString(),
            device));
    }

    public IReadOnlyList<string> Generate(string input)
    {
        var results = RunPipeline(input);
        return results.Select(result => result["generated_text"].As<string>()).ToList();
    }
}
