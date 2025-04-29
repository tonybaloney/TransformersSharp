using CSnakes.Runtime.Python;

namespace TransformersSharp;

public class TextGenerationPipeline : Pipeline
{
    internal TextGenerationPipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }

    public static TextGenerationPipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new TextGenerationPipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "text-generation",
            model,
            null,
            torchDtype?.ToString(),
            device,
            trustRemoteCode));
    }

    public IEnumerable<string> Generate(string input)
    {
        var results = RunPipeline(input);
        return results.Select(result => result["generated_text"].As<string>());
    }

    public IEnumerable<IReadOnlyDictionary<string, string>> Generate(IReadOnlyList<IReadOnlyDictionary<string, string>> messages, long? maxLength = null, long? maxNewTokens = null, long? minLength = null, long? minNewTokens = null, IReadOnlyList<string>? stopStrings = null, double? temperature = 1, long? topk = 50, double? topp = 1, double? minp = null)
    {
        var results = TransformerEnvironment.TransformersWrapper.InvokeTextGenerationPipelineWithTemplate(PipelineObject, messages, maxLength, maxNewTokens, minLength, minNewTokens, stopStrings, temperature, topk, topp, minp);
        return results;
    }
}
