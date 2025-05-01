using CSnakes.Runtime.Python;

namespace TransformersSharp.Pipelines;

public class AutomaticSpeechRecognitionPipeline : Pipeline
{
    internal AutomaticSpeechRecognitionPipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }
    public static AutomaticSpeechRecognitionPipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new AutomaticSpeechRecognitionPipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "automatic-speech-recognition",
            model,
            null,
            torchDtype?.ToString(),
            device,
            trustRemoteCode));
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="audioPath">Local file path or URL</param>
    /// <returns></returns>
    public string Transcribe(string audioPath)
    {
        return TransformerEnvironment.TransformersWrapper.InvokeAutomaticSpeechRecognitionPipeline(PipelineObject, audioPath);
    }
}
