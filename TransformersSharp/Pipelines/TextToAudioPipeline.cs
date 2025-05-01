using CommunityToolkit.HighPerformance;
using CSnakes.Runtime.Python;

namespace TransformersSharp.Pipelines;

public class TextToAudioPipeline : Pipeline
{
    public ref struct AudioResult
    {
        public ReadOnlySpan2D<float> Audio { get; set; }
        public int SamplingRate { get; set; }
    }
    internal TextToAudioPipeline(PyObject pipelineObject) : base(pipelineObject)
    {
    }

    public static TextToAudioPipeline FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new TextToAudioPipeline(TransformerEnvironment.TransformersWrapper.Pipeline(
            "text-to-audio",
            model,
            null,
            torchDtype?.ToString(),
            device,
            trustRemoteCode));
    }

    public AudioResult Generate(string text)
    {
        var (audio, sampleRate) = TransformerEnvironment.TransformersWrapper.InvokeTextToAudioPipeline(PipelineObject, text);
        return new AudioResult { 
            Audio = audio.AsFloatReadOnlySpan2D(),
            SamplingRate = (int)sampleRate
        };
    }
}
