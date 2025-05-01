using Microsoft.Extensions.AI;
using TransformersSharp.Pipelines;

namespace TransformersSharp.MEAI;

#pragma warning disable MEAI001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
public class SpeechToTextClient : ISpeechToTextClient
{
    public AutomaticSpeechRecognitionPipeline AutomaticSpeechRecognitionPipeline { get; private set; }

    public static SpeechToTextClient FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new SpeechToTextClient
        {
            AutomaticSpeechRecognitionPipeline = AutomaticSpeechRecognitionPipeline.FromModel(model, torchDtype, device, trustRemoteCode: trustRemoteCode)
        };
    }

    public void Dispose()
    {
        // Nothing to do right now
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceType is null ? throw new ArgumentNullException(nameof(serviceType)) :
        serviceKey is not null ? null :
        serviceType.IsInstanceOfType(this) ? this :
        serviceType == typeof(AutomaticSpeechRecognitionPipeline) ? this.AutomaticSpeechRecognitionPipeline :
        null;



    public async Task<SpeechToTextResponse> GetTextAsync(Stream audioSpeechStream, SpeechToTextOptions? options = null, CancellationToken cancellationToken = default)
    {
        byte[] audioBytes = new byte[audioSpeechStream.Length];
        await audioSpeechStream.ReadExactlyAsync(audioBytes, 0, (int)audioSpeechStream.Length, cancellationToken);
        var result = AutomaticSpeechRecognitionPipeline.Transcribe(audioBytes);
        return new SpeechToTextResponse(result);
    }

    public async IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(Stream audioSpeechStream, SpeechToTextOptions? options = null, CancellationToken cancellationToken = default)
    {
        // None of the models are streaming yet.
        var response = await GetTextAsync(audioSpeechStream, options, cancellationToken).ConfigureAwait(false);
        foreach (var update in response.ToSpeechToTextResponseUpdates())
        {
            yield return update;
        }
    }
}
#pragma warning restore MEAI001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

