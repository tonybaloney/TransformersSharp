using Microsoft.Extensions.AI;
using TransformersSharp.Pipelines;

namespace TransformersSharp.MEAI;

public class TextGenerationPipelineChatClient : IChatClient
{
    public TextGenerationPipeline TextGenerationPipeline { get; private set; }

    public static TextGenerationPipelineChatClient FromModel(string model, TorchDtype? torchDtype = null, string? device = null, bool trustRemoteCode = false)
    {
        return new TextGenerationPipelineChatClient
        {
            TextGenerationPipeline = TextGenerationPipeline.FromModel(model, torchDtype, device, trustRemoteCode: trustRemoteCode)
        };
    }

    public void Dispose()
    {
        // Nothing to do yet.
    }

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        return Task.Run(() =>
        {
            var result = TextGenerationPipeline.Generate(messages.Select(
                message => new Dictionary<string, string>
                {
                    { "role", message.Role.Value },
                    { "content", message.Text }
                }).ToList(),
                maxNewTokens: options?.MaxOutputTokens,
                topk: options?.TopK,
                topp: options?.TopP,
                temperature: options?.Temperature,
                stopStrings: options?.StopSequences?.AsReadOnly()
                );
            var responseMessages = result.Select(message => new ChatMessage(new ChatRole(message["role"]), message["content"])).ToList();
            return new ChatResponse(responseMessages);
        }, cancellationToken);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceType is null ? throw new ArgumentNullException(nameof(serviceType)) :
        serviceKey is not null ? null :
        serviceType.IsInstanceOfType(this) ? this :
        serviceType == typeof(TextGenerationPipeline) ? this.TextGenerationPipeline :
        null;

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        foreach (var update in response.ToChatResponseUpdates())
        {
            yield return update;
        }
    }
}
