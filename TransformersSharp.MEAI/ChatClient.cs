using Microsoft.Extensions.AI;

namespace TransformersSharp.MEAI;

public class TextGenerationPipelineChatClient : IChatClient
{
    public TextGenerationPipeline TextGenerationPipeline { get; private set; }

    public static TextGenerationPipelineChatClient FromModel(string model, TorchDtype? torchDtype = null, string? device = null)
    {
        return new TextGenerationPipelineChatClient
        {
            TextGenerationPipeline = TextGenerationPipeline.FromModel(model, torchDtype, device)
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
                }).ToList());

            var responseMessages = result.Select(text => new ChatMessage(ChatRole.Assistant, text)).ToList();
            return new ChatResponse(responseMessages);
        }, cancellationToken);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        throw new NotImplementedException();
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }
}
