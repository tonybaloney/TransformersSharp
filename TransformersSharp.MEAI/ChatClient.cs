using Microsoft.Extensions.AI;

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

    private static ChatRole GetChatRole(string role)
    {
        return role switch
        {
            "user" => ChatRole.User,
            "assistant" => ChatRole.Assistant,
            "system" => ChatRole.System,
            "tool" => ChatRole.Tool,
            _ => throw new ArgumentException($"Unknown chat role: {role}")
        };
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
            var responseMessages = result.Select(message => new ChatMessage(GetChatRole(message["role"]), message["content"])).ToList();
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
