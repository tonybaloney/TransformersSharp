using TransformersSharp.MEAI;
using Microsoft.Extensions.AI;

namespace TransformersSharp.Tests;

public class TransformerSharpMEAITests
{
    [Fact]
    public async Task TestChatClient()
    {
        var chatClient = TextGenerationPipelineChatClient.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16, trustRemoteCode: true);
        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, "You are a helpful little robot."),
            new(ChatRole.User, "how many helicopters can a human eat in one sitting?!")
        };
        var response = await chatClient.GetResponseAsync(messages, new() { Temperature = 0.7f });
        
        Assert.NotNull(response);
        Assert.Contains("helicopter", response.Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TestChatClientStreaming()
    {
        var chatClient = TextGenerationPipelineChatClient.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16, trustRemoteCode: true);
        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, "You are a helpful little robot."),
            new(ChatRole.User, "how many helicopters can a human eat in one sitting?!")
        };
        var response = chatClient.GetStreamingResponseAsync(messages, new() { Temperature = 0.7f });
        await foreach (var update in response)
        {
            Assert.NotNull(update);
            Assert.Contains("helicopter", update.Text, StringComparison.OrdinalIgnoreCase);
        }
    }
}
