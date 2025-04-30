# TransformersSharp

![Logo](docs/media/logo.png)

A little wrapper for Hugging Face Transformers in C#. This is not a comprehensive 1:1 mapping of the whole HuggingFace [transformers](https://pypi.org/transformers) package, because the API is enormous.

If you need a specific feature, toggle or pipeline API clone this repo and make adjustments.

This project was created using [CSnakes](https://github.com/tonybaloney/CSnakes) and will fetch Python, PyTorch, and Hugging Face Transformers automatically, so you don't need to install them manually.

[Full Documentation](https://tonybaloney.github.io/TransformersSharp/)

## Features

- Tokenizer API based on [`PreTrainedTokenizerBase`](https://huggingface.co/docs/transformers/v4.51.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
- Tokenizer shim to the [Microsoft.ML.Tokenizers](https://learn.microsoft.com/dotnet/api/microsoft.ml.tokenizers.tokenizer?view=ml-dotnet-preview) base class
- Generic Pipeline Factory
- TextGenerationPipeline Factory
- TextClassificationPipeline Factory

## Usage

### Text Classification


```csharp
using TransformersSharp;

var pipeline = TextClassificationPipeline.FromModel("distilbert-base-uncased-finetuned-sst-2-english");
var inputs = new List<string> { "I love programming!", "I hate bugs!" };
var results = pipeline.ClassifyBatch(inputs);

foreach (var result in results)
{
	Console.WriteLine($"Label: {result.Label}, Score: {result.Score}");
}
```

### Text Generation

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
IReadOnlyList<string>? result = pipeline.Generate("How many helicopters can a human eat in one sitting?");
Console.WriteLine(result[0]);
```

### Text Generation as Microsoft.Extensions.AI ChatClient

Using the `TransformersSharp.MEAI` package, you can use the `TextGenerationPipelineChatClient` to generate text in a chat-like format for the [`IChatClient`](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.ai.ichatclient?view=net-9.0-pp) API:

```csharp
using Microsoft.Extensions.AI;
using TransformersSharp.MEAI;

var chatClient = TextGenerationPipelineChatClient.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16);
var messages = new List<ChatMessage>
{
    new(ChatRole.System, "You are a helpful little robot."),
    new(ChatRole.User, "how many helicopters can a human eat in one sitting?!")
};
var response = await chatClient.GetResponseAsync(messages, new() { Temperature = 0.7f });
```

### Tokenizers from Pipelines

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
ReadOnlySpan<long> inputIds = pipeline.Tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
```

### Tokenizers from models

```csharp
using TransformersSharp;
using TransformersSharp.Tokenizers;

var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
ReadOnlySpan<long> inputIds = tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
Console.WriteLine($"InputIds: {string.Join(", ", InputIds)}");
```

### Tokenizers as Microsoft.ML.Tokenizers

`PreTrainedTokenizerBase` implements [`Microsoft.ML.Tokenizers.Tokenizer`](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers.tokenizer?view=ml-dotnet-preview), so you can use it with any library which uses the `Tokenizer` type, like SemanticKernel.

```csharp
var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
var input = "How many helicopters can a human eat in one sitting?";
var tokens = tokenizer.EncodeToTokens(input, out string? normalizedText);
```
