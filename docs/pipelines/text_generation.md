# Text Generation Pipeline

The `TransformersSharp.TextGenerationPipeline` class provides a high-level interface for generating text using pre-trained models from the Hugging Face Transformers library. It simplifies the process of text generation by handling tokenization, model inference, and decoding.

## What is a Text Generation Pipeline?

A text generation pipeline is designed to generate coherent and contextually relevant text based on a given input prompt. It is commonly used for tasks like:

- Autocompletion
- Story generation
- Chatbot responses
- Creative writing

### Key Features of Text Generation Pipelines:

- **Pre-trained Models**: Leverages state-of-the-art models like GPT, OPT, and others.
- **Customizable Parameters**: Allows fine-tuning of generation parameters like maximum length, temperature, and top-k sampling.
- **Batch Processing**: Supports single and batch inputs for efficient processing.

## Using the TextGenerationPipeline Class

The `TextGenerationPipeline` class in `TransformersSharp` provides methods to generate text based on input prompts. Below are examples of how to use it.

### Generating Text from a Single Input

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
var results = pipeline.Generate("Once upon a time");
foreach (var result in results)
{
    Console.WriteLine(result);
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-generation", model="facebook/opt-125m")
results = pipeline("Once upon a time")
for result in results:
    print(result["generated_text"])
```

### Generating Text with Custom Parameters

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16);
var messages = new List<IReadOnlyDictionary<string, string>>
{
    new Dictionary<string, string> { { "role", "user" }, { "content", "Tell me a story about a brave knight." } }
};
var results = pipeline.Generate(messages, maxLength: 100, temperature: 0.7);
foreach (var result in results)
{
    foreach (var kvp in result)
    {
        Console.WriteLine($"{kvp.Key}: {kvp.Value}");
    }
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline
import torch

pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)
results = pipeline("Tell me a story about a brave knight.", max_length=100, temperature=0.7)
for result in results:
    print(result["generated_text"])
```

## Accessing the Tokenizer

The `TextGenerationPipeline` class provides access to the associated tokenizer through the `Tokenizer` property. This allows users to preprocess inputs or decode outputs manually if needed.

### Example: Accessing the Tokenizer

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
var tokenizer = pipeline.Tokenizer;
var inputIds = tokenizer.Tokenize("Once upon a time");
Console.WriteLine(string.Join(", ", inputIds.ToArray()));
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-generation", model="facebook/opt-125m")
tokenizer = pipeline.tokenizer
input_ids = tokenizer("Once upon a time", return_tensors="pt")["input_ids"]
print(input_ids.tolist())
```

## Customizing Text Generation

The `Generate` method allows users to customize text generation by specifying parameters like:

- **`maxLength`**: The maximum length of the generated text.
- **`temperature`**: Controls the randomness of predictions by scaling the logits before applying softmax.
- **`topk`**: Limits the sampling pool to the top-k tokens.
- **`topp`**: Implements nucleus sampling by limiting the sampling pool to tokens with a cumulative probability of `p`.

### Example: Customizing Parameters

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
var results = pipeline.Generate("Once upon a time", maxLength: 50, temperature: 0.8, topk: 40);
foreach (var result in results)
{
    Console.WriteLine(result);
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-generation", model="facebook/opt-125m")
results = pipeline("Once upon a time", max_length=50, temperature=0.8, top_k=40)
for result in results:
    print(result["generated_text"])
```

## Microsoft Extensions AI (MEAI) compatible ChatClient

Using the `TransformersSharp.MEAI` package, you can use the `TextGenerationPipelineChatClient` to generate text in a chat-like format for the [`IChatClient`](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.ichatclient?view=net-9.0-pp) API:

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