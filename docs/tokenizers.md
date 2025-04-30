# Tokenizers

The `TransformersSharp.Tokenizers.PreTrainedTokenizerBase` class provides a mapping to the [`PreTrainedTokenizerBase`](https://huggingface.co/docs/transformers/v4.51.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)





## Tokenizers from Pipelines

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
ReadOnlySpan<long> inputIds = pipeline.Tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
```

## Tokenizers from models

```csharp
using TransformersSharp;
using TransformersSharp.Tokenizers;

var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
ReadOnlySpan<long> inputIds = tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
Console.WriteLine($"InputIds: {string.Join(", ", InputIds)}");
```

## Tokenizers as Microsoft.ML.Tokenizers

`PreTrainedTokenizerBase` implements [`Microsoft.ML.Tokenizers.Tokenizer`](https://learn.microsoft.com/dotnet/api/microsoft.ml.tokenizers.tokenizer?view=ml-dotnet-preview), so you can use it with any library which uses the `Tokenizer` type, like SemanticKernel.

```csharp
var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
var input = "How many helicopters can a human eat in one sitting?";
var tokens = tokenizer.EncodeToTokens(input, out string? normalizedText);
```
