# Tokenizers

The `TransformersSharp.Tokenizers.PreTrainedTokenizerBase` class provides a mapping to the [`PreTrainedTokenizerBase`](https://huggingface.co/docs/transformers/v4.51.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase).

## What are Hugging Face Tokenizers?

Hugging Face tokenizers are tools designed to preprocess text data for natural language processing (NLP) tasks. They convert raw text into numerical representations (token IDs) that machine learning models can understand. These tokenizers are pre-trained on large datasets and are optimized for specific models, ensuring compatibility and efficiency.

## Tokenizer Pipeline in Hugging Face Transformers SDK

The tokenizer pipeline in the Hugging Face Transformers SDK involves several steps:

1. **Normalization**: Text is normalized by lowercasing, removing accents, or applying other transformations.
2. **Pre-tokenization**: Text is split into smaller chunks, such as words or subwords.
3. **Tokenization**: Chunks are converted into tokens based on the tokenizer's vocabulary.
4. **Post-processing**: Special tokens are added, and the sequence is padded or truncated to the desired length.

## Tokenizers from Pipelines

[Text Generation Pipelines](pipelines/text_generation.md) include a tokenizer. You can access the tokenizer that comes with the model using the `Tokenizer` property on the `TextGenerationPipeline` class:

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
ReadOnlySpan<long> inputIds = pipeline.Tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
```

## Tokenizers from Models

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
foreach (var token in tokens.Tokens)
{
    Console.WriteLine($"Token: {token.Text}, Start: {token.Range.Start}, End: {token.Range.End}");
}
```

## Advanced Example: Decoding Tokens

You can also decode token IDs back into text using the `Decode` method:

```csharp
var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
var tokenIds = new List<int> { 101, 2009, 2003, 1037, 2204, 2154, 102 };
Span<char> decodedText = stackalloc char[100];
tokenizer.Decode(tokenIds, decodedText, out int idsConsumed, out int charsWritten);
Console.WriteLine(new string(decodedText.Slice(0, charsWritten)));
```

## Special Tokens and Their Usage

Special tokens are used to mark specific parts of the input text, like `<bos>` (beginning of stream). The special tokens depend on the encoding. 

By default, encoding will include the encoding's special tokens. You can turn this off:

You can customize the tokenization process by modifying parameters like `addSpecialTokens` or by using a custom vocabulary. For example:

```csharp
var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m", addSpecialTokens: false);
ReadOnlySpan<long> inputIds = tokenizer.Tokenize("Custom tokenization example.");
Console.WriteLine($"InputIds: {string.Join(", ", inputIds.ToArray())}");
```

### Options for `PreTrainedTokenizerBase.FromPretrained`

The `FromPretrained` method provides several options to customize the behavior of the tokenizer:

- **`model`** *(string, required)*: The name or path of the pre-trained model. For example, "facebook/opt-125m".
- **`cacheDir`** *(string, optional)*: The directory to cache the model files. If not specified, the default cache directory is used.
- **`forceDownload`** *(bool, optional)*: Indicates whether to force re-downloading the model files. Default is `false`.
- **`revision`** *(string, optional)*: The specific model revision to load. Default is "main". Use this to specify a particular version of the model.
- **`trustRemoteCode`** *(bool, optional)*: Indicates whether to trust remote code execution. Default is `false`. Set to `true` only if you trust the source of the model.
- **`addSpecialTokens`** *(bool, optional)*: Indicates whether to add special tokens during tokenization. Default is `true`. Special tokens are often required for tasks like sequence classification or text generation.
