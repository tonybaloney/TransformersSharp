# TransformersSharp

A little wrapper for Hugging Face Transformers in C#.

Created using 

This package will fetch Python, PyTorch, and Hugging Face Transformers automatically, so you don't need to install them manually.


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



### Tokenizers from Pipelines

```csharp
using TransformersSharp;

var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
ReadOnlySpan<long> inputIds = pipeline.Tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
```

### Tokenizers

```csharp
using TransformersSharp;
using TransformersSharp.Tokenizers;

var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
ReadOnlySpan<long> inputIds = tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
Console.WriteLine($"InputIds: {string.Join(", ", InputIds)}");
```

