# TransformersSharp

![Logo](docs/media/logo.png)

A little wrapper for Hugging Face Transformers in C#. This is not a comprehensive 1:1 mapping of the whole HuggingFace [transformers](https://pypi.org/transformers) package, because the API is enormous.

If you need a specific feature, toggle or pipeline API clone this repo and make adjustments.

This project was created using [CSnakes](https://github.com/tonybaloney/CSnakes) and will fetch Python, PyTorch, and Hugging Face Transformers automatically, so you don't need to install them manually.

[Full Documentation](https://tonybaloney.github.io/TransformersSharp/)

## Features

- Tokenizer API based on [`PreTrainedTokenizerBase`](https://huggingface.co/docs/transformers/v4.51.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
- Tokenizer shim to the [Microsoft.ML.Tokenizers](https://learn.microsoft.com/dotnet/api/microsoft.ml.tokenizers.tokenizer?view=ml-dotnet-preview) base class
- Utility classes to access pipelines like:
    - [Text Generation Pipeline (`TextGenerationPipeline`)](https://tonybaloney.github.io/TransformersSharp/docs/pipelines/text_generation)
    - [Text Classification Pipeline (`TextClassificationPipeline)](https://tonybaloney.github.io/TransformersSharp/docs/pipelines/text_classification)
    - [Image Classification Pipeline(`ImageClassificationPipeline`)](https://tonybaloney.github.io/TransformersSharp/docs/pipelines/image_classification)
    - [Object Detection Pipeline (`ObjectDetectionPipeline`)](https://tonybaloney.github.io/TransformersSharp/docs/pipelines/object_detection)
    - [Automatic Speech Recognition (`AutomaticSpeechRecognitionPipeline`)](https://tonybaloney.github.io/TransformersSharp/docs/pipelines/auto_speech_recognition)
- [Sentence Transformers (Embeddings)](https://tonybaloney.github.io/TransformersSharp/docs/sentence_transformers)

## Usage

For example, the Python code:

```python
from transformers import pipeline
import torch

pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)
results = pipeline("Tell me a story about a brave knight.", max_length=100, temperature=0.7)
for result in results:
    print(result["generated_text"])
```

Is equivalent to:

```csharp
using TransformersSharp.Pipelines;

var pipeline = TextGenerationPipeline.FromModel("Qwen/Qwen2.5-0.5B", TorchDtype.BFloat16);
var results = pipeline.Generate("Tell me a story about a brave knight.", maxLength: 100, temperature: 0.7);
foreach (var result in results)
{
    Console.WriteLine(result);
}
```

