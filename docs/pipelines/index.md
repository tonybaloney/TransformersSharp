# Pipelines

The `TransformersSharp.Pipeline` class provides a base implementation for running pre-trained models using the Hugging Face Transformers library. It acts as a bridge between the Python-based Hugging Face pipelines and .NET applications.

## What is a Pipeline?

A pipeline is a high-level abstraction that simplifies the process of using pre-trained models. Depending on the model, it handles tokenization, model inference, and decoding, allowing users to focus on their specific use cases without worrying about the underlying complexities.

## Using the Pipeline Class

The `Pipeline` base class in `TransformersSharp` provides methods to run pre-trained models on input data.

Based on the task, the `Pipeline` class should be inherited, some examples include

- [Text Generation Pipeline (`TextGenerationPipeline`)](text_generation.md)
- [Text Classification Pipeline (`TextClassificationPipeline)](text_classification.md)
- [Image Classification Pipeline(`ImageClassificationPipeline`)](image_classification.md)
- [Object Detection Pipeline (`ObjectDetectionPipeline`)](object_detection.md)
- [Text to Audio Pipeline (`TextToAudioPipeline`)](text_to_audio.md)

## Accessing the Tokenizer

The `Pipeline` class provides access to the associated tokenizer through the `Tokenizer` property. This allows users to preprocess inputs or decode outputs manually if needed.

### Example: Accessing the Tokenizer

```csharp
using TransformersSharp;

var pipeline = new Pipeline(pipelineObject); // Assume pipelineObject is initialized
var tokenizer = pipeline.Tokenizer;
var inputIds = tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
Console.WriteLine(string.Join(", ", inputIds.ToArray()));
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-classification", model="facebook/opt-125m")
tokenizer = pipeline.tokenizer
input_ids = tokenizer("How many helicopters can a human eat in one sitting?", return_tensors="pt")["input_ids"]
print(input_ids.tolist())
```

## Device Management

The `Pipeline` class automatically detects the device (CPU or GPU) being used by the underlying model. This information is accessible through the `DeviceType` property.

### Example: Checking the Device Type

```csharp
using TransformersSharp;

var pipeline = new Pipeline(pipelineObject); // Assume pipelineObject is initialized
Console.WriteLine($"Device Type: {pipeline.DeviceType}");
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-classification", model="facebook/opt-125m")
print(f"Device Type: {pipeline.device}")
```

