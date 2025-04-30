# Image Classification Pipeline

The `TransformersSharp.ImageClassificationPipeline` class provides a high-level interface for performing image classification tasks using pre-trained models from the Hugging Face Transformers library. It simplifies the process of classifying images by handling preprocessing, model inference, and decoding.

## What is an Image Classification Pipeline?

An image classification pipeline is designed to classify input images into predefined categories. It is commonly used for tasks like:

- Object recognition
- Scene classification
- Image tagging

### Key Features of Image Classification Pipelines:

- **Pre-trained Models**: Leverages state-of-the-art models like MobileNet, ResNet, and others.
- **Top-k Results**: Returns the top-k classification results with confidence scores.
- **Flexible Input**: Supports both local image files and URLs.

## Using the ImageClassificationPipeline Class

The `ImageClassificationPipeline` class in `TransformersSharp` provides methods to classify images. Below are examples of how to use it.

### Classifying a Single Image

```csharp
using TransformersSharp.Pipelines;

var pipeline = ImageClassificationPipeline.FromModel("google/mobilenet_v2_1.0_224");
var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Can be a local path
var result = pipeline.Classify(imagePath);
foreach (var classification in result)
{
    Console.WriteLine($"Label: {classification.Label}, Score: {classification.Score}");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
image_path = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # Can be a local path
results = pipeline(image_path)
for item in results:
    print(f"Label: {item['label']}, Score: {item['score']}")
```

### Classifying with Custom Parameters

```csharp
using TransformersSharp.Pipelines;

var pipeline = ImageClassificationPipeline.FromModel("google/mobilenet_v2_1.0_224");
var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Replace with a valid image path
var result = pipeline.Classify(imagePath, topk: 3);
foreach (var classification in result)
{
    Console.WriteLine($"Label: {classification.Label}, Score: {classification.Score}");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
image_path = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # Replace with a valid image path
results = pipeline(image_path, top_k=3)
for item in results:
    print(f"Label: {item['label']}, Score: {item['score']}")
```

## Accessing the Tokenizer

The `ImageClassificationPipeline` class does not use a tokenizer, as it is designed for image inputs. Instead, it preprocesses image data directly.

## Customizing Image Classification

The `Classify` method allows users to customize the classification process by specifying parameters like:

- **`functionToApply`**: Specifies the function to apply to the model's output (e.g., "softmax", "sigmoid", or "none").
- **`topk`**: Limits the number of top results to return.
- **`timeout`**: Sets a timeout for the classification process.

### Example: Customizing Parameters

```csharp
using TransformersSharp.Pipelines;

var pipeline = ImageClassificationPipeline.FromModel("google/mobilenet_v2_1.0_224");
var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Replace with a valid image path
var result = pipeline.Classify(imagePath, functionToApply: "softmax", topk: 5);
foreach (var classification in result)
{
    Console.WriteLine($"Label: {classification.Label}, Score: {classification.Score}");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
image_path = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # Replace with a valid image path
results = pipeline(image_path, function_to_apply="softmax", top_k=5)
for item in results:
    print(f"Label: {item['label']}, Score: {item['score']}")
```