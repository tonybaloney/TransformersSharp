# Object Detection Pipeline

The `TransformersSharp.ObjectDetectionPipeline` class provides a high-level interface for performing object detection tasks using pre-trained models from the Hugging Face Transformers library. It simplifies the process of detecting objects in images by handling preprocessing, model inference, and decoding.

## What is an Object Detection Pipeline?

An object detection pipeline is designed to identify and locate objects within an image. It provides both the labels of detected objects and their bounding boxes. Common use cases include:

- Autonomous vehicles
- Surveillance systems
- Image annotation

### Key Features of Object Detection Pipelines:

- **Pre-trained Models**: Leverages state-of-the-art models like DETR, YOLO, and others.
- **Bounding Boxes**: Provides precise coordinates for detected objects.
- **Confidence Scores**: Includes confidence scores for each detected object.

## Using the ObjectDetectionPipeline Class

The `ObjectDetectionPipeline` class in `TransformersSharp` provides methods to detect objects in images. Below are examples of how to use it.

### Detecting Objects in a Single Image

```csharp
using TransformersSharp.Pipelines;

var pipeline = ObjectDetectionPipeline.FromModel("facebook/detr-resnet-50");
var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Replace with a valid image path
var results = pipeline.Detect(imagePath);
foreach (var result in results)
{
    Console.WriteLine($"Label: {result.Label}, Score: {result.Score}, Box: [XMin: {result.Box.XMin}, YMin: {result.Box.YMin}, XMax: {result.Box.XMax}, YMax: {result.Box.YMax}]");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("object-detection", model="facebook/detr-resnet-50")
image_path = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # Replace with a valid image path
results = pipeline(image_path)
for item in results:
    box = item['box']
    print(f"Label: {item['label']}, Score: {item['score']}, Box: [XMin: {box['xmin']}, YMin: {box['ymin']}, XMax: {box['xmax']}, YMax: {box['ymax']}]")
```

### Customizing Detection Parameters

```csharp
using TransformersSharp.Pipelines;

var pipeline = ObjectDetectionPipeline.FromModel("facebook/detr-resnet-50");
var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Replace with a valid image path
var results = pipeline.Detect(imagePath, threshold: 0.7);
foreach (var result in results)
{
    Console.WriteLine($"Label: {result.Label}, Score: {result.Score}, Box: [XMin: {result.Box.XMin}, YMin: {result.Box.YMin}, XMax: {result.Box.XMax}, YMax: {result.Box.YMax}]");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("object-detection", model="facebook/detr-resnet-50")
image_path = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # Replace with a valid image path
results = pipeline(image_path, threshold=0.7)
for item in results:
    box = item['box']
    print(f"Label: {item['label']}, Score: {item['score']}, Box: [XMin: {box['xmin']}, YMin: {box['ymin']}, XMax: {box['xmax']}, YMax: {box['ymax']}]")
```

## Accessing the Tokenizer

The `ObjectDetectionPipeline` class does not use a tokenizer, as it is designed for image inputs. Instead, it preprocesses image data directly.

## Customizing Object Detection

The `Detect` method allows users to customize the detection process by specifying parameters like:

- **`threshold`**: Sets the confidence threshold for detected objects.
- **`timeout`**: Sets a timeout for the detection process.

### Example: Customizing Parameters

```csharp
using TransformersSharp.Pipelines;

var pipeline = ObjectDetectionPipeline.FromModel("facebook/detr-resnet-50");
var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Replace with a valid image path
var results = pipeline.Detect(imagePath, threshold: 0.8, timeout: 10);
foreach (var result in results)
{
    Console.WriteLine($"Label: {result.Label}, Score: {result.Score}, Box: [XMin: {result.Box.XMin}, YMin: {result.Box.YMin}, XMax: {result.Box.XMax}, YMax: {result.Box.YMax}]");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("object-detection", model="facebook/detr-resnet-50")
image_path = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # Replace with a valid image path
results = pipeline(image_path, threshold=0.8, timeout=10)
for item in results:
    box = item['box']
    print(f"Label: {item['label']}, Score: {item['score']}, Box: [XMin: {box['xmin']}, YMin: {box['ymin']}, XMax: {box['xmax']}, YMax: {box['ymax']}]")
```