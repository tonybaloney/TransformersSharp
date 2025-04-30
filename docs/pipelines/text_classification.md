# Text Classification Pipeline

The `TransformersSharp.TextClassificationPipeline` class provides a high-level interface for performing text classification tasks using pre-trained models from the Hugging Face Transformers library. It simplifies the process of classifying text by handling tokenization, model inference, and decoding.

## What is a Text Classification Pipeline?

A text classification pipeline is designed to classify input text into predefined categories. It is commonly used for tasks like:
- Sentiment analysis
- Topic classification
- Spam detection

### Key Features of Text Classification Pipelines:
- **Pre-trained Models**: Leverages state-of-the-art models like BERT, DistilBERT, and others.
- **Batch Processing**: Supports single and batch inputs for efficient processing.
- **Confidence Scores**: Provides confidence scores for each classification label.

## Using the TextClassificationPipeline Class

The `TextClassificationPipeline` class in `TransformersSharp` provides methods to classify text. Below are examples of how to use it.

### Classifying a Single Input

```csharp
using TransformersSharp;

var pipeline = TextClassificationPipeline.FromModel("distilbert-base-uncased-finetuned-sst-2-english");
var result = pipeline.Classify("I love programming!");
foreach (var (label, score) in result)
{
    Console.WriteLine($"Label: {label}, Score: {score}");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = pipeline("I love programming!")
for item in result:
    print(f"Label: {item['label']}, Score: {item['score']}")
```

### Classifying Batch Inputs

```csharp
using TransformersSharp;

var pipeline = TextClassificationPipeline.FromModel("distilbert-base-uncased-finetuned-sst-2-english");
var inputs = new List<string> { "I love programming!", "I hate bugs!" };
var results = pipeline.ClassifyBatch(inputs);
foreach (var (label, score) in results)
{
    Console.WriteLine($"Label: {label}, Score: {score}");
}
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
inputs = ["I love programming!", "I hate bugs!"]
results = pipeline(inputs)
for item in results:
    print(f"Label: {item['label']}, Score: {item['score']}")
```

## Accessing the Tokenizer

The `TextClassificationPipeline` class provides access to the associated tokenizer through the `Tokenizer` property. This allows users to preprocess inputs or decode outputs manually if needed.

### Example: Accessing the Tokenizer

```csharp
using TransformersSharp;

var pipeline = TextClassificationPipeline.FromModel("distilbert-base-uncased-finetuned-sst-2-english");
var tokenizer = pipeline.Tokenizer;
var inputIds = tokenizer.Tokenize("I love programming!");
Console.WriteLine(string.Join(", ", inputIds.ToArray()));
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = pipeline.tokenizer
input_ids = tokenizer("I love programming!", return_tensors="pt")["input_ids"]
print(input_ids.tolist())
```