# Sentence Transformer

The `TransformersSharp.SentenceTransformer` class provides a high-level interface for generating sentence embeddings using pre-trained models from the Hugging Face Transformers library. It simplifies the process of embedding text by handling preprocessing, model inference, and tensor conversion.

It is implemented as a `Microsoft.AI.Extensions.IEmbeddingGenerator<string, Embedding<float>>` and can be used to generate embeddings for sentences or text inputs.

## What is a Sentence Transformer?

A sentence transformer is designed to generate dense vector representations (embeddings) for input sentences. These embeddings are commonly used for tasks like:

- Semantic similarity
- Clustering
- Information retrieval
- Text classification

### Key Features of Sentence Transformers:

- **Pre-trained Models**: Leverages state-of-the-art models like `nomic-ai/nomic-embed-text-v1.5`.
- **High-Dimensional Embeddings**: Produces embeddings with hundreds of dimensions for rich semantic representation.
- **Batch Processing**: Supports generating embeddings for multiple sentences in a single call.

## Using the SentenceTransformer Class

The `SentenceTransformer` class in `TransformersSharp` provides methods to generate embeddings for sentences. Below are examples of how to use it.

### Generating Embeddings for Sentences

```csharp
using TransformersSharp;

var transformer = SentenceTransformer.FromModel("nomic-ai/nomic-embed-text-v1.5", trustRemoteCode: true);
var sentences = new List<string>
{
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are amazing for natural language processing."
};
var embeddings = await transformer.GenerateAsync(sentences);

foreach (var embedding in embeddings)
{
    Console.WriteLine($"Embedding Length: {embedding.Vector.Length}");
}
```

**Equivalent Python Code:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are amazing for natural language processing."
]
embeddings = model.encode(sentences)

for embedding in embeddings:
    print(f"Embedding Length: {len(embedding)}")
```

## Accessing Embedding Data

The `SentenceTransformer` class provides embeddings as `Embedding<float>` objects, which contain dense vectors for each input sentence.

### Example: Accessing Embedding Data

```csharp
using TransformersSharp;

var transformer = SentenceTransformer.FromModel("nomic-ai/nomic-embed-text-v1.5", trustRemoteCode: true);
var sentences = new List<string>
{
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are amazing for natural language processing."
};
var embeddings = await transformer.GenerateAsync(sentences);

foreach (var embedding in embeddings)
{
    Console.WriteLine($"Vector: {string.Join(", ", embedding.Vector.Take(10))}...");
}
```

**Equivalent Python Code:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are amazing for natural language processing."
]
embeddings = model.encode(sentences)

for embedding in embeddings:
    print(f"Vector: {embedding[:10]}...")
```