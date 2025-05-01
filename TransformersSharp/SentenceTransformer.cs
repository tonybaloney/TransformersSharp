using CSnakes.Runtime.Python;
using Microsoft.Extensions.AI;
using System.Numerics.Tensors;

namespace TransformersSharp;

public class SentenceTransformer(PyObject transformerObject) : IEmbeddingGenerator<string, Embedding<float>>
{
    public static SentenceTransformer FromModel(string model, string? device = null, string? cacheDir = null, string? revision = null, bool trustRemoteCode = false)
    {
        return new SentenceTransformer(TransformerEnvironment.SentenceTransformersWrapper.SentenceTransformer(
            model,
            device,
            cacheDir,
            revision,
            trustRemoteCode));
    }

    public void Dispose()
    {
        transformerObject.Dispose();
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        return Task.Run(() =>
        {
            var embeddings = new GeneratedEmbeddings<Embedding<float>>();
            var results = TransformerEnvironment.SentenceTransformersWrapper.EncodeSentences(transformerObject, values.ToList());
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
            ReadOnlyTensorSpan<float> tensor = results.AsFloatReadOnlyTensorSpan();
#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

            if (tensor.Lengths.Length != 2)
                throw new ArgumentException("The tensor returned is not 2-dimensional.");

            if (tensor.Lengths[0] != values.Count())
                throw new ArgumentException("The number of sentences does not match the number of embeddings returned.");

            for (int i = 0; i < tensor.Lengths[0]; i++) // Tensor for each sentence's embedding
            {
                var vector = new float[tensor.Lengths[1]];
                // TODO : Find a more efficient way to copy the tensor data to the vector
                for (int j = 0; j < tensor.Lengths[1]; j++)
                {
                    vector[j] = tensor[i, j];
                }
                embeddings.Add(new Embedding<float>(vector));
            }

            return embeddings;

        }, cancellationToken);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceType is null ? throw new ArgumentNullException(nameof(serviceType)) :
        serviceKey is not null ? null :
        serviceType.IsInstanceOfType(this) ? this :
        serviceType == typeof(SentenceTransformer) ? this :
        null;
}
