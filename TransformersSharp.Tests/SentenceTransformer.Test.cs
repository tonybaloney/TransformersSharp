using Microsoft.Extensions.AI;

namespace TransformersSharp.Tests;

public class SentenceTransformerTests
{
    [Fact]
    async public Task SentenceTransformer_ShouldGenerateEmbeddings()
    {
        var transformer = SentenceTransformer.FromModel("nomic-ai/nomic-embed-text-v1.5", trustRemoteCode: true);
        Assert.NotNull(transformer);
        Assert.IsType<SentenceTransformer>(transformer);

        var sentences = new List<string>
        {
            "The quick brown fox jumps over the lazy dog.",
            "Transformers are amazing for natural language processing."
        };

        var embeddings = await transformer.GenerateAsync(sentences);

        Assert.NotNull(embeddings);
        Assert.Equal(sentences.Count, embeddings.Count);
        Assert.All(embeddings, embedding =>
        {
            Assert.NotNull(embedding);
            Assert.IsType<Embedding<float>>(embedding);
            Assert.Equal(768, embedding.Vector.Length); // Assuming the model produces 768-dimensional embeddings
        });
    }
}
