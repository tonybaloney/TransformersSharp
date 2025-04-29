using TransformersSharp;
using TransformersSharp.Tokenizers;

namespace TransformsSharp.Tests
{
    public class TransformerEnvironmentTest
    {
        [Fact]
        public void Pipeline_ShouldReturnPipelineObject()
        {
            var pipeline = TransformerEnvironment.Pipeline("text-classification", "distilbert-base-uncased-finetuned-sst-2-english");
            Assert.NotNull(pipeline);
            Assert.IsType<Pipeline>(pipeline);

            Assert.Equal("cpu", pipeline.DeviceType);
        }

        [Fact]
        public void Pipeline_Classify()
        {
            var pipeline = TextClassificationPipeline.FromModel("distilbert-base-uncased-finetuned-sst-2-english");
            var result = pipeline.Classify("I love programming!");
            Assert.Single(result);
            Assert.Equal("POSITIVE", result[0].Label);
            Assert.InRange(result[0].Score, 0.0, 1.0);
        }

        [Fact]
        public void Pipeline_ClassifyBatch()
        {
            var pipeline = TextClassificationPipeline.FromModel("distilbert-base-uncased-finetuned-sst-2-english");
            var inputs = new List<string> { "I love programming!", "I hate bugs!" };
            var results = pipeline.ClassifyBatch(inputs);
            Assert.Equal(2, results.Count);
            Assert.Equal("POSITIVE", results[0].Label);
            Assert.InRange(results[0].Score, 0.0, 1.0);
            Assert.Equal("NEGATIVE", results[1].Label);
            Assert.InRange(results[1].Score, 0.0, 1.0);
        }

        [Fact]
        public void Pipeline_TextGeneration()
        {
            var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
            var result = pipeline.Generate("How many helicopters can a human eat in one sitting?");
            Assert.Single(result);
            Assert.Contains("helicopter", result.First(), StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void Pipeline_TokenizeFromTextGenerationPipeline()
        {
            var pipeline = TextGenerationPipeline.FromModel("facebook/opt-125m");
            var InputIds = pipeline.Tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
            Assert.Equal(12, InputIds.Length);
            Assert.Equal(2, InputIds[0]);
        }

        [Fact]
        public void Pipeline_TokenizeFromPretrained()
        {
            var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
            var InputIds = tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
            Assert.Equal(12, InputIds.Length);
            Assert.Equal(2, InputIds[0]);
        }
    }
}