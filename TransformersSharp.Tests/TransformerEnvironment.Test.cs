using TransformersSharp.Pipelines;
using TransformersSharp.Tokenizers;

namespace TransformersSharp.Tests
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


        [Fact]
        public void Tokenizer_Encode()
        {
            var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
            var input = "How many helicopters can a human eat in one sitting?";
            var tokens = tokenizer.EncodeToTokens(input, out string? normalizedText);
            Assert.NotNull(tokens);
            Assert.NotEmpty(tokens);
            Assert.Null(normalizedText);

            Assert.Equal(6179, tokens[1].Id);
            Assert.Equal("How", tokens[1].Value);

            Assert.Equal("?", tokens[tokens.Count - 1].Value);
            Assert.Equal(116, tokens[tokens.Count - 1].Id);
        }

        [Fact]
        public void Tokenizer_Decode()
        {
            var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m", addSpecialTokens: false);
            var input = "How many helicopters can a human eat in one sitting?";
            var tokens = tokenizer.EncodeToTokens(input, out string? normalizedText);
            Assert.NotNull(tokens);
            Assert.NotEmpty(tokens);
            Assert.Null(normalizedText);
            string decodedText = tokenizer.Decode(tokens.Select(et => et.Id));
            Assert.Equal(input, decodedText);
        }

        [Fact]
        public void ImageClassificationPipeline_ClassifyUrl()
        {
            var pipeline = ImageClassificationPipeline.FromModel("google/mobilenet_v2_1.0_224");
            var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Replace with a valid image path
            var result = pipeline.Classify(imagePath);
            Assert.NotNull(result);
            Assert.NotEmpty(result);
            Assert.InRange(result.First().Score, 0.5, 1.0);
            Assert.Equal("hornbill", result.First().Label);
        }

        [Fact]
        public void ObjectDetectionPipeline_DetectUrl()
        {
            var pipeline = ObjectDetectionPipeline.FromModel("facebook/detr-resnet-50");
            var imagePath = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"; // Replace with a valid image path
            var result = pipeline.Detect(imagePath);
            Assert.NotNull(result);
            Assert.NotEmpty(result);
            Assert.InRange(result.First().Score, 0.5, 1.0);
            Assert.Equal("bird", result.First().Label);
            var box = result.First().Box;
            Assert.InRange(box.XMin, 0, box.XMax);
            Assert.InRange(box.YMin, 0, box.YMax);
            Assert.InRange(box.XMax, box.XMin, 400);
            Assert.InRange(box.YMax, box.YMin, 600);
        }

        [Fact]
        public void TextToAudioPipeline_Generate()
        {
            var pipeline = TextToAudioPipeline.FromModel("suno/bark-small");
            var text = "Hello, this is a test.";
            var audioResult = pipeline.Generate(text);
            Assert.False(audioResult.Audio.IsEmpty);
            Assert.True(audioResult.Audio.Length > 0);

            Assert.Equal(24000, audioResult.SamplingRate);
        }

        [Fact]
        public void AutomaticSpeechRecognitionPipeline_Transcribe()
        {
            var pipeline = AutomaticSpeechRecognitionPipeline.FromModel("openai/whisper-tiny");
            var audioPath = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"; // Replace with a valid audio path
            var result = pipeline.Transcribe(audioPath);
            Assert.NotNull(result);
            Assert.NotEmpty(result);
            Assert.Contains("parrots", result, StringComparison.OrdinalIgnoreCase);
        }
    }
}