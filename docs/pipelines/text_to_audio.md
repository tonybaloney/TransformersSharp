# Text to Audio Pipeline

The `TransformersSharp.TextToAudioPipeline` class provides a high-level interface for generating audio from text using pre-trained models from the Hugging Face Transformers library. It simplifies the process of converting text into audio by handling preprocessing, model inference, and decoding.

## What is a Text to Audio Pipeline?

A text-to-audio pipeline is designed to generate audio representations of input text. It is commonly used for tasks like:

- Text-to-speech (TTS) systems
- Audio content generation
- Assistive technologies

### Key Features of Text to Audio Pipelines:

- **Pre-trained Models**: Leverages state-of-the-art models like Bark and others.
- **High-Quality Audio**: Generates audio with high fidelity and naturalness.
- **Customizable Parameters**: Allows fine-tuning of generation parameters.

## Using the TextToAudioPipeline Class

The `TextToAudioPipeline` class in `TransformersSharp` provides methods to generate audio from text. Below are examples of how to use it.

### Generating Audio from Text

```csharp
using TransformersSharp;

var pipeline = TextToAudioPipeline.FromModel("suno/bark-small");
var text = "Hello, this is a test.";
var audioResult = pipeline.Generate(text);
Console.WriteLine($"Audio Length: {audioResult.Audio.Length}, Sampling Rate: {audioResult.SamplingRate}");
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-to-audio", model="suno/bark-small")
text = "Hello, this is a test."
audio_result = pipeline(text)
print(f"Audio Length: {len(audio_result['audio'])}, Sampling Rate: {audio_result['sampling_rate']}")
```

## Accessing the Audio Data

The `TextToAudioPipeline` class provides the generated audio as a `ReadOnlySpan2D<float>` (a 2 channel wave) for efficient processing. The sampling rate is also returned for playback or further processing.

### Example: Accessing Audio Data

```csharp
using TransformersSharp;

var pipeline = TextToAudioPipeline.FromModel("suno/bark-small");
var text = "Hello, this is a test.";
var audioResult = pipeline.Generate(text);

// Access audio data
foreach (var sample in audioResult.Audio)
{
    Console.WriteLine(sample);
}

// Access sampling rate
Console.WriteLine($"Sampling Rate: {audioResult.SamplingRate}");
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("text-to-audio", model="suno/bark-small")
text = "Hello, this is a test."
audio_result = pipeline(text)

# Access audio data
for sample in audio_result['audio']:
    print(sample)

# Access sampling rate
print(f"Sampling Rate: {audio_result['sampling_rate']}")
```