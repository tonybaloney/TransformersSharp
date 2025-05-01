# Automatic Speech Recognition Pipeline

The `TransformersSharp.Pipelines.AutomaticSpeechRecognitionPipeline` class provides a high-level interface for transcribing audio into text using pre-trained models from the Hugging Face Transformers library. It simplifies the process of speech-to-text conversion by handling preprocessing, model inference, and decoding.

## What is an Automatic Speech Recognition Pipeline?

An automatic speech recognition (ASR) pipeline is designed to convert spoken language in audio files into written text. It is commonly used for tasks like:

- Voice assistants
- Transcription services
- Accessibility tools

### Key Features of Automatic Speech Recognition Pipelines:

- **Pre-trained Models**: Leverages state-of-the-art models like Whisper and others.
- **Flexible Input**: Supports both local audio files and URLs.
- **High Accuracy**: Provides accurate transcriptions for various languages and accents.

## Using the AutomaticSpeechRecognitionPipeline Class

The `AutomaticSpeechRecognitionPipeline` class in `TransformersSharp` provides methods to transcribe audio files. Below are examples of how to use it.

### Transcribing Audio from a File

```csharp
using TransformersSharp.Pipelines;

var pipeline = AutomaticSpeechRecognitionPipeline.FromModel("openai/whisper-small");
var audioPath = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"; // Replace with a valid audio file path
var transcription = pipeline.Transcribe(audioPath);
Console.WriteLine($"Transcription: {transcription}");
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
audio_path = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"  # Replace with a valid audio file path
transcription = pipeline(audio_path)
print(f"Transcription: {transcription}")
```

## Options for `FromModel`

The `FromModel` method provides several options to customize the behavior of the pipeline:

- **`model`** *(string, required)*: The name or path of the pre-trained model. For example, "openai/whisper-small".
- **`torchDtype`** *(TorchDtype, optional)*: Specifies the data type for PyTorch tensors. For example, `TorchDtype.Float32`.
- **`device`** *(string, optional)*: Specifies the device to run the model on. For example, "cpu" or "cuda:0".
- **`trustRemoteCode`** *(bool, optional)*: Indicates whether to trust remote code execution. Default is `false`. Set to `true` only if you trust the source of the model.

## Customizing Speech Recognition

The `Transcribe` method allows users to transcribe audio files with minimal configuration. For advanced use cases, users can preprocess audio files or adjust model parameters externally.

### Example: Transcribing with a Local File

```csharp
using TransformersSharp.Pipelines;

var pipeline = AutomaticSpeechRecognitionPipeline.FromModel("openai/whisper-small");
var audioPath = "/path/to/local/audio/file.flac"; // Replace with a valid local file path
var transcription = pipeline.Transcribe(audioPath);
Console.WriteLine($"Transcription: {transcription}");
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
audio_path = "/path/to/local/audio/file.flac"  # Replace with a valid local file path
transcription = pipeline(audio_path)
print(f"Transcription: {transcription}")
```