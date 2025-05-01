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

## Installing ffmpeg

To use the `AutomaticSpeechRecognitionPipeline`, you need to have `ffmpeg` installed on your system. 

You can install it on Windows using `winget`:

```bash
winget install ffmpeg
```

You'll probably need to restart your terminal or IDE after installing `ffmpeg` to ensure it's available in your system's PATH.

On Linux, you can install it using your package manager. For example, on Ubuntu:

```bash
sudo apt-get install ffmpeg
```

On macOS, you can use Homebrew:

```bash
brew install ffmpeg
```

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

## Using the ASR Pipeline with the SpeechToTextClient (Microsoft.Extensions.AI.ISpeechToTextClient)

The `SpeechToTextClient` class in the `TransformersSharp.MEAI` namespace provides a convenient way to use the Automatic Speech Recognition (ASR) pipeline for transcribing audio. It simplifies the process of working with audio streams and supports both synchronous and streaming transcription.

### Transcribing Audio from a File

```csharp
using TransformersSharp.MEAI;

var speechClient = SpeechToTextClient.FromModel("openai/whisper-tiny");
using var audioStream = new MemoryStream(File.ReadAllBytes("sample.flac"));
var response = await speechClient.GetTextAsync(audioStream);

Console.WriteLine($"Transcription: {response.Text}");
```

**Equivalent Python Code:**

```python
from transformers import pipeline

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
audio_path = "sample.flac"  # Replace with a valid local file path
transcription = pipeline(audio_path)
print(f"Transcription: {transcription}")
```

### Streaming Transcription

The `SpeechToTextClient` also supports streaming transcription, which is useful for processing large audio files or real-time audio streams. Although few (if any) models support streaming, the API is designed to handle it.

```csharp
using TransformersSharp.MEAI;

var speechClient = SpeechToTextClient.FromModel("openai/whisper-tiny");
using var audioStream = new MemoryStream(File.ReadAllBytes("sample.flac"));
await foreach (var update in speechClient.GetStreamingTextAsync(audioStream))
{
    Console.WriteLine($"Partial Transcription: {update.Text}");
}
```
