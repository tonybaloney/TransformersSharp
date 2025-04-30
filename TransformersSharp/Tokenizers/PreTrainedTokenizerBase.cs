using CSnakes.Runtime.Python;
using Microsoft.ML.Tokenizers;
using System.Buffers;

namespace TransformersSharp.Tokenizers;


/// <summary>
/// Represents a base class for pre-trained tokenizers, providing functionality for tokenization and decoding.
/// </summary>
/// <remarks>
/// This class is designed to integrate with the Hugging Face Transformers library, allowing users to leverage pre-trained tokenizers for text processing tasks.
/// It also implements the `Microsoft.ML.Tokenizers.Tokenizer` interface, enabling compatibility with libraries like SemanticKernel.
/// </remarks>
public class PreTrainedTokenizerBase : Tokenizer
{
	/// <summary>
	/// Gets the underlying Python tokenizer object.
	/// </summary>
	internal PyObject TokenizerObject { get; }
	private bool addSpecialTokens;

	/// <summary>
	/// Initializes a new instance of the <see cref="PreTrainedTokenizerBase"/> class with the specified tokenizer object and special token handling.
	/// </summary>
	/// <param name="tokenizerObject">The Python tokenizer object to wrap.</param>
	/// <param name="addSpecialTokens">Indicates whether to add special tokens during tokenization. Default is <c>true</c>.</param>
	internal PreTrainedTokenizerBase(PyObject tokenizerObject, bool addSpecialTokens = true)
	{
		TokenizerObject = tokenizerObject;
		this.addSpecialTokens = addSpecialTokens;
	}

	/// <summary>
	/// Loads a pre-trained tokenizer from the specified model.
	/// </summary>
	/// <param name="model">The name or path of the pre-trained model. For example, "facebook/opt-125m".</param>
	/// <param name="cacheDir">The directory to cache the model files. Optional. If not specified, the default cache directory is used.</param>
	/// <param name="forceDownload">Indicates whether to force re-downloading the model files. Default is <c>false</c>.</param>
	/// <param name="revision">The specific model revision to load. Default is "main". Use this to specify a particular version of the model.</param>
	/// <param name="trustRemoteCode">Indicates whether to trust remote code execution. Default is <c>false</c>. Set to <c>true</c> only if you trust the source of the model.</param>
	/// <param name="addSpecialTokens">Indicates whether to add special tokens during tokenization. Default is <c>true</c>. Special tokens are often required for tasks like sequence classification or text generation.</param>
	/// <returns>A new instance of <see cref="PreTrainedTokenizerBase"/> initialized with the specified model.</returns>
	/// <example>
	/// Example usage:
	/// <code>
	/// var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
	/// ReadOnlySpan<long> inputIds = tokenizer.Tokenize("Hello, world!");
	/// Console.WriteLine(string.Join(", ", inputIds.ToArray()));
	/// </code>
	/// </example>
	public static PreTrainedTokenizerBase FromPretrained(string model, string? cacheDir = null, bool forceDownload = false, string revision = "main", bool trustRemoteCode = false, bool addSpecialTokens = true)
	{
		return new PreTrainedTokenizerBase(TransformerEnvironment.TransformersWrapper.TokenizerFromPretrained(model, cacheDir, forceDownload, revision, trustRemoteCode), addSpecialTokens);
	}

	/// <summary>
	/// Tokenizes the given text into a sequence of token IDs.
	/// </summary>
	/// <param name="text">The input text to tokenize. For example, "How many helicopters can a human eat in one sitting?"</param>
	/// <returns>A read-only span of token IDs representing the tokenized text.</returns>
	/// <example>
	/// Example usage:
	/// <code>
	/// var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
	/// ReadOnlySpan<long> inputIds = tokenizer.Tokenize("How many helicopters can a human eat in one sitting?");
	/// Console.WriteLine(string.Join(", ", inputIds.ToArray()));
	/// </code>
	/// </example>
	public ReadOnlySpan<long> Tokenize(string text)
	{
		return TransformerEnvironment.TransformersWrapper.TokenizerTextAsNdarray(TokenizerObject, text).AsInt64ReadOnlySpan();
	}


	#region Microsoft.ML.Tokenizers.Tokenizer
	/// <summary>
	/// Encodes the given text into a sequence of tokens with their corresponding offsets.
	/// </summary>
	/// <param name="text">The input text to encode. If <c>null</c>, <paramref name="textSpan"/> will be used.</param>
	/// <param name="textSpan">A span of characters representing the input text.</param>
	/// <param name="settings">The encoding settings to apply.</param>
	/// <returns>An <see cref="EncodeResults{T}"/> containing the encoded tokens and their offsets.</returns>
	/// <example>
	/// Example usage:
	/// <code>
	/// var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
	/// var input = "How many helicopters can a human eat in one sitting?";
	/// var tokens = tokenizer.EncodeToTokens(input, out string? normalizedText);
	/// foreach (var token in tokens.Tokens)
	/// {
	///     Console.WriteLine($"Token: {token.Text}, Start: {token.Range.Start}, End: {token.Range.End}");
	/// }
	/// </code>
	/// </example>
	protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
    {
		// Our API only supports string input
		text ??= textSpan.ToString();

		var (inputIdsBuffer, mappingPairsBuffer) = TransformerEnvironment.TransformersWrapper.TokenizerTextWithOffsets(TokenizerObject, text, addSpecialTokens: addSpecialTokens);
		List<EncodedToken> tokens = [];
		var inputIds = inputIdsBuffer.AsInt64ReadOnlySpan();
		var mappingPairs = mappingPairsBuffer.AsInt64ReadOnlySpan2D();

		for (int i = 0; i < inputIds.Length; i++)
		{
			int start = (int)mappingPairs[i, 0];
			int end = (int)mappingPairs[i, 1];
			var token = new EncodedToken((int)inputIds[i], text[start..end], new Range(new Index(start), new Index(end)));
			tokens.Add(token);
		}

		return new EncodeResults<EncodedToken>()
		{
			Tokens = tokens,
		};
	}

	/// <summary>
	/// Decodes a sequence of token IDs into a string.
	/// </summary>
	/// <param name="ids">The sequence of token IDs to decode.</param>
	/// <param name="destination">The span to write the decoded string to.</param>
	/// <param name="idsConsumed">The number of token IDs consumed during decoding.</param>
	/// <param name="charsWritten">The number of characters written to the destination span.</param>
	/// <returns>An <see cref="OperationStatus"/> indicating the result of the decoding operation.</returns>
	/// <example>
	/// Example usage:
	/// <code>
	/// var tokenizer = PreTrainedTokenizerBase.FromPretrained("facebook/opt-125m");
	/// var tokenIds = new List<int> { 101, 2009, 2003, 1037, 2204, 2154, 102 };
	/// Span<char> decodedText = stackalloc char[100];
	/// tokenizer.Decode(tokenIds, decodedText, out int idsConsumed, out int charsWritten);
	/// Console.WriteLine(new string(decodedText.Slice(0, charsWritten)));
	/// </code>
	/// </example>
    public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
    {
		string result = TransformerEnvironment.TransformersWrapper.TokenizerDecode(TokenizerObject, [.. ids.Select(i => (long)i)], skipSpecialTokens: addSpecialTokens);
		if (result.Length > destination.Length)
		{
			idsConsumed = 0;
			charsWritten = 0;
			return OperationStatus.DestinationTooSmall;
		}
		result.CopyTo(destination);
		idsConsumed = ids.Count();
		charsWritten = result.Length;
		return OperationStatus.Done;
    }
	#endregion
}