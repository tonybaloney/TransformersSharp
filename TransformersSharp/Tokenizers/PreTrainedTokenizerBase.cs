using CSnakes.Runtime.Python;
using Microsoft.ML.Tokenizers;
using System.Buffers;

namespace TransformersSharp.Tokenizers;


public class PreTrainedTokenizerBase : Tokenizer
{
	internal PyObject TokenizerObject { get; }
	private bool addSpecialTokens;

	internal PreTrainedTokenizerBase(PyObject tokenizerObject, bool addSpecialTokens = true)
	{
		TokenizerObject = tokenizerObject;
		this.addSpecialTokens = addSpecialTokens;
	}

	public static PreTrainedTokenizerBase FromPretrained(string model, string? cacheDir = null, bool forceDownload = false, string revision = "main", bool trustRemoteCode = false, bool addSpecialTokens = true)
	{
		return new PreTrainedTokenizerBase(TransformerEnvironment.TransformersWrapper.TokenizerFromPretrained(model, cacheDir, forceDownload, revision, trustRemoteCode), addSpecialTokens);
	}

	public ReadOnlySpan<long> Tokenize(string text)
	{
		return TransformerEnvironment.TransformersWrapper.TokenizerTextAsNdarray(TokenizerObject, text).AsInt64ReadOnlySpan();
	}


	#region Microsoft.ML.Tokenizers.Tokenizer
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