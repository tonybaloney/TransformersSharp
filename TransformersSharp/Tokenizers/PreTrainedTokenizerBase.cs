using CSnakes.Runtime.Python;
using Microsoft.ML.Tokenizers;
using System.Buffers;

namespace TransformersSharp.Tokenizers;


public class PreTrainedTokenizerBase : Tokenizer
{
	internal PyObject TokenizerObject { get; }

	internal PreTrainedTokenizerBase(PyObject tokenizerObject)
	{
		TokenizerObject = tokenizerObject;
	}

	public static PreTrainedTokenizerBase FromPretrained(string model, string? cacheDir = null, bool forceDownload = false, string revision = "main", bool trustRemoteCode = false)
	{
		return new PreTrainedTokenizerBase(TransformerEnvironment.TransformersWrapper.TokenizerFromPretrained(model, cacheDir, forceDownload, revision, trustRemoteCode));
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

		// TODO: Encode settings?

		var (inputIds, mappingPairs) = TransformerEnvironment.TransformersWrapper.TokenizerTextWithOffsets(TokenizerObject, text);
		List<EncodedToken> tokens = [];

		for (int i = 0; i < inputIds.Count; i++)
		{
			int start = (int)mappingPairs[i].Item1;
			int end = (int)mappingPairs[i].Item2;
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
        throw new NotImplementedException();
    }
	#endregion
}