using CSnakes.Runtime.Python;

namespace TransformersSharp.Tokenizers;


public class PreTrainedTokenizerBase 
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
}