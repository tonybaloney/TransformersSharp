using CSnakes.Runtime.Python;
using TransformersSharp.Tokenizers;

namespace TransformersSharp.Pipelines
{
    public class Pipeline
    {
        public string DeviceType { get; private set; }

        internal PyObject PipelineObject { get; }

        internal Pipeline(PyObject pipelineObject)
        {
            PipelineObject = pipelineObject;
            DeviceType = pipelineObject.GetAttr("device").ToString();
        }

        internal IReadOnlyList<IReadOnlyDictionary<string, PyObject>> RunPipeline(string input)
        {
            return TransformerEnvironment.TransformersWrapper.CallPipeline(PipelineObject, input);
        }

        internal IReadOnlyList<IReadOnlyDictionary<string, PyObject>> RunPipeline(IReadOnlyList<string> inputs)
        {
            return TransformerEnvironment.TransformersWrapper.CallPipeline(PipelineObject, inputs);
        }

        private PreTrainedTokenizerBase? _tokenizer = null;
        public PreTrainedTokenizerBase Tokenizer
        {
            get
            {
                _tokenizer ??= new PreTrainedTokenizerBase(PipelineObject.GetAttr("tokenizer"));
                return _tokenizer;
            }
        }
    }
}
