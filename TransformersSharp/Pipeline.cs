using CSnakes.Runtime.Python;

namespace TransformersSharp
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

        internal IReadOnlyList<IReadOnlyList<IReadOnlyDictionary<string, PyObject>>> RunPipeline(IReadOnlyList<string> inputs)
        {
            return TransformerEnvironment.TransformersWrapper.CallPipelineWithList(PipelineObject, inputs);
        }
    }
}
