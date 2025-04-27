namespace TransformersSharp
{
    internal class TorchDtypeAttribute : Attribute
    {
        public string Dtype { get; }
        public TorchDtypeAttribute(string dtype)
        {
            Dtype = dtype;
        }
    }

    /// <summary>
    /// https://pytorch.org/docs/stable/tensor_attributes.html
    /// </summary>
    public enum TorchDtype
    {
        [TorchDtypeAttribute("float32")]
        Float32,
        [TorchDtypeAttribute("float64")]
        Float64,
        [TorchDtypeAttribute("complex64")]
        Complex64,
        [TorchDtypeAttribute("complex128")]
        Complex128,
        [TorchDtypeAttribute("float16")]
        Float16,
        [TorchDtypeAttribute("bfloat16")]
        BFloat16,
        [TorchDtypeAttribute("uint8")]
        UInt8,
        [TorchDtypeAttribute("int8")]
        Int8,
        [TorchDtypeAttribute("int16")]
        Int16,
        [TorchDtypeAttribute("int32")]
        Int32,
        [TorchDtypeAttribute("int64")]
        Int64,
        [TorchDtypeAttribute("bool")]
        Bool
    }
}
