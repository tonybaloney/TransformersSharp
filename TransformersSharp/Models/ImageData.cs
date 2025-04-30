namespace TransformersSharp.Models;

public enum ImagePixelMode
{
    RGB,
    Greyscale,
}

public struct ImageData
{
    public required byte[] ImageBytes { get; set; }
    public required int Width { get; set; }
    public required int Height { get; set; }
    public required ImagePixelMode PixelMode { get; set; }
}
