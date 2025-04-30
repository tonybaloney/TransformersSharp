using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TransformersSharp.Pipelines;

namespace TransformersSharp
{
    public static class TransformerEnvironment
    {
        private static IPythonEnvironment? _env;
        private static IPythonEnvironment Env {
        
            get {
                if (_env != null)
                    return _env;

                IHostBuilder builder = Host.CreateDefaultBuilder()
                    .ConfigureServices(services =>
                    {
                        var home = Path.Join(Environment.CurrentDirectory, "python"); /* Path to your Python modules */
                        services
                            .WithPython()
                            .WithHome(home)
                            .WithVirtualEnvironment(Path.Join(home, "venv"))
                            .WithUvInstaller()
                            .FromRedistributable(); // Download Python 3.12 and store it locally
                    });

                var app = builder.Build();

                _env = app.Services.GetRequiredService<IPythonEnvironment>();
                return _env;
            }
        }

        internal static ITransformersWrapper TransformersWrapper => Env.TransformersWrapper();

        /// <summary>
        /// Login to Huggingface with a token.
        /// </summary>
        /// <param name="token"></param>
        public static void Login(string token)
        {
            var wrapperModule = Env.TransformersWrapper();
            wrapperModule.HuggingfaceLogin(token);
        }

        public static Pipeline Pipeline(string? task = null, string? model = null, string? tokenizer = null, TorchDtype? torchDtype = null)
        {
            var wrapperModule = Env.TransformersWrapper();
            string? torchDtypeStr = torchDtype?.ToString() ?? null;
            var pipeline = wrapperModule.Pipeline(task, model, tokenizer, torchDtypeStr);

            return new Pipeline(pipeline);
        }

        public static void Dispose()
        {
            _env?.Dispose();
        }
    }
}
