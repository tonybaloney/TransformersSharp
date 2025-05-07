using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TransformersSharp.Pipelines;

namespace TransformersSharp
{
    public static class TransformerEnvironment
    {
        private static readonly IPythonEnvironment? _env;
        private static readonly Lock _setupLock = new();

        static TransformerEnvironment()
        {
            lock (_setupLock)
            {
                IHostBuilder builder = Host.CreateDefaultBuilder()
                    .ConfigureServices(services =>
                    {
                        // Use Local AppData folder for Python installation
                        string appDataPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TransformersSharp");

                        // Create the directory if it doesn't exist
                        if (!Directory.Exists(appDataPath))
                            Directory.CreateDirectory(appDataPath);

                        // If user has an environment variable TRANSFORMERS_SHARP_VENV_PATH, use that instead
                        string? envPath = Environment.GetEnvironmentVariable("TRANSFORMERS_SHARP_VENV_PATH");
                        string venvPath;
                        if (envPath != null)
                            venvPath = envPath;
                        else
                            venvPath = Path.Join(appDataPath, "venv");

                        // Write requirements to appDataPath
                        string requirementsPath = Path.Join(appDataPath, "requirements.txt");

                        // TODO: Make this configurable
                        string[] requirements =
                        {
                            "transformers",
                            "sentence_transformers",
                            "torch",
                            "pillow",
                            "timm",
                            "einops"
                        };

                        File.WriteAllText(requirementsPath, string.Join('\n', requirements));

                        services
                                .WithPython()
                                .WithHome(appDataPath)
                                .WithVirtualEnvironment(venvPath)
                                .WithUvInstaller()
                                .FromRedistributable(); // Download Python 3.12 and store it locally
                    });

                var app = builder.Build();

                _env = app.Services.GetRequiredService<IPythonEnvironment>();
            }
        }

        private static IPythonEnvironment Env => _env ?? throw new InvalidOperationException("Python environment is not initialized..");

        internal static ITransformersWrapper TransformersWrapper => Env.TransformersWrapper();
        internal static ISentenceTransformersWrapper SentenceTransformersWrapper => Env.SentenceTransformersWrapper();

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
