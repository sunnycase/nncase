// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Nncase.IR;
using Nncase.Targets;
using Nncase.TIR;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTLinkableModule : ILinkableModule
{
    private readonly string _moduleKind;
    private readonly IReadOnlyList<PyNTTLinkableFunction> _functions;
    private readonly CompileOptions _compileOptions;

    public PyNTTLinkableModule(string moduleKind, IReadOnlyList<PyNTTLinkableFunction> functions, CompileOptions compileOptions)
    {
        _moduleKind = moduleKind;
        _functions = functions;
        _compileOptions = compileOptions;
    }

    public IReadOnlyList<ILinkableFunction> PublicFunctions => _functions;

    public ILinkedModule Link(ILinkContext linkContext)
    {
        var metadataJson = BuildMetadataJson();
        var outputDirectory = ResolveOutputDirectory();
        WriteGeneratedModel(outputDirectory, metadataJson);

        var linkedFunctions = _functions.Select(function =>
            new LinkedFunction(function.Id, function.SourceFunction, 0, 0, function.Sections)).ToArray();
        var metadataSection = new LinkedSection(ToStream(metadataJson), ".pyntt", 0, 8, (ulong)Encoding.UTF8.GetByteCount(metadataJson));
        return new PyNTTLinkedModule(_moduleKind, linkedFunctions, new[] { metadataSection });
    }

    private static string[] GetParameterNames(BaseFunction function)
    {
        return GetParameters(function).Select(parameter => parameter.Name).ToArray();
    }

    private static IVar[] GetParameters(BaseFunction function)
    {
        return function switch
        {
            Function f => f.Parameters.ToArray(),
            Fusion f => f.Parameters.ToArray(),
            PrimFunction f => f.Parameters.ToArray(),
            _ => Array.Empty<IVar>(),
        };
    }

    private static TensorSpecMetadata[] GetInputTensorSpecs(BaseFunction function)
    {
        return GetInputTensorSpecs(function, generatedKernelCount: 0);
    }

    private static TensorSpecMetadata[] GetInputTensorSpecs(PyNTTLinkableFunction function)
    {
        return GetInputTensorSpecs(function.SourceFunction, function.GeneratedKernelSource.Kernels.Count);
    }

    private static TensorSpecMetadata[] GetInputTensorSpecs(BaseFunction function, int generatedKernelCount)
    {
        var device = generatedKernelCount > 0 ? "cuda" : "any";
        return GetParameters(function)
            .Select(parameter => BuildTensorSpec(parameter.Name, parameter.CheckedType, "input", device))
            .ToArray();
    }

    private static TensorSpecMetadata[] GetOutputTensorSpecs(BaseFunction function)
    {
        if (function.CheckedType is not CallableType callableType)
        {
            return Array.Empty<TensorSpecMetadata>();
        }

        return FlattenReturnTypes(callableType.ReturnType)
            .Select((type, index) => BuildTensorSpec($"output{index}", type, "output", "like_input"))
            .ToArray();
    }

    private static IEnumerable<IRType> FlattenReturnTypes(IRType type)
    {
        return type switch
        {
            TupleType tupleType when tupleType == TupleType.Void => Array.Empty<IRType>(),
            TupleType tupleType => tupleType.ToArray(),
            _ => new[] { type },
        };
    }

    private static TensorSpecMetadata BuildTensorSpec(string name, IRType type, string role, string device)
    {
        var tensorType = GetTensorType(type, name);
        var shape = GetStaticShape(tensorType, name);
        return new TensorSpecMetadata(
            name,
            GetPyNTTDTypeName(tensorType.DType),
            shape,
            GetContiguousStrides(shape),
            role,
            device,
            "contiguous",
            "global");
    }

    private static TensorType GetTensorType(IRType type, string name)
    {
        return type switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => throw new NotSupportedException($"PyNTT requires tensor type for {name}, got {type}."),
        };
    }

    private static long[] GetStaticShape(TensorType tensorType, string name)
    {
        if (tensorType.Shape is not RankedShape { IsFixed: true } shape)
        {
            throw new NotSupportedException($"PyNTT M2 requires static ranked shape for {name}, got {tensorType.Shape}.");
        }

        return shape.ToValueArray();
    }

    private static long[] GetContiguousStrides(IReadOnlyList<long> shape)
    {
        var strides = new long[shape.Count];
        var stride = 1L;
        for (var i = shape.Count - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    private static string GetPyNTTDTypeName(DataType dataType)
    {
        return dataType.GetDisplayName() switch
        {
            "bool" => "bool",
            "i8" => "int8",
            "u8" => "uint8",
            "i16" => "int16",
            "u16" => "uint16",
            "i32" => "int32",
            "u32" => "uint32",
            "i64" => "int64",
            "u64" => "uint64",
            "f16" => "float16",
            "bf16" => "bfloat16",
            "f32" => "float32",
            "f64" => "float64",
            var name => name,
        };
    }

    private static string PythonString(string value) => JsonSerializer.Serialize(value);

    private static string PythonBool(bool value) => value ? "True" : "False";

    private static string PythonTuple(IEnumerable<string> values)
    {
        var valueArray = values.ToArray();
        return $"({string.Join(", ", valueArray)}{(valueArray.Length == 1 ? "," : string.Empty)})";
    }

    private static string BuildFunctionSpecPython(PyNTTLinkableFunction function)
    {
        var sourceFunction = function.SourceFunction;
        var parameters = PythonTuple(GetParameterNames(sourceFunction).Select(PythonString));
        var inputs = PythonTuple(GetInputTensorSpecs(function).Select(BuildTensorSpecPython));
        var outputs = PythonTuple(GetOutputTensorSpecs(sourceFunction).Select(BuildTensorSpecPython));

        return $"        FunctionSpec(name={PythonString(sourceFunction.Name)}, module_kind={PythonString(sourceFunction.ModuleKind)}, is_entry={PythonBool(sourceFunction.IsEntry)}, parameters={parameters}, inputs={inputs}, outputs={outputs}),";
    }

    private static string BuildTensorSpecPython(TensorSpecMetadata spec)
    {
        return $"TensorSpec(name={PythonString(spec.Name)}, dtype={PythonString(spec.DType)}, shape={PythonTuple(spec.Shape.Select(value => value.ToString()))}, strides={PythonTuple(spec.Strides.Select(value => value.ToString()))}, role={PythonString(spec.Role)}, device={PythonString(spec.Device)}, layout={PythonString(spec.Layout)}, memory={PythonString(spec.Memory)})";
    }

    private static string PythonDict(IReadOnlyDictionary<string, object> values)
    {
        if (values.Count == 0)
        {
            return "{}";
        }

        return $"{{{string.Join(", ", values.Select(pair => $"{PythonString(pair.Key)}: {PythonValue(pair.Value)}"))}}}";
    }

    private static string PythonValue(object value)
    {
        return value switch
        {
            string text => PythonString(text),
            bool boolean => PythonBool(boolean),
            int integer => integer.ToString(),
            long integer => integer.ToString(),
            float number => number.ToString("R", CultureInfo.InvariantCulture),
            double number => number.ToString("R", CultureInfo.InvariantCulture),
            _ => JsonSerializer.Serialize(value),
        };
    }

    private static MemoryStream ToStream(string content) => new(Encoding.UTF8.GetBytes(content));

    private string ResolveOutputDirectory()
    {
        if (_compileOptions.TargetOptions is PyNTTTargetOptions { OutputDirectory.Length: > 0 } pynttOptions)
        {
            return Path.GetFullPath(pynttOptions.OutputDirectory);
        }

        if (!string.IsNullOrWhiteSpace(_compileOptions.DumpDir))
        {
            return Path.GetFullPath(Path.Join(_compileOptions.DumpDir, "CodeGen", _moduleKind));
        }

        return Path.GetFullPath(Path.Join(Directory.GetCurrentDirectory(), "pyntt_model"));
    }

    private string BuildMetadataJson()
    {
        var targetOptions = _compileOptions.TargetOptions as PyNTTTargetOptions;
        var metadata = new
        {
            pyntt_spec_version = 0,
            target_kind = _moduleKind,
            backend = targetOptions?.Backend ?? "triton",
            strict = targetOptions?.Strict ?? true,
            functions = _functions.Select(function => new
            {
                id = function.Id,
                name = function.SourceFunction.Name,
                module_kind = function.SourceFunction.ModuleKind,
                is_entry = function.SourceFunction.IsEntry,
                parameters = GetParameterNames(function.SourceFunction),
                inputs = GetInputTensorSpecs(function),
                outputs = GetOutputTensorSpecs(function.SourceFunction),
                generated_kernels = function.GeneratedKernelSource.Kernels,
            }).ToArray(),
        };
        return JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
    }

    private void WriteGeneratedModel(string outputDirectory, string metadataJson)
    {
        Directory.CreateDirectory(outputDirectory);

        var targetOptions = _compileOptions.TargetOptions as PyNTTTargetOptions;
        var backend = targetOptions?.Backend ?? "triton";
        var moduleName = GetModuleName();
        var runtimeConfig = $"""
            BACKEND = {PythonString(backend)}
            STRICT = {PythonBool(targetOptions?.Strict ?? true)}
            """;
        var requirements = """
            torch
            triton
            """;
        var readme = $"""
            # Generated PyNTT Model

            This directory was generated by the nncase PyNTT backend.

            Backend: `{backend}`

            M4 output validates `torch.Tensor` inputs, allocates outputs, and
            directly launches generated Triton top kernels.
            """;

        File.WriteAllText(Path.Join(outputDirectory, "__init__.py"), "from .model import load_model\n", Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "metadata.json"), metadataJson + Environment.NewLine, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "runtime_config.py"), runtimeConfig, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "requirements.txt"), requirements, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "README.md"), readme, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "specs.py"), BuildSpecsPython(moduleName, backend), Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "rdata.py"), BuildRDataPython(), Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "generated_kernels.py"), BuildGeneratedKernelsPython(), Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "model.py"), BuildModelPython(), Encoding.UTF8);
    }

    private string GetModuleName()
    {
        foreach (var function in _functions)
        {
            if (function.SourceFunction.IsEntry)
            {
                return function.SourceFunction.Name;
            }
        }

        return _functions.Count > 0 ? _functions[0].SourceFunction.Name : "model";
    }

    private string BuildSpecsPython(string moduleName, string backend)
    {
        var functionSpecs = string.Join(
            Environment.NewLine,
            _functions.Select(BuildFunctionSpecPython));

        return $$"""
            from pyntt.ir import FunctionSpec, ModuleSpec, TensorSpec


            MODULE_SPEC = ModuleSpec(
                name={{PythonString(moduleName)}},
                backend={{PythonString(backend)}},
                functions=(
            {{functionSpecs}}
                ),
            )
            """;
    }

    private string BuildModelPython()
    {
        var entry = _functions.FirstOrDefault(function => function.SourceFunction.IsEntry);
        var launchStatements = entry is null ? string.Empty : BuildModelLaunchStatements(entry);

        return $$"""
            from pyntt.runtime.tensor import allocate_outputs, validate_inputs
            from pyntt.runtime.tuning import select_tuning_parameter
            from pyntt.runtime.workspace import allocate_workspace, materialize_rdata, materialize_rdata_table
            from .rdata import RDATA_BUNDLES
            from .specs import MODULE_SPEC


            class PyNTTGeneratedModel:
                def __init__(self):
                    self.spec = MODULE_SPEC

                def __call__(self, *inputs):
                    entry = self.spec.entry
                    if entry is None:
                        raise RuntimeError(f"PyNTT module {self.spec.name} does not declare an entry function.")

                    validate_inputs(entry, inputs)
                    outputs = list(allocate_outputs(entry, inputs))
            {{launchStatements}}
                    if len(outputs) == 1:
                        return outputs[0]
                    return tuple(outputs)


            def load_model():
                return PyNTTGeneratedModel()
            """;
    }

    private string BuildModelLaunchStatements(PyNTTLinkableFunction function)
    {
        var kernels = function.GeneratedKernelSource.Kernels;
        if (kernels.Count == 0)
        {
            var outputs = GetOutputTensorSpecs(function.SourceFunction);
            if (outputs.Length == 0)
            {
                return string.Empty;
            }

            var message = $"PyNTT generated no kernels for function {function.SourceFunction.Name}.";
            return $"        raise RuntimeError({PythonString(message)})";
        }

        var parameterNames = GetParameterNames(function.SourceFunction);
        var outputNames = GetOutputTensorSpecs(function.SourceFunction).Select(output => output.Name).ToArray();
        return string.Join(
            Environment.NewLine,
            kernels.Select(kernel => BuildModelKernelLaunchPython(function.SourceFunction.Name, kernel, parameterNames, outputNames)));
    }

    private string BuildModelKernelLaunchPython(string functionName, GeneratedKernelMetadata kernel, string[] parameterNames, string[] outputNames)
    {
        var outputAliases = BuildOutputAliasStatements(kernel);
        if (kernel.Attrs.TryGetValue("pure_alias", out var pureAliasValue) && pureAliasValue is bool pureAlias && pureAlias)
        {
            return outputAliases;
        }

        var inputArgs = kernel.Inputs.Select(name => $"inputs[{FindIndex(parameterNames, name, "input")}]");
        var outputArgs = kernel.Outputs.Select(name => $"outputs[{FindIndex(outputNames, name, "output")}]");
        var numel = PythonValue(kernel.Launch.Meta["numel"]);
        var blockSize = kernel.Launch.Tuning.Parameters["block_size"];
        var blockSizeCandidates = PythonTuple(blockSize.Candidates.Select(value => value.ToString()));
        var shardCount = kernel.Launch.Sharding.Hierarchy.Aggregate(1, (product, value) => product * value);
        var isTir = kernel.Attrs.TryGetValue("tir", out var tirValue) && tirValue is bool tirBool && tirBool;
        var grid = isTir
            ? $"({Math.Max(shardCount, 1)},)"
            : shardCount > 1
            ? $"({shardCount},)"
            : "((numel + block_size - 1) // block_size,)";
        var workspaceSetup = string.Empty;
        var workspaceArgs = Array.Empty<string>();
        if (isTir)
        {
            var dataBytesPerProgram = PythonValue(kernel.Launch.Meta["data_pool_bytes"]);
            var dataDType = PythonString((string)kernel.Launch.Meta["data_dtype"]);
            workspaceSetup = $"""
                    rdata_bundle = RDATA_BUNDLES[{PythonString(functionName)}]
                    data = allocate_workspace(inputs, {dataBytesPerProgram} * grid[0], {dataDType})
                    rdata = materialize_rdata(inputs, rdata_bundle["rdata"], rdata_bundle["rdata_bytes"])
                    thread_local_rdata = materialize_rdata_table(inputs, rdata_bundle["thread_local_rdata"], rdata_bundle["thread_local_rdata_bytes"])
                    warp_local_rdata = materialize_rdata_table(inputs, rdata_bundle["warp_local_rdata"], rdata_bundle["warp_local_rdata_bytes"])
                    block_local_rdata = materialize_rdata_table(inputs, rdata_bundle["block_local_rdata"], rdata_bundle["block_local_rdata_bytes"])
            """;
            workspaceArgs = new[] { "data", "rdata", "thread_local_rdata", "warp_local_rdata", "block_local_rdata" };
        }

        var tensorArgs = string.Join(", ", inputArgs.Concat(outputArgs).Concat(workspaceArgs));
        var launchKwargs = new List<string>();
        if (kernel.Launch.NumWarps.HasValue)
        {
            launchKwargs.Add($"num_warps={kernel.Launch.NumWarps.Value}");
        }

        if (kernel.Launch.NumStages.HasValue)
        {
            launchKwargs.Add($"num_stages={kernel.Launch.NumStages.Value}");
        }

        var kwargs = launchKwargs.Count == 0 ? string.Empty : $", {string.Join(", ", launchKwargs)}";
        return $"""
                    from .generated_kernels import {kernel.Name}
                    numel = {numel}
                    block_size = select_tuning_parameter({PythonString(kernel.Name)}, "block_size", {blockSizeCandidates}, source={PythonString(blockSize.Source)})
                    grid = {grid}
            {workspaceSetup}
                    {kernel.Name}[grid]({tensorArgs}, numel, block_size{kwargs})
            {outputAliases}
            """;
    }

    private string BuildOutputAliasStatements(GeneratedKernelMetadata kernel)
    {
        if (!kernel.Attrs.TryGetValue("output_aliases", out var value) || value is not IReadOnlyDictionary<int, int> aliases || aliases.Count == 0)
        {
            return string.Empty;
        }

        return string.Join(
            Environment.NewLine,
            aliases.OrderBy(pair => pair.Key).Select(pair => $"        outputs[{pair.Key}] = inputs[{pair.Value}]"));
    }

    private int FindIndex(IReadOnlyList<string> names, string name, string role)
    {
        for (var i = 0; i < names.Count; i++)
        {
            if (names[i] == name)
            {
                return i;
            }
        }

        throw new NotSupportedException($"Generated PyNTT kernel references unknown {role}: {name}.");
    }

    private string BuildGeneratedKernelsPython()
    {
        var kernelSources = _functions
            .Select(function => function.GeneratedKernelSource.Source)
            .Where(source => !string.IsNullOrWhiteSpace(source))
            .ToArray();
        if (kernelSources.Length == 0)
        {
            return """
                # Generated PyNTT kernels.
                #
                # This model has no generated Triton top kernels.
                """;
        }

        return $$"""
            import triton
            import triton.language as tl
            from triton.language.extra import libdevice


            {{string.Join(Environment.NewLine + Environment.NewLine, kernelSources)}}
            """;
    }

    private string BuildRDataPython()
    {
        var entries = string.Join(
            Environment.NewLine,
            _functions.Select(function => $"    {PythonString(function.SourceFunction.Name)}: {BuildRDataBundlePython(function.RDataBundle)},"));
        return $$"""
            # Serialized PyNTT rdata payloads.

            RDATA_BUNDLES = {
            {{entries}}
            }
            """;
    }

    private string BuildRDataBundlePython(PyNTTRDataBundle bundle)
    {
        return "{" +
            string.Join(
                ", ",
                new[]
                {
                    $"{PythonString("rdata")}: {PythonString(bundle.RData)}",
                    $"{PythonString("rdata_bytes")}: {bundle.RDataBytes.ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("thread_local_rdata")}: {PythonTuple(bundle.ThreadLocalRDatas.Select(PythonString))}",
                    $"{PythonString("thread_local_rdata_bytes")}: {bundle.ThreadLocalRDataBytes.ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("warp_local_rdata")}: {PythonTuple(bundle.WarpLocalRDatas.Select(PythonString))}",
                    $"{PythonString("warp_local_rdata_bytes")}: {bundle.WarpLocalRDataBytes.ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("block_local_rdata")}: {PythonTuple(bundle.BlockLocalRDatas.Select(PythonString))}",
                    $"{PythonString("block_local_rdata_bytes")}: {bundle.BlockLocalRDataBytes.ToString(CultureInfo.InvariantCulture)}",
                }) +
            "}";
    }

    private sealed record TensorSpecMetadata(
        [property: JsonPropertyName("name")]
        string Name,
        [property: JsonPropertyName("dtype")]
        string DType,
        [property: JsonPropertyName("shape")]
        long[] Shape,
        [property: JsonPropertyName("strides")]
        long[] Strides,
        [property: JsonPropertyName("role")]
        string Role,
        [property: JsonPropertyName("device")]
        string Device,
        [property: JsonPropertyName("layout")]
        string Layout,
        [property: JsonPropertyName("memory")]
        string Memory);
}
