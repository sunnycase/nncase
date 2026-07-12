// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTLinkableModule : ILinkableModule
{
    private static readonly Regex AbiViewStrideArgRegex = new(@"^(?<kind>input|output)(?<index>\d+)_(?<scalar>scalar_)?stride(?<axis>\d+)$", RegexOptions.Compiled | RegexOptions.CultureInvariant);

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
        var kernelParamsJson = BuildKernelParamsJson();
        var outputDirectory = ResolveOutputDirectory();
        WriteGeneratedModel(outputDirectory, metadataJson, kernelParamsJson);

        var linkedFunctions = _functions
            .Select(function => (ILinkedFunction)new PyNTTLinkedFunctionSignature(function))
            .ToArray();
        var metadataSection = new LinkedSection(ToStream(metadataJson), ".pyntt", 0, 8, (ulong)Encoding.UTF8.GetByteCount(metadataJson));
        return new PyNTTLinkedModule(_moduleKind, linkedFunctions, new[] { metadataSection });
    }

    private static string[] GetParameterNames(BaseFunction function)
    {
        return GetInputParameters(function).Select(parameter => parameter.Name).ToArray();
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

    private static IVar[] GetInputParameters(BaseFunction function)
    {
        if (function is PrimFunction primFunction)
        {
            var abi = primFunction.GetAbiView();
            return abi.Inputs.Concat<IVar>(abi.Workspaces).ToArray();
        }

        return GetParameters(function)
            .Where(parameter => IsTensorType(((BaseExpr)parameter).CheckedType))
            .ToArray();
    }

    private static IVar[] GetInputTensorParameters(BaseFunction function)
    {
        return GetInputParameters(function)
            .Where(parameter => IsTensorType(((BaseExpr)parameter).CheckedType))
            .ToArray();
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
        return GetInputTensorParameters(function)
            .Select(parameter => BuildTensorSpec(parameter.Name, ((BaseExpr)parameter).CheckedType, "input", device))
            .ToArray();
    }

    private static TensorSpecMetadata[] GetOutputTensorSpecs(BaseFunction function)
    {
        return PyNTTFunctionOutputs.GetOutputParameters(function)
            .Select((output, index) => BuildTensorSpec($"output{index}", output.CheckedType, "output", "like_input"))
            .ToArray();
    }

    private static TensorResultSpecMetadata[] GetResultTensorSpecs(BaseFunction function)
    {
        if (function is not PrimFunction primFunction)
        {
            throw new NotSupportedException($"PyNTT requires PrimFunction result ABI, got {function.GetType().Name} {function.Name}.");
        }

        var abi = primFunction.GetAbiView();
        var inputParameters = GetInputTensorParameters(function);
        var outputParameters = abi.OutputParameters.ToArray();
        return abi.Results.Select((result, index) =>
        {
            var inputIndex = Array.FindIndex(inputParameters, parameter => ReferenceEquals(parameter, result.Storage));
            var outputIndex = Array.FindIndex(outputParameters, parameter => ReferenceEquals(parameter, result.Storage));
            var (source, sourceIndex) = (inputIndex, outputIndex) switch
            {
                (>= 0, < 0) => ("input", inputIndex),
                (< 0, >= 0) => ("output", outputIndex),
                _ => throw new InvalidOperationException(
                    $"PyNTT PrimFunction {primFunction.Name} result {index} storage {result.Storage.Name} must bind exactly one input or output parameter."),
            };
            var offset = result.Value is TIR.Buffer buffer
                ? new PyNTTDimExpressionEmitter().Emit(buffer.MemSpan.Start).ToPythonLiteral()
                : 0L;
            return new TensorResultSpecMetadata(
                BuildTensorSpec($"result{index}", result.Type, "result", "like_input"),
                source,
                sourceIndex,
                offset);
        }).ToArray();
    }

    private static TensorSpecMetadata BuildTensorSpec(string name, IRType type, string role, string device)
    {
        var tensorType = GetTensorType(type, name);
        var shape = GetRankedShape(tensorType, name).Dimensions.ToArray()
            .Select(dimension => new PyNTTDimExpressionEmitter().Emit(dimension))
            .ToArray();
        var isObject = IsObjectDataType(tensorType.DType);
        return new TensorSpecMetadata(
            name,
            GetPyNTTDTypeName(tensorType.DType),
            shape.Select(dim => dim.ToPythonLiteral()).ToArray(),
            GetContiguousStrides(shape).Select(dim => dim.ToPythonLiteral()).ToArray(),
            role,
            isObject ? "cpu" : device,
            isObject ? "object" : "contiguous",
            isObject ? "object" : "global");
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

    private static bool IsTensorType(IRType type)
    {
        return type is TensorType or DistributedType;
    }

    private static bool IsObjectDataType(DataType dataType) => dataType is ReferenceType;

    private static RankedShape GetRankedShape(TensorType tensorType, string name)
    {
        if (tensorType.Shape is not RankedShape shape)
        {
            throw new NotSupportedException($"PyNTT requires ranked shape for {name}, got {tensorType.Shape}.");
        }

        return shape;
    }

    private static PyNTTDimExpression[] GetContiguousStrides(IReadOnlyList<PyNTTDimExpression> shape)
        => PyNTTTemplateUtility.ContiguousStrides(shape);

    private static string GetPyNTTDTypeName(DataType dataType)
    {
        if (IsObjectDataType(dataType))
        {
            return "object";
        }

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

    private static string GetPyNTTScalarDTypeName(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => GetPyNTTDTypeName(vectorType.ElemType),
            MaskVectorType maskVectorType => maskVectorType.ElementBits switch
            {
                1 => "bool",
                8 => "uint8",
                16 => "uint16",
                32 => "uint32",
                64 => "uint64",
                _ => throw new NotSupportedException($"PyNTT does not support mask vector element width {maskVectorType.ElementBits}."),
            },
            _ => GetPyNTTDTypeName(dataType),
        };
    }

    private static long GetBufferOffsetBytes(TIR.Buffer buffer)
    {
        var physicalOffset = buffer.MemSpan.Buffer.Location is MemoryLocation.Input or MemoryLocation.Output
            ? 0L
            : GetFixedDimension(buffer.MemSpan.Buffer.Start, $"{buffer.Name} physical buffer offset");
        return checked(physicalOffset +
            GetFixedDimension(buffer.MemSpan.Start, $"{buffer.Name} memspan offset"));
    }

    private static long GetFixedDimension(BaseExpr expr, string context)
    {
        return expr switch
        {
            Dimension dimension => GetFixedDimension(dimension, context),
            TensorConst tensorConst when tensorConst.Value.Shape.IsScalar => ReadScalarInt64(tensorConst.Value, $"PyNTT fixed {context}"),
            _ => expr.Evaluate().AsTensor().ToScalar<long>(),
        };
    }

    private static long GetFixedDimension(Dimension dimension, string context)
    {
        if (!dimension.IsFixed)
        {
            throw new NotSupportedException($"PyNTT requires fixed {context}, got {dimension}.");
        }

        return dimension.FixedValue;
    }

    private static long ReadScalarInt64(Tensor tensor, string context)
    {
        if (!tensor.Shape.IsScalar)
        {
            throw new NotSupportedException($"{context} expects a scalar tensor, got shape {tensor.Shape}.");
        }

        return tensor[Array.Empty<long>()] switch
        {
            sbyte value => value,
            byte value => value,
            short value => value,
            ushort value => value,
            int value => value,
            uint value => value,
            long value => value,
            ulong value => checked((long)value),
            bool value => value ? 1L : 0L,
            var value when TryReadPointerValue(value, out var pointerValue) => checked((long)pointerValue),
            var value => throw new NotSupportedException($"{context} expects an integral scalar tensor, got {value?.GetType().Name ?? "null"}."),
        };
    }

    private static bool TryReadPointerValue(object? value, out ulong pointerValue)
    {
        pointerValue = 0UL;
        if (value is null)
        {
            return false;
        }

        var type = value.GetType();
        if (!type.IsGenericType || type.GetGenericTypeDefinition() != typeof(Pointer<>))
        {
            return false;
        }

        var property = type.GetProperty(nameof(Pointer<byte>.Value));
        if (property?.GetValue(value) is not ulong rawValue)
        {
            return false;
        }

        pointerValue = rawValue;
        return true;
    }

    private static int GetShardCount(GeneratedKernelMetadata kernel)
        => kernel.Launch.Sharding.Hierarchy.Aggregate(1, (product, value) => checked(product * value));

    private static bool IsTirKernel(GeneratedKernelMetadata kernel)
        => kernel.Attrs.TryGetValue("tir", out var value) && value switch
        {
            bool boolValue => boolValue,
            JsonElement { ValueKind: JsonValueKind.True } => true,
            JsonElement { ValueKind: JsonValueKind.False } => false,
            _ => false,
        };

    private static long GetMaxBlockLocalDataBytes(IReadOnlyList<GeneratedKernelMetadata> kernels)
        => kernels
            .Select(GetBlockLocalDataBytes)
            .DefaultIfEmpty(0)
            .Max();

    private static long GetBlockLocalDataBytes(GeneratedKernelMetadata kernel)
        => GetInt64LaunchMeta(kernel, "block_local_data_pool_bytes");

    private static long GetInt64LaunchMeta(GeneratedKernelMetadata kernel, string key)
    {
        if (!kernel.Launch.Meta.TryGetValue(key, out var value))
        {
            return 0;
        }

        return value switch
        {
            int intValue => intValue,
            long longValue => longValue,
            JsonElement { ValueKind: JsonValueKind.Number } jsonElement => jsonElement.GetInt64(),
            _ => throw new NotSupportedException($"PyNTT {key} must be an integer, got {value}."),
        };
    }

    private int GetTargetShardCount()
    {
        var targetOptions = PyNTTTargetOptionsUtility.Get(_compileOptions);
        var hierarchy = targetOptions.Hierarchies.Length == 0 ? new[] { 1 } : targetOptions.Hierarchies[0];
        return hierarchy.Aggregate(1, (product, value) => checked(product * value));
    }

    private int GetBlockLocalDataScopeCount()
        => GetTargetShardCount();

    private static string PythonString(string value) => JsonSerializer.Serialize(value);

    private static string PythonBool(bool value) => value ? "True" : "False";

    private static string PythonTuple(IEnumerable<string> values)
    {
        var valueArray = values.ToArray();
        return $"({string.Join(", ", valueArray)}{(valueArray.Length == 1 ? "," : string.Empty)})";
    }

    private string BuildFunctionSpecPython(PyNTTLinkableFunction function)
    {
        var sourceFunction = function.SourceFunction;
        var parameters = PythonTuple(GetParameterNames(sourceFunction).Select(PythonString));
        var inputs = PythonTuple(GetInputTensorSpecs(function).Select(BuildTensorSpecPython));
        var outputs = PythonTuple(GetOutputTensorSpecs(sourceFunction).Select(BuildTensorSpecPython));
        var results = PythonTuple(GetResultTensorSpecs(sourceFunction).Select(BuildTensorResultSpecPython));
        var shapeBindings = PythonTuple(GetShapeBindings(sourceFunction).Select(BuildShapeBindingPython));

        return $"        FunctionSpec(name={PythonString(sourceFunction.Name)}, module_kind={PythonString(sourceFunction.ModuleKind)}, is_entry={PythonBool(sourceFunction.IsEntry)}, parameters={parameters}, inputs={inputs}, outputs={outputs}, results={results}, shape_bindings={shapeBindings}),";
    }

    private static string BuildTensorSpecPython(TensorSpecMetadata spec)
    {
        return $"TensorSpec(name={PythonString(spec.Name)}, dtype={PythonString(spec.DType)}, shape={PythonTuple(spec.Shape.Select(PythonValue))}, strides={PythonTuple(spec.Strides.Select(PythonValue))}, role={PythonString(spec.Role)}, device={PythonString(spec.Device)}, layout={PythonString(spec.Layout)}, memory={PythonString(spec.Memory)})";
    }

    private static string BuildTensorResultSpecPython(TensorResultSpecMetadata spec)
        => $"TensorResultSpec(tensor={BuildTensorSpecPython(spec.Tensor)}, source={PythonString(spec.Source)}, source_index={spec.SourceIndex.ToString(CultureInfo.InvariantCulture)}, offset_bytes={PythonValue(spec.OffsetBytes)})";

    private static string BuildShapeBindingPython(ShapeBindingMetadata binding)
    {
        var minValue = binding.MinValue.HasValue ? binding.MinValue.Value.ToString(CultureInfo.InvariantCulture) : "None";
        var maxValue = binding.MaxValue.HasValue ? binding.MaxValue.Value.ToString(CultureInfo.InvariantCulture) : "None";
        return $"ShapeBinding(name={PythonString(binding.Name)}, input_index={binding.InputIndex}, axis={binding.Axis}, min_value={minValue}, max_value={maxValue})";
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
            JsonElement jsonElement => jsonElement.ToString() ?? "None",
            int integer => integer.ToString(),
            long integer => integer.ToString(),
            float number => number.ToString("R", CultureInfo.InvariantCulture),
            double number => number.ToString("R", CultureInfo.InvariantCulture),
            _ => JsonSerializer.Serialize(value),
        };
    }

    private static MemoryStream ToStream(string content) => new(Encoding.UTF8.GetBytes(content));

    private static IRType ToSerializedType(IRType type)
    {
        return type switch
        {
            DistributedType distributedType => ToSerializedType(distributedType.TensorType),
            TupleType tupleType => new TupleType(tupleType.Fields.Select(ToSerializedType).ToArray()),
            _ => type,
        };
    }

    private sealed class PyNTTLinkedFunctionSignature : ILinkedFunction
    {
        public PyNTTLinkedFunctionSignature(PyNTTLinkableFunction function)
        {
            Id = function.Id;
            var callableType = (CallableType)function.SourceFunction.CheckedType;
            ParameterTypes = callableType.Parameters.Select(ToSerializedType).ToArray();
            ReturnType = ToSerializedType(callableType.ReturnType);
            TextBegin = 0;
            TextLength = 0;
            Sections = function.Sections;
        }

        public uint Id { get; }

        public IReadOnlyList<IRType> ParameterTypes { get; }

        public IRType ReturnType { get; }

        public ulong TextBegin { get; }

        public ulong TextLength { get; }

        public IReadOnlyList<ILinkedSection> Sections { get; }
    }

    private string ResolveOutputDirectory()
    {
        var targetOptions = PyNTTTargetOptionsUtility.Get(_compileOptions);
        if (targetOptions.OutputDirectory.Length > 0)
        {
            return Path.GetFullPath(targetOptions.OutputDirectory);
        }

        if (!string.IsNullOrWhiteSpace(_compileOptions.DumpDir))
        {
            return Path.GetFullPath(Path.Join(_compileOptions.DumpDir, "CodeGen", _moduleKind));
        }

        return Path.GetFullPath(Path.Join(Directory.GetCurrentDirectory(), "pyntt_model"));
    }

    private string BuildMetadataJson()
    {
        var targetOptions = PyNTTTargetOptionsUtility.Get(_compileOptions);
        var topKernelFunctions = GetRuntimeTopKernelFunctions();
        var metadata = new
        {
            pyntt_spec_version = 0,
            target_kind = _moduleKind,
            backend = targetOptions.Backend,
            strict = targetOptions.Strict,
            functions = _functions.Select(function => new
            {
                id = function.Id,
                name = function.SourceFunction.Name,
                module_kind = function.SourceFunction.ModuleKind,
                is_entry = function.SourceFunction.IsEntry,
                parameters = GetParameterNames(function.SourceFunction),
                inputs = GetInputTensorSpecs(function),
                outputs = GetOutputTensorSpecs(function.SourceFunction),
                results = GetResultTensorSpecs(function.SourceFunction),
                generated_kernels = topKernelFunctions.Contains(function)
                    ? function.GeneratedKernelSource.Kernels
                    : Array.Empty<GeneratedKernelMetadata>(),
            }).ToArray(),
        };
        return JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
    }

    private string BuildKernelParamsJson()
    {
        var targetOptions = PyNTTTargetOptionsUtility.Get(_compileOptions);
        var topKernelFunctions = GetRuntimeTopKernelFunctions();
        var manifest = new
        {
            pyntt_codegen_manifest_version = 1,
            target_kind = _moduleKind,
            backend = targetOptions.Backend,
            functions = _functions.Select(function => new
            {
                id = function.Id,
                name = function.SourceFunction.Name,
                module_kind = function.SourceFunction.ModuleKind,
                is_entry = function.SourceFunction.IsEntry,
                render_kernels = topKernelFunctions.Contains(function)
                    ? function.GeneratedKernelSource.RenderKernels
                    : Array.Empty<KernelRenderSpec>(),
            }).ToArray(),
        };
        return JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
    }

    private HashSet<PyNTTLinkableFunction> GetRuntimeTopKernelFunctions()
    {
        var result = new HashSet<PyNTTLinkableFunction>();
        var entry = _functions.FirstOrDefault(function => function.SourceFunction.IsEntry);
        if (entry is null)
        {
            return result;
        }

        CollectRuntimeTopKernelFunctions(entry, result, new HashSet<PyNTTLinkableFunction>());
        return result;
    }

    private void CollectRuntimeTopKernelFunctions(
        PyNTTLinkableFunction function,
        HashSet<PyNTTLinkableFunction> result,
        HashSet<PyNTTLinkableFunction> active)
    {
        if (!active.Add(function))
        {
            throw new NotSupportedException($"PyNTT runtime dispatch call graph contains a recursive call involving {function.SourceFunction.Name}.");
        }

        try
        {
            if (function.GeneratedKernelSource.Kernels.Count > 0)
            {
                result.Add(function);
                return;
            }

            var callees = new List<PrimFunction>();
            CollectRuntimeDispatchCallees(function.SourceFunction, callees);
            foreach (var callee in callees)
            {
                if (PyNTTPrimFunctionRoles.IsAutoTilingDeviceFunction(callee))
                {
                    continue;
                }

                CollectRuntimeTopKernelFunctions(FindLinkableFunction(callee), result, active);
            }
        }
        finally
        {
            active.Remove(function);
        }
    }

    private static void CollectRuntimeDispatchCallees(BaseFunction function, List<PrimFunction> callees)
    {
        switch (function)
        {
            case Function f:
                CollectRuntimeDispatchCallees(f.Body, callees);
                break;
            case Fusion f:
                CollectRuntimeDispatchCallees(f.Body, callees);
                break;
            case PrimFunction f:
                CollectRuntimeDispatchCallees(f.Body, callees);
                break;
        }
    }

    private static void CollectRuntimeDispatchCallees(BaseExpr expr, List<PrimFunction> callees)
    {
        if (expr is BaseFunction)
        {
            return;
        }

        if (expr is Call { Target: PrimFunction callee })
        {
            callees.Add(callee);
            return;
        }

        if (expr is Call { Target: BaseFunction calleeFunction })
        {
            throw new NotSupportedException($"PyNTT runtime dispatch expects direct PrimFunction call targets, got {calleeFunction.GetType().Name} {calleeFunction.Name}.");
        }

        foreach (var operand in expr.Operands)
        {
            CollectRuntimeDispatchCallees(operand, callees);
        }
    }

    private void WriteGeneratedModel(string outputDirectory, string metadataJson, string kernelParamsJson)
    {
        Directory.CreateDirectory(outputDirectory);

        var targetOptions = PyNTTTargetOptionsUtility.Get(_compileOptions);
        var backend = targetOptions.Backend;
        var moduleName = GetModuleName();
        var runtimeConfig = $"""
            BACKEND = {PythonString(backend)}
            STRICT = {PythonBool(targetOptions.Strict)}
            """;
        var requirements = """
            torch
            triton
            jinja2
            """;
        var readme = $"""
            # Generated PyNTT Model

            This directory was generated by the nncase PyNTT backend.

            Backend: `{backend}`

            The generated package uses the PyNTT runtime interpreter to separate
            model load state from per-run execution.
            """;

        File.WriteAllText(Path.Join(outputDirectory, "__init__.py"), "from .model import load_model\n", Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "metadata.json"), metadataJson + Environment.NewLine, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "kernel_params.json"), kernelParamsJson + Environment.NewLine, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "runtime_config.py"), runtimeConfig, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "requirements.txt"), requirements, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "README.md"), readme, Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "specs.py"), BuildSpecsPython(moduleName, backend), Encoding.UTF8);
        File.WriteAllText(Path.Join(outputDirectory, "rdata.py"), BuildRDataPython(outputDirectory), Encoding.UTF8);
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

    private sealed record RuntimeBinding(
        string Expression,
        string? AssignmentTarget = null,
        string? PoolStrideBytes = null,
        string? PoolStrideElements = null,
        string? WorkspacePoolStrideBytes = null,
        string[]? StrideElements = null,
        string[]? ScalarStrideElements = null);

    private sealed record FunctionCallArgumentLayout(
        BaseExpr[] InputArguments,
        BaseExpr[] OutputArguments,
        BaseExpr[] WorkspaceArguments);

    private sealed record ObjectBufferKey(
        string Scope,
        string Name,
        string ElemType,
        int Rank);

    private sealed record WorkspaceUsage(bool IsReferenced, long LocalBytes);

    private sealed record PreparedWorkspaceRequirements(
        bool UsesData,
        long DataLocalBytes,
        int MaxShardCount,
        bool UsesChipLocalData,
        long ChipLocalDataBytes,
        bool UsesBlockLocalData,
        long BlockLocalDataBytes);

    private sealed class RuntimeDispatchState
    {
        private int _nextTempIndex;

        public string NewTemp(string prefix)
        {
            var index = _nextTempIndex++;
            return $"__pyntt_{prefix}_{index.ToString(CultureInfo.InvariantCulture)}";
        }
    }

    private sealed class RuntimeDispatchContext
    {
        public RuntimeDispatchContext(
            RuntimeDispatchState state,
            Dictionary<string, RuntimeBinding> parameters,
            Dictionary<string, RuntimeBinding> outputs,
            Dictionary<ObjectBufferKey, RuntimeBinding> objectBuffers,
            string scope,
            string deviceExpression,
            string rootInputsExpression)
        {
            State = state;
            Parameters = parameters;
            Outputs = outputs;
            ObjectBuffers = objectBuffers;
            Scope = scope;
            DeviceExpression = deviceExpression;
            RootInputsExpression = rootInputsExpression;
        }

        public RuntimeDispatchState State { get; }

        public Dictionary<string, RuntimeBinding> Parameters { get; }

        public Dictionary<string, RuntimeBinding> Outputs { get; }

        public Dictionary<ObjectBufferKey, RuntimeBinding> ObjectBuffers { get; }

        public string Scope { get; }

        public string DeviceExpression { get; }

        public string RootInputsExpression { get; }

        public string? Data { get; set; }

        public string? DataPoolStrideBytes { get; set; }

        public string? RData { get; set; }

        public string? ChipLocalRData { get; set; }

        public string? ChipLocalData { get; set; }

        public string? BlockLocalRData { get; set; }

        public string? BlockLocalData { get; set; }

        public string? BlockLocalDataPoolStrideBytes { get; set; }

        public RuntimeDispatchContext CreateCallee(string scope, Dictionary<string, RuntimeBinding> parameters, Dictionary<string, RuntimeBinding> outputs)
            => new(State, parameters, outputs, ObjectBuffers, scope, DeviceExpression, RootInputsExpression);
    }

    private string BuildSpecsPython(string moduleName, string backend)
    {
        var functionSpecs = string.Join(
            Environment.NewLine,
            _functions.Select(BuildFunctionSpecPython));

        return $$"""
            from pyntt.ir import FunctionSpec, ModuleSpec, ShapeBinding, TensorResultSpec, TensorSpec


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
        if (string.IsNullOrWhiteSpace(launchStatements))
        {
            launchStatements = "        pass";
        }

        var topKernelFunctions = GetRuntimeTopKernelFunctions();
        var needsGridBarrier = topKernelFunctions
            .SelectMany(function => function.GeneratedKernelSource.Kernels)
            .Any(kernel => kernel.Attrs.ContainsKey("requires_grid_barrier"));
        var tritonRuntimeImport = needsGridBarrier
            ? "from pyntt.runtime.triton import ensure_triton_allocator, select_and_validate_triton_tuning_parameter"
            : "from pyntt.runtime.triton import select_and_validate_triton_tuning_parameter";

        return $$"""
            from pathlib import Path

            from pyntt.codegen.render import render_generated_kernels
            from pyntt.runtime.interpreter import PyNTTInterpreter
            from pyntt.runtime.tensor import materialize_kv_cache_blocks_per_shard, materialize_kv_cache_metadata, materialize_kv_cache_storage, materialize_kv_cache_tensor_field, resolve_execution_device, view_typed_buffer
            {{tritonRuntimeImport}}
            from .rdata import RDATA_BUNDLES
            from .specs import MODULE_SPEC


            _BASE_DIR = Path(__file__).resolve().parent


            class PyNTTGeneratedModel(PyNTTInterpreter):
                def __init__(self):
                    render_generated_kernels(_BASE_DIR, package=__package__)
                    super().__init__(MODULE_SPEC, RDATA_BUNDLES)

                def _run_entry(self, inputs, outputs, shape_env):
            {{launchStatements}}


            def load_model(device=None):
                return PyNTTGeneratedModel().load(device=device)
            """;
    }

    private string BuildModelLaunchStatements(PyNTTLinkableFunction function)
    {
        var context = CreateEntryDispatchContext(function.SourceFunction);
        var kernels = function.GeneratedKernelSource.Kernels;
        ValidateSingleKernelFunction(function);
        if (kernels.Count == 0)
        {
            ValidateSingleRuntimeLaunchPath(function);
            var dispatch = BuildFunctionDispatch(function, context, extraIndent: 0);
            if (!string.IsNullOrWhiteSpace(dispatch))
            {
                return dispatch;
            }

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
            kernels.Select(kernel => BuildModelKernelLaunchPython(function.SourceFunction.Name, kernel, parameterNames, outputNames, context, usePreparedWorkspace: false)));
    }

    private RuntimeDispatchContext CreateEntryDispatchContext(BaseFunction function)
    {
        var state = new RuntimeDispatchState();
        var parameters = GetInputTensorParameters(function)
            .Select((parameter, index) =>
            {
                var expression = $"inputs[{index.ToString(CultureInfo.InvariantCulture)}]";
                return (parameter.Name, Binding: new RuntimeBinding(
                    expression,
                    StrideElements: BuildTorchTensorStrideExpressions(expression, ((BaseExpr)parameter).CheckedType),
                    ScalarStrideElements: BuildTorchTensorScalarStrideExpressions(expression, ((BaseExpr)parameter).CheckedType)));
            })
            .ToDictionary(item => item.Name, item => item.Binding, StringComparer.Ordinal);
        var outputs = CreateRuntimeOutputBindings(function, index => new RuntimeBinding(
            $"outputs[{index.ToString(CultureInfo.InvariantCulture)}]",
            $"outputs[{index.ToString(CultureInfo.InvariantCulture)}]",
            StrideElements: BuildTorchTensorStrideExpressions($"outputs[{index.ToString(CultureInfo.InvariantCulture)}]", PyNTTFunctionOutputs.GetOutputParameterTypes(function)[index]),
            ScalarStrideElements: BuildTorchTensorScalarStrideExpressions($"outputs[{index.ToString(CultureInfo.InvariantCulture)}]", PyNTTFunctionOutputs.GetOutputParameterTypes(function)[index])));
        return new RuntimeDispatchContext(state, parameters, outputs, new Dictionary<ObjectBufferKey, RuntimeBinding>(), function.Name, "resolve_execution_device(inputs, outputs)", "inputs");
    }

    private static Dictionary<string, RuntimeBinding> CreateRuntimeOutputBindings(BaseFunction function, Func<int, RuntimeBinding> createBinding)
    {
        var outputParameters = PyNTTFunctionOutputs.GetOutputParameters(function);
        var bindings = new Dictionary<string, RuntimeBinding>(StringComparer.Ordinal);
        for (var i = 0; i < outputParameters.Length; i++)
        {
            var outputName = $"output{i.ToString(CultureInfo.InvariantCulture)}";
            var binding = createBinding(i);
            bindings[outputName] = binding;
            bindings[outputParameters[i].Name] = binding;
        }

        return bindings;
    }

    private static WorkspaceUsage GetWorkspaceUsage(BaseFunction function, MemoryLocation location)
    {
        var buffers = new List<TIR.Buffer>();
        switch (function)
        {
            case Function f:
                CollectWorkspaceBuffers(f.Body, location, buffers);
                break;
            case Fusion f:
                CollectWorkspaceBuffers(f.Body, location, buffers);
                break;
            case PrimFunction f:
                CollectWorkspaceBuffers(f.Body, location, buffers);
                break;
        }

        if (buffers.Count == 0)
        {
            return new WorkspaceUsage(false, 0);
        }

        var localBytes = buffers
            .Select(buffer => buffer.MemSpan.Buffer)
            .Distinct((IEqualityComparer<TIR.PhysicalBuffer>)ReferenceEqualityComparer.Instance)
            .Select(buffer => checked(
                GetFixedDimension(buffer.Start, $"{location} physical buffer offset") +
                GetFixedDimension(buffer.Size, $"{location} physical buffer size")))
            .DefaultIfEmpty(0)
            .Max();
        return new WorkspaceUsage(true, localBytes);
    }

    private static bool HasWorkspaceReference(BaseFunction function, MemoryLocation location)
    {
        var buffers = new List<TIR.Buffer>();
        switch (function)
        {
            case Function f:
                CollectWorkspaceBuffers(f.Body, location, buffers);
                break;
            case Fusion f:
                CollectWorkspaceBuffers(f.Body, location, buffers);
                break;
            case PrimFunction f:
                CollectWorkspaceBuffers(f.Body, location, buffers);
                break;
        }

        return buffers.Count != 0;
    }

    private PreparedWorkspaceRequirements GetPreparedWorkspaceRequirements(PyNTTLinkableFunction function)
        => GetPreparedWorkspaceRequirements(function, new HashSet<PyNTTLinkableFunction>());

    private PreparedWorkspaceRequirements GetPreparedWorkspaceRequirements(PyNTTLinkableFunction function, HashSet<PyNTTLinkableFunction> active)
    {
        if (!active.Add(function))
        {
            throw new NotSupportedException($"PyNTT dispatch call graph contains a recursive call involving {function.SourceFunction.Name}.");
        }

        try
        {
            var kernels = function.GeneratedKernelSource.Kernels.ToArray();
            var tirKernels = kernels.Where(IsTirKernel).ToArray();
            var maxShardCount = kernels
                .Select(GetShardCount)
                .DefaultIfEmpty(GetTargetShardCount())
                .Max();
            var kernelBlockLocalDataBytes = GetMaxBlockLocalDataBytes(tirKernels);
            var usesData = false;
            var dataLocalBytes = 0L;
            var nestedDataLocalBytes = 0L;
            var usesChipLocalData = false;
            var chipLocalDataBytes = 0L;
            var nestedChipLocalDataBytes = 0L;
            var usesBlockLocalData = false;
            var blockLocalDataBytes = 0L;
            var nestedBlockLocalDataBytes = 0L;

            if (function.SourceFunction is PrimFunction primFunction)
            {
                var dataUsage = GetWorkspaceUsage(function.SourceFunction, MemoryLocation.Data);
                dataLocalBytes = Math.Max((long)primFunction.SchedResult.DataUsage, dataUsage.LocalBytes);
                usesData = tirKernels.Length > 0 || primFunction.SchedResult.DataUsage > 0 || dataUsage.IsReferenced;

                var chipLocalDataUsage = HasWorkspaceReference(function.SourceFunction, MemoryLocation.ChipLocalData);
                chipLocalDataBytes = (long)primFunction.SchedResult.ChipLocalDataPoolSize;
                usesChipLocalData = primFunction.SchedResult.ChipLocalDataPoolSize > 0 || chipLocalDataUsage;

                var blockLocalDataUsage = GetWorkspaceUsage(function.SourceFunction, MemoryLocation.BlockLocalData);
                blockLocalDataBytes = new[]
                {
                    (long)primFunction.SchedResult.BlockLocalDataPoolSize,
                    blockLocalDataUsage.LocalBytes,
                    kernelBlockLocalDataBytes,
                }.Max();
                usesBlockLocalData = tirKernels.Length > 0 || primFunction.SchedResult.BlockLocalDataPoolSize > 0 || blockLocalDataUsage.IsReferenced || kernelBlockLocalDataBytes > 0;
            }

            var callees = new List<PrimFunction>();
            CollectDirectPrimFunctionCallees(function.SourceFunction, callees);
            foreach (var directCallee in callees)
            {
                if (PyNTTPrimFunctionRoles.IsAutoTilingDeviceFunction(directCallee))
                {
                    continue;
                }

                var calleeRequirements = GetPreparedWorkspaceRequirements(FindLinkableFunction(directCallee), active);
                usesData |= calleeRequirements.UsesData;
                nestedDataLocalBytes = Math.Max(nestedDataLocalBytes, calleeRequirements.DataLocalBytes);
                maxShardCount = Math.Max(maxShardCount, calleeRequirements.MaxShardCount);
                usesChipLocalData |= calleeRequirements.UsesChipLocalData;
                nestedChipLocalDataBytes = Math.Max(nestedChipLocalDataBytes, calleeRequirements.ChipLocalDataBytes);
                usesBlockLocalData |= calleeRequirements.UsesBlockLocalData;
                nestedBlockLocalDataBytes = Math.Max(nestedBlockLocalDataBytes, calleeRequirements.BlockLocalDataBytes);
            }

            return new PreparedWorkspaceRequirements(
                usesData,
                checked(dataLocalBytes + nestedDataLocalBytes),
                maxShardCount,
                usesChipLocalData,
                checked(chipLocalDataBytes + nestedChipLocalDataBytes),
                usesBlockLocalData,
                checked(blockLocalDataBytes + nestedBlockLocalDataBytes));
        }
        finally
        {
            active.Remove(function);
        }
    }

    private static void CollectDirectPrimFunctionCallees(BaseExpr expr, List<PrimFunction> callees)
    {
        if (expr is PrimFunction primFunction)
        {
            CollectDirectPrimFunctionCallees(primFunction.Body, callees);
            return;
        }

        if (expr is BaseFunction)
        {
            return;
        }

        if (expr is Call { Target: PrimFunction callee })
        {
            callees.Add(callee);
        }

        foreach (var operand in expr.Operands)
        {
            CollectDirectPrimFunctionCallees(operand, callees);
        }
    }

    private static void CollectWorkspaceBuffers(BaseExpr expr, MemoryLocation location, List<TIR.Buffer> buffers)
    {
        if (expr is BaseFunction)
        {
            return;
        }

        if (expr is TIR.Buffer buffer && buffer.MemSpan.Buffer.Location == location)
        {
            buffers.Add(buffer);
        }

        foreach (var operand in expr.Operands)
        {
            CollectWorkspaceBuffers(operand, location, buffers);
        }
    }

    private string BuildFunctionDispatch(PyNTTLinkableFunction function, RuntimeDispatchContext context, int extraIndent)
    {
        var pieces = new List<string>();
        pieces.Add(BuildPreparedWorkspaceSetup(function, context, extraIndent));

        var body = BuildDispatchLaunchStatements(function.SourceFunction, function, context, extraIndent);
        if (!string.IsNullOrWhiteSpace(body))
        {
            pieces.Add(body);
        }

        return string.Join(Environment.NewLine, pieces.Where(piece => !string.IsNullOrWhiteSpace(piece)));
    }

    private string BuildPreparedWorkspaceSetup(PyNTTLinkableFunction function, RuntimeDispatchContext context, int extraIndent)
    {
        var statements = new List<string>();
        var indent = new string(' ', 8 + extraIndent);
        var kernels = function.GeneratedKernelSource.Kernels.ToArray();
        var tirKernels = kernels.Where(IsTirKernel).ToArray();
        var primFunction = function.SourceFunction as PrimFunction;
        if (primFunction is not null)
        {
            var requirements = GetPreparedWorkspaceRequirements(function);
            var dataDType = PythonString("uint8");
            context.DataPoolStrideBytes ??= requirements.DataLocalBytes.ToString(CultureInfo.InvariantCulture);
            if (string.IsNullOrWhiteSpace(context.Data) && requirements.UsesData)
            {
                var dataName = context.State.NewTemp("data");
                context.Data = dataName;
                statements.Add(
                    $"{indent}{dataName} = self.allocate_workspace({context.RootInputsExpression}, {PythonString(function.SourceFunction.Name + ".data")}, {requirements.DataLocalBytes.ToString(CultureInfo.InvariantCulture)} * {requirements.MaxShardCount.ToString(CultureInfo.InvariantCulture)}, {dataDType})");
            }

            if (string.IsNullOrWhiteSpace(context.ChipLocalData) && requirements.UsesChipLocalData)
            {
                var chipLocalDataName = context.State.NewTemp("chip_local_data");
                context.ChipLocalData = chipLocalDataName;
                statements.Add(
                    $"{indent}{chipLocalDataName} = self.allocate_workspace({context.RootInputsExpression}, {PythonString(function.SourceFunction.Name + ".chip_local_data")}, {requirements.ChipLocalDataBytes.ToString(CultureInfo.InvariantCulture)}, {dataDType})");
            }

            context.BlockLocalDataPoolStrideBytes ??= requirements.BlockLocalDataBytes.ToString(CultureInfo.InvariantCulture);
            if (string.IsNullOrWhiteSpace(context.BlockLocalData) && requirements.UsesBlockLocalData)
            {
                var blockLocalDataName = context.State.NewTemp("block_local_data");
                context.BlockLocalData = blockLocalDataName;
                var blockLocalDataScopeCount = GetBlockLocalDataScopeCount();
                statements.Add(
                    $"{indent}{blockLocalDataName} = self.allocate_workspace({context.RootInputsExpression}, {PythonString(function.SourceFunction.Name + ".block_local_data")}, {requirements.BlockLocalDataBytes.ToString(CultureInfo.InvariantCulture)} * {blockLocalDataScopeCount.ToString(CultureInfo.InvariantCulture)}, {dataDType})");
            }
        }

        if (tirKernels.Length > 0
            && (string.IsNullOrWhiteSpace(context.RData)
                || string.IsNullOrWhiteSpace(context.ChipLocalRData)
                || string.IsNullOrWhiteSpace(context.BlockLocalRData)))
        {
            var rdataName = context.State.NewTemp("rdata");
            var chipLocalRDataName = context.State.NewTemp("chip_local_rdata");
            var blockLocalRDataName = context.State.NewTemp("block_local_rdata");
            context.RData = rdataName;
            context.ChipLocalRData = chipLocalRDataName;
            context.BlockLocalRData = blockLocalRDataName;
            statements.Add(
                $"{indent}{rdataName}, {chipLocalRDataName}, {blockLocalRDataName} = self.materialize_rdata_bundle({context.RootInputsExpression}, {PythonString(function.SourceFunction.Name)})");
        }

        return string.Join(Environment.NewLine, statements);
    }

    private string BuildDispatchLaunchStatements(BaseExpr expr, PyNTTLinkableFunction? currentFunction, RuntimeDispatchContext context, int extraIndent)
    {
        switch (expr)
        {
            case PrimFunction primFunction:
                return BuildDispatchLaunchStatements(primFunction.Body, currentFunction, context, extraIndent);
            case Sequential sequential:
                return BuildDispatchSequential(sequential, currentFunction, context, extraIndent);
            case IfThenElse ifThenElse:
                return BuildDispatchIfThenElse(ifThenElse, currentFunction, context, extraIndent);
            case Call { Target: PrimFunction callee } when PyNTTPrimFunctionRoles.IsAutoTilingDeviceFunction(callee):
                return string.Empty;
            case Call { Target: PrimFunction callee } call:
                return BuildFunctionCallDispatch(call, FindLinkableFunction(callee), context, extraIndent);
            case Call { Target: BaseFunction callee }:
                throw new NotSupportedException($"PyNTT dispatch expects direct PrimFunction call targets, got {callee.GetType().Name} {callee.Name}.");
            case Return ret:
                return string.Join(
                    Environment.NewLine,
                    ret.Values.ToArray()
                        .Select(value => BuildDispatchLaunchStatements(value, currentFunction, context, extraIndent))
                        .Where(statement => !string.IsNullOrWhiteSpace(statement)));
            case Nncase.TIR.Buffer:
            case Const:
            case IVar:
                return string.Empty;
            default:
                return string.Empty;
        }
    }

    private string BuildDispatchSequential(Sequential sequential, PyNTTLinkableFunction? currentFunction, RuntimeDispatchContext context, int extraIndent)
    {
        var pieces = new List<string>();
        foreach (var field in sequential.Fields)
        {
            if (field is Function or Fusion or FunctionWrapper or PrimFunctionWrapper)
            {
                throw new NotSupportedException($"PyNTT dispatch expects lowered PrimFunction bodies only, but found {field.GetType().Name}. TIR selection and RemoveFunctionWrapperPass must run before PyNTT codegen.");
            }

            if (field is BaseFunction)
            {
                continue;
            }

            if (field is IfThenElse)
            {
                pieces.Add(BuildDispatchLaunchStatements(field, currentFunction, context, extraIndent));
                continue;
            }

            if (field is Call { Target: PrimFunction callee } call)
            {
                if (!PyNTTPrimFunctionRoles.IsAutoTilingDeviceFunction(callee))
                {
                    pieces.Add(BuildFunctionCallDispatch(call, FindLinkableFunction(callee), context, extraIndent));
                }

                continue;
            }

            if (field is Call { Target: BaseFunction calleeFunction })
            {
                throw new NotSupportedException($"PyNTT dispatch expects direct PrimFunction call targets, got {calleeFunction.GetType().Name} {calleeFunction.Name}.");
            }

            pieces.Add(BuildRuntimeObjectStatements(new[] { field }, context, extraIndent));
        }

        return string.Join(Environment.NewLine, pieces.Where(piece => !string.IsNullOrWhiteSpace(piece)));
    }

    private string BuildFunctionCallDispatch(Call call, PyNTTLinkableFunction callee, RuntimeDispatchContext callerContext, int extraIndent)
    {
        var calleeContext = CreateCalleeDispatchContext(call, callee, callerContext);
        var workspaceSetup = BuildPreparedWorkspaceSetup(callee, calleeContext, extraIndent);
        var kernels = callee.GeneratedKernelSource.Kernels;
        ValidateSingleKernelFunction(callee);
        if (kernels.Count == 0)
        {
            var dispatch = BuildDispatchLaunchStatements(callee.SourceFunction, callee, calleeContext, extraIndent);
            return string.Join(Environment.NewLine, new[] { workspaceSetup, dispatch }.Where(piece => !string.IsNullOrWhiteSpace(piece)));
        }

        var parameterNames = GetParameterNames(callee.SourceFunction);
        var outputNames = GetOutputTensorSpecs(callee.SourceFunction).Select(output => output.Name).ToArray();
        var launches = string.Join(
            Environment.NewLine,
            kernels.Select(kernel => IndentPythonBlock(BuildModelKernelLaunchPython(callee.SourceFunction.Name, kernel, parameterNames, outputNames, calleeContext, usePreparedWorkspace: HasPreparedKernelWorkspace(calleeContext)), extraIndent)));
        return string.Join(Environment.NewLine, new[] { workspaceSetup, launches }.Where(piece => !string.IsNullOrWhiteSpace(piece)));
    }

    private static void ValidateSingleKernelFunction(PyNTTLinkableFunction function)
    {
        var count = function.GeneratedKernelSource.Kernels.Count;
        if (count > 1)
        {
            throw new NotSupportedException($"PyNTT function {function.SourceFunction.Name} generated {count} top kernels. PyNTT requires each runtime-dispatched function to lower to at most one top kernel; nested PrimFunctions must be inlined into that kernel.");
        }
    }

    private void ValidateSingleRuntimeLaunchPath(PyNTTLinkableFunction function)
    {
        var count = CountRuntimeLaunches(function.SourceFunction, new HashSet<PyNTTLinkableFunction>());
        if (count > 1)
        {
            throw new NotSupportedException($"PyNTT entry dispatch for {function.SourceFunction.Name} can launch {count} top kernels on one runtime path. PyNTT requires one selected top kernel per model invocation; fuse the work into one PrimFunction and inline nested PrimFunctions as device-level code.");
        }
    }

    private int CountRuntimeLaunches(BaseExpr expr, HashSet<PyNTTLinkableFunction> active)
    {
        switch (expr)
        {
            case Function function:
                return CountRuntimeLaunches(function.Body, active);
            case Fusion fusion:
                return CountRuntimeLaunches(fusion.Body, active);
            case PrimFunction primFunction when PyNTTPrimFunctionRoles.IsAutoTilingDeviceFunction(primFunction):
                return 0;
            case PrimFunction primFunction:
                return CountRuntimeLaunches(FindLinkableFunction(primFunction), active);
            case Call { Target: PrimFunction callee } when PyNTTPrimFunctionRoles.IsAutoTilingDeviceFunction(callee):
                return 0;
            case Call { Target: PrimFunction callee }:
                return CountRuntimeLaunches(FindLinkableFunction(callee), active);
            case Call { Target: BaseFunction callee }:
                throw new NotSupportedException($"PyNTT runtime dispatch expects direct PrimFunction call targets, got {callee.GetType().Name} {callee.Name}.");
            case IfThenElse ifThenElse:
                return Math.Max(
                    CountRuntimeLaunches(ifThenElse.Then, active),
                    CountRuntimeLaunches(ifThenElse.Else, active));
            case Sequential sequential:
                var sequentialCount = 0;
                foreach (var field in sequential.Fields)
                {
                    sequentialCount = checked(sequentialCount + CountRuntimeLaunches(field, active));
                }

                return sequentialCount;
            case Return ret:
                var returnCount = 0;
                foreach (var value in ret.Values)
                {
                    returnCount = checked(returnCount + CountRuntimeLaunches(value, active));
                }

                return returnCount;
            case BaseFunction:
                return 0;
            default:
                var operandCount = 0;
                foreach (var operand in expr.Operands)
                {
                    operandCount = checked(operandCount + CountRuntimeLaunches(operand, active));
                }

                return operandCount;
        }
    }

    private int CountRuntimeLaunches(PyNTTLinkableFunction function, HashSet<PyNTTLinkableFunction> active)
    {
        ValidateSingleKernelFunction(function);
        var kernels = function.GeneratedKernelSource.Kernels.Count;
        if (kernels > 0)
        {
            return kernels;
        }

        if (!active.Add(function))
        {
            throw new NotSupportedException($"PyNTT runtime dispatch call graph contains a recursive call involving {function.SourceFunction.Name}.");
        }

        try
        {
            return function.SourceFunction switch
            {
                Function f => CountRuntimeLaunches(f.Body, active),
                Fusion f => CountRuntimeLaunches(f.Body, active),
                PrimFunction f => CountRuntimeLaunches(f.Body, active),
                _ => 0,
            };
        }
        finally
        {
            active.Remove(function);
        }
    }

    private string BuildDispatchIfThenElse(IfThenElse expr, PyNTTLinkableFunction? currentFunction, RuntimeDispatchContext context, int extraIndent)
    {
        var indent = new string(' ', 8 + extraIndent);
        var condition = BuildRuntimePythonScalarExpression(expr.Condition);
        var thenBody = BuildDispatchLaunchStatements(expr.Then, currentFunction, context, extraIndent + 4);
        var elseBody = BuildDispatchLaunchStatements(expr.Else, currentFunction, context, extraIndent + 4);

        if (string.IsNullOrWhiteSpace(thenBody))
        {
            thenBody = new string(' ', 12 + extraIndent) + "pass";
        }

        if (string.IsNullOrWhiteSpace(elseBody))
        {
            return $"{indent}if {condition}:{Environment.NewLine}{thenBody}";
        }

        return $"{indent}if {condition}:{Environment.NewLine}{thenBody}{Environment.NewLine}{indent}else:{Environment.NewLine}{elseBody}";
    }

    private static bool HasPreparedKernelWorkspace(RuntimeDispatchContext context)
        => !string.IsNullOrWhiteSpace(context.Data)
            && !string.IsNullOrWhiteSpace(context.DataPoolStrideBytes)
            && !string.IsNullOrWhiteSpace(context.RData)
            && !string.IsNullOrWhiteSpace(context.ChipLocalRData)
            && !string.IsNullOrWhiteSpace(context.ChipLocalData)
            && !string.IsNullOrWhiteSpace(context.BlockLocalRData)
            && !string.IsNullOrWhiteSpace(context.BlockLocalData)
            && !string.IsNullOrWhiteSpace(context.BlockLocalDataPoolStrideBytes);

    private RuntimeDispatchContext CreateCalleeDispatchContext(Call call, PyNTTLinkableFunction calleeFunction, RuntimeDispatchContext callerContext)
    {
        var callee = calleeFunction.SourceFunction;
        var inputParameters = GetInputParameters(callee);
        var outputTypes = GetOutputTypes(callee);
        var layout = ResolveFunctionCallArgumentLayout(call, callee, inputParameters, outputTypes);
        var workspaceParameters = inputParameters.Skip(layout.InputArguments.Length).ToArray();

        if (workspaceParameters.Length != layout.WorkspaceArguments.Length
            || workspaceParameters.Any(parameter => !IsRuntimeWorkspaceParameter(parameter)))
        {
            throw new NotSupportedException($"PyNTT dispatch call to {callee.Name} expects caller-provided trailing runtime workspace inputs.");
        }

        if (layout.OutputArguments.Length != outputTypes.Length)
        {
            throw new NotSupportedException($"PyNTT dispatch call to {callee.Name} expects {outputTypes.Length} outputs, got {layout.OutputArguments.Length}.");
        }

        var parameters = new Dictionary<string, RuntimeBinding>(StringComparer.Ordinal);
        for (var i = 0; i < layout.InputArguments.Length; i++)
        {
            parameters[inputParameters[i].Name] = ResolveCallArgumentBinding(layout.InputArguments[i], callerContext, $"call {callee.Name} input {inputParameters[i].Name}");
        }

        var outputs = CreateRuntimeOutputBindings(
            callee,
            index => ResolveCallOutputBinding(layout.OutputArguments[index], callerContext, $"call {callee.Name} output {index}"));
        var calleeContext = callerContext.CreateCallee(callee.Name, parameters, outputs);
        for (var i = 0; i < workspaceParameters.Length; i++)
        {
            var parameter = workspaceParameters[i];
            var binding = ResolveCallArgumentBinding(layout.WorkspaceArguments[i], callerContext, $"call {callee.Name} workspace {parameter.Name}");
            parameters[parameter.Name] = binding;
            switch (parameter.Name)
            {
                case "data":
                    calleeContext.Data = binding.Expression;
                    calleeContext.DataPoolStrideBytes = RequireWorkspaceName(binding.WorkspacePoolStrideBytes, callee.Name, "data_pool_stride_bytes");
                    break;
                case "chip_local_data":
                    calleeContext.ChipLocalData = binding.Expression;
                    break;
                case "block_local_data":
                    calleeContext.BlockLocalData = binding.Expression;
                    calleeContext.BlockLocalDataPoolStrideBytes = RequireWorkspaceName(binding.WorkspacePoolStrideBytes, callee.Name, "block_local_data_pool_stride_bytes");
                    break;
                default:
                    throw new NotSupportedException($"PyNTT dispatch call to {callee.Name} does not support runtime workspace parameter {parameter.Name}.");
            }
        }

        return calleeContext;
    }

    private static IRType[] GetOutputTypes(BaseFunction function)
        => PyNTTFunctionOutputs.GetOutputParameterTypes(function);

    private static FunctionCallArgumentLayout ResolveFunctionCallArgumentLayout(
        Call call,
        BaseFunction callee,
        IReadOnlyList<IVar> inputParameters,
        IReadOnlyList<IRType> outputTypes)
    {
        var arguments = call.Arguments.ToArray();
        var trailingWorkspaceCount = CountTrailingRuntimeWorkspaceParameters(inputParameters);
        var regularInputCount = inputParameters.Count - trailingWorkspaceCount;
        if (trailingWorkspaceCount == 0)
        {
            var expectedCount = regularInputCount + outputTypes.Count;
            if (arguments.Length == expectedCount)
            {
                var outputArguments = arguments
                    .Skip(regularInputCount)
                    .Take(outputTypes.Count)
                    .ToArray();
                if (AreCompatibleOutputArguments(outputArguments, outputTypes))
                {
                    return new FunctionCallArgumentLayout(arguments.Take(regularInputCount).ToArray(), outputArguments, Array.Empty<BaseExpr>());
                }
            }

            throw new NotSupportedException(
                $"PyNTT dispatch call to {callee.Name} expects flattened outputs with regular-inputs + outputs, got {arguments.Length} arguments.");
        }

        var expectedWithWorkspaceCount = inputParameters.Count + outputTypes.Count;
        if (arguments.Length == expectedWithWorkspaceCount)
        {
            var outputArguments = arguments
                .Skip(regularInputCount)
                .Take(outputTypes.Count)
                .ToArray();
            var workspaceArguments = arguments
                .Skip(regularInputCount + outputTypes.Count)
                .ToArray();
            if (AreCompatibleOutputArguments(outputArguments, outputTypes)
                && AreCompatibleRuntimeWorkspaceArguments(
                    inputParameters.Skip(regularInputCount).ToArray(),
                    workspaceArguments))
            {
                return new FunctionCallArgumentLayout(arguments.Take(regularInputCount).ToArray(), outputArguments, workspaceArguments);
            }
        }

        throw new NotSupportedException(
            $"PyNTT dispatch call to {callee.Name} expects flattened outputs with " +
            $"regular-inputs + outputs + caller-provided runtime-workspace, got {arguments.Length} arguments.");
    }

    private static int CountTrailingRuntimeWorkspaceParameters(IReadOnlyList<IVar> inputParameters)
    {
        var count = 0;
        for (var i = inputParameters.Count - 1; i >= 0 && IsRuntimeWorkspaceParameter(inputParameters[i]); i--)
        {
            count++;
        }

        return count;
    }

    private static bool IsRuntimeWorkspaceParameter(IVar parameter)
        => parameter is BufferVar { Role: BufferVarRole.Workspace };

    private static bool AreCompatibleRuntimeWorkspaceArguments(IReadOnlyList<IVar> parameters, IReadOnlyList<BaseExpr> arguments)
    {
        if (parameters.Count != arguments.Count)
        {
            return false;
        }

        for (var i = 0; i < parameters.Count; i++)
        {
            if (!IsCompatibleRuntimeWorkspaceArgument(parameters[i], arguments[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsCompatibleRuntimeWorkspaceArgument(IVar parameter, BaseExpr argument)
    {
        if (parameter is not BufferVar { Role: BufferVarRole.Workspace } workspace
            || UnwrapInputBoxing(argument) is not TIR.Buffer buffer
            || buffer.ElemType != DataTypes.UInt8)
        {
            return false;
        }

        return buffer.MemSpan.Buffer.Location == workspace.Location;
    }

    private static bool AreCompatibleOutputArguments(IReadOnlyList<BaseExpr> arguments, IReadOnlyList<IRType> outputTypes)
    {
        if (arguments.Count != outputTypes.Count)
        {
            return false;
        }

        for (var i = 0; i < arguments.Count; i++)
        {
            if (!IsCompatibleOutputArgument(arguments[i], outputTypes[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsCompatibleOutputArgument(BaseExpr argument, IRType outputType)
    {
        if (!TryGetTensorType(outputType, out var expectedTensorType))
        {
            return false;
        }

        argument = UnwrapInputBoxing(argument);
        var actualTensorType = argument switch
        {
            TIR.Buffer buffer => new TensorType(buffer.ElemType, new RankedShape(buffer.Dimensions.ToArray())),
            Expr typed when TryGetTensorType(typed.CheckedType, out var tensorType) => tensorType,
            _ => null,
        };
        return actualTensorType is not null
            && actualTensorType.DType == expectedTensorType.DType
            && actualTensorType.Shape.Rank == expectedTensorType.Shape.Rank;
    }

    private static bool TryGetTensorType(IRType type, out TensorType tensorType)
    {
        switch (type)
        {
            case TensorType tt:
                tensorType = tt;
                return true;
            case DistributedType dt:
                tensorType = dt.TensorType;
                return true;
            default:
                tensorType = null!;
                return false;
        }
    }

    private RuntimeBinding ResolveCallArgumentBinding(BaseExpr expr, RuntimeDispatchContext context, string contextName)
    {
        expr = UnwrapInputBoxing(expr);
        if (expr is TIR.Buffer buffer)
        {
            var tensorPoolStrideBytes = GetBufferRuntimePoolStrideBytes(buffer, context);
            var workspacePoolStrideBytes = GetWorkspacePoolStrideBytes(buffer.MemSpan.Buffer.Location, context);
            return IsObjectDataType(buffer.ElemType)
                ? ResolveObjectBinding(buffer, context, contextName)
                : new RuntimeBinding(
                    BuildTensorBufferViewExpression(buffer, context, contextName),
                    PoolStrideBytes: tensorPoolStrideBytes,
                    PoolStrideElements: BuildPoolStrideElementsExpression(tensorPoolStrideBytes, buffer.ElemType),
                    WorkspacePoolStrideBytes: workspacePoolStrideBytes,
                    StrideElements: BuildBufferStrideElementExpressions(buffer, contextName),
                    ScalarStrideElements: BuildBufferScalarStrideElementExpressions(buffer, contextName));
        }

        if (expr is IVar parameter)
        {
            if (context.Parameters.TryGetValue(parameter.Name, out var parameterBinding))
            {
                return parameterBinding;
            }
        }

        if (expr is Dimension)
        {
            return new RuntimeBinding(BuildRuntimePythonScalarExpression(expr));
        }

        throw new NotSupportedException($"PyNTT dispatch cannot bind {contextName} from {expr.GetType().Name}.");
    }

    private static string? GetWorkspacePoolStrideBytes(MemoryLocation location, RuntimeDispatchContext context)
        => location switch
        {
            MemoryLocation.Data => context.DataPoolStrideBytes,
            MemoryLocation.ChipLocalData => "0",
            MemoryLocation.BlockLocalData => context.BlockLocalDataPoolStrideBytes,
            _ => null,
        };

    private static string? GetBufferRuntimePoolStrideBytes(TIR.Buffer buffer, RuntimeDispatchContext context)
    {
        if (buffer.MemSpan.Buffer.Location == MemoryLocation.Data && buffer.DistributedType is null)
        {
            return "0";
        }

        return GetWorkspacePoolStrideBytes(buffer.MemSpan.Buffer.Location, context);
    }

    private static string? BuildPoolStrideElementsExpression(string? poolStrideBytes, DataType elementType)
    {
        if (string.IsNullOrWhiteSpace(poolStrideBytes))
        {
            return null;
        }

        var scalarElementSizeBytes = GetScalarElementSizeBytes(elementType);
        if (poolStrideBytes == "0")
        {
            return "0";
        }

        if (long.TryParse(poolStrideBytes, NumberStyles.Integer, CultureInfo.InvariantCulture, out var fixedStrideBytes))
        {
            if (fixedStrideBytes % scalarElementSizeBytes != 0)
            {
                throw new NotSupportedException($"PyNTT workspace pool stride {fixedStrideBytes} is not aligned to scalar element size {scalarElementSizeBytes}.");
            }

            return (fixedStrideBytes / scalarElementSizeBytes).ToString(CultureInfo.InvariantCulture);
        }

        return scalarElementSizeBytes == 1
            ? poolStrideBytes
            : $"(({poolStrideBytes}) // {scalarElementSizeBytes.ToString(CultureInfo.InvariantCulture)})";
    }

    private static string[] BuildBufferStrideElementExpressions(TIR.Buffer buffer, string contextName)
        => buffer.Strides.ToArray()
            .Select((stride, axis) => BuildRuntimeDimensionExpression(stride, $"{contextName} buffer {buffer.Name} stride {axis.ToString(CultureInfo.InvariantCulture)}"))
            .ToArray();

    private static string[] BuildBufferScalarStrideElementExpressions(TIR.Buffer buffer, string contextName)
        => BuildScalarStrideElementExpressions(
            BuildBufferStrideElementExpressions(buffer, contextName),
            buffer.ElemType,
            buffer.Rank,
            $"{contextName} buffer {buffer.Name}");

    private static string[] BuildTorchTensorStrideExpressions(string expression, IRType type)
    {
        if (!TryGetTensorType(type, out var tensorType))
        {
            return Array.Empty<string>();
        }

        var shape = GetRankedShape(tensorType, expression);
        return Enumerable.Range(0, shape.Rank)
            .Select(axis => $"{expression}.stride({axis.ToString(CultureInfo.InvariantCulture)})")
            .ToArray();
    }

    private static string[] BuildTorchTensorScalarStrideExpressions(string expression, IRType type)
    {
        if (!TryGetTensorType(type, out var tensorType))
        {
            return Array.Empty<string>();
        }

        var shape = GetRankedShape(tensorType, expression);
        return BuildScalarStrideElementExpressions(
            BuildTorchTensorStrideExpressions(expression, type),
            tensorType.DType,
            shape.Rank,
            $"runtime tensor {expression}");
    }

    private static string[] BuildScalarStrideElementExpressions(string[] viewStrides, DataType dataType, int rank, string context)
    {
        if (viewStrides.Length != rank)
        {
            throw new NotSupportedException($"{context} has rank {rank} but {viewStrides.Length} runtime stride expressions.");
        }

        var lanes = GetVectorLanes(dataType);
        if (lanes.Length == 0)
        {
            return viewStrides;
        }

        if (lanes.Length > rank)
        {
            throw new NotSupportedException($"{context} has vector lanes [{string.Join(",", lanes)}] but only rank {rank}.");
        }

        var laneProduct = lanes.Aggregate(1, static (product, lane) => checked(product * lane));
        var vectorAxisStart = rank - lanes.Length;
        return viewStrides
            .Select((stride, axis) =>
            {
                if (axis < vectorAxisStart)
                {
                    return MultiplyRuntimeExpression(stride, laneProduct);
                }

                var laneAxis = axis - vectorAxisStart;
                var innerLaneStride = lanes.Skip(laneAxis + 1).Aggregate(1, static (product, lane) => checked(product * lane));
                return innerLaneStride.ToString(CultureInfo.InvariantCulture);
            })
            .ToArray();
    }

    private static string MultiplyRuntimeExpression(string expression, int factor)
    {
        if (factor == 1)
        {
            return expression;
        }

        if (long.TryParse(expression.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value))
        {
            return checked(value * factor).ToString(CultureInfo.InvariantCulture);
        }

        return $"(({expression}) * {factor.ToString(CultureInfo.InvariantCulture)})";
    }

    private static int[] GetVectorLanes(DataType dataType)
        => dataType switch
        {
            VectorType vectorType => vectorType.Lanes.ToArray(),
            MaskVectorType maskVectorType => new[] { maskVectorType.Lanes },
            _ => Array.Empty<int>(),
        };

    private static int GetScalarElementSizeBytes(DataType dataType)
    {
        return dataType is VectorType vectorType
            ? vectorType.ElemType.SizeInBytes
            : dataType.SizeInBytes;
    }

    private RuntimeBinding ResolveCallOutputBinding(BaseExpr expr, RuntimeDispatchContext context, string contextName)
    {
        expr = UnwrapInputBoxing(expr);
        if (expr is TIR.Buffer buffer)
        {
            var tensorPoolStrideBytes = GetBufferRuntimePoolStrideBytes(buffer, context);
            var workspacePoolStrideBytes = GetWorkspacePoolStrideBytes(buffer.MemSpan.Buffer.Location, context);
            return IsObjectDataType(buffer.ElemType)
                ? EnsureObjectBufferBinding(buffer, context)
                : new RuntimeBinding(
                    BuildTensorBufferViewExpression(buffer, context, contextName),
                    PoolStrideBytes: tensorPoolStrideBytes,
                    PoolStrideElements: BuildPoolStrideElementsExpression(tensorPoolStrideBytes, buffer.ElemType),
                    WorkspacePoolStrideBytes: workspacePoolStrideBytes,
                    StrideElements: BuildBufferStrideElementExpressions(buffer, contextName),
                    ScalarStrideElements: BuildBufferScalarStrideElementExpressions(buffer, contextName));
        }

        if (expr is IVar parameter && context.Outputs.TryGetValue(parameter.Name, out var parameterOutputBinding))
        {
            return parameterOutputBinding;
        }

        throw new NotSupportedException($"PyNTT dispatch cannot bind {contextName} from {expr.GetType().Name}.");
    }

    private string BuildRuntimeObjectStatements(IReadOnlyList<BaseExpr> fields, RuntimeDispatchContext context, int extraIndent)
    {
        var statements = new List<string>();
        foreach (var field in fields)
        {
            CollectRuntimeObjectStatements(field, context, statements, extraIndent);
        }

        return string.Join(Environment.NewLine, statements);
    }

    private void CollectRuntimeObjectStatements(BaseExpr expr, RuntimeDispatchContext context, List<string> statements, int extraIndent)
    {
        if (expr is BaseFunction)
        {
            return;
        }

        if (expr is Call call)
        {
            var args = call.Arguments.ToArray();
            if (call.Target is Nncase.TIR.NTT.TensorLoad && args.Length >= 2)
            {
                TryAppendObjectAssignment(UnwrapInputBoxing(args[0]), UnwrapInputBoxing(args[1]), context, statements, extraIndent, "TensorLoad");
                return;
            }

            if (call.Target is Nncase.TIR.NTT.TensorStore && args.Length >= 2)
            {
                TryAppendObjectAssignment(UnwrapInputBoxing(args[1]), UnwrapInputBoxing(args[0]), context, statements, extraIndent, "TensorStore");
                return;
            }

            if (call.Target is Nncase.TIR.Memcopy && args.Length >= 2)
            {
                TryAppendObjectAssignment(UnwrapInputBoxing(args[0]), UnwrapInputBoxing(args[1]), context, statements, extraIndent, "Memcopy");
                return;
            }
        }

        foreach (var operand in expr.Operands)
        {
            CollectRuntimeObjectStatements(operand, context, statements, extraIndent);
        }
    }

    private void TryAppendObjectAssignment(BaseExpr destination, BaseExpr source, RuntimeDispatchContext context, List<string> statements, int extraIndent, string opName)
    {
        if (!IsObjectExpression(destination) && !IsObjectExpression(source))
        {
            return;
        }

        if (!IsObjectExpression(destination) || !IsObjectExpression(source))
        {
            throw new NotSupportedException($"PyNTT dispatch {opName} object assignment expects both operands to be object tensors.");
        }

        var destinationBinding = ResolveAssignableObjectBinding(destination, context, $"PyNTT dispatch {opName} destination");
        var sourceBinding = ResolveObjectExpressionBinding(source, context, $"PyNTT dispatch {opName} source");
        statements.Add($"{new string(' ', 8 + extraIndent)}{destinationBinding.AssignmentTarget} = {sourceBinding.Expression}");
    }

    private RuntimeBinding ResolveAssignableObjectBinding(BaseExpr expr, RuntimeDispatchContext context, string contextName)
    {
        expr = UnwrapInputBoxing(expr);
        if (expr is TIR.Buffer buffer)
        {
            return EnsureObjectBufferBinding(buffer, context);
        }

        if (expr is IVar parameter && context.Outputs.TryGetValue(parameter.Name, out var outputBinding) && !string.IsNullOrWhiteSpace(outputBinding.AssignmentTarget))
        {
            return outputBinding;
        }

        throw new NotSupportedException($"{contextName} is not an assignable object binding.");
    }

    private RuntimeBinding ResolveObjectExpressionBinding(BaseExpr expr, RuntimeDispatchContext context, string contextName)
    {
        expr = UnwrapInputBoxing(expr);
        if (expr is TIR.Buffer buffer)
        {
            return ResolveObjectBinding(buffer, context, contextName);
        }

        if (expr is IVar parameter && context.Parameters.TryGetValue(parameter.Name, out var parameterBinding))
        {
            return parameterBinding;
        }

        throw new NotSupportedException($"{contextName} is not a known object binding.");
    }

    private static RuntimeBinding EnsureObjectBufferBinding(TIR.Buffer buffer, RuntimeDispatchContext context)
    {
        if (!IsObjectDataType(buffer.ElemType))
        {
            throw new NotSupportedException($"PyNTT expected object buffer, got {buffer.ElemType}.");
        }

        if (TryResolveAbiBufferBinding(buffer, context, out var abiBinding))
        {
            return abiBinding;
        }

        var key = CreateObjectBufferKey(buffer, context);
        if (!context.ObjectBuffers.TryGetValue(key, out var binding))
        {
            var variableName = context.State.NewTemp("object");
            binding = new RuntimeBinding(variableName, variableName);
            context.ObjectBuffers[key] = binding;
        }

        return binding;
    }

    private static RuntimeBinding ResolveObjectBinding(TIR.Buffer buffer, RuntimeDispatchContext context, string contextName)
    {
        if (!IsObjectDataType(buffer.ElemType))
        {
            throw new NotSupportedException($"{contextName} expected object buffer, got {buffer.ElemType}.");
        }

        if (TryResolveAbiBufferBinding(buffer, context, out var abiBinding))
        {
            return abiBinding;
        }

        var key = CreateObjectBufferKey(buffer, context);
        if (context.ObjectBuffers.TryGetValue(key, out var binding))
        {
            return binding;
        }

        throw new NotSupportedException($"{contextName} references object buffer {buffer.Name} in scope {context.Scope} before it is assigned.");
    }

    private static ObjectBufferKey CreateObjectBufferKey(TIR.Buffer buffer, RuntimeDispatchContext context)
    {
        if (string.IsNullOrWhiteSpace(buffer.Name))
        {
            throw new NotSupportedException($"PyNTT object buffer in scope {context.Scope} must have a stable name.");
        }

        return new ObjectBufferKey(
            context.Scope,
            buffer.Name,
            buffer.ElemType.ToString(),
            buffer.Rank);
    }

    private string BuildTensorBufferViewExpression(TIR.Buffer buffer, RuntimeDispatchContext context, string contextName)
    {
        var baseName = buffer.MemSpan.Buffer.Location switch
        {
            MemoryLocation.Input or MemoryLocation.Output when TryResolveAbiBufferBinding(buffer, context, out var abiBinding) => abiBinding.Expression,
            MemoryLocation.Data => RequireWorkspaceName(context.Data, contextName, "data"),
            MemoryLocation.ChipLocalData => RequireWorkspaceName(context.ChipLocalData, contextName, "chip_local_data"),
            MemoryLocation.BlockLocalData => RequireWorkspaceName(context.BlockLocalData, contextName, "block_local_data"),
            MemoryLocation.Rdata => RequireWorkspaceName(context.RData, contextName, "rdata"),
            MemoryLocation.ChipLocalRdata => RequireWorkspaceName(context.ChipLocalRData, contextName, "chip_local_rdata"),
            MemoryLocation.BlockLocalRdata => RequireWorkspaceName(context.BlockLocalRData, contextName, "block_local_rdata"),
            var location => throw new NotSupportedException($"{contextName} cannot pass buffer {buffer.Name} from memory location {location} as a PyNTT runtime tensor view."),
        };
        var offsetBytes = BuildRuntimeBufferOffsetExpression(buffer, contextName);
        var sizeBytes = BuildRuntimeBufferViewSizeExpression(buffer, context, contextName);
        var dtype = GetPyNTTScalarDTypeName(buffer.ElemType);
        return $"view_typed_buffer({baseName}, {offsetBytes}, {sizeBytes}, {PythonString(dtype)})";
    }

    private string BuildRuntimeBufferViewSizeExpression(TIR.Buffer buffer, RuntimeDispatchContext context, string contextName)
        => buffer.MemSpan.Buffer.Location switch
        {
            MemoryLocation.BlockLocalData => BuildBlockLocalDataBackingSizeExpression(context, contextName),
            _ => BuildRuntimeDimensionExpression(buffer.MemSpan.Size, $"{contextName} buffer {buffer.Name} size"),
        };

    private string BuildBlockLocalDataBackingSizeExpression(RuntimeDispatchContext context, string contextName)
    {
        var strideBytes = RequireWorkspaceName(context.BlockLocalDataPoolStrideBytes, contextName, "block_local_data_pool_stride_bytes");
        return MultiplyRuntimeExpression(strideBytes, GetBlockLocalDataScopeCount());
    }

    private static bool TryResolveAbiBufferBinding(TIR.Buffer buffer, RuntimeDispatchContext context, out RuntimeBinding binding)
    {
        binding = null!;
        if (buffer.MemSpan.Buffer.Location is not (MemoryLocation.Input or MemoryLocation.Output)
            || buffer.MemSpan.Buffer.Start is not IVar parameter)
        {
            return false;
        }

        var bindings = buffer.MemSpan.Buffer.Location == MemoryLocation.Input
            ? context.Parameters
            : context.Outputs;
        if (!bindings.TryGetValue(parameter.Name, out var resolved))
        {
            return false;
        }

        binding = resolved;
        return true;
    }

    private static string BuildRuntimeBufferOffsetExpression(TIR.Buffer buffer, string context)
    {
        var physicalOffset = buffer.MemSpan.Buffer.Location is MemoryLocation.Input or MemoryLocation.Output
            ? "0"
            : BuildRuntimeDimensionExpression(buffer.MemSpan.Buffer.Start, $"{context} physical buffer offset");
        if (ContainsShardCoord(buffer.MemSpan.Buffer.Start))
        {
            throw new NotSupportedException($"{context} physical buffer offset for {buffer.Name} depends on shard coordinates, which cannot be evaluated by PyNTT host dispatch.");
        }

        var spanOffset = ContainsShardCoord(buffer.MemSpan.Start)
            ? "0"
            : BuildRuntimeDimensionExpression(buffer.MemSpan.Start, $"{context} memspan offset");
        return AddRuntimeOffsetExpressions(physicalOffset, spanOffset);
    }

    private static bool ContainsShardCoord(BaseExpr expr)
    {
        expr = UnwrapInputBoxing(expr);
        if (expr is DimVar dimVar && dimVar.Name.StartsWith("__shard_coord_", StringComparison.Ordinal))
        {
            return true;
        }

        foreach (var operand in expr.Operands)
        {
            if (ContainsShardCoord(operand))
            {
                return true;
            }
        }

        return false;
    }

    private static string BuildRuntimeDimensionExpression(BaseExpr expr, string context)
    {
        expr = UnwrapInputBoxing(expr);
        return expr switch
        {
            None => "0",
            Dimension or TensorConst or Call => BuildRuntimePythonScalarExpression(expr),
            _ => GetFixedDimension(expr, context).ToString(CultureInfo.InvariantCulture),
        };
    }

    private static string AddRuntimeOffsetExpressions(string lhs, string rhs)
    {
        if (IsZeroExpression(lhs))
        {
            return rhs;
        }

        if (IsZeroExpression(rhs))
        {
            return lhs;
        }

        return $"(({lhs}) + ({rhs}))";
    }

    private static bool IsZeroExpression(string expression)
        => string.Equals(expression.Trim(), "0", StringComparison.Ordinal);

    private static bool IsObjectExpression(BaseExpr expr)
    {
        expr = UnwrapInputBoxing(expr);
        return expr switch
        {
            TIR.Buffer buffer => IsObjectDataType(buffer.ElemType),
            Expr typedExpr when typedExpr.CheckedType is TensorType tensorType => IsObjectDataType(tensorType.DType),
            Expr typedExpr when typedExpr.CheckedType is DistributedType distributedType => IsObjectDataType(distributedType.TensorType.DType),
            _ => false,
        };
    }

    private PyNTTLinkableFunction FindLinkableFunction(PrimFunction sourceFunction)
    {
        return _functions.FirstOrDefault(function =>
                ReferenceEquals(function.SourceFunction, sourceFunction))
            ?? throw new NotSupportedException($"Generated PyNTT dispatch references unknown function {sourceFunction.Name}.");
    }

    private static string IndentPythonBlock(string text, int extraIndent)
    {
        if (extraIndent == 0 || string.IsNullOrWhiteSpace(text))
        {
            return text;
        }

        var indent = new string(' ', extraIndent);
        return string.Join(
            Environment.NewLine,
            text.Split(Environment.NewLine).Select(line => string.IsNullOrWhiteSpace(line) ? line : indent + line));
    }

    private static string BuildRuntimePythonScalarExpression(BaseExpr expr)
    {
        expr = UnwrapInputBoxing(expr);
        if (expr is Dimension dimension)
        {
            var emitter = new PyNTTDimExpressionEmitter(formatRuntimeScalar: name => $"shape_env[{PythonString(name)}]");
            return emitter.Emit(dimension).PythonExpression;
        }

        if (expr is TensorConst tensorConst)
        {
            return FormatRuntimeScalarConst(tensorConst);
        }

        if (expr is IVar parameter && parameter is Expr { CheckedType: DimensionType })
        {
            return $"shape_env[{PythonString(SanitizePythonIdentifier(parameter.Name))}]";
        }

        if (expr is Call call)
        {
            var args = call.Arguments.ToArray();
            return call.Target switch
            {
                Nncase.IR.Math.Compare compare when args.Length >= 2 =>
                    $"({BuildRuntimePythonScalarExpression(args[0])} {GetRuntimeCompareOperator(compare.CompareOp)} {BuildRuntimePythonScalarExpression(args[1])})",
                Nncase.IR.Math.Binary binary when args.Length >= 2 =>
                    BuildRuntimeScalarBinaryExpression(binary.BinaryOp, BuildRuntimePythonScalarExpression(args[0]), BuildRuntimePythonScalarExpression(args[1])),
                _ => throw new NotSupportedException($"Unsupported PyNTT dispatch scalar expression call target: {call.Target.GetType().Name}."),
            };
        }

        throw new NotSupportedException($"Unsupported PyNTT dispatch scalar expression: {expr.GetType().Name}.");
    }

    private static BaseExpr UnwrapInputBoxing(BaseExpr expr)
    {
        while (expr is Call call && call.Target is Nncase.IR.Distributed.Boxing)
        {
            expr = call.Arguments[0];
        }

        return expr;
    }

    private static string GetRuntimeCompareOperator(CompareOp op)
    {
        return op switch
        {
            CompareOp.Equal => "==",
            CompareOp.NotEqual => "!=",
            CompareOp.LowerThan => "<",
            CompareOp.LowerOrEqual => "<=",
            CompareOp.GreaterThan => ">",
            CompareOp.GreaterOrEqual => ">=",
            _ => throw new NotSupportedException($"Unsupported PyNTT dispatch compare op: {op}."),
        };
    }

    private static string BuildRuntimeScalarBinaryExpression(BinaryOp op, string lhs, string rhs)
    {
        return op switch
        {
            BinaryOp.Add => $"({lhs} + {rhs})",
            BinaryOp.Sub => $"({lhs} - {rhs})",
            BinaryOp.Mul => $"({lhs} * {rhs})",
            BinaryOp.Div => $"({lhs} / {rhs})",
            BinaryOp.Mod => $"({lhs} % {rhs})",
            BinaryOp.Min => $"min({lhs}, {rhs})",
            BinaryOp.Max => $"max({lhs}, {rhs})",
            BinaryOp.Pow => $"(({lhs}) ** ({rhs}))",
            BinaryOp.BitwiseAnd => $"({lhs} & {rhs})",
            BinaryOp.BitwiseOr => $"({lhs} | {rhs})",
            BinaryOp.BitwiseXor => $"({lhs} ^ {rhs})",
            BinaryOp.LogicalAnd => $"({lhs} and {rhs})",
            BinaryOp.LogicalOr => $"({lhs} or {rhs})",
            BinaryOp.LogicalXor => $"(({lhs}) != ({rhs}))",
            BinaryOp.LeftShift => $"({lhs} << {rhs})",
            BinaryOp.RightShift => $"({lhs} >> {rhs})",
            BinaryOp.FloorDiv => $"(({lhs}) // ({rhs}))",
            BinaryOp.CeilDiv => $"((({lhs}) + ({rhs}) - 1) // ({rhs}))",
            _ => throw new NotSupportedException($"Unsupported PyNTT dispatch scalar binary op: {op}."),
        };
    }

    private static string FormatRuntimeScalarConst(TensorConst tensorConst)
    {
        if (!tensorConst.Value.Shape.IsScalar)
        {
            throw new NotSupportedException("PyNTT dispatch scalar expression only supports scalar TensorConst values.");
        }

        var value = tensorConst.Value[Array.Empty<long>()];
        return value switch
        {
            bool boolean => boolean ? "True" : "False",
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => value?.ToString() ?? "None",
        };
    }

    private string BuildModelKernelLaunchPython(string functionName, GeneratedKernelMetadata kernel, string[] parameterNames, string[] outputNames, RuntimeDispatchContext context, bool usePreparedWorkspace)
    {
        var kvCacheFieldInputs = GetKVCacheFieldInputs(kernel).ToDictionary(input => input.Name, StringComparer.Ordinal);
        var inputBindings = kernel.Inputs.Select(name => BuildKernelInputBinding(name, parameterNames, context, kvCacheFieldInputs)).ToArray();
        var inputArgs = inputBindings.Select(binding => binding.Expression).ToArray();
        var outputAliasMap = GetKernelOutputAliases(kernel);
        var outputBindings = kernel.Outputs.Select((name, index) =>
        {
            if (!outputAliasMap.TryGetValue(index, out var inputIndex))
            {
                return ResolveOutputBinding(context, name);
            }

            if (inputIndex < 0 || inputIndex >= inputBindings.Length)
            {
                throw new NotSupportedException($"Generated PyNTT kernel {kernel.Name} aliases input index {inputIndex}, but it only has {inputBindings.Length} inputs.");
            }

            return inputBindings[inputIndex];
        }).ToArray();
        var outputArgs = outputBindings.Select(binding => binding.Expression).ToArray();
        var tensorStrideArgs = inputBindings
            .Concat(outputBindings)
            .Select(binding => string.IsNullOrWhiteSpace(binding.PoolStrideElements) ? "0" : binding.PoolStrideElements!)
            .ToArray();
        var abiViewStrideArgs = GetAbiViewStrideArgs(kernel)
            .Select(argument => ResolveAbiViewStrideArgument(argument, inputBindings, outputBindings, kernel.Name))
            .ToArray();
        var outputAliases = BuildRuntimeOutputAliasStatements(kernel, context, parameterNames, inputArgs);
        if (kernel.Attrs.TryGetValue("pure_alias", out var pureAliasValue) && pureAliasValue is bool pureAlias && pureAlias)
        {
            return outputAliases;
        }

        var runtimeShapeArgNames = GetRuntimeShapeArgs(kernel);
        var numel = RuntimePythonExpression(kernel.Launch.Meta["numel_expr"], runtimeShapeArgNames);
        var blockSize = kernel.Launch.Tuning.Parameters["block_size"];
        var blockSizeCandidates = PythonTuple(blockSize.Candidates.Select(value => value.ToString()));
        var shardCount = kernel.Launch.Sharding.Hierarchy.Aggregate(1, (product, value) => product * value);
        var isTir = IsTirKernel(kernel);
        var grid = isTir
            ? $"({Math.Max(shardCount, 1)},)"
            : shardCount > 1
            ? $"({shardCount},)"
            : "((numel + block_size - 1) // block_size,)";
        var gridBeforeTuning = isTir ? $"        grid = {grid}" : string.Empty;
        var gridForCandidate = isTir
            ? "lambda _: grid"
            : shardCount > 1
            ? $"lambda _: ({shardCount},)"
            : "lambda candidate: ((numel + candidate - 1) // candidate,)";
        var gridAfterTuning = isTir ? string.Empty : $"        grid = {grid}";
        var workspaceSetup = string.Empty;
        var workspaceArgs = Array.Empty<string>();
        var workspaceStrideArgs = Array.Empty<string>();
        if (isTir)
        {
            if (usePreparedWorkspace)
            {
                workspaceArgs = new[]
                {
                    RequireWorkspaceName(context.Data, functionName, "data"),
                    RequireWorkspaceName(context.RData, functionName, "rdata"),
                    RequireWorkspaceName(context.ChipLocalRData, functionName, "chip_local_rdata"),
                    RequireWorkspaceName(context.ChipLocalData, functionName, "chip_local_data"),
                    RequireWorkspaceName(context.BlockLocalRData, functionName, "block_local_rdata"),
                    RequireWorkspaceName(context.BlockLocalData, functionName, "block_local_data"),
                };
                workspaceStrideArgs = new[]
                {
                    RequireWorkspaceName(context.DataPoolStrideBytes, functionName, "data_pool_stride_bytes"),
                    RequireWorkspaceName(context.BlockLocalDataPoolStrideBytes, functionName, "block_local_data_pool_stride_bytes"),
                };
            }
            else
            {
                var dataBytesPerProgram = PythonValue(kernel.Launch.Meta["data_pool_bytes"]);
                var blockLocalDataBytesPerScope = PythonValue(kernel.Launch.Meta["block_local_data_pool_bytes"]);
                var blockLocalDataScopeCount = PythonValue(kernel.Launch.Meta["block_local_data_scope_count"]);
                var chipLocalDataBytes = PythonValue(kernel.Launch.Meta["chip_local_data_pool_bytes"]);
                var dataDType = PythonString((string)kernel.Launch.Meta["data_dtype"]);
                workspaceSetup = $"""
                        data = self.allocate_workspace({context.RootInputsExpression}, {PythonString(kernel.Name + ".data")}, {dataBytesPerProgram} * grid[0], {dataDType})
                        chip_local_data = self.allocate_workspace({context.RootInputsExpression}, {PythonString(kernel.Name + ".chip_local_data")}, {chipLocalDataBytes}, {dataDType})
                        block_local_data = self.allocate_workspace({context.RootInputsExpression}, {PythonString(kernel.Name + ".block_local_data")}, {blockLocalDataBytesPerScope} * {blockLocalDataScopeCount}, {dataDType})
                        rdata, chip_local_rdata, block_local_rdata = self.materialize_rdata_bundle({context.RootInputsExpression}, {PythonString(functionName)})
                """;
                workspaceArgs = new[] { "data", "rdata", "chip_local_rdata", "chip_local_data", "block_local_rdata", "block_local_data" };
                workspaceStrideArgs = new[] { dataBytesPerProgram, blockLocalDataBytesPerScope };
            }
        }

        var runtimeShapeArgs = runtimeShapeArgNames.Select(arg => ResolveRuntimeScalarArgument(arg, context)).ToArray();
        var requiresGridBarrier = kernel.Attrs.ContainsKey("requires_grid_barrier");
        var gridBarrierArgs = requiresGridBarrier ? new[] { "PYNTT_GRID_MESH" } : Array.Empty<string>();
        var tensorArgs = string.Join(", ", inputArgs.Concat(outputArgs).Concat(tensorStrideArgs).Concat(abiViewStrideArgs).Concat(workspaceArgs).Concat(workspaceStrideArgs).Concat(runtimeShapeArgs).Concat(gridBarrierArgs));
        var tritonRuntimeSetup = requiresGridBarrier
            ? $"{Environment.NewLine}        ensure_triton_allocator({context.DeviceExpression})"
            : string.Empty;
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
        var importStatement = requiresGridBarrier
            ? $"from .generated_kernels import {kernel.Name}, PYNTT_GRID_MESH"
            : $"from .generated_kernels import {kernel.Name}";
        var launchStatement = $"        {kernel.Name}[grid]({tensorArgs}, numel, block_size{kwargs})";
        var kernelArgs = string.IsNullOrWhiteSpace(tensorArgs) ? "(numel,)" : $"({tensorArgs}, numel,)";
        var tuningSelectionStatement = $"        block_size = select_and_validate_triton_tuning_parameter({PythonString(kernel.Name)}, \"block_size\", {blockSizeCandidates}, source={PythonString(blockSize.Source)}, kernel={kernel.Name}, kernel_args={kernelArgs}, grid_for_candidate={gridForCandidate}, expected_num_warps={kernel.Launch.NumWarps ?? throw new InvalidOperationException($"Generated PyNTT kernel {kernel.Name} must declare a fixed num_warps.")}, worker_width={PythonValue(kernel.Attrs["worker_width"])}, register_capacity_bytes={PythonValue(kernel.Attrs["register_capacity_bytes"])}, shared_memory_capacity_bytes={PythonValue(kernel.Attrs["shared_memory_capacity_bytes"])}, forbid_spills={PythonValue(kernel.Attrs["forbid_spills"])}{kwargs})";
        return $"""
                    {importStatement}
                    numel = {numel}
            {gridBeforeTuning}
            {workspaceSetup}
            {tritonRuntimeSetup}
            {tuningSelectionStatement}
            {gridAfterTuning}
            {launchStatement}
            {outputAliases}
            """;
    }

    private static string[] GetAbiViewStrideArgs(GeneratedKernelMetadata kernel)
    {
        if (!kernel.Attrs.TryGetValue("abi_view_stride_args", out var value))
        {
            return Array.Empty<string>();
        }

        return value switch
        {
            string[] array => array,
            IEnumerable<string> enumerable => enumerable.ToArray(),
            JsonElement jsonElement when jsonElement.ValueKind == JsonValueKind.Array =>
                jsonElement.EnumerateArray().Select(item => item.GetString() ?? throw new NotSupportedException($"Generated PyNTT kernel {kernel.Name} has a non-string ABI view stride arg.")).ToArray(),
            _ => throw new NotSupportedException($"Generated PyNTT kernel {kernel.Name} has unsupported abi_view_stride_args metadata type {value.GetType().Name}."),
        };
    }

    private static string ResolveAbiViewStrideArgument(string argument, RuntimeBinding[] inputBindings, RuntimeBinding[] outputBindings, string kernelName)
    {
        var match = AbiViewStrideArgRegex.Match(argument);
        if (!match.Success)
        {
            throw new NotSupportedException($"Generated PyNTT kernel {kernelName} has invalid ABI view stride argument {argument}.");
        }

        var bindings = match.Groups["kind"].Value == "input" ? inputBindings : outputBindings;
        var index = int.Parse(match.Groups["index"].Value, CultureInfo.InvariantCulture);
        var axis = int.Parse(match.Groups["axis"].Value, CultureInfo.InvariantCulture);
        var useScalarStride = match.Groups["scalar"].Success;
        if (index < 0 || index >= bindings.Length)
        {
            throw new NotSupportedException($"Generated PyNTT kernel {kernelName} references {argument}, but only has {bindings.Length} {match.Groups["kind"].Value} bindings.");
        }

        var binding = bindings[index];
        var strideElements = useScalarStride ? binding.ScalarStrideElements : binding.StrideElements;
        if (strideElements is { } strides && axis < strides.Length)
        {
            return strides[axis];
        }

        return $"{binding.Expression}.stride({axis.ToString(CultureInfo.InvariantCulture)})";
    }

    private static string RequireWorkspaceName(string? name, string functionName, string workspaceName)
        => string.IsNullOrWhiteSpace(name)
            ? throw new NotSupportedException($"PyNTT dispatch for {functionName} requires prepared {workspaceName} workspace.")
            : name;

    private RuntimeBinding BuildKernelInputBinding(string name, string[] parameterNames, RuntimeDispatchContext context, IReadOnlyDictionary<string, PyNTTKVCacheFieldInputMetadata> kvCacheFieldInputs)
    {
        if (!kvCacheFieldInputs.TryGetValue(name, out var kvCacheField))
        {
            return ResolveParameterBinding(context, name, parameterNames);
        }

        var sourceExpression = ResolveParameterBinding(context, kvCacheField.SourceName, parameterNames).Expression;
        var expression = kvCacheField.Field switch
        {
            "metadata" => $"materialize_kv_cache_metadata({sourceExpression}, {context.DeviceExpression})",
            "slot_mapping" or "block_tables" or "context_lens" or "seq_lens" =>
                $"materialize_kv_cache_tensor_field({sourceExpression}, {PythonString(kvCacheField.Field)}, {context.DeviceExpression})",
            "kv_caches" when kvCacheField.Storage is { } storage =>
                $"materialize_kv_cache_storage({sourceExpression}, {context.DeviceExpression}, dtype={PythonString(storage.DType)}, topology_shape={PythonTuple(storage.TopologyShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}, key_tail_shape={PythonTuple(storage.KeyTailShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}, value_tail_shape={PythonTuple(storage.ValueTailShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}, key_section_elements={storage.KeySectionElements.ToString(CultureInfo.InvariantCulture)}, value_section_elements={storage.ValueSectionElements.ToString(CultureInfo.InvariantCulture)}, block_elements={storage.BlockElements.ToString(CultureInfo.InvariantCulture)}, block_size={storage.BlockSize.ToString(CultureInfo.InvariantCulture)})",
            "kv_caches_blocks" when kvCacheField.Storage is { } storage =>
                $"materialize_kv_cache_blocks_per_shard({sourceExpression}, topology_shape={PythonTuple(storage.TopologyShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}, block_size={storage.BlockSize.ToString(CultureInfo.InvariantCulture)})",
            "kv_caches" => throw new NotSupportedException($"Generated PyNTT kernel input {name} is missing KV-cache storage metadata."),
            "kv_caches_blocks" => throw new NotSupportedException($"Generated PyNTT kernel input {name} is missing KV-cache storage metadata."),
            _ => throw new NotSupportedException($"Generated PyNTT kernel input {name} references unsupported KV-cache field {kvCacheField.Field}."),
        };
        return new RuntimeBinding(expression);
    }

    private RuntimeBinding ResolveParameterBinding(RuntimeDispatchContext context, string name, IReadOnlyList<string> parameterNames)
    {
        if (context.Parameters.TryGetValue(name, out var binding))
        {
            return binding;
        }

        var index = FindIndex(parameterNames, name, "input");
        return new RuntimeBinding($"inputs[{index.ToString(CultureInfo.InvariantCulture)}]");
    }

    private static RuntimeBinding ResolveOutputBinding(RuntimeDispatchContext context, string name)
    {
        if (context.Outputs.TryGetValue(name, out var binding))
        {
            return binding;
        }

        throw new NotSupportedException($"Generated PyNTT kernel references unknown output: {name}.");
    }

    private static IReadOnlyDictionary<int, int> GetKernelOutputAliases(GeneratedKernelMetadata kernel)
    {
        if (kernel.Attrs.TryGetValue("output_aliases", out var value) && value is IReadOnlyDictionary<int, int> aliases)
        {
            return aliases;
        }

        return new Dictionary<int, int>();
    }

    private string BuildRuntimeOutputAliasStatements(GeneratedKernelMetadata kernel, RuntimeDispatchContext context, IReadOnlyList<string> parameterNames, IReadOnlyList<string> inputArgs)
    {
        if (!kernel.Attrs.TryGetValue("runtime_output_aliases", out var value))
        {
            return string.Empty;
        }

        if (value is IReadOnlyDictionary<string, string> aliasesByName)
        {
            return string.Join(
                Environment.NewLine,
                aliasesByName.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair =>
                {
                    var output = ResolveOutputBinding(context, pair.Key);
                    var source = ResolveParameterBinding(context, pair.Value, parameterNames);
                    return string.IsNullOrWhiteSpace(output.AssignmentTarget)
                        ? $"        {output.Expression}.copy_({source.Expression})"
                        : $"        {output.AssignmentTarget} = {source.Expression}";
                }));
        }

        if (value is IReadOnlyDictionary<string, int> aliasesByIndex)
        {
            return string.Join(
                Environment.NewLine,
                aliasesByIndex.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair =>
                {
                    if (pair.Value < 0 || pair.Value >= inputArgs.Count)
                    {
                        throw new NotSupportedException($"Generated PyNTT kernel {kernel.Name} aliases input index {pair.Value}, but it only has {inputArgs.Count} inputs.");
                    }

                    var output = ResolveOutputBinding(context, pair.Key);
                    return string.IsNullOrWhiteSpace(output.AssignmentTarget)
                        ? $"        {output.Expression}.copy_({inputArgs[pair.Value]})"
                        : $"        {output.AssignmentTarget} = {inputArgs[pair.Value]}";
                }));
        }

        throw new NotSupportedException($"Generated PyNTT kernel {kernel.Name} has unsupported runtime_output_aliases metadata type {value.GetType()}.");
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
        return """
            # Generated PyNTT kernels.
            #
            # model.py refreshes this file from kernel_params.json through
            # pyntt.codegen.render before launching Triton kernels.
            """;
    }

    private int GetGridBarrierMeshSize()
    {
        var meshSizes = GetRuntimeTopKernelFunctions()
            .SelectMany(function => function.GeneratedKernelSource.Kernels)
            .Where(kernel => kernel.Attrs.ContainsKey("requires_grid_barrier"))
            .Select(kernel => kernel.Launch.Sharding.Hierarchy.Aggregate(1, (acc, value) => checked(acc * value)))
            .Distinct()
            .ToArray();
        return meshSizes.Length switch
        {
            0 => 1,
            1 => meshSizes[0],
            _ => throw new NotSupportedException($"PyNTT generated kernels with grid barriers must use one grid mesh size, got {string.Join(", ", meshSizes)}."),
        };
    }

    private string BuildRDataPython(string outputDirectory)
    {
        var moduleRData = SerializeModuleRData(primFunction => primFunction.SchedResult.Rdatas, "module_rdata");
        var moduleChipLocalRData = SerializeModuleRData(primFunction => primFunction.SchedResult.ChipLocalRdatas, "module_chip_local_rdata");
        var moduleRDataPython = BuildRDataPayloadPython(outputDirectory, "module_rdata.bin", moduleRData.Payload);
        var moduleChipLocalRDataPython = BuildRDataPayloadPython(outputDirectory, "module_chip_local_rdata.bin", moduleChipLocalRData.Payload);
        var entries = string.Join(
            Environment.NewLine,
            _functions.Select(function => $"    {PythonString(function.SourceFunction.Name)}: {BuildRDataBundlePython(outputDirectory, function, moduleRData, moduleRDataPython, moduleChipLocalRData, moduleChipLocalRDataPython)},"));
        return $$"""
            # Serialized PyNTT rdata payloads.
            from pathlib import Path


            _BASE_DIR = Path(__file__).resolve().parent

            RDATA_BUNDLES = {
            {{entries}}
            }
            """;
    }

    private string BuildRDataBundlePython(
        string outputDirectory,
        PyNTTLinkableFunction function,
        SerializedRData moduleRData,
        string moduleRDataPython,
        SerializedRData moduleChipLocalRData,
        string moduleChipLocalRDataPython)
    {
        var bundle = function.RDataBundle;
        var usesRData = UsesTransitiveModuleRData(function, static primFunction => primFunction.SchedResult.Rdatas);
        var usesChipLocalRData = UsesTransitiveModuleRData(function, static primFunction => primFunction.SchedResult.ChipLocalRdatas);
        if (usesRData && moduleRData.Bytes == 0)
        {
            throw new InvalidDataException($"Function {function.SourceFunction.Name} uses PyNTT rdata, but module rdata is empty.");
        }

        if (usesChipLocalRData && moduleChipLocalRData.Bytes == 0)
        {
            throw new InvalidDataException($"Function {function.SourceFunction.Name} uses PyNTT chip local rdata, but module chip local rdata is empty.");
        }

        return "{" +
            string.Join(
                ", ",
                new[]
                {
                    $"{PythonString("rdata")}: {(usesRData ? moduleRDataPython : PythonString(string.Empty))}",
                    $"{PythonString("rdata_bytes")}: {(usesRData ? moduleRData.Bytes : 0).ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("chip_local_rdata")}: {(usesChipLocalRData ? moduleChipLocalRDataPython : PythonString(string.Empty))}",
                    $"{PythonString("chip_local_rdata_bytes")}: {(usesChipLocalRData ? moduleChipLocalRData.Bytes : 0).ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("block_local_rdata")}: {PythonTuple(bundle.BlockLocalRDatas.Select((payload, index) => BuildRDataPayloadPython(outputDirectory, function, "block_local_rdata", payload, index)))}",
                    $"{PythonString("block_local_rdata_bytes")}: {bundle.BlockLocalRDataBytes.ToString(CultureInfo.InvariantCulture)}",
                }) +
            "}";
    }

    private bool UsesTransitiveModuleRData(
        PyNTTLinkableFunction function,
        Func<PrimFunction, IReadOnlyDictionary<Const, ValueRange<ulong>>> selector)
        => UsesTransitiveModuleRData(function, selector, new HashSet<PyNTTLinkableFunction>());

    private bool UsesTransitiveModuleRData(
        PyNTTLinkableFunction function,
        Func<PrimFunction, IReadOnlyDictionary<Const, ValueRange<ulong>>> selector,
        HashSet<PyNTTLinkableFunction> active)
    {
        if (!active.Add(function))
        {
            throw new NotSupportedException($"PyNTT rdata call graph contains a recursive call involving {function.SourceFunction.Name}.");
        }

        try
        {
            if (function.SourceFunction is PrimFunction primFunction && selector(primFunction).Count > 0)
            {
                return true;
            }

            var callees = new List<PrimFunction>();
            CollectDirectPrimFunctionCallees(function.SourceFunction, callees);
            return callees.Any(callee =>
                PyNTTPrimFunctionRoles.IsAutoTilingDeviceFunction(callee)
                    ? UsesTransitiveModuleRData(callee, selector, new HashSet<PrimFunction>(ReferenceEqualityComparer.Instance))
                    : UsesTransitiveModuleRData(FindLinkableFunction(callee), selector, active));
        }
        finally
        {
            active.Remove(function);
        }
    }

    private bool UsesTransitiveModuleRData(
        PrimFunction function,
        Func<PrimFunction, IReadOnlyDictionary<Const, ValueRange<ulong>>> selector,
        HashSet<PrimFunction> active)
    {
        if (!active.Add(function))
        {
            throw new NotSupportedException($"PyNTT rdata device call graph contains a recursive call involving {function.Name}.");
        }

        try
        {
            if (selector(function).Count > 0)
            {
                return true;
            }

            var callees = new List<PrimFunction>();
            CollectDirectPrimFunctionCallees(function, callees);
            return callees.Any(callee => UsesTransitiveModuleRData(callee, selector, active));
        }
        finally
        {
            active.Remove(function);
        }
    }

    private string BuildRDataPayloadPython(string outputDirectory, PyNTTLinkableFunction function, string section, string payload, int? index)
    {
        var suffix = index.HasValue ? $"_{index.Value.ToString(CultureInfo.InvariantCulture)}" : string.Empty;
        var assetName = $"{SanitizePythonIdentifier(function.SourceFunction.Name)}_{section}{suffix}.bin";
        return BuildRDataPayloadPython(outputDirectory, assetName, payload);
    }

    private string BuildRDataPayloadPython(string outputDirectory, string assetName, string payload)
    {
        const string filePrefix = "file:";
        if (!payload.StartsWith(filePrefix, StringComparison.Ordinal))
        {
            if (string.IsNullOrEmpty(payload))
            {
                return PythonString(payload);
            }

            throw new InvalidDataException("PyNTT rdata payloads must be emitted as binary files.");
        }

        var sourcePath = payload[filePrefix.Length..];
        var assetDirectory = Path.Join(outputDirectory, "assets");
        Directory.CreateDirectory(assetDirectory);
        var destinationPath = Path.Join(assetDirectory, assetName);
        File.Copy(sourcePath, destinationPath, overwrite: true);
        return $"\"file:\" + str(_BASE_DIR / {PythonString(Path.Join("assets", assetName))})";
    }

    private SerializedRData SerializeModuleRData(
        Func<PrimFunction, IReadOnlyDictionary<Const, ValueRange<ulong>>> selector,
        string label)
    {
        var allocations = CollectModuleRData(selector);
        if (allocations.Count == 0)
        {
            return new(string.Empty, 0);
        }

        var poolSize = checked((long)allocations.Max(allocation => allocation.Range.Max));
        using var payload = CreatePayloadStream(label);
        var stream = payload.Stream;
        stream.SetLength(poolSize);
        foreach (var allocation in allocations)
        {
            var tensor = allocation.Const.Value;
            var size = allocation.Range.Max - allocation.Range.Min;
            if ((ulong)tensor.BytesBuffer.Length != size)
            {
                throw new InvalidDataException("The PyNTT module rdata buffer size does not match the scheduled range.");
            }

            stream.Position = checked((long)allocation.Range.Min);
            tensor.Serialize(stream);
        }

        return new(FinalizePayload(payload), poolSize);
    }

    private IReadOnlyList<RDataAllocation> CollectModuleRData(
        Func<PrimFunction, IReadOnlyDictionary<Const, ValueRange<ulong>>> selector)
    {
        var allocations = new List<RDataAllocation>();
        var seenRanges = new Dictionary<RDataRangeKey, TensorConst>();
        foreach (var function in _functions)
        {
            foreach (var primFunction in EnumerateTransitivePrimFunctions(function.SourceFunction))
            {
                foreach ((var @const, var range) in selector(primFunction))
                {
                    if (@const is not TensorConst tensorConst)
                    {
                        throw new NotSupportedException($"PyNTT module rdata only supports TensorConst, got {@const.GetType().Name}.");
                    }

                    var key = new RDataRangeKey(range.Min, range.Max);
                    if (seenRanges.TryGetValue(key, out var existing))
                    {
                        EnsureSameReadOnlyData(existing, tensorConst, key);
                        continue;
                    }

                    seenRanges.Add(key, tensorConst);
                    allocations.Add(new(tensorConst, range));
                }
            }
        }

        return allocations.OrderBy(allocation => allocation.Range.Min).ToArray();
    }

    private static IReadOnlyList<PrimFunction> EnumerateTransitivePrimFunctions(BaseExpr expr)
    {
        var functions = new List<PrimFunction>();
        CollectTransitivePrimFunctions(expr, functions, new HashSet<PrimFunction>(ReferenceEqualityComparer.Instance));
        return functions;
    }

    private static void CollectTransitivePrimFunctions(BaseExpr expr, List<PrimFunction> functions, HashSet<PrimFunction> seen)
    {
        if (expr is PrimFunction primFunction)
        {
            if (!seen.Add(primFunction))
            {
                return;
            }

            functions.Add(primFunction);
            CollectTransitivePrimFunctions(primFunction.Body, functions, seen);
            return;
        }

        if (expr is BaseFunction)
        {
            return;
        }

        if (expr is Call { Target: PrimFunction callee })
        {
            CollectTransitivePrimFunctions(callee, functions, seen);
        }

        foreach (var operand in expr.Operands)
        {
            CollectTransitivePrimFunctions(operand, functions, seen);
        }
    }

    private void EnsureSameReadOnlyData(TensorConst lhs, TensorConst rhs, RDataRangeKey range)
    {
        var lhsTensor = lhs.Value;
        var rhsTensor = rhs.Value;
        if (lhsTensor.ElementType != rhsTensor.ElementType ||
            !lhsTensor.Dimensions.ToArray().SequenceEqual(rhsTensor.Dimensions.ToArray()) ||
            !lhsTensor.Strides.ToArray().SequenceEqual(rhsTensor.Strides.ToArray()) ||
            lhsTensor.BytesBuffer.Length != rhsTensor.BytesBuffer.Length ||
            !lhsTensor.BytesBuffer.SequenceEqual(rhsTensor.BytesBuffer))
        {
            throw new InvalidDataException($"PyNTT module rdata range [{range.Min}, {range.Max}) is assigned to different constants.");
        }
    }

    private PayloadStream CreatePayloadStream(string label)
    {
        var directory = Path.Join(Path.GetTempPath(), "nncase_pyntt_rdata");
        Directory.CreateDirectory(directory);
        var path = Path.Join(directory, $"module_{label}_{Guid.NewGuid():N}.bin");
        return new(new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None), path);
    }

    private string FinalizePayload(PayloadStream payload)
    {
        payload.Stream.Flush();
        if (!string.IsNullOrEmpty(payload.Path))
        {
            return $"file:{payload.Path}";
        }

        throw new InvalidOperationException("PyNTT rdata payloads must be backed by binary files.");
    }

    private ShapeBindingMetadata[] GetShapeBindings(BaseFunction function)
    {
        var parameterNames = GetParameterNames(function);
        var bindings = new List<ShapeBindingMetadata>();
        foreach (var (tensorVar, dimExprs) in _compileOptions.ShapeBucketOptions.VarMap)
        {
            var inputIndex = Array.IndexOf(parameterNames, tensorVar.Name);
            if (inputIndex < 0)
            {
                continue;
            }

            for (var axis = 0; axis < dimExprs.Length; axis++)
            {
                if (dimExprs[axis] is not DimVar dimVar)
                {
                    continue;
                }

                var name = SanitizePythonIdentifier(dimVar.Name);
                var range = dimVar.Metadata.Range;
                bindings.Add(new(
                    name,
                    inputIndex,
                    axis,
                    range.HasValue && !double.IsNegativeInfinity(range.Value.Min) ? checked((long)range.Value.Min) : null,
                    range.HasValue && !double.IsPositiveInfinity(range.Value.Max) ? checked((long)range.Value.Max) : null));
            }
        }

        return bindings
            .GroupBy(binding => (binding.Name, binding.InputIndex, binding.Axis))
            .Select(group => group.First())
            .OrderBy(binding => binding.Name, StringComparer.Ordinal)
            .ThenBy(binding => binding.InputIndex)
            .ThenBy(binding => binding.Axis)
            .ToArray();
    }

    private static string PythonExpression(object value)
        => value switch
        {
            string text => text,
            int integer => integer.ToString(CultureInfo.InvariantCulture),
            long integer => integer.ToString(CultureInfo.InvariantCulture),
            _ => throw new NotSupportedException($"PyNTT launch expression must be a string or integer, got {value.GetType().Name}."),
        };

    private static string RuntimePythonExpression(object value, IReadOnlyList<string> runtimeShapeArgs)
    {
        var expression = PythonExpression(value);
        foreach (var arg in runtimeShapeArgs.OrderByDescending(arg => arg.Length))
        {
            expression = Regex.Replace(
                expression,
                $@"(?<![A-Za-z0-9_]){Regex.Escape(arg)}(?![A-Za-z0-9_])",
                $"shape_env[{PythonString(arg)}]");
        }

        return expression;
    }

    private static string[] GetRuntimeShapeArgs(GeneratedKernelMetadata kernel)
        => kernel.Attrs.TryGetValue("runtime_shape_args", out var value) && value is string[] args
            ? args
            : Array.Empty<string>();

    private static string ResolveRuntimeScalarArgument(string name, RuntimeDispatchContext context)
        => context.Parameters.TryGetValue(name, out var binding)
            ? binding.Expression
            : $"shape_env[{PythonString(name)}]";

    private static string[] GetStringArrayAttr(GeneratedKernelMetadata kernel, string name)
        => kernel.Attrs.TryGetValue(name, out var value) && value is string[] args
            ? args
            : Array.Empty<string>();

    private static PyNTTKVCacheFieldInputMetadata[] GetKVCacheFieldInputs(GeneratedKernelMetadata kernel)
        => kernel.Attrs.TryGetValue("kv_cache_field_inputs", out var value) && value is PyNTTKVCacheFieldInputMetadata[] args
            ? args
            : Array.Empty<PyNTTKVCacheFieldInputMetadata>();

    private static string SanitizePythonIdentifier(string value)
    {
        var chars = value.Select(ch => char.IsAsciiLetterOrDigit(ch) || ch == '_' ? ch : '_').ToArray();
        if (chars.Length == 0 || char.IsDigit(chars[0]))
        {
            return "_" + new string(chars);
        }

        return new string(chars);
    }

    private sealed record TensorSpecMetadata(
        [property: JsonPropertyName("name")]
        string Name,
        [property: JsonPropertyName("dtype")]
        string DType,
        [property: JsonPropertyName("shape")]
        object[] Shape,
        [property: JsonPropertyName("strides")]
        object[] Strides,
        [property: JsonPropertyName("role")]
        string Role,
        [property: JsonPropertyName("device")]
        string Device,
        [property: JsonPropertyName("layout")]
        string Layout,
        [property: JsonPropertyName("memory")]
        string Memory);

    private sealed record TensorResultSpecMetadata(
        [property: JsonPropertyName("tensor")]
        TensorSpecMetadata Tensor,
        [property: JsonPropertyName("source")]
        string Source,
        [property: JsonPropertyName("source_index")]
        int SourceIndex,
        [property: JsonPropertyName("offset_bytes")]
        object OffsetBytes);

    private sealed record ShapeBindingMetadata(string Name, int InputIndex, int Axis, long? MinValue, long? MaxValue);

    private sealed record SerializedRData(string Payload, long Bytes);

    private sealed record RDataAllocation(TensorConst Const, ValueRange<ulong> Range);

    private sealed record RDataRangeKey(ulong Min, ulong Max);

    private sealed record PayloadStream(Stream Stream, string? Path) : IDisposable
    {
        public void Dispose()
        {
            Stream.Dispose();
        }
    }
}
