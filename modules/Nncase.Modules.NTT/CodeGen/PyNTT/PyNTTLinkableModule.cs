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
        var outputParameters = function is PrimFunction primFunction
            ? GetTensorStoreOutputParameters(primFunction)
            : new HashSet<IVar>(ReferenceEqualityComparer.Instance);
        return GetParameters(function)
            .Where(parameter => IsTensorType(((Expr)parameter).CheckedType) && !outputParameters.Contains(parameter))
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
        return GetInputParameters(function)
            .Select(parameter => BuildTensorSpec(parameter.Name, parameter.CheckedType, "input", device))
            .ToArray();
    }

    private static TensorSpecMetadata[] GetOutputTensorSpecs(BaseFunction function)
    {
        if (function.CheckedType is not CallableType callableType)
        {
            return function is PrimFunction primFunctionWithoutCallable
                ? GetTensorStoreDestinations(primFunctionWithoutCallable)
                    .Select((expr, index) => BuildTensorSpec($"output{index}", expr.CheckedType, "output", "like_input"))
                    .ToArray()
                : Array.Empty<TensorSpecMetadata>();
        }

        var outputs = FlattenReturnTypes(callableType.ReturnType)
            .Select((type, index) => BuildTensorSpec($"output{index}", type, "output", "like_input"))
            .ToArray();
        if (outputs.Length == 0 && function is PrimFunction primFunction)
        {
            return GetTensorStoreDestinations(primFunction)
                .Select((expr, index) => BuildTensorSpec($"output{index}", expr.CheckedType, "output", "like_input"))
                .ToArray();
        }

        return outputs;
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

    private static HashSet<IVar> GetTensorStoreOutputParameters(PrimFunction function)
    {
        var outputParameters = new HashSet<IVar>(ReferenceEqualityComparer.Instance);
        foreach (var destination in GetTensorStoreDestinations(function))
        {
            var unwrapped = UnwrapInputBoxing(destination);
            if (unwrapped is IVar parameter)
            {
                outputParameters.Add(parameter);
            }
        }

        return outputParameters;
    }

    private static BaseExpr[] GetTensorStoreDestinations(PrimFunction function)
    {
        var destinations = new List<BaseExpr>();
        CollectTensorStoreDestinations(function.Body, destinations);
        return destinations
            .Distinct((IEqualityComparer<BaseExpr>)ReferenceEqualityComparer.Instance)
            .ToArray();
    }

    private static void CollectTensorStoreDestinations(BaseExpr expr, List<BaseExpr> destinations)
    {
        if (expr is Call call && call.Target is Nncase.TIR.NTT.TensorStore && call.Arguments.ToArray().Length >= 2)
        {
            destinations.Add(UnwrapInputBoxing(call.Arguments[1]));
        }

        foreach (var operand in expr.Operands)
        {
            CollectTensorStoreDestinations(operand, destinations);
        }
    }

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
        var shapeBindings = PythonTuple(GetShapeBindings(sourceFunction).Select(BuildShapeBindingPython));

        return $"        FunctionSpec(name={PythonString(sourceFunction.Name)}, module_kind={PythonString(sourceFunction.ModuleKind)}, is_entry={PythonBool(sourceFunction.IsEntry)}, parameters={parameters}, inputs={inputs}, outputs={outputs}, shape_bindings={shapeBindings}),";
    }

    private static string BuildTensorSpecPython(TensorSpecMetadata spec)
    {
        return $"TensorSpec(name={PythonString(spec.Name)}, dtype={PythonString(spec.DType)}, shape={PythonTuple(spec.Shape.Select(PythonValue))}, strides={PythonTuple(spec.Strides.Select(PythonValue))}, role={PythonString(spec.Role)}, device={PythonString(spec.Device)}, layout={PythonString(spec.Layout)}, memory={PythonString(spec.Memory)})";
    }

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

    private string ResolveOutputDirectory()
    {
        var targetOptions = PyNTTTargetOptionsUtility.Normalize(_compileOptions);
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
        var targetOptions = PyNTTTargetOptionsUtility.Normalize(_compileOptions);
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
                generated_kernels = function.GeneratedKernelSource.Kernels,
            }).ToArray(),
        };
        return JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
    }

    private void WriteGeneratedModel(string outputDirectory, string metadataJson)
    {
        Directory.CreateDirectory(outputDirectory);

        var targetOptions = PyNTTTargetOptionsUtility.Normalize(_compileOptions);
        var backend = targetOptions.Backend;
        var moduleName = GetModuleName();
        var runtimeConfig = $"""
            BACKEND = {PythonString(backend)}
            STRICT = {PythonBool(targetOptions.Strict)}
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

    private string BuildSpecsPython(string moduleName, string backend)
    {
        var functionSpecs = string.Join(
            Environment.NewLine,
            _functions.Select(BuildFunctionSpecPython));

        return $$"""
            from pyntt.ir import FunctionSpec, ModuleSpec, ShapeBinding, TensorSpec


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
            from pyntt.runtime.tensor import allocate_outputs, materialize_kv_cache_blocks_per_shard, materialize_kv_cache_metadata, materialize_kv_cache_storage, materialize_kv_cache_tensor_field, resolve_shape_env, validate_inputs
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

                    shape_env = resolve_shape_env(entry, inputs)
                    validate_inputs(entry, inputs, shape_env)
                    outputs = list(allocate_outputs(entry, inputs, shape_env))
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
            var dispatch = BuildDispatchLaunchStatements(function.SourceFunction, extraIndent: 0);
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
            kernels.Select(kernel => BuildModelKernelLaunchPython(function.SourceFunction.Name, kernel, parameterNames, outputNames)));
    }

    private string BuildDispatchLaunchStatements(BaseExpr expr, int extraIndent)
    {
        switch (expr)
        {
            case PrimFunction primFunction:
                return BuildDispatchLaunchStatements(primFunction.Body, extraIndent);
            case Sequential sequential:
                return string.Join(
                    Environment.NewLine,
                    sequential.Fields.ToArray()
                        .Select(field => BuildDispatchLaunchStatements(field, extraIndent))
                        .Where(statement => !string.IsNullOrWhiteSpace(statement)));
            case IfThenElse ifThenElse:
                return BuildDispatchIfThenElse(ifThenElse, extraIndent);
            case Call { Target: PrimFunction callee }:
                return IndentPythonBlock(BuildModelLaunchStatements(FindLinkableFunction(callee)), extraIndent);
            case Return ret:
                return string.Join(
                    Environment.NewLine,
                    ret.Values.ToArray()
                        .Select(value => BuildDispatchLaunchStatements(value, extraIndent))
                        .Where(statement => !string.IsNullOrWhiteSpace(statement)));
            case Nncase.TIR.Buffer:
            case Const:
            case IVar:
                return string.Empty;
            default:
                return string.Empty;
        }
    }

    private string BuildDispatchIfThenElse(IfThenElse expr, int extraIndent)
    {
        var indent = new string(' ', 8 + extraIndent);
        var condition = BuildRuntimePythonScalarExpression(expr.Condition);
        var thenBody = BuildDispatchLaunchStatements(expr.Then, extraIndent + 4);
        var elseBody = BuildDispatchLaunchStatements(expr.Else, extraIndent + 4);

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

    private PyNTTLinkableFunction FindLinkableFunction(BaseFunction sourceFunction)
    {
        return _functions.FirstOrDefault(function => ReferenceEquals(function.SourceFunction, sourceFunction) || function.SourceFunction.Name == sourceFunction.Name)
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

    private string BuildModelKernelLaunchPython(string functionName, GeneratedKernelMetadata kernel, string[] parameterNames, string[] outputNames)
    {
        var outputAliases = BuildOutputAliasStatements(kernel);
        if (kernel.Attrs.TryGetValue("pure_alias", out var pureAliasValue) && pureAliasValue is bool pureAlias && pureAlias)
        {
            return outputAliases;
        }

        var outputArgs = kernel.Outputs.Select(name => $"outputs[{FindIndex(outputNames, name, "output")}]");
        var kvCacheFieldInputs = GetKVCacheFieldInputs(kernel).ToDictionary(input => input.Name, StringComparer.Ordinal);
        var inputArgs = kernel.Inputs.Select(name => BuildKernelInputExpression(name, parameterNames, kvCacheFieldInputs));
        var runtimeShapeArgNames = GetRuntimeShapeArgs(kernel);
        var numel = RuntimePythonExpression(kernel.Launch.Meta["numel_expr"], runtimeShapeArgNames);
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

        var runtimeShapeArgs = runtimeShapeArgNames.Select(arg => $"shape_env[{PythonString(arg)}]").ToArray();
        var tensorArgs = string.Join(", ", inputArgs.Concat(outputArgs).Concat(workspaceArgs).Concat(runtimeShapeArgs));
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
        var importStatement = $"from .generated_kernels import {kernel.Name}";
        var launchStatement = $"        {kernel.Name}[grid]({tensorArgs}, numel, block_size{kwargs})";
        return $"""
                    {importStatement}
                    numel = {numel}
                    block_size = select_tuning_parameter({PythonString(kernel.Name)}, "block_size", {blockSizeCandidates}, source={PythonString(blockSize.Source)})
                    grid = {grid}
            {workspaceSetup}
            {launchStatement}
            {outputAliases}
            """;
    }

    private string BuildKernelInputExpression(string name, string[] parameterNames, IReadOnlyDictionary<string, PyNTTKVCacheFieldInputMetadata> kvCacheFieldInputs)
    {
        if (!kvCacheFieldInputs.TryGetValue(name, out var kvCacheField))
        {
            return $"inputs[{FindIndex(parameterNames, name, "input")}]";
        }

        var sourceExpression = $"inputs[{FindIndex(parameterNames, kvCacheField.SourceName, "input")}]";
        return kvCacheField.Field switch
        {
            "metadata" => $"materialize_kv_cache_metadata({sourceExpression}, outputs[0].device)",
            "slot_mapping" or "block_tables" or "context_lens" or "seq_lens" =>
                $"materialize_kv_cache_tensor_field({sourceExpression}, {PythonString(kvCacheField.Field)}, outputs[0].device)",
            "kv_caches" when kvCacheField.Storage is { } storage =>
                $"materialize_kv_cache_storage({sourceExpression}, outputs[0].device, dtype={PythonString(storage.DType)}, topology_shape={PythonTuple(storage.TopologyShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}, tail_shape={PythonTuple(storage.TailShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}, block_size={storage.BlockSize.ToString(CultureInfo.InvariantCulture)})",
            "kv_caches_blocks" when kvCacheField.Storage is { } storage =>
                $"materialize_kv_cache_blocks_per_shard({sourceExpression}, topology_shape={PythonTuple(storage.TopologyShape.Select(value => value.ToString(CultureInfo.InvariantCulture)))}, block_size={storage.BlockSize.ToString(CultureInfo.InvariantCulture)})",
            "kv_caches" => throw new NotSupportedException($"Generated PyNTT kernel input {name} is missing KV-cache storage metadata."),
            "kv_caches_blocks" => throw new NotSupportedException($"Generated PyNTT kernel input {name} is missing KV-cache storage metadata."),
            _ => throw new NotSupportedException($"Generated PyNTT kernel input {name} references unsupported KV-cache field {kvCacheField.Field}."),
        };
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

        var imports = new List<string>
        {
            "import triton",
            "import triton.language as tl",
        };
        if (kernelSources.Any(source => source.Contains("tle.distributed_barrier()", StringComparison.Ordinal)))
        {
            imports.Add("import triton.experimental.tle.language as tle");
        }

        imports.Add("from triton.language.extra import libdevice");
        var importSource = string.Join(Environment.NewLine, imports);
        return $$"""
            {{importSource}}


            {{string.Join(Environment.NewLine + Environment.NewLine, kernelSources)}}
            """;
    }

    private string BuildRDataPython(string outputDirectory)
    {
        var entries = string.Join(
            Environment.NewLine,
            _functions.Select(function => $"    {PythonString(function.SourceFunction.Name)}: {BuildRDataBundlePython(outputDirectory, function)},"));
        return $$"""
            # Serialized PyNTT rdata payloads.
            from pathlib import Path


            _BASE_DIR = Path(__file__).resolve().parent

            RDATA_BUNDLES = {
            {{entries}}
            }
            """;
    }

    private string BuildRDataBundlePython(string outputDirectory, PyNTTLinkableFunction function)
    {
        var bundle = function.RDataBundle;
        return "{" +
            string.Join(
                ", ",
                new[]
                {
                    $"{PythonString("rdata")}: {BuildRDataPayloadPython(outputDirectory, function, "rdata", bundle.RData, null)}",
                    $"{PythonString("rdata_bytes")}: {bundle.RDataBytes.ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("thread_local_rdata")}: {PythonTuple(bundle.ThreadLocalRDatas.Select((payload, index) => BuildRDataPayloadPython(outputDirectory, function, "thread_local_rdata", payload, index)))}",
                    $"{PythonString("thread_local_rdata_bytes")}: {bundle.ThreadLocalRDataBytes.ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("warp_local_rdata")}: {PythonTuple(bundle.WarpLocalRDatas.Select((payload, index) => BuildRDataPayloadPython(outputDirectory, function, "warp_local_rdata", payload, index)))}",
                    $"{PythonString("warp_local_rdata_bytes")}: {bundle.WarpLocalRDataBytes.ToString(CultureInfo.InvariantCulture)}",
                    $"{PythonString("block_local_rdata")}: {PythonTuple(bundle.BlockLocalRDatas.Select((payload, index) => BuildRDataPayloadPython(outputDirectory, function, "block_local_rdata", payload, index)))}",
                    $"{PythonString("block_local_rdata_bytes")}: {bundle.BlockLocalRDataBytes.ToString(CultureInfo.InvariantCulture)}",
                }) +
            "}";
    }

    private string BuildRDataPayloadPython(string outputDirectory, PyNTTLinkableFunction function, string section, string payload, int? index)
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
        var suffix = index.HasValue ? $"_{index.Value.ToString(CultureInfo.InvariantCulture)}" : string.Empty;
        var assetName = $"{SanitizePythonIdentifier(function.SourceFunction.Name)}_{section}{suffix}.bin";
        var destinationPath = Path.Join(assetDirectory, assetName);
        File.Copy(sourcePath, destinationPath, overwrite: true);
        return $"\"file:\" + str(_BASE_DIR / {PythonString(Path.Join("assets", assetName))})";
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

    private sealed record ShapeBindingMetadata(string Name, int InputIndex, int Axis, long? MinValue, long? MaxValue);
}
