// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTKernelSourceConvertVisitor : ExprFunctor<Unit, Unit>
{
    private const string PoolStrideElementsSuffix = "_pool_stride_elements";

    private static readonly long[] ElementwiseBlockSizeSearchSpace = { 128, 256, 512, 1024 };
    private static readonly Regex InputReferenceRegex = new(@"\binput(?<index>\d+)(?<suffix>_pool_stride_elements|_(?:scalar_)?stride\d+)?\b", RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly Regex AbiArgumentReferenceRegex = new(@"\b(?:input|output)\d+(?:_pool_stride_elements|_(?:scalar_)?stride\d+)?\b", RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly Regex AbiArgumentSuffixRegex = new(@"_(?:pool_stride_elements|(?:scalar_)?stride\d+)$", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly List<GeneratedKernelMetadata> _generatedKernels = new();
    private readonly List<KernelRenderSpec> _renderKernels = new();
    private readonly SharedHelperRegistry _sharedHelperRegistry = new();
    private readonly PyNTTTargetOptions _targetOptions;

    public PyNTTKernelSourceConvertVisitor(CompileOptions compileOptions)
    {
        _targetOptions = PyNTTTargetOptionsUtility.Get(compileOptions);
    }

    public PyNTTGeneratedKernelSource GetKernelSource()
    {
        return new(_generatedKernels.ToArray(), _renderKernels.ToArray(), string.Empty);
    }

    protected override Unit VisitFunction(Function expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction inputs, got Function {expr.Name}.");

    protected override Unit VisitFusion(Fusion expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction inputs, got Fusion {expr.Name}.");

    protected override Unit VisitPrimFunctionWrapper(PrimFunctionWrapper expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects direct PrimFunction inputs, got PrimFunctionWrapper {expr.Name}.");

    protected override Unit VisitFunctionWrapper(FunctionWrapper expr)
        => throw new NotSupportedException($"PyNTT kernel codegen expects direct PrimFunction inputs, got FunctionWrapper {expr.Name}.");

    protected override Unit VisitPrimFunction(PrimFunction expr)
    {
        var outputs = GetOutputInfos(expr);
        if (outputs.Length == 0)
        {
            throw new NotSupportedException($"PyNTT PrimFunction {expr.Name} does not have tensor outputs.");
        }

        if (ContainsFunctionCall(expr.Body))
        {
            VisitSegmentedPrimFunction(expr, outputs);
            return default;
        }

        AddPrimFunctionKernel(expr, expr.Body, outputs, dispatchSegmentIndex: null, allowPartialOutputs: false);
        return default;
    }

    private void VisitSegmentedPrimFunction(PrimFunction function, OutputInfo[] outputs)
    {
        if (function.Body is not Sequential sequential)
        {
            if (ContainsKernelWork(function.Body))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {function.Name} contains nested function calls in a non-sequential body. Split it before PyNTT codegen.");
            }

            return;
        }

        var segmentIndex = 0;
        var segment = new List<BaseExpr>();
        foreach (var field in sequential.Fields)
        {
            if (field is Function or Fusion or FunctionWrapper or PrimFunctionWrapper)
            {
                throw new NotSupportedException($"PyNTT PrimFunction {function.Name} body contains {field.GetType().Name}. TIR selection and RemoveFunctionWrapperPass must run before PyNTT codegen.");
            }

            if (field is BaseFunction)
            {
                continue;
            }

            if (field is IfThenElse)
            {
                AddDispatchSegmentKernel(function, segment, outputs, segmentIndex);
                segment.Clear();
                if (ContainsKernelWork(field))
                {
                    throw new NotSupportedException($"PyNTT PrimFunction {function.Name} contains non-function work inside dispatch control flow. Split the branch work before PyNTT codegen.");
                }

                segmentIndex++;
                continue;
            }

            if (IsFunctionCall(field))
            {
                AddDispatchSegmentKernel(function, segment, outputs, segmentIndex);
                segment.Clear();
                segmentIndex++;
                continue;
            }

            segment.Add(field);
        }

        AddDispatchSegmentKernel(function, segment, outputs, segmentIndex);
    }

    private void AddDispatchSegmentKernel(PrimFunction function, IReadOnlyList<BaseExpr> fields, OutputInfo[] outputs, int segmentIndex)
    {
        if (fields.Count == 0)
        {
            return;
        }

        if (!fields.Any(ContainsKernelWork))
        {
            return;
        }

        var bodyFields = fields.Select((field, index) => field as Expr ??
            throw new NotSupportedException($"PyNTT dispatch segment {function.Name}[{segmentIndex}] field {index} must be an Expr, got {field.GetType().Name}.")).ToArray();
        var body = new Sequential(bodyFields);
        var generated = BuildPrimFunctionKernel(function, body, outputs, $"dispatch_{segmentIndex}", allowPartialOutputs: true, new Dictionary<string, object>
        {
            ["dispatch_segment_index"] = segmentIndex,
            ["workspace_scope"] = function.Name,
        });
        if (string.IsNullOrWhiteSpace(generated.RenderSpec.BodySource)
            && generated.RenderSpec.Helpers.Count == 0
            && !generated.Metadata.Attrs.ContainsKey("output_aliases")
            && !generated.Metadata.Attrs.ContainsKey("runtime_output_aliases"))
        {
            return;
        }

        AddPrimFunctionKernel(generated);
    }

    private void AddPrimFunctionKernel(PrimFunction function, BaseExpr body, OutputInfo[] outputs, int? dispatchSegmentIndex, bool allowPartialOutputs)
    {
        Dictionary<string, object>? extraAttrs = null;
        string? namePart = null;
        if (dispatchSegmentIndex.HasValue)
        {
            namePart = $"dispatch_{dispatchSegmentIndex.Value}";
            extraAttrs = new()
            {
                ["dispatch_segment_index"] = dispatchSegmentIndex.Value,
                ["workspace_scope"] = function.Name,
            };
        }

        AddPrimFunctionKernel(BuildPrimFunctionKernel(function, body, outputs, namePart, allowPartialOutputs, extraAttrs));
    }

    private GeneratedPrimFunctionKernel BuildPrimFunctionKernel(PrimFunction function, BaseExpr body, OutputInfo[] outputs, string? namePart, bool allowPartialOutputs, Dictionary<string, object>? extraAttrs)
    {
        var parameterNames = function.Parameters.ToArray()
            .ToDictionary(parameter => parameter, parameter => parameter.Name);
        return new PyNTTPrimFunctionSourceVisitor(function, body, parameterNames, outputs, _targetOptions, _sharedHelperRegistry, namePart, allowPartialOutputs, extraAttrs).Build();
    }

    private void AddPrimFunctionKernel(GeneratedPrimFunctionKernel kernel)
    {
        _generatedKernels.Add(kernel.Metadata);
        _renderKernels.Add(kernel.RenderSpec);
    }

    private static bool ContainsFunctionCall(BaseExpr expr)
    {
        if (IsFunctionCall(expr))
        {
            return true;
        }

        if (expr is PrimFunction primFunction)
        {
            return ContainsFunctionCall(primFunction.Body);
        }

        if (expr is Function or Fusion or FunctionWrapper or PrimFunctionWrapper)
        {
            throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction bodies only, but found {expr.GetType().Name}. TIR selection and RemoveFunctionWrapperPass must run before PyNTT codegen.");
        }

        if (expr is BaseFunction)
        {
            return false;
        }

        foreach (var operand in expr.Operands)
        {
            if (ContainsFunctionCall(operand))
            {
                return true;
            }
        }

        return false;
    }

    private static bool ContainsKernelWork(BaseExpr expr)
    {
        if (expr is Call call)
        {
            return call.Target is not BaseFunction;
        }

        if (expr is PrimFunction primFunction)
        {
            return ContainsKernelWork(primFunction.Body);
        }

        if (expr is Function or Fusion or FunctionWrapper or PrimFunctionWrapper)
        {
            throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction bodies only, but found {expr.GetType().Name}. TIR selection and RemoveFunctionWrapperPass must run before PyNTT codegen.");
        }

        if (expr is BaseFunction or Nop or TIR.Buffer or Const or IVar or Dimension or Shape or Padding or Paddings or None)
        {
            return false;
        }

        if (expr is IfThenElse ifThenElse)
        {
            return ContainsKernelWork(ifThenElse.Then) || ContainsKernelWork(ifThenElse.Else);
        }

        foreach (var operand in expr.Operands)
        {
            if (ContainsKernelWork(operand))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsFunctionCall(BaseExpr expr) => expr is Call { Target: BaseFunction };

#pragma warning disable SA1201
    private sealed class PyNTTPrimFunctionSourceVisitor : ExprFunctor<Unit, Unit>
    {
        private enum HelperBarrierKind
        {
            Block,
            Grid,
        }

        private const string ShardCoordDimPrefix = "__shard_coord_";

        private static readonly string[] WorkspaceParameterNames =
        {
            "data",
            "rdata",
            "chip_local_rdata",
            "block_local_rdata",
            "block_local_data",
            "data_pool_stride_bytes",
            "block_local_data_pool_stride_bytes",
        };

        private readonly PrimFunction _function;
        private readonly BaseExpr _bodyExpr;
        private readonly IReadOnlyDictionary<IVar, string> _parameterNames;
        private readonly OutputInfo[] _outputs;
        private readonly DistributedType?[] _outputDistributedTypes;
        private readonly PyNTTTargetOptions _targetOptions;
        private readonly string? _namePart;
        private readonly bool _allowPartialOutputs;
        private readonly Dictionary<string, object>? _extraAttrs;
        private readonly StringBuilder _body = new();
        private readonly List<HelperTemplateRenderSpec> _helpers = new();
        private readonly List<string> _inputNames = new();
        private readonly List<string> _opKinds = new();
        private readonly List<HelperKernelCallMetadata> _helperCalls = new();
        private readonly Dictionary<string, string[]> _helperArguments = new(StringComparer.Ordinal);
        private readonly List<PyNTTKVCacheFieldInputMetadata> _kvCacheFieldInputs = new();
        private readonly SortedSet<string> _runtimeScalarNames = new(StringComparer.Ordinal);
        private readonly SortedSet<string> _abiViewStrideArgNames = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _helperCounters = new();
        private readonly Dictionary<string, object> _attrs = new();
        private readonly Dictionary<TIR.Buffer, int> _bufferInputIndices = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<BufferVar, TIR.Buffer> _abiBufferMemo = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<int> _storedOutputIndices = new();
        private readonly Dictionary<int, int> _outputAliases = new();
        private readonly SharedHelperRegistry _sharedHelperRegistry;
        private readonly PyNTTDimExpressionEmitter _dimEmitter;
        private long _collectiveDataPoolBytes;
        private int _bodyIndent;

        public PyNTTPrimFunctionSourceVisitor(
            PrimFunction function,
            BaseExpr bodyExpr,
            IReadOnlyDictionary<IVar, string> parameterNames,
            OutputInfo[] outputs,
            PyNTTTargetOptions targetOptions,
            SharedHelperRegistry sharedHelperRegistry,
            string? namePart,
            bool allowPartialOutputs,
            Dictionary<string, object>? extraAttrs)
        {
            _function = function;
            _bodyExpr = bodyExpr;
            _parameterNames = parameterNames;
            _outputs = outputs;
            _outputDistributedTypes = new DistributedType?[outputs.Length];
            _targetOptions = targetOptions;
            _namePart = namePart;
            _allowPartialOutputs = allowPartialOutputs;
            _extraAttrs = extraAttrs;
            _sharedHelperRegistry = sharedHelperRegistry;
            _dimEmitter = new(RegisterRuntimeScalar, threadIdExpression: BuildThreadIdExpression(targetOptions));
        }

        public GeneratedPrimFunctionKernel Build()
        {
            Visit(_bodyExpr);
            var bodySource = _body.ToString().TrimEnd();
            var inputLayout = BuildKernelInputLayout(bodySource);
            var materializedOutputIndices = _storedOutputIndices.Concat(_outputAliases.Keys).ToHashSet();
            if (!_allowPartialOutputs && materializedOutputIndices.Count != _outputs.Length)
            {
                var missingOutputs = _outputs
                    .Select((output, index) => (output.Name, Index: index))
                    .Where(output => !materializedOutputIndices.Contains(output.Index))
                    .Select(output => output.Name);
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} does not contain TensorStore for output(s): {string.Join(", ", missingOutputs)}.");
            }

            var outputs = _outputs
                .Select((output, index) => output with
                {
                    DistributedType = _outputDistributedTypes[index] ?? output.DistributedType,
                })
                .ToArray();
            var opKind = _opKinds.Count == 1 ? _opKinds[0] : "composite";
            if (_opKinds.Count == 0)
            {
                opKind = _body.Length == 0 ? "alias" : "copy";
            }

            if (_opKinds.Count > 1)
            {
                _attrs["ops"] = _opKinds.ToArray();
            }

            if (_extraAttrs is not null)
            {
                foreach (var pair in _extraAttrs)
                {
                    _attrs[pair.Key] = pair.Value;
                }
            }

            var kernelOutputIndexes = outputs
                .Select((output, index) => (output, index))
                .Where(item => materializedOutputIndices.Contains(item.index) && !IsObjectDataType(item.output.DType))
                .ToArray();
            var kernelOutputs = kernelOutputIndexes.Select(item => item.output).ToArray();
            var kernelOutputIndexByFunctionIndex = kernelOutputIndexes
                .Select((item, kernelIndex) => (item.index, kernelIndex))
                .ToDictionary(item => item.index, item => item.kernelIndex);
            var kernelOutputAliases = new Dictionary<int, int>();
            var runtimeOutputAliases = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var pair in _outputAliases)
            {
                var sourceInputName = GetInputName(pair.Value, $"PyNTT output alias {outputs[pair.Key].Name}");
                runtimeOutputAliases[outputs[pair.Key].Name] = sourceInputName;
                if (kernelOutputIndexByFunctionIndex.TryGetValue(pair.Key, out var kernelOutputIndex))
                {
                    if (!inputLayout.IndexMap.TryGetValue(pair.Value, out var kernelInputIndex))
                    {
                        throw new NotSupportedException($"PyNTT output {outputs[pair.Key].Name} aliases object input {sourceInputName}, which cannot be passed to a Triton kernel.");
                    }

                    kernelOutputAliases[kernelOutputIndex] = kernelInputIndex;
                }
            }

            if (kernelOutputAliases.Count > 0)
            {
                _attrs["output_aliases"] = kernelOutputAliases;
            }

            if (runtimeOutputAliases.Count > 0)
            {
                _attrs["runtime_output_aliases"] = runtimeOutputAliases;
            }

            if (_runtimeScalarNames.Count > 0)
            {
                _attrs["runtime_shape_args"] = _runtimeScalarNames.ToArray();
            }

            if (_abiViewStrideArgNames.Count > 0)
            {
                _attrs["abi_view_stride_args"] = _abiViewStrideArgNames
                    .Select(argument => RemapInputReferences(argument, inputLayout.IndexMap, inputLayout.RemovedIndexes, $"PyNTT PrimFunction {_function.Name} ABI view stride argument {argument}"))
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(argument => argument.StartsWith("input", StringComparison.Ordinal) ? 0 : 1)
                    .ThenBy(ParseAbiArgumentIndex)
                    .ThenBy(argument => argument, StringComparer.Ordinal)
                    .ToArray();
            }

            if (_kvCacheFieldInputs.Count > 0)
            {
                _attrs["kv_cache_field_inputs"] = _kvCacheFieldInputs.ToArray();
            }

            if (_helperCalls.Count > 0)
            {
                _attrs["helper_calls"] = _helperCalls.ToArray();
            }

            if (opKind == "alias")
            {
                _attrs["pure_alias"] = true;
            }

            var kernelBaseName = _namePart is null ? _function.Name : $"{_function.Name}_{_namePart}";
            var metadata = new GeneratedKernelMetadata(
                SanitizePythonIdentifier($"{kernelBaseName}_{opKind}_0"),
                opKind,
                inputLayout.Names,
                kernelOutputs.Select(output => output.Name).ToArray(),
                _attrs,
                BuildLaunchMetadata(
                    kernelOutputs.Length > 0 ? kernelOutputs[0] : outputs[0],
                    _targetOptions,
                    new()
                    {
                        ["data_pool_bytes"] = checked((long)_function.SchedResult.DataUsage),
                        ["data_pool_elements"] = checked((long)_function.SchedResult.DataUsage),
                        ["data_dtype"] = "uint8",
                        ["collective_data_pool_bytes"] = _collectiveDataPoolBytes,
                        ["block_local_data_pool_bytes"] = checked((long)_function.SchedResult.BlockLocalDataPoolSize),
                        ["block_local_data_scope_count"] = GetBlockLocalDataScopeCount(_targetOptions),
                        ["rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.Rdatas),
                        ["chip_local_rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.ChipLocalRdatas),
                        ["block_local_rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.BlockLocalRdatas),
                        ["block_local_rdata_stride_bytes"] = PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.BlockLocalRdatas, _targetOptions, "b"),
                    }));
            var renderSpec = new KernelRenderSpec(
                metadata,
                inputLayout.Helpers,
                inputLayout.BodySource);
            return new(metadata, renderSpec);
        }

        protected override Unit VisitTuple(Nncase.IR.Tuple expr)
        {
            foreach (var field in expr.Fields)
            {
                Visit(field);
            }

            return default;
        }

        protected override Unit VisitSequential(Sequential expr)
        {
            foreach (var field in expr.Fields)
            {
                if (field is Function or Fusion or FunctionWrapper or PrimFunctionWrapper)
                {
                    throw new NotSupportedException($"PyNTT kernel codegen expects lowered PrimFunction bodies only, but found {field.GetType().Name}. TIR selection and RemoveFunctionWrapperPass must run before PyNTT codegen.");
                }

                if (field is BaseFunction)
                {
                    continue;
                }

                Visit(field);
            }

            return default;
        }

        protected override Unit VisitBuffer(TIR.Buffer expr)
        {
            return default;
        }

        protected override Unit VisitIfThenElse(IfThenElse expr)
        {
            WriteControlLine($"if {BuildScalarExpression(expr.Condition)}:");
            _bodyIndent++;
            Visit(expr.Then);
            _bodyIndent--;

            if (expr.Else.Count > 0)
            {
                WriteControlLine("else:");
                _bodyIndent++;
                Visit(expr.Else);
                _bodyIndent--;
            }

            return default;
        }

        protected override Unit VisitReturn(Return expr)
        {
            return default;
        }

        protected override Unit VisitCall(Call expr)
        {
            var args = expr.Arguments.ToArray();
            switch (expr.Target)
            {
                case Nncase.TIR.NTT.TensorLoad:
                    VisitTensorLoad(args);
                    break;
                case Nncase.TIR.NTT.TensorStore:
                    VisitTensorStore(args);
                    break;
                case Nncase.TIR.Memcopy:
                    VisitMemcopy(args);
                    break;
                case Nncase.TIR.NTT.Unary unary:
                    VisitUnary(unary, args);
                    break;
                case Nncase.TIR.NTT.Erf:
                    VisitElementwiseUnary(UnaryOp.Erf, "PyNTT Erf", args);
                    break;
                case Nncase.TIR.NTT.Expand:
                    VisitExpand(args);
                    break;
                case Nncase.TIR.NTT.Gather gather:
                    VisitGather(gather, args);
                    break;
                case Nncase.TIR.NTT.GatherReduceScatter gatherReduceScatter:
                    VisitGatherReduceScatter(gatherReduceScatter, args);
                    break;
                case Nncase.TIR.NTT.Pad pad:
                    VisitPad(pad, args);
                    break;
                case Nncase.TIR.NTT.ScatterND:
                    VisitScatterND(args);
                    break;
                case Nncase.TIR.NTT.Slice slice:
                    VisitSlice(slice, args);
                    break;
                case Nncase.TIR.NTT.Swish swish:
                    VisitSwish(swish, args);
                    break;
                case Nncase.TIR.NTT.VectorizedBinary binary:
                    VisitVectorizedBinary(binary, args);
                    break;
                case Nncase.TIR.NTT.Pack pack:
                    VisitPack(pack, args);
                    break;
                case Nncase.TIR.NTT.Unpack unpack:
                    VisitUnpack(unpack, args);
                    break;
                case Nncase.TIR.NTT.Cast cast:
                    VisitCast(cast, args);
                    break;
                case Nncase.TIR.NTT.Where:
                    VisitWhere(args);
                    break;
                case Nncase.TIR.NTT.Clamp clamp:
                    VisitClamp(clamp, args);
                    break;
                case Nncase.TIR.NTT.Compare compare:
                    VisitCompare(compare, args);
                    break;
                case Nncase.TIR.NTT.Concat concat:
                    VisitConcat(concat, args);
                    break;
                case Nncase.TIR.NTT.Conv2D conv2D:
                    VisitConv2D(conv2D, args);
                    break;
                case Nncase.TIR.NTT.Transpose transpose:
                    VisitTranspose(transpose, args);
                    break;
                case Nncase.TIR.NTT.Matmul matmul:
                    VisitMatmul(matmul, args);
                    break;
                case Nncase.TIR.NTT.PackedMatMul packedMatmul:
                    VisitPackedMatmul(packedMatmul, args);
                    break;
                case Nncase.TIR.NTT.QKVParallelLinear qkvParallelLinear:
                    VisitQKVParallelLinear(qkvParallelLinear, args);
                    break;
                case Nncase.TIR.NTT.PackedQKVParallelLinear packedQKVParallelLinear:
                    VisitPackedQKVParallelLinear(packedQKVParallelLinear, args);
                    break;
                case Nncase.TIR.NTT.MatMulGlu matMulGlu:
                    VisitMatMulGlu(matMulGlu, args);
                    break;
                case Nncase.TIR.NTT.PackedMatMulGlu packedMatMulGlu:
                    VisitPackedMatMulGlu(packedMatMulGlu, args);
                    break;
                case Nncase.TIR.NTT.SUMMA summa:
                    VisitSUMMA(summa, args);
                    break;
                case Nncase.TIR.NTT.Reduce reduce:
                    VisitReduce(reduce, args);
                    break;
                case Nncase.TIR.NTT.RoPE:
                    VisitRoPE(args);
                    break;
                case Nncase.TIR.NTT.GetPositionIds getPositionIds:
                    VisitGetPositionIds(getPositionIds, args);
                    break;
                case Nncase.TIR.NTT.UpdatePagedAttentionKVCache updatePagedAttentionKVCache:
                    VisitUpdatePagedAttentionKVCache(updatePagedAttentionKVCache, args);
                    break;
                case Nncase.TIR.NTT.PagedAttention pagedAttention:
                    VisitPagedAttention(pagedAttention, args);
                    break;
                case Nncase.TIR.NTT.VectorizedLayerNorm layerNorm:
                    VisitLayerNorm(layerNorm, args);
                    break;
                case Nncase.TIR.NTT.NormStats normStats:
                    VisitNormStats(normStats, args);
                    break;
                case Nncase.TIR.NTT.NormApply normApply:
                    VisitNormApply(normApply, args);
                    break;
                case Nncase.TIR.NTT.SynchronizeThreads:
                    _attrs["requires_grid_barrier"] = true;
                    WriteBarrier(HelperBarrierKind.Grid);
                    break;
                case Nncase.TIR.NTT.VectorizedSoftmax softmax:
                    VisitSoftmax(softmax.Axis, softmax.VectorizedAxes, args, "softmax");
                    break;
                case Nncase.TIR.NTT.Softmax softmax:
                    VisitSoftmax(softmax.Axis, default, args, "softmax");
                    break;
                case BaseFunction callee:
                    throw new NotSupportedException($"PyNTT kernel segment unexpectedly contains function call target {callee.GetType().Name} {callee.Name}. Caller dispatch must split nested PrimFunction calls before kernel codegen.");
                default:
                    throw new NotSupportedException($"Unsupported PyNTT PrimFunction call target: {expr.Target.GetType().Name}.");
            }

            return default;
        }

        private void VisitTensorLoad(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2 || args[0] is not TIR.Buffer dest)
            {
                throw new NotSupportedException("PyNTT TensorLoad codegen expects (dest_buffer, source_tensor).");
            }

            if (args[1] is TIR.Buffer srcBuffer)
            {
                VisitInternalTensorLoad(dest, srcBuffer);
                return;
            }

            var inputIndex = GetInputIndex(args[1]);
            _bufferInputIndices[dest] = inputIndex;
            if (IsObjectDataType(dest.ElemType))
            {
                return;
            }

            _attrs["tir"] = true;
            var localShape = GetBufferShape(dest);
            var globalShape = GetTensorShape(args[1], $"TensorLoad source input{inputIndex}");
            var helperName = GetNextHelperName("tensor_load");
            WriteHelperTemplate(
                "triton/kernels/TensorLoad.py.jinja",
                new PyNTTTensorLoadTemplateModel(
                    helperName,
                    $"input{inputIndex}",
                    0,
                    GetBufferPointer(dest),
                    GetPyNTTDTypeName(dest.ElemType),
                    GetTritonDType(dest.ElemType),
                    localShape,
                    GetBufferStrides(dest),
                    globalShape,
                    GetHierarchy(dest),
                    GetBufferSplitAxes(dest, globalShape.Length),
                    GetVectorLaneElementCount(dest.ElemType),
                    $"TensorLoad -> {dest.Name}"));
            WriteLine(BuildHelperCall(helperName, $"input{inputIndex}", $"input{inputIndex}_pool_stride_elements"));
        }

        private void VisitTensorStore(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2 || args[0] is not TIR.Buffer src)
            {
                throw new NotSupportedException("PyNTT TensorStore codegen expects (source_buffer, dest_tensor).");
            }

            if (args[1] is TIR.Buffer destBuffer)
            {
                if (destBuffer.MemSpan.Buffer.Location == MemoryLocation.Output)
                {
                    throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} must store to output BufferVar ABI parameters, but TensorStore destination {destBuffer.Name} is a TIR Buffer in Output memory.");
                }

                VisitInternalTensorStore(src, destBuffer);
                return;
            }

            var localShape = GetBufferShape(src);
            var globalShape = GetTensorShape(args[1], "TensorStore destination");
            var outputIndex = GetOutputIndex(args[1]);
            WriteTensorStore(src, outputIndex, globalShape, $"{src.Name} -> TensorStore");
        }

        private void VisitInternalTensorLoad(TIR.Buffer dest, TIR.Buffer src)
        {
            if (IsObjectDataType(dest.ElemType) || IsObjectDataType(src.ElemType))
            {
                return;
            }

            ValidateMatchingBufferDType("PyNTT TensorLoad buffer source/destination", src, dest);
            var sourcePointer = GetBufferScalarPointer(src);
            _attrs["tir"] = true;
            var localShape = GetBufferShape(dest);
            var globalShape = GetBufferShape(src);
            var helperName = GetNextHelperName("tensor_load");
            WriteHelperTemplate(
                "triton/kernels/TensorLoad.py.jinja",
                new PyNTTTensorLoadTemplateModel(
                    helperName,
                    src.Name,
                    0,
                    GetBufferPointer(dest),
                    GetPyNTTDTypeName(dest.ElemType),
                    GetTritonDType(dest.ElemType),
                    localShape,
                    GetBufferStrides(dest),
                    globalShape,
                    GetHierarchy(dest),
                    GetBufferSplitAxes(dest, globalShape.Length),
                    GetVectorLaneElementCount(dest.ElemType),
                    $"{src.Name} -> {dest.Name}")
                {
                    Source = sourcePointer,
                });
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitInternalTensorStore(TIR.Buffer src, TIR.Buffer dest)
        {
            if (IsObjectDataType(src.ElemType) || IsObjectDataType(dest.ElemType))
            {
                return;
            }

            ValidateMatchingBufferDType("PyNTT TensorStore buffer source/destination", src, dest);
            var destinationPointer = GetBufferScalarPointer(dest);
            _attrs["tir"] = true;
            var localShape = GetBufferShape(src);
            var globalShape = GetBufferShape(dest);
            var helperName = GetNextHelperName("tensor_store");
            WriteHelperTemplate(
                "triton/kernels/TensorStore.py.jinja",
                new PyNTTTensorStoreTemplateModel(
                    helperName,
                    GetBufferPointer(src),
                    dest.Name,
                    0,
                    GetPyNTTDTypeName(src.ElemType),
                    GetTritonDType(src.ElemType),
                    localShape,
                    GetBufferStrides(src),
                    globalShape,
                    GetHierarchy(src),
                    GetBufferSplitAxes(src, globalShape.Length),
                    GetVectorLaneElementCount(src.ElemType),
                    $"{src.Name} -> {dest.Name}")
                {
                    Destination = destinationPointer,
                });
            WriteLine(BuildHelperCall(helperName));
        }

        private static void ValidateMatchingBufferDType(string context, TIR.Buffer src, TIR.Buffer dest)
        {
            ValidateMatchingVectorLanes(context, src.ElemType, dest.ElemType);
            var srcScalarDType = GetPyNTTScalarDTypeName(src.ElemType);
            var destScalarDType = GetPyNTTScalarDTypeName(dest.ElemType);
            if (srcScalarDType != destScalarDType)
            {
                throw new NotSupportedException($"{context} expects matching scalar dtypes, got source {srcScalarDType} and destination {destScalarDType}.");
            }
        }

        private void WriteTensorStore(TIR.Buffer src, int outputIndex, PyNTTDimExpression[] globalShape, string comment)
        {
            _attrs["tir"] = true;
            if (!_storedOutputIndices.Add(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} contains multiple TensorStore calls for {_outputs[outputIndex].Name}.");
            }

            _outputDistributedTypes[outputIndex] = GetDistributedType(src);
            var outputName = _outputs[outputIndex].Name;
            var helperName = GetHelperName("output_tensor_store", outputIndex);
            var localShape = GetBufferShape(src);
            WriteHelperTemplate(
                "triton/kernels/TensorStore.py.jinja",
                new PyNTTTensorStoreTemplateModel(
                    helperName,
                    GetBufferPointer(src),
                    outputName,
                    0,
                    GetPyNTTDTypeName(src.ElemType),
                    GetTritonDType(src.ElemType),
                    localShape,
                    GetBufferStrides(src),
                    globalShape,
                    GetHierarchy(src),
                    GetBufferSplitAxes(src, globalShape.Length),
                    GetVectorLaneElementCount(src.ElemType),
                    comment));
            WriteLine(BuildHelperCall(helperName, outputName, $"{outputName}_pool_stride_elements"));
        }

        private void VisitMemcopy(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2)
            {
                throw new NotSupportedException("PyNTT Memcopy codegen expects (destination, source).");
            }

            var dest = UnwrapInputBoxing(args[0]);
            var src = UnwrapInputBoxing(args[1]);
            if (IsObjectExpression(dest) || IsObjectExpression(src))
            {
                VisitObjectMemcopy(dest, src);
                return;
            }

            if (dest is IVar && src is IVar)
            {
                VisitTensorOutputAliasMemcopy(dest, src);
                return;
            }

            if (dest is not TIR.Buffer destBuffer || src is not TIR.Buffer srcBuffer)
            {
                throw new NotSupportedException($"PyNTT Memcopy expects object aliases or TIR buffers, got destination {dest.GetType().Name} and source {src.GetType().Name}.");
            }

            ValidateSameShape("PyNTT Memcopy", GetBufferShape(srcBuffer), GetBufferShape(destBuffer));
            ValidateMatchingVectorLanes("PyNTT Memcopy source/destination", srcBuffer.ElemType, destBuffer.ElemType);
            var srcScalarDType = GetPyNTTScalarDTypeName(srcBuffer.ElemType);
            var destScalarDType = GetPyNTTScalarDTypeName(destBuffer.ElemType);
            if (srcScalarDType != destScalarDType)
            {
                throw new NotSupportedException($"PyNTT Memcopy expects matching scalar dtypes, got source {srcScalarDType} and destination {destScalarDType}.");
            }

            SetComputeOp("memcopy");
            _attrs["tir"] = true;
            _attrs["op"] = "memcopy";
            _attrs["dtype"] = GetPyNTTDTypeName(destBuffer.ElemType);
            _attrs["shape"] = GetBufferShape(destBuffer);
            var helperName = GetNextHelperName("memcopy");
            WriteHelperTemplate(
                "triton/kernels/Memcopy.py.jinja",
                new PyNTTMemcopyTemplateModel(
                    helperName,
                    GetBufferScalarPointer(srcBuffer),
                    GetBufferScalarPointer(destBuffer),
                    destScalarDType,
                    GetScalarTritonDType(destBuffer.ElemType),
                    GetBufferShape(destBuffer),
                    GetBufferStrides(srcBuffer),
                    GetBufferStrides(destBuffer),
                    GetVectorLaneElementCount(destBuffer.ElemType),
                    $"{srcBuffer.Name} -> {destBuffer.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitTensorOutputAliasMemcopy(BaseExpr dest, BaseExpr src)
        {
            var outputIndex = GetOutputIndex(dest);
            if (_storedOutputIndices.Contains(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} both by TensorStore and tensor Memcopy.");
            }

            var inputIndex = GetInputIndex(src);
            var destType = GetTensorType(dest.CheckedType, $"PyNTT tensor Memcopy destination {_outputs[outputIndex].Name}");
            var srcType = GetTensorType(src.CheckedType, $"PyNTT tensor Memcopy source input{inputIndex}");
            if (destType.DType != srcType.DType || destType.Shape.Rank != srcType.Shape.Rank)
            {
                throw new NotSupportedException($"PyNTT tensor Memcopy alias expects matching dtype/rank, got destination {destType} and source {srcType}.");
            }

            if (_outputAliases.TryGetValue(outputIndex, out var existingInputIndex) && existingInputIndex != inputIndex)
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} assigns output {_outputs[outputIndex].Name} to multiple input aliases.");
            }

            _outputAliases[outputIndex] = inputIndex;
        }

        private void VisitObjectMemcopy(BaseExpr dest, BaseExpr src)
        {
            if (!IsObjectExpression(dest) || !IsObjectExpression(src))
            {
                throw new NotSupportedException($"PyNTT object Memcopy expects both operands to be object tensors, got destination {dest.CheckedDataType} and source {src.CheckedDataType}.");
            }

            var outputIndex = GetOutputIndex(dest);
            if (_storedOutputIndices.Contains(outputIndex))
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} materializes output {_outputs[outputIndex].Name} both by TensorStore and object Memcopy.");
            }

            var inputIndex = GetInputIndex(src);
            if (_outputAliases.TryGetValue(outputIndex, out var existingInputIndex) && existingInputIndex != inputIndex)
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} assigns output {_outputs[outputIndex].Name} to multiple input aliases.");
            }

            _outputAliases[outputIndex] = inputIndex;
        }

        private void VisitUnary(Nncase.TIR.NTT.Unary unary, IReadOnlyList<BaseExpr> args)
        {
            VisitElementwiseUnary(unary.UnaryOp, "PyNTT Unary", args);
        }

        private void VisitElementwiseUnary(UnaryOp unaryOp, string context, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException($"{context} codegen expects TIR buffer operands.");
            }

            SetComputeOp("unary");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateSameShape(context, inputShape, shape);
            ValidateMatchingVectorLanes($"{context} input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = GetOpName(unaryOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_unary");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    shape,
                    GetUnaryExpression(unaryOp),
                    (string)_attrs["op"],
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitExpand(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Expand codegen expects input and output TIR buffers.");
            }

            SetComputeOp("expand");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateBroadcastable("PyNTT Expand input", inputShape, shape);
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT Expand input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Expand output");
            if (inputVectorLaneCount != 1 && inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT Expand expects input vector lanes to be scalar or match output lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            _attrs["op"] = "expand";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("expand_compute");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    shape,
                    "value0",
                    "expand",
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitGather(Nncase.TIR.NTT.Gather gather, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer index ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Gather codegen expects input, index, and output TIR buffers.");
            }

            SetComputeOp("gather");
            var inputShape = GetBufferShape(input);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var indexShape = GetBufferShape(index);
            var outputShape = GetBufferShape(output);
            var axis = NormalizeAxis(gather.Axis, inputShape.Length, "PyNTT Gather");
            var inputVectorLanes = GetVectorLanes(input.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var valueVectorLaneCount = 1;
            if (inputVectorLanes.Length != 0 || outputVectorLanes.Length != 0)
            {
                if (inputVectorLanes.Length != 1 || outputVectorLanes.Length != 1 || inputVectorLanes[0] != outputVectorLanes[0])
                {
                    throw new NotSupportedException($"PyNTT Gather currently supports matching one-dimensional input/output vector lanes only, got input lanes [{string.Join(",", inputVectorLanes)}] and output lanes [{string.Join(",", outputVectorLanes)}].");
                }

                if (axis == inputShape.Length - 1)
                {
                    throw new NotSupportedException("PyNTT Gather does not support gathering along the vectorized value axis yet.");
                }

                valueVectorLaneCount = inputVectorLanes[0];
            }

            ValidateGatherShape("PyNTT Gather", inputShape, indexShape, outputShape, axis);
            _attrs["op"] = "gather";
            _attrs["tir"] = true;
            _attrs["axis"] = axis;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("gather_compute");
            WriteHelperTemplate(
                "triton/kernels/Gather.py.jinja",
                new PyNTTGatherTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferPointer(index),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTDTypeName(index.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetTritonDType(index.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    inputGlobalShape,
                    indexShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(index),
                    GetBufferStrides(output),
                    axis,
                    valueVectorLaneCount,
                    GetHierarchy(input),
                    GetBufferSplitAxes(input, inputGlobalShape.Length),
                    $"{input.Name}, {index.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitGatherReduceScatter(Nncase.TIR.NTT.GatherReduceScatter gatherReduceScatter, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT GatherReduceScatter codegen expects input and output TIR buffers.");
            }

            if (!gatherReduceScatter.InType.Placement.Hierarchy.SequenceEqual(gatherReduceScatter.OutType.Placement.Hierarchy))
            {
                throw new NotSupportedException("PyNTT GatherReduceScatter expects input and output placements to use the same hierarchy.");
            }

            SetComputeOp("reshard");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var globalShape = GetRankedShapeDimensions(gatherReduceScatter.OutType.TensorType.Shape, "PyNTT GatherReduceScatter global shape")
                .Select(GetDimensionExpression)
                .ToArray();
            ValidateSameRank("PyNTT GatherReduceScatter input", inputShape, globalShape);
            ValidateSameRank("PyNTT GatherReduceScatter output", outputShape, globalShape);
            ValidateSameShape("PyNTT GatherReduceScatter global type", globalShape, GetRankedShapeDimensions(gatherReduceScatter.InType.TensorType.Shape, "PyNTT GatherReduceScatter input global shape").Select(GetDimensionExpression).ToArray());
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);
            if (inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT GatherReduceScatter expects matching input/output vector lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            var inputScalarDType = GetPyNTTScalarDTypeName(input.ElemType);
            var outputScalarDType = GetPyNTTScalarDTypeName(output.ElemType);
            if (inputScalarDType != outputScalarDType)
            {
                throw new NotSupportedException($"PyNTT GatherReduceScatter expects matching scalar dtypes, got input={inputScalarDType}, output={outputScalarDType}.");
            }

            var hierarchy = gatherReduceScatter.OutType.Placement.Hierarchy.ToArray();
            if (gatherReduceScatter.InType.Partial is { } partial)
            {
                if (partial.Op != ReduceOp.Sum)
                {
                    throw new NotSupportedException($"PyNTT GatherReduceScatter currently supports Sum partial reduction only, got {partial.Op}.");
                }

                var reduceAxes = partial.Axes.ToArray();
                foreach (var axis in reduceAxes)
                {
                    if (axis < 0 || axis >= hierarchy.Length)
                    {
                        throw new NotSupportedException($"PyNTT GatherReduceScatter partial reduce axis {axis} is outside hierarchy rank {hierarchy.Length}.");
                    }
                }

                WriteBarrier(HelperBarrierKind.Grid);
                WriteShardReduceHelper(input, reduceAxes, inputVectorLaneCount, inputScalarDType, hierarchy, false, HelperBarrierKind.Grid, $"{input.Name} partial reduce");
                WriteShardReduceHelper(input, reduceAxes, inputVectorLaneCount, inputScalarDType, hierarchy, true, HelperBarrierKind.Block, $"{input.Name} partial broadcast");
            }

            _attrs["op"] = "reshard";
            _attrs["tir"] = true;
            _attrs["requires_grid_barrier"] = true;
            _attrs["dtype"] = outputScalarDType;
            _attrs["shape"] = globalShape;
            var inputRef = ResolveBufferRef(input);
            var outputRef = ResolveBufferRef(output);
            var collectiveOffsetBytes = checked((long)_function.SchedResult.DataUsage * hierarchy.Aggregate(1L, (acc, value) => checked(acc * value)));
            var collectivePoolBytes = GetTensorMaxSizeBytes(globalShape, inputVectorLaneCount, GetScalarElementSizeBytes(output.ElemType), "PyNTT GatherReduceScatter collective pool");
            _collectiveDataPoolBytes = Math.Max(_collectiveDataPoolBytes, collectivePoolBytes);

            PyNTTReshardTemplateModel MakeModel(string helperName, string stage) => new(
                helperName,
                GetBufferScalarPointer(input, "source_shard_index"),
                GetBufferScalarPointer(output),
                inputRef.BaseName,
                inputRef.OffsetBytes,
                inputRef.PoolStrideBytes,
                outputRef.BaseName,
                outputRef.OffsetBytes,
                outputRef.PoolStrideBytes,
                GetScalarElementSizeBytes(output.ElemType),
                outputScalarDType,
                GetScalarTritonDType(output.ElemType),
                globalShape,
                inputShape,
                outputShape,
                GetBufferStrides(input),
                GetBufferStrides(output),
                inputVectorLaneCount,
                hierarchy,
                GetSplitAxes(gatherReduceScatter.InType),
                GetSplitAxes(gatherReduceScatter.OutType),
                collectiveOffsetBytes,
                collectivePoolBytes,
                stage,
                $"{input.Name} -> {output.Name}");

            var toCollectiveHelperName = GetNextHelperName("reshard_to_collective");
            WriteHelperTemplate("triton/kernels/Reshard.py.jinja", MakeModel(toCollectiveHelperName, "to_collective"));
            var fromCollectiveHelperName = GetNextHelperName("reshard_from_collective");
            WriteHelperTemplate("triton/kernels/Reshard.py.jinja", MakeModel(fromCollectiveHelperName, "from_collective"));
            WriteBarrier(HelperBarrierKind.Grid);
            WriteLine(BuildHelperCall(toCollectiveHelperName), HelperBarrierKind.Grid);
            WriteLine(BuildHelperCall(fromCollectiveHelperName), HelperBarrierKind.Grid);
        }

        private void VisitPad(Nncase.TIR.NTT.Pad pad, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not Paddings paddings ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Pad codegen expects input buffer, static paddings, and output buffer.");
            }

            SetComputeOp("pad");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var pads = GetStaticPaddings(paddings, "PyNTT Pad");
            ValidatePadShape("PyNTT Pad", inputShape, outputShape, pads);
            _attrs["op"] = "pad";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["pads"] = pads;
            _attrs["pad_value"] = pad.PadValue;
            var helperName = GetNextHelperName("pad_compute");
            WriteHelperTemplate(
                "triton/kernels/Pad.py.jinja",
                new PyNTTPadTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    pads,
                    pad.PadValue.ToString("R", CultureInfo.InvariantCulture),
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitScatterND(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer indices ||
                args[2] is not TIR.Buffer updates ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT ScatterND codegen expects input, indices, updates, and output TIR buffers.");
            }

            SetComputeOp("scatter_nd");
            var inputShape = GetBufferShape(input);
            var indicesShape = GetBufferShape(indices);
            var updatesShape = GetBufferShape(updates);
            var outputShape = GetBufferShape(output);
            ValidateScatterNDShape("PyNTT ScatterND", inputShape, indicesShape, updatesShape, outputShape);
            _attrs["op"] = "scatter_nd";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("scatter_nd_compute");
            WriteHelperTemplate(
                "triton/kernels/ScatterND.py.jinja",
                new PyNTTScatterNDTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(indices),
                    GetBufferPointer(updates),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(indices.ElemType),
                    GetPyNTTDTypeName(updates.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(indices.ElemType),
                    GetTritonDType(updates.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    indicesShape,
                    updatesShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(indices),
                    GetBufferStrides(updates),
                    GetBufferStrides(output),
                    $"{input.Name}, {indices.Name}, {updates.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitSlice(Nncase.TIR.NTT.Slice slice, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Slice codegen expects input buffer, begins, ends, and output buffer.");
            }

            SetComputeOp("slice");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var starts = GetStaticRankedShape(args[1], "PyNTT Slice begins");
            var axes = slice.Axes.ToArray();
            var strides = slice.Strides.ToArray();
            var (normalizedStarts, normalizedStrides) = NormalizeSliceParameters(inputShape, starts, axes, strides, "PyNTT Slice");
            ValidateSameRank("PyNTT Slice", inputShape, outputShape);
            _attrs["op"] = "slice";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["starts"] = normalizedStarts;
            _attrs["strides"] = normalizedStrides;
            var helperName = GetNextHelperName("slice_compute");
            WriteHelperTemplate(
                "triton/kernels/Slice.py.jinja",
                new PyNTTSliceTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    normalizedStarts,
                    normalizedStrides,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitSwish(Nncase.TIR.NTT.Swish swish, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Swish codegen expects input and output TIR buffers.");
            }

            SetComputeOp("swish");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateSameShape("PyNTT Swish", inputShape, shape);
            ValidateMatchingVectorLanes("PyNTT Swish input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            var beta = swish.Beta.ToString("R", CultureInfo.InvariantCulture);
            _attrs["op"] = "swish";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["beta"] = swish.Beta;
            var helperName = GetNextHelperName("swish_compute");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    shape,
                    $"value0_f32 / (1.0 + tl.exp(-({beta}) * value0_f32))",
                    "swish",
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitVectorizedBinary(Nncase.TIR.NTT.VectorizedBinary binary, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3)
            {
                throw new NotSupportedException("PyNTT VectorizedBinary codegen expects TIR buffer operands.");
            }

            var lhs = GetBufferOperand(args[0], "PyNTT VectorizedBinary lhs");
            var rhs = GetBufferOperand(args[1], "PyNTT VectorizedBinary rhs");
            var output = GetBufferOperand(args[2], "PyNTT VectorizedBinary output");

            SetComputeOp("binary");
            var shape = GetBufferShape(output);
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            ValidateBroadcastable("PyNTT VectorizedBinary lhs", lhsShape, shape);
            ValidateBroadcastable("PyNTT VectorizedBinary rhs", rhsShape, shape);
            ValidateScalarOrMatchingVectorLanes("PyNTT VectorizedBinary lhs", lhs.ElemType, output.ElemType);
            ValidateScalarOrMatchingVectorLanes("PyNTT VectorizedBinary rhs", rhs.ElemType, output.ElemType);
            var lhsVectorLaneCount = GetVectorLaneElementCount(lhs.ElemType);
            var rhsVectorLaneCount = GetVectorLaneElementCount(rhs.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = GetOpName(binary.BinaryOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_binary");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseBinary.py.jinja",
                new PyNTTElementwiseBinaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(lhs),
                    GetBufferScalarPointer(rhs),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(lhs.ElemType),
                    GetPyNTTScalarDTypeName(rhs.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(lhs.ElemType),
                    GetScalarTritonDType(rhs.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    shape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    lhsVectorLaneCount,
                    rhsVectorLaneCount,
                    outputVectorLaneCount,
                    shape,
                    GetBinaryExpression(binary.BinaryOp),
                    (string)_attrs["op"],
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
            MarkStoredOutput(output, "PyNTT VectorizedBinary");
        }

        private void VisitPack(Nncase.TIR.NTT.Pack pack, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Pack codegen expects input and output TIR buffers.");
            }

            SetComputeOp("pack");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var outputGlobalShape = GetBufferGlobalShape(output);
            var axes = NormalizeLayoutAxes(pack.Axes, inputShape.Length, "PyNTT Pack");
            var lanes = GetLayoutLanes(pack.Lanes, axes.Length, "PyNTT Pack");
            ValidatePackShape("PyNTT Pack global", inputGlobalShape, outputGlobalShape, axes, lanes);
            ValidateSameRank("PyNTT Pack local", inputShape, outputShape);
            var inputLanes = GetVectorLanes(input.ElemType);
            var outputLanes = GetVectorLanes(output.ElemType);
            ValidateLanePrefix("PyNTT Pack output lanes", lanes.Concat(inputLanes).ToArray(), outputLanes);

            _attrs["op"] = "pack";
            _attrs["tir"] = true;
            _attrs["from_dtype"] = GetPyNTTDTypeName(input.ElemType);
            _attrs["to_dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axes"] = axes;
            _attrs["lanes"] = lanes;
            var helperName = GetNextHelperName("pack_compute");
            WriteHelperTemplate(
                "triton/kernels/VectorLayout.py.jinja",
                new PyNTTVectorLayoutTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputLanes,
                    outputLanes,
                    axes,
                    lanes,
                    true,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitUnpack(Nncase.TIR.NTT.Unpack unpack, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Unpack codegen expects input and output TIR buffers.");
            }

            SetComputeOp("unpack");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var outputGlobalShape = GetBufferGlobalShape(output);
            var axes = NormalizeLayoutAxes(unpack.Axes, outputShape.Length, "PyNTT Unpack");
            var lanes = GetLayoutLanes(unpack.Lanes, axes.Length, "PyNTT Unpack");
            ValidateUnpackShape("PyNTT Unpack global", inputGlobalShape, outputGlobalShape, axes, lanes);
            ValidateSameRank("PyNTT Unpack local", inputShape, outputShape);
            var inputLanes = GetVectorLanes(input.ElemType);
            var outputLanes = GetVectorLanes(output.ElemType);
            ValidateLanePrefix("PyNTT Unpack input lanes", lanes.Concat(outputLanes).ToArray(), inputLanes);

            _attrs["op"] = "unpack";
            _attrs["tir"] = true;
            _attrs["from_dtype"] = GetPyNTTDTypeName(input.ElemType);
            _attrs["to_dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axes"] = axes;
            _attrs["lanes"] = lanes;
            var helperName = GetNextHelperName("unpack_compute");
            WriteHelperTemplate(
                "triton/kernels/VectorLayout.py.jinja",
                new PyNTTVectorLayoutTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputLanes,
                    outputLanes,
                    axes,
                    lanes,
                    false,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitCast(Nncase.TIR.NTT.Cast cast, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2)
            {
                throw new NotSupportedException("PyNTT Cast codegen expects TIR buffer operands.");
            }

            var input = GetBufferOperand(args[0], "PyNTT Cast input");
            var output = GetBufferOperand(args[1], "PyNTT Cast output");
            if (args.Count > 2 && args[2] is not None)
            {
                throw new NotSupportedException("PyNTT Cast codegen does not support post ops yet.");
            }

            SetComputeOp("cast");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT Cast input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Cast output");
            var vectorizedAxes = cast.VectorizeAxes.ToArray();
            if (vectorizedAxes.Length > 1)
            {
                throw new NotSupportedException($"PyNTT Cast supports at most one vectorized axis, got [{string.Join(",", vectorizedAxes)}].");
            }

            if ((inputVectorLaneCount != 1 || outputVectorLaneCount != 1) && vectorizedAxes.Length == 0)
            {
                throw new NotSupportedException("PyNTT Cast with vector dtype expects one vectorized axis.");
            }

            var logicalInputShape = vectorizedAxes.Length == 0
                ? inputShape
                : GetLogicalVectorShape(inputShape, vectorizedAxes[0], inputVectorLaneCount);
            var logicalOutputShape = vectorizedAxes.Length == 0
                ? outputShape
                : GetLogicalVectorShape(outputShape, vectorizedAxes[0], outputVectorLaneCount);
            ValidateSameShape("PyNTT Cast", logicalInputShape, logicalOutputShape);
            _attrs["op"] = "cast";
            _attrs["tir"] = true;
            _attrs["from_dtype"] = GetPyNTTScalarDTypeName(input.ElemType);
            _attrs["to_dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["cast_mode"] = GetCastModeName(cast.CastMode);
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalOutputShape;
            var helperName = GetNextHelperName("elementwise_cast");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseCast.py.jinja",
                new PyNTTElementwiseCastTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    vectorizedAxes,
                    logicalOutputShape,
                    GetCastExpression(cast.CastMode, GetScalarTritonDType(output.ElemType)),
                    (string)_attrs["cast_mode"],
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
            MarkStoredOutput(output, "PyNTT Cast");
        }

        private void VisitWhere(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer cond ||
                args[1] is not TIR.Buffer trueValue ||
                args[2] is not TIR.Buffer falseValue ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Where codegen expects TIR buffer operands.");
            }

            SetComputeOp("where");
            var shape = GetBufferShape(output);
            var condShape = GetBufferShape(cond);
            var trueShape = GetBufferShape(trueValue);
            var falseShape = GetBufferShape(falseValue);
            var condVectorLaneCount = GetSingleVectorLaneCount(cond.ElemType, "PyNTT Where cond");
            var trueVectorLaneCount = GetSingleVectorLaneCount(trueValue.ElemType, "PyNTT Where true value");
            var falseVectorLaneCount = GetSingleVectorLaneCount(falseValue.ElemType, "PyNTT Where false value");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Where output");
            var valueLaneCounts = new[] { trueVectorLaneCount, falseVectorLaneCount, outputVectorLaneCount }
                .Where(lane => lane > 1)
                .Distinct()
                .ToArray();
            if (valueLaneCounts.Length > 1)
            {
                throw new NotSupportedException($"PyNTT Where expects matching value/output vector lanes, got true={trueVectorLaneCount}, false={falseVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            var logicalShape = GetLogicalVectorShape(shape, outputVectorLaneCount);
            ValidateBroadcastable("PyNTT Where cond", GetLogicalVectorShape(condShape, condVectorLaneCount), logicalShape);
            ValidateBroadcastable("PyNTT Where true value", GetLogicalVectorShape(trueShape, trueVectorLaneCount), logicalShape);
            ValidateBroadcastable("PyNTT Where false value", GetLogicalVectorShape(falseShape, falseVectorLaneCount), logicalShape);
            _attrs["op"] = "where";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalShape;
            var helperName = GetNextHelperName("elementwise_where");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseWhere.py.jinja",
                new PyNTTElementwiseWhereTemplateModel(
                    helperName,
                    GetBufferScalarPointer(cond),
                    GetBufferScalarPointer(trueValue),
                    GetBufferScalarPointer(falseValue),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(cond.ElemType),
                    GetPyNTTScalarDTypeName(trueValue.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(cond.ElemType),
                    GetScalarTritonDType(trueValue.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    condShape,
                    trueShape,
                    falseShape,
                    shape,
                    GetBufferStrides(cond),
                    GetBufferStrides(trueValue),
                    GetBufferStrides(falseValue),
                    GetBufferStrides(output),
                    condVectorLaneCount,
                    trueVectorLaneCount,
                    falseVectorLaneCount,
                    outputVectorLaneCount,
                    logicalShape,
                    $"{cond.Name}, {trueValue.Name}, {falseValue.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitClamp(Nncase.TIR.NTT.Clamp clamp, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Clamp codegen expects input and output TIR buffers.");
            }

            SetComputeOp("clamp");
            var shape = GetBufferShape(output);
            var inputShape = GetBufferShape(input);
            ValidateSameShape("PyNTT Clamp", inputShape, shape);
            ValidateMatchingVectorLanes("PyNTT Clamp input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = "clamp";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["min"] = clamp.Min;
            _attrs["max"] = clamp.Max;
            var helperName = GetNextHelperName("elementwise_clamp");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseUnary.py.jinja",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    shape,
                    GetClampExpression(clamp.Min, clamp.Max),
                    "clamp",
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitCompare(Nncase.TIR.NTT.Compare compare, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Compare codegen expects lhs, rhs, and output TIR buffers.");
            }

            SetComputeOp("compare");
            var shape = GetBufferShape(output);
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            ValidateBroadcastable("PyNTT Compare lhs", lhsShape, shape);
            ValidateBroadcastable("PyNTT Compare rhs", rhsShape, shape);
            ValidateScalarOrMatchingVectorLanes("PyNTT Compare lhs", lhs.ElemType, output.ElemType);
            ValidateScalarOrMatchingVectorLanes("PyNTT Compare rhs", rhs.ElemType, output.ElemType);
            var lhsVectorLaneCount = GetVectorLaneElementCount(lhs.ElemType);
            var rhsVectorLaneCount = GetVectorLaneElementCount(rhs.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            _attrs["op"] = GetCompareOpName(compare.CompareOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_compare");
            WriteHelperTemplate(
                "triton/kernels/ElementwiseBinary.py.jinja",
                new PyNTTElementwiseBinaryTemplateModel(
                    helperName,
                    GetBufferScalarPointer(lhs),
                    GetBufferScalarPointer(rhs),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(lhs.ElemType),
                    GetPyNTTScalarDTypeName(rhs.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(lhs.ElemType),
                    GetScalarTritonDType(rhs.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    shape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    lhsVectorLaneCount,
                    rhsVectorLaneCount,
                    outputVectorLaneCount,
                    shape,
                    GetCompareExpression(compare.CompareOp),
                    (string)_attrs["op"],
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitConcat(Nncase.TIR.NTT.Concat concat, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[^1] is not TIR.Buffer output ||
                args.Take(args.Count - 1).Any(arg => arg is not TIR.Buffer))
            {
                throw new NotSupportedException("PyNTT Concat codegen expects one or more input TIR buffers and one output TIR buffer.");
            }

            SetComputeOp("concat");
            var inputs = args.Take(args.Count - 1).Cast<TIR.Buffer>().ToArray();
            var outputShape = GetBufferShape(output);
            if (outputShape.Length == 0)
            {
                throw new NotSupportedException("PyNTT Concat codegen requires ranked tensor buffers.");
            }

            var axis = NormalizeAxis(concat.Axis, outputShape.Length, "PyNTT Concat");
            var inputShapes = inputs.Select(GetBufferShape).ToArray();
            var inputStrides = inputs.Select(GetBufferStrides).ToArray();
            var concatAxisExtent = PyNTTDimExpression.Zero;
            for (var inputIndex = 0; inputIndex < inputShapes.Length; inputIndex++)
            {
                var inputShape = inputShapes[inputIndex];
                if (inputShape.Length != outputShape.Length)
                {
                    throw new NotSupportedException($"PyNTT Concat input{inputIndex} rank {inputShape.Length} does not match output rank {outputShape.Length}.");
                }

                for (var dim = 0; dim < outputShape.Length; dim++)
                {
                    if (dim != axis && !SameDim(inputShape[dim], outputShape[dim]))
                    {
                        throw new NotSupportedException($"PyNTT Concat input{inputIndex} dim {dim}={inputShape[dim]} does not match output dim {outputShape[dim]}.");
                    }
                }

                concatAxisExtent = AddDims(concatAxisExtent, inputShape[axis]);
            }

            if (!SameDim(concatAxisExtent, outputShape[axis]))
            {
                throw new NotSupportedException($"PyNTT Concat input axis extent sum {concatAxisExtent} does not match output axis extent {outputShape[axis]}.");
            }

            _attrs["op"] = "concat";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axis"] = axis;
            _attrs["input_count"] = inputs.Length;
            var helperName = GetNextHelperName("concat_compute");
            WriteHelperTemplate(
                "triton/kernels/Concat.py.jinja",
                new PyNTTConcatTemplateModel(
                    helperName,
                    inputs.Select(GetBufferPointer).ToArray(),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShapes,
                    inputStrides,
                    outputShape,
                    GetBufferStrides(output),
                    axis,
                    $"{string.Join(", ", inputs.Select(input => input.Name))} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitConv2D(Nncase.TIR.NTT.Conv2D conv2D, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer weights ||
                args[2] is not TIR.Buffer bias ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Conv2D codegen expects input, weights, bias, and output TIR buffers.");
            }

            if (conv2D.PadMode != PadMode.Constant)
            {
                throw new NotSupportedException($"PyNTT Conv2D currently supports constant padding only, got {conv2D.PadMode}.");
            }

            SetComputeOp("conv2d");
            var inputShape = GetBufferShape(input);
            var weightsShape = GetBufferShape(weights);
            var biasShape = GetBufferShape(bias);
            var outputShape = GetBufferShape(output);
            ValidateRank("PyNTT Conv2D input", inputShape, 4);
            ValidateRank("PyNTT Conv2D weights", weightsShape, 4);
            ValidateRank("PyNTT Conv2D bias", biasShape, 1);
            ValidateRank("PyNTT Conv2D output", outputShape, 4);

            var stride = conv2D.Stride.ToArray();
            var padding = conv2D.Padding.ToArray();
            var dilation = conv2D.Dilation.ToArray();
            if (stride.Length != 2 || padding.Length != 4 || dilation.Length != 2)
            {
                throw new NotSupportedException($"PyNTT Conv2D expects stride rank 2, padding rank 4, dilation rank 2; got stride={stride.Length}, padding={padding.Length}, dilation={dilation.Length}.");
            }

            if (conv2D.Groups <= 0)
            {
                throw new NotSupportedException($"PyNTT Conv2D requires positive groups, got {conv2D.Groups}.");
            }

            if (!SameDim(outputShape[1], weightsShape[0]) || !SameDim(biasShape[0], outputShape[1]))
            {
                throw new NotSupportedException($"PyNTT Conv2D expects output channels, weights, and bias to match; got output={outputShape[1]}, weights={weightsShape[0]}, bias={biasShape[0]}.");
            }

            var weightsInputChannels = MultiplyDim(weightsShape[1], conv2D.Groups);
            if (!SameDim(inputShape[1], weightsInputChannels))
            {
                throw new NotSupportedException($"PyNTT Conv2D expects input channels {inputShape[1]} to equal weights input channels {weightsShape[1]} * groups {conv2D.Groups}.");
            }

            var outputChannels = RequireFixedDim(outputShape[1], "PyNTT Conv2D output channels");
            if (outputChannels % conv2D.Groups != 0)
            {
                throw new NotSupportedException($"PyNTT Conv2D expects output channels {outputShape[1]} to be divisible by groups {conv2D.Groups}.");
            }

            _attrs["op"] = "conv2d";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["stride"] = stride;
            _attrs["padding"] = padding;
            _attrs["dilation"] = dilation;
            _attrs["groups"] = conv2D.Groups;
            var helperName = GetNextHelperName("conv2d_compute");
            WriteHelperTemplate(
                "triton/kernels/Conv2D.py.jinja",
                new PyNTTConv2DTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(weights),
                    GetBufferPointer(bias),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(weights.ElemType),
                    GetPyNTTDTypeName(bias.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(weights.ElemType),
                    GetTritonDType(bias.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    weightsShape,
                    biasShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(weights),
                    GetBufferStrides(bias),
                    GetBufferStrides(output),
                    stride,
                    padding,
                    dilation,
                    conv2D.Groups,
                    $"{input.Name}, {weights.Name}, {bias.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitTranspose(Nncase.TIR.NTT.Transpose transpose, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Transpose codegen expects input and output TIR buffers.");
            }

            SetComputeOp("transpose");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var perm = transpose.Perm.ToArray();
            if (perm.Length != inputShape.Length || outputShape.Length != inputShape.Length)
            {
                throw new NotSupportedException($"PyNTT Transpose expects input/output/perm ranks to match, got input={inputShape.Length}, output={outputShape.Length}, perm={perm.Length}.");
            }

            var sortedPerm = perm.OrderBy(axis => axis).ToArray();
            if (!sortedPerm.SequenceEqual(Enumerable.Range(0, perm.Length)))
            {
                throw new NotSupportedException($"PyNTT Transpose expects a valid permutation, got [{string.Join(",", perm)}].");
            }

            for (var outputAxis = 0; outputAxis < outputShape.Length; outputAxis++)
            {
                var inputAxis = perm[outputAxis];
                if (!SameDim(outputShape[outputAxis], inputShape[inputAxis]))
                {
                    throw new NotSupportedException($"PyNTT Transpose output axis {outputAxis} shape {outputShape[outputAxis]} does not match input axis {inputAxis} shape {inputShape[inputAxis]}.");
                }
            }

            _attrs["op"] = "transpose";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["perm"] = perm;
            ValidateMatchingVectorLanes("PyNTT Transpose input/output", input.ElemType, output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);

            var helperName = GetNextHelperName("transpose_compute");
            WriteHelperTemplate(
                "triton/kernels/Transpose.py.jinja",
                new PyNTTTransposeTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    perm,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitMatmul(Nncase.TIR.NTT.Matmul matmul, IReadOnlyList<BaseExpr> args)
        {
            if (matmul.FusedReduce)
            {
                throw new NotSupportedException("PyNTT Matmul codegen does not support fused reduce yet.");
            }

            if (args.Count > 5 && args[5] is not None)
            {
                throw new NotSupportedException("PyNTT Matmul codegen does not support extra workload operands yet.");
            }

            VisitMatmulLike(
                "PyNTT Matmul",
                matmul.LhsVectorizedAxes,
                matmul.RhsVectorizedAxes,
                matmul.TransposeA,
                matmul.TransposeB,
                args);
        }

        private void VisitPackedMatmul(Nncase.TIR.NTT.PackedMatMul matmul, IReadOnlyList<BaseExpr> args)
        {
            if (matmul.FusedReduce)
            {
                throw new NotSupportedException("PyNTT PackedMatMul codegen does not support fused reduce yet.");
            }

            if (args.Count < 5 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT PackedMatMul codegen expects lhs, rhs, and output TIR buffers.");
            }

            if (GetScalarBool(args[3], "PyNTT PackedMatMul loadC"))
            {
                throw new NotSupportedException("PyNTT PackedMatMul codegen does not support loadC yet.");
            }

            SetComputeOp("matmul");
            var lhsShape = GetBufferActiveShape(lhs);
            var rhsShape = GetBufferActiveShape(rhs);
            var outputShape = GetBufferActiveShape(output);
            ValidateMinimumRank("PyNTT PackedMatMul lhs", lhsShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMul rhs", rhsShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMul output", outputShape, 2);

            var lhsVectorLanes = GetVectorLanes(lhs.ElemType);
            var rhsVectorLanes = GetVectorLanes(rhs.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            if (lhsVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects scalar lhs operands, got lhs lanes [{string.Join(",", lhsVectorLanes)}].");
            }

            if (rhsVectorLanes.Length != 2)
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects RHS vector lanes [Nr,lane], got [{string.Join(",", rhsVectorLanes)}].");
            }

            if (outputVectorLanes.Length != 2 || outputVectorLanes[0] != rhsVectorLanes[0] || outputVectorLanes[1] != rhsVectorLanes[1])
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects output lanes [{rhsVectorLanes[0]},{rhsVectorLanes[1]}], got [{string.Join(",", outputVectorLanes)}].");
            }

            var nPackedLaneCount = rhsVectorLanes[0];
            var nVectorLaneCount = rhsVectorLanes[1];
            var rhsNScalarLaneCount = checked(nPackedLaneCount * nVectorLaneCount);
            var lhsK = lhsShape[^1];
            var rhsK = rhsShape[^1];
            var lhsM = lhsShape[^2];
            var rhsNOuter = rhsShape[^2];
            if (!SameDim(lhsK, rhsK) ||
                !SameDim(outputShape[^2], lhsM) ||
                !SameDim(outputShape[^1], rhsNOuter))
            {
                throw new NotSupportedException($"PyNTT PackedMatMul expects lhs=[...,M,K], rhs=[...,N,K]<Nr,lane>, output=[...,M,N]<Nr,lane>, got lhs=[{ShapeText(lhsShape)}], rhs=[{ShapeText(rhsShape)}], output=[{ShapeText(outputShape)}].");
            }

            ValidateBroadcastable("PyNTT PackedMatMul lhs batch", lhsShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedMatMul rhs batch", rhsShape[..^2], outputShape[..^2]);
            var dimInfo = Nncase.IR.NTT.VectorizedMatMul.GetDimInfo(false, true, lhsShape.Length, rhsShape.Length);
            var scale = GetScalarFloat(args[4], "PyNTT PackedMatMul scale", 1.0f);
            _attrs["op"] = "packed_matmul";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["n_pack_lane"] = nPackedLaneCount;
            _attrs["n_lane"] = nVectorLaneCount;
            _attrs["n_scalar_lane"] = rhsNScalarLaneCount;
            _attrs["scale"] = scale;
            var useGemv = IsGemvMatmul(outputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "gemv_compute" : "matmul_compute");
            var templateModel = new PyNTTMatmulTemplateModel(
                helperName,
                GetBufferScalarPointer(lhs),
                GetBufferScalarPointer(rhs),
                GetBufferScalarPointer(output),
                GetPyNTTScalarDTypeName(lhs.ElemType),
                GetPyNTTScalarDTypeName(rhs.ElemType),
                GetPyNTTScalarDTypeName(output.ElemType),
                GetScalarTritonDType(lhs.ElemType),
                GetScalarTritonDType(rhs.ElemType),
                GetScalarTritonDType(output.ElemType),
                lhsShape,
                rhsShape,
                outputShape,
                GetBufferStrides(lhs),
                GetBufferStrides(rhs),
                GetBufferStrides(output),
                false,
                true,
                GetHierarchy(output),
                nVectorLaneCount,
                outputVectorLanes[1],
                scale.ToString("R", CultureInfo.InvariantCulture),
                $"{lhs.Name}, {rhs.Name} -> {output.Name}")
            {
                RhsNPackedLaneCount = nPackedLaneCount,
                OutputNPackedLaneCount = outputVectorLanes[0],
            };

            WriteHelperTemplate(
                useGemv ? "triton/kernels/Gemv.py.jinja" : "triton/kernels/Matmul.py.jinja",
                templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitQKVParallelLinear(Nncase.TIR.NTT.QKVParallelLinear qkv, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 16 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer qWeight ||
                args[2] is not TIR.Buffer kWeight ||
                args[3] is not TIR.Buffer vWeight ||
                args[13] is not TIR.Buffer qOutput ||
                args[14] is not TIR.Buffer kOutput ||
                args[15] is not TIR.Buffer vOutput)
            {
                throw new NotSupportedException("PyNTT QKVParallelLinear codegen expects input, q/k/v weights, optional q/k/v bias and scale values, and q/k/v output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT QKVParallelLinear expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(7).Take(6).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT QKVParallelLinear codegen currently supports only None input/weight scales.");
            }

            var qBias = GetOptionalBuffer(4, "q bias");
            var kBias = GetOptionalBuffer(5, "k bias");
            var vBias = GetOptionalBuffer(6, "v bias");
            SetComputeOp("qkv_parallel_linear");
            var inputShape = GetBufferActiveShape(input);
            var qWeightShape = GetBufferActiveShape(qWeight);
            var kWeightShape = GetBufferActiveShape(kWeight);
            var vWeightShape = GetBufferActiveShape(vWeight);
            var qBiasShape = qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(qBias);
            var kBiasShape = kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(kBias);
            var vBiasShape = vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(vBias);
            var qOutputShape = GetBufferActiveShape(qOutput);
            var kOutputShape = GetBufferActiveShape(kOutput);
            var vOutputShape = GetBufferActiveShape(vOutput);
            ValidateMinimumRank("PyNTT QKVParallelLinear input", inputShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear q weight", qWeightShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear k weight", kWeightShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear v weight", vWeightShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear q output", qOutputShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear k output", kOutputShape, 2);
            ValidateMinimumRank("PyNTT QKVParallelLinear v output", vOutputShape, 2);

            foreach (var (name, buffer) in new[]
            {
                ("input", input),
                ("q weight", qWeight),
                ("k weight", kWeight),
                ("v weight", vWeight),
                ("q output", qOutput),
                ("k output", kOutput),
                ("v output", vOutput),
            })
            {
                var lanes = GetVectorLanes(buffer.ElemType);
                if (lanes.Length != 0)
                {
                    throw new NotSupportedException($"PyNTT QKVParallelLinear currently expects scalar {name} operands, got lanes [{string.Join(",", lanes)}].");
                }
            }

            ValidateProjectionShape("q", inputShape, qWeightShape, qOutputShape);
            ValidateProjectionShape("k", inputShape, kWeightShape, kOutputShape);
            ValidateProjectionShape("v", inputShape, vWeightShape, vOutputShape);
            ValidateBiasShape("q", qBiasShape, qOutputShape);
            ValidateBiasShape("k", kBiasShape, kOutputShape);
            ValidateBiasShape("v", vBiasShape, vOutputShape);
            ValidateBroadcastable("PyNTT QKVParallelLinear input/q output batch", inputShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear input/k output batch", inputShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear input/v output batch", inputShape[..^2], vOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear q weight/q output batch", qWeightShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear k weight/k output batch", kWeightShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT QKVParallelLinear v weight/v output batch", vWeightShape[..^2], vOutputShape[..^2]);

            if (input.ElemType != qWeight.ElemType || qWeight.ElemType != kWeight.ElemType || kWeight.ElemType != vWeight.ElemType)
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear expects input and q/k/v weights to have the same dtype, got input={input.ElemType}, q={qWeight.ElemType}, k={kWeight.ElemType}, v={vWeight.ElemType}.");
            }

            if (qOutput.ElemType != kOutput.ElemType || kOutput.ElemType != vOutput.ElemType)
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear expects q/k/v outputs to have the same dtype, got q={qOutput.ElemType}, k={kOutput.ElemType}, v={vOutput.ElemType}.");
            }

            var biasDType = qBias?.ElemType ?? kBias?.ElemType ?? vBias?.ElemType ?? qOutput.ElemType;
            _attrs["op"] = "qkv_parallel_linear";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(qOutput.ElemType);
            _attrs["has_bias"] = qBias is not null || kBias is not null || vBias is not null;
            _attrs["q_shape"] = qOutputShape;
            _attrs["k_shape"] = kOutputShape;
            _attrs["v_shape"] = vOutputShape;
            _attrs["num_heads"] = qkv.NumHeads;
            _attrs["num_kv_heads"] = qkv.NumKvHeads;
            var useGemv = IsGemvMatmul(qOutputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "qkv_gemv_compute" : "qkv_matmul_compute");
            var templateModel = new PyNTTQKVParallelLinearTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(qWeight),
                GetBufferScalarPointer(kWeight),
                GetBufferScalarPointer(vWeight),
                qBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(qBias),
                kBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(kBias),
                vBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(vBias),
                GetBufferScalarPointer(qOutput),
                GetBufferScalarPointer(kOutput),
                GetBufferScalarPointer(vOutput),
                qBias is not null,
                kBias is not null,
                vBias is not null,
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(qWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(qOutput.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(qWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(qOutput.ElemType),
                inputShape,
                qWeightShape,
                kWeightShape,
                vWeightShape,
                qBiasShape,
                kBiasShape,
                vBiasShape,
                qOutputShape,
                kOutputShape,
                vOutputShape,
                GetBufferStrides(input),
                GetBufferStrides(qWeight),
                GetBufferStrides(kWeight),
                GetBufferStrides(vWeight),
                qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(qBias),
                kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(kBias),
                vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(vBias),
                GetBufferStrides(qOutput),
                GetBufferStrides(kOutput),
                GetBufferStrides(vOutput),
                GetHierarchy(qOutput),
                $"{input.Name}, ({qWeight.Name}, {kWeight.Name}, {vWeight.Name}) -> ({qOutput.Name}, {kOutput.Name}, {vOutput.Name})");

            WriteHelperTemplate("triton/kernels/QKVParallelLinear.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitPackedQKVParallelLinear(Nncase.TIR.NTT.PackedQKVParallelLinear qkv, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 16 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer qWeight ||
                args[2] is not TIR.Buffer kWeight ||
                args[3] is not TIR.Buffer vWeight ||
                args[13] is not TIR.Buffer qOutput ||
                args[14] is not TIR.Buffer kOutput ||
                args[15] is not TIR.Buffer vOutput)
            {
                throw new NotSupportedException("PyNTT PackedQKVParallelLinear codegen expects input, packed q/k/v weights, optional packed q/k/v bias and scale values, and packed q/k/v output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(7).Take(6).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT PackedQKVParallelLinear codegen currently supports only None input/weight scales.");
            }

            var qBias = GetOptionalBuffer(4, "q bias");
            var kBias = GetOptionalBuffer(5, "k bias");
            var vBias = GetOptionalBuffer(6, "v bias");
            SetComputeOp("packed_qkv_parallel_linear");
            var inputShape = GetBufferActiveShape(input);
            var qWeightShape = GetBufferActiveShape(qWeight);
            var kWeightShape = GetBufferActiveShape(kWeight);
            var vWeightShape = GetBufferActiveShape(vWeight);
            var qBiasShape = qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(qBias);
            var kBiasShape = kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(kBias);
            var vBiasShape = vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(vBias);
            var qOutputShape = GetBufferActiveShape(qOutput);
            var kOutputShape = GetBufferActiveShape(kOutput);
            var vOutputShape = GetBufferActiveShape(vOutput);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear input", inputShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear q weight", qWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear k weight", kWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear v weight", vWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear q output", qOutputShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear k output", kOutputShape, 2);
            ValidateMinimumRank("PyNTT PackedQKVParallelLinear v output", vOutputShape, 2);

            var inputVectorLanes = GetVectorLanes(input.ElemType);
            if (inputVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects scalar input operands, got lanes [{string.Join(",", inputVectorLanes)}].");
            }

            var qWeightLanes = GetVectorLanes(qWeight.ElemType);
            var kWeightLanes = GetVectorLanes(kWeight.ElemType);
            var vWeightLanes = GetVectorLanes(vWeight.ElemType);
            var qOutputLanes = GetVectorLanes(qOutput.ElemType);
            var kOutputLanes = GetVectorLanes(kOutput.ElemType);
            var vOutputLanes = GetVectorLanes(vOutput.ElemType);
            ValidatePackedQKVLanes("q", qWeightLanes, qOutputLanes);
            ValidatePackedQKVLanes("k", kWeightLanes, kOutputLanes);
            ValidatePackedQKVLanes("v", vWeightLanes, vOutputLanes);
            if (!qWeightLanes.SequenceEqual(kWeightLanes) || !qWeightLanes.SequenceEqual(vWeightLanes))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects q/k/v packed lanes to match, got q=[{string.Join(",", qWeightLanes)}], k=[{string.Join(",", kWeightLanes)}], v=[{string.Join(",", vWeightLanes)}].");
            }

            foreach (var (name, bias) in new[] { ("q", qBias), ("k", kBias), ("v", vBias) })
            {
                if (bias is not null && !GetVectorLanes(bias.ElemType).SequenceEqual(qWeightLanes))
                {
                    throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} bias lanes [{string.Join(",", qWeightLanes)}], got [{string.Join(",", GetVectorLanes(bias.ElemType))}].");
                }
            }

            ValidatePackedProjectionShape("q", inputShape, qWeightShape, qOutputShape);
            ValidatePackedProjectionShape("k", inputShape, kWeightShape, kOutputShape);
            ValidatePackedProjectionShape("v", inputShape, vWeightShape, vOutputShape);
            ValidatePackedBiasShape("q", qBiasShape, qOutputShape);
            ValidatePackedBiasShape("k", kBiasShape, kOutputShape);
            ValidatePackedBiasShape("v", vBiasShape, vOutputShape);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear input/q output batch", inputShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear input/k output batch", inputShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear input/v output batch", inputShape[..^2], vOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear q weight/q output batch", qWeightShape[..^2], qOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear k weight/k output batch", kWeightShape[..^2], kOutputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedQKVParallelLinear v weight/v output batch", vWeightShape[..^2], vOutputShape[..^2]);

            if (input.ElemType != GetScalarDataType(qWeight.ElemType) ||
                GetScalarDataType(qWeight.ElemType) != GetScalarDataType(kWeight.ElemType) ||
                GetScalarDataType(kWeight.ElemType) != GetScalarDataType(vWeight.ElemType))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects input and packed q/k/v weights to have the same scalar dtype, got input={input.ElemType}, q={qWeight.ElemType}, k={kWeight.ElemType}, v={vWeight.ElemType}.");
            }

            if (GetScalarDataType(qOutput.ElemType) != GetScalarDataType(kOutput.ElemType) ||
                GetScalarDataType(kOutput.ElemType) != GetScalarDataType(vOutput.ElemType))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects q/k/v outputs to have the same scalar dtype, got q={qOutput.ElemType}, k={kOutput.ElemType}, v={vOutput.ElemType}.");
            }

            var biasDType = qBias?.ElemType ?? kBias?.ElemType ?? vBias?.ElemType ?? qOutput.ElemType;
            var nPackedLaneCount = qWeightLanes[0];
            var nVectorLaneCount = qWeightLanes[1];
            _attrs["op"] = "packed_qkv_parallel_linear";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(qOutput.ElemType);
            _attrs["has_bias"] = qBias is not null || kBias is not null || vBias is not null;
            _attrs["q_shape"] = qOutputShape;
            _attrs["k_shape"] = kOutputShape;
            _attrs["v_shape"] = vOutputShape;
            _attrs["n_pack_lane"] = nPackedLaneCount;
            _attrs["n_lane"] = nVectorLaneCount;
            _attrs["n_scalar_lane"] = checked(nPackedLaneCount * nVectorLaneCount);
            _attrs["num_heads"] = qkv.NumHeads;
            _attrs["num_kv_heads"] = qkv.NumKvHeads;
            var qLogicalOutputShape = qOutputShape.ToArray();
            qLogicalOutputShape[^1] = MultiplyDim(qLogicalOutputShape[^1], checked(nPackedLaneCount * nVectorLaneCount));
            var useGemv = IsGemvMatmul(qLogicalOutputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "packed_qkv_gemv_compute" : "packed_qkv_matmul_compute");
            var templateModel = new PyNTTQKVParallelLinearTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(qWeight),
                GetBufferScalarPointer(kWeight),
                GetBufferScalarPointer(vWeight),
                qBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(qBias),
                kBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(kBias),
                vBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(vBias),
                GetBufferScalarPointer(qOutput),
                GetBufferScalarPointer(kOutput),
                GetBufferScalarPointer(vOutput),
                qBias is not null,
                kBias is not null,
                vBias is not null,
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(qWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(qOutput.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(qWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(qOutput.ElemType),
                inputShape,
                qWeightShape,
                kWeightShape,
                vWeightShape,
                qBiasShape,
                kBiasShape,
                vBiasShape,
                qOutputShape,
                kOutputShape,
                vOutputShape,
                GetBufferStrides(input),
                GetBufferStrides(qWeight),
                GetBufferStrides(kWeight),
                GetBufferStrides(vWeight),
                qBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(qBias),
                kBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(kBias),
                vBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(vBias),
                GetBufferStrides(qOutput),
                GetBufferStrides(kOutput),
                GetBufferStrides(vOutput),
                GetHierarchy(qOutput),
                $"{input.Name}, packed({qWeight.Name}, {kWeight.Name}, {vWeight.Name}) -> packed({qOutput.Name}, {kOutput.Name}, {vOutput.Name})")
            {
                PackedN = true,
                NPackedLaneCount = nPackedLaneCount,
                NVectorLaneCount = nVectorLaneCount,
            };

            WriteHelperTemplate("triton/kernels/PackedQKVParallelLinear.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitMatMulGlu(Nncase.TIR.NTT.MatMulGlu matMulGlu, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 10 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer gateWeight ||
                args[2] is not TIR.Buffer upWeight ||
                args[9] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT MatMulGlu codegen expects input, gate/up weights, optional gate/up bias and scale values, and output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT MatMulGlu expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(5).Take(4).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT MatMulGlu codegen currently supports only None input/weight scales.");
            }

            var gateBias = GetOptionalBuffer(3, "gate bias");
            var upBias = GetOptionalBuffer(4, "up bias");
            SetComputeOp("matmul_glu");
            var inputShape = GetBufferActiveShape(input);
            var gateWeightShape = GetBufferActiveShape(gateWeight);
            var upWeightShape = GetBufferActiveShape(upWeight);
            var gateBiasShape = gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(gateBias);
            var upBiasShape = upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(upBias);
            var outputShape = GetBufferActiveShape(output);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var gateWeightGlobalShape = GetBufferGlobalShape(gateWeight);
            var upWeightGlobalShape = GetBufferGlobalShape(upWeight);
            var gateBiasGlobalShape = gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferGlobalShape(gateBias);
            var upBiasGlobalShape = upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferGlobalShape(upBias);
            var outputGlobalShape = GetBufferGlobalShape(output);
            var inputSplitAxes = GetBufferSplitAxes(input, inputGlobalShape.Length);
            var gateWeightSplitAxes = GetBufferSplitAxes(gateWeight, gateWeightGlobalShape.Length);
            var upWeightSplitAxes = GetBufferSplitAxes(upWeight, upWeightGlobalShape.Length);
            var gateBiasSplitAxes = gateBias is null ? Array.Empty<int[]>() : GetBufferSplitAxes(gateBias, gateBiasGlobalShape.Length);
            var upBiasSplitAxes = upBias is null ? Array.Empty<int[]>() : GetBufferSplitAxes(upBias, upBiasGlobalShape.Length);
            var outputSplitAxes = GetBufferSplitAxes(output, outputGlobalShape.Length);
            ValidateMinimumRank("PyNTT MatMulGlu input", inputShape, 2);
            ValidateMinimumRank("PyNTT MatMulGlu gate weight", gateWeightShape, 2);
            ValidateMinimumRank("PyNTT MatMulGlu up weight", upWeightShape, 2);
            ValidateMinimumRank("PyNTT MatMulGlu output", outputShape, 2);

            foreach (var (name, buffer) in new[]
            {
                ("input", input),
                ("gate weight", gateWeight),
                ("up weight", upWeight),
                ("output", output),
            })
            {
                var lanes = GetVectorLanes(buffer.ElemType);
                if (lanes.Length != 0)
                {
                    throw new NotSupportedException($"PyNTT MatMulGlu currently expects scalar {name} operands, got lanes [{string.Join(",", lanes)}].");
                }
            }

            ValidateMatMulGluProjectionShape("gate", inputShape, gateWeightShape, outputShape, inputGlobalShape, gateWeightGlobalShape, outputGlobalShape, inputSplitAxes, gateWeightSplitAxes, outputSplitAxes, packed: false);
            ValidateMatMulGluProjectionShape("up", inputShape, upWeightShape, outputShape, inputGlobalShape, upWeightGlobalShape, outputGlobalShape, inputSplitAxes, upWeightSplitAxes, outputSplitAxes, packed: false);
            ValidateMatMulGluBiasShape("gate", gateBiasShape, outputShape, gateBiasGlobalShape, outputGlobalShape, gateBiasSplitAxes, outputSplitAxes, packed: false);
            ValidateMatMulGluBiasShape("up", upBiasShape, outputShape, upBiasGlobalShape, outputGlobalShape, upBiasSplitAxes, outputSplitAxes, packed: false);
            ValidateBroadcastable("PyNTT MatMulGlu input/output batch", inputShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT MatMulGlu gate weight/output batch", gateWeightShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT MatMulGlu up weight/output batch", upWeightShape[..^2], outputShape[..^2]);

            if (input.ElemType != gateWeight.ElemType || gateWeight.ElemType != upWeight.ElemType)
            {
                throw new NotSupportedException($"PyNTT MatMulGlu expects input and gate/up weights to have the same dtype, got input={input.ElemType}, gate={gateWeight.ElemType}, up={upWeight.ElemType}.");
            }

            var biasDType = gateBias?.ElemType ?? upBias?.ElemType ?? output.ElemType;
            _attrs["op"] = "matmul_glu";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["has_bias"] = gateBias is not null || upBias is not null;
            _attrs["shape"] = outputShape;
            _attrs["glu_type"] = GetGluTypeName(matMulGlu.GluType);
            var useGemv = IsGemvMatmul(outputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "matmul_glu_gemv_compute" : "matmul_glu_matmul_compute");
            var templateModel = new PyNTTMatMulGluTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(gateWeight),
                GetBufferScalarPointer(upWeight),
                gateBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(gateBias),
                upBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(upBias),
                GetBufferScalarPointer(output),
                gateBias is not null,
                upBias is not null,
                GetGluTypeName(matMulGlu.GluType),
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(gateWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(output.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(gateWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(output.ElemType),
                inputShape,
                gateWeightShape,
                upWeightShape,
                gateBiasShape,
                upBiasShape,
                outputShape,
                GetBufferStrides(input),
                GetBufferStrides(gateWeight),
                GetBufferStrides(upWeight),
                gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(gateBias),
                upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(upBias),
                GetBufferStrides(output),
                GetHierarchy(output),
                $"{input.Name}, ({gateWeight.Name}, {upWeight.Name}) -> {output.Name}")
            {
                InputGlobalShape = inputGlobalShape,
                GateWeightGlobalShape = gateWeightGlobalShape,
                UpWeightGlobalShape = upWeightGlobalShape,
                GateBiasGlobalShape = gateBiasGlobalShape,
                UpBiasGlobalShape = upBiasGlobalShape,
                OutputGlobalShape = outputGlobalShape,
                InputSplitAxes = inputSplitAxes,
                GateWeightSplitAxes = gateWeightSplitAxes,
                UpWeightSplitAxes = upWeightSplitAxes,
                GateBiasSplitAxes = gateBiasSplitAxes,
                UpBiasSplitAxes = upBiasSplitAxes,
                OutputSplitAxes = outputSplitAxes,
            };

            WriteHelperTemplate("triton/kernels/MatMulGlu.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitPackedMatMulGlu(Nncase.TIR.NTT.PackedMatMulGlu matMulGlu, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 10 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer gateWeight ||
                args[2] is not TIR.Buffer upWeight ||
                args[9] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT PackedMatMulGlu codegen expects input, packed gate/up weights, optional packed gate/up bias and scale values, and packed output TIR buffers.");
            }

            TIR.Buffer? GetOptionalBuffer(int index, string name) => args[index] switch
            {
                TIR.Buffer buffer => buffer,
                None => null,
                _ => throw new NotSupportedException($"PyNTT PackedMatMulGlu expects {name} to be a TIR buffer or None, got {args[index].GetType().Name}."),
            };

            if (args.Skip(5).Take(4).Any(arg => arg is not None))
            {
                throw new NotSupportedException("PyNTT PackedMatMulGlu codegen currently supports only None input/weight scales.");
            }

            var gateBias = GetOptionalBuffer(3, "gate bias");
            var upBias = GetOptionalBuffer(4, "up bias");
            SetComputeOp("packed_matmul_glu");
            var inputShape = GetBufferActiveShape(input);
            var gateWeightShape = GetBufferActiveShape(gateWeight);
            var upWeightShape = GetBufferActiveShape(upWeight);
            var gateBiasShape = gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(gateBias);
            var upBiasShape = upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferActiveShape(upBias);
            var outputShape = GetBufferActiveShape(output);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var gateWeightGlobalShape = GetBufferGlobalShape(gateWeight);
            var upWeightGlobalShape = GetBufferGlobalShape(upWeight);
            var gateBiasGlobalShape = gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferGlobalShape(gateBias);
            var upBiasGlobalShape = upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferGlobalShape(upBias);
            var outputGlobalShape = GetBufferGlobalShape(output);
            var inputSplitAxes = GetBufferSplitAxes(input, inputGlobalShape.Length);
            var gateWeightSplitAxes = GetBufferSplitAxes(gateWeight, gateWeightGlobalShape.Length);
            var upWeightSplitAxes = GetBufferSplitAxes(upWeight, upWeightGlobalShape.Length);
            var gateBiasSplitAxes = gateBias is null ? Array.Empty<int[]>() : GetBufferSplitAxes(gateBias, gateBiasGlobalShape.Length);
            var upBiasSplitAxes = upBias is null ? Array.Empty<int[]>() : GetBufferSplitAxes(upBias, upBiasGlobalShape.Length);
            var outputSplitAxes = GetBufferSplitAxes(output, outputGlobalShape.Length);
            ValidateMinimumRank("PyNTT PackedMatMulGlu input", inputShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMulGlu gate weight", gateWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMulGlu up weight", upWeightShape, 2);
            ValidateMinimumRank("PyNTT PackedMatMulGlu output", outputShape, 2);

            var inputVectorLanes = GetVectorLanes(input.ElemType);
            if (inputVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT PackedMatMulGlu expects scalar input operands, got lanes [{string.Join(",", inputVectorLanes)}].");
            }

            var gateWeightLanes = GetVectorLanes(gateWeight.ElemType);
            var upWeightLanes = GetVectorLanes(upWeight.ElemType);
            var outputLanes = GetVectorLanes(output.ElemType);
            ValidatePackedQKVLanes("gate", gateWeightLanes, outputLanes);
            ValidatePackedQKVLanes("up", upWeightLanes, outputLanes);
            if (!gateWeightLanes.SequenceEqual(upWeightLanes))
            {
                throw new NotSupportedException($"PyNTT PackedMatMulGlu expects gate/up packed lanes to match, got gate=[{string.Join(",", gateWeightLanes)}], up=[{string.Join(",", upWeightLanes)}].");
            }

            foreach (var (name, bias) in new[] { ("gate", gateBias), ("up", upBias) })
            {
                if (bias is not null && !GetVectorLanes(bias.ElemType).SequenceEqual(gateWeightLanes))
                {
                    throw new NotSupportedException($"PyNTT PackedMatMulGlu expects {name} bias lanes [{string.Join(",", gateWeightLanes)}], got [{string.Join(",", GetVectorLanes(bias.ElemType))}].");
                }
            }

            ValidateMatMulGluProjectionShape("gate", inputShape, gateWeightShape, outputShape, inputGlobalShape, gateWeightGlobalShape, outputGlobalShape, inputSplitAxes, gateWeightSplitAxes, outputSplitAxes, packed: true);
            ValidateMatMulGluProjectionShape("up", inputShape, upWeightShape, outputShape, inputGlobalShape, upWeightGlobalShape, outputGlobalShape, inputSplitAxes, upWeightSplitAxes, outputSplitAxes, packed: true);
            ValidateMatMulGluBiasShape("gate", gateBiasShape, outputShape, gateBiasGlobalShape, outputGlobalShape, gateBiasSplitAxes, outputSplitAxes, packed: true);
            ValidateMatMulGluBiasShape("up", upBiasShape, outputShape, upBiasGlobalShape, outputGlobalShape, upBiasSplitAxes, outputSplitAxes, packed: true);
            ValidateBroadcastable("PyNTT PackedMatMulGlu input/output batch", inputShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedMatMulGlu gate weight/output batch", gateWeightShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT PackedMatMulGlu up weight/output batch", upWeightShape[..^2], outputShape[..^2]);

            if (input.ElemType != GetScalarDataType(gateWeight.ElemType) ||
                GetScalarDataType(gateWeight.ElemType) != GetScalarDataType(upWeight.ElemType))
            {
                throw new NotSupportedException($"PyNTT PackedMatMulGlu expects input and packed gate/up weights to have the same scalar dtype, got input={input.ElemType}, gate={gateWeight.ElemType}, up={upWeight.ElemType}.");
            }

            var biasDType = gateBias?.ElemType ?? upBias?.ElemType ?? output.ElemType;
            var nPackedLaneCount = gateWeightLanes[0];
            var nVectorLaneCount = gateWeightLanes[1];
            _attrs["op"] = "packed_matmul_glu";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["has_bias"] = gateBias is not null || upBias is not null;
            _attrs["shape"] = outputShape;
            _attrs["n_pack_lane"] = nPackedLaneCount;
            _attrs["n_lane"] = nVectorLaneCount;
            _attrs["n_scalar_lane"] = checked(nPackedLaneCount * nVectorLaneCount);
            _attrs["glu_type"] = GetGluTypeName(matMulGlu.GluType);
            var logicalOutputShape = outputShape.ToArray();
            logicalOutputShape[^1] = MultiplyDim(logicalOutputShape[^1], checked(nPackedLaneCount * nVectorLaneCount));
            var useGemv = IsGemvMatmul(logicalOutputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "packed_matmul_glu_gemv_compute" : "packed_matmul_glu_matmul_compute");
            var templateModel = new PyNTTMatMulGluTemplateModel(
                helperName,
                GetBufferScalarPointer(input),
                GetBufferScalarPointer(gateWeight),
                GetBufferScalarPointer(upWeight),
                gateBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(gateBias),
                upBias is null ? new PyNTTBufferPointerTemplateModel("None") : GetBufferScalarPointer(upBias),
                GetBufferScalarPointer(output),
                gateBias is not null,
                upBias is not null,
                GetGluTypeName(matMulGlu.GluType),
                GetPyNTTScalarDTypeName(input.ElemType),
                GetPyNTTScalarDTypeName(gateWeight.ElemType),
                GetPyNTTScalarDTypeName(biasDType),
                GetPyNTTScalarDTypeName(output.ElemType),
                GetScalarTritonDType(input.ElemType),
                GetScalarTritonDType(gateWeight.ElemType),
                GetScalarTritonDType(biasDType),
                GetScalarTritonDType(output.ElemType),
                inputShape,
                gateWeightShape,
                upWeightShape,
                gateBiasShape,
                upBiasShape,
                outputShape,
                GetBufferStrides(input),
                GetBufferStrides(gateWeight),
                GetBufferStrides(upWeight),
                gateBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(gateBias),
                upBias is null ? Array.Empty<PyNTTDimExpression>() : GetBufferStrides(upBias),
                GetBufferStrides(output),
                GetHierarchy(output),
                $"{input.Name}, packed({gateWeight.Name}, {upWeight.Name}) -> packed({output.Name})")
            {
                PackedN = true,
                NPackedLaneCount = nPackedLaneCount,
                NVectorLaneCount = nVectorLaneCount,
                InputGlobalShape = inputGlobalShape,
                GateWeightGlobalShape = gateWeightGlobalShape,
                UpWeightGlobalShape = upWeightGlobalShape,
                GateBiasGlobalShape = gateBiasGlobalShape,
                UpBiasGlobalShape = upBiasGlobalShape,
                OutputGlobalShape = outputGlobalShape,
                InputSplitAxes = inputSplitAxes,
                GateWeightSplitAxes = gateWeightSplitAxes,
                UpWeightSplitAxes = upWeightSplitAxes,
                GateBiasSplitAxes = gateBiasSplitAxes,
                UpBiasSplitAxes = upBiasSplitAxes,
                OutputSplitAxes = outputSplitAxes,
            };

            WriteHelperTemplate("triton/kernels/PackedMatMulGlu.py.jinja", templateModel);
            WriteLine(BuildHelperCall(helperName));
        }

        private static void ValidateMatMulGluProjectionShape(
            string name,
            PyNTTDimExpression[] inputShape,
            PyNTTDimExpression[] weightShape,
            PyNTTDimExpression[] outputShape,
            PyNTTDimExpression[] inputGlobalShape,
            PyNTTDimExpression[] weightGlobalShape,
            PyNTTDimExpression[] outputGlobalShape,
            int[][] inputSplitAxes,
            int[][] weightSplitAxes,
            int[][] outputSplitAxes,
            bool packed)
        {
            var inputGlobalK = inputGlobalShape[^1];
            var inputGlobalM = inputGlobalShape[^2];
            var weightGlobalK = packed ? weightGlobalShape[^1] : weightGlobalShape[^2];
            var weightGlobalN = packed ? weightGlobalShape[^2] : weightGlobalShape[^1];
            var outputGlobalM = outputGlobalShape[^2];
            var outputGlobalN = outputGlobalShape[^1];
            if (!SameDim(inputGlobalK, weightGlobalK) ||
                !SameDim(outputGlobalM, inputGlobalM) ||
                !SameDim(outputGlobalN, weightGlobalN))
            {
                var weightLayout = packed ? "weight=[...,N,K]<Nr,lane>" : "weight=[...,K,N]";
                var outputLayout = packed ? "output=[...,M,N]<Nr,lane>" : "output=[...,M,N]";
                throw new NotSupportedException($"PyNTT MatMulGlu {name} projection expects compatible global shapes input=[...,M,K], {weightLayout}, {outputLayout}, got input=[{ShapeText(inputGlobalShape)}], weight=[{ShapeText(weightGlobalShape)}], output=[{ShapeText(outputGlobalShape)}].");
            }

            ValidateMatMulGluShardAxis(
                $"PyNTT MatMulGlu {name} input M",
                inputShape[^2],
                inputSplitAxes[^2],
                outputShape[^2],
                outputSplitAxes[^2]);
            ValidateMatMulGluShardAxis(
                $"PyNTT MatMulGlu {name} weight K",
                packed ? weightShape[^1] : weightShape[^2],
                packed ? weightSplitAxes[^1] : weightSplitAxes[^2],
                inputShape[^1],
                inputSplitAxes[^1]);
            ValidateMatMulGluShardAxis(
                $"PyNTT MatMulGlu {name} weight N",
                packed ? weightShape[^2] : weightShape[^1],
                packed ? weightSplitAxes[^2] : weightSplitAxes[^1],
                outputShape[^1],
                outputSplitAxes[^1]);
        }

        private static void ValidateMatMulGluBiasShape(
            string name,
            PyNTTDimExpression[] biasShape,
            PyNTTDimExpression[] outputShape,
            PyNTTDimExpression[] biasGlobalShape,
            PyNTTDimExpression[] outputGlobalShape,
            int[][] biasSplitAxes,
            int[][] outputSplitAxes,
            bool packed)
        {
            if (biasShape.Length == 0)
            {
                return;
            }

            ValidateRank($"PyNTT MatMulGlu {name} bias", biasShape, 1);
            if (!SameDim(biasGlobalShape[^1], outputGlobalShape[^1]))
            {
                var outputLayout = packed ? "packed output N" : "output N";
                throw new NotSupportedException($"PyNTT MatMulGlu {name} bias last dimension should match global {outputLayout}, got bias=[{ShapeText(biasGlobalShape)}], output=[{ShapeText(outputGlobalShape)}].");
            }

            ValidateMatMulGluShardAxis(
                $"PyNTT MatMulGlu {name} bias N",
                biasShape[^1],
                biasSplitAxes[^1],
                outputShape[^1],
                outputSplitAxes[^1]);
        }

        private static void ValidateMatMulGluShardAxis(
            string context,
            PyNTTDimExpression tensorLocal,
            int[] tensorSplitAxes,
            PyNTTDimExpression canonicalLocal,
            int[] canonicalSplitAxes)
        {
            if (tensorSplitAxes.Length == 0)
            {
                return;
            }

            if (!tensorSplitAxes.SequenceEqual(canonicalSplitAxes))
            {
                if (IsPrefix(tensorSplitAxes, canonicalSplitAxes))
                {
                    return;
                }

                throw new NotSupportedException($"{context} split axes [{string.Join(",", tensorSplitAxes)}] must either match canonical split axes [{string.Join(",", canonicalSplitAxes)}], be a prefix of them, or be broadcast.");
            }

            if (!SameDim(tensorLocal, canonicalLocal))
            {
                throw new NotSupportedException($"{context} local extent should match canonical local extent when split axes match, got local={tensorLocal.PythonExpression}, canonical={canonicalLocal.PythonExpression}.");
            }
        }

        private static bool IsPrefix(int[] prefix, int[] values)
        {
            if (prefix.Length > values.Length)
            {
                return false;
            }

            for (int i = 0; i < prefix.Length; i++)
            {
                if (prefix[i] != values[i])
                {
                    return false;
                }
            }

            return true;
        }

        private static void ValidateBiasShape(string name, PyNTTDimExpression[] biasShape, PyNTTDimExpression[] outputShape)
        {
            if (biasShape.Length == 0)
            {
                return;
            }

            ValidateRank($"PyNTT QKVParallelLinear {name} bias", biasShape, 1);
            if (!SameDim(biasShape[^1], outputShape[^1]))
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear {name} bias last dimension should match output N, got bias=[{ShapeText(biasShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidateProjectionShape(string name, PyNTTDimExpression[] inputShape, PyNTTDimExpression[] weightShape, PyNTTDimExpression[] outputShape)
        {
            var inputK = inputShape[^1];
            var inputM = inputShape[^2];
            var weightK = weightShape[^2];
            var weightN = weightShape[^1];
            if (!SameDim(inputK, weightK) ||
                !SameDim(outputShape[^2], inputM) ||
                !SameDim(outputShape[^1], weightN))
            {
                throw new NotSupportedException($"PyNTT QKVParallelLinear {name} projection expects input=[...,M,K], weight=[...,K,N], output=[...,M,N], got input=[{ShapeText(inputShape)}], weight=[{ShapeText(weightShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidatePackedQKVLanes(string name, IReadOnlyList<int> weightLanes, IReadOnlyList<int> outputLanes)
        {
            if (weightLanes.Count != 2)
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} weight lanes [Nr,lane], got [{string.Join(",", weightLanes)}].");
            }

            if (!weightLanes.SequenceEqual(outputLanes))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear expects {name} output lanes [{string.Join(",", weightLanes)}], got [{string.Join(",", outputLanes)}].");
            }
        }

        private static void ValidatePackedBiasShape(string name, PyNTTDimExpression[] biasShape, PyNTTDimExpression[] outputShape)
        {
            if (biasShape.Length == 0)
            {
                return;
            }

            ValidateRank($"PyNTT PackedQKVParallelLinear {name} bias", biasShape, 1);
            if (!SameDim(biasShape[^1], outputShape[^1]))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear {name} bias last dimension should match packed output N, got bias=[{ShapeText(biasShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private static void ValidatePackedProjectionShape(string name, PyNTTDimExpression[] inputShape, PyNTTDimExpression[] weightShape, PyNTTDimExpression[] outputShape)
        {
            var inputK = inputShape[^1];
            var inputM = inputShape[^2];
            var weightN = weightShape[^2];
            var weightK = weightShape[^1];
            if (!SameDim(inputK, weightK) ||
                !SameDim(outputShape[^2], inputM) ||
                !SameDim(outputShape[^1], weightN))
            {
                throw new NotSupportedException($"PyNTT PackedQKVParallelLinear {name} projection expects input=[...,M,K], weight=[...,N,K]<Nr,lane>, output=[...,M,N]<Nr,lane>, got input=[{ShapeText(inputShape)}], weight=[{ShapeText(weightShape)}], output=[{ShapeText(outputShape)}].");
            }
        }

        private void VisitSUMMA(Nncase.TIR.NTT.SUMMA summa, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT SUMMA codegen expects lhs, rhs, and output TIR buffers.");
            }

            if (summa.TransposeA || summa.TransposeB)
            {
                throw new NotSupportedException("PyNTT SUMMA codegen does not support transposed inputs yet.");
            }

            if (GetScalarBool(args[3], "PyNTT SUMMA loadC"))
            {
                throw new NotSupportedException("PyNTT SUMMA codegen does not support loadC yet.");
            }

            SetComputeOp("matmul");
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            var outputShape = GetBufferShape(output);
            var lhsGlobalShape = GetBufferGlobalShape(lhs);
            var rhsGlobalShape = GetBufferGlobalShape(rhs);
            var outputGlobalShape = GetBufferGlobalShape(output);
            ValidateMinimumRank("PyNTT SUMMA lhs", lhsShape, 2);
            ValidateMinimumRank("PyNTT SUMMA rhs", rhsShape, 2);
            ValidateMinimumRank("PyNTT SUMMA output", outputShape, 2);
            if (lhsShape.Length != 2 || rhsShape.Length != 2 || outputShape.Length != 2)
            {
                throw new NotSupportedException($"PyNTT SUMMA currently supports rank-2 matmul only, got lhs rank {lhsShape.Length}, rhs rank {rhsShape.Length}, output rank {outputShape.Length}.");
            }

            var dimInfo = Nncase.IR.NTT.VectorizedMatMul.GetDimInfo(false, false, lhsShape.Length, rhsShape.Length);
            var lhsVectorLanes = GetVectorLanes(lhs.ElemType);
            var rhsVectorLanes = GetVectorLanes(rhs.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var rhsNVectorLaneCount = 1;
            var outputNVectorLaneCount = 1;
            if (!summa.LhsVectorizedAxes.IsDefaultOrEmpty)
            {
                throw new NotSupportedException($"PyNTT SUMMA currently supports only RHS N-axis vectorization, got lhs axes [{string.Join(",", summa.LhsVectorizedAxes)}].");
            }

            if (!summa.RhsVectorizedAxes.IsDefaultOrEmpty)
            {
                if (summa.RhsVectorizedAxes.Count != 1 || summa.RhsVectorizedAxes[0] != dimInfo.Rn || rhsVectorLanes.Length != 1)
                {
                    throw new NotSupportedException($"PyNTT SUMMA currently supports only one RHS N-axis vector lane, got rhs axes [{string.Join(",", summa.RhsVectorizedAxes)}] and lanes [{string.Join(",", rhsVectorLanes)}].");
                }

                rhsNVectorLaneCount = rhsVectorLanes[0];
            }

            if (lhsVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT SUMMA currently supports scalar lhs operands only, got lhs lanes [{string.Join(",", lhsVectorLanes)}].");
            }

            if (outputVectorLanes.Length != 0)
            {
                if (outputVectorLanes.Length != 1 || outputVectorLanes[0] != rhsNVectorLaneCount || rhsNVectorLaneCount == 1)
                {
                    throw new NotSupportedException($"PyNTT SUMMA currently supports only RHS N-axis vectorization producing the same output N lane, got output lanes [{string.Join(",", outputVectorLanes)}] and rhs N lane {rhsNVectorLaneCount}.");
                }

                outputNVectorLaneCount = outputVectorLanes[0];
            }

            var rhsFullGlobalN = MultiplyDim(rhsGlobalShape[^1], rhsNVectorLaneCount);
            var outputGlobalNCompatible = outputNVectorLaneCount == 1
                ? rhsNVectorLaneCount == 1 ? SameDim(outputGlobalShape[^1], rhsGlobalShape[^1]) : CanFitPaddedDim(outputGlobalShape[^1], rhsFullGlobalN)
                : SameDim(outputGlobalShape[^1], rhsGlobalShape[^1]);
            if (!SameDim(lhsGlobalShape[^1], rhsGlobalShape[^2]) ||
                !SameDim(outputGlobalShape[^2], lhsGlobalShape[^2]) ||
                !outputGlobalNCompatible)
            {
                throw new NotSupportedException($"PyNTT SUMMA expects compatible global matrix shapes, got lhs=[{ShapeText(lhsGlobalShape)}], rhs=[{ShapeText(rhsGlobalShape)}], output=[{ShapeText(outputGlobalShape)}].");
            }

            var lhsRef = ResolveBufferRef(lhs);
            var rhsRef = ResolveBufferRef(rhs);
            var outputRef = ResolveBufferRef(output);
            var scale = GetScalarFloat(args[4], "PyNTT SUMMA scale", 1.0f);
            _attrs["op"] = "matmul";
            _attrs["tir"] = true;
            _attrs["summa"] = true;
            _attrs["requires_grid_barrier"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["scale"] = scale;
            var helperName = GetNextHelperName("summa_compute");
            WriteHelperTemplate(
                "triton/kernels/Summa.py.jinja",
                new PyNTTSummaTemplateModel(
                    helperName,
                    lhsRef.BaseName,
                    lhsRef.OffsetBytes,
                    lhsRef.PoolStrideBytes,
                    rhsRef.BaseName,
                    rhsRef.OffsetBytes,
                    rhsRef.PoolStrideBytes,
                    outputRef.BaseName,
                    outputRef.OffsetBytes,
                    outputRef.PoolStrideBytes,
                    GetPyNTTScalarDTypeName(lhs.ElemType),
                    GetPyNTTScalarDTypeName(rhs.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(lhs.ElemType),
                    GetScalarTritonDType(rhs.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    outputShape,
                    lhsGlobalShape,
                    rhsGlobalShape,
                    outputGlobalShape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    GetBufferSplitAxes(lhs, lhsShape.Length),
                    GetBufferSplitAxes(rhs, rhsShape.Length),
                    GetBufferSplitAxes(output, outputShape.Length),
                    GetHierarchy(output),
                    rhsNVectorLaneCount,
                    outputNVectorLaneCount,
                    scale.ToString("R", CultureInfo.InvariantCulture),
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteBarrier(HelperBarrierKind.Grid);
            WriteLine(BuildHelperCall(helperName), HelperBarrierKind.Grid);
        }

        private void VisitMatmulLike(
            string context,
            IRArray<int> lhsVectorizedAxes,
            IRArray<int> rhsVectorizedAxes,
            bool transposeA,
            bool transposeB,
            IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException($"{context} codegen expects lhs, rhs, and output TIR buffers.");
            }

            if (GetScalarBool(args[3], $"{context} loadC"))
            {
                throw new NotSupportedException($"{context} codegen does not support loadC yet.");
            }

            SetComputeOp("matmul");
            var lhsShape = GetBufferActiveShape(lhs);
            var rhsShape = GetBufferActiveShape(rhs);
            var outputShape = GetBufferActiveShape(output);
            ValidateMinimumRank($"{context} lhs", lhsShape, 2);
            ValidateMinimumRank($"{context} rhs", rhsShape, 2);
            ValidateMinimumRank($"{context} output", outputShape, 2);
            var dimInfo = Nncase.IR.NTT.VectorizedMatMul.GetDimInfo(transposeA, transposeB, lhsShape.Length, rhsShape.Length);
            var lhsVectorLanes = GetVectorLanes(lhs.ElemType);
            var rhsVectorLanes = GetVectorLanes(rhs.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var rhsNVectorLaneCount = 1;
            var outputNVectorLaneCount = 1;
            if (!lhsVectorizedAxes.IsDefaultOrEmpty)
            {
                throw new NotSupportedException($"{context} currently supports only RHS N-axis vectorization, got lhs axes [{string.Join(",", lhsVectorizedAxes)}].");
            }

            if (!rhsVectorizedAxes.IsDefaultOrEmpty)
            {
                if (rhsVectorizedAxes.Count != 1 || rhsVectorizedAxes[0] != dimInfo.Rn || rhsVectorLanes.Length != 1)
                {
                    throw new NotSupportedException($"{context} currently supports only one RHS N-axis vector lane, got rhs axes [{string.Join(",", rhsVectorizedAxes)}] and lanes [{string.Join(",", rhsVectorLanes)}].");
                }

                rhsNVectorLaneCount = rhsVectorLanes[0];
            }

            if (lhsVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"{context} currently supports scalar lhs operands only, got lhs lanes [{string.Join(",", lhsVectorLanes)}].");
            }

            if (outputVectorLanes.Length != 0)
            {
                if (outputVectorLanes.Length != 1 || outputVectorLanes[0] != rhsNVectorLaneCount || rhsNVectorLaneCount == 1)
                {
                    throw new NotSupportedException($"{context} currently supports only RHS N-axis vectorization producing the same output N lane, got output lanes [{string.Join(",", outputVectorLanes)}] and rhs N lane {rhsNVectorLaneCount}.");
                }

                outputNVectorLaneCount = outputVectorLanes[0];
            }

            var lhsK = transposeA ? lhsShape[^2] : lhsShape[^1];
            var rhsK = transposeB ? rhsShape[^1] : rhsShape[^2];
            var lhsM = transposeA ? lhsShape[^1] : lhsShape[^2];
            var rhsN = transposeB ? rhsShape[^2] : rhsShape[^1];
            var rhsFullN = MultiplyDim(rhsN, rhsNVectorLaneCount);
            var outputNCompatible = outputNVectorLaneCount == 1
                ? rhsNVectorLaneCount == 1 ? SameDim(outputShape[^1], rhsN) : CanFitPaddedDim(outputShape[^1], rhsFullN)
                : SameDim(outputShape[^1], rhsN);
            if (!SameDim(lhsK, rhsK) || !SameDim(outputShape[^2], lhsM) || !outputNCompatible)
            {
                throw new NotSupportedException($"{context} expects compatible matrix shapes, got lhs=[{ShapeText(lhsShape)}], rhs=[{ShapeText(rhsShape)}], output=[{ShapeText(outputShape)}].");
            }

            ValidateBroadcastable($"{context} lhs batch", lhsShape[..^2], outputShape[..^2]);
            ValidateBroadcastable($"{context} rhs batch", rhsShape[..^2], outputShape[..^2]);
            var scale = GetScalarFloat(args[4], $"{context} scale", 1.0f);
            _attrs["op"] = "matmul";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["transpose_a"] = transposeA;
            _attrs["transpose_b"] = transposeB;
            _attrs["scale"] = scale;
            var useGemv = IsGemvMatmul(outputShape);
            if (useGemv)
            {
                _attrs["gemv"] = true;
            }

            var helperName = GetNextHelperName(useGemv ? "gemv_compute" : "matmul_compute");
            WriteHelperTemplate(
                useGemv ? "triton/kernels/Gemv.py.jinja" : "triton/kernels/Matmul.py.jinja",
                new PyNTTMatmulTemplateModel(
                    helperName,
                    GetBufferScalarPointer(lhs),
                    GetBufferScalarPointer(rhs),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(lhs.ElemType),
                    GetPyNTTScalarDTypeName(rhs.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(lhs.ElemType),
                    GetScalarTritonDType(rhs.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    outputShape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    transposeA,
                    transposeB,
                    GetHierarchy(output),
                    rhsNVectorLaneCount,
                    outputNVectorLaneCount,
                    scale.ToString("R", CultureInfo.InvariantCulture),
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private static bool IsGemvMatmul(IReadOnlyList<PyNTTDimExpression> outputShape)
            => outputShape.Count >= 2 && outputShape[^2].MaxValue == 1;

        private void WriteShardReduceHelper(
            TIR.Buffer buffer,
            int[] reduceAxes,
            int vectorLaneCount,
            string scalarDType,
            int[] hierarchy,
            bool broadcast,
            HelperBarrierKind postBarrier,
            string comment)
        {
            var bufferRef = ResolveBufferRef(buffer);
            if (!string.Equals(bufferRef.BaseName, "data", StringComparison.Ordinal))
            {
                throw new NotSupportedException($"PyNTT Matmul shard reduce currently supports data workspace buffers only, got {bufferRef.BaseName}.");
            }

            var localShape = GetBufferShape(buffer);
            var strides = GetBufferStrides(buffer);
            var tritonDType = GetScalarTritonDType(buffer.ElemType);
            var helperKey = BuildShardReduceHelperKey(
                bufferRef.BaseName,
                scalarDType,
                tritonDType,
                localShape,
                strides,
                vectorLaneCount,
                hierarchy,
                reduceAxes,
                broadcast);
            var hasSharedHelper = _sharedHelperRegistry.TryGetName(helperKey, out var helperName);
            if (!hasSharedHelper)
            {
                helperName = _sharedHelperRegistry.Add(helperKey, _function.Name, broadcast ? "shard_broadcast" : "shard_reduce");
                WriteHelperTemplate(
                    "triton/kernels/ShardReduce.py.jinja",
                    new PyNTTShardReduceTemplateModel(
                        helperName,
                        bufferRef.BaseName,
                        scalarDType,
                        tritonDType,
                        localShape,
                        strides,
                        vectorLaneCount,
                        hierarchy,
                        reduceAxes,
                        broadcast,
                        comment));
            }

            WriteLine(
                BuildHelperCall(
                    helperName,
                    BuildRawPythonArgument(bufferRef.PoolStrideBytes),
                    BuildRawPythonArgument(bufferRef.OffsetBytes),
                    BuildRawPythonArgument(bufferRef.OffsetBytes)),
                postBarrier);
        }

        private void VisitReduce(Nncase.TIR.NTT.Reduce reduce, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Reduce codegen expects input and output TIR buffers.");
            }

            if (GetScalarBool(args[2], "reduce loadPrevious"))
            {
                throw new NotSupportedException("PyNTT Reduce codegen does not support loadPrevious yet.");
            }

            EnsureEmpty("PyNTT Reduce vectorized axes", reduce.VectorizedAxes);
            SetComputeOp("reduce");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var axes = NormalizeAxes(reduce.Axes.ToArray(), inputShape.Length, "PyNTT Reduce");
            ValidateReduceShape("PyNTT Reduce", inputShape, outputShape, axes, reduce.KeepDims);
            _attrs["op"] = GetReduceOpName(reduce.ReduceOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axes"] = axes;
            if (axes.Length == 1)
            {
                _attrs["axis"] = axes[0];
            }

            _attrs["keep_dims"] = reduce.KeepDims;
            var helperName = GetNextHelperName("reduce_compute");
            var reduceOp = (string)_attrs["op"];
            var reduceElementCount = Product(axes.Select(axis => inputShape[axis]));
            WriteHelperTemplate(
                "triton/kernels/Reduce.py.jinja",
                new PyNTTReduceTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    axes,
                    reduce.KeepDims,
                    reduceOp,
                    GetReduceInitValue(reduce.ReduceOp),
                    GetReduceUpdateExpression(reduce.ReduceOp),
                    GetReduceFinalizeExpression(reduce.ReduceOp, reduceElementCount),
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitRoPE(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 4 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer cos ||
                args[2] is not TIR.Buffer sin ||
                args[3] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT RoPE codegen expects input, cos, sin, and output TIR buffers.");
            }

            SetComputeOp("rope");
            var inputShape = GetBufferShape(input);
            var cosShape = GetBufferShape(cos);
            var sinShape = GetBufferShape(sin);
            var outputShape = GetBufferShape(output);
            var inputVectorLanes = GetVectorLanes(input.ElemType);
            var cosVectorLanes = GetVectorLanes(cos.ElemType);
            var sinVectorLanes = GetVectorLanes(sin.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var inputVectorLaneCount = GetVectorLaneElementCount(input.ElemType);
            var cosVectorLaneCount = GetVectorLaneElementCount(cos.ElemType);
            var sinVectorLaneCount = GetVectorLaneElementCount(sin.ElemType);
            var outputVectorLaneCount = GetVectorLaneElementCount(output.ElemType);
            var sinCosVectorPackFactor = GetRoPESinCosVectorPackFactor(inputVectorLanes, cosVectorLanes, sinVectorLanes, outputVectorLanes);

            var rotaryAxis = outputShape.Length - 1;
            if (rotaryAxis < 0)
            {
                throw new NotSupportedException($"PyNTT RoPE requires a non-scalar output, got [{ShapeText(outputShape)}].");
            }

            ValidateRoPEShape("PyNTT RoPE", inputShape, cosShape, sinShape, outputShape, rotaryAxis, outputVectorLaneCount, sinCosVectorPackFactor);
            if (GetShardAxis(output) == rotaryAxis || GetShardAxis(input) == rotaryAxis)
            {
                throw new NotSupportedException("PyNTT RoPE codegen does not support sharding along the rotary dimension yet.");
            }

            _attrs["op"] = "rope";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("rope_compute");
            WriteHelperTemplate(
                "triton/kernels/RoPE.py.jinja",
                new PyNTTRoPETemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(cos),
                    GetBufferScalarPointer(sin),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(cos.ElemType),
                    GetPyNTTScalarDTypeName(sin.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(cos.ElemType),
                    GetScalarTritonDType(sin.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    cosShape,
                    sinShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(cos),
                    GetBufferStrides(sin),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    cosVectorLaneCount,
                    sinVectorLaneCount,
                    outputVectorLaneCount,
                    sinCosVectorPackFactor,
                    rotaryAxis,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitLayerNorm(Nncase.TIR.NTT.VectorizedLayerNorm layerNorm, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer scale ||
                args[2] is not TIR.Buffer bias ||
                args[4] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT LayerNorm codegen expects input, scale, bias, postScale, and output TIR buffers.");
            }

            if (args[3] is not None)
            {
                throw new NotSupportedException("PyNTT LayerNorm codegen does not support postScale yet.");
            }

            SetComputeOp("layer_norm");
            var inputShape = GetBufferShape(input);
            var scaleShape = GetBufferShape(scale);
            var biasShape = GetBufferShape(bias);
            var outputShape = GetBufferShape(output);
            var normalizedAxis = NormalizeAxis(layerNorm.Axis, outputShape.Length, "PyNTT LayerNorm");
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT LayerNorm input");
            var scaleVectorLaneCount = GetSingleVectorLaneCount(scale.ElemType, "PyNTT LayerNorm scale");
            var biasVectorLaneCount = GetSingleVectorLaneCount(bias.ElemType, "PyNTT LayerNorm bias");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT LayerNorm output");
            var valueLaneCounts = new[] { inputVectorLaneCount, scaleVectorLaneCount, biasVectorLaneCount, outputVectorLaneCount }
                .Where(lane => lane > 1)
                .Distinct()
                .ToArray();
            if (valueLaneCounts.Length > 1)
            {
                throw new NotSupportedException($"PyNTT LayerNorm expects matching input/scale/bias/output vector lanes, got input={inputVectorLaneCount}, scale={scaleVectorLaneCount}, bias={biasVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            if (valueLaneCounts.Length == 1)
            {
                var vectorizedAxes = layerNorm.VectorizedAxes.IsDefaultOrEmpty ? Array.Empty<int>() : layerNorm.VectorizedAxes.ToArray();
                if (vectorizedAxes.Length != 1 || NormalizeAxis(vectorizedAxes[0], outputShape.Length, "PyNTT LayerNorm vectorized axis") != outputShape.Length - 1)
                {
                    throw new NotSupportedException($"PyNTT LayerNorm currently supports vectorization only on the last normalized axis, got [{string.Join(",", vectorizedAxes)}].");
                }

                if (normalizedAxis > outputShape.Length - 1)
                {
                    throw new NotSupportedException("PyNTT LayerNorm vectorized axis must be inside the normalized dimensions.");
                }
            }

            var logicalInputShape = GetLogicalVectorShape(inputShape, inputVectorLaneCount);
            var logicalScaleShape = GetLogicalVectorShape(scaleShape, scaleVectorLaneCount);
            var logicalBiasShape = GetLogicalVectorShape(biasShape, biasVectorLaneCount);
            var logicalOutputShape = GetLogicalVectorShape(outputShape, outputVectorLaneCount);
            ValidateSameShape("PyNTT LayerNorm", logicalInputShape, logicalOutputShape);
            ValidateLayerNormShape("PyNTT LayerNorm scale", logicalScaleShape, logicalOutputShape, normalizedAxis);
            ValidateLayerNormShape("PyNTT LayerNorm bias", logicalBiasShape, logicalOutputShape, normalizedAxis);
            if ((GetShardAxis(input) is int inputShardAxis && inputShardAxis >= normalizedAxis) ||
                (GetShardAxis(output) is int outputShardAxis && outputShardAxis >= normalizedAxis) ||
                GetShardAxis(scale).HasValue ||
                GetShardAxis(bias).HasValue)
            {
                throw new NotSupportedException("PyNTT LayerNorm codegen does not support sharding along normalized dimensions or sharded scale/bias yet.");
            }

            _attrs["op"] = layerNorm.UseMean ? "layer_norm" : "rms_norm";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalOutputShape;
            _attrs["axis"] = normalizedAxis;
            _attrs["epsilon"] = layerNorm.Epsilon;
            _attrs["use_mean"] = layerNorm.UseMean;
            var helperName = GetNextHelperName("layer_norm_compute");
            WriteHelperTemplate(
                "triton/kernels/LayerNorm.py.jinja",
                new PyNTTLayerNormTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(scale),
                    GetBufferScalarPointer(bias),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(scale.ElemType),
                    GetPyNTTScalarDTypeName(bias.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(scale.ElemType),
                    GetScalarTritonDType(bias.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    scaleShape,
                    biasShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(scale),
                    GetBufferStrides(bias),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    scaleVectorLaneCount,
                    biasVectorLaneCount,
                    outputVectorLaneCount,
                    normalizedAxis,
                    layerNorm.Epsilon,
                    layerNorm.UseMean,
                    $"{input.Name}, {scale.Name}, {bias.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitNormStats(Nncase.TIR.NTT.NormStats normStats, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT NormStats codegen expects input and output TIR buffers.");
            }

            SetComputeOp(normStats.UseMean ? "norm_stats" : "rms_norm_stats");
            var inputShape = GetBufferShape(input);
            var outputShape = GetBufferShape(output);
            var normalizedAxis = NormalizeAxis(normStats.Axis, inputShape.Length, "PyNTT NormStats");
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT NormStats input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT NormStats output");
            if (outputVectorLaneCount != 1)
            {
                throw new NotSupportedException("PyNTT NormStats expects scalar stats output buffer dtype.");
            }

            ValidateNormStatsShape("PyNTT NormStats", inputShape, outputShape, normalizedAxis, normStats.UseMean);

            _attrs["op"] = normStats.UseMean ? "norm_stats" : "rms_norm_stats";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["axis"] = normalizedAxis;
            _attrs["use_mean"] = normStats.UseMean;
            var helperName = GetNextHelperName("norm_stats_compute");
            WriteHelperTemplate(
                "triton/kernels/NormStats.py.jinja",
                new PyNTTNormStatsTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    outputVectorLaneCount,
                    normalizedAxis,
                    normStats.UseMean,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private void VisitNormApply(Nncase.TIR.NTT.NormApply normApply, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5)
            {
                throw new NotSupportedException("PyNTT NormApply codegen expects input, stats, scale, bias, and output TIR buffers.");
            }

            var input = GetBufferOperand(args[0], "PyNTT NormApply input");
            var stats = GetBufferOperand(args[1], "PyNTT NormApply stats");
            var scale = GetBufferOperand(args[2], "PyNTT NormApply scale");
            var bias = GetBufferOperand(args[3], "PyNTT NormApply bias");
            var output = GetBufferOperand(args[4], "PyNTT NormApply output");
            SetComputeOp(normApply.UseMean ? "norm_apply" : "rms_norm_apply");
            var inputShape = GetBufferActiveShape(input);
            var inputGlobalShape = GetBufferGlobalShape(input);
            var statsShape = GetBufferActiveShape(stats);
            var scaleShape = GetBufferActiveShape(scale);
            var biasShape = GetBufferActiveShape(bias);
            var outputShape = GetBufferActiveShape(output);
            var normalizedAxis = NormalizeAxis(normApply.Axis, outputShape.Length, "PyNTT NormApply");
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT NormApply input");
            var statsVectorLaneCount = GetSingleVectorLaneCount(stats.ElemType, "PyNTT NormApply stats");
            var scaleVectorLaneCount = GetSingleVectorLaneCount(scale.ElemType, "PyNTT NormApply scale");
            var biasVectorLaneCount = GetSingleVectorLaneCount(bias.ElemType, "PyNTT NormApply bias");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT NormApply output");
            if (statsVectorLaneCount != 1)
            {
                throw new NotSupportedException("PyNTT NormApply expects scalar stats buffer dtype.");
            }

            if (new[] { inputVectorLaneCount, scaleVectorLaneCount, biasVectorLaneCount, outputVectorLaneCount }.Distinct().Count() != 1)
            {
                throw new NotSupportedException($"PyNTT NormApply expects matching input/scale/bias/output vector lanes, got input={inputVectorLaneCount}, scale={scaleVectorLaneCount}, bias={biasVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            if (inputVectorLaneCount != 1 && normalizedAxis > outputShape.Length - 1)
            {
                throw new NotSupportedException("PyNTT NormApply vectorized axis must be inside the normalized dimensions.");
            }

            var logicalInputShape = GetLogicalVectorShape(inputShape, inputVectorLaneCount);
            var logicalStatsShape = GetLogicalVectorShape(statsShape, statsVectorLaneCount);
            var logicalScaleShape = GetLogicalVectorShape(scaleShape, scaleVectorLaneCount);
            var logicalBiasShape = GetLogicalVectorShape(biasShape, biasVectorLaneCount);
            var logicalOutputShape = GetLogicalVectorShape(outputShape, outputVectorLaneCount);
            ValidateSameShape("PyNTT NormApply", logicalInputShape, logicalOutputShape);
            ValidateNormStatsShape("PyNTT NormApply stats", logicalOutputShape, logicalStatsShape, normalizedAxis, normApply.UseMean);
            ValidateLayerNormShape("PyNTT NormApply scale", logicalScaleShape, logicalOutputShape, normalizedAxis);
            ValidateLayerNormShape("PyNTT NormApply bias", logicalBiasShape, logicalOutputShape, normalizedAxis);

            _attrs["op"] = normApply.UseMean ? "norm_apply" : "rms_norm_apply";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = logicalOutputShape;
            _attrs["axis"] = normalizedAxis;
            _attrs["epsilon"] = normApply.Epsilon;
            _attrs["use_mean"] = normApply.UseMean;
            var helperName = GetNextHelperName("norm_apply_compute");
            WriteHelperTemplate(
                "triton/kernels/NormApply.py.jinja",
                new PyNTTNormApplyTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input),
                    GetBufferScalarPointer(stats),
                    GetBufferScalarPointer(scale),
                    GetBufferScalarPointer(bias),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(input.ElemType),
                    GetPyNTTScalarDTypeName(stats.ElemType),
                    GetPyNTTScalarDTypeName(scale.ElemType),
                    GetPyNTTScalarDTypeName(bias.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(input.ElemType),
                    GetScalarTritonDType(stats.ElemType),
                    GetScalarTritonDType(scale.ElemType),
                    GetScalarTritonDType(bias.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    inputShape,
                    inputGlobalShape,
                    statsShape,
                    scaleShape,
                    biasShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(stats),
                    GetBufferStrides(scale),
                    GetBufferStrides(bias),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    statsVectorLaneCount,
                    scaleVectorLaneCount,
                    biasVectorLaneCount,
                    outputVectorLaneCount,
                    normalizedAxis,
                    normApply.Epsilon,
                    normApply.UseMean,
                    $"{input.Name}, {stats.Name}, {scale.Name}, {bias.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
            MarkStoredOutput(output, "PyNTT NormApply");
        }

        private void VisitGetPositionIds(Nncase.TIR.NTT.GetPositionIds getPositionIds, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException($"PyNTT GetPositionIds codegen expects kv-cache input and output TIR buffer, got ({string.Join(", ", args.Select(arg => arg.GetType().Name))}).");
            }

            var outputShape = GetBufferShape(output);
            ValidateRank("PyNTT GetPositionIds output", outputShape, 1);
            var globalShape = GetRankedShapeDimensions(getPositionIds.DistributedType.TensorType.Shape, "PyNTT GetPositionIds global shape")
                .Select(GetDimensionExpression)
                .ToArray();
            ValidateRank("PyNTT GetPositionIds global output", globalShape, 1);
            var cacheMetaInputIndex = RegisterKVCacheFieldInput(args[0], "metadata");
            SetComputeOp("get_position_ids");
            _attrs["op"] = "get_position_ids";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("get_position_ids_compute");
            WriteHelperTemplate(
                "triton/kernels/GetPositionIds.py.jinja",
                new PyNTTGetPositionIdsTemplateModel(
                    helperName,
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(output.ElemType),
                    outputShape,
                    globalShape,
                    GetBufferStrides(output),
                    GetHierarchy(output),
                    GetBufferSplitAxes(output, outputShape.Length),
                    GetShardAxis(output),
                    $"kv-cache -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName, $"input{cacheMetaInputIndex}"));
        }

        private void VisitUpdatePagedAttentionKVCache(Nncase.TIR.NTT.UpdatePagedAttentionKVCache update, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer slots)
            {
                throw new NotSupportedException("PyNTT UpdatePagedAttentionKVCache codegen expects slots buffer, kv-cache object, and layer id.");
            }

            var cache = GetPagedAttentionCacheTemplateModel(args[1], "PyNTT UpdatePagedAttentionKVCache");
            var layerIdExpression = GetDimensionExpression(args[2], "PyNTT UpdatePagedAttentionKVCache layer id").TritonExpression;
            var storage = GetKVCacheStorageMetadata(cache);
            var metaInputIndex = RegisterKVCacheFieldInput(args[1], "metadata");
            var slotMappingInputIndex = RegisterKVCacheFieldInput(args[1], "slot_mapping");
            var storageInputIndex = RegisterKVCacheFieldInput(args[1], "kv_caches", storage);
            var storageBlocksInputIndex = RegisterKVCacheFieldInput(args[1], "kv_caches_blocks", storage);
            var layout = update.Layout.ToArray();
            var seqAxis = Array.IndexOf(layout, AttentionDimKind.Seq);
            var headAxis = Array.IndexOf(layout, AttentionDimKind.Head);
            var dimAxis = Array.IndexOf(layout, AttentionDimKind.Dim);
            if (seqAxis < 0 || headAxis < 0 || dimAxis < 0)
            {
                throw new NotSupportedException("PyNTT UpdatePagedAttentionKVCache layout must contain Seq, Head, and Dim.");
            }

            SetComputeOp("update_paged_attention_kv_cache");
            _attrs["op"] = "update_paged_attention_kv_cache";
            _attrs["tir"] = true;
            _attrs["cache_kind"] = update.CacheKind.ToString();
            _attrs["layer_id"] = layerIdExpression;
            var helperName = GetNextHelperName("update_paged_attention_kv_cache");
            var slotsRef = ResolveBufferRef(slots);
            WriteHelperTemplate(
                "triton/kernels/UpdatePagedAttentionKVCache.py.jinja",
                new PyNTTUpdatePagedAttentionKVCacheTemplateModel(
                    helperName,
                    GetBufferScalarPointer(slots),
                    GetPyNTTScalarDTypeName(slots.ElemType),
                    GetScalarTritonDType(slots.ElemType),
                    GetBufferShape(slots),
                    GetBufferGlobalShape(slots),
                    GetBufferStrides(slots),
                    GetBufferSplitAxes(slots, slots.Dimensions.Length),
                    GetHierarchy(slots),
                    slotsRef.BaseName,
                    slotsRef.OffsetBytes,
                    slotsRef.PoolStrideBytes,
                    GetScalarElementSizeBytes(slots.ElemType),
                    seqAxis,
                    headAxis,
                    dimAxis,
                    layerIdExpression,
                    update.CacheKind == AttentionCacheKind.Key ? 0 : 1,
                    GetVectorLaneElementCount(slots.ElemType),
                    cache,
                    $"{slots.Name} -> kv-cache"));
            _attrs["requires_grid_barrier"] = true;
            WriteBarrier(HelperBarrierKind.Grid);
            WriteLine(BuildHelperCall(helperName, $"input{slotMappingInputIndex}", $"input{storageInputIndex}", $"input{storageBlocksInputIndex}", $"input{metaInputIndex}"), HelperBarrierKind.Grid);
        }

        private void VisitPagedAttention(Nncase.TIR.NTT.PagedAttention pagedAttention, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 6 ||
                args[0] is not TIR.Buffer query ||
                args[2] is not TIR.Buffer ||
                args[3] is not TIR.Buffer scale ||
                args[5] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT PagedAttention codegen expects query, kv-cache, extra, scale, layer id, and output buffers.");
            }

            var cache = GetPagedAttentionCacheTemplateModel(args[1], "PyNTT PagedAttention");
            var layerIdExpression = GetDimensionExpression(args[4], "PyNTT PagedAttention layer id").TritonExpression;
            var storage = GetKVCacheStorageMetadata(cache);
            var metaInputIndex = RegisterKVCacheFieldInput(args[1], "metadata");
            var blockTablesInputIndex = RegisterKVCacheFieldInput(args[1], "block_tables");
            var storageInputIndex = RegisterKVCacheFieldInput(args[1], "kv_caches", storage);
            var storageBlocksInputIndex = RegisterKVCacheFieldInput(args[1], "kv_caches_blocks", storage);
            var layout = pagedAttention.Layout.ToArray();
            var seqAxis = Array.IndexOf(layout, AttentionDimKind.Seq);
            var headAxis = Array.IndexOf(layout, AttentionDimKind.Head);
            var dimAxis = Array.IndexOf(layout, AttentionDimKind.Dim);
            if (seqAxis < 0 || headAxis < 0 || dimAxis < 0)
            {
                throw new NotSupportedException("PyNTT PagedAttention layout must contain Seq, Head, and Dim.");
            }

            var querySplitAxes = GetBufferSplitAxes(query, query.Dimensions.Length);
            var outputSplitAxes = GetBufferSplitAxes(output, output.Dimensions.Length);
            if (querySplitAxes[dimAxis].Length != 0 || outputSplitAxes[dimAxis].Length != 0)
            {
                throw new NotSupportedException("PyNTT PagedAttention codegen requires the attention Dim axis to be unsplit.");
            }

            SetComputeOp("paged_attention");
            _attrs["op"] = "paged_attention";
            _attrs["tir"] = true;
            _attrs["layer_id"] = layerIdExpression;
            _attrs["hidden_size"] = pagedAttention.HiddenSize;
            var helperName = GetNextHelperName("paged_attention");
            WriteHelperTemplate(
                "triton/kernels/PagedAttention.py.jinja",
                new PyNTTPagedAttentionTemplateModel(
                    helperName,
                    GetBufferScalarPointer(query),
                    GetBufferScalarPointer(scale),
                    GetBufferScalarPointer(output),
                    GetPyNTTScalarDTypeName(query.ElemType),
                    GetScalarTritonDType(query.ElemType),
                    GetPyNTTScalarDTypeName(output.ElemType),
                    GetScalarTritonDType(output.ElemType),
                    GetBufferShape(query),
                    GetBufferShape(output),
                    GetBufferGlobalShape(output),
                    GetBufferStrides(query),
                    GetBufferStrides(output),
                    outputSplitAxes,
                    GetHierarchy(output),
                    seqAxis,
                    headAxis,
                    dimAxis,
                    GetGlobalNumQueryHeads(pagedAttention, cache),
                    layerIdExpression,
                    cache,
                    $"{query.Name}, kv-cache -> {output.Name}"));
            _attrs["requires_grid_barrier"] = true;
            WriteBarrier(HelperBarrierKind.Grid);
            WriteLine(BuildHelperCall(helperName, $"input{blockTablesInputIndex}", $"input{storageInputIndex}", $"input{storageBlocksInputIndex}", $"input{metaInputIndex}"));
        }

        private void VisitSoftmax(int axis, IRArray<int> vectorizedAxes, IReadOnlyList<BaseExpr> args, string opKind)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Softmax codegen expects input and output TIR buffers.");
            }

            EnsureEmpty("PyNTT Softmax vectorized axes", vectorizedAxes);
            SetComputeOp(opKind);
            var shape = GetBufferShape(output);
            ValidateSameShape("PyNTT Softmax", GetBufferShape(input), shape);
            var normalizedAxis = NormalizeAxis(axis, shape.Length, "PyNTT Softmax");
            if (GetShardAxis(output) == normalizedAxis || GetShardAxis(input) == normalizedAxis)
            {
                throw new NotSupportedException("PyNTT Softmax codegen does not support sharding along the softmax axis yet.");
            }

            _attrs["op"] = "softmax";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["axis"] = normalizedAxis;
            var helperName = GetNextHelperName("softmax_compute");
            WriteHelperTemplate(
                "triton/kernels/Softmax.py.jinja",
                new PyNTTSoftmaxTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    normalizedAxis,
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
        }

        private int GetInputIndex(BaseExpr expr)
        {
            expr = UnwrapInputBoxing(expr);

            var inputName = GetTensorName(expr, _parameterNames);
            var inputIndex = _inputNames.IndexOf(inputName);
            if (inputIndex < 0)
            {
                inputIndex = _inputNames.Count;
                _inputNames.Add(inputName);
            }

            return inputIndex;
        }

        private TIR.Buffer GetBufferOperand(BaseExpr expr, string context)
        {
            expr = UnwrapInputBoxing(expr);
            return expr switch
            {
                TIR.Buffer buffer => buffer,
                BufferVar bufferVar => GetAbiBuffer(bufferVar, context),
                _ => throw new NotSupportedException($"{context} expects a TIR buffer or buffer ABI parameter, got {expr.GetType().Name}."),
            };
        }

        private TIR.Buffer GetAbiBuffer(BufferVar bufferVar, string context)
        {
            if (_abiBufferMemo.TryGetValue(bufferVar, out var buffer))
            {
                return buffer;
            }

            var tensorType = GetTensorType(bufferVar.CheckedType, context);
            var distributedType = bufferVar.CheckedType as DistributedType;
            buffer = T.AttachBuffer(bufferVar, tensorType, bufferVar.Location, 0, out _, $"{bufferVar.Name}_abi", distributedType);
            _abiBufferMemo.Add(bufferVar, buffer);
            return buffer;
        }

        private void MarkStoredOutput(TIR.Buffer buffer, string context)
        {
            if (buffer.MemSpan.Buffer.Location != MemoryLocation.Output)
            {
                return;
            }

            var outputIndex = GetOutputIndex(buffer);
            if (!_storedOutputIndices.Add(outputIndex))
            {
                throw new NotSupportedException($"{context} stores output {_outputs[outputIndex].Name} more than once.");
            }

            _outputDistributedTypes[outputIndex] = GetDistributedType(buffer);
        }

        private string GetInputName(int inputIndex, string context)
        {
            if (inputIndex < 0 || inputIndex >= _inputNames.Count)
            {
                throw new NotSupportedException($"{context} references input index {inputIndex}, but PyNTT PrimFunction {_function.Name} only has {_inputNames.Count} inputs.");
            }

            return _inputNames[inputIndex];
        }

        private int GetOutputIndex(BaseExpr expr)
        {
            expr = UnwrapInputBoxing(expr);
            var outputName = GetTensorName(expr, _parameterNames);
            for (var i = 0; i < _outputs.Length; i++)
            {
                if (_outputs[i].Name == outputName || _outputs[i].AbiName == outputName)
                {
                    return i;
                }
            }

            throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} references unknown output parameter {outputName}.");
        }

        private int GetBufferInputIndex(TIR.Buffer buffer, string context)
        {
            if (_bufferInputIndices.TryGetValue(buffer, out var inputIndex))
            {
                return inputIndex;
            }

            throw new NotSupportedException($"{context} must be loaded from a function input before use.");
        }

        private int RegisterKVCacheFieldInput(BaseExpr expr, string field, PyNTTKVCacheStorageMetadata? storage = null)
        {
            var sourceName = GetKVCacheSourceInputName(expr, $"PyNTT KV-cache field {field}");
            var syntheticName = $"{sourceName}.__{field}";
            var inputIndex = _inputNames.IndexOf(syntheticName);
            if (inputIndex < 0)
            {
                inputIndex = _inputNames.Count;
                _inputNames.Add(syntheticName);
            }

            if (!_kvCacheFieldInputs.Any(input => input.Name == syntheticName))
            {
                _kvCacheFieldInputs.Add(new(syntheticName, sourceName, field, storage));
            }

            return inputIndex;
        }

        private string GetKVCacheSourceInputName(BaseExpr expr, string context)
        {
            expr = UnwrapInputBoxing(expr);
            if (expr is TIR.Buffer kvCache)
            {
                if (IsAbiBuffer(kvCache))
                {
                    return _inputNames[GetInputIndex(kvCache)];
                }

                return _inputNames[GetBufferInputIndex(kvCache, context)];
            }

            return GetTensorName(expr, _parameterNames);
        }

        private KernelInputLayout BuildKernelInputLayout(string bodySource)
        {
            var names = new List<string>(_inputNames.Count);
            var indexMap = new Dictionary<int, int>();
            var removedIndexes = new HashSet<int>();

            for (var i = 0; i < _inputNames.Count; i++)
            {
                var name = _inputNames[i];
                if (IsObjectKernelInput(name))
                {
                    removedIndexes.Add(i);
                    continue;
                }

                indexMap[i] = names.Count;
                names.Add(name);
            }

            var remappedBodySource = RemapInputReferences(bodySource, indexMap, removedIndexes, $"PyNTT PrimFunction {_function.Name} body");
            var helpers = _helpers
                .Select((helper, index) => RemapHelperInputReferences(helper, indexMap, removedIndexes, $"PyNTT PrimFunction {_function.Name} helper {index}"))
                .ToArray();
            return new(names.ToArray(), indexMap, removedIndexes, remappedBodySource, helpers);
        }

        private bool IsObjectKernelInput(string name)
        {
            foreach (var pair in _parameterNames)
            {
                if (pair.Value == name && pair.Key is Expr expr)
                {
                    return IsObjectExpression(expr);
                }
            }

            return false;
        }

        private static HelperTemplateRenderSpec RemapHelperInputReferences(
            HelperTemplateRenderSpec helper,
            IReadOnlyDictionary<int, int> indexMap,
            IReadOnlySet<int> removedIndexes,
            string context)
        {
            var model = JsonSerializer.SerializeToNode(helper.Model);
            var remappedModel = RemapJsonInputReferences(model, indexMap, removedIndexes, context);
            var remappedArguments = helper.Arguments
                .Select(argument => RemapInputReferences(argument, indexMap, removedIndexes, context))
                .ToArray();
            return helper with { Model = remappedModel ?? new JsonObject(), Arguments = remappedArguments };
        }

        private static JsonNode? RemapJsonInputReferences(
            JsonNode? node,
            IReadOnlyDictionary<int, int> indexMap,
            IReadOnlySet<int> removedIndexes,
            string context)
        {
            switch (node)
            {
                case null:
                    return null;
                case JsonValue value when value.TryGetValue<string>(out var text):
                    return JsonValue.Create(RemapInputReferences(text, indexMap, removedIndexes, context));
                case JsonValue value:
                    return value.DeepClone();
                case JsonArray array:
                    var remappedArray = new JsonArray();
                    foreach (var item in array)
                    {
                        remappedArray.Add(RemapJsonInputReferences(item, indexMap, removedIndexes, context));
                    }

                    return remappedArray;
                case JsonObject obj:
                    var remappedObject = new JsonObject();
                    foreach (var pair in obj)
                    {
                        remappedObject[pair.Key] = RemapJsonInputReferences(pair.Value, indexMap, removedIndexes, context);
                    }

                    return remappedObject;
                default:
                    return node.DeepClone();
            }
        }

        private static string RemapInputReferences(
            string source,
            IReadOnlyDictionary<int, int> indexMap,
            IReadOnlySet<int> removedIndexes,
            string context)
        {
            return InputReferenceRegex.Replace(source, match =>
            {
                var inputIndex = int.Parse(match.Groups["index"].Value, CultureInfo.InvariantCulture);
                if (removedIndexes.Contains(inputIndex))
                {
                    throw new NotSupportedException($"{context} references object input{inputIndex}, which cannot be passed to a Triton kernel.");
                }

                return indexMap.TryGetValue(inputIndex, out var remappedInputIndex)
                    ? $"input{remappedInputIndex.ToString(CultureInfo.InvariantCulture)}{match.Groups["suffix"].Value}"
                    : match.Value;
            });
        }

        private PyNTTPagedAttentionCacheTemplateModel GetPagedAttentionCacheTemplateModel(BaseExpr expr, string context)
        {
            var config = GetPagedAttentionConfig(expr, context);
            ValidatePagedAttentionConfig(config, context);

            var key = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Key, context);
            var value = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Value, context);
            var topologyShape = GetPagedAttentionTopologyShape(config);
            var numBlocksSplitAxes = GetPagedAttentionNumBlocksSplitAxes(config);
            var valueSectionOffset = key.SectionElements;
            var blockElements = checked(key.SectionElements + value.SectionElements);
            return new(
                GetPyNTTScalarDTypeName(config.KVPrimType),
                GetScalarTritonDType(config.KVPrimType),
                config.NumLayers,
                config.NumKVHeads,
                config.HeadDim,
                config.BlockSize,
                key.LaneCount,
                value.LaneCount,
                (int)key.VectorizedDim,
                (int)value.VectorizedDim,
                key.HeadDimBlocks,
                value.HeadDimBlocks,
                0,
                valueSectionOffset,
                key.SectionElements,
                value.SectionElements,
                blockElements,
                key.LayerStride,
                key.HeadStride,
                key.DimBlockStride,
                key.BlockOffsetStride,
                value.LayerStride,
                value.HeadStride,
                value.DimBlockStride,
                value.BlockOffsetStride,
                key.TailShape,
                value.TailShape,
                config.ShardingAxes.Count + 1,
                topologyShape,
                numBlocksSplitAxes);
        }

        private static IPagedAttentionConfig GetPagedAttentionConfig(BaseExpr expr, string context)
        {
            if (expr is TIR.Buffer buffer)
            {
                return GetPagedAttentionConfig(buffer.ElemType, context);
            }

            var tensorType = GetTensorType(expr.CheckedType, context);
            return GetPagedAttentionConfig(tensorType.DType, context);
        }

        private static IPagedAttentionConfig GetPagedAttentionConfig(DataType dataType, string context)
        {
            if (dataType is ReferenceType { ElemType: PagedAttentionKVCacheType kvCacheType } &&
                kvCacheType.Config is { } config)
            {
                return config;
            }

            throw new NotSupportedException($"{context} expects Reference<PagedAttentionKVCacheType>, got {dataType}.");
        }

        private sealed record PagedAttentionCacheKindLayout(
            int LaneCount,
            PagedKVCacheDimKind VectorizedDim,
            int HeadDimBlocks,
            int[] TailShape,
            int SectionElements,
            int LayerStride,
            int HeadStride,
            int DimBlockStride,
            int BlockOffsetStride);

        private static PagedAttentionCacheKindLayout GetPagedAttentionCacheKindLayout(IPagedAttentionConfig config, AttentionCacheKind kind, string context)
        {
            var vectorizedAxes = config.GetVectorizedAxes(kind).ToArray();
            var lanes = config.GetLanes(kind).ToArray();
            int laneCount;
            PagedKVCacheDimKind vectorizedDim;
            if (vectorizedAxes.Length == 0 && lanes.Length == 0)
            {
                laneCount = 1;
                vectorizedDim = (PagedKVCacheDimKind)(-1);
            }
            else if (vectorizedAxes.Length == 1 && lanes.Length == 1 &&
                vectorizedAxes[0] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.BlockSize)
            {
                laneCount = checked((int)lanes[0]);
                vectorizedDim = vectorizedAxes[0];
            }
            else
            {
                throw new NotSupportedException($"{context} currently supports only {kind} HeadDim or BlockSize vectorization with zero or one lane value.");
            }

            if (laneCount <= 0)
            {
                throw new NotSupportedException($"{context} requires {kind} lane to be positive, got {laneCount}.");
            }

            if (vectorizedDim == PagedKVCacheDimKind.HeadDim && config.HeadDim % laneCount != 0)
            {
                throw new NotSupportedException($"{context} requires {kind} head_dim divisible by lane, got head_dim={config.HeadDim}, lane={laneCount}.");
            }

            if (vectorizedDim == PagedKVCacheDimKind.BlockSize && config.BlockSize % laneCount != 0)
            {
                throw new NotSupportedException($"{context} requires {kind} block_size divisible by lane, got block_size={config.BlockSize}, lane={laneCount}.");
            }

            var headDimBlocks = vectorizedDim == PagedKVCacheDimKind.HeadDim
                ? checked(config.HeadDim / laneCount)
                : config.HeadDim;
            var tailDims = new List<int>();
            var tailDimKinds = new List<PagedKVCacheDimKind>();
            foreach (var dimKind in config.GetCacheLayout(kind))
            {
                if (dimKind is PagedKVCacheDimKind.NumBlocks or PagedKVCacheDimKind.KV)
                {
                    continue;
                }

                tailDimKinds.Add(dimKind);
                var dimExtent = dimKind switch
                {
                    PagedKVCacheDimKind.NumLayers => checked((int)config.NumLayers),
                    PagedKVCacheDimKind.BlockSize => checked((int)config.BlockSize),
                    PagedKVCacheDimKind.NumKVHeads => checked((int)config.NumKVHeads),
                    PagedKVCacheDimKind.HeadDim => checked((int)config.HeadDim),
                    _ => throw new NotSupportedException($"{context} does not support {kind} cache dimension {dimKind}."),
                };
                if (dimKind == vectorizedDim)
                {
                    dimExtent = checked(dimExtent / laneCount);
                }

                tailDims.Add(dimExtent);
            }

            if (tailDims.Count != 4)
            {
                throw new NotSupportedException($"{context} expects {kind} cache layout to contain exactly NumLayers, NumKVHeads, HeadDim, and BlockSize after removing NumBlocks/KV.");
            }

            var strides = ComputeContiguousStrides(tailDims);
            int GetStride(PagedKVCacheDimKind dimKind)
            {
                var index = tailDimKinds.IndexOf(dimKind);
                if (index < 0)
                {
                    throw new NotSupportedException($"{context} {kind} cache layout is missing {dimKind}.");
                }

                return strides[index];
            }

            var sectionVectorElements = checked(tailDims.Aggregate(1, static (acc, dim) => checked(acc * dim)));
            var sectionElements = checked(sectionVectorElements * laneCount);
            return new(
                laneCount,
                vectorizedDim,
                headDimBlocks,
                tailDims.ToArray(),
                sectionElements,
                GetStride(PagedKVCacheDimKind.NumLayers),
                GetStride(PagedKVCacheDimKind.NumKVHeads),
                GetStride(PagedKVCacheDimKind.HeadDim),
                GetStride(PagedKVCacheDimKind.BlockSize));
        }

        private static int[] ComputeContiguousStrides(IReadOnlyList<int> shape)
        {
            var strides = new int[shape.Count];
            var stride = 1;
            for (int i = shape.Count - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride = checked(stride * shape[i]);
            }

            return strides;
        }

        private PyNTTKVCacheStorageMetadata GetKVCacheStorageMetadata(PyNTTPagedAttentionCacheTemplateModel cache)
        {
            return new(
                cache.DType,
                cache.TopologyShape,
                cache.KeyTailShape,
                cache.ValueTailShape,
                cache.KeySectionElements,
                cache.ValueSectionElements,
                cache.BlockElements,
                cache.BlockSize);
        }

        private static int GetGlobalNumQueryHeads(Nncase.TIR.NTT.PagedAttention pagedAttention, PyNTTPagedAttentionCacheTemplateModel cache)
        {
            if (pagedAttention.HiddenSize <= 0 || pagedAttention.HiddenSize % cache.HeadDim != 0)
            {
                throw new NotSupportedException($"PyNTT PagedAttention requires hidden_size divisible by head_dim, got hidden_size={pagedAttention.HiddenSize}, head_dim={cache.HeadDim}.");
            }

            var numQueryHeads = checked(pagedAttention.HiddenSize / cache.HeadDim);
            if (numQueryHeads <= 0 || numQueryHeads % cache.NumKVHeads != 0)
            {
                throw new NotSupportedException($"PyNTT PagedAttention requires num_query_heads divisible by num_kv_heads, got num_query_heads={numQueryHeads}, num_kv_heads={cache.NumKVHeads}.");
            }

            return numQueryHeads;
        }

        private void ValidatePagedAttentionConfig(IPagedAttentionConfig config, string context)
        {
            ValidatePagedAttentionLayout(config.KeyCacheLayout, AttentionCacheKind.Key, context);
            ValidatePagedAttentionLayout(config.ValueCacheLayout, AttentionCacheKind.Value, context);
            _ = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Key, context);
            _ = GetPagedAttentionCacheKindLayout(config, AttentionCacheKind.Value, context);

            if (config.ShardingAxes.Count > 1 ||
                (config.ShardingAxes.Count == 1 && config.ShardingAxes[0] != PagedKVCacheDimKind.NumBlocks))
            {
                throw new NotSupportedException($"{context} currently supports no KV-cache sharding or NumBlocks-only sharding.");
            }

            if (config.AxisPolicies.Count != config.ShardingAxes.Count)
            {
                throw new NotSupportedException($"{context} requires one axis policy per KV-cache sharding axis.");
            }

            if (config.ShardingAxes.Count == 1)
            {
                var hierarchy = GetBlockHierarchy(_targetOptions);
                var policy = config.AxisPolicies[0];
                foreach (var axis in policy.Axes)
                {
                    if (axis < 0 || axis >= hierarchy.Length)
                    {
                        throw new NotSupportedException($"{context} NumBlocks sharding axis {axis} is outside PyNTT hierarchy rank {hierarchy.Length}.");
                    }
                }
            }
        }

        private static void ValidatePagedAttentionLayout(IRArray<PagedKVCacheDimKind> layout, AttentionCacheKind kind, string context)
        {
            if (layout.Count != 6)
            {
                throw new NotSupportedException($"{context} {kind} cache layout must have 6 dimensions, got {layout.Count}.");
            }

            var expected = Enum.GetValues<PagedKVCacheDimKind>().OrderBy(x => (int)x).ToArray();
            var actual = layout.ToArray().OrderBy(x => (int)x).ToArray();
            if (!actual.SequenceEqual(expected))
            {
                throw new NotSupportedException($"{context} {kind} cache layout must be a permutation of [{string.Join(", ", expected)}], got [{string.Join(", ", layout)}].");
            }
        }

        private int[] GetPagedAttentionTopologyShape(IPagedAttentionConfig config)
        {
            if (config.ShardingAxes.Count == 0)
            {
                return Array.Empty<int>();
            }

            var hierarchy = GetBlockHierarchy(_targetOptions);
            return config.AxisPolicies
                .Select(policy => policy.Axes.Aggregate(1, (product, axis) => checked(product * hierarchy[axis])))
                .ToArray();
        }

        private static int[] GetPagedAttentionNumBlocksSplitAxes(IPagedAttentionConfig config)
        {
            if (config.ShardingAxes.Count == 0)
            {
                return Array.Empty<int>();
            }

            return config.AxisPolicies[0].Axes.ToArray();
        }

        private PyNTTBufferPointerTemplateModel GetBufferPointer(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            return new(BuildPointerExpression(bufferRef, GetTritonDType(buffer.ElemType)), bufferRef.ShardCoordHierarchy);
        }

        private PyNTTBufferPointerTemplateModel GetBufferScalarPointer(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            return new(BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType)), bufferRef.ShardCoordHierarchy);
        }

        private PyNTTBufferPointerTemplateModel GetBufferScalarPointer(TIR.Buffer buffer, string indexExpression)
        {
            var bufferRef = ResolveBufferRef(buffer) with { IndexExpression = indexExpression };
            return new(BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType)), bufferRef.ShardCoordHierarchy);
        }

        private PyNTTDimExpression[] GetTensorShape(BaseExpr expr, string name)
        {
            var tensorType = GetTensorType(expr.CheckedType, name);
            return GetRankedShape(tensorType, name).Dimensions.ToArray()
                .Select(_dimEmitter.Emit)
                .ToArray();
        }

        private BufferRef ResolveBufferRef(TIR.Buffer buffer)
        {
            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Input)
            {
                return ResolveAbiBufferRef(buffer, $"input{GetInputIndex(buffer).ToString(CultureInfo.InvariantCulture)}");
            }

            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Output)
            {
                return ResolveAbiBufferRef(buffer, $"output{GetOutputIndex(buffer).ToString(CultureInfo.InvariantCulture)}");
            }

            var offsetBytes = GetBufferOffsetBytes(buffer);
            var shardCoordHierarchy = RequiresShardCoords(offsetBytes)
                ? GetShardCoordHierarchy(buffer)
                : null;
            return buffer.MemSpan.Buffer.Location switch
            {
                MemoryLocation.Data when buffer.DistributedType is null => new("data", offsetBytes, "0", null, shardCoordHierarchy),
                MemoryLocation.Data => new("data", offsetBytes, "data_pool_stride_bytes", "shard_index", shardCoordHierarchy),
                MemoryLocation.BlockLocalData => new("block_local_data", offsetBytes, "block_local_data_pool_stride_bytes", BuildBlockLocalDataIndexExpression(_targetOptions), shardCoordHierarchy),
                MemoryLocation.Rdata => new("rdata", offsetBytes, "0", null, shardCoordHierarchy),
                MemoryLocation.ChipLocalRdata => new("chip_local_rdata", offsetBytes, "0", null, shardCoordHierarchy),
                MemoryLocation.BlockLocalRdata => new("block_local_rdata", offsetBytes, PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.BlockLocalRdatas, _targetOptions, "b").ToString(CultureInfo.InvariantCulture), "shard_index", shardCoordHierarchy),
                var location => throw new NotSupportedException($"PyNTT does not support buffer memory location {location} for Triton template operands yet."),
            };
        }

        private BufferRef ResolveAbiBufferRef(TIR.Buffer buffer, string baseName)
        {
            var spanOffsetElements = GetBufferSpanOffsetElements(buffer);
            if (buffer.DistributedType is not { })
            {
                return new(baseName, spanOffsetElements, "0", null, null);
            }

            var poolStrideElements = $"{baseName}{PoolStrideElementsSuffix}";
            var localOffsetElements = GetDistributedCompactLocalOffsetElements(buffer);
            var offsetElements = spanOffsetElements;
            if (!IsZeroOffset(localOffsetElements))
            {
                offsetElements = AddOffsetExpressions(offsetElements, $"tl.where({poolStrideElements} == 0, {localOffsetElements}, 0)");
            }

            var shardCoordHierarchy = RequiresShardCoords(offsetElements)
                ? GetShardCoordHierarchy(buffer)
                : null;
            return new(baseName, offsetElements, poolStrideElements, "shard_index", shardCoordHierarchy);
        }

        private string BuildPointerExpression(BufferRef bufferRef, string tritonDType)
        {
            var expression = bufferRef.BaseName;
            if (!string.IsNullOrWhiteSpace(bufferRef.IndexExpression) && bufferRef.PoolStrideBytes != "0")
            {
                expression += $" + {bufferRef.IndexExpression} * {bufferRef.PoolStrideBytes}";
            }

            if (!IsZeroOffset(bufferRef.OffsetBytes))
            {
                expression += $" + {bufferRef.OffsetBytes}";
            }

            return $"({expression}).to(tl.pointer_type({tritonDType}))";
        }

        private string BuildHelperCall(string helperName, params string[] leadingArguments)
        {
            var helperArguments = _helperArguments.TryGetValue(helperName, out var arguments)
                ? arguments
                : Array.Empty<string>();
            var callArguments = leadingArguments.Concat(helperArguments).ToArray();
            _helperCalls.Add(new(helperName, callArguments));
            var args = leadingArguments
                .Select(FormatHelperCallArgument)
                .Concat(helperArguments)
                .Concat(WorkspaceParameterNames)
                .Concat(_runtimeScalarNames)
                .Concat(new[] { "block_size" });
            return $"{helperName}({string.Join(", ", args)})";
        }

        private static string BuildRawPythonArgument(string expression) => $"py:{expression}";

        private static string FormatHelperCallArgument(string argument)
            => argument.StartsWith("py:", StringComparison.Ordinal) ? argument[3..] : argument;

        private PyNTTDimExpression[] GetBufferShape(TIR.Buffer buffer)
        {
            if (buffer.DistributedType is { } distributedType)
            {
                return GetRankedShapeDimensions(DistributedUtility.GetDividedTensorType(distributedType).Shape, $"{buffer.Name} distributed local shape")
                    .Select(GetDimensionExpression)
                    .ToArray();
            }

            return buffer.Dimensions.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetBufferActiveShape(TIR.Buffer buffer)
        {
            if (buffer.DistributedType is { } distributedType)
            {
                var shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                    .Select(axis => (Dimension)new DimVar($"{ShardCoordDimPrefix}{axis}"))
                    .ToArray();
                var (_, activeShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var hierarchy = distributedType.Placement.Hierarchy.ToArray();
                return activeShape
                    .Select(dimension => GetLocalRegionDimensionExpression(dimension, hierarchy))
                    .ToArray();
            }

            return buffer.Dimensions.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetBufferGlobalShape(TIR.Buffer buffer)
        {
            if (buffer.DistributedType is { } distributedType)
            {
                return GetRankedShapeDimensions(distributedType.TensorType.Shape, $"{buffer.Name} distributed global shape")
                    .Select(GetDimensionExpression)
                    .ToArray();
            }

            return buffer.Dimensions.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetBufferStrides(TIR.Buffer buffer)
        {
            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Input)
            {
                var inputIndex = GetInputIndex(buffer).ToString(CultureInfo.InvariantCulture);
                return GetAbiBufferStrideExpressions($"input{inputIndex}", buffer);
            }

            if (buffer.MemSpan.Buffer.Location == MemoryLocation.Output)
            {
                var outputIndex = GetOutputIndex(buffer).ToString(CultureInfo.InvariantCulture);
                return GetAbiBufferStrideExpressions($"output{outputIndex}", buffer);
            }

            return buffer.Strides.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
        }

        private PyNTTDimExpression[] GetAbiBufferStrideExpressions(string prefix, TIR.Buffer buffer)
        {
            var stridePrefix = GetVectorLaneElementCount(buffer.ElemType) == 1
                ? $"{prefix}_scalar_stride"
                : $"{prefix}_stride";
            var strides = new PyNTTDimExpression[buffer.Rank];
            for (var axis = 0; axis < buffer.Rank; axis++)
            {
                var name = $"{stridePrefix}{axis.ToString(CultureInfo.InvariantCulture)}";
                _abiViewStrideArgNames.Add(name);
                strides[axis] = new PyNTTDimExpression(name, name);
            }

            return strides;
        }

        private string GetBufferOffsetBytes(TIR.Buffer buffer)
        {
            var physicalOffset = GetDimensionExpression(buffer.MemSpan.Buffer.Start, $"{buffer.MemSpan.Buffer.Location} physical buffer offset");
            var spanOffset = GetLocalRegionDimensionExpression(buffer.MemSpan.Start, GetShardCoordHierarchy(buffer));
            return AddOffsetExpressions(physicalOffset.TritonExpression, spanOffset.TritonExpression);
        }

        private string GetBufferSpanOffsetBytes(TIR.Buffer buffer)
        {
            var spanOffset = GetLocalRegionDimensionExpression(buffer.MemSpan.Start, GetShardCoordHierarchy(buffer));
            return spanOffset.TritonExpression;
        }

        private string GetBufferSpanOffsetElements(TIR.Buffer buffer)
        {
            var spanOffset = GetBufferSpanOffsetBytes(buffer);
            return DivideOffsetExpression(spanOffset, GetScalarElementSizeBytes(buffer.ElemType));
        }

        private string GetDistributedCompactLocalOffsetElements(TIR.Buffer buffer)
        {
            if (buffer.DistributedType is not { } distributedType)
            {
                return "0";
            }

            var shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                .Select(axis => (Dimension)new DimVar($"{ShardCoordDimPrefix}{axis}"))
                .ToArray();
            var (localOffset, _) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
            var globalShape = GetRankedShapeDimensions(distributedType.TensorType.Shape, $"{buffer.Name} distributed global shape").ToArray();
            var globalStrides = TensorUtilities.GetDefaultStrides(globalShape);
            var offsetElements = TensorUtilities.GetLinearOffset(globalStrides, localOffset) * GetVectorLaneElementCount(buffer.ElemType);
            return GetLocalRegionDimensionExpression(offsetElements, distributedType.Placement.Hierarchy.ToArray()).TritonExpression;
        }

        private int[] GetHierarchy(TIR.Buffer buffer)
        {
            return buffer.DistributedType is { } distributedType
                ? distributedType.Placement.Hierarchy.ToArray()
                : GetBlockHierarchy(_targetOptions);
        }

        private static int[][] GetBufferSplitAxes(TIR.Buffer buffer, int rank)
        {
            if (buffer.DistributedType is not { } distributedType)
            {
                return Enumerable.Range(0, rank).Select(_ => Array.Empty<int>()).ToArray();
            }

            return GetSplitAxes(distributedType);
        }

        private static int[] GetMatmulReduceAxes(TIR.Buffer lhs, TIR.Buffer rhs, TIR.Buffer output, Nncase.IR.Math.MatMulDimInfo dimInfo)
        {
            if (lhs.DistributedType is not { } lhsType ||
                dimInfo.Lk >= lhsType.AxisPolicies.Count ||
                lhsType.AxisPolicies[dimInfo.Lk] is not SBPSplit lhsSplit)
            {
                return Array.Empty<int>();
            }

            var reduceAxes = lhsSplit.Axes.ToArray();
            if (reduceAxes.Length == 0)
            {
                return Array.Empty<int>();
            }

            if (reduceAxes.Any(axis => axis < 0 || axis >= lhsType.Placement.Rank))
            {
                throw new NotSupportedException($"PyNTT Matmul K-axis split uses placement axes [{string.Join(",", reduceAxes)}] outside placement rank {lhsType.Placement.Rank}.");
            }

            if (rhs.DistributedType is not { } rhsType ||
                dimInfo.Rk >= rhsType.AxisPolicies.Count ||
                rhsType.AxisPolicies[dimInfo.Rk] is not SBPSplit rhsSplit ||
                !rhsSplit.Axes.ToArray().SequenceEqual(reduceAxes))
            {
                throw new NotSupportedException("PyNTT Matmul K-axis split expects lhs and rhs K axes to use the same placement split axes.");
            }

            if (output.DistributedType is { } outputType &&
                outputType.AxisPolicies
                    .OfType<SBPSplit>()
                    .Any(split => split.Axes.Any(axis => reduceAxes.Contains(axis))))
            {
                throw new NotSupportedException("PyNTT Matmul K-axis split cannot write an output that is also split by the K reduction placement axes.");
            }

            return reduceAxes;
        }

        private string BuildShardReduceHelperKey(
            string baseName,
            string scalarDType,
            string tritonDType,
            IReadOnlyList<PyNTTDimExpression> localShape,
            IReadOnlyList<PyNTTDimExpression> strides,
            int vectorLaneCount,
            IReadOnlyList<int> hierarchy,
            IReadOnlyList<int> reduceAxes,
            bool broadcast)
        {
            static string DimsKey(IEnumerable<PyNTTDimExpression> dims)
                => string.Join(",", dims.Select(dim => dim.TritonExpression));

            return string.Join(
                "|",
                broadcast ? "broadcast" : "reduce",
                baseName,
                scalarDType,
                tritonDType,
                $"shape={DimsKey(localShape)}",
                $"strides={DimsKey(strides)}",
                $"lane={vectorLaneCount.ToString(CultureInfo.InvariantCulture)}",
                $"hierarchy={string.Join(",", hierarchy)}",
                $"reduce={string.Join(",", reduceAxes)}",
                $"runtime={string.Join(",", _runtimeScalarNames)}");
        }

        private PyNTTDimExpression GetDimensionExpression(Dimension dimension) => _dimEmitter.Emit(dimension);

        private PyNTTDimExpression GetLocalRegionDimensionExpression(Dimension dimension, IReadOnlyList<int> hierarchy)
        {
            var emitter = new PyNTTDimExpressionEmitter(
                RegisterLocalRegionRuntimeScalar,
                name => FormatLocalRegionRuntimeScalar(name, hierarchy),
                BuildThreadIdExpression(_targetOptions));
            return emitter.Emit(dimension);
        }

        private PyNTTDimExpression GetDimensionExpression(BaseExpr expr, string name)
        {
            expr = UnwrapInputBoxing(expr);
            return expr switch
            {
                None => PyNTTDimExpression.Zero,
                Dimension dimension => _dimEmitter.Emit(dimension),
                TensorConst tensorConst when tensorConst.Value.Shape.IsScalar => FormatDimensionConst(tensorConst),
                _ => throw new NotSupportedException($"PyNTT requires scalar dimension expression for {name}, got {expr.GetType().Name}."),
            };
        }

        private static PyNTTDimExpression FormatDimensionConst(TensorConst tensorConst)
        {
            var value = tensorConst.Value.ToScalar<long>();
            var text = value.ToString(CultureInfo.InvariantCulture);
            return new(text, text, value);
        }

        private static string AddOffsetExpressions(string lhs, string rhs)
        {
            if (IsZeroOffset(lhs))
            {
                return rhs;
            }

            if (IsZeroOffset(rhs))
            {
                return lhs;
            }

            return $"(({lhs}) + ({rhs}))";
        }

        private static string DivideOffsetExpression(string expression, int divisor)
        {
            if (divisor <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(divisor), divisor, "Offset divisor must be positive.");
            }

            if (divisor == 1 || IsZeroOffset(expression))
            {
                return expression;
            }

            if (long.TryParse(expression.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value))
            {
                if (value % divisor != 0)
                {
                    throw new NotSupportedException($"PyNTT ABI buffer byte offset {value} is not aligned to scalar element size {divisor}.");
                }

                return (value / divisor).ToString(CultureInfo.InvariantCulture);
            }

            return $"(({expression}) // {divisor.ToString(CultureInfo.InvariantCulture)})";
        }

        private static bool IsZeroOffset(string expression)
            => string.Equals(expression.Trim(), "0", StringComparison.Ordinal);

        private static bool RequiresShardCoords(string expression)
            => expression.Contains("shard_coord", StringComparison.Ordinal);

        private int[] GetShardCoordHierarchy(TIR.Buffer buffer)
            => buffer.DistributedType is { } distributedType
                ? distributedType.Placement.Hierarchy.ToArray()
                : GetBlockHierarchy(_targetOptions);

        private static string BuildThreadIdExpression(PyNTTTargetOptions targetOptions)
            => "0";

        private static string BuildBlockLocalDataIndexExpression(PyNTTTargetOptions targetOptions)
            => BuildScopeIndexExpression(GetBlockLocalDataScopeSize(targetOptions));

        private static int GetBlockLocalDataScopeCount(PyNTTTargetOptions targetOptions)
            => GetScopeCount(targetOptions, GetBlockLocalDataScopeSize(targetOptions));

        private static string BuildScopeIndexExpression(int scopeSize)
            => scopeSize <= 1 ? "shard_index" : $"(shard_index // {scopeSize.ToString(CultureInfo.InvariantCulture)})";

        private static int GetScopeCount(PyNTTTargetOptions targetOptions, int scopeSize)
        {
            var hierarchy = GetBlockHierarchy(targetOptions);
            var shardCount = hierarchy.Aggregate(1, (product, value) => checked(product * value));
            return Math.Max(1, shardCount / Math.Max(1, scopeSize));
        }

        private static int GetBlockLocalDataScopeSize(PyNTTTargetOptions targetOptions)
            => 1;

        private static int GetScalarElementSizeBytes(DataType dataType)
        {
            return dataType is VectorType vectorType
                ? vectorType.ElemType.SizeInBytes
                : dataType.SizeInBytes;
        }

        private void RegisterRuntimeScalar(string name)
        {
            _runtimeScalarNames.Add(name);
        }

        private void RegisterLocalRegionRuntimeScalar(string name)
        {
            if (!TryGetShardCoordDimAxis(name, out _))
            {
                RegisterRuntimeScalar(name);
            }
        }

        private static string FormatLocalRegionRuntimeScalar(string name, IReadOnlyList<int> hierarchy)
            => TryGetShardCoordDimAxis(name, out var axis) ? BuildShardCoordExpression(axis, hierarchy) : name;

        private static string BuildShardCoordExpression(int axis, IReadOnlyList<int> hierarchy)
        {
            if (axis < 0 || axis >= hierarchy.Count)
            {
                throw new NotSupportedException($"PyNTT local shard coordinate axis {axis} is outside hierarchy rank {hierarchy.Count}.");
            }

            var divisor = 1;
            for (var i = axis + 1; i < hierarchy.Count; i++)
            {
                divisor = checked(divisor * hierarchy[i]);
            }

            var dividend = divisor == 1
                ? "shard_index"
                : $"(shard_index // {divisor.ToString(CultureInfo.InvariantCulture)})";
            var extent = hierarchy[axis];
            return extent == 1
                ? "0"
                : $"(({dividend}) % {extent.ToString(CultureInfo.InvariantCulture)})";
        }

        private static bool TryGetShardCoordDimAxis(string name, out int axis)
        {
            axis = default;
            if (!name.StartsWith(ShardCoordDimPrefix, StringComparison.Ordinal))
            {
                return false;
            }

            return int.TryParse(name[ShardCoordDimPrefix.Length..], NumberStyles.None, CultureInfo.InvariantCulture, out axis);
        }

        private string BuildScalarExpression(BaseExpr expr)
        {
            expr = UnwrapInputBoxing(expr);
            if (expr is Dimension dimension)
            {
                return _dimEmitter.Emit(dimension).TritonExpression;
            }

            if (expr is TensorConst tensorConst)
            {
                return FormatScalarConst(tensorConst);
            }

            if (expr is IVar parameter && _parameterNames.TryGetValue(parameter, out var parameterName))
            {
                return SanitizePythonIdentifier(parameterName);
            }

            if (expr is Call call)
            {
                var args = call.Arguments.ToArray();
                return call.Target switch
                {
                    Nncase.IR.Math.Compare compare when args.Length >= 2 =>
                        $"({BuildScalarExpression(args[0])} {GetCompareOperator(compare.CompareOp)} {BuildScalarExpression(args[1])})",
                    Nncase.IR.Math.Binary binary when args.Length >= 2 =>
                        BuildScalarBinaryExpression(binary.BinaryOp, BuildScalarExpression(args[0]), BuildScalarExpression(args[1])),
                    _ => throw new NotSupportedException($"Unsupported PyNTT scalar expression call target: {call.Target.GetType().Name}."),
                };
            }

            throw new NotSupportedException($"Unsupported PyNTT scalar expression: {expr.GetType().Name}.");
        }

        private void SetComputeOp(string opKind)
        {
            _opKinds.Add(opKind);
        }

        private void WriteControlLine(string line)
        {
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine(line);
        }

        private void WriteLine(string line, HelperBarrierKind barrierKind = HelperBarrierKind.Block)
        {
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine(line);
            WriteBarrier(barrierKind);
        }

        private void WriteBarrier(HelperBarrierKind barrierKind)
        {
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine(barrierKind switch
            {
                HelperBarrierKind.Block => "tl.debug_barrier()",
                HelperBarrierKind.Grid => "tle.distributed_barrier(pyntt_grid_mesh)",
                _ => throw new ArgumentOutOfRangeException(nameof(barrierKind), barrierKind, null),
            });
        }

        private void WriteHelperTemplate(string templatePath, object model)
        {
            var runtimeShapeArgs = _runtimeScalarNames.ToArray();
            var runtimeShapeArgsProperty = model.GetType().GetProperty("RuntimeShapeArgs");
            runtimeShapeArgsProperty?.SetValue(model, runtimeShapeArgs);

            var arguments = CollectHelperArguments(model);
            var functionName = GetHelperFunctionName(model);
            _helperArguments[functionName] = arguments;
            _helpers.Add(new(GetJinjaTemplateName(templatePath), model, arguments));
        }

        private static string GetHelperFunctionName(object model)
        {
            var property = model.GetType().GetProperty("FunctionName")
                ?? throw new NotSupportedException($"PyNTT helper model {model.GetType().Name} must expose FunctionName.");
            return property.GetValue(model) as string
                ?? throw new NotSupportedException($"PyNTT helper model {model.GetType().Name} has non-string FunctionName.");
        }

        private static string[] CollectHelperArguments(object model)
        {
            var arguments = new HashSet<string>(StringComparer.Ordinal);
            CollectHelperArguments(JsonSerializer.SerializeToNode(model), arguments);
            return arguments
                .OrderBy(argument => argument.StartsWith("input", StringComparison.Ordinal) ? 0 : 1)
                .ThenBy(ParseAbiArgumentIndex)
                .ThenBy(GetAbiArgumentKind)
                .ThenBy(argument => argument, StringComparer.Ordinal)
                .ToArray();
        }

        private static void CollectHelperArguments(JsonNode? node, HashSet<string> arguments)
        {
            switch (node)
            {
                case null:
                    return;
                case JsonValue value when value.TryGetValue<string>(out var text):
                    foreach (Match match in AbiArgumentReferenceRegex.Matches(text))
                    {
                        arguments.Add(match.Value);
                    }

                    return;
                case JsonValue:
                    return;
                case JsonArray array:
                    foreach (var item in array)
                    {
                        CollectHelperArguments(item, arguments);
                    }

                    return;
                case JsonObject obj:
                    foreach (var property in obj)
                    {
                        CollectHelperArguments(property.Value, arguments);
                    }

                    return;
            }
        }

        private static int ParseAbiArgumentIndex(string argument)
        {
            argument = StripAbiArgumentSuffix(argument);
            var indexStart = argument.StartsWith("input", StringComparison.Ordinal) ? "input".Length : "output".Length;
            return int.Parse(argument[indexStart..], CultureInfo.InvariantCulture);
        }

        private static int GetAbiArgumentKind(string argument)
        {
            if (argument.Contains("_stride", StringComparison.Ordinal))
            {
                return 1;
            }

            if (argument.EndsWith(PoolStrideElementsSuffix, StringComparison.Ordinal))
            {
                return 2;
            }

            return 0;
        }

        private static string StripAbiArgumentSuffix(string argument)
            => AbiArgumentSuffixRegex.Replace(argument, string.Empty);

        private static string GetJinjaTemplateName(string templatePath)
        {
            const string prefix = "triton/kernels/";
            const string suffix = ".py.jinja";
            if (!templatePath.StartsWith(prefix, StringComparison.Ordinal) ||
                !templatePath.EndsWith(suffix, StringComparison.Ordinal))
            {
                throw new NotSupportedException($"Unsupported PyNTT Jinja template path: {templatePath}.");
            }

            return templatePath;
        }

        private string GetHelperName(string kind, int index)
        {
            var helperBaseName = _namePart is null ? _function.Name : $"{_function.Name}_{_namePart}";
            return SanitizePythonIdentifier($"{helperBaseName}_{kind}_{index}");
        }

        private string GetNextHelperName(string kind)
        {
            var index = _helperCounters.TryGetValue(kind, out var current) ? current : 0;
            _helperCounters[kind] = index + 1;
            return GetHelperName(kind, index);
        }

        private BaseExpr UnwrapInputBoxing(BaseExpr expr)
        {
            while (expr is Call call && call.Target is Boxing)
            {
                expr = call.Arguments[0];
            }

            return expr;
        }

        private sealed record BufferRef(string BaseName, string OffsetBytes, string PoolStrideBytes, string? IndexExpression, int[]? ShardCoordHierarchy);
    }

    private static LaunchMetadata BuildLaunchMetadata(OutputInfo output, PyNTTTargetOptions targetOptions, Dictionary<string, object>? extraMeta = null)
    {
        var numel = Product(output.Shape);
        var meta = new Dictionary<string, object>
        {
            ["numel_expr"] = numel.PythonExpression,
            ["placement"] = "cuda",
        };
        if (extraMeta is not null)
        {
            foreach (var pair in extraMeta)
            {
                meta[pair.Key] = pair.Value;
            }
        }

        return new LaunchMetadata(
            meta,
            new(
                new()
                {
                    ["block_size"] = new TuningParameterMetadata("search_space", ElementwiseBlockSizeSearchSpace),
                }),
            BuildShardingMetadata(output, targetOptions),
            8,
            null);
    }

    private static ShardMetadata BuildShardingMetadata(OutputInfo output, PyNTTTargetOptions targetOptions)
    {
        if (output.DistributedType is { } distributedType)
        {
            return new ShardMetadata(
                "local_shard",
                string.IsNullOrWhiteSpace(distributedType.Placement.NormalizedHierarchyNames) ? "b" : distributedType.Placement.NormalizedHierarchyNames,
                GetTensorAxis(distributedType),
                "grid[0]",
                distributedType.Placement.Hierarchy.ToArray(),
                distributedType.Placement.NormalizedHierarchyLevels,
                ToPythonExpressions(output.Shape));
        }

        return new ShardMetadata(
            "local_shard",
            GetBlockPlacementAxis(targetOptions),
            0,
            "grid[0]",
            GetBlockHierarchy(targetOptions),
            GetBlockHierarchyLevels(targetOptions),
            ToPythonExpressions(output.Shape));
    }

    private static int GetTensorAxis(DistributedType distributedType)
    {
        for (var index = 0; index < distributedType.AxisPolicies.Count; index++)
        {
            if (distributedType.AxisPolicies[index] is SBPSplit)
            {
                return index;
            }
        }

        return 0;
    }

    private static string GetBlockPlacementAxis(PyNTTTargetOptions targetOptions)
    {
        var hierarchy = GetBlockHierarchy(targetOptions);
        var hierarchyNames = Placement.NormalizeAxisString(string.IsNullOrWhiteSpace(targetOptions.HierarchyNames)
            ? "b"
            : targetOptions.HierarchyNames);
        var hierarchyLevels = GetBlockHierarchyLevels(targetOptions);
        if (hierarchyNames.Length != hierarchy.Length)
        {
            hierarchyNames = new string('b', hierarchy.Length);
        }

        var blockAxes = hierarchyNames.Zip(hierarchyLevels).Where(pair => pair.Second == 'b').Select(pair => pair.First);
        var placementAxis = string.Concat(blockAxes);
        return string.IsNullOrEmpty(placementAxis) ? "b" : placementAxis;
    }

    private static int[] GetBlockHierarchy(PyNTTTargetOptions targetOptions)
    {
        var hierarchies = targetOptions.Hierarchies;
        return hierarchies.Length == 0 ? new[] { 1 } : hierarchies[0];
    }

    private static string GetBlockHierarchyLevels(PyNTTTargetOptions targetOptions)
    {
        var hierarchy = GetBlockHierarchy(targetOptions);
        return Placement.NormalizeHierarchyLevels(targetOptions.HierarchyLevels, targetOptions.HierarchyNames, hierarchy.Length);
    }

    private static string BuildGeneratedTopKernelPython(GeneratedKernelMetadata kernel, string bodySource, string helperSource)
    {
        var inputs = kernel.Inputs.Select((_, index) => $"input{index}").ToArray();
        var outputs = kernel.Outputs.Select((_, index) => $"output{index}").ToArray();
        var workspaceParameters = new[] { "data", "rdata", "chip_local_rdata", "block_local_rdata", "block_local_data" };
        var runtimeShapeArgs = GetRuntimeShapeArgs(kernel);
        var gridBarrierParameters = kernel.Attrs.ContainsKey("requires_grid_barrier")
            ? new[] { "pyntt_grid_mesh: tl.constexpr" }
            : Array.Empty<string>();
        var parameters = string.Join(", ", inputs.Concat(outputs).Concat(workspaceParameters).Concat(runtimeShapeArgs).Concat(gridBarrierParameters).Concat(new[] { "numel", "block_size: tl.constexpr" }));
        if (kernel.Attrs.TryGetValue("abi_view_stride_args", out var abiStrideArgsValue))
        {
            var abiStrideArgs = abiStrideArgsValue switch
            {
                string[] array => array,
                IEnumerable<string> enumerable => enumerable.ToArray(),
                _ => throw new NotSupportedException($"PyNTT kernel {kernel.Name} has unsupported abi_view_stride_args metadata type {abiStrideArgsValue.GetType().Name}."),
            };
            parameters = string.Join(", ", inputs.Concat(outputs).Concat(abiStrideArgs).Concat(workspaceParameters).Concat(runtimeShapeArgs).Concat(gridBarrierParameters).Concat(new[] { "numel", "block_size: tl.constexpr" }));
        }

        var call = IndentGeneratedCall(bodySource);

        var topKernelSource = $$"""
            @triton.jit
            def {{kernel.Name}}({{parameters}}):
                {{call}}
            """;
        return string.IsNullOrWhiteSpace(helperSource)
            ? topKernelSource
            : $"{helperSource.TrimEnd()}{Environment.NewLine}{Environment.NewLine}{topKernelSource}";
    }

    private static string IndentGeneratedCall(string call)
    {
        return call.Replace(Environment.NewLine, Environment.NewLine + "    ", StringComparison.Ordinal);
    }

    private static string GetTensorName(BaseExpr expr, IReadOnlyDictionary<IVar, string> parameterNames)
    {
        while (expr is Call call && call.Target is Boxing)
        {
            expr = call.Arguments[0];
        }

        if (expr is TIR.Buffer buffer
            && IsAbiBuffer(buffer)
            && buffer.MemSpan.Buffer.Start is IVar bufferParameter
            && parameterNames.TryGetValue(bufferParameter, out var bufferParameterName))
        {
            return bufferParameterName;
        }

        if (expr is IVar parameter && parameterNames.TryGetValue(parameter, out var name))
        {
            return name;
        }

        throw new NotSupportedException($"PyNTT M3 elementwise kernels only support direct function parameter operands, got {expr.GetType().Name}.");
    }

    private static bool IsAbiBuffer(TIR.Buffer buffer)
        => buffer.MemSpan.Buffer.Location is MemoryLocation.Input or MemoryLocation.Output;

    private static DistributedType? GetDistributedType(BaseExpr expr)
    {
        return expr switch
        {
            Nncase.TIR.Buffer buffer => buffer.DistributedType,
            _ => expr.CheckedType as DistributedType,
        };
    }

    private static PyNTTDimExpression[] GetTensorShape(BaseExpr expr, string name)
    {
        var tensorType = GetTensorType(expr.CheckedType, name);
        return GetRankedShape(tensorType, name).Dimensions.ToArray()
            .Select(dimension => new PyNTTDimExpressionEmitter().Emit(dimension))
            .ToArray();
    }

    private static int? GetShardAxis(BaseExpr expr)
    {
        return GetDistributedType(expr) is { } distributedType ? GetShardAxis(distributedType) : null;
    }

    private static int? GetShardAxis(DistributedType distributedType)
    {
        for (var index = 0; index < distributedType.AxisPolicies.Count; index++)
        {
            if (distributedType.AxisPolicies[index] is SBPSplit)
            {
                return index;
            }
        }

        return null;
    }

    private static long GetBufferOffsetBytes(TIR.Buffer buffer)
    {
        var byteOffset = GetFixedDimension(buffer.MemSpan.Buffer.Start, $"{buffer.MemSpan.Buffer.Location} physical buffer offset") +
            GetFixedDimension(buffer.MemSpan.Start, $"{buffer.MemSpan.Buffer.Location} memspan offset");
        return byteOffset;
    }

    private static long GetPoolSizeBytes(IReadOnlyDictionary<Const, ValueRange<ulong>> ranges)
    {
        return ranges.Count == 0 ? 0L : checked((long)ranges.Values.Max(range => range.Max));
    }

    private static long GetTensorMaxSizeBytes(IReadOnlyList<PyNTTDimExpression> shape, int vectorLaneCount, int scalarElementSizeBytes, string context)
    {
        var elements = 1L;
        foreach (var dimension in shape)
        {
            if (dimension.MaxValue is not { } maxValue)
            {
                throw new NotSupportedException($"{context} requires bounded dimension for collective staging, got {dimension}.");
            }

            elements = checked(elements * maxValue);
        }

        return checked(elements * vectorLaneCount * scalarElementSizeBytes);
    }

    private static long GetFixedDimension(BaseExpr expr, string name)
    {
        try
        {
            return expr switch
            {
                None => 0,
                DimConst dimConst => dimConst.Value,
                Dimension dimension when dimension.IsFixed => dimension.FixedValue,
                TensorConst tensorConst => tensorConst.Value.ToScalar<long>(),
                _ => expr.Evaluate().AsTensor().ToScalar<long>(),
            };
        }
        catch (Exception ex) when (ex is InvalidCastException or NotSupportedException)
        {
            throw new NotSupportedException($"PyNTT requires fixed {name}, got {expr}.", ex);
        }
    }

    private static string GetOpName(Enum op)
    {
        return op switch
        {
            UnaryOp.Abs => "abs",
            UnaryOp.Acos => "acos",
            UnaryOp.Acosh => "acosh",
            UnaryOp.Asin => "asin",
            UnaryOp.Asinh => "asinh",
            UnaryOp.Ceil => "ceil",
            UnaryOp.Cos => "cos",
            UnaryOp.Cosh => "cosh",
            UnaryOp.Erf => "erf",
            UnaryOp.Exp => "exp",
            UnaryOp.Floor => "floor",
            UnaryOp.Log => "log",
            UnaryOp.Neg => "neg",
            UnaryOp.Round => "round",
            UnaryOp.Rsqrt => "rsqrt",
            UnaryOp.Sin => "sin",
            UnaryOp.Sinh => "sinh",
            UnaryOp.Sqrt => "sqrt",
            UnaryOp.Square => "square",
            UnaryOp.Tanh => "tanh",
            UnaryOp.LogicalNot => "logical_not",
            UnaryOp.Sign => "sign",
            BinaryOp.Add => "add",
            BinaryOp.Sub => "sub",
            BinaryOp.Mul => "mul",
            BinaryOp.Div => "div",
            BinaryOp.Mod => "mod",
            BinaryOp.Min => "min",
            BinaryOp.Max => "max",
            _ => throw new NotSupportedException($"Unsupported PyNTT elementwise op: {op}."),
        };
    }

    private static string GetUnaryExpression(UnaryOp op)
    {
        return op switch
        {
            UnaryOp.Abs => "tl.abs(value0)",
            UnaryOp.Acos => "libdevice.acos(value0_f32)",
            UnaryOp.Acosh => "tl.log(value0_f32 + tl.sqrt(value0_f32 * value0_f32 - 1.0))",
            UnaryOp.Asin => "libdevice.asin(value0_f32)",
            UnaryOp.Asinh => "tl.log(value0_f32 + tl.sqrt(value0_f32 * value0_f32 + 1.0))",
            UnaryOp.Ceil => "tl.ceil(value0)",
            UnaryOp.Cos => "tl.cos(value0_f32)",
            UnaryOp.Cosh => "(tl.exp(value0_f32) + tl.exp(-value0_f32)) * 0.5",
            UnaryOp.Erf => "tl.erf(value0_f32)",
            UnaryOp.Exp => "tl.exp(value0_f32)",
            UnaryOp.Floor => "tl.floor(value0)",
            UnaryOp.Log => "tl.log(value0_f32)",
            UnaryOp.Neg => "-value0",
            UnaryOp.Round => "libdevice.round(value0)",
            UnaryOp.Rsqrt => "tl.rsqrt(value0_f32)",
            UnaryOp.Sin => "tl.sin(value0_f32)",
            UnaryOp.Sinh => "(tl.exp(value0_f32) - tl.exp(-value0_f32)) * 0.5",
            UnaryOp.Sqrt => "tl.sqrt(value0_f32)",
            UnaryOp.Square => "value0 * value0",
            UnaryOp.Tanh => "(tl.exp(value0_f32 + value0_f32) - 1.0) / (tl.exp(value0_f32 + value0_f32) + 1.0)",
            UnaryOp.LogicalNot => "value0 == 0",
            UnaryOp.Sign => "tl.where(value0 > 0, 1.0, tl.where(value0 < 0, -1.0, 0.0))",
            _ => throw new NotSupportedException($"Unsupported PyNTT unary template op: {op}."),
        };
    }

    private static string GetGluTypeName(GluType gluType)
    {
        return gluType switch
        {
            GluType.SwiGLU => "swiglu",
            _ => throw new NotSupportedException($"Unsupported PyNTT MatMulGlu type: {gluType}."),
        };
    }

    private static string GetBinaryExpression(BinaryOp op)
    {
        return op switch
        {
            BinaryOp.Add => "value0 + value1",
            BinaryOp.Sub => "value0 - value1",
            BinaryOp.Mul => "value0 * value1",
            BinaryOp.Div => "value0 / value1",
            BinaryOp.Mod => "value0 % value1",
            BinaryOp.Min => "tl.minimum(value0, value1)",
            BinaryOp.Max => "tl.maximum(value0, value1)",
            _ => throw new NotSupportedException($"Unsupported PyNTT binary template op: {op}."),
        };
    }

    private static string GetCompareOpName(CompareOp op)
    {
        return op switch
        {
            CompareOp.Equal => "equal",
            CompareOp.NotEqual => "not_equal",
            CompareOp.LowerThan => "less",
            CompareOp.LowerOrEqual => "less_or_equal",
            CompareOp.GreaterThan => "greater",
            CompareOp.GreaterOrEqual => "greater_or_equal",
            _ => throw new NotSupportedException($"Unsupported PyNTT compare op: {op}."),
        };
    }

    private static string GetCompareExpression(CompareOp op)
    {
        return op switch
        {
            CompareOp.Equal => "value0 == value1",
            CompareOp.NotEqual => "value0 != value1",
            CompareOp.LowerThan => "value0 < value1",
            CompareOp.LowerOrEqual => "value0 <= value1",
            CompareOp.GreaterThan => "value0 > value1",
            CompareOp.GreaterOrEqual => "value0 >= value1",
            _ => throw new NotSupportedException($"Unsupported PyNTT compare template op: {op}."),
        };
    }

    private static string GetCompareOperator(CompareOp op)
    {
        return op switch
        {
            CompareOp.Equal => "==",
            CompareOp.NotEqual => "!=",
            CompareOp.LowerThan => "<",
            CompareOp.LowerOrEqual => "<=",
            CompareOp.GreaterThan => ">",
            CompareOp.GreaterOrEqual => ">=",
            _ => throw new NotSupportedException($"Unsupported PyNTT compare op: {op}."),
        };
    }

    private static string BuildScalarBinaryExpression(BinaryOp op, string lhs, string rhs)
    {
        return op switch
        {
            BinaryOp.Add => $"({lhs} + {rhs})",
            BinaryOp.Sub => $"({lhs} - {rhs})",
            BinaryOp.Mul => $"({lhs} * {rhs})",
            BinaryOp.Div => $"({lhs} / {rhs})",
            BinaryOp.Mod => $"({lhs} % {rhs})",
            BinaryOp.Min => $"tl.minimum({lhs}, {rhs})",
            BinaryOp.Max => $"tl.maximum({lhs}, {rhs})",
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
            _ => throw new NotSupportedException($"Unsupported PyNTT scalar binary op: {op}."),
        };
    }

    private static string FormatScalarConst(TensorConst tensorConst)
    {
        if (!tensorConst.Value.Shape.IsScalar)
        {
            throw new NotSupportedException("PyNTT scalar expression only supports scalar TensorConst values.");
        }

        var value = tensorConst.Value[Array.Empty<long>()];
        return value switch
        {
            bool boolean => boolean ? "True" : "False",
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => value?.ToString() ?? "None",
        };
    }

    private static string GetClampExpression(float min, float max)
    {
        return (float.IsNegativeInfinity(min), float.IsPositiveInfinity(max)) switch
        {
            (true, true) => "value0",
            (true, false) => $"tl.minimum(value0, {FormatFloat(max)})",
            (false, true) => $"tl.maximum(value0, {FormatFloat(min)})",
            (false, false) => $"tl.minimum(tl.maximum(value0, {FormatFloat(min)}), {FormatFloat(max)})",
        };
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    private static string GetCastModeName(CastMode castMode)
    {
        return castMode switch
        {
            CastMode.KDefault => "kdefault",
            CastMode.Exact => "exact",
            CastMode.CheckOverflow => "check_overflow",
            CastMode.Reinterpret => "reinterpret",
            _ => throw new NotSupportedException($"Unsupported PyNTT cast mode: {castMode}."),
        };
    }

    private static string GetCastExpression(CastMode castMode, string outputTritonDType)
    {
        return castMode switch
        {
            CastMode.KDefault or CastMode.Exact or CastMode.CheckOverflow => $"value0.to({outputTritonDType})",
            CastMode.Reinterpret => $"value0.to({outputTritonDType}, bitcast=True)",
            _ => throw new NotSupportedException($"Unsupported PyNTT cast mode: {castMode}."),
        };
    }

    private static string GetReduceOpName(ReduceOp op)
    {
        return op switch
            {
                ReduceOp.Sum => "sum",
                ReduceOp.Mean => "mean",
                ReduceOp.Max => "max",
                ReduceOp.Min => "min",
                _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static string GetReduceInitValue(ReduceOp op)
    {
            return op switch
            {
                ReduceOp.Sum => "0.0",
                ReduceOp.Mean => "0.0",
                ReduceOp.Max => "-float(\"inf\")",
                ReduceOp.Min => "float(\"inf\")",
                _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static string GetReduceUpdateExpression(ReduceOp op)
    {
        return op switch
        {
            ReduceOp.Sum => "acc + value0",
            ReduceOp.Mean => "acc + value0",
            ReduceOp.Max => "tl.maximum(acc, value0)",
            ReduceOp.Min => "tl.minimum(acc, value0)",
            _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static string GetReduceFinalizeExpression(ReduceOp op, PyNTTDimExpression reduceElementCount)
    {
        return op switch
        {
            ReduceOp.Sum or ReduceOp.Max or ReduceOp.Min => "acc",
            ReduceOp.Mean => $"acc / (({reduceElementCount.TritonExpression}) + lane * 0).to(tl.float32)",
            _ => throw new NotSupportedException($"Unsupported PyNTT reduce op: {op}."),
        };
    }

    private static bool GetScalarBool(BaseExpr expr, string name)
    {
        if (expr is None)
        {
            return false;
        }

        try
        {
            return expr.Evaluate().AsTensor().ToScalar<bool>();
        }
        catch (Exception ex) when (ex is InvalidCastException or NotSupportedException)
        {
            throw new NotSupportedException($"PyNTT requires a bool scalar for {name}.", ex);
        }
    }

    private static float GetScalarFloat(BaseExpr expr, string name, float defaultValue)
    {
        if (expr is None)
        {
            return defaultValue;
        }

        try
        {
            return expr.Evaluate().AsTensor().ToScalar<float>();
        }
        catch (Exception ex) when (ex is InvalidCastException or NotSupportedException)
        {
            throw new NotSupportedException($"PyNTT requires a float32 scalar for {name}.", ex);
        }
    }

    private static int NormalizeAxis(int axis, int rank, string context)
    {
        var normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
        {
            throw new NotSupportedException($"{context} axis {axis} is out of range for rank {rank}.");
        }

        return normalized;
    }

    private static int[] NormalizeAxes(IReadOnlyList<int> axes, int rank, string context)
    {
        if (axes.Count == 0)
        {
            throw new NotSupportedException($"{context} requires at least one reduction axis.");
        }

        var normalizedAxes = axes
            .Select(axis => NormalizeAxis(axis, rank, context))
            .OrderBy(axis => axis)
            .ToArray();
        for (var i = 1; i < normalizedAxes.Length; i++)
        {
            if (normalizedAxes[i] == normalizedAxes[i - 1])
            {
                throw new NotSupportedException($"{context} contains duplicated axis {normalizedAxes[i]}.");
            }
        }

        return normalizedAxes;
    }

    private static bool SameDim(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (lhs.FixedValue.HasValue && rhs.FixedValue.HasValue)
        {
            return lhs.FixedValue.Value == rhs.FixedValue.Value;
        }

        return lhs.PythonExpression == rhs.PythonExpression ||
            lhs.TritonExpression == rhs.TritonExpression;
    }

    private static bool CanFitPaddedDim(PyNTTDimExpression actual, PyNTTDimExpression padded)
    {
        if (actual.FixedValue.HasValue && padded.FixedValue.HasValue)
        {
            return actual.FixedValue.Value <= padded.FixedValue.Value;
        }

        return SameDim(actual, padded);
    }

    private static string ShapeText(IEnumerable<PyNTTDimExpression> shape)
        => string.Join(",", shape.Select(dim => dim.PythonExpression));

    private static long RequireFixedDim(PyNTTDimExpression dimension, string context)
        => dimension.FixedValue ?? throw new NotSupportedException($"{context} must be fixed for the current PyNTT codegen path, got {dimension.PythonExpression}.");

    private static PyNTTDimExpression ToDim(long value)
    {
        var text = value.ToString(CultureInfo.InvariantCulture);
        return new(text, text, value);
    }

    private static PyNTTDimExpression AddDims(PyNTTDimExpression lhs, PyNTTDimExpression rhs)
    {
        if (lhs.FixedValue == 0)
        {
            return rhs;
        }

        if (rhs.FixedValue == 0)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue && rhs.FixedValue.HasValue
            ? checked(lhs.FixedValue.Value + rhs.FixedValue.Value)
            : null;
        return new($"({lhs.PythonExpression} + {rhs.PythonExpression})", $"({lhs.TritonExpression} + {rhs.TritonExpression})", fixedValue);
    }

    private static PyNTTDimExpression MultiplyDim(PyNTTDimExpression lhs, long rhs)
    {
        if (rhs == 0 || lhs.FixedValue == 0)
        {
            return PyNTTDimExpression.Zero;
        }

        if (rhs == 1)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue ? checked(lhs.FixedValue.Value * rhs) : null;
        return new($"({lhs.PythonExpression} * {rhs.ToString(CultureInfo.InvariantCulture)})", $"({lhs.TritonExpression} * {rhs.ToString(CultureInfo.InvariantCulture)})", fixedValue);
    }

    private static int GetSingleVectorLaneCount(DataType dataType, string context)
    {
        var lanes = GetVectorLanes(dataType);
        return lanes.Length switch
        {
            0 => 1,
            1 => lanes[0],
            _ => throw new NotSupportedException($"{context} expects at most one vector lane dimension, got [{string.Join(",", lanes)}]."),
        };
    }

    private static int GetVectorLaneElementCount(DataType dataType)
    {
        var lanes = GetVectorLanes(dataType);
        return lanes.Length == 0 ? 1 : lanes.Aggregate(1, (acc, lane) => checked(acc * lane));
    }

    private static void ValidateMatchingVectorLanes(string context, DataType lhs, DataType rhs)
    {
        var lhsLanes = GetVectorLanes(lhs);
        var rhsLanes = GetVectorLanes(rhs);
        if (!lhsLanes.SequenceEqual(rhsLanes))
        {
            throw new NotSupportedException($"{context} expects matching vector lanes, got lhs=[{string.Join(",", lhsLanes)}], rhs=[{string.Join(",", rhsLanes)}].");
        }
    }

    private static void ValidateScalarOrMatchingVectorLanes(string context, DataType operand, DataType output)
    {
        var operandLanes = GetVectorLanes(operand);
        var outputLanes = GetVectorLanes(output);
        if (operandLanes.Length != 0 && !operandLanes.SequenceEqual(outputLanes))
        {
            throw new NotSupportedException($"{context} expects scalar lanes or lanes matching output, got operand=[{string.Join(",", operandLanes)}], output=[{string.Join(",", outputLanes)}].");
        }
    }

    private static PyNTTDimExpression[] GetLogicalVectorShape(IReadOnlyList<PyNTTDimExpression> physicalShape, int laneCount)
    {
        if (laneCount == 1)
        {
            return physicalShape.ToArray();
        }

        if (physicalShape.Count == 0)
        {
            throw new NotSupportedException("PyNTT vector scalar buffers are not supported by the current Triton templates.");
        }

        var logicalShape = physicalShape.ToArray();
        logicalShape[^1] = MultiplyDim(logicalShape[^1], laneCount);
        return logicalShape;
    }

    private static PyNTTDimExpression[] GetLogicalVectorShape(IReadOnlyList<PyNTTDimExpression> physicalShape, int vectorizedAxis, int laneCount)
    {
        if (laneCount == 1)
        {
            return physicalShape.ToArray();
        }

        if (physicalShape.Count == 0)
        {
            throw new NotSupportedException("PyNTT vector scalar buffers are not supported by the current Triton templates.");
        }

        var axis = NormalizeAxis(vectorizedAxis, physicalShape.Count, "PyNTT vectorized axis");
        var logicalShape = physicalShape.ToArray();
        logicalShape[axis] = MultiplyDim(logicalShape[axis], laneCount);
        return logicalShape;
    }

    private static PyNTTDimExpression CeilDivDim(PyNTTDimExpression lhs, long rhs)
    {
        if (rhs <= 0)
        {
            throw new NotSupportedException($"PyNTT dimension ceil-div requires positive divisor, got {rhs}.");
        }

        if (rhs == 1)
        {
            return lhs;
        }

        long? fixedValue = lhs.FixedValue.HasValue ? checked((lhs.FixedValue.Value + rhs - 1) / rhs) : null;
        var rhsText = rhs.ToString(CultureInfo.InvariantCulture);
        return new(
            $"(({lhs.PythonExpression} + {rhsText} - 1) // ({rhsText}))",
            $"(({lhs.TritonExpression} + {rhsText} - 1) // ({rhsText}))",
            fixedValue);
    }

    private static void ValidateSameShape(string context, IReadOnlyList<PyNTTDimExpression> actual, IReadOnlyList<PyNTTDimExpression> expected)
    {
        if (actual.Count != expected.Count || actual.Zip(expected).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} requires matching shapes, got [{ShapeText(actual)}] and [{ShapeText(expected)}].");
        }
    }

    private static void ValidatePackShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
    {
        ValidateSameRank(context, inputShape, outputShape);
        for (var axis = 0; axis < inputShape.Count; axis++)
        {
            var laneProduct = GetLayoutAxisLaneProduct(axes, lanes, axis);
            var expected = laneProduct > 1 ? CeilDivDim(inputShape[axis], laneProduct) : inputShape[axis];
            if (!SameDim(outputShape[axis], expected))
            {
                throw new NotSupportedException($"{context} output shape mismatch at axis {axis}, expected {expected}, got {outputShape[axis]}.");
            }
        }
    }

    private static void ValidateUnpackShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, IReadOnlyList<int> axes, IReadOnlyList<int> lanes)
    {
        ValidateSameRank(context, inputShape, outputShape);
        for (var axis = 0; axis < outputShape.Count; axis++)
        {
            var laneProduct = GetLayoutAxisLaneProduct(axes, lanes, axis);
            var expected = laneProduct > 1 ? MultiplyDim(inputShape[axis], laneProduct) : inputShape[axis];
            if (!SameDim(outputShape[axis], expected))
            {
                throw new NotSupportedException($"{context} output shape mismatch at axis {axis}, expected {expected}, got {outputShape[axis]}.");
            }
        }
    }

    private static int[] NormalizeLayoutAxes(IRArray<int> axes, int rank, string context)
    {
        var normalizedAxes = axes.IsDefaultOrEmpty
            ? Array.Empty<int>()
            : axes.ToArray().Select(axis => NormalizeAxis(axis, rank, context)).ToArray();

        return normalizedAxes;
    }

    private static int GetLayoutAxisLaneProduct(IReadOnlyList<int> axes, IReadOnlyList<int> lanes, int axis)
    {
        var product = 1;
        for (var i = 0; i < axes.Count; i++)
        {
            if (axes[i] == axis)
            {
                product = checked(product * lanes[i]);
            }
        }

        return product;
    }

    private static int[] GetLayoutLanes(IRArray<int> lanes, int axesCount, string context)
    {
        var laneArray = lanes.IsDefaultOrEmpty ? Array.Empty<int>() : lanes.ToArray();
        if (laneArray.Length != axesCount)
        {
            throw new NotSupportedException($"{context} expects lanes rank {axesCount}, got {laneArray.Length}.");
        }

        if (laneArray.Any(lane => lane <= 0))
        {
            throw new NotSupportedException($"{context} requires positive lanes, got [{string.Join(",", laneArray)}].");
        }

        return laneArray;
    }

    private static void ValidateLanePrefix(string context, IReadOnlyList<int> expected, IReadOnlyList<int> actual)
    {
        if (!expected.SequenceEqual(actual))
        {
            throw new NotSupportedException($"{context} mismatch, expected [{string.Join(",", expected)}], got [{string.Join(",", actual)}].");
        }
    }

    private static void ValidateRoPEShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> cosShape, IReadOnlyList<PyNTTDimExpression> sinShape, IReadOnlyList<PyNTTDimExpression> outputShape, int rotaryAxis, int laneCount, int sinCosVectorPackFactor)
    {
        ValidateSameShape(context, inputShape, outputShape);
        var sinCosOutputShape = outputShape.ToArray();
        if (sinCosVectorPackFactor > 1)
        {
            sinCosOutputShape[rotaryAxis] = CeilDivDim(sinCosOutputShape[rotaryAxis], sinCosVectorPackFactor);
        }

        ValidateBroadcastable($"{context} cos", cosShape, sinCosOutputShape);
        ValidateBroadcastable($"{context} sin", sinShape, sinCosOutputShape);
        if (outputShape.Count == 0)
        {
            throw new NotSupportedException($"{context} requires a non-scalar output, got [{ShapeText(outputShape)}].");
        }

        var rotaryDim = RequireFixedDim(MultiplyDim(outputShape[rotaryAxis], laneCount), $"{context} rotary dimension");
        if (rotaryDim % 2 != 0)
        {
            throw new NotSupportedException($"{context} requires an even rotary dimension, got axis {rotaryAxis} with shape [{ShapeText(outputShape)}] and lanes {laneCount}.");
        }
    }

    private static int GetRoPESinCosVectorPackFactor(IReadOnlyList<int> inputLanes, IReadOnlyList<int> cosLanes, IReadOnlyList<int> sinLanes, IReadOnlyList<int> outputLanes)
    {
        if (!inputLanes.SequenceEqual(outputLanes))
        {
            throw new NotSupportedException($"PyNTT RoPE expects matching input/output vector lanes, got input=[{string.Join(",", inputLanes)}], output=[{string.Join(",", outputLanes)}].");
        }

        if (!cosLanes.SequenceEqual(sinLanes))
        {
            throw new NotSupportedException($"PyNTT RoPE expects matching cos/sin vector lanes, got cos=[{string.Join(",", cosLanes)}], sin=[{string.Join(",", sinLanes)}].");
        }

        if (cosLanes.Count == 0 && outputLanes.Count == 0)
        {
            return 1;
        }

        if (cosLanes.SequenceEqual(outputLanes))
        {
            return 1;
        }

        if (outputLanes.Count == 1 && cosLanes.Count == 2 && cosLanes[0] == 2 && cosLanes[1] == outputLanes[0])
        {
            return 2;
        }

        throw new NotSupportedException($"PyNTT RoPE expects sin/cos lanes to be scalar, [{string.Join(",", outputLanes)}], or [2,{string.Join(",", outputLanes)}], got [{string.Join(",", cosLanes)}].");
    }

    private static void ValidateLayerNormShape(string context, IReadOnlyList<PyNTTDimExpression> parameterShape, IReadOnlyList<PyNTTDimExpression> outputShape, int axis)
    {
        var expectedShape = outputShape.Skip(axis).ToArray();
        if (parameterShape.Count != expectedShape.Length || parameterShape.Zip(expectedShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} requires shape [{ShapeText(expectedShape)}], got [{ShapeText(parameterShape)}].");
        }
    }

    private static void ValidateNormStatsShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> statsShape, int axis, bool useMean)
    {
        var expectedRank = inputShape.Count + 1;
        if (statsShape.Count != expectedRank)
        {
            throw new NotSupportedException($"{context} stats rank must be {expectedRank}, got [{ShapeText(statsShape)}].");
        }

        var expectedComponents = useMean ? 2 : 1;
        if (statsShape[0].FixedValue != expectedComponents)
        {
            throw new NotSupportedException($"{context} stats component dimension must be {expectedComponents}, got [{ShapeText(statsShape)}].");
        }

        for (var i = 0; i < inputShape.Count; i++)
        {
            if (i < axis)
            {
                if (!SameDim(inputShape[i], statsShape[i + 1]))
                {
                    throw new NotSupportedException($"{context} stats outer axis {i} must match input shape [{ShapeText(inputShape)}], got [{ShapeText(statsShape)}].");
                }
            }
            else if (!statsShape[i + 1].IsFixedOne)
            {
                throw new NotSupportedException($"{context} stats normalized axis {i} must have extent 1, got [{ShapeText(statsShape)}].");
            }
        }
    }

    private static void ValidateBroadcastable(string context, IReadOnlyList<PyNTTDimExpression> actual, IReadOnlyList<PyNTTDimExpression> expected)
    {
        if (actual.Count > expected.Count)
        {
            throw new NotSupportedException($"{context} requires broadcastable shape, got [{ShapeText(actual)}] for output [{ShapeText(expected)}].");
        }

        var axisOffset = expected.Count - actual.Count;
        for (var i = 0; i < actual.Count; i++)
        {
            var actualDim = actual[i];
            var expectedDim = expected[axisOffset + i];
            if (!actualDim.IsFixedOne && !SameDim(actualDim, expectedDim))
            {
                throw new NotSupportedException($"{context} requires broadcastable shape, got [{ShapeText(actual)}] for output [{ShapeText(expected)}].");
            }
        }
    }

    private static void ValidateReduceShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, IReadOnlyList<int> axes, bool keepDims)
    {
        var axisSet = axes.ToHashSet();
        if (keepDims)
        {
            if (outputShape.Count != inputShape.Count)
            {
                throw new NotSupportedException($"{context} with keep_dims expects output rank {inputShape.Count}, got [{ShapeText(outputShape)}].");
            }

            for (var axis = 0; axis < inputShape.Count; axis++)
            {
                var expected = axisSet.Contains(axis) ? PyNTTDimExpression.One : inputShape[axis];
                if (!SameDim(outputShape[axis], expected))
                {
                    throw new NotSupportedException($"{context} output shape mismatch at axis {axis}, expected {expected}, got {outputShape[axis]}.");
                }
            }

            return;
        }

        var expectedShape = inputShape
            .Select((dimension, axis) => (dimension, axis))
            .Where(item => !axisSet.Contains(item.axis))
            .Select(item => item.dimension)
            .ToArray();
        if (outputShape.Count != expectedShape.Length || outputShape.Zip(expectedShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} output shape mismatch, expected [{ShapeText(expectedShape)}], got [{ShapeText(outputShape)}].");
        }
    }

    private static void ValidatePadShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> outputShape, long[][] pads)
    {
        if (pads.Length != inputShape.Count || outputShape.Count != inputShape.Count)
        {
            throw new NotSupportedException($"{context} expects input, output, and pads to have the same rank.");
        }

        for (var axis = 0; axis < inputShape.Count; axis++)
        {
            var expected = AddDims(AddDims(inputShape[axis], ToDim(pads[axis][0])), ToDim(pads[axis][1]));
            if (!SameDim(outputShape[axis], expected))
            {
                throw new NotSupportedException($"{context} output shape mismatch at axis {axis}, expected {expected}, got {outputShape[axis]}.");
            }
        }
    }

    private static void ValidateSameRank(string context, IReadOnlyList<PyNTTDimExpression> lhsShape, IReadOnlyList<PyNTTDimExpression> rhsShape)
    {
        if (lhsShape.Count != rhsShape.Count)
        {
            throw new NotSupportedException($"{context} expects matching ranks, got {lhsShape.Count} and {rhsShape.Count}.");
        }
    }

    private static void ValidateScatterNDShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> indicesShape, IReadOnlyList<PyNTTDimExpression> updatesShape, IReadOnlyList<PyNTTDimExpression> outputShape)
    {
        if (inputShape.Count != outputShape.Count || inputShape.Zip(outputShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} expects input and output to have the same shape.");
        }

        if (indicesShape.Count < 1)
        {
            throw new NotSupportedException($"{context} expects indices rank >= 1.");
        }

        var indexDepth = RequireFixedDim(indicesShape[^1], $"{context} index depth");
        if (indexDepth < 1 || indexDepth > inputShape.Count)
        {
            throw new NotSupportedException($"{context} index depth {indexDepth} is out of range for input rank {inputShape.Count}.");
        }

        var expectedUpdatesShape = indicesShape.Take(indicesShape.Count - 1)
            .Concat(inputShape.Skip((int)indexDepth))
            .ToArray();
        if (updatesShape.Count != expectedUpdatesShape.Length || updatesShape.Zip(expectedUpdatesShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} updates shape mismatch, expected [{ShapeText(expectedUpdatesShape)}], got [{ShapeText(updatesShape)}].");
        }
    }

    private static (long[] Starts, long[] Strides) NormalizeSliceParameters(IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<long> starts, IReadOnlyList<int> axes, IReadOnlyList<int> strides, string context)
    {
        if (starts.Count != axes.Count || strides.Count != axes.Count)
        {
            throw new NotSupportedException($"{context} expects begins, axes, and strides to have the same length.");
        }

        var normalizedStarts = new long[inputShape.Count];
        var normalizedStrides = Enumerable.Repeat(1L, inputShape.Count).ToArray();
        var usedAxes = new HashSet<int>();
        for (var i = 0; i < axes.Count; i++)
        {
            var axis = NormalizeAxis(axes[i], inputShape.Count, context);
            if (!usedAxes.Add(axis))
            {
                throw new NotSupportedException($"{context} contains duplicated axis {axis}.");
            }

            var stride = strides[i];
            if (stride == 0)
            {
                throw new NotSupportedException($"{context} stride must not be zero.");
            }

            normalizedStarts[axis] = NormalizeSliceStart(starts[i], RequireFixedDim(inputShape[axis], $"{context} input dimension {axis}"), stride);
            normalizedStrides[axis] = stride;
        }

        return (normalizedStarts, normalizedStrides);
    }

    private static long NormalizeSliceStart(long start, long dim, long stride)
    {
        var normalized = start < 0 ? start + dim : start;
        if (stride > 0)
        {
            return Math.Clamp(normalized, 0, dim);
        }

        return Math.Clamp(normalized, 0, dim - 1);
    }

    private static long[] GetStaticRankedShape(BaseExpr expr, string context)
    {
        if (expr is not RankedShape shape)
        {
            throw new NotSupportedException($"{context} must be a static ranked shape.");
        }

        return shape.ToValueArray();
    }

    private static void ValidateGatherShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> indexShape, IReadOnlyList<PyNTTDimExpression> outputShape, int axis)
    {
        var expectedRank = inputShape.Count + indexShape.Count - 1;
        if (outputShape.Count != expectedRank)
        {
            throw new NotSupportedException($"{context} expects output rank {expectedRank}, got [{ShapeText(outputShape)}].");
        }

        for (var i = 0; i < axis; i++)
        {
            if (!SameDim(outputShape[i], inputShape[i]))
            {
                throw new NotSupportedException($"{context} output pre-axis shape mismatch, got [{ShapeText(outputShape)}] for input [{ShapeText(inputShape)}].");
            }
        }

        for (var i = 0; i < indexShape.Count; i++)
        {
            if (!SameDim(outputShape[axis + i], indexShape[i]))
            {
                throw new NotSupportedException($"{context} output index shape mismatch, got [{ShapeText(outputShape)}] for index [{ShapeText(indexShape)}].");
            }
        }

        for (var i = axis + 1; i < inputShape.Count; i++)
        {
            var outputAxis = i + indexShape.Count - 1;
            if (!SameDim(outputShape[outputAxis], inputShape[i]))
            {
                throw new NotSupportedException($"{context} output post-axis shape mismatch, got [{ShapeText(outputShape)}] for input [{ShapeText(inputShape)}].");
            }
        }
    }

    private static long[][] GetStaticPaddings(Paddings paddings, string context)
    {
        if (!paddings.IsFixed)
        {
            throw new NotSupportedException($"{context} requires fixed paddings.");
        }

        var pads = paddings.ToValueArray();
        return Enumerable.Range(0, paddings.Count)
            .Select(axis => new[] { pads[axis, 0], pads[axis, 1] })
            .ToArray();
    }

    private static void ValidateRank(string context, IReadOnlyList<PyNTTDimExpression> shape, int rank)
    {
        if (shape.Count != rank)
        {
            throw new NotSupportedException($"{context} requires rank {rank}, got [{ShapeText(shape)}].");
        }
    }

    private static void ValidateMinimumRank(string context, IReadOnlyList<PyNTTDimExpression> shape, int rank)
    {
        if (shape.Count < rank)
        {
            throw new NotSupportedException($"{context} requires rank >= {rank}, got [{ShapeText(shape)}].");
        }
    }

    private static void EnsureEmpty(string context, IRArray<int> values)
    {
        var valueArray = values.IsDefaultOrEmpty ? Array.Empty<int>() : values.ToArray();
        if (valueArray.Length > 0)
        {
            throw new NotSupportedException($"{context} must be empty for the current PyNTT Triton templates, got [{string.Join(",", valueArray)}].");
        }
    }

    private static string GetTritonDType(DataType dataType)
    {
        return GetPyNTTDTypeName(GetScalarDataType(dataType)) switch
        {
            "bool" => "tl.int1",
            "int8" => "tl.int8",
            "uint8" => "tl.uint8",
            "int16" => "tl.int16",
            "uint16" => "tl.uint16",
            "int32" => "tl.int32",
            "uint32" => "tl.uint32",
            "int64" => "tl.int64",
            "uint64" => "tl.uint64",
            "float16" => "tl.float16",
            "bfloat16" => "tl.bfloat16",
            "float32" => "tl.float32",
            "float64" => "tl.float64",
            var name => throw new NotSupportedException($"Unsupported PyNTT Triton dtype: {name}."),
        };
    }

    private static string GetPyNTTScalarDTypeName(DataType dataType) => GetPyNTTDTypeName(GetScalarDataType(dataType));

    private static string GetScalarTritonDType(DataType dataType) => GetTritonDType(GetScalarDataType(dataType));

    private static DataType GetScalarDataType(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => vectorType.ElemType,
            MaskVectorType => DataTypes.Boolean,
            _ => dataType,
        };
    }

    private static int[] GetVectorLanes(DataType dataType)
    {
        return dataType switch
        {
            VectorType vectorType => vectorType.Lanes.ToArray(),
            MaskVectorType maskVectorType => new[] { maskVectorType.Lanes },
            _ => Array.Empty<int>(),
        };
    }

    private static int[][] GetSplitAxes(DistributedType distributedType)
    {
        var rank = distributedType.TensorType.Shape.Rank;
        return Enumerable.Range(0, rank)
            .Select(axis => axis < distributedType.AxisPolicies.Count && distributedType.AxisPolicies[axis] is SBPSplit split
                ? split.Axes.ToArray()
                : Array.Empty<int>())
            .ToArray();
    }

    private static bool IsObjectExpression(BaseExpr expr) => IsObjectDataType(expr.CheckedDataType);

    private static bool IsObjectDataType(DataType dataType) => dataType is ReferenceType;

    private static OutputInfo[] GetOutputInfos(BaseFunction function)
    {
        return BuildOutputInfos(PyNTTFunctionOutputs.GetOutputs(function));
    }

    private static OutputInfo[] BuildOutputInfos(IEnumerable<BufferVar> outputs)
    {
        return outputs
            .Select((output, index) =>
            {
                var type = output.CheckedType;
                var tensorType = GetTensorType(type, $"output{index}");
                return new OutputInfo(
                    $"output{index}",
                    output.Name,
                    GetRankedShape(tensorType, $"output{index}").Dimensions.ToArray()
                        .Select(dimension => new PyNTTDimExpressionEmitter().Emit(dimension))
                        .ToArray(),
                    tensorType.DType,
                    type as DistributedType);
            })
            .ToArray();
    }

    private static BaseExpr UnwrapInputBoxing(BaseExpr expr)
    {
        while (expr is Call call && call.Target is Boxing)
        {
            expr = call.Arguments[0];
        }

        return expr;
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

    private static RankedShape GetRankedShape(TensorType tensorType, string name)
    {
        if (tensorType.Shape is not RankedShape shape)
        {
            throw new NotSupportedException($"PyNTT requires ranked shape for {name}, got {tensorType.Shape}.");
        }

        return shape;
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

    private static Dimension[] GetRankedShapeDimensions(Shape shape, string name)
    {
        if (shape is not RankedShape rankedShape)
        {
            throw new NotSupportedException($"PyNTT requires ranked shape for {name}, got {shape}.");
        }

        return rankedShape.Dimensions.ToArray();
    }

    private static PyNTTDimExpression Product(IEnumerable<PyNTTDimExpression> values)
    {
        var array = values.ToArray();
        if (array.Length == 0)
        {
            return PyNTTDimExpression.One;
        }

        long? fixedValue = array.All(value => value.FixedValue.HasValue)
            ? array.Aggregate(1L, (product, value) => checked(product * value.FixedValue!.Value))
            : null;
        return new(
            string.Join(" * ", array.Select(value => $"({value.PythonExpression})")),
            string.Join(" * ", array.Select(value => $"({value.TritonExpression})")),
            fixedValue);
    }

    private static string[] ToPythonExpressions(IEnumerable<PyNTTDimExpression> values)
        => values.Select(value => value.PythonExpression).ToArray();

    private static string[] GetRuntimeShapeArgs(GeneratedKernelMetadata kernel)
        => kernel.Attrs.TryGetValue("runtime_shape_args", out var value) && value is string[] args
            ? args
            : Array.Empty<string>();

    private static string SanitizePythonIdentifier(string value)
    {
        var chars = value.Select(ch => char.IsAsciiLetterOrDigit(ch) || ch == '_' ? ch : '_').ToArray();
        if (chars.Length == 0 || char.IsDigit(chars[0]))
        {
            return "_" + new string(chars);
        }

        return new string(chars);
    }

    private sealed record OutputInfo(string Name, string AbiName, PyNTTDimExpression[] Shape, DataType DType, DistributedType? DistributedType);

    private sealed class SharedHelperRegistry
    {
        private readonly Dictionary<string, string> _names = new(StringComparer.Ordinal);
        private int _nextIndex;

        public bool TryGetName(string key, out string name) => _names.TryGetValue(key, out name!);

        public string Add(string key, string ownerName, string kind)
        {
            var name = SanitizePythonIdentifier($"{ownerName}_shared_{kind}_{_nextIndex++}");
            _names.Add(key, name);
            return name;
        }
    }
}
#pragma warning restore SA1201

internal sealed record PyNTTGeneratedKernelSource(
    [property: JsonPropertyName("generated_kernels")]
    IReadOnlyList<GeneratedKernelMetadata> Kernels,
    [property: JsonPropertyName("render_kernels")]
    IReadOnlyList<KernelRenderSpec> RenderKernels,
    [property: JsonPropertyName("source")]
    string Source);

internal sealed record GeneratedPrimFunctionKernel(
    GeneratedKernelMetadata Metadata,
    KernelRenderSpec RenderSpec);

internal sealed record KernelRenderSpec(
    [property: JsonPropertyName("metadata")]
    GeneratedKernelMetadata Metadata,
    [property: JsonPropertyName("helpers")]
    IReadOnlyList<HelperTemplateRenderSpec> Helpers,
    [property: JsonPropertyName("body_source")]
    string BodySource);

internal sealed record KernelInputLayout(
    string[] Names,
    IReadOnlyDictionary<int, int> IndexMap,
    IReadOnlySet<int> RemovedIndexes,
    string BodySource,
    HelperTemplateRenderSpec[] Helpers);

internal sealed record HelperTemplateRenderSpec(
    [property: JsonPropertyName("template")]
    string Template,
    [property: JsonPropertyName("model")]
    object Model,
    [property: JsonPropertyName("arguments")]
    string[] Arguments);

internal sealed record GeneratedKernelMetadata(
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("op_kind")]
    string OpKind,
    [property: JsonPropertyName("inputs")]
    string[] Inputs,
    [property: JsonPropertyName("outputs")]
    string[] Outputs,
    [property: JsonPropertyName("attrs")]
    Dictionary<string, object> Attrs,
    [property: JsonPropertyName("launch")]
    LaunchMetadata Launch);

internal sealed record HelperKernelCallMetadata(
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("arguments")]
    string[] Arguments);

internal sealed record LaunchMetadata(
    [property: JsonPropertyName("meta")]
    Dictionary<string, object> Meta,
    [property: JsonPropertyName("tuning")]
    KernelTuningMetadata Tuning,
    [property: JsonPropertyName("sharding")]
    ShardMetadata Sharding,
    [property: JsonPropertyName("num_warps")]
    int? NumWarps,
    [property: JsonPropertyName("num_stages")]
    int? NumStages);

internal sealed record ShardMetadata(
    [property: JsonPropertyName("strategy")]
    string Strategy,
    [property: JsonPropertyName("placement_axis")]
    string PlacementAxis,
    [property: JsonPropertyName("tensor_axis")]
    int TensorAxis,
    [property: JsonPropertyName("extent")]
    string Extent,
    [property: JsonPropertyName("hierarchy")]
    int[] Hierarchy,
    [property: JsonPropertyName("hierarchy_levels")]
    string HierarchyLevels,
    [property: JsonPropertyName("global_shape")]
    string[] GlobalShape);

internal sealed record KernelTuningMetadata(
    [property: JsonPropertyName("parameters")]
    Dictionary<string, TuningParameterMetadata> Parameters);

internal sealed record TuningParameterMetadata(
    [property: JsonPropertyName("source")]
    string Source,
    [property: JsonPropertyName("candidates")]
    long[] Candidates);
