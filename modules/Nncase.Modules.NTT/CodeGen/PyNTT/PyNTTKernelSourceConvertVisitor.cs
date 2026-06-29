// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using System.Text;
using System.Text.Json.Serialization;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTKernelSourceConvertVisitor : ExprFunctor<Unit, Unit>
{
    private static readonly long[] ElementwiseBlockSizeSearchSpace = { 128, 256, 512, 1024 };

    private readonly List<GeneratedKernelMetadata> _generatedKernels = new();
    private readonly StringBuilder _sourceBuilder = new();
    private readonly SharedHelperRegistry _sharedHelperRegistry = new();
    private readonly PyNTTTargetOptions _targetOptions;

    public PyNTTKernelSourceConvertVisitor(CompileOptions compileOptions)
    {
        _targetOptions = PyNTTTargetOptionsUtility.Normalize(compileOptions);
    }

    public PyNTTGeneratedKernelSource GetKernelSource()
    {
        return new(_generatedKernels.ToArray(), _sourceBuilder.ToString());
    }

    protected override Unit VisitFunction(Function expr) => default;

    protected override Unit VisitFusion(Fusion expr) => default;

    protected override Unit VisitPrimFunction(PrimFunction expr)
    {
        if (ContainsPrimFunctionCall(expr.Body))
        {
            return default;
        }

        var outputs = GetOutputInfos(expr);
        if (outputs.Length == 0)
        {
            outputs = GetTensorStoreOutputInfos(expr);
        }

        if (outputs.Length == 0)
        {
            throw new NotSupportedException($"PyNTT PrimFunction {expr.Name} does not have tensor outputs.");
        }

        var parameterNames = expr.Parameters.ToArray()
            .ToDictionary(parameter => parameter, parameter => parameter.Name);
        var lowered = new PyNTTPrimFunctionSourceVisitor(expr, parameterNames, outputs, _targetOptions, _sharedHelperRegistry).Build();
        _generatedKernels.Add(lowered.Metadata);
        if (!string.IsNullOrWhiteSpace(lowered.BodySource))
        {
            AppendKernelSource(BuildGeneratedTopKernelPython(lowered.Metadata, lowered.BodySource, lowered.HelperSource));
        }

        return default;
    }

    private static bool ContainsPrimFunctionCall(BaseExpr expr)
    {
        if (expr is Call { Target: PrimFunction })
        {
            return true;
        }

        foreach (var operand in expr.Operands)
        {
            if (ContainsPrimFunctionCall(operand))
            {
                return true;
            }
        }

        return false;
    }

#pragma warning disable SA1201
    private sealed class PyNTTPrimFunctionSourceVisitor : ExprFunctor<Unit, Unit>
    {
        private static readonly string[] WorkspaceParameterNames = { "data", "rdata", "thread_local_rdata", "warp_local_rdata", "block_local_rdata" };

        private readonly PrimFunction _function;
        private readonly IReadOnlyDictionary<IVar, string> _parameterNames;
        private readonly OutputInfo[] _outputs;
        private readonly DistributedType?[] _outputDistributedTypes;
        private readonly PyNTTTargetOptions _targetOptions;
        private readonly StringBuilder _body = new();
        private readonly StringBuilder _helperSource = new();
        private readonly List<string> _inputNames = new();
        private readonly List<string> _opKinds = new();
        private readonly List<HelperKernelCallMetadata> _helperCalls = new();
        private readonly List<PyNTTKVCacheFieldInputMetadata> _kvCacheFieldInputs = new();
        private readonly SortedSet<string> _runtimeScalarNames = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _helperCounters = new();
        private readonly Dictionary<string, object> _attrs = new();
        private readonly Dictionary<TIR.Buffer, int> _returnOutputBufferIndices = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.Buffer, int> _bufferInputIndices = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<int> _storedOutputIndices = new();
        private readonly Dictionary<int, int> _outputAliases = new();
        private readonly SharedHelperRegistry _sharedHelperRegistry;
        private readonly PyNTTDimExpressionEmitter _dimEmitter;
        private int _nextStoreIndex;
        private int _bodyIndent;

        public PyNTTPrimFunctionSourceVisitor(
            PrimFunction function,
            IReadOnlyDictionary<IVar, string> parameterNames,
            OutputInfo[] outputs,
            PyNTTTargetOptions targetOptions,
            SharedHelperRegistry sharedHelperRegistry)
        {
            _function = function;
            _parameterNames = parameterNames;
            _outputs = outputs;
            _outputDistributedTypes = new DistributedType?[outputs.Length];
            _targetOptions = targetOptions;
            _sharedHelperRegistry = sharedHelperRegistry;
            _dimEmitter = new(RegisterRuntimeScalar);
        }

        public GeneratedPrimFunctionKernel Build()
        {
            AnalyzeReturnValues(_function.Body);
            Visit(_function.Body);
            var materializedOutputIndices = _storedOutputIndices.Concat(_outputAliases.Keys).ToHashSet();
            if (materializedOutputIndices.Count != _outputs.Length)
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

            if (_outputAliases.Count > 0)
            {
                _attrs["output_aliases"] = new Dictionary<int, int>(_outputAliases);
            }

            if (_runtimeScalarNames.Count > 0)
            {
                _attrs["runtime_shape_args"] = _runtimeScalarNames.ToArray();
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

            var metadata = new GeneratedKernelMetadata(
                SanitizePythonIdentifier($"{_function.Name}_{opKind}_0"),
                opKind,
                _inputNames.ToArray(),
                outputs.Select(output => output.Name).ToArray(),
                _attrs,
                BuildLaunchMetadata(
                    outputs[0],
                    _targetOptions,
                    new()
                    {
                        ["data_pool_bytes"] = checked((long)_function.SchedResult.DataUsage),
                        ["data_pool_elements"] = checked((long)_function.SchedResult.DataUsage),
                        ["data_dtype"] = "uint8",
                        ["rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.Rdatas),
                        ["thread_local_rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.ThreadLocalRdatas),
                        ["warp_local_rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.WarpLocalRdatas),
                        ["block_local_rdata_pool_bytes"] = GetPoolSizeBytes(_function.SchedResult.BlockLocalRdatas),
                        ["thread_local_rdata_stride_bytes"] = PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.ThreadLocalRdatas, _targetOptions, "t"),
                        ["warp_local_rdata_stride_bytes"] = PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.WarpLocalRdatas, _targetOptions, "w"),
                        ["block_local_rdata_stride_bytes"] = PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.BlockLocalRdatas, _targetOptions, "b"),
                    }));
            return new(metadata, _helperSource.ToString().TrimEnd(), _body.ToString().TrimEnd());
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
                case Nncase.TIR.NTT.VectorizedSoftmax softmax:
                    VisitSoftmax(softmax.Axis, softmax.VectorizedAxes, args, "softmax");
                    break;
                case Nncase.TIR.NTT.Softmax softmax:
                    VisitSoftmax(softmax.Axis, default, args, "softmax");
                    break;
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/TensorLoad.py.cshtml",
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
                    $"TensorLoad -> {dest.Name}"));
            WriteLine(BuildHelperCall(helperName, $"input{inputIndex}"));
        }

        private void VisitTensorStore(IReadOnlyList<BaseExpr> args)
        {
            if (args.Count != 2 || args[0] is not TIR.Buffer src)
            {
                throw new NotSupportedException("PyNTT TensorStore codegen expects (source_buffer, dest_tensor).");
            }

            var localShape = GetBufferShape(src);
            var globalShape = GetTensorShape(args[1], "TensorStore destination");
            var outputIndex = GetOutputIndex(args[1]);
            WriteTensorStore(src, outputIndex, globalShape, $"{src.Name} -> TensorStore");
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
            var helperName = GetHelperName("tensor_store", outputIndex);
            var localShape = GetBufferShape(src);
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/TensorStore.py.cshtml",
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
                    comment));
            WriteLine(BuildHelperCall(helperName, outputName));
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
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, $"{context} input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, $"{context} output");
            if (inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"{context} expects matching input/output vector lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            _attrs["op"] = GetOpName(unaryOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_unary");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Gather.py.cshtml",
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
                    indexShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(index),
                    GetBufferStrides(output),
                    axis,
                    valueVectorLaneCount,
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

            if (gatherReduceScatter.InType.AxisPolicies.Any(policy => policy is SBPPartial))
            {
                throw new NotSupportedException("PyNTT GatherReduceScatter does not support partial/reduction reshard yet.");
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
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT GatherReduceScatter input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT GatherReduceScatter output");
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

            _attrs["op"] = "reshard";
            _attrs["tir"] = true;
            _attrs["requires_split_launch"] = true;
            _attrs["dtype"] = outputScalarDType;
            _attrs["shape"] = globalShape;
            var helperName = GetNextHelperName("reshard");
            var inputRef = ResolveBufferRef(input);
            var outputRef = ResolveBufferRef(output);
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Reshard.py.cshtml",
                new PyNTTReshardTemplateModel(
                    helperName,
                    GetBufferScalarPointer(input, "source_shard_index"),
                    GetBufferScalarPointer(output),
                    inputRef.BaseName,
                    inputRef.OffsetBytes,
                    inputRef.PoolBytes,
                    outputRef.BaseName,
                    outputRef.OffsetBytes,
                    outputRef.PoolBytes,
                    GetScalarElementSizeBytes(output.ElemType),
                    outputScalarDType,
                    GetScalarTritonDType(output.ElemType),
                    globalShape,
                    inputShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    inputVectorLaneCount,
                    gatherReduceScatter.OutType.Placement.Hierarchy.ToArray(),
                    GetSplitAxes(gatherReduceScatter.InType),
                    GetSplitAxes(gatherReduceScatter.OutType),
                    $"{input.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Pad.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ScatterND.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Slice.py.cshtml",
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
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT Swish input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Swish output");
            if (inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT Swish expects matching input/output vector lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            var beta = swish.Beta.ToString("R", CultureInfo.InvariantCulture);
            _attrs["op"] = "swish";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["beta"] = swish.Beta;
            var helperName = GetNextHelperName("swish_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
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
            if (args.Count < 3 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT VectorizedBinary codegen expects TIR buffer operands.");
            }

            SetComputeOp("binary");
            var shape = GetBufferShape(output);
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            ValidateBroadcastable("PyNTT VectorizedBinary lhs", lhsShape, shape);
            ValidateBroadcastable("PyNTT VectorizedBinary rhs", rhsShape, shape);
            var lhsVectorLaneCount = GetSingleVectorLaneCount(lhs.ElemType, "PyNTT VectorizedBinary lhs");
            var rhsVectorLaneCount = GetSingleVectorLaneCount(rhs.ElemType, "PyNTT VectorizedBinary rhs");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT VectorizedBinary output");
            if (lhsVectorLaneCount != 1 && lhsVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT VectorizedBinary expects lhs vector lanes to be scalar or match output lanes, got lhs={lhsVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            if (rhsVectorLaneCount != 1 && rhsVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT VectorizedBinary expects rhs vector lanes to be scalar or match output lanes, got rhs={rhsVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            _attrs["op"] = GetOpName(binary.BinaryOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_binary");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseBinary.py.cshtml",
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
            var axes = NormalizeLayoutAxes(pack.Axes, inputShape.Length, "PyNTT Pack");
            var lanes = GetLayoutLanes(pack.Lanes, axes.Length, "PyNTT Pack");
            ValidatePackShape("PyNTT Pack", inputShape, outputShape, axes, lanes);
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/VectorLayout.py.cshtml",
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
            var axes = NormalizeLayoutAxes(unpack.Axes, outputShape.Length, "PyNTT Unpack");
            var lanes = GetLayoutLanes(unpack.Lanes, axes.Length, "PyNTT Unpack");
            ValidateUnpackShape("PyNTT Unpack", inputShape, outputShape, axes, lanes);
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/VectorLayout.py.cshtml",
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
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer input ||
                args[1] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Cast codegen expects TIR buffer operands.");
            }

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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseCast.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseWhere.py.cshtml",
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
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT Clamp input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Clamp output");
            if (inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT Clamp expects matching input/output vector lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            _attrs["op"] = "clamp";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["min"] = clamp.Min;
            _attrs["max"] = clamp.Max;
            var helperName = GetNextHelperName("elementwise_clamp");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
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
            var lhsVectorLaneCount = GetSingleVectorLaneCount(lhs.ElemType, "PyNTT Compare lhs");
            var rhsVectorLaneCount = GetSingleVectorLaneCount(rhs.ElemType, "PyNTT Compare rhs");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Compare output");
            if (lhsVectorLaneCount != 1 && lhsVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT Compare expects lhs vector lanes to be scalar or match output lanes, got lhs={lhsVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            if (rhsVectorLaneCount != 1 && rhsVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT Compare expects rhs vector lanes to be scalar or match output lanes, got rhs={rhsVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            _attrs["op"] = GetCompareOpName(compare.CompareOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_compare");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseBinary.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Concat.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Conv2D.py.cshtml",
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
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT Transpose input");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Transpose output");
            if (inputVectorLaneCount != outputVectorLaneCount)
            {
                throw new NotSupportedException($"PyNTT Transpose expects matching input/output vector lanes, got input={inputVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            var helperName = GetNextHelperName("transpose_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Transpose.py.cshtml",
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
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer lhs ||
                args[1] is not TIR.Buffer rhs ||
                args[2] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT Matmul codegen expects lhs, rhs, and output TIR buffers.");
            }

            if (matmul.FusedReduce)
            {
                throw new NotSupportedException("PyNTT Matmul codegen does not support fused reduce yet.");
            }

            if (GetScalarBool(args[3], "matmul loadC"))
            {
                throw new NotSupportedException("PyNTT Matmul codegen does not support loadC yet.");
            }

            if (args.Count > 5 && args[5] is not None)
            {
                throw new NotSupportedException("PyNTT Matmul codegen does not support extra workload operands yet.");
            }

            SetComputeOp("matmul");
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            var outputShape = GetBufferShape(output);
            ValidateMinimumRank("PyNTT Matmul lhs", lhsShape, 2);
            ValidateMinimumRank("PyNTT Matmul rhs", rhsShape, 2);
            ValidateMinimumRank("PyNTT Matmul output", outputShape, 2);
            var dimInfo = Nncase.IR.NTT.VectorizedMatMul.GetDimInfo(matmul.TransposeA, matmul.TransposeB, lhsShape.Length, rhsShape.Length);
            var lhsVectorLanes = GetVectorLanes(lhs.ElemType);
            var rhsVectorLanes = GetVectorLanes(rhs.ElemType);
            var outputVectorLanes = GetVectorLanes(output.ElemType);
            var rhsNVectorLaneCount = 1;
            var outputNVectorLaneCount = 1;
            if (!matmul.LhsVectorizedAxes.IsDefaultOrEmpty)
            {
                throw new NotSupportedException($"PyNTT Matmul currently supports only RHS N-axis vectorization, got lhs axes [{string.Join(",", matmul.LhsVectorizedAxes)}].");
            }

            if (!matmul.RhsVectorizedAxes.IsDefaultOrEmpty)
            {
                if (matmul.RhsVectorizedAxes.Count != 1 || matmul.RhsVectorizedAxes[0] != dimInfo.Rn || rhsVectorLanes.Length != 1)
                {
                    throw new NotSupportedException($"PyNTT Matmul currently supports only one RHS N-axis vector lane, got rhs axes [{string.Join(",", matmul.RhsVectorizedAxes)}] and lanes [{string.Join(",", rhsVectorLanes)}].");
                }

                rhsNVectorLaneCount = rhsVectorLanes[0];
            }

            if (lhsVectorLanes.Length != 0)
            {
                throw new NotSupportedException($"PyNTT Matmul currently supports scalar lhs operands only, got lhs lanes [{string.Join(",", lhsVectorLanes)}].");
            }

            if (outputVectorLanes.Length != 0)
            {
                if (outputVectorLanes.Length != 1 || outputVectorLanes[0] != rhsNVectorLaneCount || rhsNVectorLaneCount == 1)
                {
                    throw new NotSupportedException($"PyNTT Matmul currently supports only RHS N-axis vectorization producing the same output N lane, got output lanes [{string.Join(",", outputVectorLanes)}] and rhs N lane {rhsNVectorLaneCount}.");
                }

                outputNVectorLaneCount = outputVectorLanes[0];
            }

            var lhsK = matmul.TransposeA ? lhsShape[^2] : lhsShape[^1];
            var rhsK = matmul.TransposeB ? rhsShape[^1] : rhsShape[^2];
            var lhsM = matmul.TransposeA ? lhsShape[^1] : lhsShape[^2];
            var rhsN = matmul.TransposeB ? rhsShape[^2] : rhsShape[^1];
            var rhsFullN = MultiplyDim(rhsN, rhsNVectorLaneCount);
            var outputNCompatible = outputNVectorLaneCount == 1
                ? rhsNVectorLaneCount == 1 ? SameDim(outputShape[^1], rhsN) : CanFitPaddedDim(outputShape[^1], rhsFullN)
                : SameDim(outputShape[^1], rhsN);
            if (!SameDim(lhsK, rhsK) || !SameDim(outputShape[^2], lhsM) || !outputNCompatible)
            {
                throw new NotSupportedException($"PyNTT Matmul expects compatible matrix shapes, got lhs=[{ShapeText(lhsShape)}], rhs=[{ShapeText(rhsShape)}], output=[{ShapeText(outputShape)}].");
            }

            ValidateBroadcastable("PyNTT Matmul lhs batch", lhsShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT Matmul rhs batch", rhsShape[..^2], outputShape[..^2]);
            var reduceAxes = GetMatmulReduceAxes(lhs, rhs, output, dimInfo);
            var scale = GetScalarFloat(args[4], "matmul scale", 1.0f);
            _attrs["op"] = "matmul";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTScalarDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["transpose_a"] = matmul.TransposeA;
            _attrs["transpose_b"] = matmul.TransposeB;
            _attrs["scale"] = scale;
            var helperName = GetNextHelperName("matmul_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Matmul.py.cshtml",
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
                    matmul.TransposeA,
                    matmul.TransposeB,
                    rhsNVectorLaneCount,
                    outputNVectorLaneCount,
                    scale.ToString("R", CultureInfo.InvariantCulture),
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
            WriteMatmulShardReduce(output, reduceAxes, $"{output.Name} K-axis shard reduce");
        }

        private void WriteMatmulShardReduce(TIR.Buffer output, int[] reduceAxes, string comment)
        {
            if (reduceAxes.Length == 0)
            {
                return;
            }

            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT Matmul shard reduce output");
            var outputScalarDType = GetPyNTTScalarDTypeName(output.ElemType);
            var hierarchy = GetHierarchy(output);
            foreach (var axis in reduceAxes)
            {
                if (axis < 0 || axis >= hierarchy.Length)
                {
                    throw new NotSupportedException($"PyNTT Matmul shard reduce axis {axis} is outside hierarchy rank {hierarchy.Length}.");
                }
            }

            WriteShardReduceHelper(output, reduceAxes, outputVectorLaneCount, outputScalarDType, hierarchy, broadcast: false, $"{comment}: reduce");
            WriteShardReduceHelper(output, reduceAxes, outputVectorLaneCount, outputScalarDType, hierarchy, broadcast: true, $"{comment}: broadcast");
        }

        private void WriteShardReduceHelper(
            TIR.Buffer buffer,
            int[] reduceAxes,
            int vectorLaneCount,
            string scalarDType,
            int[] hierarchy,
            bool broadcast,
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
                    "~/CodeGen/PyNTT/Templates/Triton/Kernels/ShardReduce.py.cshtml",
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

            WriteLine(BuildHelperCall(
                helperName,
                BuildRawPythonArgument(bufferRef.PoolBytes.ToString(CultureInfo.InvariantCulture)),
                BuildRawPythonArgument(bufferRef.OffsetBytes.ToString(CultureInfo.InvariantCulture)),
                BuildRawPythonArgument(bufferRef.OffsetBytes.ToString(CultureInfo.InvariantCulture))));
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Reduce.py.cshtml",
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
            var inputVectorLaneCount = GetSingleVectorLaneCount(input.ElemType, "PyNTT RoPE input");
            var cosVectorLaneCount = GetSingleVectorLaneCount(cos.ElemType, "PyNTT RoPE cos");
            var sinVectorLaneCount = GetSingleVectorLaneCount(sin.ElemType, "PyNTT RoPE sin");
            var outputVectorLaneCount = GetSingleVectorLaneCount(output.ElemType, "PyNTT RoPE output");
            var valueLaneCounts = new[] { inputVectorLaneCount, cosVectorLaneCount, sinVectorLaneCount, outputVectorLaneCount }
                .Where(lane => lane > 1)
                .Distinct()
                .ToArray();
            if (valueLaneCounts.Length > 1)
            {
                throw new NotSupportedException($"PyNTT RoPE expects matching input/cos/sin/output vector lanes, got input={inputVectorLaneCount}, cos={cosVectorLaneCount}, sin={sinVectorLaneCount}, output={outputVectorLaneCount}.");
            }

            var rotaryAxis = outputVectorLaneCount > 1
                ? outputShape.Length - 2
                : outputShape.Length - 1;
            if (rotaryAxis < 0)
            {
                throw new NotSupportedException($"PyNTT RoPE requires a non-scalar output, got [{ShapeText(outputShape)}].");
            }

            ValidateRoPEShape("PyNTT RoPE", inputShape, cosShape, sinShape, outputShape, rotaryAxis, outputVectorLaneCount);
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/RoPE.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/LayerNorm.py.cshtml",
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/GetPositionIds.py.cshtml",
                new PyNTTGetPositionIdsTemplateModel(
                    helperName,
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(output.ElemType),
                    outputShape,
                    globalShape,
                    GetBufferStrides(output),
                    GetShardAxis(output),
                    $"kv-cache -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName, $"input{cacheMetaInputIndex}"));
        }

        private void VisitUpdatePagedAttentionKVCache(Nncase.TIR.NTT.UpdatePagedAttentionKVCache update, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 2 ||
                args[0] is not TIR.Buffer slots)
            {
                throw new NotSupportedException("PyNTT UpdatePagedAttentionKVCache codegen expects slots buffer and kv-cache object.");
            }

            var cache = GetPagedAttentionCacheTemplateModel(args[1], "PyNTT UpdatePagedAttentionKVCache");
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
            _attrs["layer_id"] = update.LayerId;
            var helperName = GetNextHelperName("update_paged_attention_kv_cache");
            var slotsRef = ResolveBufferRef(slots);
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/UpdatePagedAttentionKVCache.py.cshtml",
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
                    slotsRef.PoolBytes,
                    GetScalarElementSizeBytes(slots.ElemType),
                    seqAxis,
                    headAxis,
                    dimAxis,
                    update.LayerId,
                    update.CacheKind == AttentionCacheKind.Key ? 0 : 1,
                    cache,
                    $"{slots.Name} -> kv-cache"));
            WriteLine(BuildHelperCall(helperName, $"input{slotMappingInputIndex}", $"input{storageInputIndex}", $"input{storageBlocksInputIndex}", $"input{metaInputIndex}"));
        }

        private void VisitPagedAttention(Nncase.TIR.NTT.PagedAttention pagedAttention, IReadOnlyList<BaseExpr> args)
        {
            if (args.Count < 5 ||
                args[0] is not TIR.Buffer query ||
                args[2] is not TIR.Buffer ||
                args[3] is not TIR.Buffer scale ||
                args[4] is not TIR.Buffer output)
            {
                throw new NotSupportedException("PyNTT PagedAttention codegen expects query, kv-cache, extra, scale, and output buffers.");
            }

            var cache = GetPagedAttentionCacheTemplateModel(args[1], "PyNTT PagedAttention");
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

            SetComputeOp("paged_attention");
            _attrs["op"] = "paged_attention";
            _attrs["tir"] = true;
            _attrs["layer_id"] = pagedAttention.LayerId;
            _attrs["hidden_size"] = pagedAttention.HiddenSize;
            var helperName = GetNextHelperName("paged_attention");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/PagedAttention.py.cshtml",
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
                    GetBufferSplitAxes(output, output.Dimensions.Length),
                    GetHierarchy(output),
                    seqAxis,
                    headAxis,
                    dimAxis,
                    GetGlobalNumQueryHeads(pagedAttention, cache),
                    pagedAttention.LayerId,
                    cache,
                    $"{query.Name}, kv-cache -> {output.Name}"));
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
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Softmax.py.cshtml",
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

        private int GetOutputIndex(BaseExpr expr)
        {
            expr = UnwrapInputBoxing(expr);
            if (expr is TIR.Buffer buffer && _returnOutputBufferIndices.TryGetValue(buffer, out var returnOutputIndex))
            {
                return returnOutputIndex;
            }

            try
            {
                var outputName = GetTensorName(expr, _parameterNames);
                for (var i = 0; i < _outputs.Length; i++)
                {
                    if (_outputs[i].Name == outputName)
                    {
                        return i;
                    }
                }
            }
            catch (NotSupportedException)
            {
                // Some TIR forms do not preserve the destination tensor as a named
                // parameter. In that case the TensorStore order matches the flattened
                // function return order produced by TIR lowering.
            }

            while (_nextStoreIndex < _outputs.Length && _storedOutputIndices.Contains(_nextStoreIndex))
            {
                _nextStoreIndex++;
            }

            if (_nextStoreIndex >= _outputs.Length)
            {
                throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} has more TensorStore calls than flattened outputs.");
            }

            return _nextStoreIndex++;
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
                return _inputNames[GetBufferInputIndex(kvCache, context)];
            }

            return GetTensorName(expr, _parameterNames);
        }

        private PyNTTPagedAttentionCacheTemplateModel GetPagedAttentionCacheTemplateModel(BaseExpr expr, string context)
        {
            var config = GetPagedAttentionConfig(expr, context);
            ValidatePagedAttentionConfig(config, context);

            var laneCount = config.Lanes.Count == 0 ? 1 : checked((int)config.Lanes[0]);
            var headDimBlocks = checked(config.HeadDim / laneCount);
            var topologyShape = GetPagedAttentionTopologyShape(config);
            var numBlocksSplitAxes = GetPagedAttentionNumBlocksSplitAxes(config);
            return new(
                GetPyNTTScalarDTypeName(config.KVType),
                GetScalarTritonDType(config.KVType),
                config.NumLayers,
                config.NumKVHeads,
                config.HeadDim,
                config.BlockSize,
                laneCount,
                headDimBlocks,
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

        private PyNTTKVCacheStorageMetadata GetKVCacheStorageMetadata(PyNTTPagedAttentionCacheTemplateModel cache)
        {
            // [topology, num_blocks, num_layers, kv, num_kv_heads, head_dim_blocks, block_size, lane].
            var tailShape = new[] { cache.NumLayers, 2, cache.NumKVHeads, cache.HeadDimBlocks, cache.BlockSize, cache.LaneCount };
            return new(cache.DType, cache.TopologyShape, tailShape, cache.BlockSize);
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
            var expectedLayout = new[]
            {
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.HeadDim,
                PagedKVCacheDimKind.BlockSize,
            };
            if (!config.CacheLayout.ToArray().SequenceEqual(expectedLayout))
            {
                throw new NotSupportedException($"{context} currently supports cache layout [NumBlocks, NumLayers, KV, NumKVHeads, HeadDim, BlockSize], got [{string.Join(", ", config.CacheLayout)}].");
            }

            if (config.VectorizedAxes.Count != 1 || config.VectorizedAxes[0] != PagedKVCacheDimKind.HeadDim || config.Lanes.Count != 1)
            {
                throw new NotSupportedException($"{context} currently supports only HeadDim vectorization with one lane value.");
            }

            if (config.HeadDim % config.Lanes[0] != 0)
            {
                throw new NotSupportedException($"{context} requires head_dim divisible by lane, got head_dim={config.HeadDim}, lane={config.Lanes[0]}.");
            }

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

        private void AnalyzeReturnValues(BaseExpr expr)
        {
            switch (expr)
            {
                case Return ret:
                    AnalyzeReturn(ret);
                    break;
                case Sequential sequential:
                    foreach (var field in sequential.Fields)
                    {
                        AnalyzeReturnValues(field);
                    }

                    break;
            }
        }

        private void AnalyzeReturn(Return expr)
        {
            var values = expr.Values.ToArray();
            for (var i = 0; i < values.Length && i < _outputs.Length; i++)
            {
                var value = UnwrapInputBoxing(values[i]);
                if (value is TIR.Buffer buffer)
                {
                    _returnOutputBufferIndices[buffer] = i;
                    continue;
                }

                try
                {
                    _outputAliases[i] = GetInputIndex(value);
                }
                catch (NotSupportedException)
                {
                    throw new NotSupportedException($"PyNTT PrimFunction {_function.Name} cannot materialize return output {_outputs[i].Name} from {value.GetType().Name}.");
                }
            }
        }

        private PyNTTBufferPointerTemplateModel GetBufferPointer(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            return new(BuildPointerExpression(bufferRef, GetTritonDType(buffer.ElemType)));
        }

        private PyNTTBufferPointerTemplateModel GetBufferScalarPointer(TIR.Buffer buffer)
        {
            var bufferRef = ResolveBufferRef(buffer);
            return new(BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType)));
        }

        private PyNTTBufferPointerTemplateModel GetBufferScalarPointer(TIR.Buffer buffer, string indexExpression)
        {
            var bufferRef = ResolveBufferRef(buffer) with { IndexExpression = indexExpression };
            return new(BuildPointerExpression(bufferRef, GetScalarTritonDType(buffer.ElemType)));
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
            var offsetBytes = GetBufferOffsetBytes(buffer);
            return buffer.MemSpan.Buffer.Location switch
            {
                MemoryLocation.Data => new("data", offsetBytes, checked((long)_function.SchedResult.DataUsage), "shard_index"),
                MemoryLocation.Rdata => new("rdata", offsetBytes, 0, null),
                MemoryLocation.ThreadLocalRdata => new("thread_local_rdata", offsetBytes, PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.ThreadLocalRdatas, _targetOptions, "t"), "shard_index"),
                MemoryLocation.WarpLocalRdata => new("warp_local_rdata", offsetBytes, PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.WarpLocalRdatas, _targetOptions, "w"), "shard_index"),
                MemoryLocation.BlockLocalRdata => new("block_local_rdata", offsetBytes, PyNTTRDataUtility.GetLocalRDataTableStrideBytes(_function.SchedResult.BlockLocalRdatas, _targetOptions, "b"), "shard_index"),
                var location => throw new NotSupportedException($"PyNTT does not support buffer memory location {location} for Triton template operands yet."),
            };
        }

        private string BuildPointerExpression(BufferRef bufferRef, string tritonDType)
        {
            var expression = bufferRef.BaseName;
            if (!string.IsNullOrWhiteSpace(bufferRef.IndexExpression) && bufferRef.PoolBytes != 0)
            {
                expression += $" + {bufferRef.IndexExpression} * {bufferRef.PoolBytes.ToString(CultureInfo.InvariantCulture)}";
            }

            if (bufferRef.OffsetBytes != 0)
            {
                expression += $" + {bufferRef.OffsetBytes.ToString(CultureInfo.InvariantCulture)}";
            }

            return $"({expression}).to(tl.pointer_type({tritonDType}))";
        }

        private string BuildHelperCall(string helperName, params string[] leadingArguments)
        {
            _helperCalls.Add(new(helperName, leadingArguments));
            var args = leadingArguments
                .Select(FormatHelperCallArgument)
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
            return buffer.Strides.ToArray()
                .Select(GetDimensionExpression)
                .ToArray();
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

        private void WriteLine(string line)
        {
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine(line);
            _body.Append(new string(' ', _bodyIndent * 4));
            _body.AppendLine("tl.debug_barrier()");
        }

        private void WriteHelperTemplate(string templatePath, object model)
        {
            var runtimeShapeArgs = _runtimeScalarNames.ToArray();
            var runtimeShapeArgsProperty = model.GetType().GetProperty("RuntimeShapeArgs");
            runtimeShapeArgsProperty?.SetValue(model, runtimeShapeArgs);

            if (_helperSource.Length > 0)
            {
                _helperSource.AppendLine();
                _helperSource.AppendLine();
            }

            _helperSource.Append(RazorTemplateEngine.RenderAsync(templatePath, model).Result.TrimEnd());
        }

        private string GetHelperName(string kind, int index)
        {
            return SanitizePythonIdentifier($"{_function.Name}_{kind}_{index}");
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

        private sealed record BufferRef(string BaseName, long OffsetBytes, long PoolBytes, string? IndexExpression);
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
            4,
            null);
    }

    private static ShardMetadata BuildShardingMetadata(OutputInfo output, PyNTTTargetOptions targetOptions)
    {
        if (output.DistributedType is { } distributedType)
        {
            return new ShardMetadata(
                "local_shard",
                string.IsNullOrWhiteSpace(distributedType.Placement.Name) ? "b" : distributedType.Placement.Name,
                GetTensorAxis(distributedType),
                "grid[0]",
                distributedType.Placement.Hierarchy.ToArray(),
                ToPythonExpressions(output.Shape));
        }

        return new ShardMetadata(
            "local_shard",
            GetBlockPlacementAxis(targetOptions),
            0,
            "grid[0]",
            GetBlockHierarchy(targetOptions),
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
        var hierarchyNames = string.IsNullOrWhiteSpace(targetOptions.HierarchyNames)
            ? "b"
            : targetOptions.HierarchyNames;
        return hierarchyNames.Contains("b", StringComparison.Ordinal) ? "b" : hierarchyNames[0].ToString();
    }

    private static int[] GetBlockHierarchy(PyNTTTargetOptions targetOptions)
    {
        var hierarchies = targetOptions.Hierarchies;
        return hierarchies.Length == 0 ? new[] { 1 } : hierarchies[0];
    }

    private static string BuildGeneratedTopKernelPython(GeneratedKernelMetadata kernel, string bodySource, string helperSource)
    {
        var inputs = kernel.Inputs.Select((_, index) => $"input{index}").ToArray();
        var outputs = kernel.Outputs.Select((_, index) => $"output{index}").ToArray();
        var workspaceParameters = new[] { "data", "rdata", "thread_local_rdata", "warp_local_rdata", "block_local_rdata" };
        var runtimeShapeArgs = GetRuntimeShapeArgs(kernel);
        var parameters = string.Join(", ", inputs.Concat(outputs).Concat(workspaceParameters).Concat(runtimeShapeArgs).Concat(new[] { "numel", "block_size: tl.constexpr" }));
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

        if (expr is IVar parameter && parameterNames.TryGetValue(parameter, out var name))
        {
            return name;
        }

        throw new NotSupportedException($"PyNTT M3 elementwise kernels only support direct function parameter operands, got {expr.GetType().Name}.");
    }

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
            var packedAxis = axes.IndexOf(axis);
            var expected = packedAxis >= 0 ? CeilDivDim(inputShape[axis], lanes[packedAxis]) : inputShape[axis];
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
            var unpackedAxis = axes.IndexOf(axis);
            var expected = unpackedAxis >= 0 ? MultiplyDim(inputShape[axis], lanes[unpackedAxis]) : inputShape[axis];
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
        for (var i = 0; i < normalizedAxes.Length; i++)
        {
            for (var j = i + 1; j < normalizedAxes.Length; j++)
            {
                if (normalizedAxes[i] == normalizedAxes[j])
                {
                    throw new NotSupportedException($"{context} contains duplicated axis {normalizedAxes[i]}.");
                }
            }
        }

        return normalizedAxes;
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

    private static void ValidateRoPEShape(string context, IReadOnlyList<PyNTTDimExpression> inputShape, IReadOnlyList<PyNTTDimExpression> cosShape, IReadOnlyList<PyNTTDimExpression> sinShape, IReadOnlyList<PyNTTDimExpression> outputShape, int rotaryAxis, int laneCount)
    {
        ValidateSameShape(context, inputShape, outputShape);
        ValidateBroadcastable($"{context} cos", cosShape, outputShape);
        ValidateBroadcastable($"{context} sin", sinShape, outputShape);
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

    private static void ValidateLayerNormShape(string context, IReadOnlyList<PyNTTDimExpression> parameterShape, IReadOnlyList<PyNTTDimExpression> outputShape, int axis)
    {
        var expectedShape = outputShape.Skip(axis).ToArray();
        if (parameterShape.Count != expectedShape.Length || parameterShape.Zip(expectedShape).Any(pair => !SameDim(pair.First, pair.Second)))
        {
            throw new NotSupportedException($"{context} requires shape [{ShapeText(expectedShape)}], got [{ShapeText(parameterShape)}].");
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
        return GetPyNTTDTypeName(dataType) switch
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

    private static bool IsObjectDataType(DataType dataType) => dataType is ReferenceType;

    private static OutputInfo[] GetOutputInfos(BaseFunction function)
    {
        if (function.CheckedType is not CallableType callableType)
        {
            return Array.Empty<OutputInfo>();
        }

        return FlattenReturnTypes(callableType.ReturnType)
            .Select((type, index) =>
            {
                var tensorType = GetTensorType(type, $"output{index}");
                return new OutputInfo(
                    $"output{index}",
                    GetRankedShape(tensorType, $"output{index}").Dimensions.ToArray()
                        .Select(dimension => new PyNTTDimExpressionEmitter().Emit(dimension))
                        .ToArray(),
                    type as DistributedType);
            })
            .ToArray();
    }

    private static OutputInfo[] GetTensorStoreOutputInfos(PrimFunction function)
    {
        return GetTensorStoreDestinations(function)
            .Select((expr, index) =>
            {
                var tensorType = GetTensorType(expr.CheckedType, $"output{index}");
                return new OutputInfo(
                    $"output{index}",
                    GetRankedShape(tensorType, $"output{index}").Dimensions.ToArray()
                        .Select(dimension => new PyNTTDimExpressionEmitter().Emit(dimension))
                        .ToArray(),
                    expr.CheckedType as DistributedType);
            })
            .ToArray();
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

    private static BaseExpr UnwrapInputBoxing(BaseExpr expr)
    {
        while (expr is Call call && call.Target is Boxing)
        {
            expr = call.Arguments[0];
        }

        return expr;
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

    private void AppendKernelSource(string source)
    {
        if (_sourceBuilder.Length > 0)
        {
            _sourceBuilder.AppendLine();
            _sourceBuilder.AppendLine();
        }

        _sourceBuilder.Append(source);
    }

    private sealed record OutputInfo(string Name, PyNTTDimExpression[] Shape, DistributedType? DistributedType);

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
    [property: JsonPropertyName("source")]
    string Source);

internal sealed record GeneratedPrimFunctionKernel(
    GeneratedKernelMetadata Metadata,
    string HelperSource,
    string BodySource);

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
