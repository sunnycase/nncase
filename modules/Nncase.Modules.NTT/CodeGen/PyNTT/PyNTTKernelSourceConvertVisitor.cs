// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reactive;
using System.Text;
using System.Text.Json.Serialization;
using Nncase.IR;
using Nncase.IR.Distributed;
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
    private readonly PyNTTTargetOptions _targetOptions;

    public PyNTTKernelSourceConvertVisitor(CompileOptions compileOptions)
    {
        _targetOptions = compileOptions.TargetOptions as PyNTTTargetOptions ?? new PyNTTTargetOptions();
    }

    public PyNTTGeneratedKernelSource GetKernelSource()
    {
        return new(_generatedKernels.ToArray(), _sourceBuilder.ToString());
    }

    protected override Unit VisitFunction(Function expr) => default;

    protected override Unit VisitFusion(Fusion expr) => default;

    protected override Unit VisitPrimFunction(PrimFunction expr)
    {
        var outputs = GetOutputInfos(expr);
        if (outputs.Length == 0)
        {
            throw new NotSupportedException($"PyNTT PrimFunction {expr.Name} does not have tensor outputs.");
        }

        var parameterNames = expr.Parameters.ToArray()
            .ToDictionary(parameter => parameter, parameter => parameter.Name);
        var lowered = new PyNTTPrimFunctionSourceVisitor(expr, parameterNames, outputs, _targetOptions).Build();
        _generatedKernels.Add(lowered.Metadata);
        if (!string.IsNullOrWhiteSpace(lowered.BodySource))
        {
            AppendKernelSource(BuildGeneratedTopKernelPython(lowered.Metadata, lowered.BodySource, lowered.HelperSource));
        }

        return default;
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
        private readonly Dictionary<string, int> _helperCounters = new();
        private readonly Dictionary<string, object> _attrs = new();
        private readonly Dictionary<TIR.Buffer, int> _returnOutputBufferIndices = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<int> _storedOutputIndices = new();
        private readonly Dictionary<int, int> _outputAliases = new();
        private int _nextStoreIndex;

        public PyNTTPrimFunctionSourceVisitor(
            PrimFunction function,
            IReadOnlyDictionary<IVar, string> parameterNames,
            OutputInfo[] outputs,
            PyNTTTargetOptions targetOptions)
        {
            _function = function;
            _parameterNames = parameterNames;
            _outputs = outputs;
            _outputDistributedTypes = new DistributedType?[outputs.Length];
            _targetOptions = targetOptions;
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
            _attrs["tir"] = true;
            var localShape = GetBufferShape(dest);
            var globalShape = GetTensorShape(args[1], $"TensorLoad source input{inputIndex}");
            var helperName = GetHelperName("tensor_load", inputIndex);
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
                    GetShardAxis(dest),
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
            WriteTensorStore(src, outputIndex, globalShape, GetShardAxis(src), $"{src.Name} -> TensorStore");
        }

        private void WriteTensorStore(TIR.Buffer src, int outputIndex, long[] globalShape, int? shardAxis, string comment)
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
                    shardAxis,
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
            ValidateSameShape(context, GetBufferShape(input), shape);
            _attrs["op"] = GetOpName(unaryOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_unary");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    GetBufferShape(input),
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
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
            _attrs["op"] = "expand";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("expand_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
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
            ValidateGatherShape("PyNTT Gather", inputShape, indexShape, outputShape, axis);
            _attrs["op"] = "gather";
            _attrs["tir"] = true;
            _attrs["axis"] = axis;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            var helperName = GetNextHelperName("gather_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Gather.py.cshtml",
                new PyNTTGatherTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(index),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(index.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(index.ElemType),
                    GetTritonDType(output.ElemType),
                    inputShape,
                    indexShape,
                    outputShape,
                    GetBufferStrides(input),
                    GetBufferStrides(index),
                    GetBufferStrides(output),
                    axis,
                    $"{input.Name}, {index.Name} -> {output.Name}"));
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
            ValidateSameShape("PyNTT Swish", GetBufferShape(input), shape);
            var beta = swish.Beta.ToString("R", CultureInfo.InvariantCulture);
            _attrs["op"] = "swish";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["beta"] = swish.Beta;
            var helperName = GetNextHelperName("swish_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    GetBufferShape(input),
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    shape,
                    $"value0 / (1.0 + tl.exp(-({beta}) * value0))",
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
            EnsureEmpty("PyNTT VectorizedBinary lhs vectorized axes", binary.LhsVectorizedAxes);
            EnsureEmpty("PyNTT VectorizedBinary rhs vectorized axes", binary.RhsVectorizedAxes);
            _attrs["op"] = GetOpName(binary.BinaryOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_binary");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseBinary.py.cshtml",
                new PyNTTElementwiseBinaryTemplateModel(
                    helperName,
                    GetBufferPointer(lhs),
                    GetBufferPointer(rhs),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(lhs.ElemType),
                    GetPyNTTDTypeName(rhs.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(lhs.ElemType),
                    GetTritonDType(rhs.ElemType),
                    GetTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    shape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    shape,
                    GetBinaryExpression(binary.BinaryOp),
                    (string)_attrs["op"],
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
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

            EnsureEmpty("PyNTT Cast vectorized axes", cast.VectorizeAxes);
            SetComputeOp("cast");
            var shape = GetBufferShape(output);
            ValidateSameShape("PyNTT Cast", GetBufferShape(input), shape);
            _attrs["op"] = "cast";
            _attrs["tir"] = true;
            _attrs["from_dtype"] = GetPyNTTDTypeName(input.ElemType);
            _attrs["to_dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["cast_mode"] = GetCastModeName(cast.CastMode);
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_cast");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseCast.py.cshtml",
                new PyNTTElementwiseCastTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    GetBufferShape(input),
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
                    shape,
                    GetCastExpression(cast.CastMode, GetTritonDType(output.ElemType)),
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
            ValidateBroadcastable("PyNTT Where cond", condShape, shape);
            ValidateBroadcastable("PyNTT Where true value", trueShape, shape);
            ValidateBroadcastable("PyNTT Where false value", falseShape, shape);
            _attrs["op"] = "where";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_where");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseWhere.py.cshtml",
                new PyNTTElementwiseWhereTemplateModel(
                    helperName,
                    GetBufferPointer(cond),
                    GetBufferPointer(trueValue),
                    GetBufferPointer(falseValue),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(cond.ElemType),
                    GetPyNTTDTypeName(trueValue.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(cond.ElemType),
                    GetTritonDType(trueValue.ElemType),
                    GetTritonDType(output.ElemType),
                    condShape,
                    trueShape,
                    falseShape,
                    shape,
                    GetBufferStrides(cond),
                    GetBufferStrides(trueValue),
                    GetBufferStrides(falseValue),
                    GetBufferStrides(output),
                    shape,
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
            ValidateSameShape("PyNTT Clamp", GetBufferShape(input), shape);
            _attrs["op"] = "clamp";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            _attrs["min"] = clamp.Min;
            _attrs["max"] = clamp.Max;
            var helperName = GetNextHelperName("elementwise_clamp");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseUnary.py.cshtml",
                new PyNTTElementwiseUnaryTemplateModel(
                    helperName,
                    GetBufferPointer(input),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(input.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(input.ElemType),
                    GetTritonDType(output.ElemType),
                    GetBufferShape(input),
                    shape,
                    GetBufferStrides(input),
                    GetBufferStrides(output),
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
            _attrs["op"] = GetCompareOpName(compare.CompareOp);
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = shape;
            var helperName = GetNextHelperName("elementwise_compare");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/ElementwiseBinary.py.cshtml",
                new PyNTTElementwiseBinaryTemplateModel(
                    helperName,
                    GetBufferPointer(lhs),
                    GetBufferPointer(rhs),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(lhs.ElemType),
                    GetPyNTTDTypeName(rhs.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(lhs.ElemType),
                    GetTritonDType(rhs.ElemType),
                    GetTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    shape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
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
            long concatAxisExtent = 0;
            for (var inputIndex = 0; inputIndex < inputShapes.Length; inputIndex++)
            {
                var inputShape = inputShapes[inputIndex];
                if (inputShape.Length != outputShape.Length)
                {
                    throw new NotSupportedException($"PyNTT Concat input{inputIndex} rank {inputShape.Length} does not match output rank {outputShape.Length}.");
                }

                for (var dim = 0; dim < outputShape.Length; dim++)
                {
                    if (dim != axis && inputShape[dim] != outputShape[dim])
                    {
                        throw new NotSupportedException($"PyNTT Concat input{inputIndex} dim {dim}={inputShape[dim]} does not match output dim {outputShape[dim]}.");
                    }
                }

                concatAxisExtent += inputShape[axis];
            }

            if (concatAxisExtent != outputShape[axis])
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

            if (outputShape[1] != weightsShape[0] || biasShape[0] != outputShape[1])
            {
                throw new NotSupportedException($"PyNTT Conv2D expects output channels, weights, and bias to match; got output={outputShape[1]}, weights={weightsShape[0]}, bias={biasShape[0]}.");
            }

            if (inputShape[1] != weightsShape[1] * conv2D.Groups)
            {
                throw new NotSupportedException($"PyNTT Conv2D expects input channels {inputShape[1]} to equal weights input channels {weightsShape[1]} * groups {conv2D.Groups}.");
            }

            if (outputShape[1] % conv2D.Groups != 0)
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
                if (outputShape[outputAxis] != inputShape[inputAxis])
                {
                    throw new NotSupportedException($"PyNTT Transpose output axis {outputAxis} shape {outputShape[outputAxis]} does not match input axis {inputAxis} shape {inputShape[inputAxis]}.");
                }
            }

            _attrs["op"] = "transpose";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["perm"] = perm;
            var helperName = GetNextHelperName("transpose_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Transpose.py.cshtml",
                new PyNTTTransposeTemplateModel(
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

            EnsureEmpty("PyNTT Matmul lhs vectorized axes", matmul.LhsVectorizedAxes);
            EnsureEmpty("PyNTT Matmul rhs vectorized axes", matmul.RhsVectorizedAxes);
            SetComputeOp("matmul");
            var lhsShape = GetBufferShape(lhs);
            var rhsShape = GetBufferShape(rhs);
            var outputShape = GetBufferShape(output);
            ValidateMinimumRank("PyNTT Matmul lhs", lhsShape, 2);
            ValidateMinimumRank("PyNTT Matmul rhs", rhsShape, 2);
            ValidateMinimumRank("PyNTT Matmul output", outputShape, 2);
            var lhsK = matmul.TransposeA ? lhsShape[^2] : lhsShape[^1];
            var rhsK = matmul.TransposeB ? rhsShape[^1] : rhsShape[^2];
            var lhsM = matmul.TransposeA ? lhsShape[^1] : lhsShape[^2];
            var rhsN = matmul.TransposeB ? rhsShape[^2] : rhsShape[^1];
            if (lhsK != rhsK || outputShape[^2] != lhsM || outputShape[^1] != rhsN)
            {
                throw new NotSupportedException($"PyNTT Matmul expects compatible matrix shapes, got lhs=[{string.Join(",", lhsShape)}], rhs=[{string.Join(",", rhsShape)}], output=[{string.Join(",", outputShape)}].");
            }

            ValidateBroadcastable("PyNTT Matmul lhs batch", lhsShape[..^2], outputShape[..^2]);
            ValidateBroadcastable("PyNTT Matmul rhs batch", rhsShape[..^2], outputShape[..^2]);
            var scale = GetScalarFloat(args[4], "matmul scale", 1.0f);
            _attrs["op"] = "matmul";
            _attrs["tir"] = true;
            _attrs["dtype"] = GetPyNTTDTypeName(output.ElemType);
            _attrs["shape"] = outputShape;
            _attrs["transpose_a"] = matmul.TransposeA;
            _attrs["transpose_b"] = matmul.TransposeB;
            _attrs["scale"] = scale;
            var helperName = GetNextHelperName("matmul_compute");
            WriteHelperTemplate(
                "~/CodeGen/PyNTT/Templates/Triton/Kernels/Matmul.py.cshtml",
                new PyNTTMatmulTemplateModel(
                    helperName,
                    GetBufferPointer(lhs),
                    GetBufferPointer(rhs),
                    GetBufferPointer(output),
                    GetPyNTTDTypeName(lhs.ElemType),
                    GetPyNTTDTypeName(rhs.ElemType),
                    GetPyNTTDTypeName(output.ElemType),
                    GetTritonDType(lhs.ElemType),
                    GetTritonDType(rhs.ElemType),
                    GetTritonDType(output.ElemType),
                    lhsShape,
                    rhsShape,
                    outputShape,
                    GetBufferStrides(lhs),
                    GetBufferStrides(rhs),
                    GetBufferStrides(output),
                    matmul.TransposeA,
                    matmul.TransposeB,
                    scale.ToString("R", CultureInfo.InvariantCulture),
                    $"{lhs.Name}, {rhs.Name} -> {output.Name}"));
            WriteLine(BuildHelperCall(helperName));
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

        private BufferRef ResolveBufferRef(TIR.Buffer buffer)
        {
            var offsetBytes = GetBufferOffsetBytes(buffer);
            return buffer.MemSpan.Buffer.Location switch
            {
                MemoryLocation.Data => new("data", offsetBytes, checked((long)_function.SchedResult.DataUsage), "shard_index"),
                MemoryLocation.Rdata => new("rdata", offsetBytes, 0, null),
                MemoryLocation.BlockLocalRdata => new("block_local_rdata", offsetBytes, GetPoolSizeBytes(_function.SchedResult.BlockLocalRdatas), "shard_index"),
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
            var args = leadingArguments
                .Concat(WorkspaceParameterNames)
                .Concat(new[] { "block_size" });
            return $"{helperName}({string.Join(", ", args)})";
        }

        private long[] GetBufferShape(TIR.Buffer buffer)
        {
            if (buffer.DistributedType is { } distributedType)
            {
                return DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
            }

            return buffer.Dimensions.ToArray()
                .Select((dimension, index) => GetFixedDimension(dimension, $"{buffer.Name} dimension {index}"))
                .ToArray();
        }

        private long[] GetBufferStrides(TIR.Buffer buffer)
        {
            return buffer.Strides.ToArray()
                .Select((stride, index) => GetFixedDimension(stride, $"{buffer.Name} stride {index}"))
                .ToArray();
        }

        private void SetComputeOp(string opKind)
        {
            _opKinds.Add(opKind);
        }

        private void WriteLine(string line)
        {
            _body.AppendLine(line);
            _body.AppendLine("tl.debug_barrier()");
        }

        private void WriteHelperTemplate(string templatePath, object model)
        {
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
            ["numel"] = numel,
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
                output.Shape);
        }

        return new ShardMetadata(
            "local_shard",
            GetBlockPlacementAxis(targetOptions),
            0,
            "grid[0]",
            GetBlockHierarchy(targetOptions),
            output.Shape);
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
        var parameters = string.Join(", ", inputs.Concat(outputs).Concat(workspaceParameters).Concat(new[] { "numel: tl.constexpr", "block_size: tl.constexpr" }));
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

    private static long[] GetTensorShape(BaseExpr expr, string name)
    {
        var tensorType = GetTensorType(expr.CheckedType, name);
        return GetStaticShape(tensorType, name);
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
            UnaryOp.Acos => "libdevice.acos(value0)",
            UnaryOp.Acosh => "tl.log(value0 + tl.sqrt(value0 * value0 - 1.0))",
            UnaryOp.Asin => "libdevice.asin(value0)",
            UnaryOp.Asinh => "tl.log(value0 + tl.sqrt(value0 * value0 + 1.0))",
            UnaryOp.Ceil => "tl.ceil(value0)",
            UnaryOp.Cos => "tl.cos(value0)",
            UnaryOp.Cosh => "(tl.exp(value0) + tl.exp(-value0)) * 0.5",
            UnaryOp.Erf => "tl.erf(value0)",
            UnaryOp.Exp => "tl.exp(value0)",
            UnaryOp.Floor => "tl.floor(value0)",
            UnaryOp.Log => "tl.log(value0)",
            UnaryOp.Neg => "-value0",
            UnaryOp.Round => "libdevice.round(value0)",
            UnaryOp.Rsqrt => "tl.rsqrt(value0)",
            UnaryOp.Sin => "tl.sin(value0)",
            UnaryOp.Sinh => "(tl.exp(value0) - tl.exp(-value0)) * 0.5",
            UnaryOp.Sqrt => "tl.sqrt(value0)",
            UnaryOp.Square => "value0 * value0",
            UnaryOp.Tanh => "(tl.exp(value0 + value0) - 1.0) / (tl.exp(value0 + value0) + 1.0)",
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

    private static string GetReduceFinalizeExpression(ReduceOp op, long reduceElementCount)
    {
        return op switch
        {
            ReduceOp.Sum or ReduceOp.Max or ReduceOp.Min => "acc",
            ReduceOp.Mean => $"acc / {reduceElementCount.ToString(CultureInfo.InvariantCulture)}.0",
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

    private static void ValidateSameShape(string context, IReadOnlyList<long> actual, IReadOnlyList<long> expected)
    {
        if (!actual.SequenceEqual(expected))
        {
            throw new NotSupportedException($"{context} requires matching shapes, got [{string.Join(",", actual)}] and [{string.Join(",", expected)}].");
        }
    }

    private static void ValidateBroadcastable(string context, IReadOnlyList<long> actual, IReadOnlyList<long> expected)
    {
        if (actual.Count > expected.Count)
        {
            throw new NotSupportedException($"{context} requires broadcastable shape, got [{string.Join(",", actual)}] for output [{string.Join(",", expected)}].");
        }

        var axisOffset = expected.Count - actual.Count;
        for (var i = 0; i < actual.Count; i++)
        {
            var actualDim = actual[i];
            var expectedDim = expected[axisOffset + i];
            if (actualDim != 1 && actualDim != expectedDim)
            {
                throw new NotSupportedException($"{context} requires broadcastable shape, got [{string.Join(",", actual)}] for output [{string.Join(",", expected)}].");
            }
        }
    }

    private static void ValidateReduceShape(string context, IReadOnlyList<long> inputShape, IReadOnlyList<long> outputShape, IReadOnlyList<int> axes, bool keepDims)
    {
        var axisSet = axes.ToHashSet();
        if (keepDims)
        {
            if (outputShape.Count != inputShape.Count)
            {
                throw new NotSupportedException($"{context} with keep_dims expects output rank {inputShape.Count}, got [{string.Join(",", outputShape)}].");
            }

            for (var axis = 0; axis < inputShape.Count; axis++)
            {
                var expected = axisSet.Contains(axis) ? 1 : inputShape[axis];
                if (outputShape[axis] != expected)
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
        if (!outputShape.SequenceEqual(expectedShape))
        {
            throw new NotSupportedException($"{context} output shape mismatch, expected [{string.Join(",", expectedShape)}], got [{string.Join(",", outputShape)}].");
        }
    }

    private static void ValidatePadShape(string context, IReadOnlyList<long> inputShape, IReadOnlyList<long> outputShape, long[][] pads)
    {
        if (pads.Length != inputShape.Count || outputShape.Count != inputShape.Count)
        {
            throw new NotSupportedException($"{context} expects input, output, and pads to have the same rank.");
        }

        for (var axis = 0; axis < inputShape.Count; axis++)
        {
            var expected = inputShape[axis] + pads[axis][0] + pads[axis][1];
            if (outputShape[axis] != expected)
            {
                throw new NotSupportedException($"{context} output shape mismatch at axis {axis}, expected {expected}, got {outputShape[axis]}.");
            }
        }
    }

    private static void ValidateSameRank(string context, IReadOnlyList<long> lhsShape, IReadOnlyList<long> rhsShape)
    {
        if (lhsShape.Count != rhsShape.Count)
        {
            throw new NotSupportedException($"{context} expects matching ranks, got {lhsShape.Count} and {rhsShape.Count}.");
        }
    }

    private static void ValidateScatterNDShape(string context, IReadOnlyList<long> inputShape, IReadOnlyList<long> indicesShape, IReadOnlyList<long> updatesShape, IReadOnlyList<long> outputShape)
    {
        if (!inputShape.SequenceEqual(outputShape))
        {
            throw new NotSupportedException($"{context} expects input and output to have the same shape.");
        }

        if (indicesShape.Count < 1)
        {
            throw new NotSupportedException($"{context} expects indices rank >= 1.");
        }

        var indexDepth = indicesShape[^1];
        if (indexDepth < 1 || indexDepth > inputShape.Count)
        {
            throw new NotSupportedException($"{context} index depth {indexDepth} is out of range for input rank {inputShape.Count}.");
        }

        var expectedUpdatesShape = indicesShape.Take(indicesShape.Count - 1)
            .Concat(inputShape.Skip((int)indexDepth))
            .ToArray();
        if (!updatesShape.SequenceEqual(expectedUpdatesShape))
        {
            throw new NotSupportedException($"{context} updates shape mismatch, expected [{string.Join(",", expectedUpdatesShape)}], got [{string.Join(",", updatesShape)}].");
        }
    }

    private static (long[] Starts, long[] Strides) NormalizeSliceParameters(IReadOnlyList<long> inputShape, IReadOnlyList<long> starts, IReadOnlyList<int> axes, IReadOnlyList<int> strides, string context)
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

            normalizedStarts[axis] = NormalizeSliceStart(starts[i], inputShape[axis], stride);
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

    private static void ValidateGatherShape(string context, IReadOnlyList<long> inputShape, IReadOnlyList<long> indexShape, IReadOnlyList<long> outputShape, int axis)
    {
        var expectedRank = inputShape.Count + indexShape.Count - 1;
        if (outputShape.Count != expectedRank)
        {
            throw new NotSupportedException($"{context} expects output rank {expectedRank}, got [{string.Join(",", outputShape)}].");
        }

        for (var i = 0; i < axis; i++)
        {
            if (outputShape[i] != inputShape[i])
            {
                throw new NotSupportedException($"{context} output pre-axis shape mismatch, got [{string.Join(",", outputShape)}] for input [{string.Join(",", inputShape)}].");
            }
        }

        for (var i = 0; i < indexShape.Count; i++)
        {
            if (outputShape[axis + i] != indexShape[i])
            {
                throw new NotSupportedException($"{context} output index shape mismatch, got [{string.Join(",", outputShape)}] for index [{string.Join(",", indexShape)}].");
            }
        }

        for (var i = axis + 1; i < inputShape.Count; i++)
        {
            var outputAxis = i + indexShape.Count - 1;
            if (outputShape[outputAxis] != inputShape[i])
            {
                throw new NotSupportedException($"{context} output post-axis shape mismatch, got [{string.Join(",", outputShape)}] for input [{string.Join(",", inputShape)}].");
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

    private static void ValidateRank(string context, IReadOnlyList<long> shape, int rank)
    {
        if (shape.Count != rank)
        {
            throw new NotSupportedException($"{context} requires rank {rank}, got [{string.Join(",", shape)}].");
        }
    }

    private static void ValidateMinimumRank(string context, IReadOnlyList<long> shape, int rank)
    {
        if (shape.Count < rank)
        {
            throw new NotSupportedException($"{context} requires rank >= {rank}, got [{string.Join(",", shape)}].");
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
                return new OutputInfo($"output{index}", GetStaticShape(tensorType, $"output{index}"), type as DistributedType);
            })
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

    private static long Product(IEnumerable<long> values)
    {
        return values.Aggregate(1L, (product, value) => product * value);
    }

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

    private sealed record OutputInfo(string Name, long[] Shape, DistributedType? DistributedType);
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
    long[] GlobalShape);

internal sealed record KernelTuningMetadata(
    [property: JsonPropertyName("parameters")]
    Dictionary<string, TuningParameterMetadata> Parameters);

internal sealed record TuningParameterMetadata(
    [property: JsonPropertyName("source")]
    string Source,
    [property: JsonPropertyName("candidates")]
    long[] Candidates);
