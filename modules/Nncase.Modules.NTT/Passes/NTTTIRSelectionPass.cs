// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes;

public sealed class NTTTIRSelectionPass : TIRSelectionPass
{
    private readonly CompileOptions _compileOptions;
    private readonly Dictionary<TIR.Buffer, Dimension> _shardedComponentBases =
        new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Placement, Dimension[]> _shardCoordinates = new();
    private int _bufferIndex;
    private int _shardedViewIndex;

    public NTTTIRSelectionPass(CompileOptions compileOptions, string moduleKind = CPUTarget.Kind)
        : base(moduleKind)
    {
        _compileOptions = compileOptions;
    }

    protected override TIR.Buffer CreateIntermediateBuffer(
        Expr sourceExpr,
        IRType type,
        MemoryLocation defaultLocation,
        string name)
    {
        if (_compileOptions.TargetOptions is PyNTTTargetOptions &&
            type is DistributedType distributedType &&
            (distributedType.Partial is not null ||
             IsCompactPartialReshardResult(sourceExpr)))
        {
            return T.CreateCompactPerOwnerBuffer(distributedType, out _, name);
        }

        return base.CreateIntermediateBuffer(sourceExpr, type, defaultLocation, name);
    }

    protected override BufferLayoutAnnotation? GetBufferParameterLayout(
        IRType type,
        BufferVarRole role,
        bool isEntry)
    {
        if (!isEntry &&
            _compileOptions.TargetOptions is PyNTTTargetOptions &&
            type is DistributedType distributedType &&
            (distributedType.Partial is not null ||
             (role == BufferVarRole.Output && DistributedUtility.IsFullyShardedAcrossPlacement(distributedType))))
        {
            (_, var strides) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(
                distributedType.TensorType,
                distributedType);
            return BufferLayoutAnnotation.ExactStrided(
                strides,
                DistributedBufferStorageKind.CompactPerOwner);
        }

        return base.GetBufferParameterLayout(type, role, isEntry);
    }

    protected override Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, ref Expr output, TIRSelectionContext context)
    {
        var op = call.Target;
        switch (op)
        {
            case IR.Math.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.CustomNTT.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.Math.Clamp clamp:
                return GenerateClamp(call, arguments, output);
            case IR.Distributed.Boxing boxing:
                return GenerateBoxing(call, boxing, arguments, ref output);
            case IR.Distributed.ShardedView shardedView:
                return GenerateShardedView(call, shardedView, arguments, ref output, context);
            case IR.Distributed.ForceBoxing forceBoxing:
                return T.Memcopy(output, (Expr)arguments[0]);
            case IR.Math.Binary binary:
                return TIR.F.NTT.VectorizedBinary((Expr)arguments[0], (Expr)arguments[1], output, None.Default, binary.BinaryOp, Array.Empty<int>(), Array.Empty<Dimension>(), Array.Empty<int>(), Array.Empty<Dimension>());
            case IR.Tensors.Bitcast:
                return GenerateBufferView("Bitcast", (Expr)arguments[0], ref output);
            case IR.Tensors.Pack pack:
                return TIR.F.NTT.Pack((Expr)arguments[0], output, pack.Lanes, pack.Axes);
            case IR.Tensors.VectorizeMask pack:
                return TIR.F.NTT.Pack((Expr)arguments[0], output, new[] { pack.Lanes }, new[] { pack.Axis });
            case IR.Tensors.Unpack unpack:
                return TIR.F.NTT.Unpack((Expr)arguments[0], output, unpack.Lanes, unpack.Axes);
            case IR.NTT.VectorizedBinary vectorizedBinary:
                return TIR.F.NTT.VectorizedBinary((Expr)arguments[0], (Expr)arguments[1], output, (Expr)arguments[2], vectorizedBinary.BinaryOp, vectorizedBinary.LhsVectorizedAxes, vectorizedBinary.LhsPadedNums, vectorizedBinary.RhsVectorizedAxes, vectorizedBinary.RhsPadedNums);
            case IR.NTT.VectorizedMatMul vectorizedMatMul when GetArgumentType(arguments[0]) is DistributedType dta && GetArgumentType(arguments[1]) is DistributedType dtb:
                var dinfo = vectorizedMatMul.GetDimInfo(dta.TensorType.Shape.Rank, dtb.TensorType.Shape.Rank);
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    DistributedUtility.IsSamePolicy(dta.AxisPolicies[dinfo.Lk], dtb.AxisPolicies[dinfo.Rn], false) &&
                    DistributedUtility.IsSamePolicy(dta.AxisPolicies[dinfo.Lm], dtb.AxisPolicies[dinfo.Rk], false))
                {
                    return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.VectorizedMatMul.Scale], vectorizedMatMul.LhsVectorizedAxes, vectorizedMatMul.RhsVectorizedAxes, vectorizedMatMul.TransposeA, vectorizedMatMul.TransposeB);
                }
                else
                {
                    return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.VectorizedMatMul.Scale], None.Default, vectorizedMatMul.LhsVectorizedAxes, vectorizedMatMul.RhsVectorizedAxes, vectorizedMatMul.TransposeA, vectorizedMatMul.TransposeB, vectorizedMatMul.FusedReduce);
                }

            case IR.Math.MatMul when GetArgumentType(arguments[0]) is DistributedType dta && GetArgumentType(arguments[1]) is DistributedType dtb:
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    DistributedUtility.IsSamePolicy(dta.AxisPolicies[^2], dtb.AxisPolicies[^2], false) &&
                    DistributedUtility.IsSamePolicy(dta.AxisPolicies[^1], dtb.AxisPolicies[^1], false))
                {
                    return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.Math.MatMul.Scale]);
                }
                else
                {
                    return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.Math.MatMul.Scale]);
                }

            case IR.CustomNTT.MatMul matmul:
                return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.CustomNTT.MatMul.Scale], (Expr)arguments[3], matmul.LhsVectorizedAxes, matmul.RhsVectorizedAxes, matmul.TransposeA, matmul.TransposeB, false, matmul.CSourcePath, matmul.FuncName);
            case IR.NTT.PackedMatMul matmul:
                return TIR.F.NTT.PackedMatMul(
                    (Expr)arguments[0],
                    (Expr)arguments[1],
                    output,
                    None.Default,
                    (Expr)call[IR.NTT.PackedMatMul.Scale],
                    matmul.FusedReduce,
                    matmul.RhsLayout,
                    (Expr)arguments[IR.NTT.PackedMatMul.Addend.Index]);
            case IR.NTT.PackedMatMulNormStats matmulNormStats:
                {
                    var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
                    if (outputBase is not IR.Tuple outputs || outputs.Count != 2)
                    {
                        throw new NotSupportedException(
                            "PackedMatMulNormStats TIR selection expects value and local-statistics outputs.");
                    }

                    return TIR.F.NTT.PackedMatMulNormStats(
                        (Expr)arguments[0],
                        (Expr)arguments[1],
                        (Expr)outputs[0],
                        (Expr)outputs[1],
                        None.Default,
                        (Expr)call[IR.NTT.PackedMatMulNormStats.Scale],
                        matmulNormStats.RhsLayout,
                        matmulNormStats.Axis,
                        matmulNormStats.UseMean,
                        (Expr)arguments[IR.NTT.PackedMatMulNormStats.Addend.Index]);
                }
            case IR.NTT.PackedMatMulSamplingPartial matmulSamplingPartial:
                return GeneratePackedMatMulSamplingPartial(
                    call,
                    matmulSamplingPartial,
                    arguments,
                    output);
            case IR.NTT.PackedMatMulGlu matmulGlu:
                return TIR.F.NTT.PackedMatMulGlu(
                    (Expr)arguments[0],
                    (Expr)arguments[1],
                    (Expr)arguments[2],
                    (Expr)arguments[3],
                    (Expr)arguments[4],
                    (Expr)arguments[5],
                    (Expr)arguments[6],
                    (Expr)arguments[7],
                    (Expr)arguments[8],
                    output,
                    matmulGlu.GluType,
                    matmulGlu.RhsLayout);
            case IR.NTT.PackedQKVParallelLinear qkv:
                {
                    var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
                    if (outputBase is not IR.Tuple outputs || outputs.Count != 3)
                    {
                        throw new NotSupportedException("PackedQKVParallelLinear TIR selection expects a tuple of 3 outputs.");
                    }

                    return TIR.F.NTT.PackedQKVParallelLinear(
                        (Expr)arguments[0],
                        (Expr)arguments[1],
                        (Expr)arguments[2],
                        (Expr)arguments[3],
                        (Expr)arguments[4],
                        (Expr)arguments[5],
                        (Expr)arguments[6],
                        (Expr)arguments[7],
                        (Expr)arguments[8],
                        (Expr)arguments[9],
                        (Expr)arguments[10],
                        (Expr)arguments[11],
                        (Expr)arguments[12],
                        (Expr)outputs[0],
                        (Expr)outputs[1],
                        (Expr)outputs[2],
                        qkv.NumHeads,
                        qkv.NumKvHeads,
                        qkv.RhsLayout);
                }

            case IR.NN.QKVParallelLinear qkv:
                {
                    var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
                    if (outputBase is not IR.Tuple outputs || outputs.Count != 3)
                    {
                        throw new NotSupportedException("QKVParallelLinear TIR selection expects a tuple of 3 outputs.");
                    }

                    return TIR.F.NTT.QKVParallelLinear(
                        (Expr)arguments[0],
                        (Expr)arguments[1],
                        (Expr)arguments[2],
                        (Expr)arguments[3],
                        (Expr)arguments[4],
                        (Expr)arguments[5],
                        (Expr)arguments[6],
                        (Expr)arguments[7],
                        (Expr)arguments[8],
                        (Expr)arguments[9],
                        (Expr)arguments[10],
                        (Expr)arguments[11],
                        (Expr)arguments[12],
                        (Expr)outputs[0],
                        (Expr)outputs[1],
                        (Expr)outputs[2],
                        qkv.NumHeads,
                        qkv.NumKvHeads);
                }

            case IR.NN.QKVRoPEWithCache fused:
                {
                    if (arguments[IR.NN.QKVRoPEWithCache.QKV.Index] is not IR.Tuple { Count: 3 } qkvInputs)
                    {
                        throw new NotSupportedException(
                            "QKVRoPEWithCache TIR selection expects a tuple of Q, K, and V buffers.");
                    }

                    var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
                    if (outputBase is not IR.Tuple { Count: 2 } outputs)
                    {
                        throw new NotSupportedException(
                            "QKVRoPEWithCache TIR selection expects Q and KV-cache outputs.");
                    }

                    var qOutput = (Expr)outputs[0];
                    Unsafe.As<Expr, BaseExpr>(ref output) =
                        new IR.Tuple(qOutput, (Expr)arguments[IR.NN.QKVRoPEWithCache.KVCaches.Index]);
                    return TIR.F.NTT.QKVRoPEWithCache(
                        (Expr)qkvInputs[0],
                        (Expr)qkvInputs[1],
                        (Expr)qkvInputs[2],
                        (Expr)arguments[IR.NN.QKVRoPEWithCache.QScale.Index],
                        (Expr)arguments[IR.NN.QKVRoPEWithCache.KScale.Index],
                        (Expr)arguments[IR.NN.QKVRoPEWithCache.QBias.Index],
                        (Expr)arguments[IR.NN.QKVRoPEWithCache.KBias.Index],
                        (Expr)arguments[IR.NN.QKVRoPEWithCache.Cos.Index],
                        (Expr)arguments[IR.NN.QKVRoPEWithCache.Sin.Index],
                        (Expr)arguments[IR.NN.QKVRoPEWithCache.KVCaches.Index],
                        (Dimension)arguments[IR.NN.QKVRoPEWithCache.LayerId.Index],
                        qOutput,
                        fused.QAxis,
                        fused.QEpsilon,
                        fused.QUseMean,
                        fused.KAxis,
                        fused.KEpsilon,
                        fused.KUseMean,
                        fused.QKVLayout,
                        fused.AttentionLayout);
                }

            case IR.NN.GatedDeltaNetProjection projection:
                {
                    var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
                    if (outputBase is not IR.Tuple { Count: 2 } outputs)
                    {
                        throw new NotSupportedException(
                            "GatedDeltaNetProjection TIR selection expects QKV and state outputs.");
                    }

                    var qkvOutput = (Expr)outputs[0];
                    Unsafe.As<Expr, BaseExpr>(ref output) = new IR.Tuple(
                        qkvOutput,
                        (Expr)arguments[IR.NN.GatedDeltaNetProjection.State.Index]);

                    return TIR.F.NTT.GatedDeltaNetProjection(
                        (Expr)arguments[IR.NN.GatedDeltaNetProjection.Input.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetProjection.State.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetProjection.QKVWeight.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetProjection.ConvWeight.Index],
                        qkvOutput,
                        (Dimension)arguments[IR.NN.GatedDeltaNetProjection.LayerId.Index],
                        projection.ConvKernelSize);
                }

            case IR.NN.GatedDeltaNetRecurrentCore recurrent:
                {
                    var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
                    if (outputBase is not IR.Tuple { Count: 2 } outputs)
                    {
                        throw new NotSupportedException(
                            "GatedDeltaNetRecurrentCore TIR selection expects gated activation and state outputs.");
                    }

                    var gatedOutput = (Expr)outputs[0];
                    Unsafe.As<Expr, BaseExpr>(ref output) = new IR.Tuple(
                        gatedOutput,
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.State.Index]);

                    return TIR.F.NTT.GatedDeltaNetRecurrentCore(
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.Input.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.State.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.QKV.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.ZWeight.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.BWeight.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.AWeight.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.ALog.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.DtBias.Index],
                        (Expr)arguments[IR.NN.GatedDeltaNetRecurrentCore.NormWeight.Index],
                        gatedOutput,
                        (Dimension)arguments[IR.NN.GatedDeltaNetRecurrentCore.LayerId.Index],
                        recurrent.NumKeyHeads,
                        recurrent.NumValueHeads,
                        recurrent.KeyHeadDim,
                        recurrent.ValueHeadDim,
                        recurrent.Epsilon);
                }

            case IR.NN.MatMulGlu matmulGlu:
                return TIR.F.NTT.MatMulGlu(
                    (Expr)arguments[0],
                    (Expr)arguments[1],
                    (Expr)arguments[2],
                    (Expr)arguments[3],
                    (Expr)arguments[4],
                    (Expr)arguments[5],
                    (Expr)arguments[6],
                    (Expr)arguments[7],
                    (Expr)arguments[8],
                    output,
                    matmulGlu.GluType);

            case IR.NN.Conv2D conv:
                {
                    var input = call[IR.NN.Conv2D.Input];
                    var weights = call[IR.NN.Conv2D.Weights];
                    var bias = call[IR.NN.Conv2D.Bias];
                    var strides = ((RankedShape)call[IR.NN.Conv2D.Stride]).ToValueArray();
                    var padding = Tensor.From(((Paddings)call[IR.NN.Conv2D.Padding]).ToValueArray()).ToArray();
                    var dilation = ((RankedShape)call[IR.NN.Conv2D.Dilation]).ToValueArray();
                    var groups = ((Dimension)call[IR.NN.Conv2D.Groups]).FixedValue;
                    var fusedClamp = ((TensorConst)call[IR.NN.Conv2D.FusedClamp]).Value.ToArray<float>();
                    var wShape = weights.CheckedShape.ToValueArray();
                    var outShape = call.CheckedShape.ToValueArray();
                    if (fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity || conv.PadMode != PadMode.Constant)
                    {
                        throw new NotSupportedException("not support this conv2d");
                    }

                    return TIR.F.NTT.Conv2D((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, strides, padding, dilation, groups, conv.PadMode, call.CheckedType is DistributedType dt_conv ? dt_conv : null!);
                }

            case IR.NTT.Im2col im2col:
                return TIR.F.NTT.Im2col((Expr)arguments[0], output, im2col.Kernel, im2col.Stride, im2col.Padding, im2col.VectorizedAxes, im2col.PadedNums);
            case IR.NTT.VectorizedRoPE rope:
                return TIR.F.NTT.RoPE((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.NN.RoPE when ModuleKind == PyNTTTarget.Kind:
                return TIR.F.NTT.RoPE((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Imaging.ResizeImage resize:
                if ((call[IR.Imaging.ResizeImage.Roi] is not None && ((RankedShape)call[IR.Imaging.ResizeImage.Roi].CheckedShape).Size != 0) || resize.IsTFResize)
                {
                    throw new NotSupportedException("not support tf resize");
                }

                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, Array.Empty<int>(), Array.Empty<Dimension>(), ((RankedShape)call[IR.Imaging.ResizeImage.NewSize]).ToValueArray().ToInts(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.NTT.ResizeImage resize:
                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, resize.VectorizedAxes.ToArray(), ((RankedShape)call[IR.NTT.ResizeImage.PadedNums]).Dimensions.ToArray(), resize.NewSize.ToArray(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.Tensors.Slice slice:
                return TIR.F.NTT.Slice((Expr)arguments[0], (RankedShape)arguments[1], (RankedShape)arguments[2], output, ((RankedShape)call[IR.Tensors.Slice.Axes]).ToValueArray().ToInts(), ((RankedShape)call[IR.Tensors.Slice.Strides]).ToValueArray().ToInts());
            case IR.Tensors.Concat concat:
                return TIR.F.NTT.Concat(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, concat.Axis);
            case IR.Tensors.Transpose trans:
                return TIR.F.NTT.Transpose((Expr)arguments[0], output, ((RankedShape)call[IR.Tensors.Transpose.Perm]).ToValueArray().ToInts());
            case IR.NN.Swish swish:
                return TIR.F.NTT.Swish((Expr)arguments[0], output, ((TensorConst)call[IR.NN.Swish.Beta]).Value.ToScalar<float>());
            case IR.Tensors.Gather gather:
                return TIR.F.NTT.Gather((Expr)arguments[0], (Expr)arguments[1], output, gather.Axis);
            case IR.NN.Pad pad:
                var paddings = (Paddings)call[IR.NN.Pad.Pads];
                var actualPadAxes = Enumerable.Range(0, paddings.Count).Where(i => !(paddings[i] is { IsFixed: true } pad && pad.Sum() == 0)).ToArray();
                return TIR.F.NTT.Pad((Expr)arguments[0], output, paddings, ((TensorConst)call[IR.NN.Pad.Value]).Value.ToArray<float>()[0], actualPadAxes);
            case IR.Math.Reduce reduce:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, Array.Empty<int>(), Array.Empty<Dimension>(), ((RankedShape)call[IR.Math.Reduce.Axes]).ToValueArray().Select(x => Util.PositiveIndex(x, arguments[0].CheckedTensorType)).OrderBy(a => a).ToArray().ToInts(), ((TensorConst)call[IR.Math.Reduce.KeepDims]).Value.ToArray<bool>()[0], reduce.ReduceOp);
            case IR.Math.ReduceArg reduceArg:
                return TIR.F.NTT.ReduceArg((Expr)arguments[0], output, (int)((DimConst)call[IR.Math.ReduceArg.Axis]).FixedValue, ((TensorConst)call[IR.Math.ReduceArg.KeepDims]).Value.ToArray<bool>()[0], ((TensorConst)call[IR.Math.ReduceArg.SelectLastIndex]).Value.ToArray<bool>()[0], reduceArg.ReduceArgOp, reduceArg.DestType);
            case IR.Tensors.Cast cast:
                return TIR.F.NTT.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode, Array.Empty<int>(), None.Default);
            case IR.NTT.VectorizedCast cast:
                return TIR.F.NTT.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode, cast.VectorizeAxes, (Expr)arguments[1]);
            case IR.Tensors.Where where:
                return TIR.F.NTT.Where((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Expand expand:
                return TIR.F.NTT.Expand((Expr)arguments[0], output);
            case IR.NN.Erf erf:
                return TIR.F.NTT.Erf((Expr)arguments[0], output);
            case IR.NN.Sigmoid:
                return GenerateUnary(UnaryOp.Sigmoid, arguments, output);
            case IR.NN.Softplus:
                return GenerateUnary(UnaryOp.Softplus, arguments, output);
            case IR.NTT.VectorizedReduce pr:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, pr.VectorizedAxes.ToArray(), ((RankedShape)call[IR.NTT.VectorizedReduce.PadedNums]).Dimensions.ToArray(), pr.Axes, pr.KeepDims, pr.ReduceOp);
            case IR.Math.Compare compare:
                return TIR.F.NTT.Compare(compare.CompareOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.GetItem getItem:
                return TIR.F.NTT.GetItem((Expr)arguments[0], arguments[1], output);
            case IR.Tensors.Reshape:
                return GenerateBufferView("Reshape", (Expr)arguments[0], ref output);
            case IR.Tensors.ScatterND scatterND:
                return TIR.F.NTT.ScatterND((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Stack stack:
                return TIR.F.NTT.Stack(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, ((TensorConst)call[IR.Tensors.Stack.Axis]).Value.ToScalar<int>());
            case IR.Tensors.Unsqueeze:
                return GenerateBufferView("Unsqueeze", (Expr)arguments[0], ref output);
            case IR.NN.UpdatePagedAttentionKVCache upkv:
                output = (Expr)arguments[1];
                return TIR.F.NTT.UpdatePagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], (Dimension)arguments[2], upkv.CacheKind, upkv.Layout);
            case IR.NN.GatherPagedAttentionKVCache gakv:
                return TIR.F.NTT.GatherPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], output);
            case IR.NN.CreatePagedAttentionKVCache ctkv:
                return TIR.F.NTT.CreatePagedAttentionKVCache(ctkv.Config, (Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], output);
            case IR.NN.IdentityPagedAttentionKVCache ctkv:
                output = (Expr)arguments[0];
                return TIR.F.NTT.IdentityPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8]);
            case IR.NN.PagedAttention pgat:
                return GeneratePagedAttention(call, pgat, arguments, ref output, context);
            case IR.NTT.PagedAttentionPartial partialAttention:
                return GeneratePagedAttentionPartial(partialAttention, arguments, output);
            case IR.NTT.PagedAttentionCombine combineAttention:
                return TIR.F.NTT.PagedAttentionMerge(
                    (Expr)arguments[IR.NTT.PagedAttentionCombine.MaxState.Index],
                    (Expr)arguments[IR.NTT.PagedAttentionCombine.SumState.Index],
                    (Expr)arguments[IR.NTT.PagedAttentionCombine.AccState.Index],
                    output,
                    combineAttention.Layout,
                    combineAttention.HiddenSize,
                    combineAttention.SplitHierarchyAxis,
                    combineAttention.SplitCount);
            case IR.NTT.SamplingPartial samplingPartial:
                return GenerateSamplingPartial(samplingPartial, arguments, output);
            case IR.NTT.SamplingCombine samplingCombine:
                return GenerateSamplingCombine(samplingCombine, arguments, ref output);
            case IR.Tensors.ConstantOfShape constantOfShape:
                return TIR.F.NTT.ConstantOfShape((Shape)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.Range range:
                return TIR.F.NTT.Range((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Buffers.Uninitialized uninitialized:
                return T.Nop();
            case IR.Shapes.AsTensor asTensor:
                output = call;
                return call;
            case IR.NN.GetPositionIds getPositionIds:
                return TIR.F.NTT.GetPositionIds((Expr)arguments[1], output, (DistributedType)call.CheckedType);
            case IR.NN.NormStats normStats:
                return TIR.F.NTT.NormStats((Expr)arguments[0], output, normStats.Axis, normStats.UseMean);
            case IR.NN.NormApply normApply:
                return TIR.F.NTT.NormApply((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], output, normApply.Axis, normApply.Epsilon, normApply.UseMean);
            case IR.NN.LayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, Array.Empty<int>(), Array.Empty<Dimension>());
            case IR.NTT.VectorizedLayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, ln.VectorizedAxes, ((RankedShape)call[IR.NTT.VectorizedLayerNorm.PadedNums]).Dimensions.ToArray());
            case IR.CustomNTT.LayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, ln.VectorizedAxes, Array.Empty<Dimension>(), (Expr)call[IR.CustomNTT.LayerNorm.PostScale], ln.CSourcePath, ln.FuncName);
            case IR.NN.Softmax softmax:
                return TIR.F.NTT.VectorizedSoftmax((Expr)arguments[0], output, (int)((DimConst)call[IR.NN.Softmax.Axis]).FixedValue, Array.Empty<int>());
            case IR.NTT.VectorizedSoftmax softmax:
                return TIR.F.NTT.VectorizedSoftmax((Expr)arguments[0], output, softmax.Axis, softmax.VectorizedAxes);
            case IR.NN.Qwen3MoE moe:
                return TIR.F.NTT.Qwen3MoE((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8], (Expr)arguments[9], (Expr)arguments[10], output, moe.LayerId, moe.HiddenSize, moe.IntermediateSize, moe.MoEIntermediateSize, moe.NumExpert, moe.NumTopK, moe.IsNormTopkProb);
            case IR.NN.SparseExpertsGateUp gateUp:
                return TIR.F.NTT.SparseExpertsGateUp((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], output, gateUp.HiddenSize, gateUp.MoEIntermediateSize, gateUp.NumExpert, gateUp.NumTopK, gateUp.ChunkSize);
            case IR.NN.SparseExpertsDown down:
                return TIR.F.NTT.SparseExpertsDown((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], output, down.HiddenSize, down.MoEIntermediateSize, down.NumExpert, down.NumTopK, down.ChunkSize);
            case IR.Tensors.TopK topk:
                return TIR.F.NTT.TopK((Expr)arguments[0], (Expr)arguments[1], output, ((TensorConst)call[IR.Tensors.TopK.Axis]).Value.ToScalar<long>(), ((TensorConst)call[IR.Tensors.TopK.Largest]).Value.ToScalar<long>(), ((TensorConst)call[IR.Tensors.TopK.Sorted]).Value.ToScalar<long>());
            case IR.CustomNTT.SparseExperts sparseExperts:
                var oldExtraBuffer = ((TIR.Buffer)arguments[12]).MemSpan.Buffer;
                var newExtraBuffer = oldExtraBuffer.With(location: MemoryLocation.BlockLocalData);
                var userBuffers = (from memSpan in oldExtraBuffer.Users.OfType<TIR.MemSpan>()
                                   from userBuffer in memSpan.Users.OfType<TIR.Buffer>()
                                   select userBuffer).ToArray();
                foreach (var userBuffer in userBuffers)
                {
                    var newBuffer = userBuffer.With(memSpan: userBuffer.MemSpan.With(buffer: newExtraBuffer));
                    ReplaceUtility.ReplaceAllUsesWith(userBuffer, newBuffer);
                    context.ReplaceSelectedValue(userBuffer, newBuffer);
                }

                var extraNew = ((TIR.Buffer)arguments[12]).With(memSpan: ((TIR.Buffer)arguments[12]).MemSpan.With(newExtraBuffer));
                return TIR.F.NTT.SparseExperts((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8], (Expr)arguments[9], (Expr)arguments[10], (Expr)arguments[11], extraNew, output, sparseExperts.QVectorizedAxes, sparseExperts.GateVectorizedAxes, sparseExperts.DownVectorizedAxes, sparseExperts.UpVectorizedAxes, sparseExperts.QSBPs, sparseExperts.GateSBPs, sparseExperts.DownSBPs, sparseExperts.UpSBPs, sparseExperts.HiddenSize, sparseExperts.MoEIntermediateSize, sparseExperts.NumExpert, sparseExperts.NumTopK, sparseExperts.ChunkSize, sparseExperts.Cost, sparseExperts.CSourcePath, sparseExperts.FuncName);
            default:
                throw new NotSupportedException($"Not supported: {op}");
        }
    }

    protected override IReadOnlyList<BaseExpr> PrepareCallerAllocatedPrimFunctionArguments(
        TIR.PrimFunction primFunction,
        IReadOnlyList<BaseExpr> arguments,
        TIRSelectionContext context)
    {
        var parameters = primFunction.Parameters.ToArray();
        var prepared = arguments.ToArray();
        for (var index = 0; index < parameters.Length; index++)
        {
            if (parameters[index] is BufferVar
                {
                    Role: BufferVarRole.Output,
                    LayoutAnnotation.DistributedStorageKind: DistributedBufferStorageKind.CompactPerOwner,
                } compactOutputParameter)
            {
                if (index >= prepared.Length ||
                    prepared[index] is not TIR.Buffer
                    {
                        DistributedStorageKind: DistributedBufferStorageKind.CompactPerOwner,
                        MemSpan.Buffer.Location: MemoryLocation.ChipLocalData,
                    })
                {
                    throw new InvalidOperationException(
                        $"Compact-per-owner output {primFunction.Name}.{compactOutputParameter.Name} " +
                        "requires caller-allocated ChipLocalData backing.");
                }

                continue;
            }

            if (parameters[index] is not BufferVar
                {
                    Role: BufferVarRole.Output,
                    LayoutAnnotation.DistributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal,
                } outputParameter)
            {
                continue;
            }

            if (index >= prepared.Length)
            {
                throw new InvalidOperationException(
                    $"Caller-allocated PrimFunction {primFunction.Name} is missing canonical output argument " +
                    $"{outputParameter.Name} at index {index}.");
            }

            if (prepared[index] is not TIR.Buffer actualBuffer ||
                actualBuffer.DistributedType is null)
            {
                throw new InvalidOperationException(
                    $"Canonical output {primFunction.Name}.{outputParameter.Name} requires a distributed " +
                    $"TIR.Buffer argument, got {prepared[index].GetType().Name}.");
            }

            if (actualBuffer.DistributedStorageKind == DistributedBufferStorageKind.CanonicalGlobal &&
                actualBuffer.MemSpan.Buffer.Location != MemoryLocation.Data)
            {
                continue;
            }

            (prepared[index], _) = EnsureChipLocalShardedBacking(actualBuffer, context);
        }

        return prepared;
    }

    private Expr GeneratePagedAttention(
        Call sourceCall,
        IR.NN.PagedAttention pagedAttention,
        IReadOnlyList<BaseExpr> arguments,
        ref Expr output,
        TIRSelectionContext context)
    {
        var direct = TIR.F.NTT.PagedAttention(
            (Expr)arguments[0],
            (Expr)arguments[1],
            (Expr)arguments[2],
            (Expr)arguments[3],
            (Dimension)arguments[4],
            output,
            pagedAttention.Layout,
            pagedAttention.HiddenSize);
        if (_compileOptions.TargetOptions is not IPagedAttentionExecutionPlanProvider provider ||
            sourceCall[IR.NN.PagedAttention.KVCaches].CheckedType is not TensorType kvCachesType ||
            !PagedAttentionExecutionPlanQuery.TryCreate(
                sourceCall[IR.NN.PagedAttention.Q].CheckedType,
                sourceCall[IR.NN.PagedAttention.Extra].CheckedType,
                kvCachesType,
                pagedAttention.Layout,
                pagedAttention.HiddenSize,
                out var query))
        {
            return direct;
        }

        var plan = provider.GetPagedAttentionExecutionPlan(query);
        plan.Validate(query);
        if (plan.Kind != PagedAttentionExecutionKind.SplitKV)
        {
            return direct;
        }

        if (arguments[0] is not TIR.Buffer queryBuffer ||
            queryBuffer.DistributedType is not { } queryDistributedType)
        {
            throw new InvalidOperationException(
                "Split-KV PagedAttention requires a distributed query buffer at TIR selection.");
        }

        var seqAxis = pagedAttention.Layout.IndexOf(IR.NN.AttentionDimKind.Seq);
        var headAxis = pagedAttention.Layout.IndexOf(IR.NN.AttentionDimKind.Head);
        var dimAxis = pagedAttention.Layout.IndexOf(IR.NN.AttentionDimKind.Dim);
        if (seqAxis < 0 || headAxis < 0 || dimAxis < 0)
        {
            throw new InvalidOperationException(
                "Split-KV PagedAttention layout must contain Seq, Head, and Dim.");
        }

        var partialMaxType = CreatePagedAttentionStateType(
            queryDistributedType,
            dimAxis,
            plan,
            accumulator: false,
            ReduceOp.Max);
        var partialSumType = CreatePagedAttentionStateType(
            queryDistributedType,
            dimAxis,
            plan,
            accumulator: false,
            ReduceOp.Sum);
        var partialAccType = CreatePagedAttentionStateType(
            queryDistributedType,
            dimAxis,
            plan,
            accumulator: true,
            ReduceOp.Sum);
        var partialMax = CreateMetadataBuffer(partialMaxType, MemoryLocation.Data, "paged_attention_max");
        var partialSum = CreateMetadataBuffer(partialSumType, MemoryLocation.Data, "paged_attention_sum");
        var partialAcc = CreateMetadataBuffer(partialAccType, MemoryLocation.Data, "paged_attention_acc");

        var mergeOutput = output;
        var requiresPublicationBarrier = false;
        if (output is TIR.Buffer outputBuffer &&
            outputBuffer.DistributedType is not null &&
            outputBuffer.MemSpan.Buffer.Location == MemoryLocation.Data &&
            outputBuffer.MemSpan.Buffer.Start is None)
        {
            var (canonicalOutput, _) = EnsureChipLocalShardedBacking(outputBuffer, context);
            output = canonicalOutput;
            mergeOutput = canonicalOutput;
            requiresPublicationBarrier = true;
        }

        var partial = TIR.F.NTT.PagedAttentionPartial(
            queryBuffer,
            (Expr)arguments[1],
            (Expr)arguments[2],
            (Expr)arguments[3],
            (Dimension)arguments[4],
            partialMax,
            partialSum,
            partialAcc,
            mergeOutput,
            pagedAttention.Layout,
            pagedAttention.HiddenSize,
            plan.SplitHierarchyAxis,
            plan.SplitCount,
            plan.DirectContextThreshold);
        var merge = TIR.F.NTT.PagedAttentionMerge(
            partialMax,
            partialSum,
            partialAcc,
            mergeOutput,
            pagedAttention.Layout,
            pagedAttention.HiddenSize,
            plan.SplitHierarchyAxis,
            plan.SplitCount);
        var splitFields = requiresPublicationBarrier
            ? new Expr[]
            {
                partial,
                merge,
                TIR.F.NTT.Barrier(
                    TIR.NTT.BarrierScope.Chip,
                    new IRArray<int>(new[] { plan.SplitHierarchyAxis })),
            }
            : new Expr[] { partial, merge };
        if (plan.DirectContextThreshold > 0)
        {
            var useSplitKV = TIR.F.NTT.PagedAttentionUseSplitKV(
                (Expr)arguments[1],
                plan.DirectContextThreshold);
            return new Sequential(
                [
                    partial,
                    new IfThenElse(
                        useSplitKV,
                        new Sequential(
                            splitFields[1..],
                            traceScopeName: "paged_attention_split_kv_merge")),
                ],
                traceScopeName: "paged_attention_adaptive");
        }

        return new Sequential(
            splitFields,
            traceScopeName: "paged_attention_split_kv");
    }

    private Expr GeneratePagedAttentionPartial(
        IR.NTT.PagedAttentionPartial partial,
        IReadOnlyList<BaseExpr> arguments,
        Expr output)
    {
        var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
        if (outputBase is not IR.Tuple { Count: 3 } states)
        {
            throw new InvalidOperationException(
                "PagedAttentionPartial TIR selection expects max, sum, and accumulator output buffers.");
        }

        return TIR.F.NTT.PagedAttentionPartial(
            (Expr)arguments[IR.NTT.PagedAttentionPartial.Q.Index],
            (Expr)arguments[IR.NTT.PagedAttentionPartial.KVCaches.Index],
            (Expr)arguments[IR.NTT.PagedAttentionPartial.Extra.Index],
            (Expr)arguments[IR.NTT.PagedAttentionPartial.Scale.Index],
            (Dimension)arguments[IR.NTT.PagedAttentionPartial.LayerId.Index],
            (Expr)states[0],
            (Expr)states[1],
            (Expr)states[2],
            None.Default,
            partial.Layout,
            partial.HiddenSize,
            partial.SplitHierarchyAxis,
            partial.SplitCount,
            directContextThreshold: 0);
    }

    private DistributedType CreatePagedAttentionStateType(
        DistributedType queryType,
        int dimAxis,
        PagedAttentionExecutionPlan plan,
        bool accumulator,
        ReduceOp reduceOp)
    {
        if (queryType.TensorType.Shape is not RankedShape queryShape ||
            queryType.AxisPolicies.Count != queryShape.Rank)
        {
            throw new InvalidOperationException(
                "Split-KV PagedAttention requires a ranked query with one SBP policy per tensor axis.");
        }

        var dimensions = queryShape.Dimensions.ToArray();
        dimensions[dimAxis] = accumulator
            ? (dimensions[dimAxis] * GetVectorLaneCount(queryType.TensorType.DType)).Simplify()
            : Dimension.One;
        var stateTensorType = new TensorType(
            DataTypes.Float32,
            new RankedShape(dimensions));
        return new DistributedType(
            stateTensorType,
            queryType.AxisPolicies,
            queryType.Placement,
            SBP.P([plan.SplitHierarchyAxis], reduceOp));
    }

    private int GetVectorLaneCount(DataType dataType)
        => dataType is VectorType vectorType
            ? vectorType.Lanes.Aggregate(1, static (acc, lane) => checked(acc * lane)) *
                GetVectorLaneCount(vectorType.ElemType)
            : 1;

    private Expr GenerateShardedView(
        Call call,
        IR.Distributed.ShardedView shardedView,
        IReadOnlyList<BaseExpr> arguments,
        ref Expr output,
        TIRSelectionContext context)
    {
        if (call[IR.Distributed.ShardedView.Input] is TensorConst tensorConst)
        {
            var constView = T.AttachShardedConstView(
                tensorConst,
                shardedView.NewType,
                out _,
                $"const_sharded_view_{_shardedViewIndex++}");
            if (IsCallerOutputBuffer(output))
            {
                return GenerateTensorStore(constView, output);
            }

            output = constView;
            return T.Nop();
        }

        if (call[IR.Distributed.ShardedView.Input].CheckedType is not DistributedType sourceType)
        {
            throw new InvalidOperationException(
                $"ShardedView TIR selection expects a TensorConst or DistributedType source, got " +
                $"{call[IR.Distributed.ShardedView.Input].CheckedType}.");
        }

        if (arguments[IR.Distributed.ShardedView.Input.Index] is not TIR.Buffer sourceBuffer)
        {
            throw new InvalidOperationException(
                $"ShardedView TIR selection expects its distributed source to lower to TIR.Buffer, got " +
                $"{arguments[IR.Distributed.ShardedView.Input.Index].GetType().Name}.");
        }

        if (!DistributedUtility.TryValidateShardedView(sourceType, shardedView.NewType, out var reason))
        {
            throw new InvalidOperationException($"Invalid ShardedView reached TIR selection: {reason}");
        }

        if (DistributedUtility.IsLocalShardSubview(sourceType, shardedView.NewType))
        {
            var aliasSource = sourceBuffer;
            if (RequiresCanonicalBackingForLocalAlias(sourceType, shardedView.NewType) &&
                sourceBuffer.DistributedStorageKind != DistributedBufferStorageKind.CanonicalGlobal &&
                sourceType.AxisPolicies.Any(policy => policy is not SBPBroadCast))
            {
                (aliasSource, _) = EnsureChipLocalShardedBacking(sourceBuffer, context);
            }

            var localAlias = CreateLocalShardedAlias(
                aliasSource,
                sourceType,
                shardedView.NewType,
                $"local_sharded_view_{_shardedViewIndex++}");
            var localComponentBase = _shardedComponentBases.TryGetValue(aliasSource, out var knownComponentBase)
                ? knownComponentBase
                : aliasSource.MemSpan.Start;
            _shardedComponentBases.TryAdd(aliasSource, localComponentBase);
            _shardedComponentBases.Add(localAlias, localComponentBase);
            output = localAlias;
            return T.Nop();
        }

        var (canonicalSource, componentBase) = EnsureChipLocalShardedBacking(sourceBuffer, context);
        var alias = CreateShardedAlias(
            canonicalSource,
            shardedView.NewType,
            componentBase,
            $"sharded_view_{_shardedViewIndex++}");
        _shardedComponentBases.Add(alias, componentBase);
        output = alias;
        return T.Nop();
    }

    private (TIR.Buffer Buffer, Dimension ComponentBase) EnsureChipLocalShardedBacking(
        TIR.Buffer source,
        TIRSelectionContext context)
    {
        var oldPhysicalBuffer = source.MemSpan.Buffer;
        if (oldPhysicalBuffer.Location == MemoryLocation.Output &&
            oldPhysicalBuffer.Start is BufferVar { Role: BufferVarRole.Output } outputParameter)
        {
            if (source.DistributedType is not { } distributedType)
            {
                throw new InvalidOperationException(
                    $"Caller output {source.Name} reached ShardedView without a DistributedType descriptor.");
            }

            if (source.DistributedStorageKind != DistributedBufferStorageKind.CanonicalGlobal)
            {
                source = CanonicalizeCallerOutputBacking(
                    source,
                    outputParameter,
                    distributedType,
                    context);
            }

            var outputComponentBase = _shardedComponentBases.TryGetValue(source, out var knownComponentBase)
                ? knownComponentBase
                : source.MemSpan.Start;
            _shardedComponentBases.TryAdd(source, outputComponentBase);
            return (source, outputComponentBase);
        }

        if (oldPhysicalBuffer.Location == MemoryLocation.ChipLocalData)
        {
            if (!_shardedComponentBases.TryGetValue(source, out var componentBase))
            {
                throw new InvalidOperationException(
                    $"ChipLocalData buffer {source.Name} reached ShardedView without a canonical component-base binding.");
            }

            return (source, componentBase);
        }

        if (oldPhysicalBuffer.Location != MemoryLocation.Data ||
            oldPhysicalBuffer.Start is not None)
        {
            throw new InvalidOperationException(
                $"ShardedView mutable source must own internal Data or ChipLocalData storage, got " +
                $"{oldPhysicalBuffer.Location}/{oldPhysicalBuffer.Start.GetType().Name}.");
        }

        var userBuffers = oldPhysicalBuffer.Users
            .OfType<TIR.MemSpan>()
            .SelectMany(memSpan => memSpan.Users.OfType<TIR.Buffer>())
            .Distinct((IEqualityComparer<TIR.Buffer>)ReferenceEqualityComparer.Instance)
            .ToArray();
        if (!userBuffers.Contains(source, (IEqualityComparer<TIR.Buffer>)ReferenceEqualityComparer.Instance))
        {
            throw new InvalidOperationException(
                $"ShardedView source {source.Name} is not attached to its physical buffer.");
        }

        var alignment = oldPhysicalBuffer.Alignment;
        Dimension physicalSize = oldPhysicalBuffer.Size;
        foreach (var buffer in userBuffers)
        {
            if (buffer.DistributedType is not { } distributedType)
            {
                throw new InvalidOperationException(
                    $"ShardedView cannot promote physical storage shared with non-distributed buffer {buffer.Name}.");
            }

            var (tensorSize, _) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(
                distributedType.TensorType,
                null);
            var componentBase = _shardedComponentBases.TryGetValue(buffer, out var knownComponentBase)
                ? knownComponentBase
                : buffer.MemSpan.Start;
            alignment = Math.Max(alignment, distributedType.TensorType.DType.SizeInBytes);
            physicalSize = Dimension.Max(
                physicalSize,
                (componentBase + tensorSize).Simplify());
        }

        var chipLocalBuffer = new TIR.PhysicalBuffer(
            alignment,
            physicalSize.Simplify(),
            MemoryLocation.ChipLocalData);
        var rewrittenBuffers = new Dictionary<TIR.Buffer, TIR.Buffer>(
            ReferenceEqualityComparer.Instance);
        foreach (var buffer in userBuffers)
        {
            var componentBase = _shardedComponentBases.TryGetValue(buffer, out var knownComponentBase)
                ? knownComponentBase
                : buffer.MemSpan.Start;
            var rewritten = CreateCanonicalShardedBuffer(
                buffer,
                chipLocalBuffer,
                buffer.DistributedType!,
                componentBase);
            rewrittenBuffers.Add(buffer, rewritten);
            _shardedComponentBases.Add(rewritten, componentBase);
        }

        foreach (var (buffer, rewritten) in rewrittenBuffers)
        {
            ReplaceUtility.ReplaceAllUsesWith(buffer, rewritten);
            context.ReplaceSelectedValue(buffer, rewritten);
        }

        var canonicalSource = rewrittenBuffers[source];
        return (canonicalSource, _shardedComponentBases[canonicalSource]);
    }

    private TIR.Buffer CanonicalizeCallerOutputBacking(
        TIR.Buffer source,
        BufferVar outputParameter,
        DistributedType distributedType,
        TIRSelectionContext context)
    {
        if (outputParameter.Role != BufferVarRole.Output || outputParameter.Location != MemoryLocation.Output)
        {
            throw new InvalidOperationException(
                $"ShardedView source must be backed by a caller-allocated output, got " +
                $"{outputParameter.Role}/{outputParameter.Location}.");
        }

        var (localShape, globalStrides, byteSpanSize) = GetShardedBufferLayout(distributedType);
        var canonicalLayout = BufferLayoutAnnotation.ExactStrided(
            globalStrides,
            DistributedBufferStorageKind.CanonicalGlobal);
        var canonicalParameter = outputParameter.With(layoutAnnotation: canonicalLayout);
        var physicalBuffer = new TIR.PhysicalBuffer(
            distributedType.TensorType.DType.SizeInBytes,
            canonicalParameter,
            byteSpanSize,
            MemoryLocation.Output);
        var canonicalSource = new TIR.Buffer(
            source.Name,
            distributedType.TensorType.DType,
            new TIR.MemSpan(physicalBuffer, 0, byteSpanSize),
            localShape,
            globalStrides,
            distributedType,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
        ReplaceUtility.ReplaceAllUsesWith(outputParameter, canonicalParameter);
        ReplaceUtility.ReplaceAllUsesWith(source, canonicalSource);
        context.ReplaceSelectedValue(outputParameter, canonicalParameter);
        context.ReplaceSelectedValue(source, canonicalSource);
        _shardedComponentBases.Add(canonicalSource, Dimension.Zero);
        return canonicalSource;
    }

    private TIR.Buffer CreateLocalShardedAlias(
        TIR.Buffer source,
        DistributedType sourceType,
        DistributedType targetType,
        string name)
    {
        var sourceDescriptor = GetLocalShardDescriptor(sourceType);
        var targetDescriptor = GetLocalShardDescriptor(targetType);
        var targetShape = targetDescriptor.LocalCapacityShape.Dimensions.ToArray();
        if (!sourceDescriptor.TryGetContiguousRegion(out var sourceOffset, out _) ||
            !targetDescriptor.TryGetContiguousRegion(out var targetOffset, out _))
        {
            if (sourceType.AxisPolicies.SequenceEqual(targetType.AxisPolicies))
            {
                return source.With(name: name, distributedType: targetType);
            }

            if (!DistributedUtility.IsLocalShardSubview(sourceType, targetType))
            {
                throw new InvalidOperationException(
                    $"Local ShardedView cannot alias non-contiguous mapping {sourceType} -> {targetType}.");
            }

            if (source.DistributedStorageKind != DistributedBufferStorageKind.CanonicalGlobal &&
                sourceType.AxisPolicies.Any(policy => policy is not SBPBroadCast))
            {
                throw new InvalidOperationException(
                    $"Local ShardedView requires canonical backing for non-contiguous refinement {sourceType} -> {targetType}.");
            }

            var (_, globalStrides) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(
                targetType.TensorType,
                null);
            return new TIR.Buffer(
                name,
                targetType.TensorType.DType,
                source.MemSpan,
                targetShape,
                globalStrides,
                targetType,
                source.StorageEncoding,
                distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
        }

        if (sourceOffset.Length != source.Rank || targetOffset.Length != source.Rank)
        {
            throw new InvalidOperationException(
                $"Local ShardedView rank mismatch: source buffer={source.Rank}, " +
                $"source type={sourceOffset.Length}, target type={targetOffset.Length}.");
        }

        var relativeOffset = targetOffset
            .Zip(sourceOffset, (target, source) => (target - source).Simplify())
            .ToArray();
        var relativeElementOffset = TensorUtilities.GetLinearOffset(source.Strides, relativeOffset);
        var byteOffset = (relativeElementOffset * source.ElemType.SizeInBytes).Simplify();
        var byteSpanSize = BufferViewUtility.GetByteSpanSize(
            targetShape,
            source.Strides,
            source.ElemType.SizeInBytes);
        return T.CreateBufferView(
            source,
            targetType.TensorType.DType,
            targetShape,
            source.Strides,
            byteOffset,
            byteSpanSize,
            targetType,
            name);
    }

    private bool RequiresCanonicalBackingForLocalAlias(
        DistributedType sourceType,
        DistributedType targetType)
    {
        if (sourceType.AxisPolicies.SequenceEqual(targetType.AxisPolicies))
        {
            return false;
        }

        var sourceDescriptor = GetLocalShardDescriptor(sourceType);
        var targetDescriptor = GetLocalShardDescriptor(targetType);
        return !sourceDescriptor.TryGetContiguousRegion(out _, out _) ||
            !targetDescriptor.TryGetContiguousRegion(out _, out _);
    }

    private LocalShardDescriptor GetLocalShardDescriptor(DistributedType distributedType)
    {
        if (!_shardCoordinates.TryGetValue(distributedType.Placement, out var shardIndex))
        {
            shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                .Select(axis => (Dimension)new DimVar($"__shard_coord_{axis}"))
                .ToArray();
            _shardCoordinates.Add(distributedType.Placement, shardIndex);
        }

        return DistributedUtility.GetLocalShardDescriptor(
            distributedType,
            shardIndex,
            DistributedUtility.DivideFlags.MaxShape);
    }

    private TIR.Buffer CreateCanonicalShardedBuffer(
        TIR.Buffer source,
        TIR.PhysicalBuffer physicalBuffer,
        DistributedType distributedType,
        Dimension componentBase)
    {
        var (localShape, globalStrides, byteSpanSize) = GetShardedBufferLayout(distributedType);
        return source.With(
            memSpan: new TIR.MemSpan(
                physicalBuffer,
                componentBase,
                byteSpanSize),
            dimensions: localShape,
            strides: globalStrides,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
    }

    private TIR.Buffer CreateShardedAlias(
        TIR.Buffer source,
        DistributedType targetType,
        Dimension componentBase,
        string name)
    {
        var (targetShape, targetStrides, targetByteSpanSize) = GetShardedBufferLayout(targetType);
        return new TIR.Buffer(
            name,
            targetType.TensorType.DType,
            new TIR.MemSpan(
                source.MemSpan.Buffer,
                componentBase,
                targetByteSpanSize),
            targetShape,
            targetStrides,
            targetType,
            source.StorageEncoding,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
    }

    private (Dimension[] Shape, Dimension[] Strides, Dimension ByteSpanSize) GetShardedBufferLayout(
        DistributedType distributedType)
    {
        if (!_shardCoordinates.TryGetValue(distributedType.Placement, out var shardIndex))
        {
            shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                .Select(axis => (Dimension)new DimVar($"__shard_coord_{axis}"))
                .ToArray();
            _shardCoordinates.Add(distributedType.Placement, shardIndex);
        }

        var shardDescriptor = GetLocalShardDescriptor(distributedType);
        var localShape = shardDescriptor.LocalCapacityShape.Dimensions.ToArray();
        var (globalSize, globalStrides) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(
            distributedType.TensorType,
            null);
        return (localShape, globalStrides, globalSize);
    }

    private Expr GenerateBufferView(string opName, Expr input, ref Expr output)
    {
        if (input is not TIR.Buffer inputBuffer)
        {
            throw new NotSupportedException($"{opName} only supports buffer input, got {input.GetType().Name}.");
        }

        if (IsCallerOutputBuffer(output))
        {
            return GenerateTensorStore(CreateLogicalView(opName, inputBuffer, (TIR.Buffer)output), output);
        }

        if (output is not TIR.Buffer selectedOutput)
        {
            throw new NotSupportedException(
                $"{opName} output must be a buffer or caller output BufferVar, got {output.GetType().Name}.");
        }

        var logicalView = CreateLogicalView(opName, inputBuffer, selectedOutput);
        if (_shardedComponentBases.TryGetValue(inputBuffer, out var componentBase))
        {
            _shardedComponentBases.Add(logicalView, componentBase);
        }

        output = logicalView;
        return T.Nop();
    }

    private static TIR.Buffer CreateLogicalView(string opName, TIR.Buffer input, TIR.Buffer output)
    {
        var inputType = GetBufferType(input);
        var outputType = GetBufferType(output);
        if (!BufferViewUtility.TryCreate(inputType, outputType, out var transform))
        {
            throw new InvalidOperationException(
                $"{opName} cannot create a storage-preserving buffer view from {inputType} to {outputType}.");
        }

        return BufferViewUtility.CreateLogicalBufferView(input, outputType, transform, output.Name);
    }

    private TIR.Buffer CreateMetadataBuffer(Expr expr, MemoryLocation location, string namePrefix)
        => CreateMetadataBuffer(expr.CheckedType, location, namePrefix);

    private TIR.Buffer CreateMetadataBuffer(IRType type, MemoryLocation location, string namePrefix)
    {
        var (tensorType, distributedType) = GetTensorTypeAndDistributedType(type, namePrefix);
        T.CreateBuffer(tensorType, location, out var buffer, $"{namePrefix}_{_bufferIndex++}", distributedType);
        return buffer;
    }

    private Expr GenerateSamplingPartial(
        IR.NTT.SamplingPartial sampling,
        IReadOnlyList<BaseExpr> arguments,
        Expr output)
    {
        var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
        if (outputBase is not IR.Tuple { Count: 2 } outputs)
        {
            throw new NotSupportedException(
                "SamplingPartial TIR selection expects processed logits and a partial maximum state.");
        }

        if (arguments[IR.NTT.SamplingPartial.Logits.Index] is not TIR.Buffer { DistributedType: { } logitsType } ||
            !Evaluator.IR.NTT.SamplingSplitTypeUtility.IsFullyVocabularySharded(logitsType))
        {
            throw new NotSupportedException(
                "SamplingPartial requires its vocabulary axis to cover every placement axis exactly once.");
        }

        return TIR.F.NTT.SamplingPartial(
            (Expr)arguments[IR.NTT.SamplingPartial.Logits.Index],
            (Expr)arguments[IR.NTT.SamplingPartial.State.Index],
            (Expr)outputs[0],
            (Expr)outputs[1],
            sampling.Config);
    }

    private Expr GeneratePackedMatMulSamplingPartial(
        Call call,
        IR.NTT.PackedMatMulSamplingPartial sampling,
        IReadOnlyList<BaseExpr> arguments,
        Expr output)
    {
        var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
        if (outputBase is not IR.Tuple { Count: 3 } outputs)
        {
            throw new NotSupportedException(
                "PackedMatMulSamplingPartial TIR selection expects raw logits, processed logits, and a partial maximum state.");
        }

        if (_compileOptions.TargetOptions is not PyNTTTargetOptions)
        {
            throw new NotSupportedException(
                "PackedMatMulSamplingPartial is supported only by the PyNTT target.");
        }

        var packedOutput = Evaluator.IR.NTT.PackedMatMulEvaluator.InferType(
            new IR.NTT.PackedMatMul(
                sampling.OutputDataType,
                false,
                sampling.RhsLayout),
            call[IR.NTT.PackedMatMulSamplingPartial.Lhs].CheckedType,
            call[IR.NTT.PackedMatMulSamplingPartial.Rhs].CheckedType,
            call[IR.NTT.PackedMatMulSamplingPartial.Scale].CheckedType,
            call[IR.NTT.PackedMatMulSamplingPartial.Addend].CheckedType);
        if (packedOutput is not DistributedType { Partial: null } packedOutputType)
        {
            throw new NotSupportedException(
                $"PackedMatMulSamplingPartial requires a non-partial distributed packed output, got {packedOutput}.");
        }

        var logits = BitcastUtility.InferType(
            packedOutputType,
            sampling.OutputDataType);
        if (logits is not DistributedType logitsType ||
            !Evaluator.IR.NTT.SamplingSplitTypeUtility.IsFullyVocabularySharded(logitsType))
        {
            throw new NotSupportedException(
                "PackedMatMulSamplingPartial requires its scalar vocabulary axis to cover every placement axis exactly once.");
        }

        return TIR.F.NTT.PackedMatMulSamplingPartial(
            (Expr)arguments[IR.NTT.PackedMatMulSamplingPartial.Lhs.Index],
            (Expr)arguments[IR.NTT.PackedMatMulSamplingPartial.Rhs.Index],
            (Expr)arguments[IR.NTT.PackedMatMulSamplingPartial.State.Index],
            (Expr)outputs[0],
            (Expr)outputs[1],
            (Expr)outputs[2],
            (Expr)arguments[IR.NTT.PackedMatMulSamplingPartial.Scale.Index],
            (Expr)arguments[IR.NTT.PackedMatMulSamplingPartial.Addend.Index],
            sampling.RhsLayout,
            packedOutputType,
            logitsType,
            sampling.Config);
    }

    private Expr GenerateSamplingCombine(
        IR.NTT.SamplingCombine sampling,
        IReadOnlyList<BaseExpr> arguments,
        ref Expr output)
    {
        var outputBase = Unsafe.As<Expr, BaseExpr>(ref output);
        if (outputBase is not IR.Tuple outputs || outputs.Count != 6)
        {
            throw new NotSupportedException(
                "SamplingCombine TIR selection expects five tensor outputs followed by the sampler state.");
        }

        if (_compileOptions.TargetOptions is not PyNTTTargetOptions)
        {
            throw new NotSupportedException("SamplingCombine is currently supported only by the PyNTT target.");
        }

        if (arguments[IR.NTT.SamplingCombine.Logits.Index] is not TIR.Buffer { DistributedType: { } logitsType } ||
            !Evaluator.IR.NTT.SamplingSplitTypeUtility.IsFullyVocabularySharded(logitsType))
        {
            throw new NotSupportedException(
                "SamplingCombine requires its vocabulary axis to cover every placement axis exactly once.");
        }

        if (arguments[IR.NTT.SamplingCombine.ArgMaxState.Index] is not TIR.Buffer { DistributedType: { } argMaxType } ||
            !Evaluator.IR.NTT.SamplingSplitTypeUtility.IsArgMaxPartial(argMaxType, logitsType.Placement))
        {
            throw new NotSupportedException(
                "SamplingCombine requires a full-mesh P(Max) state.");
        }

        var blockCount = logitsType.Placement.Hierarchy
            .Aggregate(1, (product, extent) => checked(product * extent));
        const int radixBits = 8;
        var radixBins = 1 << radixBits;
        var summary = CreateMetadataBuffer(
            new TensorType(
                DataTypes.Float32,
                new RankedShape(sampling.Config.MaxBatchSize, blockCount, radixBins + 16)),
            MemoryLocation.ChipLocalData,
            "sampling_summary");

        var materializedOutputs = outputs.Fields[..5]
            .AsValueEnumerable()
            .Select((field, index) => CreateCanonicalReplicatedOutput(
                field,
                $"SamplingCombine output {index}"))
            .ToArray();

        Unsafe.As<Expr, BaseExpr>(ref output) = new IR.Tuple(
            materializedOutputs[0],
            materializedOutputs[1],
            materializedOutputs[2],
            materializedOutputs[3],
            materializedOutputs[4],
            arguments[IR.NTT.SamplingCombine.State.Index]);
        return TIR.F.NTT.SamplingCombine(
            (Expr)arguments[IR.NTT.SamplingCombine.Logits.Index],
            (Expr)arguments[IR.NTT.SamplingCombine.ProcessedLogits.Index],
            (Expr)arguments[IR.NTT.SamplingCombine.ArgMaxState.Index],
            (Expr)arguments[IR.NTT.SamplingCombine.State.Index],
            summary,
            materializedOutputs[0],
            materializedOutputs[1],
            materializedOutputs[2],
            materializedOutputs[3],
            materializedOutputs[4],
            sampling.Config,
            blockCount,
            radixBits);
    }

    private TIR.Buffer CreateCanonicalReplicatedOutput(BaseExpr output, string context)
    {
        if (output is not TIR.Buffer
            {
                DistributedType:
                {
                    Partial: null,
                } distributedType,
            } outputBuffer ||
            !distributedType.AxisPolicies.AsValueEnumerable().All(policy => policy is SBPBroadCast))
        {
            throw new InvalidOperationException(
                $"{context} must be a fully broadcast distributed tensor buffer, got {output.CheckedType}.");
        }

        var (_, _, byteSpanSize) = GetShardedBufferLayout(distributedType);
        var physicalBuffer = new TIR.PhysicalBuffer(
            distributedType.TensorType.DType.SizeInBytes,
            byteSpanSize,
            MemoryLocation.ChipLocalData);
        var canonical = CreateCanonicalShardedBuffer(
            outputBuffer,
            physicalBuffer,
            distributedType,
            Dimension.Zero);
        _shardedComponentBases.Add(canonical, Dimension.Zero);
        return canonical;
    }

    private static (TensorType TensorType, DistributedType? DistributedType) GetTensorTypeAndDistributedType(IRType type, string context)
        => type switch
        {
            DistributedType distributedType => (distributedType.TensorType, distributedType),
            TensorType tensorType => (tensorType, null),
            _ => throw new NotSupportedException($"{context} expects a tensor type, got {type}."),
        };

    private static IRType GetBufferType(TIR.Buffer buffer)
        => (IRType?)buffer.DistributedType ?? new TensorType(buffer.ElemType, buffer.Dimensions.ToArray());

    private static bool IsCallerOutputBuffer(Expr expr)
        => expr is TIR.Buffer
        {
            MemSpan.Buffer.Start: BufferVar { Role: BufferVarRole.Output },
        };

    private static bool IsCompactPartialReshardResult(Expr sourceExpr)
        => sourceExpr is Call sourceCall &&
            sourceCall.Target is IR.Distributed.Boxing &&
            sourceCall[IR.Distributed.Boxing.Input].CheckedType is DistributedType { Partial: not null } &&
            sourceCall.CheckedType is DistributedType { Partial: null } outputType &&
            outputType.AxisPolicies.AsValueEnumerable().All(policy => policy is SBPBroadCast);

    private static Expr GenerateTensorStore(TIR.Buffer source, Expr destination)
    {
        var distributedType = source.DistributedType;
        return TIR.F.NTT.TensorStore(
            source,
            destination,
            distributedType?.AxisPolicies ?? new IRArray<SBP>(),
            distributedType?.Placement ?? new Placement(new IRArray<int>(), string.Empty, string.Empty));
    }

    private static Expr GenerateBoxing(
        Call call,
        IR.Distributed.Boxing boxing,
        IReadOnlyList<BaseExpr> arguments,
        ref Expr output)
    {
        return (call[IR.Distributed.Boxing.Input].CheckedType, boxing.NewType) switch
        {
            (TensorType, DistributedType outputType) => TIR.F.NTT.TensorLoad(output, (Expr)arguments[0], outputType.AxisPolicies, outputType.Placement),
            (DistributedType inputType, TensorType) => TIR.F.NTT.TensorStore((Expr)arguments[0], output, inputType.AxisPolicies, inputType.Placement),
            (DistributedType inputType, DistributedType outputType) => GenerateReshard((Expr)arguments[0], output, inputType, outputType),
            _ => throw new NotSupportedException(
                $"Unsupported Boxing TIR selection from {call[IR.Distributed.Boxing.Input].CheckedType} to {boxing.NewType}."),
        };
    }

    private static Expr GenerateReshard(Expr input, Expr output, DistributedType inputType, DistributedType outputType)
        => TIR.F.NTT.GatherReduceScatter(input, output, inputType, outputType);

    private Expr GenerateUnary(UnaryOp unaryOp, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var input = (Expr)arguments[IR.Math.Unary.Input.Index];
        return TIR.F.NTT.Unary(unaryOp, input, output);
    }

    private Expr GenerateClamp(Call call, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var min = ((TensorConst)call[IR.Math.Clamp.Min]).Value.ToScalar<float>();
        var max = ((TensorConst)call[IR.Math.Clamp.Max]).Value.ToScalar<float>();
        return TIR.F.NTT.Clamp((Expr)arguments[0], output, min, max);
    }
}
