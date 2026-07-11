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
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes;

public sealed class NTTTIRSelectionPass : TIRSelectionPass
{
    private readonly CompileOptions _compileOptions;
    private int _bufferIndex;

    public NTTTIRSelectionPass(CompileOptions compileOptions, string moduleKind = CPUTarget.Kind)
        : base(moduleKind)
    {
        _compileOptions = compileOptions;
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
            case IR.Distributed.Boxing:
                throw new InvalidOperationException("Boxing must be lowered to an affine transfer before TIR selection.");
            case AffineView affineView:
                return GenerateAffineView(call, affineView, arguments, ref output);
            case IR.Distributed.ForceBoxing forceBoxing:
                return T.Memcopy(output, (Expr)arguments[0]);
            case IR.Math.Binary binary:
                return TIR.F.NTT.VectorizedBinary((Expr)arguments[0], (Expr)arguments[1], output, None.Default, binary.BinaryOp, Array.Empty<int>(), Array.Empty<Dimension>(), Array.Empty<int>(), Array.Empty<Dimension>());
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
                return TIR.F.NTT.PackedMatMul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.PackedMatMul.Scale], matmul.FusedReduce);
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
                    matmulGlu.GluType);
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
                        qkv.NumKvHeads);
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
            case IR.NTT.VectorizedReduce pr:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, pr.VectorizedAxes.ToArray(), ((RankedShape)call[IR.NTT.VectorizedReduce.PadedNums]).Dimensions.ToArray(), pr.Axes, pr.KeepDims, pr.ReduceOp);
            case IR.Math.Compare compare:
                return TIR.F.NTT.Compare(compare.CompareOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.GetItem getItem:
                return TIR.F.NTT.GetItem((Expr)arguments[0], arguments[1], output);
            case IR.Tensors.ScatterND scatterND:
                return TIR.F.NTT.ScatterND((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Stack stack:
                return TIR.F.NTT.Stack(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, ((TensorConst)call[IR.Tensors.Stack.Axis]).Value.ToScalar<int>());
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
                return TIR.F.NTT.PagedAttention((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Dimension)arguments[4], output, pgat.Layout, pgat.HiddenSize);
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
            case IR.NN.SparseExperts sparseExperts:
                return TIR.F.NTT.SparseExperts((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8], (Expr)arguments[9], (Expr)arguments[10], (Expr)arguments[11], output, sparseExperts.HiddenSize, sparseExperts.MoEIntermediateSize, sparseExperts.NumExpert, sparseExperts.NumTopK, sparseExperts.ChunkSize);
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

    private Expr GenerateAffineView(Call call, AffineView affineView, IReadOnlyList<BaseExpr> arguments, ref Expr output)
    {
        if (arguments[AffineView.Input.Index] is not TIR.Buffer inputBuffer)
        {
            throw new NotSupportedException($"AffineView input must lower to TIR.Buffer, got {arguments[AffineView.Input.Index].GetType().Name}.");
        }

        if (call[AffineView.Input] is TensorConst tensorConst && affineView.NewType is DistributedType shardedType)
        {
            output = T.AttachShardedConstView(tensorConst, shardedType, out _, $"affine_const_view_{_bufferIndex++}");
            return T.Nop();
        }

        var resultTensorType = affineView.NewType switch
        {
            DistributedType distributedType => distributedType.TensorType,
            TensorType tensorType => tensorType,
            _ => throw new NotSupportedException($"AffineView result must be a tensor type, got {affineView.NewType}."),
        };
        if (resultTensorType.Shape is not RankedShape resultShape)
        {
            throw new NotSupportedException("AffineView TIR lowering requires a ranked result shape.");
        }

        var resultDimensions = resultShape.Dimensions.ToArray();
        var resultStrides = CreateAffineViewStrides(inputBuffer, resultTensorType, affineView.Transform);
        var name = output switch
        {
            BufferVar outputVar => $"{outputVar.Name}_view",
            TIR.Buffer outputBuffer => outputBuffer.Name,
            _ => $"affine_view_{_bufferIndex++}",
        };
        output = T.CreateBufferView(
            inputBuffer,
            resultTensorType.DType,
            resultDimensions,
            resultStrides,
            0,
            inputBuffer.MemSpan.Size,
            affineView.NewType as DistributedType,
            name);
        return T.Nop();
    }

    private static Dimension[] CreateAffineViewStrides(TIR.Buffer source, TensorType resultType, AffineViewTransform transform)
    {
        var sourceDefaultStrides = TensorUtilities.GetDefaultStrides(source.Dimensions);
        var prefixRank = 0;
        var comparableRank = System.Math.Min(transform.SourceMap.Results.Length, transform.ResultMap.Results.Length);
        while (prefixRank < comparableRank && transform.SourceMap.Results[prefixRank].Equals(transform.ResultMap.Results[prefixRank]))
        {
            prefixRank++;
        }

        for (var axis = prefixRank; axis < source.Rank; axis++)
        {
            if (!source.Strides[axis].Equals(sourceDefaultStrides[axis]))
            {
                throw new NotSupportedException(
                    $"AffineView cannot reshape non-contiguous source suffix at axis {axis}: stride={source.Strides[axis]}, expected={sourceDefaultStrides[axis]}.");
            }
        }

        var resultDimensions = ((RankedShape)resultType.Shape).Dimensions.ToArray();
        var resultStrides = TensorUtilities.GetDefaultStrides(resultDimensions);
        var sharedPrefixRank = System.Math.Min(prefixRank, System.Math.Min(source.Rank, resultDimensions.Length));
        for (var axis = 0; axis < sharedPrefixRank; axis++)
        {
            var sourceByteStride = source.Strides[axis] * source.ElemType.SizeInBytes;
            if (sourceByteStride is DimConst byteStride && byteStride.Value % resultType.DType.SizeInBytes != 0)
            {
                throw new NotSupportedException(
                    $"AffineView byte stride {byteStride.Value} at axis {axis} is not aligned to result element size {resultType.DType.SizeInBytes}.");
            }

            resultStrides[axis] = (sourceByteStride / resultType.DType.SizeInBytes).Simplify();
        }

        return resultStrides;
    }

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
