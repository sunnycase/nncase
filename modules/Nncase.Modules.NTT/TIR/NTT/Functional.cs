// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Shapes;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.TIR.F;

public partial class NTT
{
    private static Call KernelCall(NTTKernelOp op, params BaseExpr[] arguments)
        => op.CreateCall(arguments);

    public static Expr GatedDeltaNetConvolution(
        Expr qkv,
        Expr state,
        Expr convWeight,
        Expr qkvOutput,
        Dimension layerId,
        long convKernelSize) =>
        KernelCall(
            new TIR.NTT.GatedDeltaNetConvolution(convKernelSize),
            qkv,
            state,
            convWeight,
            qkvOutput,
            layerId);

    public static Expr GatedDeltaNetRecurrentCore(
        Expr state,
        Expr qkv,
        Expr z,
        Expr projectionInput,
        Expr bWeight,
        Expr aWeight,
        Expr aLog,
        Expr dtBias,
        Expr normWeight,
        Expr coreScratch,
        Expr gatedOutput,
        Dimension layerId,
        long numKeyHeads,
        long numValueHeads,
        long keyHeadDim,
        long valueHeadDim,
        float epsilon) =>
        KernelCall(
            new TIR.NTT.GatedDeltaNetRecurrentCore(
                numKeyHeads,
                numValueHeads,
                keyHeadDim,
                valueHeadDim,
                epsilon),
            state,
            qkv,
            z,
            projectionInput,
            bWeight,
            aWeight,
            aLog,
            dtBias,
            normWeight,
            coreScratch,
            gatedOutput,
            layerId);

    /// <summary>
    /// the ptr of can create the *PtrName in the c code.
    /// </summary>
    /// <param name="name">c pointer name.</param>
    /// <param name="primType">type.</param>
    /// <returns>call.</returns>
    public static Call PtrOf(string name, DataType primType) => new Call(new PtrOf(name, primType));

    public static Call SramPtr(Expr input, DataType primType) => new Call(new SramPtr(primType), input);

    public static Call TensorLoad(Expr dest, Expr src, IRArray<SBP> ndsbp, Placement placement)
    {
        return KernelCall(new TensorLoad(ndsbp, placement), dest, src);
    }

    public static Call TensorStore(Expr src, Expr dest, IRArray<SBP> ndsbp, Placement placement)
    {
        return KernelCall(new TensorStore(ndsbp, placement), src, dest);
    }

    public static Call Unary(UnaryOp unaryOp, Expr input, Expr output)
    {
        return KernelCall(new TIR.NTT.Unary(unaryOp), input, output);
    }

    public static Call Reshape(Expr input, Expr output)
    {
        return KernelCall(new TIR.NTT.Reshape(), input, output);
    }

    public static Call Bitcast(Expr input, Expr output)
    {
        return KernelCall(new TIR.NTT.Bitcast(), input, output);
    }

    public static Call Matmul(Expr lhs, Expr rhs, Expr output, Expr loadC, Expr scale, Expr extra, IRArray<int> lhsVectorizedAxes, IRArray<int> rhsVectorizedAxes, bool transA = false, bool transB = false, bool fusedReduce = false, string cSourcePath = "", string funcName = "")
    {
        return KernelCall(new Matmul(lhsVectorizedAxes, rhsVectorizedAxes, transA, transB, fusedReduce, cSourcePath, funcName), lhs, rhs, output, loadC, scale, extra);
    }

    public static Call Matmul(Expr lhs, Expr rhs, Expr output, Expr loadC, Expr scale)
    {
        return KernelCall(new Matmul(new IRArray<int>(), new IRArray<int>(), false, false, false, null, null), lhs, rhs, output, loadC, scale, None.Default);
    }

    public static Call PackedMatMul(
        Expr lhs,
        Expr rhs,
        Expr output,
        Expr loadC,
        Expr scale,
        bool fusedReduce = false,
        IR.NTT.PackedMatMulRhsLayout rhsLayout = IR.NTT.PackedMatMulRhsLayout.NMajor,
        Expr? addend = null)
    {
        return KernelCall(
            new PackedMatMul(fusedReduce, rhsLayout),
            lhs,
            rhs,
            output,
            loadC,
            scale,
            addend ?? None.Default);
    }

    public static Call PackedScaledMatMul(
        Expr lhs,
        Expr rhs,
        Expr lhsScale,
        Expr rhsScale,
        Expr output,
        IR.NTT.PackedMatMulRhsLayout rhsLayout)
    {
        return KernelCall(
            new PackedScaledMatMul(rhsLayout),
            lhs,
            rhs,
            lhsScale,
            rhsScale,
            output);
    }

    public static Call PackedBlockScaledMatMul(
        Expr lhs,
        Expr rhs,
        Expr rhsScale,
        Expr output,
        IR.NTT.PackedMatMulRhsLayout rhsLayout,
        int outputNVectorLaneCount,
        long weightBlockN,
        long weightBlockK,
        Expr? addend = null)
    {
        return KernelCall(
            new PackedBlockScaledMatMul(
                rhsLayout,
                outputNVectorLaneCount,
                weightBlockN,
                weightBlockK),
            lhs,
            rhs,
            rhsScale,
            output,
            addend ?? None.Default);
    }

    public static Call NVFP4MatMul(
        Expr lhs,
        Expr rhsPacked,
        Expr rhsScale,
        Expr lhsGlobalScale,
        Expr rhsGlobalScale,
        Expr output,
        long groupSize)
    {
        return KernelCall(
            new Nncase.TIR.NTT.NVFP4MatMul(groupSize),
            lhs,
            rhsPacked,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            output);
    }

    public static Call PackedBlockScaledMatMulNormStats(
        Expr lhs,
        Expr rhs,
        Expr rhsScale,
        Expr output,
        Expr stats,
        Expr addend,
        IR.NTT.PackedMatMulRhsLayout rhsLayout,
        int outputNVectorLaneCount,
        long weightBlockN,
        long weightBlockK,
        int axis,
        bool useMean)
    {
        return KernelCall(
            new PackedBlockScaledMatMulNormStats(
                rhsLayout,
                outputNVectorLaneCount,
                weightBlockN,
                weightBlockK,
                axis,
                useMean),
            lhs,
            rhs,
            rhsScale,
            output,
            stats,
            addend);
    }

    public static Call PackedMatMulNormStats(
        Expr lhs,
        Expr rhs,
        Expr output,
        Expr stats,
        Expr loadC,
        Expr scale,
        IR.NTT.PackedMatMulRhsLayout rhsLayout,
        int axis,
        bool useMean,
        Expr? addend = null)
    {
        return KernelCall(
            new PackedMatMulNormStats(rhsLayout, axis, useMean),
            lhs,
            rhs,
            output,
            stats,
            loadC,
            scale,
            addend ?? None.Default);
    }

    public static Call PackedMatMulSamplingPartial(
        Expr lhs,
        Expr rhs,
        Expr state,
        Expr logits,
        Expr processedLogits,
        Expr argMaxState,
        Expr scale,
        Expr addend,
        IR.NTT.PackedMatMulRhsLayout rhsLayout,
        DistributedType packedOutputType,
        DistributedType logitsType,
        IR.NN.SamplerConfig config)
    {
        return KernelCall(
            new PackedMatMulSamplingPartial(
                rhsLayout,
                packedOutputType,
                logitsType,
                config),
            lhs,
            rhs,
            state,
            logits,
            processedLogits,
            argMaxState,
            scale,
            addend);
    }

    public static Call QKVParallelLinear(
        Expr input,
        Expr qWeight,
        Expr kWeight,
        Expr vWeight,
        Expr qBias,
        Expr kBias,
        Expr vBias,
        Expr qInputScale,
        Expr kInputScale,
        Expr vInputScale,
        Expr qWeightScale,
        Expr kWeightScale,
        Expr vWeightScale,
        Expr qOutput,
        Expr kOutput,
        Expr vOutput,
        long numHeads,
        long numKvHeads)
    {
        return KernelCall(
            new QKVParallelLinear(numHeads, numKvHeads),
            input,
            qWeight,
            kWeight,
            vWeight,
            qBias,
            kBias,
            vBias,
            qInputScale,
            kInputScale,
            vInputScale,
            qWeightScale,
            kWeightScale,
            vWeightScale,
            qOutput,
            kOutput,
            vOutput);
    }

    public static Call PackedQKVParallelLinear(
        Expr input,
        Expr qWeight,
        Expr kWeight,
        Expr vWeight,
        Expr qBias,
        Expr kBias,
        Expr vBias,
        Expr qInputScale,
        Expr kInputScale,
        Expr vInputScale,
        Expr qWeightScale,
        Expr kWeightScale,
        Expr vWeightScale,
        Expr qOutput,
        Expr kOutput,
        Expr vOutput,
        long numHeads,
        long numKvHeads,
        IR.NTT.PackedMatMulRhsLayout rhsLayout = IR.NTT.PackedMatMulRhsLayout.NMajor)
    {
        return KernelCall(
            new PackedQKVParallelLinear(numHeads, numKvHeads, rhsLayout),
            input,
            qWeight,
            kWeight,
            vWeight,
            qBias,
            kBias,
            vBias,
            qInputScale,
            kInputScale,
            vInputScale,
            qWeightScale,
            kWeightScale,
            vWeightScale,
            qOutput,
            kOutput,
            vOutput);
    }

    public static Call PackedQKVParallelLinearFusedRhs(
        Expr input,
        Expr weight,
        Expr qBias,
        Expr kBias,
        Expr vBias,
        Expr qInputScale,
        Expr kInputScale,
        Expr vInputScale,
        Expr qWeightScale,
        Expr kWeightScale,
        Expr vWeightScale,
        Expr qOutput,
        Expr kOutput,
        Expr vOutput,
        long numHeads,
        long numKvHeads,
        IR.NTT.PackedMatMulRhsLayout rhsLayout,
        IRArray<long> projectionNCapacities)
    {
        return KernelCall(
            new PackedQKVParallelLinearFusedRhs(
                numHeads,
                numKvHeads,
                rhsLayout,
                projectionNCapacities),
            input,
            weight,
            qBias,
            kBias,
            vBias,
            qInputScale,
            kInputScale,
            vInputScale,
            qWeightScale,
            kWeightScale,
            vWeightScale,
            qOutput,
            kOutput,
            vOutput);
    }

    public static Call MatMulGlu(
        Expr input,
        Expr gateWeight,
        Expr upWeight,
        Expr gateBias,
        Expr upBias,
        Expr gateInputScale,
        Expr upInputScale,
        Expr gateWeightScale,
        Expr upWeightScale,
        Expr output,
        IR.NN.GluType gluType,
        global::Nncase.IR.Math.MatMulQuantizationMode quantizationMode =
            global::Nncase.IR.Math.MatMulQuantizationMode.None,
        long weightBlockN = 0,
        long weightBlockK = 0)
    {
        return KernelCall(
            new MatMulGlu(gluType, quantizationMode, weightBlockN, weightBlockK),
            input,
            gateWeight,
            upWeight,
            gateBias,
            upBias,
            gateInputScale,
            upInputScale,
            gateWeightScale,
            upWeightScale,
            output);
    }

    public static Call NVFP4MatMulGlu(
        Expr input,
        Expr gateWeightPacked,
        Expr upWeightPacked,
        Expr gateWeightScale,
        Expr upWeightScale,
        Expr gateInputGlobalScale,
        Expr upInputGlobalScale,
        Expr gateWeightGlobalScale,
        Expr upWeightGlobalScale,
        Expr output,
        IR.NN.GluType gluType,
        long groupSize)
    {
        return KernelCall(
            new NVFP4MatMulGlu(gluType, groupSize),
            input,
            gateWeightPacked,
            upWeightPacked,
            gateWeightScale,
            upWeightScale,
            gateInputGlobalScale,
            upInputGlobalScale,
            gateWeightGlobalScale,
            upWeightGlobalScale,
            output);
    }

    public static Call PackedMatMulGlu(
        Expr input,
        Expr gateWeight,
        Expr upWeight,
        Expr gateBias,
        Expr upBias,
        Expr gateInputScale,
        Expr upInputScale,
        Expr gateWeightScale,
        Expr upWeightScale,
        Expr output,
        IR.NN.GluType gluType,
        IR.NTT.PackedMatMulRhsLayout rhsLayout = IR.NTT.PackedMatMulRhsLayout.NMajor,
        global::Nncase.IR.Math.MatMulQuantizationMode quantizationMode =
            global::Nncase.IR.Math.MatMulQuantizationMode.None,
        long weightBlockN = 0,
        long weightBlockK = 0)
    {
        return KernelCall(
            new PackedMatMulGlu(
                gluType,
                rhsLayout,
                quantizationMode,
                weightBlockN,
                weightBlockK),
            input,
            gateWeight,
            upWeight,
            gateBias,
            upBias,
            gateInputScale,
            upInputScale,
            gateWeightScale,
            upWeightScale,
            output);
    }

    public static Call SUMMA(Expr lhs, Expr rhs, Expr output, Expr loadC, Expr scale, IRArray<int> lhsVectorizedAxes, IRArray<int> rhsVectorizedAxes, bool transA = false, bool transB = false)
    {
        return KernelCall(new SUMMA(lhsVectorizedAxes, rhsVectorizedAxes, transA, transB), lhs, rhs, output, loadC, scale);
    }

    public static Call SUMMA(Expr lhs, Expr rhs, Expr output, Expr loadC, Expr scale)
    {
        return KernelCall(new SUMMA(new IRArray<int>(), new IRArray<int>(), false, false), lhs, rhs, output, loadC, scale);
    }

    public static Expr Pack(Expr input, Expr output, IRArray<int> lanes, IRArray<int> axes)
    {
        return KernelCall(new Pack(lanes, axes), input, output);
    }

    public static Call Conv2D(Expr input, Expr weights, Expr bias, Expr output, long[] stride, long[] padding, long[] dilation, long groups, PadMode padMode, DistributedType distributedType) => KernelCall(new Conv2D(stride, padding, dilation, groups, padMode, distributedType), input, weights, bias, output);

    public static Expr Unpack(Expr input, Expr output, IRArray<int> lanes, IRArray<int> axes)
    {
        return KernelCall(new Unpack(lanes, axes), input, output);
    }

    public static Expr VectorizedSoftmax(Expr input, Expr output, int axis, IRArray<int> vectorizedAxes)
    {
        return KernelCall(new VectorizedSoftmax(axis, vectorizedAxes), input, output);
    }

    public static Expr VectorizedLayerNorm(Expr input, Expr scale, Expr bias, Expr output, int axis, float epsilon, bool usemean, IRArray<int> vectorizedAxes, IRArray<Dimension> padedNums, string cSourcePath = "", string funcName = "")
    {
        return KernelCall(new VectorizedLayerNorm(axis, epsilon, usemean, vectorizedAxes, padedNums, null!, cSourcePath, funcName), input, scale, bias, None.Default, output);
    }

    public static Expr VectorizedLayerNorm(Expr input, Expr scale, Expr bias, Expr output, int axis, float epsilon, bool usemean, IRArray<int> vectorizedAxes, IRArray<Dimension> padedNums, Expr postScale, string cSourcePath = "", string funcName = "")
    {
        return KernelCall(new VectorizedLayerNorm(axis, epsilon, usemean, vectorizedAxes, padedNums, null!, cSourcePath, funcName), input, scale, bias, postScale, output);
    }

    public static Expr NormStats(Expr input, Expr output, int axis, bool useMean)
    {
        return KernelCall(new NormStats(axis, useMean), input, output);
    }

    public static Expr NormApply(Expr input, Expr stats, Expr scale, Expr bias, Expr output, int axis, float epsilon, bool useMean)
    {
        return KernelCall(new NormApply(axis, epsilon, useMean), input, stats, scale, bias, output);
    }

    public static Expr GatherReduceNormApply(
        Expr partialStats,
        Expr input,
        Expr scale,
        Expr bias,
        Expr output,
        DistributedType inStatsType,
        DistributedType outStatsType,
        int axis,
        float epsilon,
        bool useMean,
        bool hasBias)
    {
        return KernelCall(
            new GatherReduceNormApply(inStatsType, outStatsType, axis, epsilon, useMean, hasBias),
            partialStats,
            input,
            scale,
            bias,
            output);
    }

    public static Expr QKVRoPEWithCache(
        Expr q,
        Expr k,
        Expr v,
        Expr qScale,
        Expr kScale,
        Expr qBias,
        Expr kBias,
        Expr cos,
        Expr sin,
        Expr kvCaches,
        Dimension layerId,
        Expr qOutput,
        int qAxis,
        float qEpsilon,
        bool qUseMean,
        int kAxis,
        float kEpsilon,
        bool kUseMean,
        IRArray<IR.NN.AttentionDimKind> qkvLayout,
        IRArray<IR.NN.AttentionDimKind> attentionLayout)
    {
        return KernelCall(
            new QKVRoPEWithCache(
                qAxis,
                qEpsilon,
                qUseMean,
                kAxis,
                kEpsilon,
                kUseMean,
                qkvLayout,
                attentionLayout),
            q,
            k,
            v,
            qScale,
            kScale,
            qBias,
            kBias,
            cos,
            sin,
            kvCaches,
            layerId,
            qOutput);
    }

    public static Expr GatherReduceQKVRoPEWithCache(
        Expr q,
        Expr k,
        Expr v,
        Expr qScale,
        Expr kScale,
        Expr qBias,
        Expr kBias,
        Expr cos,
        Expr sin,
        Expr kvCaches,
        Dimension layerId,
        Expr qOutput,
        DistributedType qInType,
        DistributedType qLogicalType,
        DistributedType kInType,
        DistributedType kLogicalType,
        DistributedType vInType,
        DistributedType vLogicalType,
        IRArray<Dimension> qShape,
        IRArray<Dimension> qStrides,
        IRArray<Dimension> kShape,
        IRArray<Dimension> kStrides,
        IRArray<Dimension> vShape,
        IRArray<Dimension> vStrides,
        int qAxis,
        float qEpsilon,
        bool qUseMean,
        int kAxis,
        float kEpsilon,
        bool kUseMean,
        IRArray<IR.NN.AttentionDimKind> qkvLayout,
        IRArray<IR.NN.AttentionDimKind> attentionLayout)
    {
        return KernelCall(
            new GatherReduceQKVRoPEWithCache(
                qInType,
                qLogicalType,
                kInType,
                kLogicalType,
                vInType,
                vLogicalType,
                qShape,
                qStrides,
                kShape,
                kStrides,
                vShape,
                vStrides,
                qAxis,
                qEpsilon,
                qUseMean,
                kAxis,
                kEpsilon,
                kUseMean,
                qkvLayout,
                attentionLayout),
            q,
            k,
            v,
            qScale,
            kScale,
            qBias,
            kBias,
            cos,
            sin,
            kvCaches,
            layerId,
            qOutput);
    }

    public static Expr InstanceNorm(Expr input, Expr scale, Expr bias, Expr output, float epsilon, IRArray<int> vectorizedAxes, IRArray<Dimension> padedNums, DistributedType distributedType)
    {
        return KernelCall(new InstanceNorm(epsilon, vectorizedAxes, padedNums, distributedType), input, scale, bias, output);
    }

    public static Expr VectorizedBinary(Expr lhs, Expr rhs, Expr output, BaseExpr postOps, BinaryOp binaryOp, IRArray<int>? lhsVectorizedAxes = null, IRArray<Dimension>? lhsPadedNums = null, IRArray<int>? rhsVectorizedAxes = null, IRArray<Dimension>? rhsPadedNums = null)
    {
        return KernelCall(new VectorizedBinary(binaryOp, lhsVectorizedAxes ?? Array.Empty<int>(), lhsPadedNums ?? Array.Empty<Dimension>(), rhsVectorizedAxes ?? Array.Empty<int>(), rhsPadedNums ?? Array.Empty<Dimension>()), lhs, rhs, output, postOps);
    }

    public static Call ResizeImage(Expr input, Expr output, int[] vectorizedAxes, Dimension[] padedNums, int[] newSize, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode)
    {
        return KernelCall(new ResizeImage(vectorizedAxes, padedNums, newSize, resizeMode, transformationMode, nearestMode), input, output);
    }

    public static Expr Slice(Expr input, RankedShape begins, RankedShape ends, Expr ret, int[] axes, int[] strides)
    {
        return KernelCall(new Slice(axes, strides), input, begins, ends, ret);
    }

    public static Expr Concat(Expr[] inputs, Expr ret, int axis)
    {
        return KernelCall(new Concat(axis), inputs.Concat(new[] { ret }).ToArray());
    }

    public static Expr PagedAttention(Expr q, Expr kvcache, Expr extra, Expr scale, Dimension layerId, Expr ret, IRArray<IR.NN.AttentionDimKind> layout, int hiddenSize)
    {
        return KernelCall(new PagedAttention(layout, hiddenSize), q, kvcache, extra, scale, layerId, ret);
    }

    public static Call PagedAttentionUseSplitKV(Expr kvcache, long directContextThreshold)
        => new(new PagedAttentionUseSplitKV(directContextThreshold), kvcache);

    public static Expr PagedAttentionPartial(
        Expr q,
        Expr kvcache,
        Expr extra,
        Expr scale,
        Dimension layerId,
        Expr maxState,
        Expr sumState,
        Expr accState,
        Expr output,
        IRArray<IR.NN.AttentionDimKind> layout,
        int hiddenSize,
        int splitHierarchyAxis,
        int splitCount,
        long directContextThreshold)
    {
        return KernelCall(
            new PagedAttentionPartial(
                layout,
                hiddenSize,
                splitHierarchyAxis,
                splitCount,
                directContextThreshold),
            q,
            kvcache,
            extra,
            scale,
            layerId,
            maxState,
            sumState,
            accState,
            output);
    }

    public static Expr PagedAttentionMerge(
        Expr maxState,
        Expr sumState,
        Expr accState,
        Expr output,
        IRArray<IR.NN.AttentionDimKind> layout,
        int hiddenSize,
        int splitHierarchyAxis,
        int splitCount)
    {
        return KernelCall(
            new PagedAttentionMerge(layout, hiddenSize, splitHierarchyAxis, splitCount),
            maxState,
            sumState,
            accState,
            output);
    }

    public static Call PagedAttentionMergePackedMatMul(
        Expr maxState,
        Expr sumState,
        Expr accState,
        Expr mergeOutputLayout,
        Expr mergedLhsLayout,
        Expr rhs,
        Expr output,
        Expr loadC,
        Expr scale,
        Expr addend,
        IRArray<IR.NN.AttentionDimKind> layout,
        int hiddenSize,
        int splitHierarchyAxis,
        int splitCount,
        IR.NTT.PackedMatMulRhsLayout rhsLayout)
    {
        return KernelCall(
            new PagedAttentionMergePackedMatMul(
                layout,
                hiddenSize,
                splitHierarchyAxis,
                splitCount,
                rhsLayout),
            maxState,
            sumState,
            accState,
            mergeOutputLayout,
            mergedLhsLayout,
            rhs,
            output,
            loadC,
            scale,
            addend);
    }

    public static Expr UpdatePagedAttentionKVCache(Expr value, Expr kvcache, Dimension layerId, IR.NN.AttentionCacheKind kind, IRArray<IR.NN.AttentionDimKind> layout)
    {
        return KernelCall(new UpdatePagedAttentionKVCache(kind, layout), value, kvcache, layerId);
    }

    public static Expr GatherPagedAttentionKVCache(Expr value, Expr kvcache, Expr output)
    {
        return KernelCall(new GatherPagedAttentionKVCache(), value, kvcache, output);
    }

    public static Expr CreatePagedAttentionKVCache(IR.NN.PagedAttentionConfig config, Expr numSeqs, Expr numTokens, Expr contextLens, Expr seqLens, Expr blockTable, Expr slotMapping, Expr numBlocks, Expr kvCaches, Expr output)
    {
        return KernelCall(new CreatePagedAttentionKVCache(config), numSeqs, numTokens, contextLens, seqLens, blockTable, slotMapping, numBlocks, kvCaches, output);
    }

    public static Expr IdentityPagedAttentionKVCache(Expr input, Expr numSeqs, Expr numTokens, Expr contextLens, Expr seqLens, Expr blockTable, Expr slotMapping, Expr numBlocks, Expr kvCaches)
    {
        return KernelCall(new IdentityPagedAttentionKVCache(), input, numSeqs, numTokens, contextLens, seqLens, blockTable, slotMapping, numBlocks, kvCaches);
    }

    public static Expr Swish(Expr buffer, Expr ret, float v)
    {
        return KernelCall(new Swish(v), buffer, ret);
    }

    public static Expr Gather(Expr input, Expr indcies, Expr ret, int axis)
    {
        return KernelCall(new Gather(axis), input, indcies, ret);
    }

    public static Expr GetItem(Expr input, BaseExpr index, Expr ret)
    {
        return KernelCall(new GetItem(), input, index, ret);
    }

    public static Expr Transpose(Expr buffer, Expr ret, int[] perm)
    {
        return KernelCall(new Transpose(perm), buffer, ret);
    }

    public static Expr Pad(Expr input, Expr ret, Paddings pads, float padValue, IRArray<int> actualPadAxes)
    {
        return KernelCall(new Pad(padValue, actualPadAxes), input, pads, ret);
    }

    public static Expr Im2col(Expr input, Expr output, IRArray<long> kernel, IRArray<int> stride, IRArray<int> padding, IRArray<int> vectorizedAxes, IRArray<int> padedNums)
    {
        return KernelCall(new Im2col(kernel, stride, padding, vectorizedAxes, padedNums), input, output);
    }

    public static Expr Reduce(Expr input, Expr ret, Expr loadPrevious, int[] vectorizedAxes, Dimension[] padedNums, IRArray<int> axis, bool keepDims, ReduceOp reduceOp)
    {
        return KernelCall(new TIR.NTT.Reduce(vectorizedAxes, padedNums, axis, keepDims, reduceOp), input, ret, loadPrevious);
    }

    public static Expr ReduceArg(Expr input, Expr ret, int axis, bool keepDims, bool selectLastIndex, ReduceArgOp reduceArgOp, DataType destType)
    {
        return KernelCall(new TIR.NTT.ReduceArg(axis, keepDims, selectLastIndex, reduceArgOp, destType), input, ret);
    }

    public static Call RoPE(Expr input, Expr cos, Expr sin, Expr output)
    {
        return KernelCall(new TIR.NTT.RoPE(), input, cos, sin, output);
    }

    public static Call GatherReduceScatter(Expr input, Expr output, DistributedType inType, DistributedType outType)
    {
        return KernelCall(new TIR.NTT.GatherReduceScatter(inType, outType), input, output);
    }

    public static Call GatherReduceAddNormStats(
        Expr input,
        Expr collective,
        Expr addend,
        Expr valueOutput,
        Expr statsOutput,
        DistributedType inType,
        DistributedType outType,
        int axis,
        bool useMean)
    {
        return KernelCall(
            new TIR.NTT.GatherReduceAddNormStats(inType, outType, axis, useMean),
            input,
            collective,
            addend,
            valueOutput,
            statsOutput);
    }

    public static Call GatherReduceAddNormApply(
        Expr input,
        Expr collective,
        Expr addend,
        Expr valueOutput,
        Expr statsWorkspace,
        Expr scale,
        Expr bias,
        Expr normOutput,
        DistributedType inType,
        DistributedType outType,
        int axis,
        float epsilon,
        bool useMean)
    {
        return KernelCall(
            new TIR.NTT.GatherReduceAddNormApply(inType, outType, axis, epsilon, useMean),
            input,
            collective,
            addend,
            valueOutput,
            statsWorkspace,
            scale,
            bias,
            normOutput);
    }

    public static Call Clamp(Expr input, Expr output, float min, float max)
    {
        return KernelCall(new TIR.NTT.Clamp(min, max), input, output);
    }

    public static Call Cast(Expr input, Expr output, DataType newType, CastMode castMode, IRArray<int> vectorizeAxes = default, Expr? postOps = null)
    {
        return KernelCall(new TIR.NTT.Cast(newType, castMode, vectorizeAxes.IsDefaultOrEmpty ? Array.Empty<int>() : vectorizeAxes), input, output, postOps ?? None.Default);
    }

    public static Call SynchronizeThreads()
    {
        return KernelCall(new TIR.NTT.SynchronizeThreads());
    }

    public static Call Barrier(BarrierScope scope, IRArray<int> axisGroupAxes = default)
    {
        var normalizedAxes = axisGroupAxes.IsDefaultOrEmpty
            ? new IRArray<int>()
            : new IRArray<int>(axisGroupAxes.Distinct().Order().ToArray());
        if (scope == BarrierScope.Block && normalizedAxes.Count != 0)
        {
            throw new ArgumentException(
                "Block barriers cannot carry chip axis-group axes.",
                nameof(axisGroupAxes));
        }

        return KernelCall(new TIR.NTT.Barrier(scope, normalizedAxes));
    }

    public static Call Where(Expr cond, Expr x, Expr y, Expr output)
    {
        return KernelCall(new TIR.NTT.Where(), cond, x, y, output);
    }

    public static Call Expand(Expr input, Expr output)
    {
        return KernelCall(new TIR.NTT.Expand(), input, output);
    }

    public static Call Erf(Expr input, Expr output)
    {
        return KernelCall(new TIR.NTT.Erf(), input, output);
    }

    public static Call Compare(CompareOp compareOp, Expr lhs, Expr rhs, Expr output)
    {
        return KernelCall(new TIR.NTT.Compare(compareOp), lhs, rhs, output);
    }

    public static Call ScatterND(Expr input, Expr indices, Expr updates, Expr output)
    {
        return KernelCall(new TIR.NTT.ScatterND(), input, indices, updates, output);
    }

    public static Expr Stack(Expr[] inputs, Expr ret, int axis)
    {
        return KernelCall(new Stack(axis), inputs.Concat(new[] { ret }).ToArray());
    }

    public static Expr ShapeOf(Expr inputs, Expr ret)
    {
        return KernelCall(new TIR.NTT.ShapeOf(), inputs, ret);
    }

    public static Expr ConstantOfShape(Shape shape, Expr value, Expr ret)
    {
        return KernelCall(new TIR.NTT.ConstantOfShape(), shape, value, ret);
    }

    public static Expr Range(Expr begin, Expr end, Expr step, Expr ret)
    {
        return KernelCall(new TIR.NTT.Range(), begin, end, step, ret);
    }

    public static Expr GetPositionIds(Expr kvCache, Expr ret, DistributedType distributedType)
    {
        return KernelCall(new TIR.NTT.GetPositionIds(distributedType), kvCache, ret);
    }

    public static Expr Qwen3MoE(Expr hiddenStates, Expr moeGateW, Expr moeExpertGateInputScale, Expr moeExpertGateProjW, Expr moeExpertGateProjScale, Expr moeExpertDownInputScale, Expr moeExpertDownProjW, Expr moeExpertDownProjScale, Expr moeExpertUpInputScale, Expr moeExpertUpProjW, Expr moeExpertUpProjScale, Expr ret, long layerId, long hiddenSize, long intermediateSize, long moeIntermediateSize, long numExpert, long numTopK, long isNormTopkProb)
    {
        return KernelCall(new TIR.NTT.Qwen3MoE(layerId, hiddenSize, intermediateSize, moeIntermediateSize, numExpert, numTopK, isNormTopkProb), hiddenStates, moeGateW, moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale, moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale, moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale, ret);
    }

    public static Expr SparseExperts(Expr q, Expr routerIdx, Expr routerWeights, Expr moeExpertGateInputScale, Expr moeExpertGateProjW, Expr moeExpertGateProjScale, Expr moeExpertDownInputScale, Expr moeExpertDownProjW, Expr moeExpertDownProjScale, Expr moeExpertUpInputScale, Expr moeExpertUpProjW, Expr moeExpertUpProjScale, Expr ret, long hiddenSize, long moeIntermediateSize, long numExpert, long numTopK, long chunkSize)
    {
        return KernelCall(new TIR.NTT.SparseExperts(Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>(), Array.Empty<int>(), Array.Empty<SBP>(), Array.Empty<SBP>(), Array.Empty<SBP>(), Array.Empty<SBP>(), hiddenSize, moeIntermediateSize, numExpert, numTopK, chunkSize, null, string.Empty, string.Empty), q, routerIdx, routerWeights, moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale, moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale, moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale, None.Default, ret);
    }

    public static Expr SparseExpertsGateUp(Expr q, Expr routerExpertIds, Expr moeExpertGateInputScale, Expr moeExpertGateProjW, Expr moeExpertGateProjScale, Expr moeExpertUpInputScale, Expr moeExpertUpProjW, Expr moeExpertUpProjScale, Expr output, long hiddenSize, long moeIntermediateSize, long numExpert, long numTopK, long chunkSize)
    {
        return KernelCall(new TIR.NTT.SparseExpertsGateUp(hiddenSize, moeIntermediateSize, numExpert, numTopK, chunkSize), q, routerExpertIds, moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale, moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale, output);
    }

    public static Expr SparseExpertsDown(Expr activations, Expr routerExpertIds, Expr routerExpertWeights, Expr moeExpertDownInputScale, Expr moeExpertDownProjW, Expr moeExpertDownProjScale, Expr output, long hiddenSize, long moeIntermediateSize, long numExpert, long numTopK, long chunkSize)
    {
        return KernelCall(new TIR.NTT.SparseExpertsDown(hiddenSize, moeIntermediateSize, numExpert, numTopK, chunkSize), activations, routerExpertIds, routerExpertWeights, moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale, output);
    }

    public static Expr SparseExperts(Expr q, Expr routerIdx, Expr routerWeights, Expr moeExpertGateInputScale, Expr moeExpertGateProjW, Expr moeExpertGateProjScale, Expr moeExpertDownInputScale, Expr moeExpertDownProjW, Expr moeExpertDownProjScale, Expr moeExpertUpInputScale, Expr moeExpertUpProjW, Expr moeExpertUpProjScale, Expr extra, Expr ret, IRArray<int> qVectorizedAxes, IRArray<int> gateVectorizedAxes, IRArray<int> downVectorizedAxes, IRArray<int> upVectorizedAxes, IRArray<SBP> qSBPs, IRArray<SBP> gateSBPs, IRArray<SBP> downSBPs, IRArray<SBP> upSBPs, long hiddenSize, long moeIntermediateSize, long numExpert, long numTopK, long chunkSize, Cost costmodel, string cSourcePath = "", string funcName = "")
    {
        return KernelCall(new TIR.NTT.SparseExperts(qVectorizedAxes, gateVectorizedAxes, downVectorizedAxes, upVectorizedAxes, qSBPs, gateSBPs, downSBPs, upSBPs, hiddenSize, moeIntermediateSize, numExpert, numTopK, chunkSize, costmodel, cSourcePath, funcName), q, routerIdx, routerWeights, moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale, moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale, moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale, extra, ret);
    }

    public static Expr TopK(Expr x, Expr k, Expr output, long axis, long largest, long sorted)
    {
        return KernelCall(new TIR.NTT.TopK(axis, largest, sorted), x, k, output);
    }

    public static Expr SamplingPartial(
        Expr logits,
        Expr state,
        Expr processedLogits,
        Expr argMaxState,
        IR.NN.SamplerConfig config)
    {
        return KernelCall(
            new TIR.NTT.SamplingPartial(config),
            logits,
            state,
            processedLogits,
            argMaxState);
    }

    public static Expr SamplingCombine(
        Expr logits,
        Expr processedLogits,
        Expr argMaxState,
        Expr state,
        Expr summary,
        Expr sampledIds,
        Expr logprobIds,
        Expr logprobs,
        Expr ranks,
        Expr counts,
        IR.NN.SamplerConfig config,
        int blockCount,
        int radixBits = 8)
    {
        return KernelCall(
            new TIR.NTT.SamplingCombine(config, blockCount, radixBits),
            logits,
            processedLogits,
            argMaxState,
            state,
            summary,
            sampledIds,
            logprobIds,
            logprobs,
            ranks,
            counts);
    }
}
