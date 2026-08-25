// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;

namespace Nncase.IR.F;

public partial class NTT
{
    public static Call Load(Expr input)
    {
        return new Call(new Load(), input);
    }

    public static Call Store(Expr input)
    {
        return new Call(new Store(), input);
    }

    public static Expr VectorizedSoftmax(Expr input, int axis, IRArray<int> vectorizedAxes)
    {
        return new Call(new VectorizedSoftmax(axis, vectorizedAxes), input);
    }

    public static Expr VectorizedLayerNorm(Expr input, Expr scale, Expr bias, int axis, float epsilon, bool usemean, IRArray<int> vectorizedAxes, BaseExpr padedNums)
    {
        return new Call(new VectorizedLayerNorm(axis, epsilon, usemean, vectorizedAxes), input, scale, bias, padedNums);
    }

    public static Call VectorizedReduce(Expr input, ReduceOp reduceOp, IRArray<int> axes, float initValue, bool keepDims, IRArray<int> vectorizedAxes, BaseExpr padedNums)
    {
        return new Call(new VectorizedReduce(reduceOp, axes, initValue, keepDims, vectorizedAxes), input, padedNums);
    }

    public static Expr VectorizedRoPE(Expr input, Expr cos, Expr sin)
    {
        return new Call(new VectorizedRoPE(), input, cos, sin);
    }

    public static Expr InstacneNorm(Expr input, Expr scale, Expr bias, float epsilon, IRArray<int> vectorizedAxes, BaseExpr padedNums)
    {
        return new Call(new InstacneNorm(epsilon, vectorizedAxes), input, scale, bias, padedNums);
    }

    public static Expr PackedMatMul(
        Expr lhs,
        Expr rhs,
        bool fusedReduce = false,
        DataType? outDataType = null,
        Expr? scale = null,
        PackedMatMulRhsLayout rhsLayout = PackedMatMulRhsLayout.NMajor,
        Expr? addend = null)
    {
        return new Call(
            new PackedMatMul(outDataType ?? DataTypes.Float32, fusedReduce, rhsLayout),
            lhs,
            rhs,
            scale ?? None.Default,
            addend ?? None.Default);
    }

    public static Expr PackedScaledMatMul(
        Expr lhs,
        Expr rhs,
        Expr lhsScale,
        Expr rhsScale,
        DataType outputDataType,
        PackedMatMulRhsLayout rhsLayout = PackedMatMulRhsLayout.KMajor)
        => new Call(
            new PackedScaledMatMul(outputDataType, rhsLayout),
            lhs,
            rhs,
            lhsScale,
            rhsScale);

    public static Expr PackedBlockScaledMatMul(
        Expr lhs,
        Expr rhs,
        Expr rhsScale,
        DataType outputDataType,
        long weightBlockN,
        long weightBlockK,
        PackedMatMulRhsLayout rhsLayout = PackedMatMulRhsLayout.KMajor,
        int outputNVectorLaneCount = 1,
        Expr? addend = null)
        => new Call(
            new PackedBlockScaledMatMul(
                outputDataType,
                rhsLayout,
                outputNVectorLaneCount,
                weightBlockN,
                weightBlockK),
            lhs,
            rhs,
            rhsScale,
            addend ?? None.Default);

    public static Expr PackedMatMulNormStats(
        Expr lhs,
        Expr rhs,
        DataType outputDataType,
        PackedMatMulRhsLayout rhsLayout,
        int axis,
        bool useMean,
        Expr? scale = null,
        Expr? addend = null)
    {
        return new Call(
            new PackedMatMulNormStats(outputDataType, rhsLayout, axis, useMean),
            lhs,
            rhs,
            scale ?? None.Default,
            addend ?? None.Default);
    }

    public static Expr PackedBlockScaledMatMulNormStats(
        Expr lhs,
        Expr rhs,
        Expr rhsScale,
        DataType outputDataType,
        long weightBlockN,
        long weightBlockK,
        PackedMatMulRhsLayout rhsLayout,
        int outputNVectorLaneCount,
        int axis,
        bool useMean,
        Expr? addend = null)
    {
        return new Call(
            new PackedBlockScaledMatMulNormStats(
                outputDataType,
                rhsLayout,
                outputNVectorLaneCount,
                weightBlockN,
                weightBlockK,
                axis,
                useMean),
            lhs,
            rhs,
            rhsScale,
            addend ?? None.Default);
    }

    public static Expr PackedMatMulSamplingPartial(
        Expr lhs,
        Expr rhs,
        Expr state,
        DataType outputDataType,
        PackedMatMulRhsLayout rhsLayout,
        IR.NN.SamplerConfig config,
        Expr? scale = null,
        Expr? addend = null)
    {
        return new Call(
            new PackedMatMulSamplingPartial(outputDataType, rhsLayout, config),
            lhs,
            rhs,
            state,
            scale ?? None.Default,
            addend ?? None.Default);
    }

    public static Expr PagedAttentionPartial(
        Expr q,
        Expr kvCaches,
        Expr extra,
        Expr scale,
        Dimension layerId,
        IRArray<IR.NN.AttentionDimKind> layout,
        int hiddenSize,
        int splitHierarchyAxis,
        int splitCount)
    {
        return new Call(
            new PagedAttentionPartial(
                layout,
                hiddenSize,
                splitHierarchyAxis,
                splitCount),
            q,
            kvCaches,
            extra,
            scale,
            layerId);
    }

    public static Expr PagedAttentionCombine(
        Expr maxState,
        Expr sumState,
        Expr accState,
        IRArray<IR.NN.AttentionDimKind> layout,
        int hiddenSize,
        DataType outputDataType,
        IRType outputType,
        int splitHierarchyAxis,
        int splitCount)
    {
        return new Call(
            new PagedAttentionCombine(
                layout,
                hiddenSize,
                outputDataType,
                outputType,
                splitHierarchyAxis,
                splitCount),
            maxState,
            sumState,
            accState);
    }

    public static Expr SamplingPartial(Expr logits, Expr state, IR.NN.SamplerConfig config)
        => new Call(new SamplingPartial(config), logits, state);

    public static Expr SamplingCombine(
        Expr logits,
        Expr processedLogits,
        Expr argMaxState,
        Expr state,
        IR.NN.SamplerConfig config)
        => new Call(
            new SamplingCombine(config),
            logits,
            processedLogits,
            argMaxState,
            state);

    public static Expr PackedQKVParallelLinear(
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
        long numHeads,
        long numKvHeads,
        DataType? outDataType = null,
        PackedMatMulRhsLayout rhsLayout = PackedMatMulRhsLayout.NMajor)
    {
        return new Call(
            new PackedQKVParallelLinear(numHeads, numKvHeads, outDataType ?? DataTypes.Float32, rhsLayout),
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
            vWeightScale);
    }

    public static Expr PackedMatMulGlu(
        Expr input,
        Expr gateWeight,
        Expr upWeight,
        Expr gateBias,
        Expr upBias,
        Expr gateInputScale,
        Expr upInputScale,
        Expr gateWeightScale,
        Expr upWeightScale,
        IR.NN.GluType gluType,
        DataType? outDataType = null,
        PackedMatMulRhsLayout rhsLayout = PackedMatMulRhsLayout.NMajor,
        global::Nncase.IR.Math.MatMulQuantizationMode quantizationMode =
            global::Nncase.IR.Math.MatMulQuantizationMode.None,
        long weightBlockN = 0,
        long weightBlockK = 0)
    {
        return new Call(
            new PackedMatMulGlu(
                gluType,
                outDataType ?? DataTypes.Float32,
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
            upWeightScale);
    }

    public static Expr VectorizedMatMul(Expr lhs, Expr rhs, IRArray<int> lhsVectorizedAxes, IRArray<int> rhsVectorizedAxes, bool transA = false, bool transB = false, bool fusedReduce = false, DataType? outDataType = null, Expr? scale = null)
    {
        return new Call(new VectorizedMatMul(outDataType ?? DataTypes.Float32, lhsVectorizedAxes, rhsVectorizedAxes, transA, transB, fusedReduce), lhs, rhs, scale ?? None.Default);
    }

    public static Expr VectorizedBinary(Expr lhs, Expr rhs, BaseExpr postOps, BinaryOp binaryOp, IRArray<int>? lhsVectorizedAxes = null, IRArray<Dimension>? lhsPadedNums = null, IRArray<int>? rhsVectorizedAxes = null, IRArray<Dimension>? rhsPadedNums = null)
    {
        return new Call(new VectorizedBinary(binaryOp, lhsVectorizedAxes ?? Array.Empty<int>(), lhsPadedNums ?? Array.Empty<Dimension>(), rhsVectorizedAxes ?? Array.Empty<int>(), rhsPadedNums ?? Array.Empty<Dimension>()), lhs, rhs, postOps);
    }

    public static Call VectorizedCast(Expr input, DataType newType, CastMode castMode, IRArray<int> vectorizeAxes, Expr postOps)
    {
        return new Call(new VectorizedCast(newType, castMode, vectorizeAxes), input, postOps);
    }

    public static Call ResizeImage(Expr input, BaseExpr paddedNums, int[] vectorizedAxes, int[] newSize, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode)
    {
        return new Call(new ResizeImage(vectorizedAxes, newSize, resizeMode, transformationMode, nearestMode), input, paddedNums);
    }

    public static Expr Im2col(Expr input, long[] kernel, int[] stride, int[] padding)
    {
        return new Call(new Im2col(kernel, stride, padding, Array.Empty<int>(), Array.Empty<int>()), input);
    }

    public static Expr Im2col(Expr input, long[] kernel, int[] stride, int[] padding, int[] vectorizedAxes, int[] padedNums)
    {
        return new Call(new Im2col(kernel, stride, padding, vectorizedAxes, padedNums), input);
    }
}
