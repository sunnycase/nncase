// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;

namespace Nncase.IR.F;

/// <summary>
/// NN functional helper.
/// </summary>
public static class NN
{
    public static Call GatedDeltaNet(
        Expr input,
        Expr state,
        Expr qkvWeight,
        Expr zWeight,
        Expr bWeight,
        Expr aWeight,
        Expr convWeight,
        Expr aLog,
        Expr dtBias,
        Expr normWeight,
        Expr outputWeight,
        Dimension layerId,
        long numKeyHeads,
        long numValueHeads,
        long keyHeadDim,
        long valueHeadDim,
        long convKernelSize,
        float epsilon,
        Expr? qkvWeightScale = null,
        Expr? zWeightScale = null,
        Expr? outputWeightScale = null,
        global::Nncase.IR.Math.MatMulQuantizationMode quantizationMode =
            global::Nncase.IR.Math.MatMulQuantizationMode.None,
        long weightBlockN = 0,
        long weightBlockK = 0) =>
        new(
            new GatedDeltaNet(
                numKeyHeads,
                numValueHeads,
                keyHeadDim,
                valueHeadDim,
                convKernelSize,
                epsilon,
                quantizationMode,
                weightBlockN,
                weightBlockK),
            input,
            state,
            qkvWeight,
            qkvWeightScale ?? None.Default,
            zWeight,
            zWeightScale ?? None.Default,
            bWeight,
            aWeight,
            convWeight,
            aLog,
            dtBias,
            normWeight,
            outputWeight,
            outputWeightScale ?? None.Default,
            layerId);

    public static Call GatedDeltaNetConvolution(
        Expr qkv,
        Expr state,
        Expr convWeight,
        Dimension layerId,
        long convKernelSize) =>
        new(
            new GatedDeltaNetConvolution(convKernelSize),
            qkv,
            state,
            convWeight,
            layerId);

    public static Call GatedDeltaNetRecurrentCore(
        Expr state,
        Expr qkv,
        Expr z,
        Expr projectionInput,
        Expr bWeight,
        Expr aWeight,
        Expr aLog,
        Expr dtBias,
        Expr normWeight,
        Dimension layerId,
        long numKeyHeads,
        long numValueHeads,
        long keyHeadDim,
        long valueHeadDim,
        float epsilon) =>
        new(
            new GatedDeltaNetRecurrentCore(
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
            layerId);

    public static Call Sampling(Expr logits, Expr state, SamplerConfig config)
        => new(new Sampling(config), logits, state);

    public static Call Conv2D(Expr input, Expr weights, Expr bias, Shape stride, Paddings padding, Shape dilation, PadMode padMode, Dimension groups) => new Call(new Conv2D(padMode), input, weights, bias, stride, padding, dilation, groups, (Expr)new[] { ValueRange<float>.Full.Min, ValueRange<float>.Full.Max });

    public static Call Conv2D(Expr input, Expr weights, Expr bias, Shape stride, Paddings padding, Shape dilation, PadMode padMode, Dimension groups, Expr fusedClamp) => new Call(new Conv2D(padMode), input, weights, bias, stride, padding, dilation, groups, fusedClamp);

    public static Call Celu(Expr input, Expr alpha) => new Call(new Celu(), input, alpha);

    public static Call Conv2DTranspose(Expr input, Expr weights, Expr bias, Shape outShape, Shape stride, Paddings padding, Shape outputPadding, Shape dilation, PadMode padMode, Dimension groups) => new Call(new Conv2DTranspose(padMode), input, weights, bias, outShape, stride, padding, outputPadding, dilation, groups, (Expr)new[] { ValueRange<float>.Full.Min, ValueRange<float>.Full.Max });

    public static Call Elu(Expr input, Expr alpha) => new Call(new Elu(), input, alpha);

    public static Call Hardmax(Expr input, Expr axis) => new Call(new Hardmax(), input, axis);

    public static Call LeakyRelu(Expr input, Expr alpha) => new Call(new LeakyRelu(), input, alpha);

    public static Call L2Normalization(Expr input) => new Call(new L2Normalization(), input);

    public static Call BatchNormalization(Expr input, Expr scale, Expr bias, Expr input_mean, Expr input_var, Expr epsilon, Expr momentum) => new Call(new BatchNormalization(), input, scale, bias, input_mean, input_var, epsilon, momentum);

    public static Call LayerNorm(int axis, float epsilon, Expr input, Expr scale, Expr bias, bool hasMean = true, bool channelFirst = false) => new Call(new LayerNorm(axis, epsilon, hasMean, channelFirst), input, scale, bias);

    public static Call NormStats(int axis, Expr input, bool useMean) => new Call(new NormStats(axis, useMean), input);

    public static Call BindNormStats(int axis, Expr input, Expr stats, bool useMean) => new Call(new BindNormStats(axis, useMean), input, stats);

    public static Call NormApply(int axis, float epsilon, Expr input, Expr stats, Expr scale, Expr bias, bool useMean) => new Call(new NormApply(axis, epsilon, useMean), input, stats, scale, bias);

    public static Call QKVRoPEWithCache(
        BaseExpr qkv,
        Expr qScale,
        Expr kScale,
        Expr qBias,
        Expr kBias,
        Expr cos,
        Expr sin,
        Expr kvCaches,
        Dimension layerId,
        int qAxis,
        float qEpsilon,
        bool qUseMean,
        int kAxis,
        float kEpsilon,
        bool kUseMean,
        IRArray<AttentionDimKind> qkvLayout,
        IRArray<AttentionDimKind> attentionLayout) =>
        new Call(
            new QKVRoPEWithCache(
                qAxis,
                qEpsilon,
                qUseMean,
                kAxis,
                kEpsilon,
                kUseMean,
                qkvLayout,
                attentionLayout),
            qkv,
            qScale,
            kScale,
            qBias,
            kBias,
            cos,
            sin,
            kvCaches,
            layerId);

    public static Call BatchToSpace(Expr input, Shape blockShape, Paddings crops) => new Call(new BatchToSpace(), input, blockShape, crops);

    public static Call InstanceNormalization(Expr input, Expr scale, Expr bias, Expr eps) => new Call(new InstanceNormalization(), input, scale, bias, eps);

    public static Call LpNormalization(Expr input, Expr axis, Expr p) => new Call(new LpNormalization(), input, axis, p);

    public static Call LRN(Expr input, Expr alpha, Expr beta, Expr bias, Expr size) => new Call(new LRN(), input, alpha, beta, bias, size);

    public static Call Mish(Expr input) => input * Math.Tanh(Softplus(input));

    public static Call HardSigmoid(Expr input, Expr alpha, Expr beta) => new Call(new HardSigmoid(), input, alpha, beta);

    public static Call HardSwish(Expr input) => new Call(new HardSwish(), input);

    public static Call OneHot(OneHotMode oneHotMode, Expr indices, Expr depth, Expr values, Expr axis) => new Call(new OneHot(oneHotMode), indices, depth, values, axis);

    /// <summary>
    /// Pads is Const tensor, shape = [channels, 2(before, after)].
    /// </summary>
    public static Call Pad(Expr input, Paddings pads, PadMode mode, Expr value) => new Call(new Pad(mode), input, pads, value);

    public static Call ReduceWindow2D(ReduceOp reduceOp, Expr input, Expr initValue, Shape filter, Shape stride, Paddings padding, Shape dilation, Expr ceilMode, Expr countIncludePad) =>
        new Call(new ReduceWindow2D(reduceOp), input, initValue, filter, stride, padding, dilation, ceilMode, countIncludePad);

    public static Call Relu(Expr input) => new Call(new Relu(), input);

    public static Call Relu6(Expr input) => new Call(new Relu6(), input);

    public static Call RoPE(Expr input, Expr cos, Expr sin) => new Call(new RoPE(), input, cos, sin);

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
        long numHeads,
        long numKvHeads,
        DataType outputDataType) =>
        new Call(
            new QKVParallelLinear(numHeads, numKvHeads, outputDataType),
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
        GluType gluType,
        DataType outputDataType,
        global::Nncase.IR.Math.MatMulQuantizationMode quantizationMode =
            global::Nncase.IR.Math.MatMulQuantizationMode.None,
        long weightBlockN = 0,
        long weightBlockK = 0) =>
        new Call(
            new MatMulGlu(
                gluType,
                outputDataType,
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

    public static Call PRelu(Expr input, Expr slope) => new Call(new PRelu(), input, slope);

    public static Call Selu(Expr input, Expr alpha, Expr gamma) => new Call(new Selu(), input, alpha, gamma);

    public static Call Sigmoid(Expr expr) => new Call(new Sigmoid(), expr);

    public static Call Softmax(Expr expr, Dimension axis) => new Call(new Softmax(), expr, axis);

    public static Call Softplus(Expr expr) => new Call(new Softplus(), expr);

    public static Call Softsign(Expr expr) => new Call(new Softsign(), expr);

    // same like tensorflow
    public static Call SpaceToBatch(Expr input, Shape blockShape, Paddings paddings) => new Call(new SpaceToBatch(), input, blockShape, paddings);

    public static Call LogSoftmax(Expr expr, Dimension axis) => new Call(new LogSoftmax(), expr, axis);

    // public static Call LSTM(Expr input,Expr w, Expr r, Expr b,
    //     Expr initH, Expr initC, Expr has_static, lstm_direction lstmDirection,string str) =>
    //     new Call(new IR.NN.LSTM(lstmDirection,str), input, w, r, b,  initH, initC, has_static);

    /// <summary>
    /// create custom call.
    /// </summary>
    public static Call CustomCall(CustomOp op, params Expr[] args) => new Call(op, args);

    /// <summary>
    /// create Erf call.
    /// </summary>
    public static Call Erf(Expr expr) => new Call(new Erf(), expr);

    /// <summary>
    /// create Gelu call.
    /// </summary>
    public static Call Gelu(Expr expr, Expr alpha) => new Call(new Gelu(), expr, alpha);

    /// <summary>
    /// create Swish call.
    /// </summary>
    public static Call Swish(Expr input) => new Call(new Swish(), input, (Expr)1f);

    /// <summary>
    /// create Swish call.
    /// </summary>
    public static Call Swish(Expr input, Expr beta) => new Call(new Swish(), input, beta);

    public static Call GetPositionIds(Dimension sequenceLength, Expr kvCache) => new Call(new GetPositionIds(new IRArray<SBP>(), new Placement(new IRArray<int>(), string.Empty, string.Empty)), sequenceLength, kvCache);

    public static Call GetPositionIds(Dimension sequenceLength, Expr kvCache, IRArray<SBP> ndsbp, Placement placement) => new Call(new GetPositionIds(ndsbp, placement), sequenceLength, kvCache);

    public static Expr UpdatePagedAttentionKVCache(Expr slots, Expr kvCaches, AttentionCacheKind cacheKind, int layerId, AttentionDimKind[] layout) => UpdatePagedAttentionKVCache(slots, kvCaches, new DimConst(layerId), cacheKind, layout);

    public static Expr UpdatePagedAttentionKVCache(Expr slots, Expr kvCaches, Dimension layerId, AttentionCacheKind cacheKind, AttentionDimKind[] layout) => new Call(new UpdatePagedAttentionKVCache(cacheKind, layout), slots, kvCaches, layerId);

    public static Expr GatherPagedAttentionKVCache(Expr shardId, Expr kvCaches, int numBlocks) => new Call(new GatherPagedAttentionKVCache(numBlocks), shardId, kvCaches);

    public static Expr CreatePagedAttentionKVCache(PagedAttentionConfig config, Expr numSeqs, Expr numTokens, Expr contextLens, Expr seqLens, Expr blockTable, Expr slotMapping, Expr numBlocks, Expr kvCaches) => new Call(new CreatePagedAttentionKVCache(config), numSeqs, numTokens, contextLens, seqLens, blockTable, slotMapping, numBlocks, kvCaches);

    public static Expr IdentityPagedAttentionKVCache(Expr input, Expr numSeqs, Expr numTokens, Expr contextLens, Expr seqLens, Expr blockTable, Expr slotMapping, Expr numBlocks, Expr kvCaches) => new Call(new IdentityPagedAttentionKVCache(), input, numSeqs, numTokens, contextLens, seqLens, blockTable, slotMapping, numBlocks, kvCaches);

    public static Expr PagedAttention(Expr q, Expr kvCaches, Expr extra, Expr scale, int layerId, AttentionDimKind[] qlayout, int hiddenSize) => PagedAttention(q, kvCaches, extra, scale, new DimConst(layerId), qlayout, hiddenSize);

    public static Expr PagedAttention(Expr q, Expr kvCaches, Expr extra, Expr scale, Dimension layerId, AttentionDimKind[] qlayout, int hiddenSize) => new Call(new PagedAttention(new IRArray<AttentionDimKind>(qlayout), hiddenSize), q, kvCaches, extra, scale, layerId);

    public static Expr Qwen3MoE(Expr q, Expr moeGateW, Expr moeExpertGateInputScale, Expr moeExpertGateProjW, Expr moeExpertGateProjScale, Expr moeExpertDownInputScale, Expr moeExpertDownProjW, Expr moeExpertDownProjScale, Expr moeExpertUpInputScale, Expr moeExpertUpProjW, Expr moeExpertUpProjScale, long layerId, long hiddenSize, long intermediateSize, long moeIntermediateSize, long numExpert, long numTopK, long isNormTopkProb) => new Call(new Qwen3MoE(layerId, hiddenSize, intermediateSize, moeIntermediateSize, numExpert, numTopK, isNormTopkProb), q, moeGateW, moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale, moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale, moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale);

    public static Call SparseExperts(Expr q, Expr routerExpertIds, Expr routerExpertWeights, Expr moeExpertGateInputScale, Expr moeExpertGateProjW, Expr moeExpertGateProjScale, Expr moeExpertDownInputScale, Expr moeExpertDownProjW, Expr moeExpertDownProjScale, Expr moeExpertUpInputScale, Expr moeExpertUpProjW, Expr moeExpertUpProjScale, long hiddenSize, long moeIntermediateSize, long numExpert, long numTopK, long chunkSize) => new Call(new SparseExperts(hiddenSize, moeIntermediateSize, numExpert, numTopK, chunkSize), q, routerExpertIds, routerExpertWeights, moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale, moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale, moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale);

    public static Call SparseExpertsGateUp(Expr q, Expr routerExpertIds, Expr moeExpertGateInputScale, Expr moeExpertGateProjW, Expr moeExpertGateProjScale, Expr moeExpertUpInputScale, Expr moeExpertUpProjW, Expr moeExpertUpProjScale, DataType outputDataType, long hiddenSize, long moeIntermediateSize, long numExpert, long numTopK, long chunkSize) => new Call(new SparseExpertsGateUp(outputDataType, hiddenSize, moeIntermediateSize, numExpert, numTopK, chunkSize), q, routerExpertIds, moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale, moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale);

    public static Call SparseExpertsDown(Expr activations, Expr routerExpertIds, Expr routerExpertWeights, Expr moeExpertDownInputScale, Expr moeExpertDownProjW, Expr moeExpertDownProjScale, DataType outputDataType, long hiddenSize, long moeIntermediateSize, long numExpert, long numTopK, long chunkSize) => new Call(new SparseExpertsDown(outputDataType, hiddenSize, moeIntermediateSize, numExpert, numTopK, chunkSize), activations, routerExpertIds, routerExpertWeights, moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale);
}
