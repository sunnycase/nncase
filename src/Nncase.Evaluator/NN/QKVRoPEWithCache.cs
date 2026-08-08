// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="QKVRoPEWithCache"/>.
/// </summary>
public sealed class QKVRoPEWithCacheEvaluator :
    IEvaluator<QKVRoPEWithCache>,
    ITypeInferencer<QKVRoPEWithCache>,
    ICostEvaluator<QKVRoPEWithCache>
{
    public static IRType InferType(
        QKVRoPEWithCache target,
        IRType qkv,
        IRType qStats,
        IRType kStats,
        IRType qScale,
        IRType kScale,
        IRType qBias,
        IRType kBias,
        IRType cos,
        IRType sin,
        IRType kvCaches)
    {
        if (qkv is not TupleType { Count: 3 } tuple)
        {
            return new InvalidType($"QKVRoPEWithCache QKV must be a tuple of three tensors, got {qkv}.");
        }

        var qNorm = NormApplyEvaluator.InferType(
            new NormApply(target.QAxis, target.QEpsilon, target.QUseMean),
            tuple[0],
            qStats,
            qScale,
            qBias);
        if (qNorm is InvalidType)
        {
            return qNorm;
        }

        var kNorm = NormApplyEvaluator.InferType(
            new NormApply(target.KAxis, target.KEpsilon, target.KUseMean),
            tuple[1],
            kStats,
            kScale,
            kBias);
        if (kNorm is InvalidType)
        {
            return kNorm;
        }

        var qOutput = RoPEEvaluator.InferType(qNorm, cos, sin);
        if (qOutput is InvalidType)
        {
            return qOutput;
        }

        var kOutput = RoPEEvaluator.InferType(kNorm, cos, sin);
        if (kOutput is InvalidType)
        {
            return kOutput;
        }

        var afterKey = UpdatePagedAttentionKVCacheEvaluator.InferType(
            new UpdatePagedAttentionKVCache(AttentionCacheKind.Key, target.Layout),
            kOutput,
            kvCaches);
        if (afterKey is InvalidType)
        {
            return afterKey;
        }

        var afterValue = UpdatePagedAttentionKVCacheEvaluator.InferType(
            new UpdatePagedAttentionKVCache(AttentionCacheKind.Value, target.Layout),
            tuple[2],
            afterKey);
        return afterValue is InvalidType
            ? afterValue
            : new TupleType([qOutput, afterValue]);
    }

    public IValue Visit(IEvaluateContext context, QKVRoPEWithCache target)
    {
        var qkv = context.GetArgumentValueAsTensors(target, QKVRoPEWithCache.QKV);
        if (qkv.Length != 3)
        {
            throw new InvalidOperationException($"QKVRoPEWithCache expects three QKV tensors, got {qkv.Length}.");
        }

        var qStats = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.QStats);
        var kStats = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.KStats);
        var qScale = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.QScale);
        var kScale = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.KScale);
        var qBias = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.QBias);
        var kBias = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.KBias);
        var cos = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.Cos);
        var sin = context.GetArgumentValueAsTensor(target, QKVRoPEWithCache.Sin);
        var kvCaches = context.GetArgumentValue(target, QKVRoPEWithCache.KVCaches);
        var layerId = checked((int)context.GetArgumentValue(target, QKVRoPEWithCache.LayerId).AsTensor().ToScalar<long>());
        var qkvType = context.CurrentCall.Arguments[QKVRoPEWithCache.QKV.Index].CheckedType as TupleType
            ?? throw new InvalidOperationException("QKVRoPEWithCache QKV argument must have a tuple type.");

        var q = NormApplyEvaluator.Evaluate(
            qkv[0],
            qStats,
            qScale,
            qBias,
            GetTensorType(qkvType[0]).DType,
            target.QAxis,
            target.QEpsilon,
            target.QUseMean,
            GetNormalizationSize(qkvType[0], qkv[0], target.QAxis));
        q = RoPEEvaluator.Evaluate(q, cos, sin);

        var k = NormApplyEvaluator.Evaluate(
            qkv[1],
            kStats,
            kScale,
            kBias,
            GetTensorType(qkvType[1]).DType,
            target.KAxis,
            target.KEpsilon,
            target.KUseMean,
            GetNormalizationSize(qkvType[1], qkv[1], target.KAxis));
        k = RoPEEvaluator.Evaluate(k, cos, sin);

        var cacheTensor = kvCaches.AsTensor().Cast<Reference<IPagedAttentionKVCache>>();
        UpdatePagedAttentionKVCacheEvaluator.UpdateCache(k, cacheTensor, AttentionCacheKind.Key, layerId, target.Layout);
        UpdatePagedAttentionKVCacheEvaluator.UpdateCache(qkv[2], cacheTensor, AttentionCacheKind.Value, layerId, target.Layout);
        return new TupleValue([Value.FromTensor(q), kvCaches]);
    }

    public IRType Visit(ITypeInferenceContext context, QKVRoPEWithCache target)
    {
        var qkv = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.QKV);
        var qStats = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.QStats);
        var kStats = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.KStats);
        var qScale = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.QScale);
        var kScale = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.KScale);
        var qBias = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.QBias);
        var kBias = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.KBias);
        var cos = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.Cos);
        var sin = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.Sin);
        var kvCaches = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.KVCaches);
        _ = context.CheckArgumentType<DimensionType>(target, QKVRoPEWithCache.LayerId);
        return InferType(target, qkv, qStats, kStats, qScale, kScale, qBias, kBias, cos, sin, kvCaches);
    }

    public Cost Visit(ICostEvaluateContext context, QKVRoPEWithCache target)
    {
        var qkv = context.GetArgumentType<TupleType>(target, QKVRoPEWithCache.QKV);
        if (qkv.Count != 3)
        {
            return Cost.Zero;
        }

        var q = qkv[0];
        var k = qkv[1];
        var v = qkv[2];
        var qStats = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.QStats);
        var kStats = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.KStats);
        var qScale = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.QScale);
        var kScale = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.KScale);
        var qBias = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.QBias);
        var kBias = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.KBias);
        var cos = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.Cos);
        var sin = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.Sin);
        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                CostUtility.GetMemoryAccess(q) + CostUtility.GetMemoryAccess(k) + CostUtility.GetMemoryAccess(v) +
                CostUtility.GetMemoryAccess(qStats) + CostUtility.GetMemoryAccess(kStats) +
                CostUtility.GetMemoryAccess(qScale) + CostUtility.GetMemoryAccess(kScale) +
                CostUtility.GetMemoryAccess(qBias) + CostUtility.GetMemoryAccess(kBias) +
                (2 * (CostUtility.GetMemoryAccess(cos) + CostUtility.GetMemoryAccess(sin))),
            [CostFactorNames.BlockLocalMemoryStoreBytes] =
                CostUtility.GetMemoryAccess(q) + CostUtility.GetMemoryAccess(k) + CostUtility.GetMemoryAccess(v),
            [CostFactorNames.CPUCycles] =
                CostUtility.GetCPUCycles(q, target.QUseMean ? 11U : 9U) +
                CostUtility.GetCPUCycles(k, target.KUseMean ? 11U : 9U),
        };
    }

    private static TensorType GetTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => distributed.TensorType,
        _ => throw new InvalidOperationException($"Expected a tensor-like QKV field, got {type}."),
    };

    private static long GetNormalizationSize(IRType type, Tensor value, int axis)
    {
        var shape = value.Shape.ToValueArray();
        var normalizedAxis = NormUtility.NormalizeAxis(axis, shape.Length);
        var fallback = TensorUtilities.GetProduct(shape.AsSpan(normalizedAxis));
        return NormUtility.GetNormalizationSize(GetTensorType(type), axis, fallback);
    }
}
