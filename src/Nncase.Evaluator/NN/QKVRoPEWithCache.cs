// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.Evaluator.Tensors;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
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

        var qStats = NormStatsEvaluator.InferType(
            new NormStats(target.QAxis, target.QUseMean),
            tuple[0]);
        if (qStats is InvalidType)
        {
            return qStats;
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

        var kStats = NormStatsEvaluator.InferType(
            new NormStats(target.KAxis, target.KUseMean),
            tuple[1]);
        if (kStats is InvalidType)
        {
            return kStats;
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

        var qRoPE = RoPEEvaluator.InferType(qNorm, cos, sin);
        if (qRoPE is InvalidType)
        {
            return qRoPE;
        }

        var kRoPE = RoPEEvaluator.InferType(kNorm, cos, sin);
        if (kRoPE is InvalidType)
        {
            return kRoPE;
        }

        if (!TryGetCacheConfig(kvCaches, out var cacheConfig, out var cacheError))
        {
            return cacheError;
        }

        var qOutput = TransformAttentionLayoutType(
            qRoPE,
            cacheConfig,
            AttentionCacheKind.Key,
            target.QKVLayout,
            target.AttentionLayout);
        if (qOutput is InvalidType)
        {
            return qOutput;
        }

        var kSlots = TransformAttentionLayoutType(
            kRoPE,
            cacheConfig,
            AttentionCacheKind.Key,
            target.QKVLayout,
            target.AttentionLayout);
        if (kSlots is InvalidType)
        {
            return kSlots;
        }

        var vSlots = TransformAttentionLayoutType(
            tuple[2],
            cacheConfig,
            AttentionCacheKind.Value,
            target.QKVLayout,
            target.AttentionLayout);
        if (vSlots is InvalidType)
        {
            return vSlots;
        }

        var afterKey = UpdatePagedAttentionKVCacheEvaluator.InferType(
            new UpdatePagedAttentionKVCache(AttentionCacheKind.Key, target.AttentionLayout),
            kSlots,
            kvCaches);
        if (afterKey is InvalidType)
        {
            return afterKey;
        }

        var afterValue = UpdatePagedAttentionKVCacheEvaluator.InferType(
            new UpdatePagedAttentionKVCache(AttentionCacheKind.Value, target.AttentionLayout),
            vSlots,
            afterKey);
        return afterValue is InvalidType
            ? afterValue
            : new TupleType([qOutput, afterValue]);
    }

    public static IRType TransformAttentionLayoutType(
        IRType input,
        IPagedAttentionConfig cacheConfig,
        AttentionCacheKind cacheKind,
        IRArray<AttentionDimKind> inputLayout,
        IRArray<AttentionDimKind> outputLayout)
    {
        try
        {
            var permutation = AttentionLayoutUtility.GetPermutation(inputLayout, outputLayout);
            Shape permutationShape = new RankedShape(permutation.Select(axis => (Dimension)axis).ToArray());
            var transposed = input switch
            {
                TensorType tensor => TransposeEvaluator.Visit(tensor, permutationShape),
                DistributedType distributed => TransposeEvaluator.Visit(distributed, permutationShape),
                _ => new InvalidType($"Expected a tensor-like attention input, got {input}."),
            };
            if (transposed is InvalidType)
            {
                return transposed;
            }

            var transposedTensor = GetTensorType(transposed);
            var expectedDType = cacheConfig.GetKVType(cacheKind);
            if (transposedTensor.DType == expectedDType)
            {
                return transposed;
            }

            if (transposedTensor.DType != cacheConfig.KVPrimType)
            {
                return new InvalidType(
                    $"QKVRoPEWithCache {cacheKind} input dtype {transposedTensor.DType} " +
                    $"must be either {cacheConfig.KVPrimType} or {expectedDType}.");
            }

            var (lanes, axes) = AttentionLayoutUtility.GetVectorizeParams(
                cacheConfig,
                outputLayout,
                cacheKind);
            return transposed switch
            {
                TensorType tensor => TypeInference.PackType(tensor, lanes, axes),
                DistributedType distributed => TypeInference.PackType(distributed, lanes, axes),
                _ => throw new InvalidOperationException(),
            };
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            return new InvalidType(exception.Message);
        }
    }

    public IValue Visit(IEvaluateContext context, QKVRoPEWithCache target)
    {
        var qkv = context.GetArgumentValueAsTensors(target, QKVRoPEWithCache.QKV);
        if (qkv.Length != 3)
        {
            throw new InvalidOperationException($"QKVRoPEWithCache expects three QKV tensors, got {qkv.Length}.");
        }

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
        var cacheTensor = kvCaches.AsTensor().Cast<Reference<IPagedAttentionKVCache>>();
        var cacheConfig = cacheTensor.Single().Value.Config;

        var (qLanes, qAxes) = AttentionLayoutUtility.GetVectorizeParams(
            cacheConfig,
            target.QKVLayout,
            AttentionCacheKind.Key);
        var qInput = ToLogicalTensor(qkv[0], qLanes, qAxes, "Q");
        var qScaleInput = ToLogicalNormParameter(
            qScale,
            qLanes,
            qAxes,
            target.QAxis,
            qkv[0].Shape.Count,
            "Q scale");
        var qBiasInput = ToLogicalNormParameter(
            qBias,
            qLanes,
            qAxes,
            target.QAxis,
            qkv[0].Shape.Count,
            "Q bias");
        var qCos = ToLogicalRoPEParameter(cos, qLanes, qAxes, qkv[0].Shape.Count, "cos");
        var qSin = ToLogicalRoPEParameter(sin, qLanes, qAxes, qkv[0].Shape.Count, "sin");
        var qStats = NormStatsEvaluator.Evaluate(qInput, target.QAxis, target.QUseMean);

        var q = NormApplyEvaluator.Evaluate(
            qInput,
            qStats,
            qScaleInput,
            qBiasInput,
            GetScalarDataType(GetTensorType(qkvType[0]).DType),
            target.QAxis,
            target.QEpsilon,
            target.QUseMean,
            GetNormalizationSize(qkvType[0], qkv[0], target.QAxis));
        q = RoPEEvaluator.Evaluate(q, qCos, qSin);

        var (kLanes, kAxes) = AttentionLayoutUtility.GetVectorizeParams(
            cacheConfig,
            target.QKVLayout,
            AttentionCacheKind.Key);
        var kInput = ToLogicalTensor(qkv[1], kLanes, kAxes, "K");
        var kScaleInput = ToLogicalNormParameter(
            kScale,
            kLanes,
            kAxes,
            target.KAxis,
            qkv[1].Shape.Count,
            "K scale");
        var kBiasInput = ToLogicalNormParameter(
            kBias,
            kLanes,
            kAxes,
            target.KAxis,
            qkv[1].Shape.Count,
            "K bias");
        var kCos = ToLogicalRoPEParameter(cos, kLanes, kAxes, qkv[1].Shape.Count, "cos");
        var kSin = ToLogicalRoPEParameter(sin, kLanes, kAxes, qkv[1].Shape.Count, "sin");
        var kStats = NormStatsEvaluator.Evaluate(kInput, target.KAxis, target.KUseMean);

        var k = NormApplyEvaluator.Evaluate(
            kInput,
            kStats,
            kScaleInput,
            kBiasInput,
            GetScalarDataType(GetTensorType(qkvType[1]).DType),
            target.KAxis,
            target.KEpsilon,
            target.KUseMean,
            GetNormalizationSize(qkvType[1], qkv[1], target.KAxis));
        k = RoPEEvaluator.Evaluate(k, kCos, kSin);

        var qOutput = TransformAttentionLayoutValue(
            q,
            cacheConfig,
            AttentionCacheKind.Key,
            target.QKVLayout,
            target.AttentionLayout);
        var kSlots = TransformAttentionLayoutValue(
            k,
            cacheConfig,
            AttentionCacheKind.Key,
            target.QKVLayout,
            target.AttentionLayout);
        var vSlots = TransformAttentionLayoutValue(
            qkv[2],
            cacheConfig,
            AttentionCacheKind.Value,
            target.QKVLayout,
            target.AttentionLayout);
        UpdatePagedAttentionKVCacheEvaluator.UpdateCache(
            kSlots,
            cacheTensor,
            AttentionCacheKind.Key,
            layerId,
            target.AttentionLayout);
        UpdatePagedAttentionKVCacheEvaluator.UpdateCache(
            vSlots,
            cacheTensor,
            AttentionCacheKind.Value,
            layerId,
            target.AttentionLayout);
        return new TupleValue([Value.FromTensor(qOutput), kvCaches]);
    }

    public IRType Visit(ITypeInferenceContext context, QKVRoPEWithCache target)
    {
        var qkv = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.QKV);
        var qScale = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.QScale);
        var kScale = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.KScale);
        var qBias = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.QBias);
        var kBias = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.KBias);
        var cos = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.Cos);
        var sin = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.Sin);
        var kvCaches = context.CheckArgumentType<IRType>(target, QKVRoPEWithCache.KVCaches);
        _ = context.CheckArgumentType<DimensionType>(target, QKVRoPEWithCache.LayerId);
        return InferType(target, qkv, qScale, kScale, qBias, kBias, cos, sin, kvCaches);
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
        var qScale = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.QScale);
        var kScale = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.KScale);
        var qBias = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.QBias);
        var kBias = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.KBias);
        var cos = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.Cos);
        var sin = context.GetArgumentType<IRType>(target, QKVRoPEWithCache.Sin);
        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] =
                (2 * CostUtility.GetMemoryAccess(q)) + (2 * CostUtility.GetMemoryAccess(k)) + CostUtility.GetMemoryAccess(v) +
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

    private static DataType GetScalarDataType(DataType dataType) => dataType switch
    {
        VectorType vectorType => vectorType.ElemType,
        MaskVectorType => DataTypes.Boolean,
        _ => dataType,
    };

    private static Tensor ToLogicalTensor(
        Tensor value,
        int[] expectedLanes,
        int[] vectorizedAxes,
        string argumentName)
    {
        if (value.ElementType is not VectorType vectorType)
        {
            return value;
        }

        if (!vectorType.Lanes.SequenceEqual(expectedLanes))
        {
            throw new InvalidOperationException(
                $"QKVRoPEWithCache {argumentName} lanes [{string.Join(",", vectorType.Lanes)}] " +
                $"do not match cache lanes [{string.Join(",", expectedLanes)}].");
        }

        if (vectorizedAxes.Length != vectorType.Lanes.Count ||
            vectorizedAxes.Any(axis => axis < 0 || axis >= value.Shape.Count))
        {
            throw new InvalidOperationException(
                $"QKVRoPEWithCache {argumentName} vector axes [{string.Join(",", vectorizedAxes)}] " +
                $"are invalid for rank {value.Shape.Count} and lanes [{string.Join(",", vectorType.Lanes)}].");
        }

        return value.ToOrtTensor()
            .Unpack(vectorType.Lanes.Count, vectorizedAxes)
            .ToValue(vectorType.ElemType)
            .AsTensor();
    }

    private static Tensor ToLogicalNormParameter(
        Tensor value,
        int[] expectedLanes,
        int[] inputVectorizedAxes,
        int normalizationAxis,
        int inputRank,
        string argumentName)
    {
        if (value.ElementType is not VectorType)
        {
            return value;
        }

        var normalizedAxis = NormUtility.NormalizeAxis(normalizationAxis, inputRank);
        var parameterAxes = inputVectorizedAxes.Select(axis => axis - normalizedAxis).ToArray();
        return ToLogicalTensor(value, expectedLanes, parameterAxes, argumentName);
    }

    private static Tensor ToLogicalRoPEParameter(
        Tensor value,
        int[] inputLanes,
        int[] inputVectorizedAxes,
        int inputRank,
        string argumentName)
    {
        if (value.ElementType is not VectorType)
        {
            return value;
        }

        if (inputLanes.Length != 1 || inputVectorizedAxes.Length != 1)
        {
            throw new NotSupportedException(
                $"QKVRoPEWithCache vectorized {argumentName} requires one rotary vector axis.");
        }

        var parameterAxis = inputVectorizedAxes[0] - (inputRank - value.Shape.Count);
        return ToLogicalTensor(
            value,
            [2, inputLanes[0]],
            [parameterAxis, parameterAxis],
            argumentName);
    }

    private static long GetNormalizationSize(IRType type, Tensor value, int axis)
    {
        var shape = value.Shape.ToValueArray();
        var normalizedAxis = NormUtility.NormalizeAxis(axis, shape.Length);
        var fallback = TensorUtilities.GetProduct(shape.AsSpan(normalizedAxis));
        return NormUtility.GetNormalizationSize(GetTensorType(type), axis, fallback);
    }

    private static Tensor TransformAttentionLayoutValue(
        Tensor input,
        IPagedAttentionConfig cacheConfig,
        AttentionCacheKind cacheKind,
        IRArray<AttentionDimKind> inputLayout,
        IRArray<AttentionDimKind> outputLayout)
    {
        var permutation = AttentionLayoutUtility.GetPermutation(inputLayout, outputLayout);
        var transposed = input.Transpose(permutation.Select(axis => (long)axis).ToArray());
        var expectedDType = cacheConfig.GetKVType(cacheKind);
        if (transposed.ElementType == expectedDType)
        {
            return transposed;
        }

        if (transposed.ElementType != cacheConfig.KVPrimType)
        {
            throw new InvalidOperationException(
                $"QKVRoPEWithCache {cacheKind} input dtype {transposed.ElementType} " +
                $"must be either {cacheConfig.KVPrimType} or {expectedDType}.");
        }

        var (lanes, axes) = AttentionLayoutUtility.GetVectorizeParams(
            cacheConfig,
            outputLayout,
            cacheKind);
        if (lanes.Length == 0)
        {
            return transposed;
        }

        return transposed.ToOrtTensor()
            .Pack(0, lanes, axes)
            .ToValue(expectedDType)
            .AsTensor();
    }

    private static bool TryGetCacheConfig(
        IRType kvCaches,
        out IPagedAttentionConfig config,
        out InvalidType error)
    {
        if (kvCaches is TensorType
            {
                DType: ReferenceType
                {
                    ElemType: PagedAttentionKVCacheType { Config: IPagedAttentionConfig cacheConfig },
                },
            })
        {
            config = cacheConfig;
            error = null!;
            return true;
        }

        config = null!;
        error = new InvalidType($"QKVRoPEWithCache requires a configured paged-attention cache, got {kvCaches}.");
        return false;
    }
}
