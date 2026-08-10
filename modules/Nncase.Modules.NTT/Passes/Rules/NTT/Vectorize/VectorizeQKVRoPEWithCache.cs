// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;

using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

/// <summary>
/// Vectorizes the semantic Q/K normalization, RoPE, and cache update region
/// according to the cache storage contract.
/// </summary>
[RuleGenerator]
public sealed partial class VectorizeQKVRoPEWithCache : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsCallWildcard("call", IsOp<QKVRoPEWithCache>("target"));

    private Expr? GetReplace(QKVRoPEWithCache target, Call call)
    {
        if (call[QKVRoPEWithCache.QKV] is not IR.Tuple { Count: 3 } qkv ||
            !TryGetCacheConfig(call[QKVRoPEWithCache.KVCaches], out var cacheConfig) ||
            !TryPackTensor((Expr)qkv[0], cacheConfig, AttentionCacheKind.Key, target.QKVLayout, out var q) ||
            !TryPackTensor((Expr)qkv[1], cacheConfig, AttentionCacheKind.Key, target.QKVLayout, out var k) ||
            !TryPackTensor((Expr)qkv[2], cacheConfig, AttentionCacheKind.Value, target.QKVLayout, out var v) ||
            !TryGetRoPEVectorization(
                (Expr)qkv[0],
                cacheConfig,
                target.QKVLayout,
                out var rotaryAxis,
                out var rotaryLane) ||
            !TryPackNormParameter(
                (Expr)call[QKVRoPEWithCache.QScale],
                target.QAxis,
                ((Expr)qkv[0]).CheckedShape.Rank,
                rotaryAxis,
                rotaryLane,
                out var qScale) ||
            !TryPackNormParameter(
                (Expr)call[QKVRoPEWithCache.QBias],
                target.QAxis,
                ((Expr)qkv[0]).CheckedShape.Rank,
                rotaryAxis,
                rotaryLane,
                out var qBias) ||
            !TryPackNormParameter(
                (Expr)call[QKVRoPEWithCache.KScale],
                target.KAxis,
                ((Expr)qkv[1]).CheckedShape.Rank,
                rotaryAxis,
                rotaryLane,
                out var kScale) ||
            !TryPackNormParameter(
                (Expr)call[QKVRoPEWithCache.KBias],
                target.KAxis,
                ((Expr)qkv[1]).CheckedShape.Rank,
                rotaryAxis,
                rotaryLane,
                out var kBias) ||
            !TryPackSinCos(
                (Expr)call[QKVRoPEWithCache.Cos],
                ((Expr)qkv[0]).CheckedShape.Rank,
                rotaryAxis,
                rotaryLane,
                out var cos) ||
            !TryPackSinCos(
                (Expr)call[QKVRoPEWithCache.Sin],
                ((Expr)qkv[0]).CheckedShape.Rank,
                rotaryAxis,
                rotaryLane,
                out var sin))
        {
            return null;
        }

        var candidate = call.WithArguments([
            (QKVRoPEWithCache.QKV, new IR.Tuple(q, k, v)),
            (QKVRoPEWithCache.QScale, qScale),
            (QKVRoPEWithCache.KScale, kScale),
            (QKVRoPEWithCache.QBias, qBias),
            (QKVRoPEWithCache.KBias, kBias),
            (QKVRoPEWithCache.Cos, cos),
            (QKVRoPEWithCache.Sin, sin),
        ]);
        return candidate.CheckedType is InvalidType || candidate.CheckedType != call.CheckedType
            ? null
            : candidate;
    }

    private static bool TryPackTensor(
        Expr input,
        IPagedAttentionConfig config,
        AttentionCacheKind cacheKind,
        IRArray<AttentionDimKind> layout,
        out Expr packed)
    {
        packed = input;
        if (input.CheckedDataType is not PrimType || input.CheckedShape.IsUnranked)
        {
            return false;
        }

        try
        {
            var (lanes, axes) = AttentionLayoutUtility.GetVectorizeParams(config, layout, cacheKind);
            packed = IR.F.Tensors.Pack(input, lanes, axes);
            return lanes.Length > 0 && packed.CheckedType is not InvalidType;
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            return false;
        }
    }

    private static bool TryGetRoPEVectorization(
        Expr input,
        IPagedAttentionConfig config,
        IRArray<AttentionDimKind> layout,
        out int rotaryAxis,
        out int lane)
    {
        rotaryAxis = -1;
        lane = 0;
        if (input.CheckedShape.IsUnranked ||
            input.CheckedShape.Rank == 0 ||
            layout.Count != input.CheckedShape.Rank)
        {
            return false;
        }

        try
        {
            var (lanes, axes) = AttentionLayoutUtility.GetVectorizeParams(
                config,
                layout,
                AttentionCacheKind.Key);
            if (lanes.Length != 1 || axes.Length != 1 || axes[0] != input.CheckedShape.Rank - 1)
            {
                return false;
            }

            rotaryAxis = axes[0];
            lane = lanes[0];
            return lane > 1;
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            return false;
        }
    }

    private static bool TryPackNormParameter(
        Expr parameter,
        int normalizationAxis,
        int inputRank,
        int vectorizedAxis,
        int lane,
        out Expr packed)
    {
        packed = parameter;
        if (parameter.CheckedDataType is not PrimType || parameter.CheckedShape.IsUnranked)
        {
            return false;
        }

        var normalizedAxis = normalizationAxis < 0 ? normalizationAxis + inputRank : normalizationAxis;
        var parameterAxis = vectorizedAxis - normalizedAxis;
        if (normalizedAxis < 0 ||
            normalizedAxis >= inputRank ||
            parameterAxis < 0 ||
            parameterAxis >= parameter.CheckedShape.Rank)
        {
            return false;
        }

        packed = IR.F.Tensors.Pack(parameter, [lane], [parameterAxis]);
        return packed.CheckedType is not InvalidType;
    }

    private static bool TryPackSinCos(
        Expr input,
        int outputRank,
        int rotaryAxis,
        int lane,
        out Expr packed)
    {
        packed = input;
        if (input.CheckedShape.IsUnranked || input.CheckedShape.Rank == 0)
        {
            return false;
        }

        var inputAxis = rotaryAxis - (outputRank - input.CheckedShape.Rank);
        if (inputAxis < 0 || inputAxis >= input.CheckedShape.Rank)
        {
            return false;
        }

        var f32 = input.CheckedDataType == DataTypes.Float32
            ? input
            : IR.F.Tensors.Cast(input, DataTypes.Float32);
        packed = IR.F.Tensors.Pack(f32, [2, lane], [inputAxis, inputAxis]);
        return packed.CheckedType is not InvalidType;
    }

    private static bool TryGetCacheConfig(BaseExpr value, out IPagedAttentionConfig config)
    {
        if (value.CheckedType is TensorType
            {
                DType: ReferenceType
                {
                    ElemType: PagedAttentionKVCacheType { Config: IPagedAttentionConfig cacheConfig },
                },
            })
        {
            config = cacheConfig;
            return true;
        }

        config = null!;
        return false;
    }
}
