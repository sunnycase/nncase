// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Forms semantic Q/K normalization, RoPE, attention-layout conversion, and
/// K/V cache update before target vectorization and packing.
/// </summary>
public sealed class FormQKVRoPEWithCachePass : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function)
        {
            return Task.FromResult(input);
        }

        var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        foreach (var pagedAttentionCall in ExprCollector.Collect(function.Body)
                     .OfType<Call>()
                     .Where(call => call.Target is PagedAttention))
        {
            if (!TryCreateFusion(
                    pagedAttentionCall,
                    out var queryView,
                    out var valueUpdate,
                    out var fusedCall) ||
                replacements.ContainsKey(queryView) ||
                replacements.ContainsKey(valueUpdate))
            {
                continue;
            }

            if (!CompilerServices.InferenceType(fusedCall))
            {
                throw new InvalidOperationException(
                    $"Failed to infer QKVRoPEWithCache formed from {function.Name}.");
            }

            if (fusedCall.CheckedType is InvalidType invalid)
            {
                throw new InvalidOperationException(
                    $"Failed to infer QKVRoPEWithCache formed from {function.Name}: {invalid}.");
            }

            var qOutput = IR.F.Tensors.GetItem(fusedCall, 0).InheritMetaData(queryView);
            qOutput.Metadata.SemanticRegion = fusedCall.Metadata.SemanticRegion;
            var cacheOutput = IR.F.Tensors.GetItem(fusedCall, 1);
            cacheOutput.Metadata = valueUpdate.Metadata.Clone();
            if (!CompilerServices.InferenceType(qOutput) ||
                !CompilerServices.InferenceType(cacheOutput) ||
                qOutput.CheckedType != queryView.CheckedType ||
                cacheOutput.CheckedType != valueUpdate.CheckedType)
            {
                throw new InvalidOperationException(
                    $"QKVRoPEWithCache formed from {function.Name} does not preserve its region boundary types.");
            }

            replacements.Add(queryView, qOutput);
            replacements.Add(valueUpdate, cacheOutput);
        }

        if (replacements.Count == 0)
        {
            return Task.FromResult(input);
        }

        var rewritten = (BaseFunction)new ReplacementRewriter(replacements).Rewrite(function);
        if (!CompilerServices.InferenceType(rewritten))
        {
            throw new InvalidOperationException(
                $"QKVRoPEWithCache formation failed to infer function {function.Name}.");
        }

        if (rewritten.CheckedType is InvalidType invalidFunction)
        {
            throw new InvalidOperationException(
                $"QKVRoPEWithCache formation produced an invalid function {function.Name}: {invalidFunction}.");
        }

        return Task.FromResult(rewritten);
    }

    private static bool TryCreateFusion(
        Call pagedAttentionCall,
        out Expr queryView,
        out Call valueUpdateCall,
        out Call fusedCall)
    {
        queryView = null!;
        valueUpdateCall = null!;
        fusedCall = null!;
        var pagedAttention = (PagedAttention)pagedAttentionCall.Target;
        if (!AttentionLayoutUtility.IsValid(pagedAttention.Layout) ||
            !TryGetCacheUpdate(
                pagedAttentionCall[PagedAttention.KVCaches],
                AttentionCacheKind.Value,
                out var valueUpdateCandidate,
                out var valueUpdate) ||
            !TryGetCacheUpdate(
                valueUpdateCandidate[UpdatePagedAttentionKVCache.KVCaches],
                AttentionCacheKind.Key,
                out var keyUpdateCall,
                out var keyUpdate) ||
            !keyUpdate.Layout.SequenceEqual(pagedAttention.Layout) ||
            !valueUpdate.Layout.SequenceEqual(pagedAttention.Layout) ||
            !TryGetCacheConfig(keyUpdateCall[UpdatePagedAttentionKVCache.KVCaches], out var cacheConfig))
        {
            return false;
        }

        if (!TryMatchLayoutView(
                (Expr)pagedAttentionCall[PagedAttention.Q],
                pagedAttention.Layout,
                cacheConfig,
                AttentionCacheKind.Key,
                out var queryLayoutView) ||
            !TryMatchLayoutView(
                (Expr)keyUpdateCall[UpdatePagedAttentionKVCache.Slots],
                pagedAttention.Layout,
                cacheConfig,
                AttentionCacheKind.Key,
                out var keyLayoutView) ||
            !TryMatchLayoutView(
                (Expr)valueUpdateCandidate[UpdatePagedAttentionKVCache.Slots],
                pagedAttention.Layout,
                cacheConfig,
                AttentionCacheKind.Value,
                out var valueLayoutView) ||
            !queryLayoutView.InputLayout.SequenceEqual(keyLayoutView.InputLayout) ||
            !queryLayoutView.InputLayout.SequenceEqual(valueLayoutView.InputLayout))
        {
            return false;
        }

        if (queryLayoutView.Source is not Call qRoPE)
        {
            return false;
        }

        if (!TryGetRoPE(qRoPE, out var qNormOutput, out var qCos, out var qSin))
        {
            return false;
        }

        if (qNormOutput is not Call { Target: NormApply qNorm } qNormCall)
        {
            return false;
        }

        if (keyLayoutView.Source is not Call kRoPE)
        {
            return false;
        }

        if (!TryGetRoPE(kRoPE, out var kNormOutput, out var kCos, out var kSin))
        {
            return false;
        }

        if (kNormOutput is not Call { Target: NormApply kNorm } kNormCall)
        {
            return false;
        }

        if (!HasMatchingNormStats(qNormCall, qNorm) ||
            !HasMatchingNormStats(kNormCall, kNorm) ||
            !SameValue(qCos, kCos) ||
            !SameValue(qSin, kSin) ||
            !SameValue(pagedAttentionCall[PagedAttention.LayerId], keyUpdateCall[UpdatePagedAttentionKVCache.LayerId]) ||
            !SameValue(pagedAttentionCall[PagedAttention.LayerId], valueUpdateCandidate[UpdatePagedAttentionKVCache.LayerId]) ||
            !queryLayoutView.HasOnlyUser(pagedAttentionCall) ||
            !keyLayoutView.HasOnlyUser(keyUpdateCall) ||
            !valueLayoutView.HasOnlyUser(valueUpdateCandidate) ||
            !HasOnlyUser(qRoPE, queryLayoutView.GetSourceUser(pagedAttentionCall)) ||
            !HasOnlyUser(qNormCall, qRoPE) ||
            !HasOnlyUser(kRoPE, keyLayoutView.GetSourceUser(keyUpdateCall)) ||
            !HasOnlyUser(kNormCall, kRoPE) ||
            !HasOnlyUser(keyUpdateCall, valueUpdateCandidate))
        {
            return false;
        }

        var absorbedCalls = queryLayoutView.Nodes
            .Concat(keyLayoutView.Nodes)
            .Concat(valueLayoutView.Nodes)
            .Concat(
            [
                qRoPE,
                qNormCall,
                (Call)qNormCall[NormApply.Stats],
                kRoPE,
                kNormCall,
                (Call)kNormCall[NormApply.Stats],
                keyUpdateCall,
                valueUpdateCandidate,
            ]);
        if (!SemanticRegionUtility.HaveUniformRegion(absorbedCalls))
        {
            return false;
        }

        var qkv = new IR.Tuple(
            (Expr)qNormCall[NormApply.Input],
            (Expr)kNormCall[NormApply.Input],
            valueLayoutView.Source);
        fusedCall = IR.F.NN.QKVRoPEWithCache(
                qkv,
                (Expr)qNormCall[NormApply.Scale],
                (Expr)kNormCall[NormApply.Scale],
                (Expr)qNormCall[NormApply.Bias],
                (Expr)kNormCall[NormApply.Bias],
                (Expr)qCos,
                (Expr)qSin,
                (Expr)keyUpdateCall[UpdatePagedAttentionKVCache.KVCaches],
                (Dimension)pagedAttentionCall[PagedAttention.LayerId],
                qNorm.Axis,
                qNorm.Epsilon,
                qNorm.UseMean,
                kNorm.Axis,
                kNorm.Epsilon,
                kNorm.UseMean,
                queryLayoutView.InputLayout,
                pagedAttention.Layout);
        fusedCall.Metadata = valueUpdateCandidate.Metadata.Clone();
        queryView = queryLayoutView.Root;
        valueUpdateCall = valueUpdateCandidate;
        return true;
    }

    private static bool HasMatchingNormStats(Call normApplyCall, NormApply normApply)
    {
        if (normApplyCall[NormApply.Stats] is not Call { Target: NormStats normStats } normStatsCall ||
            !SameValue(normApplyCall[NormApply.Input], normStatsCall[NormStats.Input]) ||
            normApply.UseMean != normStats.UseMean)
        {
            return false;
        }

        var input = (Expr)normApplyCall[NormApply.Input];
        if (input.CheckedShape.IsUnranked)
        {
            return false;
        }

        var rank = input.CheckedShape.Rank;
        var applyAxis = normApply.Axis < 0 ? normApply.Axis + rank : normApply.Axis;
        var statsAxis = normStats.Axis < 0 ? normStats.Axis + rank : normStats.Axis;
        return applyAxis >= 0 && applyAxis < rank && applyAxis == statsAxis;
    }

    private static bool TryGetCacheUpdate(
        BaseExpr value,
        AttentionCacheKind cacheKind,
        out Call call,
        out UpdatePagedAttentionKVCache update)
    {
        if (value is Call { Target: UpdatePagedAttentionKVCache candidate } candidateCall &&
            candidate.CacheKind == cacheKind)
        {
            call = candidateCall;
            update = candidate;
            return true;
        }

        call = null!;
        update = null!;
        return false;
    }

    private static bool TryMatchLayoutView(
        Expr root,
        IRArray<AttentionDimKind> outputLayout,
        IPagedAttentionConfig cacheConfig,
        AttentionCacheKind cacheKind,
        out LayoutView view)
    {
        view = null!;
        Expr current = root;
        var nodes = new List<Call>();
        int[] expectedLanes;
        int[] expectedAxes;
        try
        {
            (expectedLanes, expectedAxes) = AttentionLayoutUtility.GetVectorizeParams(
                cacheConfig,
                outputLayout,
                cacheKind);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            return false;
        }

        if (expectedLanes.Length > 0)
        {
            if (current is not Call { Target: Pack pack } packCall ||
                !pack.Lanes.SequenceEqual(expectedLanes) ||
                !pack.Axes.SequenceEqual(expectedAxes))
            {
                return false;
            }

            nodes.Add(packCall);
            current = (Expr)packCall[Pack.Input];
        }

        IRArray<AttentionDimKind> inputLayout;
        if (current is Call { Target: Transpose } transposeCall)
        {
            if (transposeCall[Transpose.Perm] is not Shape { IsFixed: true } permutationShape)
            {
                return false;
            }

            var permutation = permutationShape.ToValueArray().Select(value => checked((int)value)).ToArray();
            try
            {
                inputLayout = AttentionLayoutUtility.GetInputLayout(outputLayout, permutation);
            }
            catch (ArgumentException)
            {
                return false;
            }

            nodes.Add(transposeCall);
            current = (Expr)transposeCall[Transpose.Input];
        }
        else
        {
            inputLayout = outputLayout;
        }

        view = new LayoutView(root, current, inputLayout, nodes);
        return true;
    }

    private static bool TryGetRoPE(Call call, out BaseExpr input, out BaseExpr cos, out BaseExpr sin)
    {
        if (call.Target is not RoPE)
        {
            input = null!;
            cos = null!;
            sin = null!;
            return false;
        }

        input = call[RoPE.Input];
        cos = call[RoPE.Cos];
        sin = call[RoPE.Sin];
        return true;
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

    private static bool HasOnlyUser(BaseExpr value, BaseExpr expectedUser) =>
        value.Users.Count() == 1 && ReferenceEquals(value.Users.Single(), expectedUser);

    private static bool SameValue(BaseExpr lhs, BaseExpr rhs) =>
        ReferenceEquals(lhs, rhs) || lhs.Equals(rhs);

    private sealed record LayoutView(
        Expr Root,
        Expr Source,
        IRArray<AttentionDimKind> InputLayout,
        IReadOnlyList<Call> Nodes)
    {
        public BaseExpr GetSourceUser(BaseExpr terminalUser) =>
            Nodes.Count == 0 ? terminalUser : Nodes[^1];

        public bool HasOnlyUser(BaseExpr terminalUser)
        {
            BaseExpr expectedUser = terminalUser;
            foreach (var node in Nodes)
            {
                if (!FormQKVRoPEWithCachePass.HasOnlyUser(node, expectedUser))
                {
                    return false;
                }

                expectedUser = node;
            }

            return true;
        }
    }

    private sealed class ReplacementRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;

        public ReplacementRewriter(IReadOnlyDictionary<BaseExpr, BaseExpr> replacements)
        {
            _replacements = replacements;
        }

        protected override BaseExpr DefaultRewriteLeaf(BaseExpr expr) =>
            _replacements.TryGetValue(expr, out var replacement) ? replacement : expr;
    }
}
