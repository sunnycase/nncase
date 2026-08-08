// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Transforms;
using Nncase.Utilities;

namespace Nncase.Passes;

/// <summary>
/// Forms the semantic Q/K normalization, RoPE, and K/V cache-update operation
/// after physical vector/packing layouts have been selected.
/// </summary>
public sealed class FuseQKVRoPEWithCachePass : FunctionPass
{
    private readonly string _moduleKind;

    public FuseQKVRoPEWithCachePass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function { ModuleKind: var moduleKind } function || moduleKind != _moduleKind)
        {
            return Task.FromResult(input);
        }

        var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        foreach (var pagedAttentionCall in ExprCollector.Collect(function.Body)
                     .OfType<Call>()
                     .Where(call => call.Target is PagedAttention))
        {
            if (!TryCreateFusion(pagedAttentionCall, out var qRoPECall, out var valueUpdateCall, out var fusedCall) ||
                replacements.ContainsKey(qRoPECall) ||
                replacements.ContainsKey(valueUpdateCall))
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

            var qOutput = IR.F.Tensors.GetItem(fusedCall, 0).InheritMetaData(qRoPECall);
            var cacheOutput = IR.F.Tensors.GetItem(fusedCall, 1).InheritMetaData(valueUpdateCall);
            if (!CompilerServices.InferenceType(qOutput) || !CompilerServices.InferenceType(cacheOutput))
            {
                throw new InvalidOperationException(
                    $"Failed to infer QKVRoPEWithCache results formed from {function.Name}.");
            }

            replacements.Add(qRoPECall, qOutput);
            replacements.Add(valueUpdateCall, cacheOutput);
        }

        if (replacements.Count == 0)
        {
            return Task.FromResult(input);
        }

        var rewritten = (BaseFunction)new ReplacementRewriter(replacements).Rewrite(function);
        if (!CompilerServices.InferenceType(rewritten))
        {
            throw new InvalidOperationException(
                $"Failed to infer function {function.Name} after QKVRoPEWithCache fusion.");
        }

        if (rewritten.CheckedType is InvalidType invalidFunction)
        {
            throw new InvalidOperationException(
                $"QKVRoPEWithCache fusion produced an invalid function {function.Name}: {invalidFunction}.");
        }

        return Task.FromResult(rewritten);
    }

    private static bool TryCreateFusion(
        Call pagedAttentionCall,
        out Call qRoPECall,
        out Call valueUpdateCall,
        out Call fusedCall)
    {
        qRoPECall = null!;
        valueUpdateCall = null!;
        fusedCall = null!;
        var pagedAttention = (PagedAttention)pagedAttentionCall.Target;
        if (pagedAttentionCall[PagedAttention.Q] is not Call qRoPE ||
            !TryGetRoPE(qRoPE, out var qNormOutput, out var qCos, out var qSin) ||
            qNormOutput is not Call { Target: NormApply qNorm } qNormCall ||
            pagedAttentionCall[PagedAttention.KVCaches] is not Call { Target: UpdatePagedAttentionKVCache valueUpdate } valueUpdateCandidate ||
            valueUpdate.CacheKind != AttentionCacheKind.Value ||
            valueUpdateCandidate[UpdatePagedAttentionKVCache.KVCaches] is not Call { Target: UpdatePagedAttentionKVCache keyUpdate } keyUpdateCall ||
            keyUpdate.CacheKind != AttentionCacheKind.Key ||
            keyUpdateCall[UpdatePagedAttentionKVCache.Slots] is not Call kRoPE ||
            !TryGetRoPE(kRoPE, out var kNormOutput, out var kCos, out var kSin) ||
            kNormOutput is not Call { Target: NormApply kNorm } kNormCall)
        {
            return false;
        }

        if (!SameValue(qCos, kCos) ||
            !SameValue(qSin, kSin) ||
            !SameValue(pagedAttentionCall[PagedAttention.LayerId], keyUpdateCall[UpdatePagedAttentionKVCache.LayerId]) ||
            !SameValue(pagedAttentionCall[PagedAttention.LayerId], valueUpdateCandidate[UpdatePagedAttentionKVCache.LayerId]) ||
            !keyUpdate.Layout.SequenceEqual(valueUpdate.Layout) ||
            !HasOnlyUser(qRoPE, pagedAttentionCall) ||
            !HasOnlyUser(qNormCall, qRoPE) ||
            !HasOnlyUser(kRoPE, keyUpdateCall) ||
            !HasOnlyUser(kNormCall, kRoPE) ||
            !HasOnlyUser(keyUpdateCall, valueUpdateCandidate))
        {
            return false;
        }

        var qkv = new IR.Tuple(
            (Expr)qNormCall[NormApply.Input],
            (Expr)kNormCall[NormApply.Input],
            (Expr)valueUpdateCandidate[UpdatePagedAttentionKVCache.Slots]);
        fusedCall = IR.F.NN.QKVRoPEWithCache(
                qkv,
                (Expr)qNormCall[NormApply.Stats],
                (Expr)kNormCall[NormApply.Stats],
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
                keyUpdate.Layout)
            .InheritMetaData(valueUpdateCandidate);
        qRoPECall = qRoPE;
        valueUpdateCall = valueUpdateCandidate;
        return true;
    }

    private static bool TryGetRoPE(Call call, out BaseExpr input, out BaseExpr cos, out BaseExpr sin)
    {
        switch (call.Target)
        {
            case RoPE:
                input = call[RoPE.Input];
                cos = call[RoPE.Cos];
                sin = call[RoPE.Sin];
                return true;
            case IR.NTT.VectorizedRoPE:
                input = call[IR.NTT.VectorizedRoPE.Input];
                cos = call[IR.NTT.VectorizedRoPE.Cos];
                sin = call[IR.NTT.VectorizedRoPE.Sin];
                return true;
            default:
                input = null!;
                cos = null!;
                sin = null!;
                return false;
        }
    }

    private static bool HasOnlyUser(BaseExpr value, BaseExpr expectedUser) =>
        value.Users.Count() == 1 && ReferenceEquals(value.Users.Single(), expectedUser);

    private static bool SameValue(BaseExpr lhs, BaseExpr rhs) =>
        ReferenceEquals(lhs, rhs) || lhs.Equals(rhs);

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
