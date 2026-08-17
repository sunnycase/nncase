// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses a post-distribution packed matmul with normalization statistics over
/// its (possibly broadcast-viewed) output. The fused op emits local additive
/// statistics; the existing Boxing op owns any required cross-block reduction.
/// </summary>
public sealed class FusePackedMatMulNormStatsPass : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function)
        {
            return Task.FromResult(input);
        }

        var candidatesByProducer = CollectCandidates(function);
        var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        foreach (var (producer, candidates) in candidatesByProducer)
        {
            var first = candidates[0];
            if (candidates.Any(candidate =>
                    candidate.Axis != first.Axis || candidate.UseMean != first.UseMean))
            {
                continue;
            }

            var packed = (PackedMatMul)producer.Target;
            var fused = AssertValidCall(
                IR.F.NTT.PackedMatMulNormStats(
                    (Expr)producer[PackedMatMul.Lhs],
                    (Expr)producer[PackedMatMul.Rhs],
                    packed.OutputDataType,
                    packed.RhsLayout,
                    first.Axis,
                    first.UseMean,
                    (Expr)producer[PackedMatMul.Scale],
                    (Expr)producer[PackedMatMul.Addend])
                .InheritMetaData(producer),
                $"forming PackedMatMulNormStats in {function.Name}");
            if (fused.CheckedType is not TupleType { Fields.Count: 2 } fusedType)
            {
                throw new InvalidOperationException(
                    $"PackedMatMulNormStats in {function.Name} must infer two outputs, got {fused.CheckedType}.");
            }

            if (!Equals(fusedType.Fields[0], producer.CheckedType))
            {
                throw new InvalidOperationException(
                    $"PackedMatMulNormStats in {function.Name} changed the matmul boundary type from " +
                    $"{producer.CheckedType} to {fusedType.Fields[0]}.");
            }

            var valueOutput = AssertValidExpr(
                IR.F.Tensors.GetItem(fused, 0).InheritMetaData(producer),
                $"selecting PackedMatMulNormStats value output in {function.Name}");
            var localStats = AssertValidExpr(
                IR.F.Tensors.GetItem(fused, 1),
                $"selecting PackedMatMulNormStats statistics output in {function.Name}");
            replacements.Add(producer, valueOutput);

            foreach (var candidate in candidates)
            {
                Expr replacement = localStats;
                if (!Equals(localStats.CheckedType, candidate.NormStats.CheckedType))
                {
                    replacement = AssertValidCall(
                        IR.F.Distributed.Boxing(localStats, candidate.NormStats.CheckedType),
                        $"reducing PackedMatMulNormStats statistics in {function.Name}");
                }

                if (!Equals(replacement.CheckedType, candidate.NormStats.CheckedType))
                {
                    throw new InvalidOperationException(
                        $"PackedMatMulNormStats in {function.Name} cannot preserve NormStats boundary type " +
                        $"{candidate.NormStats.CheckedType}; got {replacement.CheckedType}.");
                }

                replacements.Add(
                    candidate.NormStats,
                    replacement.InheritMetaData(candidate.NormStats));
            }
        }

        if (replacements.Count == 0)
        {
            return Task.FromResult(input);
        }

        var rewritten = (BaseFunction)new ReplacementRewriter(replacements).Rewrite(function);
        if (!CompilerServices.InferenceType(rewritten))
        {
            throw new InvalidOperationException(
                $"PackedMatMulNormStats fusion could not infer function {function.Name}.");
        }

        if (rewritten.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException(
                $"PackedMatMulNormStats fusion produced an invalid function {function.Name}: {invalid}.");
        }

        return Task.FromResult(rewritten);
    }

    private static Dictionary<Call, List<Candidate>> CollectCandidates(Function function)
    {
        var result = new Dictionary<Call, List<Candidate>>(ReferenceEqualityComparer.Instance);
        foreach (var normStatsCall in ExprCollector.Collect(function.Body)
                     .OfType<Call>()
                     .Where(call => call.Target is NormStats))
        {
            var normStats = (NormStats)normStatsCall.Target;
            if (!TryGetPackedMatMul(
                    normStatsCall[NormStats.Input],
                    out var packedCall,
                    out var packed) ||
                packed.FusedReduce ||
                packedCall.CheckedType is DistributedType { Partial: not null } ||
                packedCall.CheckedType is not (TensorType or DistributedType) ||
                !TryNormalizeAxis(normStats.Axis, packedCall.CheckedShape.Rank, out var axis) ||
                axis != packedCall.CheckedShape.Rank - 1)
            {
                continue;
            }

            if (!result.TryGetValue(packedCall, out var candidates))
            {
                candidates = new List<Candidate>();
                result.Add(packedCall, candidates);
            }

            candidates.Add(new Candidate(normStatsCall, axis, normStats.UseMean));
        }

        return result;
    }

    private static bool TryGetPackedMatMul(
        BaseExpr input,
        out Call packedCall,
        out PackedMatMul packed)
    {
        if (input is Call { Target: PackedMatMul direct } directCall)
        {
            packedCall = directCall;
            packed = direct;
            return true;
        }

        if (input is Call { Target: ShardedView } viewCall &&
            viewCall[ShardedView.Input] is Call { Target: PackedMatMul viewed } viewedCall)
        {
            packedCall = viewedCall;
            packed = viewed;
            return true;
        }

        packedCall = null!;
        packed = null!;
        return false;
    }

    private static bool TryNormalizeAxis(int axis, int rank, out int normalizedAxis)
    {
        normalizedAxis = axis < 0 ? axis + rank : axis;
        return normalizedAxis >= 0 && normalizedAxis < rank;
    }

    private static Call AssertValidCall(Expr expression, string context) =>
        (Call)AssertValidExpr(expression, context);

    private static Expr AssertValidExpr(Expr expression, string context)
    {
        if (!CompilerServices.InferenceType(expression))
        {
            throw new InvalidOperationException($"Failed to infer expression while {context}.");
        }

        if (expression.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException($"Failed while {context}: {invalid}.");
        }

        return expression;
    }

    private sealed record Candidate(Call NormStats, int Axis, bool UseMean);

    private sealed class ReplacementRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;
        private readonly HashSet<BaseExpr> _activeReplacements = new(ReferenceEqualityComparer.Instance);

        public ReplacementRewriter(IReadOnlyDictionary<BaseExpr, BaseExpr> replacements)
        {
            _replacements = replacements;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (!_replacements.TryGetValue(expr, out var replacement))
            {
                return base.DispatchVisit(expr, context);
            }

            if (!_activeReplacements.Add(expr))
            {
                throw new InvalidOperationException(
                    $"PackedMatMulNormStats replacement graph contains a cycle at {expr}.");
            }

            try
            {
                // Replacements are formed from the original DAG. Rewrite their
                // operands as well so dependent fused producers remain shared.
                return Visit(replacement, context);
            }
            finally
            {
                _activeReplacements.Remove(expr);
            }
        }
    }
}
