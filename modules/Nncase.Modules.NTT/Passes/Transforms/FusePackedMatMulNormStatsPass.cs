// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Heterogeneous;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses a post-distribution packed matmul with normalization statistics over
/// its (possibly broadcast-viewed) output. The fused op emits local additive
/// statistics; the existing Boxing op owns any required cross-block reduction.
/// </summary>
public sealed class FusePackedMatMulNormStatsPass : FunctionPass
{
    private readonly bool _enablePackedMatMul;
    private readonly bool _enableBlockScaledMatMul;
    private readonly bool _enableNVFP4MatMul;

    public FusePackedMatMulNormStatsPass(
        bool enablePackedMatMul,
        bool enableBlockScaledMatMul,
        bool enableNVFP4MatMul)
    {
        _enablePackedMatMul = enablePackedMatMul;
        _enableBlockScaledMatMul = enableBlockScaledMatMul;
        _enableNVFP4MatMul = enableNVFP4MatMul;
    }

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

            var fused = AssertValidCall(
                CreateFusedProducer(producer, first.Axis, first.UseMean)
                    .InheritMetaData(producer),
                $"forming packed matmul normalization statistics in {function.Name}");
            if (fused.CheckedType is not TupleType { Fields.Count: 2 } fusedType)
            {
                throw new InvalidOperationException(
                    $"Fused packed matmul normalization statistics in {function.Name} must infer two outputs, got {fused.CheckedType}.");
            }

            if (!Equals(fusedType.Fields[0], producer.CheckedType))
            {
                throw new InvalidOperationException(
                    $"Fused packed matmul normalization statistics in {function.Name} changed the matmul boundary type from " +
                    $"{producer.CheckedType} to {fusedType.Fields[0]}.");
            }

            var valueOutput = AssertValidExpr(
                IR.F.Tensors.GetItem(fused, 0).InheritMetaData(producer),
                $"selecting fused packed matmul value output in {function.Name}");
            var localStats = AssertValidExpr(
                IR.F.Tensors.GetItem(fused, 1),
                $"selecting fused packed matmul statistics output in {function.Name}");
            replacements.Add(producer, valueOutput);

            foreach (var candidate in candidates)
            {
                Expr replacement = localStats;
                foreach (var dependency in candidate.OrderingDependencies.Reverse())
                {
                    replacement = AssertValidCall(
                        IR.F.Heterogeneous.PipelineYield(replacement, dependency),
                        $"preserving packed matmul normalization-statistics ordering in {function.Name}");
                }

                if (!Equals(localStats.CheckedType, candidate.NormStats.CheckedType))
                {
                    replacement = AssertValidCall(
                        IR.F.Distributed.Boxing(localStats, candidate.NormStats.CheckedType),
                        $"reducing fused packed matmul statistics in {function.Name}");
                }

                if (!Equals(replacement.CheckedType, candidate.NormStats.CheckedType))
                {
                    throw new InvalidOperationException(
                        $"Fused packed matmul normalization statistics in {function.Name} cannot preserve NormStats boundary type " +
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

    private Dictionary<Call, List<Candidate>> CollectCandidates(Function function)
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
                    out var orderingDependencies) ||
                packedCall.Target is not Op packedTarget ||
                !CanFuseProducer(packedTarget) ||
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

            candidates.Add(new Candidate(
                normStatsCall,
                axis,
                normStats.UseMean,
                orderingDependencies));
        }

        return result;
    }

    private static bool TryGetPackedMatMul(
        BaseExpr input,
        out Call packedCall,
        out IReadOnlyList<Expr> orderingDependencies)
    {
        var dependencies = new List<Expr>();
        var current = input;
        while (current is Call call)
        {
            if (call.Target is PackedMatMul or PackedBlockScaledMatMul or PackedNVFP4MatMul)
            {
                if (!HaveSameLogicalTensorType(input.CheckedType, call.CheckedType))
                {
                    break;
                }

                packedCall = call;
                orderingDependencies = dependencies;
                return true;
            }

            switch (call.Target)
            {
                case ShardedView:
                    current = call[ShardedView.Input];
                    break;
                case PipelineYield:
                    if (call[PipelineYield.Dependency] is not Expr dependency)
                    {
                        packedCall = null!;
                        orderingDependencies = Array.Empty<Expr>();
                        return false;
                    }

                    dependencies.Add(dependency);
                    current = call[PipelineYield.Value];
                    break;
                case Pack:
                    current = call[Pack.Input];
                    break;
                case Unpack:
                    current = call[Unpack.Input];
                    break;
                case Bitcast:
                    current = call[Bitcast.Input];
                    break;
                case Reshape:
                    current = call[Reshape.Input];
                    break;
                default:
                    packedCall = null!;
                    orderingDependencies = Array.Empty<Expr>();
                    return false;
            }
        }

        packedCall = null!;
        orderingDependencies = Array.Empty<Expr>();
        return false;
    }

    private static bool HaveSameLogicalTensorType(IRType lhs, IRType rhs)
    {
        var lhsTensor = lhs switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null,
        };
        var rhsTensor = rhs switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null,
        };

        return lhsTensor is not null && Equals(lhsTensor, rhsTensor);
    }

    private bool CanFuseProducer(Op target) => target switch
    {
        PackedMatMul { FusedReduce: false } => _enablePackedMatMul,
        PackedBlockScaledMatMul => _enableBlockScaledMatMul,
        PackedNVFP4MatMul => _enableNVFP4MatMul,
        _ => false,
    };

    private static Expr CreateFusedProducer(Call producer, int axis, bool useMean) =>
        producer.Target switch
        {
            PackedMatMul packed => IR.F.NTT.PackedMatMulNormStats(
                (Expr)producer[PackedMatMul.Lhs],
                (Expr)producer[PackedMatMul.Rhs],
                packed.OutputDataType,
                packed.RhsLayout,
                axis,
                useMean,
                (Expr)producer[PackedMatMul.Scale],
                (Expr)producer[PackedMatMul.Addend]),
            PackedBlockScaledMatMul packed => IR.F.NTT.PackedBlockScaledMatMulNormStats(
                (Expr)producer[PackedBlockScaledMatMul.Lhs],
                (Expr)producer[PackedBlockScaledMatMul.Rhs],
                (Expr)producer[PackedBlockScaledMatMul.RhsScale],
                packed.OutputDataType,
                packed.WeightBlockN,
                packed.WeightBlockK,
                packed.RhsLayout,
                packed.OutputNVectorLaneCount,
                axis,
                useMean,
                (Expr)producer[PackedBlockScaledMatMul.Addend]),
            PackedNVFP4MatMul packed => IR.F.NTT.PackedNVFP4MatMulNormStats(
                (Expr)producer[PackedNVFP4MatMul.Lhs],
                (Expr)producer[PackedNVFP4MatMul.RhsPacked],
                (Expr)producer[PackedNVFP4MatMul.RhsScale],
                (Expr)producer[PackedNVFP4MatMul.LhsGlobalScale],
                (Expr)producer[PackedNVFP4MatMul.RhsGlobalScale],
                packed.OutputDataType,
                packed.GroupSize,
                packed.InputKVectorLaneCount,
                packed.RhsKPackLaneCount,
                packed.RhsKVectorLaneCount,
                packed.OutputNVectorLaneCount,
                axis,
                useMean,
                (Expr)producer[PackedNVFP4MatMul.Addend]),
            _ => throw new InvalidOperationException(
                $"Unsupported packed matmul normalization-statistics producer {producer.Target.GetType().Name}."),
        };

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

    private sealed record Candidate(
        Call NormStats,
        int Axis,
        bool UseMean,
        IReadOnlyList<Expr> OrderingDependencies);

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
