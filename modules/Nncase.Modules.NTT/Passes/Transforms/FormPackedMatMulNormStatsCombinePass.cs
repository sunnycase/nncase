// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Makes packed-matmul partial materialization explicit before AutoDistributed
/// so split-K remains searchable across residual addition and NormStats.
/// </summary>
public sealed class FormPackedMatMulNormStatsCombinePass : FunctionPass
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

            var addend = GetAddend(producer);
            if (addend.CheckedType is NoneType)
            {
                continue;
            }

            var partialCapable = AssertValidExpr(
                CreatePartialCapableProducer(producer),
                $"forming partial-capable PackedMatMul in {function.Name}");
            if (!Equals(partialCapable.CheckedType, producer.CheckedType))
            {
                throw new InvalidOperationException(
                    $"Removing the packed-matmul addend in {function.Name} changed its value type from " +
                    $"{producer.CheckedType} to {partialCapable.CheckedType}.");
            }

            var outputType = new TupleType(new[]
            {
                producer.CheckedType,
                first.NormStats.CheckedType,
            });
            var combine = AssertValidExpr(
                IR.F.NTT.PackedMatMulNormStatsCombine(
                    partialCapable,
                    addend,
                    outputType,
                    first.Axis,
                    first.UseMean)
                .InheritMetaData(producer),
                $"forming PackedMatMulNormStatsCombine in {function.Name}");
            var value = AssertValidExpr(
                IR.F.Tensors.GetItem(combine, 0).InheritMetaData(producer),
                $"selecting PackedMatMulNormStatsCombine value in {function.Name}");
            var stats = AssertValidExpr(
                IR.F.Tensors.GetItem(combine, 1),
                $"selecting PackedMatMulNormStatsCombine statistics in {function.Name}");
            replacements.Add(producer, value);
            foreach (var candidate in candidates)
            {
                replacements.Add(candidate.NormStats, stats.InheritMetaData(candidate.NormStats));
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
                $"PackedMatMulNormStatsCombine formation could not infer function {function.Name}.");
        }

        if (rewritten.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException(
                $"PackedMatMulNormStatsCombine formation produced an invalid function {function.Name}: {invalid}.");
        }

        return Task.FromResult(rewritten);
    }

    private static Dictionary<Call, List<Candidate>> CollectCandidates(Function function)
    {
        var result = new Dictionary<Call, List<Candidate>>(ReferenceEqualityComparer.Instance);
        foreach (var statsCall in ExprCollector.Collect(function.Body)
                     .OfType<Call>()
                     .Where(call => call.Target is NormStats))
        {
            var stats = (NormStats)statsCall.Target;
            if (statsCall[NormStats.Input] is not Call producer ||
                producer.Target is not Op producerTarget ||
                !IsPartialCapableProducer(producerTarget) ||
                producer.CheckedType is not (TensorType or DistributedType) ||
                !TryNormalizeAxis(stats.Axis, producer.CheckedShape.Rank, out var axis) ||
                axis != producer.CheckedShape.Rank - 1)
            {
                continue;
            }

            if (!result.TryGetValue(producer, out var candidates))
            {
                candidates = new List<Candidate>();
                result.Add(producer, candidates);
            }

            candidates.Add(new Candidate(statsCall, axis, stats.UseMean));
        }

        return result;
    }

    private static bool IsPartialCapableProducer(Op target) => target switch
    {
        PackedMatMul { FusedReduce: false } => true,
        PackedBlockScaledMatMul => true,
        _ => false,
    };

    private static Expr GetAddend(Call producer) => producer.Target switch
    {
        PackedMatMul => (Expr)producer[PackedMatMul.Addend],
        PackedBlockScaledMatMul => (Expr)producer[PackedBlockScaledMatMul.Addend],
        _ => throw new InvalidOperationException(
            $"Unsupported partial-capable packed matmul {producer.Target.GetType().Name}."),
    };

    private static Expr CreatePartialCapableProducer(Call producer) => producer.Target switch
    {
        PackedMatMul packed => IR.F.NTT.PackedMatMul(
            (Expr)producer[PackedMatMul.Lhs],
            (Expr)producer[PackedMatMul.Rhs],
            fusedReduce: false,
            outDataType: packed.OutputDataType,
            scale: (Expr)producer[PackedMatMul.Scale],
            rhsLayout: packed.RhsLayout),
        PackedBlockScaledMatMul packed => IR.F.NTT.PackedBlockScaledMatMul(
            (Expr)producer[PackedBlockScaledMatMul.Lhs],
            (Expr)producer[PackedBlockScaledMatMul.Rhs],
            (Expr)producer[PackedBlockScaledMatMul.RhsScale],
            packed.OutputDataType,
            packed.WeightBlockN,
            packed.WeightBlockK,
            packed.RhsLayout,
            packed.OutputNVectorLaneCount),
        _ => throw new InvalidOperationException(
            $"Unsupported partial-capable packed matmul {producer.Target.GetType().Name}."),
    };

    private static bool TryNormalizeAxis(int axis, int rank, out int normalizedAxis)
    {
        normalizedAxis = axis < 0 ? axis + rank : axis;
        return normalizedAxis >= 0 && normalizedAxis < rank;
    }

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
        private readonly HashSet<BaseExpr> _active = new(ReferenceEqualityComparer.Instance);

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

            if (!_active.Add(expr))
            {
                throw new InvalidOperationException(
                    $"PackedMatMulNormStatsCombine replacement graph contains a cycle at {expr}.");
            }

            try
            {
                return Visit(replacement, context);
            }
            finally
            {
                _active.Remove(expr);
            }
        }
    }
}
