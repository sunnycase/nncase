// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses a single-use normalization-statistics publication into the collective
/// that produces the residual value consumed by NormApply.
/// </summary>
public sealed class FuseGatherReduceAddNormApplyPass : ModulePass
{
    private readonly string _moduleKind;

    public FuseGatherReduceAddNormApplyPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var function in input.Functions.OfType<PrimFunction>().Where(function => function.ModuleKind == _moduleKind))
        {
            var useCounts = CollectPhysicalBufferCallUseCounts(function.Body);
            var rewriter = new FusionRewriter(useCounts);
            rewriter.Rewrite(function);
            if (rewriter.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after fusing GatherReduceAddNormStats with NormApply in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private static IReadOnlyDictionary<PhysicalBuffer, int> CollectPhysicalBufferCallUseCounts(BaseExpr body)
    {
        var counts = new Dictionary<PhysicalBuffer, int>(ReferenceEqualityComparer.Instance);
        foreach (var call in ExprCollector.Collect(body).OfType<Call>())
        {
            foreach (var buffer in call.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                counts.TryGetValue(buffer.MemSpan.Buffer, out var count);
                counts[buffer.MemSpan.Buffer] = count + 1;
            }
        }

        return counts;
    }

    private sealed class FusionRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<PhysicalBuffer, int> _useCounts;

        public FusionRewriter(IReadOnlyDictionary<PhysicalBuffer, int> useCounts)
            : base(visitOtherFunctions: false)
        {
            _useCounts = useCounts;
        }

        protected override BaseExpr RewriteLeafSequential(Sequential expr)
        {
            var source = expr.Fields.ToArray();
            var result = new List<Expr>(source.Length);
            for (var index = 0; index < source.Length; index++)
            {
                var consumerIndex = index + 1;
                while (consumerIndex < source.Length && IsNop(source[consumerIndex]))
                {
                    consumerIndex++;
                }

                if (consumerIndex < source.Length &&
                    TryFuse(source[index], source[consumerIndex], out var fused))
                {
                    result.Add(fused);
                    index = consumerIndex;
                    SetMutated();
                    continue;
                }

                result.Add(source[index]);
            }

            return result.Count == source.Length
                ? expr
                : expr.With(fields: result.ToArray());
        }

        private static bool IsNop(Expr expression)
            => expression is Call { Target: Nop };

        private bool TryFuse(Expr producer, Expr consumer, out Expr fused)
        {
            fused = null!;
            if (producer is not Call { Target: TIR.NTT.GatherReduceAddNormStats gather } gatherCall ||
                consumer is not Call { Target: TIR.NTT.NormApply norm } normCall)
            {
                return false;
            }

            var gatherArguments = gatherCall.Arguments.ToArray();
            var normArguments = normCall.Arguments.ToArray();
            if (gatherArguments.Length < 5 ||
                normArguments.Length < 5 ||
                gatherArguments[3] is not TIR.Buffer valueOutput ||
                gatherArguments[4] is not TIR.Buffer statsOutput ||
                normArguments[0] is not TIR.Buffer normInput ||
                normArguments[1] is not TIR.Buffer normStats ||
                normArguments[4] is not TIR.Buffer { DistributedType: { Partial: null } normOutputType } ||
                !ReferenceEquals(valueOutput.MemSpan.Buffer, normInput.MemSpan.Buffer) ||
                !ReferenceEquals(statsOutput.MemSpan.Buffer, normStats.MemSpan.Buffer) ||
                !_useCounts.TryGetValue(statsOutput.MemSpan.Buffer, out var statsUseCount) ||
                statsUseCount != 2 ||
                gather.Axis != norm.Axis ||
                gather.UseMean != norm.UseMean)
            {
                return false;
            }

            fused = TIR.F.NTT.GatherReduceAddNormApply(
                    (Expr)gatherArguments[0],
                    (Expr)gatherArguments[1],
                    (Expr)gatherArguments[2],
                    valueOutput,
                    statsOutput,
                    (Expr)normArguments[2],
                    (Expr)normArguments[3],
                    (Expr)normArguments[4],
                    gather.InType,
                    gather.OutType,
                    normOutputType,
                    gather.Axis,
                    norm.Epsilon,
                    gather.UseMean)
                .InheritMetaData(normCall);
            return true;
        }
    }
}
