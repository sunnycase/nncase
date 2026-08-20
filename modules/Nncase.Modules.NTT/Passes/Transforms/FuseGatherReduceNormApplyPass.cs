// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.Passes.Mutators;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses a single-use partial-to-broadcast statistics materialization with its
/// immediately following normalization apply operation.
/// </summary>
public sealed class FuseGatherReduceNormApplyPass : ModulePass
{
    private readonly string _moduleKind;

    public FuseGatherReduceNormApplyPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var function in input.Functions.OfType<PrimFunction>().Where(function => function.ModuleKind == _moduleKind))
        {
            var useCounts = CollectBufferCallUseCounts(function.Body);
            var rewriter = new FusionRewriter(useCounts);
            rewriter.Rewrite(function);
            if (rewriter.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after fusing GatherReduceScatter with NormApply in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private static IReadOnlyDictionary<TIR.Buffer, int> CollectBufferCallUseCounts(BaseExpr body)
    {
        var counts = new Dictionary<TIR.Buffer, int>(ReferenceEqualityComparer.Instance);
        foreach (var call in ExprCollector.Collect(body).OfType<Call>())
        {
            foreach (var buffer in call.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                counts.TryGetValue(buffer, out var count);
                counts[buffer] = count + 1;
            }
        }

        return counts;
    }

    private sealed class FusionRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<TIR.Buffer, int> _useCounts;

        public FusionRewriter(IReadOnlyDictionary<TIR.Buffer, int> useCounts)
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

        private static bool CanFuseCollective(
            TIR.NTT.GatherReduceScatter gather,
            TIR.Buffer partialStats,
            TIR.Buffer broadcastStats)
        {
            if (gather.InType.Partial is not { Op: ReduceOp.Sum } partial ||
                partial.Axes.Count == 0 ||
                gather.OutType.Partial is not null ||
                gather.InType.AxisPolicies.Any(policy => policy is not SBPBroadCast) ||
                gather.OutType.AxisPolicies.Any(policy => policy is not SBPBroadCast))
            {
                return false;
            }

            if (!gather.InType.Placement.Equals(gather.OutType.Placement) ||
                !gather.InType.TensorType.Equals(gather.OutType.TensorType))
            {
                return false;
            }

            if (partialStats.DistributedStorageKind != DistributedBufferStorageKind.CompactPerOwner ||
                broadcastStats.DistributedStorageKind != DistributedBufferStorageKind.CompactPerOwner)
            {
                return false;
            }

            return true;
        }

        private bool TryFuse(Expr producer, Expr consumer, out Expr fused)
        {
            fused = null!;
            if (producer is not Call { Target: TIR.NTT.GatherReduceScatter gather } gatherCall ||
                consumer is not Call { Target: TIR.NTT.NormApply normApply } normApplyCall)
            {
                return false;
            }

            var gatherArguments = gatherCall.Arguments.ToArray();
            var normArguments = normApplyCall.Arguments.ToArray();
            if (gatherArguments.Length < 2 || normArguments.Length < 5 ||
                gatherArguments[0] is not TIR.Buffer partialStats ||
                gatherArguments[1] is not TIR.Buffer broadcastStats ||
                !ReferenceEquals(normArguments[1], broadcastStats) ||
                !_useCounts.TryGetValue(broadcastStats, out var useCount) ||
                useCount != 2)
            {
                return false;
            }

            if (!CanFuseCollective(gather, partialStats, broadcastStats))
            {
                return false;
            }

            fused = TIR.F.NTT.GatherReduceNormApply(
                    partialStats,
                    (Expr)normArguments[0],
                    (Expr)normArguments[2],
                    (Expr)normArguments[3],
                    (Expr)normArguments[4],
                    gather.InType,
                    gather.OutType,
                    normApply.Axis,
                    normApply.Epsilon,
                    normApply.UseMean,
                    HasNonZeroBias((TIR.Buffer)normArguments[3]))
                .InheritMetaData(normApplyCall);
            return true;
        }

        private static bool HasNonZeroBias(TIR.Buffer bias)
        {
            if (bias.MemSpan.Buffer.Start is not Call { Target: AddressOf } addressOf ||
                addressOf[AddressOf.Input] is not TensorConst tensor)
            {
                return true;
            }

            return tensor.Value.BytesBuffer.IndexOfAnyExcept((byte)0) >= 0;
        }
    }
}
