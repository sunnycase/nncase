// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
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
        var counts = new Dictionary<TIR.Buffer, int>(ExactBufferRegionComparer.Instance);
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

    private sealed class ExactBufferRegionComparer : IEqualityComparer<TIR.Buffer>
    {
        public static readonly ExactBufferRegionComparer Instance = new();

        public bool Equals(TIR.Buffer? lhs, TIR.Buffer? rhs)
        {
            if (ReferenceEquals(lhs, rhs))
            {
                return true;
            }

            return lhs is not null &&
                rhs is not null &&
                HasSameBacking(lhs.MemSpan.Buffer, rhs.MemSpan.Buffer) &&
                lhs.MemSpan.Start.Equals(rhs.MemSpan.Start) &&
                lhs.MemSpan.Size.Equals(rhs.MemSpan.Size);
        }

        public int GetHashCode(TIR.Buffer buffer)
        {
            var physical = buffer.MemSpan.Buffer;
            var backingHash = HasStableBackingAddress(physical)
                ? HashCode.Combine(
                    physical.Location,
                    physical.Hierarchy,
                    physical.Start,
                    physical.Size,
                    physical.BlockLocalRDataMaterialization is null
                        ? 0
                        : RuntimeHelpers.GetHashCode(physical.BlockLocalRDataMaterialization))
                : RuntimeHelpers.GetHashCode(physical);
            return HashCode.Combine(backingHash, buffer.MemSpan.Start, buffer.MemSpan.Size);
        }

        private static bool HasSameBacking(PhysicalBuffer lhs, PhysicalBuffer rhs)
        {
            if (ReferenceEquals(lhs, rhs))
            {
                return true;
            }

            return HasStableBackingAddress(lhs) &&
                HasStableBackingAddress(rhs) &&
                lhs.Location == rhs.Location &&
                lhs.Hierarchy == rhs.Hierarchy &&
                ReferenceEquals(lhs.BlockLocalRDataMaterialization, rhs.BlockLocalRDataMaterialization) &&
                lhs.Start.Equals(rhs.Start) &&
                lhs.Size.Equals(rhs.Size);
        }

        private static bool HasStableBackingAddress(PhysicalBuffer buffer)
            => buffer.Start is not None and not Var;
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
            var mutated = false;
            for (var index = 0; index < source.Length; index++)
            {
                var current = source[index];
                while (true)
                {
                    var consumerIndex = index + 1;
                    while (consumerIndex < source.Length && IsNop(source[consumerIndex]))
                    {
                        consumerIndex++;
                    }

                    if (consumerIndex >= source.Length ||
                        !TryExtractTrailingGather(current, out var gatherCall, out var producerPrefix) ||
                        !TryGetLeadingNormApply(source[consumerIndex], out var normApplyCall) ||
                        !TryFuse(gatherCall, normApplyCall, out var fused))
                    {
                        break;
                    }

                    if (producerPrefix is not null)
                    {
                        result.Add(producerPrefix);
                    }

                    current = ReplaceLeadingNormApply(source[consumerIndex], fused);
                    index = consumerIndex;
                    mutated = true;
                    SetMutated();
                }

                result.Add(current);
            }

            return mutated ? expr.With(fields: result.ToArray()) : expr;
        }

        private static bool IsNop(Expr expression)
            => expression is Call { Target: Nop };

        private static bool TryExtractTrailingGather(
            Expr expression,
            out Call gatherCall,
            out Expr? prefix)
        {
            if (expression is Call { Target: TIR.NTT.GatherReduceScatter } directGather)
            {
                gatherCall = directGather;
                prefix = null;
                return true;
            }

            if (expression is not Sequential sequential)
            {
                gatherCall = null!;
                prefix = null;
                return false;
            }

            var fields = sequential.Fields.ToArray();
            var producerIndex = fields.Length - 1;
            while (producerIndex >= 0 && IsNop(fields[producerIndex]))
            {
                producerIndex--;
            }

            gatherCall = null!;
            if (producerIndex < 0 ||
                !TryExtractTrailingGather(fields[producerIndex], out gatherCall, out var nestedPrefix))
            {
                prefix = null;
                return false;
            }

            var prefixFields = new List<Expr>(fields.Length);
            prefixFields.AddRange(fields.Take(producerIndex));
            if (nestedPrefix is not null)
            {
                prefixFields.Add(nestedPrefix);
            }

            prefixFields.AddRange(fields.Skip(producerIndex + 1));
            prefix = sequential.With(fields: prefixFields.ToArray());
            return true;
        }

        private static bool TryGetLeadingNormApply(Expr expression, out Call normApplyCall)
        {
            if (expression is Call { Target: TIR.NTT.NormApply } directNormApply)
            {
                normApplyCall = directNormApply;
                return true;
            }

            if (expression is Sequential sequential)
            {
                foreach (var field in sequential.Fields)
                {
                    if (IsNop(field))
                    {
                        continue;
                    }

                    return TryGetLeadingNormApply(field, out normApplyCall);
                }
            }

            normApplyCall = null!;
            return false;
        }

        private static Expr ReplaceLeadingNormApply(Expr expression, Expr replacement)
        {
            if (expression is Call { Target: TIR.NTT.NormApply })
            {
                return replacement;
            }

            if (expression is not Sequential sequential)
            {
                throw new InvalidOperationException(
                    $"Expected a leading NormApply in {expression.GetType().Name}.");
            }

            var fields = sequential.Fields.ToArray();
            for (var index = 0; index < fields.Length; index++)
            {
                if (IsNop(fields[index]))
                {
                    continue;
                }

                fields[index] = ReplaceLeadingNormApply(fields[index], replacement);
                return sequential.With(fields: fields);
            }

            throw new InvalidOperationException("Expected a leading NormApply in a non-empty Sequential.");
        }

        private static bool CanFuseCollective(
            TIR.NTT.GatherReduceScatter gather,
            TIR.Buffer partialStats)
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

            if (partialStats.DistributedStorageKind != DistributedBufferStorageKind.CompactPerOwner)
            {
                return false;
            }

            return true;
        }

        private bool TryFuse(Call gatherCall, Call normApplyCall, out Expr fused)
        {
            fused = null!;
            if (gatherCall.Target is not TIR.NTT.GatherReduceScatter gather ||
                normApplyCall.Target is not TIR.NTT.NormApply normApply)
            {
                return false;
            }

            var gatherArguments = gatherCall.Arguments.ToArray();
            var normArguments = normApplyCall.Arguments.ToArray();
            if (gatherArguments.Length < 2 || normArguments.Length < 5 ||
                gatherArguments[0] is not TIR.Buffer partialStats ||
                gatherArguments[1] is not TIR.Buffer broadcastStats ||
                normArguments[1] is not TIR.Buffer normStats ||
                !ExactBufferRegionComparer.Instance.Equals(normStats, broadcastStats) ||
                !_useCounts.TryGetValue(broadcastStats, out var useCount) ||
                useCount != 2)
            {
                return false;
            }

            if (!CanFuseCollective(gather, partialStats))
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
