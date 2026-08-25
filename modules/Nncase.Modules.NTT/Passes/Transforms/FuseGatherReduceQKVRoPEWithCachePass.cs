// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses three single-use partial Q/K/V materializations into the immediately
/// following QKV normalization, RoPE, and cache update.
/// </summary>
public sealed class FuseGatherReduceQKVRoPEWithCachePass : ModulePass
{
    private readonly string _moduleKind;

    public FuseGatherReduceQKVRoPEWithCachePass(string moduleKind)
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
                    $"Type inference failed after fusing partial QKV combine with QKVRoPEWithCache in {function.Name}.");
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
                var storage = buffer.MemSpan.Buffer;
                counts.TryGetValue(storage, out var count);
                counts[storage] = count + 1;
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
            var removed = new HashSet<int>();
            var replacements = new Dictionary<int, Expr>();
            for (var consumerIndex = 0; consumerIndex < source.Length; consumerIndex++)
            {
                if (source[consumerIndex] is not Call { Target: TIR.NTT.QKVRoPEWithCache } consumer)
                {
                    continue;
                }

                if (!TryGetProducerGroup(
                        source,
                        consumerIndex,
                        out var producerIndices,
                        out var producers) ||
                    !TryFuse(
                        producers,
                        consumer,
                        out var fused))
                {
                    continue;
                }

                foreach (var producerIndex in producerIndices)
                {
                    removed.Add(producerIndex);
                }

                replacements.Add(consumerIndex, fused);
                SetMutated();
            }

            if (replacements.Count == 0)
            {
                return expr;
            }

            var result = new List<Expr>(source.Length - removed.Count);
            for (var index = 0; index < source.Length; index++)
            {
                if (removed.Contains(index))
                {
                    continue;
                }

                result.Add(replacements.TryGetValue(index, out var replacement) ? replacement : source[index]);
            }

            return expr.With(fields: result.ToArray());
        }

        private static bool TryGetProducerGroup(
            Expr[] source,
            int consumerIndex,
            out int[] producerIndices,
            out Expr[] producers)
        {
            producerIndices = PreviousExecutableIndices(source, consumerIndex, 1);
            producers = Array.Empty<Expr>();
            if (producerIndices.Length != 1)
            {
                return false;
            }

            if (source[producerIndices[0]] is Sequential group)
            {
                producers = group.Fields.ToArray()
                    .Where(field => field is not Call { Target: Nop })
                    .ToArray();
                return producers.Length == 3;
            }

            producerIndices = PreviousExecutableIndices(source, consumerIndex, 3);
            if (producerIndices.Length != 3)
            {
                return false;
            }

            producers = producerIndices.Select(index => source[index]).ToArray();
            return true;
        }

        private static int[] PreviousExecutableIndices(Expr[] source, int end, int count)
        {
            var indices = new List<int>(count);
            for (var index = end - 1; index >= 0 && indices.Count < count; index--)
            {
                if (source[index] is Call { Target: Nop })
                {
                    continue;
                }

                indices.Add(index);
            }

            indices.Reverse();
            return indices.ToArray();
        }

        private bool TryFuse(
            IReadOnlyList<Expr> producers,
            Call consumerCall,
            out Expr fused)
        {
            fused = null!;
            if (producers.Count != 3 ||
                producers.Any(producer => producer is not Call { Target: TIR.NTT.GatherReduceScatter }) ||
                consumerCall.Target is not TIR.NTT.QKVRoPEWithCache consumer)
            {
                return false;
            }

            var consumerArguments = consumerCall.Arguments.ToArray();
            if (consumerArguments.Length < 12 ||
                consumerArguments[0] is not TIR.Buffer qView ||
                consumerArguments[1] is not TIR.Buffer kView ||
                consumerArguments[2] is not TIR.Buffer vView)
            {
                return false;
            }

            var gatherByOutput = new Dictionary<PhysicalBuffer, Call>(ReferenceEqualityComparer.Instance);
            foreach (var producer in producers.Cast<Call>())
            {
                var arguments = producer.Arguments.ToArray();
                if (arguments.Length < 2 ||
                    arguments[0] is not TIR.Buffer ||
                    arguments[1] is not TIR.Buffer output ||
                    !gatherByOutput.TryAdd(output.MemSpan.Buffer, producer))
                {
                    return false;
                }
            }

            if (!TryGetSingleUseGather(qView, gatherByOutput, out var qGather, out var qInput) ||
                !TryGetSingleUseGather(kView, gatherByOutput, out var kGather, out var kInput) ||
                !TryGetSingleUseGather(vView, gatherByOutput, out var vGather, out var vInput) ||
                qView.DistributedType is not { } qLogicalType ||
                kView.DistributedType is not { } kLogicalType ||
                vView.DistributedType is not { } vLogicalType ||
                !CanFuseCollective(qGather, qInput) ||
                !CanFuseCollective(kGather, kInput) ||
                !CanFuseCollective(vGather, vInput))
            {
                return false;
            }

            fused = TIR.F.NTT.GatherReduceQKVRoPEWithCache(
                    qInput,
                    kInput,
                    vInput,
                    (Expr)consumerArguments[3],
                    (Expr)consumerArguments[4],
                    (Expr)consumerArguments[5],
                    (Expr)consumerArguments[6],
                    (Expr)consumerArguments[7],
                    (Expr)consumerArguments[8],
                    (Expr)consumerArguments[9],
                    (Dimension)consumerArguments[10],
                    (Expr)consumerArguments[11],
                    qGather.InType,
                    qLogicalType,
                    kGather.InType,
                    kLogicalType,
                    vGather.InType,
                    vLogicalType,
                    qView.Dimensions.ToArray(),
                    qView.Strides.ToArray(),
                    kView.Dimensions.ToArray(),
                    kView.Strides.ToArray(),
                    vView.Dimensions.ToArray(),
                    vView.Strides.ToArray(),
                    consumer.QAxis,
                    consumer.QEpsilon,
                    consumer.QUseMean,
                    consumer.KAxis,
                    consumer.KEpsilon,
                    consumer.KUseMean,
                    consumer.QKVLayout,
                    consumer.AttentionLayout)
                .InheritMetaData(consumerCall);
            return true;
        }

        private bool TryGetSingleUseGather(
            TIR.Buffer view,
            IReadOnlyDictionary<PhysicalBuffer, Call> gatherByOutput,
            out TIR.NTT.GatherReduceScatter gather,
            out TIR.Buffer input)
        {
            gather = null!;
            input = null!;
            var storage = view.MemSpan.Buffer;
            if (!gatherByOutput.TryGetValue(storage, out var gatherCall) ||
                !_useCounts.TryGetValue(storage, out var useCount) ||
                useCount != 2 ||
                gatherCall.Target is not TIR.NTT.GatherReduceScatter target ||
                gatherCall.Arguments[0] is not TIR.Buffer source)
            {
                return false;
            }

            gather = target;
            input = source;
            return true;
        }

        private static bool CanFuseCollective(TIR.NTT.GatherReduceScatter gather, TIR.Buffer input)
        {
            if (gather.InType.Partial is not { Op: ReduceOp.Sum } partial ||
                partial.Axes.Count == 0 ||
                gather.OutType.Partial is not null ||
                gather.InType.AxisPolicies.Any(policy => policy is SBPPartial) ||
                gather.OutType.AxisPolicies.Any(policy => policy is SBPPartial) ||
                !gather.InType.Placement.Equals(gather.OutType.Placement) ||
                !gather.InType.TensorType.Equals(gather.OutType.TensorType))
            {
                return false;
            }

            return input.DistributedStorageKind == DistributedBufferStorageKind.CompactPerOwner &&
                input.MemSpan.Buffer.Location == MemoryLocation.ChipLocalData;
        }
    }
}
