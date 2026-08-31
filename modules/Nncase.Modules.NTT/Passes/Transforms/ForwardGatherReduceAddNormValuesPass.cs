// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Reuses the canonical gather-reduce collective for its materialized residual value.
/// </summary>
public sealed class ForwardGatherReduceAddNormValuesPass : ModulePass
{
    private readonly string _moduleKind;

    public ForwardGatherReduceAddNormValuesPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var function in input.Functions
                     .OfType<PrimFunction>()
                     .Where(function => function.ModuleKind == _moduleKind))
        {
            var expressions = ExprCollector.Collect(function).ToArray();
            var plans = BuildPlans(expressions);
            if (plans.Count == 0)
            {
                continue;
            }

            var rewriter = new ValueForwardingRewriter(plans);
            rewriter.Rewrite(function);
            if (rewriter.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after forwarding gather-reduce values in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private static IReadOnlyDictionary<PhysicalBuffer, TIR.Buffer> BuildPlans(
        IReadOnlyList<BaseExpr> expressions)
    {
        var buffers = expressions
            .OfType<TIR.Buffer>()
            .Distinct(new ReferenceEqualityComparer<TIR.Buffer>())
            .ToArray();
        var calls = expressions.OfType<Call>().Where(call => call.Target is Op).ToArray();
        var plans = new Dictionary<PhysicalBuffer, TIR.Buffer>(
            new ReferenceEqualityComparer<PhysicalBuffer>());

        foreach (var call in calls.Where(call =>
                     call.Target is TIR.NTT.GatherReduceAddNormStats or
                         TIR.NTT.GatherReduceAddNormApply))
        {
            var arguments = call.Arguments.ToArray();
            if (arguments.Length < 4 ||
                arguments[1] is not TIR.Buffer collective ||
                arguments[3] is not TIR.Buffer value ||
                !CanForward(call, collective, value, buffers, calls))
            {
                continue;
            }

            if (!plans.TryAdd(value.MemSpan.Buffer, collective))
            {
                throw new InvalidOperationException(
                    $"Gather-reduce value backing {value.Name} has multiple canonical forwarding targets.");
            }
        }

        return plans;
    }

    private static bool CanForward(
        Call writer,
        TIR.Buffer collective,
        TIR.Buffer value,
        IReadOnlyList<TIR.Buffer> buffers,
        IReadOnlyList<Call> calls)
    {
        if (collective.DistributedStorageKind != DistributedBufferStorageKind.CanonicalGlobal ||
            collective.DistributedType is not { Partial: null } collectiveType ||
            value.DistributedType is not { Partial: null } valueType ||
            !collectiveType.Equals(valueType) ||
            !collective.ElemType.Equals(value.ElemType) ||
            !collective.Dimensions.SequenceEqual(value.Dimensions) ||
            !collective.Strides.SequenceEqual(value.Strides) ||
            collective.StorageEncoding is not null ||
            collective.StagedLayout is not null ||
            value.StorageEncoding is not null ||
            value.StagedLayout is not null ||
            !collective.MemSpan.Start.Equals(value.MemSpan.Start) ||
            !collective.MemSpan.Size.Equals(value.MemSpan.Size))
        {
            return false;
        }

        var aliases = buffers
            .Where(buffer => ReferenceEquals(buffer.MemSpan.Buffer, value.MemSpan.Buffer))
            .ToArray();
        if (aliases.Length == 0 || aliases.Any(alias =>
                !alias.MemSpan.Start.Equals(value.MemSpan.Start) ||
                !alias.MemSpan.Size.Equals(value.MemSpan.Size) ||
                !alias.ElemType.Equals(value.ElemType) ||
                !alias.Dimensions.SequenceEqual(value.Dimensions) ||
                !alias.Strides.SequenceEqual(value.Strides)))
        {
            return false;
        }

        var writerCount = 0;
        foreach (var call in calls)
        {
            MemoryEffectUtility.VisitCallEffects(call, (argument, _, argumentIndex, effect) =>
            {
                if (argument is not TIR.Buffer buffer ||
                    !ReferenceEquals(buffer.MemSpan.Buffer, value.MemSpan.Buffer))
                {
                    return;
                }

                var mode = MemoryEffectUtility.GetPhysicalBufferAccessMode(effect);
                if (mode == MemoryAccessMode.Write)
                {
                    if (ReferenceEquals(call, writer) && argumentIndex == 3 &&
                        effect.Kind == MemoryEffectKind.Direct)
                    {
                        writerCount++;
                    }
                    else
                    {
                        writerCount = int.MinValue;
                    }
                }
            });
        }

        return writerCount == 1;
    }

    private sealed class ValueForwardingRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<PhysicalBuffer, TIR.Buffer> _plans;

        public ValueForwardingRewriter(
            IReadOnlyDictionary<PhysicalBuffer, TIR.Buffer> plans)
            : base(visitOtherFunctions: false)
        {
            _plans = plans;
        }

        protected override BaseExpr RewriteLeafBuffer(TIR.Buffer expr)
        {
            if (!_plans.TryGetValue(expr.MemSpan.Buffer, out var collective))
            {
                return expr;
            }

            SetMutated();
            return expr.With(
                memSpan: collective.MemSpan,
                distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
        }
    }
}
