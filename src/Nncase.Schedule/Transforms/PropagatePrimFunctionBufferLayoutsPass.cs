// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Propagates compatible physical layouts across PrimFunction input
/// boundaries before residual layout specialization.
/// </summary>
public sealed class PropagatePrimFunctionBufferLayoutsPass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var functions = input.Functions.OfType<PrimFunction>().ToArray();
        var callSites = functions
            .SelectMany(function => CollectDirectCalls(function)
                .Select(call => new CallSite(function, call)))
            .ToArray();
        var plans = new Dictionary<TIR.Buffer, ForwardingPlan>(
            ReferenceEqualityComparer.Instance);
        var conflicts = new HashSet<TIR.Buffer>(ReferenceEqualityComparer.Instance);

        foreach (var calleeSites in callSites.GroupBy(site => (PrimFunction)site.Call.Target))
        {
            var callee = calleeSites.Key;
            var parameters = callee.Parameters.ToArray();
            var sites = calleeSites.ToArray();
            for (var parameterIndex = 0; parameterIndex < parameters.Length; parameterIndex++)
            {
                if (parameters[parameterIndex] is not BufferVar
                    {
                        Role: BufferVarRole.Input or BufferVarRole.InOut,
                    })
                {
                    continue;
                }

                var actuals = sites
                    .Select(site => new ActualBuffer(
                        site.Caller,
                        AssertBufferArgument(callee, site.Call, parameterIndex)))
                    .ToArray();
                var compactLayouts = actuals
                    .Select(actual => GetExactLayout(actual.Buffer))
                    .Where(layout => layout.DistributedStorageKind ==
                        DistributedBufferStorageKind.CompactPerOwner)
                    .Distinct()
                    .ToArray();
                if (compactLayouts.Length != 1)
                {
                    continue;
                }

                var targetLayout = compactLayouts[0];
                if (!actuals.All(actual =>
                        GetExactLayout(actual.Buffer) == targetLayout ||
                        CanForwardToCompactPerOwner(actual.Buffer, targetLayout)))
                {
                    continue;
                }

                foreach (var actual in actuals.Where(actual =>
                             GetExactLayout(actual.Buffer) != targetLayout))
                {
                    AddPlan(actual, targetLayout, plans, conflicts);
                }
            }
        }

        var mutatedFunctions = new HashSet<PrimFunction>(ReferenceEqualityComparer.Instance);
        foreach (var (source, plan) in plans)
        {
            if (conflicts.Contains(source))
            {
                continue;
            }

            var replacement = ForwardToCompactPerOwner(source, plan.Layout);
            ReplaceUtility.ReplaceAllUsesWith(source, replacement);
            mutatedFunctions.UnionWith(plan.Callers);
        }

        foreach (var function in mutatedFunctions)
        {
            if (!CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after propagating PrimFunction buffer layouts in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private static IEnumerable<Call> CollectDirectCalls(PrimFunction function)
    {
        var calls = new List<Call>();
        var collector = new DirectPrimFunctionCallCollector(calls);
        collector.Visit(function.Body);
        collector.Visit(function.Results);
        return calls;
    }

    private static TIR.Buffer AssertBufferArgument(
        PrimFunction callee,
        Call call,
        int parameterIndex)
    {
        if (parameterIndex >= call.Arguments.Length)
        {
            throw new InvalidOperationException(
                $"PrimFunction call ABI mismatch for {callee.Name}: parameter {parameterIndex} has no argument.");
        }

        return call.Arguments[parameterIndex] as TIR.Buffer
            ?? throw new InvalidOperationException(
                $"PrimFunction {callee.Name} tensor parameter {parameterIndex} expects a TIR.Buffer argument, " +
                $"got {call.Arguments[parameterIndex].GetType().Name}.");
    }

    private static BufferLayoutAnnotation GetExactLayout(TIR.Buffer buffer)
        => BufferLayoutAnnotation.ExactStrided(
            buffer.Strides,
            buffer.DistributedStorageKind);

    private static bool CanForwardToCompactPerOwner(
        TIR.Buffer source,
        BufferLayoutAnnotation targetLayout)
    {
        if (targetLayout.Kind != BufferLayoutKind.ExactStrided ||
            targetLayout.DistributedStorageKind !=
                DistributedBufferStorageKind.CompactPerOwner ||
            source.DistributedType is not { Partial: null } distributedType ||
            !DistributedUtility.IsFullyShardedAcrossPlacement(distributedType) ||
            source.DistributedStorageKind != DistributedBufferStorageKind.CompactLocal ||
            source.MemSpan.Buffer.Location != MemoryLocation.Data ||
            source.MemSpan.Buffer.Start is not None ||
            !source.MemSpan.Start.Simplify().Equals(Dimension.Zero) ||
            !source.MemSpan.Size.Simplify().Equals(source.MemSpan.Buffer.Size.Simplify()) ||
            !source.Strides.SequenceEqual(targetLayout.Strides))
        {
            return false;
        }

        var aliases = source.MemSpan.Buffer.Users
            .OfType<MemSpan>()
            .SelectMany(span => span.Users.OfType<TIR.Buffer>())
            .Distinct(ReferenceEqualityComparer.Instance)
            .ToArray();
        return aliases.Length == 1 && ReferenceEquals(aliases[0], source);
    }

    private static TIR.Buffer ForwardToCompactPerOwner(
        TIR.Buffer source,
        BufferLayoutAnnotation targetLayout)
    {
        if (!CanForwardToCompactPerOwner(source, targetLayout))
        {
            throw new InvalidOperationException(
                $"Buffer {source.Name} is no longer eligible for compact-per-owner layout propagation.");
        }

        var distributedType = source.DistributedType!;
        var ownerCount = distributedType.Placement.Hierarchy.Aggregate(
            1L,
            (product, extent) => checked(product * extent));
        var physicalBuffer = new PhysicalBuffer(
            source.MemSpan.Buffer.Alignment,
            (source.MemSpan.Size * ownerCount).Simplify(),
            MemoryLocation.ChipLocalData);
        return source.With(
            name: $"{source.Name}_compact_per_owner",
            memSpan: source.MemSpan.With(buffer: physicalBuffer),
            distributedStorageKind: DistributedBufferStorageKind.CompactPerOwner);
    }

    private static void AddPlan(
        ActualBuffer actual,
        BufferLayoutAnnotation targetLayout,
        IDictionary<TIR.Buffer, ForwardingPlan> plans,
        ISet<TIR.Buffer> conflicts)
    {
        if (!plans.TryGetValue(actual.Buffer, out var existing))
        {
            plans.Add(
                actual.Buffer,
                new ForwardingPlan(
                    targetLayout,
                    new HashSet<PrimFunction>(
                        [actual.Caller],
                        ReferenceEqualityComparer.Instance)));
            return;
        }

        if (existing.Layout != targetLayout)
        {
            conflicts.Add(actual.Buffer);
            return;
        }

        existing.Callers.Add(actual.Caller);
    }

    private sealed class DirectPrimFunctionCallCollector : ExprWalker
    {
        private readonly ICollection<Call> _calls;

        public DirectPrimFunctionCallCollector(ICollection<Call> calls)
            : base(visitOtherFunctions: false)
        {
            _calls = calls;
        }

        protected override Unit VisitLeafCall(Call expr)
        {
            if (expr.Target is PrimFunction)
            {
                _calls.Add(expr);
            }

            return base.VisitLeafCall(expr);
        }
    }

    private sealed record CallSite(PrimFunction Caller, Call Call);

    private sealed record ActualBuffer(PrimFunction Caller, TIR.Buffer Buffer);

    private sealed record ForwardingPlan(
        BufferLayoutAnnotation Layout,
        HashSet<PrimFunction> Callers);
}
