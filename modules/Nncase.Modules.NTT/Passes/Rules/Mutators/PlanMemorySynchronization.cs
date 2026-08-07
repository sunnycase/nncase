// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Transforms;
using Nncase.Schedule;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Mutators;

internal readonly record struct MemoryArena(MemoryLocation Location, int Hierarchy);

internal readonly record struct MemoryByteRange(long Start, long End)
{
    public bool Overlaps(MemoryByteRange other) => Start < other.End && other.Start < End;
}

internal readonly record struct EffectInfo(
    MemoryAccessMode Mode,
    TIR.NTT.BarrierScope Scope,
    bool RequiresFullChipSynchronization)
{
    public EffectInfo Merge(EffectInfo other)
        => new(
            Mode | other.Mode,
            MemoryEffectAnalyzer.MergeScope(Scope, other.Scope),
            RequiresFullChipSynchronization || other.RequiresFullChipSynchronization);
}

internal readonly record struct ResolvedMemoryEffect(MemoryResource Resource, EffectInfo Effect);

internal sealed class MemoryEffectAnalyzer
{
    private readonly HashSet<PrimFunction> _functions;
    private readonly Dictionary<PrimFunction, FunctionEffectSummary> _summaries = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<PrimFunction> _active = new(ReferenceEqualityComparer.Instance);

    public MemoryEffectAnalyzer(IEnumerable<PrimFunction> functions)
    {
        _functions = new HashSet<PrimFunction>(functions, ReferenceEqualityComparer.Instance);
    }

    public void AnalyzeAll()
    {
        foreach (var function in _functions)
        {
            _ = GetFunctionSummary(function);
        }
    }

    public EffectSet GetEffects(Expr expr, bool suppressReductionAccumulatorEffects = false)
        => GetEffects(expr, ResourceBindingScope.Empty, suppressReductionAccumulatorEffects, false);

    public EffectSet GetIterationLocalEffects(
        Sequential body,
        bool suppressReductionAccumulatorEffects)
        => GetEffects(body, ResourceBindingScope.Empty, suppressReductionAccumulatorEffects, true);

    internal static MemoryByteRange? TryGetAbsoluteByteRange(MemSpan span)
    {
        if (!TryGetFixedInt64(span.Buffer.Start, out var allocationStart) ||
            !TryGetFixedDimension(span.Start, out var spanStart) ||
            !TryGetFixedDimension(span.Size, out var spanSize) ||
            spanSize < 0)
        {
            return null;
        }

        try
        {
            var start = checked(allocationStart + spanStart);
            return new MemoryByteRange(start, checked(start + spanSize));
        }
        catch (OverflowException)
        {
            return null;
        }
    }

    private EffectSet GetEffects(
        Expr expr,
        ResourceBindingScope bindings,
        bool suppressReductionAccumulatorEffects,
        bool stopAtNestedLoops)
    {
        switch (expr)
        {
            case Block block:
                return Union(
                    [
                        GetEffects(block.InitBody, bindings, suppressReductionAccumulatorEffects, stopAtNestedLoops),
                        GetEffects(block.Body, bindings, suppressReductionAccumulatorEffects, stopAtNestedLoops),
                    ]);
            case Sequential sequential:
                return Union(sequential.Fields.ToArray().Select(
                    field => GetEffects(field, bindings, suppressReductionAccumulatorEffects, stopAtNestedLoops)));
            case PipelineFor pipelineFor:
                if (stopAtNestedLoops)
                {
                    return new EffectSet();
                }

                var pipelineBindings = bindings;
                var pipelineAccesses = pipelineFor.StagedAccesses;
                var pipelineAllocations = pipelineFor.StagedAllocations;
                for (var index = 0; index < pipelineAccesses.Length; index++)
                {
                    pipelineBindings = pipelineBindings.Bind(
                        (BaseExpr)pipelineAccesses[index],
                        pipelineAllocations[index]);
                }

                return Union(
                [
                    GetEffects(pipelineFor.ProduceBody, pipelineBindings, suppressReductionAccumulatorEffects, false),
                    GetEffects(pipelineFor.ConsumeBody, pipelineBindings, suppressReductionAccumulatorEffects, false),
                ]);
            case Nncase.TIR.For @for:
                if (stopAtNestedLoops)
                {
                    return new EffectSet();
                }

                return GetEffects(@for.Body, bindings, suppressReductionAccumulatorEffects, false);
            case Let let:
                var expressionEffects = let.Expression is Expr bindingExpression
                    ? GetEffects(bindingExpression, bindings, suppressReductionAccumulatorEffects, stopAtNestedLoops)
                    : new EffectSet();
                return Union(
                    [
                        expressionEffects,
                        GetEffects(
                            let.Body,
                            bindings.Bind((BaseExpr)let.Var, let.Expression),
                            suppressReductionAccumulatorEffects,
                            stopAtNestedLoops),
                    ]);
            case IfThenElse ifThenElse:
                return Union(
                    [
                        GetEffects(ifThenElse.Then, bindings, suppressReductionAccumulatorEffects, stopAtNestedLoops),
                        GetEffects(ifThenElse.Else, bindings, suppressReductionAccumulatorEffects, stopAtNestedLoops),
                    ]);
            case Call { Target: PrimFunction callee } call when _functions.Contains(callee):
                return Instantiate(GetFunctionSummary(callee), call.Arguments, bindings);
            case Call { Target: PrimFunctionWrapper }:
                throw new InvalidOperationException("PrimFunctionWrapper must be eliminated before memory synchronization planning.");
            case Call { Target: Op } call:
                return GetCallEffects(call, bindings, suppressReductionAccumulatorEffects);
            default:
                return new EffectSet();
        }
    }

    private FunctionEffectSummary GetFunctionSummary(PrimFunction function)
    {
        if (_summaries.TryGetValue(function, out var summary))
        {
            return summary;
        }

        if (!_active.Add(function))
        {
            throw new InvalidOperationException($"Recursive PrimFunction call graph is not supported by memory synchronization planning: {function.Name}.");
        }

        var effects = GetEffects(function.Body, ResourceBindingScope.Empty, false, false);
        var parameterEffects = new Dictionary<int, EffectInfo>();
        foreach (var item in effects.Items)
        {
            if (item.Resource.ExpressionIdentity is not { } identity)
            {
                continue;
            }

            var parameterIndex = FindParameterIndex(function, identity);
            if (parameterIndex < 0)
            {
                continue;
            }

            if (parameterEffects.TryGetValue(parameterIndex, out var existing))
            {
                parameterEffects[parameterIndex] = existing.Merge(item.Effect);
            }
            else
            {
                parameterEffects.Add(parameterIndex, item.Effect);
            }
        }

        _active.Remove(function);
        summary = new FunctionEffectSummary(parameterEffects);
        _summaries.Add(function, summary);
        return summary;
    }

    private static int FindParameterIndex(PrimFunction function, BaseExpr identity)
    {
        for (var index = 0; index < function.Parameters.Length; index++)
        {
            if (ReferenceEquals((BaseExpr)function.Parameters[index], identity))
            {
                return index;
            }
        }

        return -1;
    }

    private static EffectSet GetCallEffects(
        Call call,
        ResourceBindingScope bindings,
        bool suppressReductionAccumulatorEffects)
    {
        var effects = new EffectSet();
        MemoryEffectUtility.VisitCallEffects(
            call,
            (argument, _, effect) =>
            {
                if (suppressReductionAccumulatorEffects &&
                    effect.Kind == MemoryEffectKind.ReductionAccumulator)
                {
                    return;
                }

                var resource = ResolveResource(argument, effect.Scope, bindings);
                effects.Add(
                    resource,
                    new EffectInfo(
                        effect.Mode,
                        resource.Scope,
                        effect.Scope == MemoryAccessScope.Chip));
            });

        return effects;
    }

    private static EffectSet Instantiate(
        FunctionEffectSummary summary,
        ReadOnlySpan<BaseExpr> arguments,
        ResourceBindingScope bindings)
    {
        var effects = new EffectSet();
        foreach (var (parameterIndex, effect) in summary.ParameterEffects)
        {
            if (parameterIndex >= arguments.Length || arguments[parameterIndex] is not Expr argument)
            {
                throw new InvalidOperationException($"Cannot map memory effect for PrimFunction parameter {parameterIndex}.");
            }

            var resource = ResolveResource(argument, MemoryAccessScope.Inferred, bindings);
            effects.Add(
                resource with { Scope = MergeScope(resource.Scope, effect.Scope) },
                effect);
        }

        return effects;
    }

    private static EffectSet Union(IEnumerable<EffectSet> sets)
    {
        var result = new EffectSet();
        foreach (var set in sets)
        {
            result.UnionWith(set);
        }

        return result;
    }

    private static MemoryResource ResolveResource(
        Expr expression,
        MemoryAccessScope synchronizationScope = MemoryAccessScope.Inferred,
        ResourceBindingScope? bindings = null)
        => ResolveResource(
            expression,
            synchronizationScope,
            bindings ?? ResourceBindingScope.Empty,
            new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance));

    private static MemoryResource ResolveResource(
        Expr expression,
        MemoryAccessScope synchronizationScope,
        ResourceBindingScope bindings,
        HashSet<BaseExpr> resolving)
    {
        var explicitScope = synchronizationScope switch
        {
            MemoryAccessScope.Block => TIR.NTT.BarrierScope.Block,
            MemoryAccessScope.Chip => TIR.NTT.BarrierScope.Chip,
            MemoryAccessScope.Inferred => (TIR.NTT.BarrierScope?)null,
            _ => throw new ArgumentOutOfRangeException(nameof(synchronizationScope)),
        };
        if (expression is IVar && bindings.TryGet(expression, out var boundExpression))
        {
            if (!resolving.Add(expression))
            {
                throw new InvalidOperationException($"Cyclic TIR resource alias binding detected at '{((IVar)expression).Name}'.");
            }

            if (boundExpression is not Expr boundResource)
            {
                throw new InvalidOperationException(
                    $"TIR resource alias '{((IVar)expression).Name}' is bound to non-expression {boundExpression.GetType().Name}.");
            }

            var aliasDistributedType = GetDistributedType(expression);
            var resource = ResolveResource(boundResource, synchronizationScope, bindings, resolving);
            resolving.Remove(expression);
            return aliasDistributedType is null
                ? resource
                : resource with { DistributedType = aliasDistributedType };
        }

        switch (expression)
        {
            case Call { Target: IR.Buffers.BufferSubview or IR.Buffers.AllocateBufferView } call
                when call.Arguments.Length > 0 && call.Arguments[0] is Expr source:
                var viewDistributedType = GetDistributedType(expression);
                var sourceResource = ResolveResource(source, synchronizationScope, bindings, resolving);
                return viewDistributedType is null
                    ? sourceResource
                    : sourceResource with { DistributedType = viewDistributedType };
            case TIR.Buffer buffer:
                var physicalBuffer = buffer.MemSpan.Buffer;
                var scope = explicitScope ?? (physicalBuffer.Location is MemoryLocation.ChipLocalData or MemoryLocation.ChipLocalRdata ||
                    buffer.ElemType is ReferenceType
                    ? TIR.NTT.BarrierScope.Chip
                    : TIR.NTT.BarrierScope.Block);
                if (TryGetSingleVariable(physicalBuffer.Start) is { } identity)
                {
                    var relativeRange = ReferenceEquals(physicalBuffer.Start, identity)
                        ? TryGetRelativeByteRange(buffer.MemSpan)
                        : null;
                    return new MemoryResource(identity, buffer, null, relativeRange, scope, buffer.DistributedType);
                }

                return new MemoryResource(
                    null,
                    buffer,
                    new MemoryArena(physicalBuffer.Location, physicalBuffer.Hierarchy),
                    TryGetAbsoluteByteRange(buffer.MemSpan),
                    scope,
                    buffer.DistributedType);
            case IVar variable:
                var variableExpr = (BaseExpr)variable;
                var variableScope = explicitScope ?? (variableExpr.CheckedDataType is ReferenceType
                    ? TIR.NTT.BarrierScope.Chip
                    : TIR.NTT.BarrierScope.Block);
                return new MemoryResource(
                    variableExpr,
                    variableExpr,
                    null,
                    null,
                    variableScope,
                    GetDistributedType(variableExpr));
            default:
                return new MemoryResource(
                    expression,
                    expression,
                    null,
                    null,
                    explicitScope ?? (expression.CheckedDataType is ReferenceType
                        ? TIR.NTT.BarrierScope.Chip
                        : TIR.NTT.BarrierScope.Block),
                    GetDistributedType(expression));
        }
    }

    private static DistributedType? GetDistributedType(BaseExpr expression)
        => expression switch
        {
            TIR.Buffer buffer => buffer.DistributedType,
            BufferVar bufferVar => bufferVar.TypeAnnotation as DistributedType,
            _ => expression.CheckedType as DistributedType,
        };

    private static MemoryByteRange? TryGetRelativeByteRange(MemSpan span)
    {
        if (!TryGetFixedDimension(span.Start, out var start) ||
            !TryGetFixedDimension(span.Size, out var size) ||
            size < 0)
        {
            return null;
        }

        try
        {
            return new MemoryByteRange(start, checked(start + size));
        }
        catch (OverflowException)
        {
            return null;
        }
    }

    private static bool TryGetFixedDimension(Dimension dimension, out long value)
    {
        if (dimension.IsFixed)
        {
            value = dimension.FixedValue;
            return true;
        }

        value = 0;
        return false;
    }

    private static bool TryGetFixedInt64(BaseExpr expression, out long value)
    {
        try
        {
            switch (expression)
            {
                case None:
                    value = 0;
                    return true;
                case DimConst dimConst:
                    value = dimConst.Value;
                    return true;
                case Dimension dimension:
                    return TryGetFixedDimension(dimension, out value);
                case TensorConst { Value.Shape.IsScalar: true } tensorConst:
                    return TryReadScalarInt64(tensorConst.Value, out value);
                default:
                    value = 0;
                    return false;
            }
        }
        catch (OverflowException)
        {
            value = 0;
            return false;
        }
    }

    private static bool TryReadScalarInt64(Tensor tensor, out long result)
    {
        var value = tensor[Array.Empty<long>()];
        switch (value)
        {
            case sbyte scalar:
                result = scalar;
                return true;
            case byte scalar:
                result = scalar;
                return true;
            case short scalar:
                result = scalar;
                return true;
            case ushort scalar:
                result = scalar;
                return true;
            case int scalar:
                result = scalar;
                return true;
            case uint scalar:
                result = scalar;
                return true;
            case long scalar:
                result = scalar;
                return true;
            case ulong scalar when scalar <= long.MaxValue:
                result = (long)scalar;
                return true;
        }

        if (value is not null)
        {
            var type = value.GetType();
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Pointer<>))
            {
                var pointer = (ulong)type.GetProperty(nameof(Pointer<byte>.Value))!.GetValue(value)!;
                if (pointer <= long.MaxValue)
                {
                    result = (long)pointer;
                    return true;
                }
            }
        }

        result = 0;
        return false;
    }

    private static BaseExpr? TryGetSingleVariable(BaseExpr expression)
    {
        var variables = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<BaseExpr>();
        stack.Push(expression);
        while (stack.Count > 0)
        {
            var current = stack.Pop();
            if (current is IVar)
            {
                variables.Add(current);
                continue;
            }

            foreach (var operand in current.Operands)
            {
                stack.Push(operand);
            }
        }

        return variables.Count == 1 ? variables.Single() : null;
    }

    internal static TIR.NTT.BarrierScope MergeScope(TIR.NTT.BarrierScope lhs, TIR.NTT.BarrierScope rhs)
        => lhs == TIR.NTT.BarrierScope.Chip || rhs == TIR.NTT.BarrierScope.Chip
            ? TIR.NTT.BarrierScope.Chip
            : TIR.NTT.BarrierScope.Block;

    private sealed class ResourceBindingScope
    {
        private ResourceBindingScope(
            ResourceBindingScope? parent,
            BaseExpr? variable,
            BaseExpr? expression)
        {
            Parent = parent;
            Variable = variable;
            Expression = expression;
        }

        public static ResourceBindingScope Empty { get; } = new(null, null, null);

        private ResourceBindingScope? Parent { get; }

        private BaseExpr? Variable { get; }

        private BaseExpr? Expression { get; }

        public ResourceBindingScope Bind(BaseExpr variable, BaseExpr expression)
            => new(this, variable, expression);

        public bool TryGet(BaseExpr variable, out BaseExpr expression)
        {
            for (var scope = this; scope?.Variable is not null; scope = scope.Parent)
            {
                if (ReferenceEquals(scope.Variable, variable))
                {
                    expression = scope.Expression!;
                    return true;
                }
            }

            expression = null!;
            return false;
        }
    }
}

internal readonly record struct SynchronizationRequirement(
    TIR.NTT.BarrierScope Scope,
    Placement? Placement,
    IRArray<int> AxisGroupAxes,
    bool IsFullChip)
{
    public static SynchronizationRequirement Block { get; } = new(
        TIR.NTT.BarrierScope.Block,
        null,
        new IRArray<int>(),
        false);

    public static SynchronizationRequirement FullChip(Placement? placement = null)
        => new(
            TIR.NTT.BarrierScope.Chip,
            placement,
            new IRArray<int>(),
            true);

    public static SynchronizationRequirement ChipAxisGroup(
        Placement placement,
        IEnumerable<int> axisGroupAxes)
    {
        var axes = axisGroupAxes.Distinct().Order().ToArray();
        if (axes.Length == 0)
        {
            return Block;
        }

        var blockAxes = GetBlockAxes(placement);
        if (axes.Any(axis => !blockAxes.Contains(axis)))
        {
            return FullChip(placement);
        }

        return axes.SequenceEqual(blockAxes)
            ? FullChip(placement)
            : new(
                TIR.NTT.BarrierScope.Chip,
                placement,
                new IRArray<int>(axes),
                false);
    }

    public static SynchronizationRequirement FromBarrier(
        TIR.NTT.Barrier barrier,
        Placement? placement = null)
        => barrier.Scope switch
        {
            TIR.NTT.BarrierScope.Block => Block,
            TIR.NTT.BarrierScope.Chip when barrier.AxisGroupAxes.IsDefaultOrEmpty => FullChip(placement),
            TIR.NTT.BarrierScope.Chip => new(
                TIR.NTT.BarrierScope.Chip,
                placement,
                new IRArray<int>(barrier.AxisGroupAxes.Distinct().Order().ToArray()),
                false),
            _ => throw new ArgumentOutOfRangeException(nameof(barrier)),
        };

    public SynchronizationRequirement Merge(SynchronizationRequirement other)
    {
        if (Scope == TIR.NTT.BarrierScope.Block)
        {
            return other;
        }

        if (other.Scope == TIR.NTT.BarrierScope.Block)
        {
            return this;
        }

        if (IsFullChip || other.IsFullChip ||
            Placement is null || other.Placement is null || Placement != other.Placement)
        {
            return FullChip(Placement == other.Placement ? Placement : null);
        }

        return ChipAxisGroup(Placement, AxisGroupAxes.Concat(other.AxisGroupAxes));
    }

    public IRArray<int> ToBarrierAxisGroupAxes()
        => Scope == TIR.NTT.BarrierScope.Chip && !IsFullChip
            ? AxisGroupAxes
            : new IRArray<int>();

    public static int[] GetBlockAxes(Placement placement)
        => Enumerable.Range(0, placement.Rank)
            .Where(placement.IsPhysicalBlockAxis)
            .ToArray();
}

internal static class SynchronizationRequirementInference
{
    public static SynchronizationRequirement ForHazard(
        ResolvedMemoryEffect producer,
        ResolvedMemoryEffect consumer)
    {
        var mergedScope = MemoryEffectAnalyzer.MergeScope(
            producer.Effect.Scope,
            consumer.Effect.Scope);
        if (mergedScope == TIR.NTT.BarrierScope.Block)
        {
            return SynchronizationRequirement.Block;
        }

        if (producer.Effect.RequiresFullChipSynchronization ||
            consumer.Effect.RequiresFullChipSynchronization)
        {
            var fullPlacement = producer.Resource.DistributedType?.Placement ==
                consumer.Resource.DistributedType?.Placement
                ? producer.Resource.DistributedType?.Placement
                : null;
            return SynchronizationRequirement.FullChip(fullPlacement);
        }

        var producerReads = producer.Effect.Mode.HasFlag(MemoryAccessMode.Read);
        var producerWrites = producer.Effect.Mode.HasFlag(MemoryAccessMode.Write);
        var consumerReads = consumer.Effect.Mode.HasFlag(MemoryAccessMode.Read);
        var consumerWrites = consumer.Effect.Mode.HasFlag(MemoryAccessMode.Write);
        var hasRaw = producerWrites && consumerReads;
        var hasOtherHazard = (producerReads && consumerWrites) ||
            (producerWrites && consumerWrites &&
                !producer.Resource.HasSameLogicalResource(consumer.Resource));
        if (hasRaw && !hasOtherHazard &&
            TryInferRawAxisGroup(
                producer.Resource.DistributedType,
                consumer.Resource.DistributedType,
                out var requirement))
        {
            return requirement;
        }

        var placement = producer.Resource.DistributedType?.Placement ==
            consumer.Resource.DistributedType?.Placement
            ? producer.Resource.DistributedType?.Placement
            : null;
        return SynchronizationRequirement.FullChip(placement);
    }

    internal static bool TryInferRawAxisGroup(
        DistributedType? producer,
        DistributedType? consumer,
        out SynchronizationRequirement requirement)
    {
        requirement = default;
        if (producer is null || consumer is null ||
            producer.Placement != consumer.Placement ||
            producer.Partial != consumer.Partial)
        {
            return false;
        }

        var placement = producer.Placement;
        if (!TryGetSplitAssignments(producer, out var producerAssignments, out var producerOrders) ||
            !TryGetSplitAssignments(consumer, out var consumerAssignments, out var consumerOrders))
        {
            return false;
        }

        for (var tensorAxis = 0; tensorAxis < Math.Max(producer.AxisPolicies.Count, consumer.AxisPolicies.Count); tensorAxis++)
        {
            var producerOrder = producerOrders.GetValueOrDefault(tensorAxis, Array.Empty<int>());
            var consumerOrder = consumerOrders.GetValueOrDefault(tensorAxis, Array.Empty<int>());
            var common = producerOrder.Where(consumerOrder.Contains).ToArray();
            if (!common.SequenceEqual(consumerOrder.Where(producerOrder.Contains)))
            {
                return false;
            }

            var removesProducerAxis = producerOrder.Any(axis => !consumerOrder.Contains(axis));
            if (removesProducerAxis &&
                (consumerOrder.Length > producerOrder.Length ||
                    !producerOrder.Take(consumerOrder.Length).SequenceEqual(consumerOrder)))
            {
                // A zero-copy coarsening forms a valid axis group only when
                // it removes a suffix of the producer's split order.
                return false;
            }
        }

        var requiredAxes = new List<int>();
        foreach (var meshAxis in producerAssignments.Keys.Union(consumerAssignments.Keys).Order())
        {
            var hasProducer = producerAssignments.TryGetValue(meshAxis, out var producerTensorAxis);
            var hasConsumer = consumerAssignments.TryGetValue(meshAxis, out var consumerTensorAxis);
            if (hasProducer && hasConsumer && producerTensorAxis != consumerTensorAxis)
            {
                return false;
            }

            if (hasProducer && !hasConsumer)
            {
                requiredAxes.Add(meshAxis);
            }
        }

        requirement = SynchronizationRequirement.ChipAxisGroup(placement, requiredAxes);
        return true;

        static bool TryGetSplitAssignments(
            DistributedType type,
            out Dictionary<int, int> assignments,
            out Dictionary<int, int[]> orders)
        {
            assignments = new Dictionary<int, int>();
            orders = new Dictionary<int, int[]>();
            for (var tensorAxis = 0; tensorAxis < type.AxisPolicies.Count; tensorAxis++)
            {
                if (type.AxisPolicies[tensorAxis] is not SBPSplit split)
                {
                    continue;
                }

                var axes = split.Axes.ToArray();
                if (axes.Distinct().Count() != axes.Length ||
                    axes.Any(axis => axis < 0 ||
                        axis >= type.Placement.Rank ||
                        !type.Placement.IsPhysicalBlockAxis(axis)))
                {
                    return false;
                }

                orders.Add(tensorAxis, axes);
                foreach (var meshAxis in axes)
                {
                    if (!assignments.TryAdd(meshAxis, tensorAxis))
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}

internal sealed class MemorySynchronizationPlanner
{
    private readonly MemoryEffectAnalyzer _analyzer;
    private readonly MemorySynchronizationScopes _materializedScopes;

    public MemorySynchronizationPlanner(
        MemoryEffectAnalyzer analyzer,
        MemorySynchronizationScopes materializedScopes)
    {
        _analyzer = analyzer;
        _materializedScopes = materializedScopes;
    }

    public PrimFunction Rewrite(PrimFunction function)
        => function.With(body: RewriteSequential(function.Body, false, false).Expression);

    private SequentialRewrite RewriteSequential(
        Sequential sequential,
        bool insideLoop,
        bool insideReduction)
    {
        var fields = new List<Expr>();
        var pendingAccesses = new PendingEffectSet();
        foreach (var field in sequential.Fields)
        {
            if (TryGetBarrier(field, out var explicitBarrier))
            {
                var explicitRequirement = SynchronizationRequirement.FromBarrier(explicitBarrier);
                if (ShouldMaterialize(explicitRequirement.Scope) &&
                    pendingAccesses.HasUnsynchronizedAccesses(explicitRequirement))
                {
                    AppendBarrier(fields, explicitRequirement);
                    pendingAccesses.Synchronize(explicitRequirement);
                }

                continue;
            }

            var effects = _analyzer.GetEffects(field, insideReduction);
            if (pendingAccesses.TryGetConflict(effects, out var requirement))
            {
                if (ShouldMaterialize(requirement.Scope) &&
                    insideLoop && requirement.Scope == TIR.NTT.BarrierScope.Chip)
                {
                    throw new InvalidOperationException("A chip-wide synchronization dependence remains inside a tiled loop. Split the producer and consumer into separate scheduling phases.");
                }

                if (ShouldMaterialize(requirement.Scope))
                {
                    AppendBarrier(fields, requirement);
                    pendingAccesses.Synchronize(requirement);
                }
            }

            var rewritten = RewriteStatement(field, insideLoop, insideReduction);
            if (rewritten.Expression is Sequential { CanFlatten: true } nested)
            {
                fields.AddRange(nested.Fields.ToArray());
            }
            else if (rewritten.Expression is not Call { Target: Nop })
            {
                fields.Add(rewritten.Expression);
            }

            var remainingEffects = effects.Clone();
            remainingEffects.RemoveAccessesAtExactScopes(rewritten.SynchronizedScopes);
            pendingAccesses.Add(remainingEffects);
        }

        return new(
            sequential.With(fields: fields.ToArray()),
            pendingAccesses.GetScopesWithoutAccesses());
    }

    private StatementRewrite RewriteStatement(
        Expr expression,
        bool insideLoop,
        bool insideReduction)
    {
        switch (expression)
        {
            case Block block:
                var initBody = RewriteSequential(block.InitBody, insideLoop, insideReduction);
                var body = RewriteSequential(block.Body, insideLoop, insideReduction);
                return new(
                    block.With(body: body.Expression, initBody: initBody.Expression),
                    initBody.SynchronizedScopes & body.SynchronizedScopes);
            case PipelineFor pipelineFor:
                return RewritePipelineFor(pipelineFor, insideReduction);
            case Nncase.TIR.For @for:
                return RewriteFor(@for, insideReduction);
            case Let let:
                var letBody = RewriteSequential(let.Body, insideLoop, insideReduction);
                var expressionScopes = let.Expression is Expr bindingExpression
                    ? _analyzer.GetEffects(bindingExpression, insideReduction).GetScopesWithoutAccesses()
                    : MemorySynchronizationScopes.All;
                return new(
                    let.With(body: letBody.Expression),
                    expressionScopes & letBody.SynchronizedScopes);
            case IfThenElse ifThenElse:
                var thenBody = RewriteSequential(ifThenElse.Then, insideLoop, insideReduction);
                var elseBody = RewriteSequential(ifThenElse.Else, insideLoop, insideReduction);
                return new(
                    ifThenElse.With(then: thenBody.Expression, @else: elseBody.Expression),
                    thenBody.SynchronizedScopes & elseBody.SynchronizedScopes);
            case Sequential sequential:
                var rewritten = RewriteSequential(sequential, insideLoop, insideReduction);
                return new(rewritten.Expression, rewritten.SynchronizedScopes);
            default:
                return new(
                    expression,
                    _analyzer.GetEffects(expression, insideReduction).GetScopesWithoutAccesses());
        }
    }

    private StatementRewrite RewriteFor(Nncase.TIR.For @for, bool insideReduction)
    {
        var isReduction = insideReduction || @for.Mode == LoopMode.Reduction;
        var body = RewriteLoopPartition(@for.Body, isReduction, $"loop '{@for.LoopVar.Name}'");
        return new(
            @for.With(body: body.Expression),
            body.SynchronizedScopes);
    }

    private StatementRewrite RewritePipelineFor(PipelineFor pipelineFor, bool insideReduction)
    {
        var isReduction = insideReduction || pipelineFor.Mode == LoopMode.Reduction;

        // Cross-phase ordering belongs to the target pipeline template.
        // Generic synchronization remains responsible only for hazards wholly
        // contained in one semantic phase.
        var produceBody = RewriteLoopPartition(
            pipelineFor.ProduceBody,
            isReduction,
            $"pipeline {pipelineFor.Plan.ScheduleId} produce phase");
        var consumeBody = RewriteLoopPartition(
            pipelineFor.ConsumeBody,
            isReduction,
            $"pipeline {pipelineFor.Plan.ScheduleId} consume phase");
        var synchronizedScopes = _analyzer
            .GetEffects(pipelineFor, insideReduction)
            .GetScopesWithoutAccesses();
        if (pipelineFor.Plan.Synchronization.RequiresConsumerRelease)
        {
            synchronizedScopes |= MemorySynchronizationScopes.Block;
        }

        return new(
            pipelineFor.With(
                produceBody: produceBody.Expression,
                consumeBody: consumeBody.Expression),
            synchronizedScopes);
    }

    private SequentialRewrite RewriteLoopPartition(
        Sequential originalBody,
        bool isReduction,
        string context)
    {
        var body = RewriteSequential(originalBody, true, isReduction);
        var loopEffects = _analyzer.GetIterationLocalEffects(originalBody, isReduction);
        if (loopEffects.TryGetReadWriteAlias(out var requiredScope) &&
            !body.SynchronizedScopes.HasFlag(ToSynchronizationScope(requiredScope)))
        {
            if (ShouldMaterialize(requiredScope) && requiredScope == TIR.NTT.BarrierScope.Chip)
            {
                throw new InvalidOperationException(
                    $"A chip-wide loop-carried memory dependence remains in {context}. " +
                    "Split the producer and consumer into separate scheduling phases.");
            }

            if (ShouldMaterialize(requiredScope))
            {
                var fields = body.Expression.Fields.ToArray().ToList();
                AppendBarrier(
                    fields,
                    requiredScope == TIR.NTT.BarrierScope.Chip
                        ? SynchronizationRequirement.FullChip()
                        : SynchronizationRequirement.Block);
                body = new(
                    body.Expression.With(fields: fields.ToArray()),
                    body.SynchronizedScopes | GetScopesSatisfiedBy(requiredScope));
            }
        }

        return body;
    }

    private static MemorySynchronizationScopes ToSynchronizationScope(TIR.NTT.BarrierScope scope)
        => scope switch
        {
            TIR.NTT.BarrierScope.Block => MemorySynchronizationScopes.Block,
            TIR.NTT.BarrierScope.Chip => MemorySynchronizationScopes.Chip,
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, null),
        };

    private static MemorySynchronizationScopes GetScopesSatisfiedBy(TIR.NTT.BarrierScope scope)
        => scope switch
        {
            TIR.NTT.BarrierScope.Block => MemorySynchronizationScopes.Block,
            TIR.NTT.BarrierScope.Chip => MemorySynchronizationScopes.All,
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, null),
        };

    private bool ShouldMaterialize(TIR.NTT.BarrierScope scope)
        => scope switch
        {
            TIR.NTT.BarrierScope.Block => _materializedScopes.HasFlag(MemorySynchronizationScopes.Block),
            TIR.NTT.BarrierScope.Chip => _materializedScopes.HasFlag(MemorySynchronizationScopes.Chip),
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, null),
        };

    private static void AppendBarrier(
        List<Expr> fields,
        SynchronizationRequirement requirement)
    {
        if (fields.Count > 0 && TryGetBarrier(fields[^1], out var previousBarrier))
        {
            var previous = SynchronizationRequirement.FromBarrier(
                previousBarrier,
                requirement.Placement);
            var merged = previous.Merge(requirement);
            fields[^1] = TIR.F.NTT.Barrier(merged.Scope, merged.ToBarrierAxisGroupAxes());
            return;
        }

        fields.Add(TIR.F.NTT.Barrier(requirement.Scope, requirement.ToBarrierAxisGroupAxes()));
    }

    private static bool TryGetBarrier(Expr expression, out TIR.NTT.Barrier barrier)
    {
        if (expression is Call { Target: TIR.NTT.Barrier target })
        {
            barrier = target;
            return true;
        }

        barrier = null!;
        return false;
    }

    private readonly record struct StatementRewrite(
        Expr Expression,
        MemorySynchronizationScopes SynchronizedScopes);

    private readonly record struct SequentialRewrite(
        Sequential Expression,
        MemorySynchronizationScopes SynchronizedScopes);
}

internal sealed record MemoryResource(
    BaseExpr? ExpressionIdentity,
    BaseExpr? LogicalIdentity,
    MemoryArena? Arena,
    MemoryByteRange? ByteRange,
    TIR.NTT.BarrierScope Scope,
    DistributedType? DistributedType)
{
    public bool HasSameRegion(MemoryResource other)
        => HasSameLogicalResource(other) && HasSameBacking(other) && ByteRange == other.ByteRange;

    public bool HasSameLogicalResource(MemoryResource other)
        => (LogicalIdentity is not null && ReferenceEquals(LogicalIdentity, other.LogicalIdentity)) ||
            (ExpressionIdentity is not null && ReferenceEquals(ExpressionIdentity, other.ExpressionIdentity));

    public bool MayAlias(MemoryResource other)
    {
        if (!HasSameBacking(other))
        {
            return false;
        }

        return ByteRange is not { } lhs || other.ByteRange is not { } rhs || lhs.Overlaps(rhs);
    }

    private bool HasSameBacking(MemoryResource other)
    {
        if (ExpressionIdentity is not null || other.ExpressionIdentity is not null)
        {
            return ExpressionIdentity is not null &&
                other.ExpressionIdentity is not null &&
                ReferenceEquals(ExpressionIdentity, other.ExpressionIdentity);
        }

        return Arena is not null && Arena == other.Arena;
    }
}

internal sealed record FunctionEffectSummary(IReadOnlyDictionary<int, EffectInfo> ParameterEffects);

internal sealed class EffectSet
{
    private readonly List<ResolvedMemoryEffect> _items = new();

    public IEnumerable<ResolvedMemoryEffect> Items => _items;

    public void Add(MemoryResource resource, EffectInfo effect)
    {
        var index = _items.FindIndex(item => item.Resource.HasSameRegion(resource));
        if (index >= 0)
        {
            var existing = _items[index];
            _items[index] = existing with { Effect = existing.Effect.Merge(effect) };
        }
        else
        {
            _items.Add(new ResolvedMemoryEffect(resource, effect));
        }
    }

    public void UnionWith(EffectSet other)
    {
        foreach (var item in other._items)
        {
            Add(item.Resource, item.Effect);
        }
    }

    public EffectSet Clone()
    {
        var result = new EffectSet();
        result.UnionWith(this);
        return result;
    }

    public bool TryGetReadWriteAlias(out TIR.NTT.BarrierScope scope)
    {
        var found = false;
        scope = TIR.NTT.BarrierScope.Block;
        for (var lhsIndex = 0; lhsIndex < _items.Count; lhsIndex++)
        {
            var lhs = _items[lhsIndex];
            for (var rhsIndex = lhsIndex; rhsIndex < _items.Count; rhsIndex++)
            {
                var rhs = _items[rhsIndex];
                if (!lhs.Resource.MayAlias(rhs.Resource) ||
                    !HasReadWriteConflict(lhs.Effect.Mode, rhs.Effect.Mode))
                {
                    continue;
                }

                found = true;
                scope = MemoryEffectAnalyzer.MergeScope(
                    scope,
                    MemoryEffectAnalyzer.MergeScope(lhs.Effect.Scope, rhs.Effect.Scope));
            }
        }

        return found;

        static bool HasReadWriteConflict(MemoryAccessMode lhs, MemoryAccessMode rhs)
            => (lhs.HasFlag(MemoryAccessMode.Read) && rhs.HasFlag(MemoryAccessMode.Write)) ||
                (lhs.HasFlag(MemoryAccessMode.Write) && rhs.HasFlag(MemoryAccessMode.Read));
    }

    public void RemoveAccessesAtExactScopes(MemorySynchronizationScopes scopes)
    {
        for (var index = _items.Count - 1; index >= 0; index--)
        {
            var effect = _items[index].Effect;
            if (effect.Mode != MemoryAccessMode.None && scopes.HasFlag(ToSynchronizationScope(effect.Scope)))
            {
                _items.RemoveAt(index);
            }
        }
    }

    public MemorySynchronizationScopes GetScopesWithoutAccesses()
    {
        var result = MemorySynchronizationScopes.All;
        foreach (var item in _items)
        {
            if (item.Effect.Mode != MemoryAccessMode.None)
            {
                result &= ~ToSynchronizationScope(item.Effect.Scope);
            }
        }

        return result;
    }

    private static MemorySynchronizationScopes ToSynchronizationScope(TIR.NTT.BarrierScope scope)
        => scope switch
        {
            TIR.NTT.BarrierScope.Block => MemorySynchronizationScopes.Block,
            TIR.NTT.BarrierScope.Chip => MemorySynchronizationScopes.Chip,
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, null),
        };

}

internal sealed class PendingEffectSet
{
    private readonly List<PendingEffect> _items = new();

    public void Add(EffectSet effects)
    {
        foreach (var effect in effects.Items)
        {
            if (effect.Effect.Mode != MemoryAccessMode.None)
            {
                _items.Add(new PendingEffect(effect));
            }
        }
    }

    public bool TryGetConflict(
        EffectSet consumer,
        out SynchronizationRequirement requirement)
    {
        SynchronizationRequirement? merged = null;
        foreach (var consumerEffect in consumer.Items)
        {
            foreach (var pending in _items)
            {
                if (!pending.Access.Resource.MayAlias(consumerEffect.Resource) ||
                    !RequiresSynchronization(pending.Access, consumerEffect))
                {
                    continue;
                }

                var current = SynchronizationRequirementInference.ForHazard(
                    pending.Access,
                    consumerEffect);
                if (pending.Coverage.Covers(current))
                {
                    continue;
                }

                merged = merged is { } existing
                    ? existing.Merge(current)
                    : current;
            }
        }

        requirement = merged ?? default;
        return merged is not null;
    }

    public bool HasUnsynchronizedAccesses(SynchronizationRequirement requirement)
        => _items.Any(item =>
            IsSatisfiedBy(item.Access.Effect.Scope, requirement.Scope) &&
            !item.Coverage.Covers(requirement));

    public void Synchronize(SynchronizationRequirement requirement)
    {
        foreach (var item in _items)
        {
            item.Coverage.Apply(requirement, item.Access.Resource.DistributedType);
        }

        _items.RemoveAll(item =>
            item.Access.Effect.Scope == TIR.NTT.BarrierScope.Block
                ? item.Coverage.BlockSynchronized
                : item.Coverage.FullChipSynchronized);
    }

    public MemorySynchronizationScopes GetScopesWithoutAccesses()
    {
        var result = MemorySynchronizationScopes.All;
        foreach (var item in _items)
        {
            result &= ~ToSynchronizationScope(item.Access.Effect.Scope);
        }

        return result;
    }

    private static bool RequiresSynchronization(
        ResolvedMemoryEffect producer,
        ResolvedMemoryEffect consumer)
    {
        var producerReads = producer.Effect.Mode.HasFlag(MemoryAccessMode.Read);
        var producerWrites = producer.Effect.Mode.HasFlag(MemoryAccessMode.Write);
        var consumerReads = consumer.Effect.Mode.HasFlag(MemoryAccessMode.Read);
        var consumerWrites = consumer.Effect.Mode.HasFlag(MemoryAccessMode.Write);
        return (producerWrites && consumerReads) ||
            (producerReads && consumerWrites) ||
            (producerWrites && consumerWrites &&
                !producer.Resource.HasSameLogicalResource(consumer.Resource));
    }

    private static MemorySynchronizationScopes ToSynchronizationScope(
        TIR.NTT.BarrierScope scope)
        => scope switch
        {
            TIR.NTT.BarrierScope.Block => MemorySynchronizationScopes.Block,
            TIR.NTT.BarrierScope.Chip => MemorySynchronizationScopes.Chip,
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, null),
        };

    private static bool IsSatisfiedBy(
        TIR.NTT.BarrierScope required,
        TIR.NTT.BarrierScope actual)
        => actual == TIR.NTT.BarrierScope.Chip ||
            required == TIR.NTT.BarrierScope.Block;

    private sealed class PendingEffect
    {
        public PendingEffect(ResolvedMemoryEffect access)
        {
            Access = access;
        }

        public ResolvedMemoryEffect Access { get; }

        public BarrierCoverage Coverage { get; } = new();
    }

    private sealed class BarrierCoverage
    {
        private readonly HashSet<int> _axisGroupAxes = new();
        private Placement? _placement;

        public bool BlockSynchronized { get; private set; }

        public bool FullChipSynchronized { get; private set; }

        public bool Covers(SynchronizationRequirement requirement)
        {
            if (requirement.Scope == TIR.NTT.BarrierScope.Block)
            {
                return BlockSynchronized;
            }

            if (FullChipSynchronized)
            {
                return true;
            }

            return !requirement.IsFullChip &&
                requirement.Placement is not null &&
                _placement == requirement.Placement &&
                requirement.AxisGroupAxes.All(_axisGroupAxes.Contains);
        }

        public void Apply(
            SynchronizationRequirement requirement,
            DistributedType? distributedType)
        {
            BlockSynchronized = true;
            if (requirement.Scope == TIR.NTT.BarrierScope.Block)
            {
                return;
            }

            if (requirement.IsFullChip)
            {
                FullChipSynchronized = true;
                return;
            }

            var placement = requirement.Placement ?? distributedType?.Placement;
            if (placement is null ||
                (distributedType is not null && distributedType.Placement != placement))
            {
                return;
            }

            if (_placement is not null && _placement != placement)
            {
                return;
            }

            _placement = placement;
            _axisGroupAxes.UnionWith(requirement.AxisGroupAxes);
            var blockAxes = SynchronizationRequirement.GetBlockAxes(placement);
            FullChipSynchronized = blockAxes.All(_axisGroupAxes.Contains);
        }
    }
}
