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

internal readonly record struct EffectInfo(MemoryAccessMode Mode, TIR.NTT.BarrierScope Scope)
{
    public EffectInfo Merge(EffectInfo other)
        => new(Mode | other.Mode, MemoryEffectAnalyzer.MergeScope(Scope, other.Scope));
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

                effects.Add(ResolveResource(argument, effect.Scope, bindings), effect.Mode);
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
            effects.Add(resource with { Scope = MergeScope(resource.Scope, effect.Scope) }, effect.Mode);
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

            var resource = ResolveResource(boundResource, synchronizationScope, bindings, resolving);
            resolving.Remove(expression);
            return resource;
        }

        switch (expression)
        {
            case Call { Target: IR.Buffers.BufferSubview or IR.Buffers.AllocateBufferView } call
                when call.Arguments.Length > 0 && call.Arguments[0] is Expr source:
                return ResolveResource(source, synchronizationScope, bindings, resolving);
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
                    return new MemoryResource(identity, buffer, null, relativeRange, scope);
                }

                return new MemoryResource(
                    null,
                    buffer,
                    new MemoryArena(physicalBuffer.Location, physicalBuffer.Hierarchy),
                    TryGetAbsoluteByteRange(buffer.MemSpan),
                    scope);
            case IVar variable:
                var variableExpr = (BaseExpr)variable;
                var variableScope = explicitScope ?? (variableExpr.CheckedDataType is ReferenceType
                    ? TIR.NTT.BarrierScope.Chip
                    : TIR.NTT.BarrierScope.Block);
                return new MemoryResource(variableExpr, variableExpr, null, null, variableScope);
            default:
                return new MemoryResource(
                    expression,
                    expression,
                    null,
                    null,
                    explicitScope ?? (expression.CheckedDataType is ReferenceType
                        ? TIR.NTT.BarrierScope.Chip
                        : TIR.NTT.BarrierScope.Block));
        }
    }

    private static MemoryByteRange? TryGetAbsoluteByteRange(MemSpan span)
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
        var pendingAccesses = new EffectSet();
        foreach (var field in sequential.Fields)
        {
            if (TryGetBarrierScope(field, out var explicitScope))
            {
                if (ShouldMaterialize(explicitScope) &&
                    pendingAccesses.HasAccessesAtOrBelow(explicitScope))
                {
                    AppendBarrier(fields, explicitScope);
                    pendingAccesses.RemoveAccessesAtOrBelow(explicitScope);
                }

                continue;
            }

            var effects = _analyzer.GetEffects(field, insideReduction);
            if (pendingAccesses.TryGetConflict(effects, out var requiredScope))
            {
                if (ShouldMaterialize(requiredScope) &&
                    insideLoop && requiredScope == TIR.NTT.BarrierScope.Chip)
                {
                    throw new InvalidOperationException("A chip-wide synchronization dependence remains inside a tiled loop. Split the producer and consumer into separate scheduling phases.");
                }

                if (ShouldMaterialize(requiredScope))
                {
                    AppendBarrier(fields, requiredScope);
                    pendingAccesses.RemoveAccessesAtOrBelow(requiredScope);
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
            pendingAccesses.UnionWith(remainingEffects);
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
                AppendBarrier(fields, requiredScope);
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

    private static void AppendBarrier(List<Expr> fields, TIR.NTT.BarrierScope scope)
    {
        if (fields.Count > 0 && TryGetBarrierScope(fields[^1], out var previousScope))
        {
            fields[^1] = TIR.F.NTT.Barrier(MemoryEffectAnalyzer.MergeScope(previousScope, scope));
            return;
        }

        fields.Add(TIR.F.NTT.Barrier(scope));
    }

    private static bool TryGetBarrierScope(Expr expression, out TIR.NTT.BarrierScope scope)
    {
        if (expression is Call { Target: TIR.NTT.Barrier barrier })
        {
            scope = barrier.Scope;
            return true;
        }

        scope = default;
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
    TIR.NTT.BarrierScope Scope)
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

    public void Add(MemoryResource resource, MemoryAccessMode mode)
        => Add(resource, new EffectInfo(mode, resource.Scope));

    private void Add(MemoryResource resource, EffectInfo effect)
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

    public bool TryGetConflict(EffectSet consumer, out TIR.NTT.BarrierScope scope)
    {
        var found = false;
        scope = TIR.NTT.BarrierScope.Block;
        foreach (var consumerEffect in consumer._items)
        {
            foreach (var pendingAccess in _items)
            {
                if (!pendingAccess.Resource.MayAlias(consumerEffect.Resource) ||
                    !RequiresSynchronization(pendingAccess, consumerEffect))
                {
                    continue;
                }

                found = true;
                scope = MemoryEffectAnalyzer.MergeScope(
                    scope,
                    MemoryEffectAnalyzer.MergeScope(consumerEffect.Effect.Scope, pendingAccess.Effect.Scope));
            }
        }

        return found;

        static bool RequiresSynchronization(
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

    public bool HasAccessesAtOrBelow(TIR.NTT.BarrierScope scope)
        => _items.Any(item => item.Effect.Mode != MemoryAccessMode.None && IsSatisfiedBy(item.Effect.Scope, scope));

    public void RemoveAccessesAtOrBelow(TIR.NTT.BarrierScope scope)
    {
        for (var index = _items.Count - 1; index >= 0; index--)
        {
            var effect = _items[index].Effect;
            if (effect.Mode != MemoryAccessMode.None && IsSatisfiedBy(effect.Scope, scope))
            {
                _items.RemoveAt(index);
            }
        }
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

    private static bool IsSatisfiedBy(TIR.NTT.BarrierScope required, TIR.NTT.BarrierScope actual)
        => actual == TIR.NTT.BarrierScope.Chip || required == TIR.NTT.BarrierScope.Block;
}
