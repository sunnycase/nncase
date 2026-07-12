// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Schedule.TileGraph;

public readonly record struct GridBufferAlias(int SourceAccessIndex, int ResultAccessIndex);

internal sealed record GridBodyAnalysis(
    ImmutableArray<MemoryEffect> Effects,
    ImmutableArray<GridBufferAlias> BufferAliases);

/// <summary>
/// Resolves the effects of a grid body back to its declared access parameters.
/// GridAccess describes external dataflow; this analysis describes local tile
/// accesses performed by the body calls.
/// </summary>
internal sealed class GridMemoryEffectAnalysis
{
    private readonly Dictionary<PrimFunction, GridBodyAnalysis> _functionSummaries = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<PrimFunction> _activeFunctions = new(ReferenceEqualityComparer.Instance);

    public GridBodyAnalysis Analyze(Grid grid)
    {
        var roots = grid.Accesses.ToArray().Select(access => (BaseExpr)access.Parameter).ToArray();
        var analysis = AnalyzeExpression(grid.Body, roots, ResourceBindings.Empty);
        foreach (var alias in analysis.BufferAliases)
        {
            if (!grid.Accesses[alias.SourceAccessIndex].IsRead || !grid.Accesses[alias.ResultAccessIndex].IsWrite)
            {
                throw new InvalidOperationException(
                    $"Grid buffer alias must map a readable access to a writable access, got {alias.SourceAccessIndex}->{alias.ResultAccessIndex}.");
            }
        }

        for (var index = 0; index < analysis.Effects.Length; index++)
        {
            ValidateAccess(grid, index, analysis.Effects[index], analysis.BufferAliases);
        }

        return analysis;
    }

    private static void ValidateAccess(Grid grid, int index, MemoryEffect effect, IReadOnlyList<GridBufferAlias> aliases)
    {
        var access = grid.Accesses[index];
        if (effect.Mode == MemoryAccessMode.None)
        {
            if (aliases.Any(alias => alias.SourceAccessIndex == index || alias.ResultAccessIndex == index))
            {
                return;
            }

            throw new InvalidOperationException($"Grid access {index} ({access.AccessMode}) is not referenced by any memory-effect operand in the grid body.");
        }

        if (access.AccessMode == GridAccessMode.Read && !effect.Mode.HasFlag(MemoryAccessMode.Read))
        {
            throw new InvalidOperationException($"Grid read access {index} is only used with local effect {effect.Mode}.");
        }

        if (access.AccessMode == GridAccessMode.Write && !effect.Mode.HasFlag(MemoryAccessMode.Write))
        {
            throw new InvalidOperationException($"Grid write access {index} is only used with local effect {effect.Mode}.");
        }
    }

    private GridBodyAnalysis AnalyzeExpression(Expr expression, IReadOnlyList<BaseExpr> roots, ResourceBindings bindings)
    {
        var effects = Enumerable.Repeat(MemoryEffect.None, roots.Count).ToArray();
        var bufferAliases = new List<GridBufferAlias>();
        Visit(expression, bindings);
        return new(ImmutableArray.Create(effects), ImmutableArray.CreateRange(bufferAliases.Distinct()));

        void Visit(Expr current, ResourceBindings currentBindings)
        {
            switch (current)
            {
                case Block block:
                    Visit(block.InitBody, currentBindings);
                    Visit(block.Body, currentBindings);
                    return;
                case Sequential sequential:
                    foreach (var field in sequential.Fields)
                    {
                        Visit(field, currentBindings);
                    }

                    return;
                case Nncase.TIR.For @for:
                    Visit(@for.Body, currentBindings);
                    return;
                case Let let:
                    if (let.Expression is Expr bindingExpression)
                    {
                        Visit(bindingExpression, currentBindings);
                    }

                    Visit(let.Body, currentBindings.Bind((BaseExpr)let.Var, let.Expression));
                    return;
                case IfThenElse ifThenElse:
                    Visit(ifThenElse.Condition, currentBindings);
                    Visit(ifThenElse.Then, currentBindings);
                    Visit(ifThenElse.Else, currentBindings);
                    return;
                case Call { Target: PrimFunction callee } call:
                    VisitCallArguments(call, currentBindings);
                    InstantiateFunction(callee, call.Arguments, currentBindings);
                    return;
                case Call { Target: PrimFunctionWrapper }:
                    throw new InvalidOperationException("PrimFunctionWrapper must be eliminated before grid memory-effect analysis.");
                case Call { Target: Op } call:
                    VisitCallArguments(call, currentBindings);
                    MemoryEffectUtility.VisitCallEffects(call, (argument, _, effect) => AddEffect(argument, effect, currentBindings));
                    VisitBufferAliases(call, currentBindings);
                    return;
                default:
                    foreach (var operand in current.Operands)
                    {
                        if (operand is Expr operandExpression)
                        {
                            Visit(operandExpression, currentBindings);
                        }
                    }

                    return;
            }
        }

        void VisitCallArguments(Call call, ResourceBindings currentBindings)
        {
            foreach (var argument in call.Arguments)
            {
                if (argument is Expr argumentExpression)
                {
                    Visit(argumentExpression, currentBindings);
                }
            }
        }

        void InstantiateFunction(PrimFunction callee, ReadOnlySpan<BaseExpr> arguments, ResourceBindings currentBindings)
        {
            var summary = GetFunctionSummary(callee);
            if (summary.Effects.Length != callee.Parameters.Length)
            {
                throw new InvalidOperationException($"Invalid memory-effect summary for PrimFunction {callee.Name}.");
            }

            for (var parameterIndex = 0; parameterIndex < summary.Effects.Length; parameterIndex++)
            {
                var effect = summary.Effects[parameterIndex];
                if (effect.Mode == MemoryAccessMode.None)
                {
                    continue;
                }

                if (parameterIndex >= arguments.Length || arguments[parameterIndex] is not Expr argument)
                {
                    throw new InvalidOperationException($"Cannot map memory effect for PrimFunction {callee.Name} parameter {parameterIndex}.");
                }

                AddEffect(argument, effect, currentBindings);
            }

            foreach (var alias in summary.BufferAliases)
            {
                if (alias.SourceAccessIndex >= arguments.Length || alias.ResultAccessIndex >= arguments.Length ||
                    arguments[alias.SourceAccessIndex] is not Expr sourceArgument ||
                    arguments[alias.ResultAccessIndex] is not Expr resultArgument)
                {
                    throw new InvalidOperationException(
                        $"Cannot map buffer alias for PrimFunction {callee.Name} parameters {alias.SourceAccessIndex}->{alias.ResultAccessIndex}.");
                }

                AddBufferAlias(sourceArgument, resultArgument, currentBindings, $"PrimFunction {callee.Name}");
            }
        }

        void AddEffect(Expr argument, MemoryEffect effect, ResourceBindings currentBindings)
        {
            foreach (var rootIndex in ResolveRoots(argument, roots, currentBindings))
            {
                effects[rootIndex] = MemoryEffectUtility.Merge(effects[rootIndex], effect);
            }
        }

        void VisitBufferAliases(Call call, ResourceBindings currentBindings)
        {
            if (call.Target is not IBufferAliasOp aliasOp)
            {
                return;
            }

            foreach (var alias in aliasOp.BufferAliases)
            {
                AddBufferAlias(
                    (Expr)call.Arguments[alias.Source.Index],
                    (Expr)call.Arguments[alias.Result.Index],
                    currentBindings,
                    $"{call.Target.GetType().Name}.{alias.Result.Name}->{alias.Source.Name}");
            }
        }

        void AddBufferAlias(Expr source, Expr result, ResourceBindings currentBindings, string context)
        {
            var sourceRoots = ResolveRoots(source, roots, currentBindings).Distinct().ToArray();
            var resultRoots = ResolveRoots(result, roots, currentBindings).Distinct().ToArray();
            if (sourceRoots.Length != 1 || resultRoots.Length != 1)
            {
                throw new InvalidOperationException(
                    $"Buffer alias {context} must resolve to exactly one Grid access on each side.");
            }

            bufferAliases.Add(new(sourceRoots[0], resultRoots[0]));
        }
    }

    private GridBodyAnalysis GetFunctionSummary(PrimFunction function)
    {
        if (_functionSummaries.TryGetValue(function, out var summary))
        {
            return summary;
        }

        if (!_activeFunctions.Add(function))
        {
            throw new InvalidOperationException($"Recursive PrimFunction call graph is not supported by grid memory-effect analysis: {function.Name}.");
        }

        var roots = function.Parameters.ToArray().Select(parameter => (BaseExpr)parameter).ToArray();
        summary = AnalyzeExpression(function.Body, roots, ResourceBindings.Empty);
        _activeFunctions.Remove(function);
        _functionSummaries.Add(function, summary);
        return summary;
    }

    private IEnumerable<int> ResolveRoots(Expr expression, IReadOnlyList<BaseExpr> roots, ResourceBindings bindings)
    {
        var result = new HashSet<int>();
        var active = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
        Resolve(expression);
        return result;

        void Resolve(BaseExpr current)
        {
            for (var index = 0; index < roots.Count; index++)
            {
                if (ReferenceEquals(current, roots[index]))
                {
                    result.Add(index);
                    return;
                }
            }

            if (!active.Add(current))
            {
                throw new InvalidOperationException("Cyclic grid resource alias detected during memory-effect analysis.");
            }

            if (current is IVar && bindings.TryGet(current, out var boundExpression))
            {
                Resolve(boundExpression);
            }
            else if (current is Call { Target: IR.Buffers.BufferSubview or IR.Buffers.AllocateBufferView } view && view.Arguments.Length > 0)
            {
                Resolve(view.Arguments[0]);
            }
            else
            {
                foreach (var operand in current.Operands)
                {
                    Resolve(operand);
                }
            }

            active.Remove(current);
        }
    }

    private sealed class ResourceBindings
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _bindings;

        private ResourceBindings(IReadOnlyDictionary<BaseExpr, BaseExpr> bindings)
        {
            _bindings = bindings;
        }

        public static ResourceBindings Empty { get; } = new(new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance));

        public ResourceBindings Bind(BaseExpr variable, BaseExpr value)
        {
            var bindings = new Dictionary<BaseExpr, BaseExpr>(_bindings, ReferenceEqualityComparer.Instance)
            {
                [variable] = value,
            };
            return new(bindings);
        }

        public bool TryGet(BaseExpr variable, out BaseExpr value)
            => _bindings.TryGetValue(variable, out value!);
    }
}
