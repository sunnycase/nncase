// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Assigns every semantic operation to one module kind before target-specific
/// lowering starts.
/// </summary>
public sealed class AssignModulePlacementPass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var target = CompileSession.Target;
        var options = CompileSession.CompileOptions;
        var compilers = target.ModuleCompilers.ToDictionary(
            compiler => compiler.ModuleKind,
            StringComparer.Ordinal);

        foreach (var owner in input.Functions)
        {
            var semanticRegions = SemanticRegionCollector.Collect(owner);
            var collector = new LocalOpCallCollector();
            collector.Visit(owner);
            foreach (var call in collector.Calls)
            {
                if (call.Target is not Op)
                {
                    continue;
                }

                if (semanticRegions.TryGetValue(call, out var semanticRegion))
                {
                    call.Metadata.SemanticRegion = semanticRegion;
                }

                var moduleKind = target.GetPreferredModuleKind(owner, call, options);
                if (!compilers.TryGetValue(moduleKind, out var compiler))
                {
                    throw new InvalidOperationException(
                        $"Target {target.Name} placed {call.Target.GetType().Name} in unknown " +
                        $"module kind {moduleKind}.");
                }

                var moduleOptions = target.GetModuleCompileOptions(moduleKind, options);
                if (!compiler.IsSupportedCall(call, moduleOptions))
                {
                    var outputNames = call.Metadata.OutputNames is { Count: > 0 }
                        ? string.Join(", ", call.Metadata.OutputNames)
                        : "<unnamed>";
                    var region = (call.Metadata.SemanticRegion ?? owner.Metadata.SemanticRegion) is { } failedRegion
                        ? $" in semantic region {failedRegion.Kind}:{failedRegion.Instance}"
                        : string.Empty;
                    throw new NotSupportedException(
                        $"Target {target.Name} placed operation {call.Target.GetType().Name} " +
                        $"({outputNames}){region} in module {moduleKind}, but that module compiler " +
                        "does not support the operation's inferred signature.");
                }

                call.Metadata.ExecutionModuleKind = moduleKind;
            }
        }

        StripSemanticRegionMarkers(input);
        AssignFunctionPlacement(input);

        return Task.FromResult(input);
    }

    private static void StripSemanticRegionMarkers(IRModule module)
    {
        var functions = module.Functions
            .Select((function, index) => (Function: function as Function, Index: index))
            .Where(item => item.Function is not null)
            .Select(item => (Function: item.Function!, item.Index))
            .ToArray();
        foreach (var (function, index) in functions)
        {
            var stripper = new SemanticRegionMarkerStripper();
            var body = stripper.Rewrite(function.Body);
            if (!stripper.IsMutated)
            {
                continue;
            }

            module.Replace(
                index,
                function.With(
                    function.Name,
                    function.ModuleKind,
                    body,
                    function.Parameters.ToArray(),
                    function.Role));
        }
    }

    private static void AssignFunctionPlacement(IRModule module)
    {
        var indexedFunctions = module.Functions
            .Select((function, index) => (Function: function as Function, Index: index))
            .Where(item => item.Function is not null)
            .Select(item => (Function: item.Function!, item.Index))
            .ToArray();
        var functions = indexedFunctions.Select(item => item.Function).ToArray();
        var functionSet = new HashSet<Function>(functions, ReferenceEqualityComparer.Instance);
        var functionIndices = new Dictionary<Function, int>(ReferenceEqualityComparer.Instance);
        foreach (var (function, index) in indexedFunctions)
        {
            functionIndices.Add(function, index);
        }

        var placementKinds = new Dictionary<Function, HashSet<string>>(
            ReferenceEqualityComparer.Instance);

        foreach (var function in FunctionCallGraphUtility.GetCalleeFirstOrder(functions))
        {
            var kinds = new HashSet<string>(StringComparer.Ordinal);
            var collector = new LocalOpCallCollector();
            collector.Visit(function.Body);
            foreach (var call in collector.Calls)
            {
                if (call.Target is Op)
                {
                    if (string.IsNullOrEmpty(call.Metadata.ExecutionModuleKind))
                    {
                        throw new InvalidOperationException(
                            $"Operation {call.Target.GetType().Name} in function {function.Name} " +
                            "has no assigned execution module.");
                    }

                    kinds.Add(call.Metadata.ExecutionModuleKind);
                    continue;
                }

                if (TryGetFunctionTarget(call.Target, out var callee) && functionSet.Contains(callee))
                {
                    foreach (var calleeKind in placementKinds[callee])
                    {
                        kinds.Add(calleeKind);
                    }

                    continue;
                }

                if (call.Target is BaseFunction externalCallee &&
                    externalCallee.Role != FunctionRole.ModuleDispatch)
                {
                    kinds.Add(externalCallee.ModuleKind);
                }
            }

            placementKinds.Add(function, kinds);
        }

        foreach (var original in FunctionCallGraphUtility.GetCalleeFirstOrder(functions))
        {
            var index = functionIndices[original];
            var current = (Function)module.Functions[index];
            var kinds = placementKinds[original];
            var moduleKind = kinds.Count == 1 ? kinds.Single() : current.ModuleKind;
            var role = kinds.Count > 1 ? FunctionRole.ModuleDispatch : current.Role;
            if (!string.Equals(current.ModuleKind, moduleKind, StringComparison.Ordinal) ||
                current.Role != role)
            {
                module.Replace(
                    index,
                    current.With(
                        current.Name,
                        moduleKind,
                        current.Body,
                        current.Parameters.ToArray(),
                        role));
            }
        }
    }

    private static bool TryGetFunctionTarget(BaseExpr target, out Function function)
    {
        switch (target)
        {
            case Function direct:
                function = direct;
                return true;
            case FunctionWrapper { Target: Function wrapped }:
                function = wrapped;
                return true;
            default:
                function = null!;
                return false;
        }
    }

    private sealed class LocalOpCallCollector : ExprWalker
    {
        public LocalOpCallCollector()
            : base(visitOtherFunctions: false)
        {
        }

        public List<Call> Calls { get; } = new();

        protected override System.Reactive.Unit VisitLeafCall(Call expr)
        {
            Calls.Add(expr);

            return base.VisitLeafCall(expr);
        }
    }

    private sealed class SemanticRegionMarkerStripper : ExprRewriter
    {
        protected override BaseExpr RewriteLeafMarker(Marker expr)
        {
            if (!SemanticRegionUtility.TryGet(expr, out _, out _))
            {
                return expr;
            }

            SetMutated();
            return expr.Target;
        }
    }

    private sealed class SemanticRegionCollector : ExprWalker
    {
        private readonly Dictionary<Call, SemanticRegion> _regions =
            new(ReferenceEqualityComparer.Instance);

        private SemanticRegionCollector()
            : base(visitOtherFunctions: false)
        {
        }

        public static IReadOnlyDictionary<Call, SemanticRegion> Collect(BaseFunction owner)
        {
            var collector = new SemanticRegionCollector();
            collector.Visit(owner);
            return collector._regions;
        }

        protected override Unit VisitLeafMarker(Marker expr)
        {
            if (SemanticRegionUtility.TryGet(expr, out var region, out var inputs))
            {
                CollectRegion(expr.Target, inputs, region);
            }

            return base.VisitLeafMarker(expr);
        }

        private void CollectRegion(
            BaseExpr output,
            IReadOnlyList<BaseExpr> inputs,
            SemanticRegion region)
        {
            var boundaries = inputs.ToHashSet(ReferenceEqualityComparer.Instance);
            var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            var pending = new Stack<BaseExpr>();
            pending.Push(output);
            while (pending.TryPop(out var current))
            {
                if (!visited.Add(current) || boundaries.Contains(current))
                {
                    continue;
                }

                if (current is Marker marker &&
                    SemanticRegionUtility.TryGet(marker, out _, out _))
                {
                    continue;
                }

                if (current is Call { Target: Op } call)
                {
                    if (_regions.TryGetValue(call, out var existing) && existing != region)
                    {
                        throw new InvalidOperationException(
                            $"Operation {call.Target.GetType().Name} belongs to conflicting semantic regions " +
                            $"{existing.Kind}:{existing.Instance} and {region.Kind}:{region.Instance}.");
                    }

                    _regions[call] = region;
                }

                foreach (var operand in current.Operands)
                {
                    if (operand is not BaseFunction)
                    {
                        pending.Push(operand);
                    }
                }
            }
        }
    }
}
