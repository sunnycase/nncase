// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Tensors;
using Nncase.Passes.Distributed;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Propagate packed tensor layouts across internal high-level function boundaries.
/// </summary>
public sealed class FunctionBoundaryLayoutPropagationPass : ModulePass
{
    private const int MaxPlanningIterations = 8;
    private const int IdentityVariantPenalty = 1000;
    private const int LocalTransformCostScale = 10;
    private readonly bool _enableCallerOutputDemandLayouts;
    private readonly bool _enableInternalOutputLayouts;

    public FunctionBoundaryLayoutPropagationPass(
        bool enableCallerOutputDemandLayouts = true,
        bool enableInternalOutputLayouts = true)
    {
        _enableCallerOutputDemandLayouts = enableCallerOutputDemandLayouts;
        _enableInternalOutputLayouts = enableInternalOutputLayouts;
    }

    private enum BoundaryTransformKind
    {
        Pack,
        Unpack,
        Transpose,
        Bitcast,
        Boxing,
    }

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var reshardContext = CompileSession.CompileOptions.TargetOptions is INTTTargetOptions targetOptions
            ? new BoundaryReshardContext(targetOptions, CompileSession.Target.Name)
            : null;
        var compilerInsertedRestoreTransforms = new HashSet<Call>(ReferenceEqualityComparer.Instance);
        for (int iteration = 0; iteration < MaxPlanningIterations; iteration++)
        {
            var planningFunctions = GetPlanningFunctions(input);
            var callSites = FunctionCallGraphUtility.CollectCallSites(planningFunctions);
            var specializer = new FunctionLayoutSpecializer(input, reshardContext, compilerInsertedRestoreTransforms);
            var enableCallerOutputDemandLayouts = _enableCallerOutputDemandLayouts && iteration == 0;
            var selectedLayouts = ModuleLayoutPlanner.Plan(
                input,
                planningFunctions,
                callSites,
                specializer,
                compilerInsertedRestoreTransforms,
                enableCallerOutputDemandLayouts,
                _enableInternalOutputLayouts);
            if (selectedLayouts.Count == 0)
            {
                return Task.FromResult(input);
            }

            var anyMutated = false;
            foreach (var function in planningFunctions)
            {
                var rewriteTarget = selectedLayouts.TryGetValue(function, out var selectedLayout)
                    ? specializer.GetOrCreate(function, selectedLayout)
                    : function;
                var rewriter = new CallBoundaryLayoutRewriter(selectedLayouts, specializer, reshardContext);
                rewriter.Rewrite(rewriteTarget);
                if (rewriter.IsMutated)
                {
                    anyMutated = true;
                    if (!CompilerServices.InferenceType(rewriteTarget))
                    {
                        throw new InvalidOperationException($"Type inference failed after propagating function boundary layouts in {rewriteTarget.Name}.");
                    }
                }

                var cleanup = new BoundaryTransformCleanupRewriter();
                cleanup.Rewrite(rewriteTarget);
                if (cleanup.IsMutated)
                {
                    anyMutated = true;
                    if (!CompilerServices.InferenceType(rewriteTarget))
                    {
                        throw new InvalidOperationException($"Type inference failed after cleaning function boundary layouts in {rewriteTarget.Name}.");
                    }
                }
            }

            anyMutated |= specializer.HasReplacements;
            specializer.CommitReplacements();
            if (!anyMutated)
            {
                return Task.FromResult(input);
            }
        }

        var remainingFunctions = GetPlanningFunctions(input);
        var remainingLayoutMap = ModuleLayoutPlanner.CollectCandidateLayouts(
            input,
            remainingFunctions,
            FunctionCallGraphUtility.CollectCallSites(remainingFunctions),
            compilerInsertedRestoreTransforms,
            _enableCallerOutputDemandLayouts,
            _enableInternalOutputLayouts);
        var remainingLayoutFunctions = remainingLayoutMap.Keys.ToArray();
        var remainingLayouts = string.Join(", ", remainingLayoutMap.Select(kv => $"{kv.Key.Name}: {string.Join(" | ", kv.Value.Select(value => value.ToString()))}"));
        var lastSignature = remainingLayoutFunctions.LastOrDefault() is { } last
            ? string.Join(", ", last.Parameters.ToArray().Select(parameter => $"{parameter.Name}: {((BaseExpr)parameter).CheckedType}"))
            : string.Empty;
        var lastBody = remainingLayoutFunctions.LastOrDefault() is { } lastBodyFunction ? CompilerServices.Print(lastBodyFunction.Body) : string.Empty;
        throw new InvalidOperationException($"Function boundary layout planning did not converge within {MaxPlanningIterations} iterations. Remaining layout functions: {remainingLayouts}. Last signature: {lastSignature}. Last body: {lastBody}");
    }

    private Function[] GetPlanningFunctions(IRModule module)
    {
        if (CompileSession.ActiveModuleKind is not { } activeModuleKind)
        {
            return module.Functions.OfType<Function>().ToArray();
        }

        return module.Functions
            .OfType<Function>()
            .Where(function =>
                function.Role != FunctionRole.ModuleDispatch &&
                string.Equals(function.ModuleKind, activeModuleKind, StringComparison.Ordinal))
            .ToArray();
    }

    private static Call[] GetFunctionCalls(Function function)
    {
        var calls = new HashSet<Call>(ReferenceEqualityComparer.Instance);
        foreach (var user in function.Users)
        {
            if (user is Call call &&
                TryGetTargetFunction(call.Target, out var directTarget, out _) &&
                ReferenceEquals(directTarget, function))
            {
                calls.Add(call);
                continue;
            }

            if (user is not FunctionWrapper { ReturnOutput: true, Target: Function wrapperTarget } wrapper ||
                !ReferenceEquals(wrapperTarget, function))
            {
                continue;
            }

            foreach (var wrapperCall in wrapper.Users.OfType<Call>())
            {
                if (ReferenceEquals(wrapperCall.Target, wrapper))
                {
                    calls.Add(wrapperCall);
                }
            }
        }

        return calls.ToArray();
    }

    private static bool TryGetTargetFunction(Expr target, out Function function, out FunctionWrapper? wrapper)
    {
        switch (target)
        {
            case Function fn:
                function = fn;
                wrapper = null;
                return true;
            case FunctionWrapper { ReturnOutput: true, Target: Function fn } fw:
                function = fn;
                wrapper = fw;
                return true;
            default:
                function = null!;
                wrapper = null;
                return false;
        }
    }

    private static bool TryStripInverseFromArgument(BaseExpr expr, BoundaryTransform transform, out BaseExpr transformed)
    {
        if (transform.TryStripInverse(expr, out transformed))
        {
            return true;
        }

        if (expr is Call { Target: GetItem, Arguments: var getItemArgs }
            && getItemArgs[GetItem.Input.Index] is IR.Tuple tuple
            && TryGetItemIndex(getItemArgs[GetItem.Index.Index], out var index)
            && index >= 0
            && index < tuple.Count)
        {
            return TryStripInverseFromArgument(tuple.Fields[index], transform, out transformed);
        }

        transformed = null!;
        return false;
    }

    private static bool TryGetItemIndex(BaseExpr expr, out int index)
    {
        switch (expr)
        {
            case DimConst dim:
                index = checked((int)dim.Value);
                return true;
            case TensorConst { Value: { Rank: 0 } tensor }:
                index = checked((int)tensor.ToScalar<long>());
                return true;
            default:
                index = -1;
                return false;
        }
    }

    private static BaseExpr MakeBoundaryTransform(
        Expr input,
        BoundaryTransform transform,
        BoundaryReshardContext? reshardContext,
        DistributedReshardUsageKind usageKind = DistributedReshardUsageKind.FunctionBoundary)
    {
        var expr = transform.Apply(input, reshardContext, usageKind);
        Infer(expr, $"{transform.Kind} inserted at function boundary");
        return expr;
    }

    private static BaseExpr MakeInverseBoundaryTransform(Expr input, BoundaryTransform transform, IRType originalType)
    {
        var expr = transform.ApplyInverse(input, originalType);
        Infer(expr, $"inverse {transform.Kind} inserted for raw parameter use");
        return expr;
    }

    private static void Infer(BaseExpr expr, string context)
    {
        if (!CompilerServices.InferenceType(expr))
        {
            throw new InvalidOperationException($"Type inference failed for {context}.");
        }
    }

    private static bool TryFoldPackUnpack(Call expr, out BaseExpr folded)
    {
        if (expr.Target is Pack pack
            && BoundaryTransform.Pack(pack).TryStripInverse(expr.Arguments[Pack.Input.Index], out folded))
        {
            return true;
        }

        if (expr.Target is Unpack unpack
            && BoundaryTransform.Unpack(unpack).TryStripInverse(expr.Arguments[Unpack.Input.Index], out folded))
        {
            return true;
        }

        folded = null!;
        return false;
    }

    private static bool TryFoldNoOpBoxing(Call expr, out BaseExpr folded)
    {
        if (expr.Target is IR.Distributed.Boxing boxing
            && expr[IR.Distributed.Boxing.Input] is Expr input
            && Equals(input.CheckedType, boxing.NewType))
        {
            folded = input;
            return true;
        }

        folded = null!;
        return false;
    }

    private sealed class ModuleLayoutPlanner
    {
        public static IReadOnlyDictionary<Function, FunctionBoundaryLayout> Plan(
            IRModule module,
            IReadOnlyList<Function> planningFunctions,
            IReadOnlyDictionary<Function, List<Call>> callSites,
            FunctionLayoutSpecializer specializer,
            IReadOnlySet<Call> compilerInsertedRestoreTransforms,
            bool enableCallerOutputDemandLayouts,
            bool enableInternalOutputLayouts)
        {
            var candidates = CollectCandidateLayouts(
                module,
                planningFunctions,
                callSites,
                compilerInsertedRestoreTransforms,
                enableCallerOutputDemandLayouts,
                enableInternalOutputLayouts);
            if (candidates.Count == 0)
            {
                return new Dictionary<Function, FunctionBoundaryLayout>(ReferenceEqualityComparer.Instance);
            }

            var model = new CpModel();
            var variables = new Dictionary<FunctionLayoutVariant, BoolVar>();
            foreach (var perFunction in candidates)
            {
                var choiceVars = new List<BoolVar>();
                foreach (var layout in perFunction.Value)
                {
                    var variant = new FunctionLayoutVariant(perFunction.Key, layout);
                    var variable = model.NewBoolVar($"{SanitizeName(perFunction.Key.Name)}_{choiceVars.Count}");
                    variables.Add(variant, variable);
                    choiceVars.Add(variable);
                }

                model.Add(LinearExpr.Sum(choiceVars) == 1);
            }

            var weightedVars = new List<BoolVar>();
            var weights = new List<long>();
            foreach (var (variant, variable) in variables)
            {
                weightedVars.Add(variable);
                weights.Add(EstimateVariantCost(
                    variant.Function,
                    variant.Layout,
                    specializer,
                    callSites));
            }

            model.Minimize(LinearExpr.WeightedSum(weightedVars, weights));
            var validation = model.Validate();
            if (!string.IsNullOrEmpty(validation))
            {
                throw new InvalidOperationException($"Function boundary layout CP-SAT model is invalid: {validation}");
            }

            var solver = new CpSolver
            {
                StringParameters = $"max_time_in_seconds:{GetSolveMaxTimeSeconds()},num_workers:{Math.Max(Environment.ProcessorCount / 2, 1)}",
            };
            var status = solver.Solve(model);
            if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Feasible))
            {
                throw new InvalidOperationException($"Function boundary layout CP-SAT solve failed: {status}.");
            }

            var selected = new Dictionary<Function, FunctionBoundaryLayout>(ReferenceEqualityComparer.Instance);
            foreach (var (variant, variable) in variables)
            {
                if (solver.BooleanValue(variable) && !variant.Layout.IsIdentity)
                {
                    selected.Add(variant.Function, variant.Layout);
                }
            }

            return selected;
        }

        public static Dictionary<Function, FunctionBoundaryLayout[]> CollectCandidateLayouts(
            IRModule module,
            IReadOnlyList<Function> planningFunctions,
            IReadOnlyDictionary<Function, List<Call>> callSites,
            IReadOnlySet<Call>? compilerInsertedRestoreTransforms = null,
            bool enableCallerOutputDemandLayouts = true,
            bool enableInternalOutputLayouts = true)
        {
            var layouts = new Dictionary<Function, FunctionBoundaryLayout[]>(ReferenceEqualityComparer.Instance);
            foreach (var function in planningFunctions)
            {
                if (ReferenceEquals(function, module.Entry) || function.IsEntry || !callSites.TryGetValue(function, out var functionCallSites))
                {
                    continue;
                }

                var candidates = new List<FunctionBoundaryLayout> { FunctionBoundaryLayout.Identity(function) };
                var internalLayout = FunctionBoundaryLayout.TryCreate(
                    function,
                    compilerInsertedRestoreTransforms,
                    enableInternalOutputLayouts);
                if (internalLayout is not null)
                {
                    candidates.Add(internalLayout);
                }

                if (enableCallerOutputDemandLayouts)
                {
                    foreach (var demandLayout in CollectCallDemandLayouts(function, functionCallSites, internalLayout))
                    {
                        candidates.Add(demandLayout);
                    }
                }

                var distinct = candidates
                    .Distinct()
                    .OrderBy(candidate => candidate.IsIdentity ? 0 : 1)
                    .ThenBy(candidate => candidate.ToString(), StringComparer.Ordinal)
                    .ToArray();
                if (distinct.Length > 1)
                {
                    layouts.Add(function, distinct);
                }
            }

            return layouts;
        }

        private static IEnumerable<FunctionBoundaryLayout> CollectCallDemandLayouts(
            Function function,
            IReadOnlyList<Call> calls,
            FunctionBoundaryLayout? internalLayout)
        {
            foreach (var call in calls)
            {
                var outputs = FunctionBoundaryLayout.CreateEmptyOutputs(call.CheckedType);
                if (call.CheckedType is TupleType tupleType)
                {
                    foreach (var user in call.Users.OfType<Call>())
                    {
                        if (user.Target is GetItem
                            && ReferenceEquals(user.Arguments[GetItem.Input.Index], call)
                            && TryGetItemIndex(user.Arguments[GetItem.Index.Index], out var outputIndex)
                            && outputIndex >= 0
                            && outputIndex < tupleType.Count)
                        {
                            TryCollectOutputDemand(user, outputIndex, outputs);
                        }
                    }
                }
                else
                {
                    TryCollectOutputDemand(call, 0, outputs);
                }

                if (outputs.Any(output => output is not null))
                {
                    yield return FunctionBoundaryLayout.Merge(function, internalLayout, outputs);
                }
            }
        }

        private static void TryCollectOutputDemand(BaseExpr value, int outputIndex, PortLayout?[] outputs)
        {
            foreach (var user in value.Users.OfType<Call>())
            {
                if (BoundaryTransform.TryCreate(user, out var consumerTransform, out var input)
                    && ReferenceEquals(input, value))
                {
                    if (consumerTransform.TryCreateRestoreTransform(value.CheckedType, out var restoreTransform))
                    {
                        outputs[outputIndex] = new PortLayout(restoreTransform, user.CheckedType, consumerTransform);
                        return;
                    }

                    if (consumerTransform.IsMaterializingCollective(value.CheckedType)
                        && AllValueConsumersMatch(value, consumerTransform))
                    {
                        outputs[outputIndex] = new PortLayout(
                            consumerTransform,
                            user.CheckedType,
                            consumerTransform,
                            restoreAtCaller: false);
                        return;
                    }
                }
            }
        }

        private static bool AllValueConsumersMatch(BaseExpr value, BoundaryTransform expectedTransform)
        {
            var consumers = value.Users.Where(user => user is not BaseFunction).ToArray();
            return consumers.Length != 0 && consumers.All(user =>
                user is Call call
                && BoundaryTransform.TryCreate(call, out var transform, out var input)
                && ReferenceEquals(input, value)
                && transform.Equals(expectedTransform));
        }

        private static long EstimateVariantCost(
            Function function,
            FunctionBoundaryLayout layout,
            FunctionLayoutSpecializer specializer,
            IReadOnlyDictionary<Function, List<Call>> callSites)
        {
            var callCount = callSites.TryGetValue(function, out var calls)
                ? Math.Max(calls.Count, 1)
                : 1;
            if (layout.IsIdentity)
            {
                return checked((IdentityVariantPenalty + (CountBoundaryTransforms(function.Body) * LocalTransformCostScale)) * callCount);
            }

            var specialized = specializer.CreateDetached(function, layout);
            var localCost = CountBoundaryTransforms(specialized.Body) * LocalTransformCostScale;
            var adapterCost = layout.AdapterTransformCount;
            var satisfiedDemandBenefit = CountSatisfiedCallerOutputDemands(function, layout) * LocalTransformCostScale;
            return checked(((localCost + adapterCost) * callCount) - satisfiedDemandBenefit);
        }

        private static int CountSatisfiedCallerOutputDemands(
            Function function,
            FunctionBoundaryLayout layout)
        {
            var count = 0;
            foreach (var call in GetFunctionCalls(function))
            {
                if (call.CheckedType is TupleType tupleType)
                {
                    foreach (var projection in call.Users.OfType<Call>())
                    {
                        if (projection.Target is not GetItem ||
                            !ReferenceEquals(projection[GetItem.Input], call) ||
                            !TryGetItemIndex(projection[GetItem.Index], out var outputIndex) ||
                            outputIndex < 0 ||
                            outputIndex >= tupleType.Count)
                        {
                            continue;
                        }

                        count += CountMatchingDemandTransforms(
                            projection,
                            layout.Outputs[outputIndex]?.SourceTransform);
                    }
                }
                else
                {
                    count += CountMatchingDemandTransforms(call, layout.Outputs[0]?.SourceTransform);
                }
            }

            return count;
        }

        private static int CountMatchingDemandTransforms(
            BaseExpr value,
            BoundaryTransform? sourceTransform)
        {
            if (sourceTransform is null)
            {
                return 0;
            }

            return value.Users.OfType<Call>().Count(user =>
                BoundaryTransform.TryCreate(user, out var transform, out var input) &&
                ReferenceEquals(input, value) &&
                transform.Equals(sourceTransform));
        }

        private static int CountBoundaryTransforms(BaseExpr expr)
        {
            return ExprCollector.Collect(expr)
                .OfType<Call>()
                .Count(call => BoundaryTransform.TryCreate(call, out _, out _));
        }

        private static int GetSolveMaxTimeSeconds()
        {
            if (Environment.GetEnvironmentVariable("NNCASE_LAYOUT_SOLVE_MAX_TIME") is { } value
                && int.TryParse(value, out var seconds)
                && seconds > 0)
            {
                return seconds;
            }

            return 10;
        }

        private static string SanitizeName(string name)
        {
            return new string(name.Select(ch => char.IsLetterOrDigit(ch) ? ch : '_').ToArray());
        }

        private sealed record FunctionLayoutVariant(Function Function, FunctionBoundaryLayout Layout);
    }

    private sealed class BoundaryTransformCleanupRewriter : ExprRewriter
    {
        public BoundaryTransformCleanupRewriter()
            : base(visitOtherFunctions: false)
        {
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
        {
            if (FunctionBoundaryLayoutPropagationPass.TryFoldPackUnpack(expr, out var folded))
            {
                SetMutated();
                return folded;
            }

            if (TryFoldBoxingOfInverseShardedView(expr, out folded))
            {
                SetMutated();
                return folded;
            }

            if (FunctionBoundaryLayoutPropagationPass.TryFoldNoOpBoxing(expr, out folded))
            {
                SetMutated();
                return folded;
            }

            return expr;
        }

        private static bool TryFoldBoxingOfInverseShardedView(Call expr, out BaseExpr folded)
        {
            if (expr.Target is IR.Distributed.Boxing boxing &&
                expr[IR.Distributed.Boxing.Input] is Call { Target: IR.Distributed.ShardedView } view &&
                view[IR.Distributed.ShardedView.Input] is Expr viewSource &&
                Equals(boxing.NewType, viewSource.CheckedType))
            {
                folded = viewSource;
                return true;
            }

            folded = null!;
            return false;
        }
    }

    private sealed class CallBoundaryLayoutRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<Function, FunctionBoundaryLayout> _layouts;
        private readonly FunctionLayoutSpecializer _specializer;
        private readonly BoundaryReshardContext? _reshardContext;

        public CallBoundaryLayoutRewriter(
            IReadOnlyDictionary<Function, FunctionBoundaryLayout> layouts,
            FunctionLayoutSpecializer specializer,
            BoundaryReshardContext? reshardContext)
            : base(visitOtherFunctions: false)
        {
            _layouts = layouts;
            _specializer = specializer;
            _reshardContext = reshardContext;
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
        {
            if (TryFoldPackUnpack(expr, out var folded))
            {
                return folded;
            }

            if (FunctionBoundaryLayoutPropagationPass.TryFoldNoOpBoxing(expr, out folded))
            {
                return folded;
            }

            if (!TryGetTargetFunction(expr.Target, out var function, out var wrapper)
                || !_layouts.TryGetValue(function, out var layout))
            {
                return expr;
            }

            var specialized = _specializer.GetOrCreate(function, layout);
            Expr target = wrapper is null ? specialized : wrapper.With(target: specialized);
            Infer(target, $"specialized target {specialized.Name}");

            var args = expr.Arguments.ToArray();
            for (int i = 0; i < layout.Inputs.Length; i++)
            {
                var inputLayout = layout.Inputs[i];
                if (inputLayout is null)
                {
                    continue;
                }

                if (TryStripInverseFromArgument(args[i], inputLayout.Transform, out var transformedArg))
                {
                    args[i] = transformedArg;
                    continue;
                }

                if (args[i] is not Expr argExpr)
                {
                    throw new InvalidOperationException($"Cannot pack non-expression argument {i} for call to {function.Name}.");
                }

                args[i] = MakeBoundaryTransform(argExpr, inputLayout.Transform, _reshardContext);
            }

            var rawCall = expr.With(target: target, arguments: args);
            Infer(rawCall, $"call to specialized function {specialized.Name}");

            return WrapOutputs(rawCall, layout);
        }

        private static bool TryGetTargetFunction(Expr target, out Function function, out FunctionWrapper? wrapper)
        {
            switch (target)
            {
                case Function fn:
                    function = fn;
                    wrapper = null;
                    return true;
                case FunctionWrapper { ReturnOutput: true, Target: Function fn } fw:
                    function = fn;
                    wrapper = fw;
                    return true;
                default:
                    function = null!;
                    wrapper = null;
                    return false;
            }
        }

        private static bool TryFoldPackUnpack(Call expr, out BaseExpr folded)
        {
            return FunctionBoundaryLayoutPropagationPass.TryFoldPackUnpack(expr, out folded);
        }

        private BaseExpr WrapOutputs(Call rawCall, FunctionBoundaryLayout layout)
        {
            if (!layout.HasOutputLayout)
            {
                return rawCall;
            }

            if (rawCall.CheckedType is not TupleType)
            {
                var outputLayout = layout.Outputs[0] ?? throw new InvalidOperationException("Single-output call has no output layout to wrap.");
                if (!outputLayout.RestoreAtCaller)
                {
                    return rawCall;
                }

                return MakeBoundaryTransform(
                    rawCall,
                    outputLayout.Transform,
                    _reshardContext,
                    DistributedReshardUsageKind.Internal);
            }

            var fields = new BaseExpr[layout.Outputs.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                BaseExpr field = IR.F.Tensors.GetItem(rawCall, i);
                Infer(field, $"tuple field {i} from {rawCall.Target}");
                if (layout.Outputs[i] is { RestoreAtCaller: true } outputLayout)
                {
                    if (field is not Expr fieldExpr)
                    {
                        throw new InvalidOperationException($"Cannot unpack non-expression output {i} from call to {rawCall.Target}.");
                    }

                    field = MakeBoundaryTransform(
                        fieldExpr,
                        outputLayout.Transform,
                        _reshardContext,
                        DistributedReshardUsageKind.Internal);
                }

                fields[i] = field;
            }

            var tuple = new IR.Tuple(fields);
            Infer(tuple, "tuple output wrapper");
            return tuple;
        }
    }

    private sealed class FunctionLayoutSpecializer
    {
        private readonly IRModule _module;
        private readonly BoundaryReshardContext? _reshardContext;
        private readonly ISet<Call> _compilerInsertedRestoreTransforms;
        private readonly Dictionary<Function, Dictionary<FunctionBoundaryLayout, Function>> _cache = new(ReferenceEqualityComparer.Instance);

        public FunctionLayoutSpecializer(
            IRModule module,
            BoundaryReshardContext? reshardContext,
            ISet<Call> compilerInsertedRestoreTransforms)
        {
            _module = module;
            _reshardContext = reshardContext;
            _compilerInsertedRestoreTransforms = compilerInsertedRestoreTransforms;
        }

        public bool HasReplacements => _cache.Count != 0;

        public Function GetOrCreate(Function function, FunctionBoundaryLayout layout)
        {
            if (!_cache.TryGetValue(function, out var perFunction))
            {
                perFunction = new Dictionary<FunctionBoundaryLayout, Function>();
                _cache.Add(function, perFunction);
            }

            if (perFunction.TryGetValue(layout, out var existing))
            {
                return existing;
            }

            var specialized = Create(function, layout, _compilerInsertedRestoreTransforms);
            perFunction.Add(layout, specialized);
            return specialized;
        }

        public Function CreateDetached(Function function, FunctionBoundaryLayout layout) => Create(function, layout, null);

        public void CommitReplacements()
        {
            foreach (var (function, perFunction) in _cache)
            {
                if (perFunction.Count != 1)
                {
                    throw new InvalidOperationException($"Function {function.Name} has {perFunction.Count} layout replacements in one planning iteration.");
                }

                var replacement = perFunction.Values.Single();
                var index = _module.Functions.ToList().FindIndex(candidate => ReferenceEquals(candidate, function));
                if (index < 0)
                {
                    throw new InvalidOperationException($"Function {function.Name} is not in the module during layout replacement.");
                }

                _module.Replace(index, replacement);
            }
        }

        private Function Create(
            Function function,
            FunctionBoundaryLayout layout,
            ISet<Call>? compilerInsertedRestoreTransforms)
        {
            var parameters = function.Parameters.ToArray();
            var mappedParameters = new Dictionary<Var, Var>(ReferenceEqualityComparer.Instance);
            var specializedParameters = new IVar[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is not Var parameter)
                {
                    if (layout.Inputs[i] is not null)
                    {
                        throw new InvalidOperationException($"Function boundary layout unexpectedly contains a tensor layout for non-Var parameter {i} in {function.Name}.");
                    }

                    specializedParameters[i] = parameters[i];
                    continue;
                }

                var type = layout.Inputs[i]?.TransformedType ?? parameter.CheckedType;
                var specializedParameter = parameter.With(typeAnnotation: type);
                mappedParameters.Add(parameter, specializedParameter);
                specializedParameters[i] = specializedParameter;
            }

            var packedInputs = new Dictionary<Var, PortLayout>(ReferenceEqualityComparer.Instance);
            for (int i = 0; i < layout.Inputs.Length; i++)
            {
                if (layout.Inputs[i] is { } inputLayout)
                {
                    packedInputs.Add((Var)parameters[i], inputLayout);
                }
            }

            var cloner = new SpecializedBodyCloner(
                mappedParameters,
                packedInputs,
                _reshardContext,
                compilerInsertedRestoreTransforms);
            var body = CloneBodyWithPackedOutputs(function.Body, layout, cloner);
            var varMap = RewriteVarMap(function.VarMap, mappedParameters);
            var specialized = new Function(function.Name, function.ModuleKind, body, specializedParameters, varMap)
            {
                Role = function.Role,
                Metadata = function.Metadata.Clone(),
            };
            if (!CompilerServices.InferenceType(specialized))
            {
                throw new InvalidOperationException($"Type inference failed for specialized function {specialized.Name} generated from {function.Name}.");
            }

            return specialized;
        }

        private BaseExpr CloneBodyWithPackedOutputs(BaseExpr body, FunctionBoundaryLayout layout, SpecializedBodyCloner cloner)
        {
            var fields = FunctionBoundaryLayout.FlattenReturn(body);
            if (fields.Length != layout.Outputs.Length)
            {
                throw new InvalidOperationException("Function output layout metadata is out of sync with function body.");
            }

            var clonedFields = new BaseExpr[fields.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                var source = fields[i];
                if (layout.Outputs[i] is { SourceTransform: { } sourceTransform } outputLayout)
                {
                    if (source is not Expr sourceExpr)
                    {
                        throw new InvalidOperationException($"Cannot apply output source transform {sourceTransform} to non-expression output.");
                    }

                    source = sourceTransform.Apply(
                        sourceExpr,
                        _reshardContext,
                        DistributedReshardUsageKind.Internal);
                    Infer(source, $"output source transform {sourceTransform}");
                    if (!Equals(source.CheckedType, outputLayout.TransformedType))
                    {
                        throw new InvalidOperationException($"Output source transform {sourceTransform} produced {source.CheckedType}, expected {outputLayout.TransformedType}.");
                    }
                }
                else if (layout.Outputs[i] is { } strippedOutputLayout)
                {
                    if (strippedOutputLayout.Transform.TryStrip(source, out var transformedOutput))
                    {
                        source = transformedOutput;
                    }
                    else if (!Equals(source.CheckedType, strippedOutputLayout.TransformedType))
                    {
                        throw new InvalidOperationException($"Cannot strip output transform {strippedOutputLayout.Transform} from {source.CheckedType}; expected specialized output type {strippedOutputLayout.TransformedType}.");
                    }
                }

                clonedFields[i] = cloner.Clone(source, default);
            }

            if (body is IR.Tuple)
            {
                var tuple = new IR.Tuple(clonedFields);
                Infer(tuple, "specialized tuple body");
                return tuple;
            }

            return clonedFields[0];
        }

        private Dictionary<IVar, Dimension[]> RewriteVarMap(Dictionary<IVar, Dimension[]>? source, IReadOnlyDictionary<Var, Var> mappedParameters)
        {
            var result = new Dictionary<IVar, Dimension[]>();
            if (source is null)
            {
                return result;
            }

            foreach (var (key, value) in source)
            {
                if (key is Var var && mappedParameters.TryGetValue(var, out var mapped))
                {
                    result[mapped] = value;
                }
                else
                {
                    result[key] = value;
                }
            }

            return result;
        }
    }

    private sealed class SpecializedBodyCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Var, Var> _mappedParameters;
        private readonly Dictionary<Var, PortLayout> _packedInputs;
        private readonly BoundaryReshardContext? _reshardContext;
        private readonly ISet<Call>? _compilerInsertedRestoreTransforms;

        public SpecializedBodyCloner(
            IReadOnlyDictionary<Var, Var> mappedParameters,
            Dictionary<Var, PortLayout> packedInputs,
            BoundaryReshardContext? reshardContext,
            ISet<Call>? compilerInsertedRestoreTransforms)
            : base(cloneOtherFunctions: false)
        {
            _mappedParameters = mappedParameters;
            _packedInputs = packedInputs;
            _reshardContext = reshardContext;
            _compilerInsertedRestoreTransforms = compilerInsertedRestoreTransforms;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (expr is Call call
                && BoundaryTransform.TryCreate(call, out var transform, out var input)
                && input is Var parameter
                && _packedInputs.TryGetValue(parameter, out var inputLayout))
            {
                if (inputLayout.Transform.Equals(transform))
                {
                    return _mappedParameters[parameter];
                }

                if (inputLayout.Transform.IsDistributedBoxing
                    && transform.IsDistributedBoxing)
                {
                    return MakeBoundaryTransform(
                        _mappedParameters[parameter],
                        transform,
                        _reshardContext,
                        DistributedReshardUsageKind.Internal);
                }
            }

            return base.DispatchVisit(expr, context);
        }

        protected override BaseExpr VisitLeafVar(Var expr, Unit context)
        {
            if (!_mappedParameters.TryGetValue(expr, out var mapped))
            {
                return base.VisitLeafVar(expr, context);
            }

            if (_packedInputs.TryGetValue(expr, out var inputLayout))
            {
                var restore = MakeInverseBoundaryTransform(
                    mapped,
                    inputLayout.Transform,
                    expr.CheckedType);
                if (restore is Call restoreCall)
                {
                    _compilerInsertedRestoreTransforms?.Add(restoreCall);
                }

                return restore;
            }

            return mapped;
        }

        protected override BaseExpr VisitLeafDimVar(DimVar expr, Unit context) => expr;
    }

    private sealed class FunctionBoundaryLayout : IEquatable<FunctionBoundaryLayout>
    {
        public FunctionBoundaryLayout(PortLayout?[] inputs, PortLayout?[] outputs, bool isIdentity = false)
        {
            Inputs = inputs;
            Outputs = outputs;
            IsIdentity = isIdentity;
        }

        public PortLayout?[] Inputs { get; }

        public PortLayout?[] Outputs { get; }

        public bool IsIdentity { get; }

        public bool HasOutputLayout => Outputs.Any(x => x is not null);

        public int AdapterTransformCount => Inputs.Count(input => input is not null) + Outputs.Count(output => output is { RestoreAtCaller: true });

        public static FunctionBoundaryLayout Identity(Function function)
        {
            return new FunctionBoundaryLayout(new PortLayout?[function.Parameters.Length], CreateEmptyOutputs(function.Body.CheckedType), true);
        }

        public static FunctionBoundaryLayout? TryCreate(
            Function function,
            IReadOnlySet<Call>? compilerInsertedRestoreTransforms = null,
            bool enableOutputs = true)
        {
            var inputs = AnalyzeInputs(function, compilerInsertedRestoreTransforms);
            var outputs = enableOutputs
                ? AnalyzeOutputs(function.Body)
                : CreateEmptyOutputs(function.Body.CheckedType);
            if (inputs.All(x => x is null) && outputs.All(x => x is null))
            {
                return null;
            }

            return new FunctionBoundaryLayout(inputs, outputs);
        }

        public static FunctionBoundaryLayout Merge(Function function, FunctionBoundaryLayout? baseLayout, IReadOnlyList<PortLayout?> demandedOutputs)
        {
            var inputs = new PortLayout?[function.Parameters.Length];
            if (baseLayout is not null)
            {
                Array.Copy(baseLayout.Inputs, inputs, inputs.Length);
            }

            var outputs = CreateEmptyOutputs(function.Body.CheckedType);
            if (baseLayout is not null)
            {
                Array.Copy(baseLayout.Outputs, outputs, outputs.Length);
            }

            if (demandedOutputs.Count != outputs.Length)
            {
                throw new InvalidOperationException($"Output demand layout rank mismatch for function {function.Name}: expected {outputs.Length}, got {demandedOutputs.Count}.");
            }

            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] ??= demandedOutputs[i];
            }

            return new FunctionBoundaryLayout(inputs, outputs);
        }

        public static PortLayout?[] CreateEmptyOutputs(IRType returnType)
        {
            return returnType is TupleType tupleType ? new PortLayout?[tupleType.Count] : new PortLayout?[1];
        }

        public static BaseExpr[] FlattenReturn(BaseExpr body) => body is IR.Tuple tuple ? tuple.Fields.ToArray() : [body];

        public bool Equals(FunctionBoundaryLayout? other)
        {
            return other is not null
                && IsIdentity == other.IsIdentity
                && LayoutArraysEqual(Inputs, other.Inputs)
                && LayoutArraysEqual(Outputs, other.Outputs);
        }

        public override bool Equals(object? obj) => Equals(obj as FunctionBoundaryLayout);

        public override int GetHashCode()
        {
            var hash = default(HashCode);
            hash.Add(IsIdentity);
            AddLayoutArrayHash(ref hash, Inputs);
            AddLayoutArrayHash(ref hash, Outputs);
            return hash.ToHashCode();
        }

        public override string ToString()
        {
            return IsIdentity
                ? "identity"
                : $"inputs=[{string.Join(", ", Inputs.Select(x => x?.ToString() ?? "-"))}], outputs=[{string.Join(", ", Outputs.Select(x => x?.ToString() ?? "-"))}]";
        }

        private static PortLayout?[] AnalyzeInputs(
            Function function,
            IReadOnlySet<Call>? compilerInsertedRestoreTransforms)
        {
            var parameters = function.Parameters.ToArray();
            var result = new PortLayout?[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is not Var parameter)
                {
                    continue;
                }

                if (parameter.CheckedType is DistributedType)
                {
                    continue;
                }

                var candidates = new List<PortLayout>();
                foreach (var user in parameter.Users)
                {
                    if (ReferenceEquals(user, function))
                    {
                        continue;
                    }

                    if (user is Call call
                        && !(compilerInsertedRestoreTransforms?.Contains(call) ?? false)
                        && BoundaryTransform.TryCreateInputTransform(call, out var transform, out var input)
                        && ReferenceEquals(input, parameter))
                    {
                        candidates.Add(new PortLayout(transform, user.CheckedType));
                    }
                }

                result[i] = SelectCanonicalInputLayout(candidates);
            }

            return result;
        }

        private static PortLayout? SelectCanonicalInputLayout(IReadOnlyList<PortLayout> candidates)
        {
            if (candidates.Count == 0)
            {
                return null;
            }

            var first = candidates[0];
            if (candidates.All(candidate => candidate.Transform.Equals(first.Transform)))
            {
                return first;
            }

            if (!candidates.All(candidate => candidate.Transform.IsDistributedBoxing))
            {
                return null;
            }

            return candidates
                .GroupBy(candidate => candidate.Transform)
                .OrderByDescending(group => group.Count())
                .ThenBy(group => group.Key.ToString(), StringComparer.Ordinal)
                .Select(group => group.First())
                .First();
        }

        private static PortLayout?[] AnalyzeOutputs(BaseExpr body)
        {
            var fields = FlattenReturn(body);
            var result = new PortLayout?[fields.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                if (fields[i] is Call call
                    && BoundaryTransform.TryCreateOutputTransform(call, out var transform, out var input))
                {
                    result[i] = new PortLayout(transform, input.CheckedType);
                }
            }

            return result;
        }

        private static bool LayoutArraysEqual(IReadOnlyList<PortLayout?> lhs, IReadOnlyList<PortLayout?> rhs)
        {
            if (lhs.Count != rhs.Count)
            {
                return false;
            }

            for (int i = 0; i < lhs.Count; i++)
            {
                var lhsLayout = lhs[i];
                var rhsLayout = rhs[i];
                if (lhsLayout is null || rhsLayout is null)
                {
                    if (lhsLayout is not null || rhsLayout is not null)
                    {
                        return false;
                    }

                    continue;
                }

                if (!lhsLayout.Equals(rhsLayout))
                {
                    return false;
                }
            }

            return true;
        }

        private static void AddLayoutArrayHash(ref HashCode hash, IEnumerable<PortLayout?> values)
        {
            foreach (var value in values)
            {
                hash.Add(value);
            }
        }
    }

    private sealed class PortLayout : IEquatable<PortLayout>
    {
        public PortLayout(
            BoundaryTransform transform,
            IRType transformedType,
            BoundaryTransform? sourceTransform = null,
            bool restoreAtCaller = true)
        {
            Transform = transform;
            TransformedType = transformedType;
            SourceTransform = sourceTransform;
            RestoreAtCaller = restoreAtCaller;
        }

        public BoundaryTransform Transform { get; }

        public IRType TransformedType { get; }

        public BoundaryTransform? SourceTransform { get; }

        public bool RestoreAtCaller { get; }

        public bool Equals(PortLayout? other)
        {
            return other is not null
                && Transform.Equals(other.Transform)
                && Equals(TransformedType, other.TransformedType)
                && Equals(SourceTransform, other.SourceTransform)
                && RestoreAtCaller == other.RestoreAtCaller;
        }

        public override bool Equals(object? obj) => Equals(obj as PortLayout);

        public override int GetHashCode() => HashCode.Combine(Transform, TransformedType, SourceTransform, RestoreAtCaller);

        public override string ToString()
        {
            return SourceTransform is null
                ? $"{Transform}->{TransformedType}"
                : RestoreAtCaller
                    ? $"{SourceTransform}=>{TransformedType}; restore {Transform}"
                    : $"{SourceTransform}=>{TransformedType}; direct";
        }
    }

    private sealed class BoundaryTransform : IEquatable<BoundaryTransform>
    {
        private BoundaryTransform(BoundaryTransformKind kind, IReadOnlyList<int>? lanes = null, IReadOnlyList<int>? axes = null, IReadOnlyList<int>? perm = null, DataType? newType = null, IRType? newIRType = null)
        {
            Kind = kind;
            Lanes = lanes?.ToArray() ?? Array.Empty<int>();
            Axes = axes?.ToArray() ?? Array.Empty<int>();
            Perm = perm?.ToArray() ?? Array.Empty<int>();
            NewType = newType;
            NewIRType = newIRType;
        }

        public BoundaryTransformKind Kind { get; }

        public int[] Lanes { get; }

        public int[] Axes { get; }

        public int[] Perm { get; }

        public DataType? NewType { get; }

        public IRType? NewIRType { get; }

        public bool IsDistributedBoxing => Kind is BoundaryTransformKind.Boxing && NewIRType is DistributedType;

        public static BoundaryTransform Pack(Pack pack) => new(BoundaryTransformKind.Pack, lanes: pack.Lanes.ToArray(), axes: pack.Axes.ToArray());

        public static BoundaryTransform Unpack(Unpack unpack) => new(BoundaryTransformKind.Unpack, lanes: unpack.Lanes.ToArray(), axes: unpack.Axes.ToArray());

        public static BoundaryTransform Transpose(IReadOnlyList<int> perm) => new(BoundaryTransformKind.Transpose, perm: perm);

        public static BoundaryTransform Bitcast(Bitcast bitcast) => new(BoundaryTransformKind.Bitcast, newType: bitcast.NewType);

        public static BoundaryTransform Boxing(IR.Distributed.Boxing boxing) => new(BoundaryTransformKind.Boxing, newIRType: boxing.NewType);

        public static bool TryCreate(Call call, out BoundaryTransform transform, out BaseExpr input)
        {
            switch (call.Target)
            {
                case IR.Tensors.Pack pack:
                    transform = Pack(pack);
                    input = call.Arguments[IR.Tensors.Pack.Input.Index];
                    return true;
                case IR.Tensors.Unpack unpack:
                    transform = Unpack(unpack);
                    input = call.Arguments[IR.Tensors.Unpack.Input.Index];
                    return true;
                case IR.Tensors.Transpose when TryGetFixedPerm(call.Arguments[IR.Tensors.Transpose.Perm.Index], out var perm):
                    transform = Transpose(perm);
                    input = call.Arguments[IR.Tensors.Transpose.Input.Index];
                    return true;
                case IR.Tensors.Bitcast bitcast:
                    transform = Bitcast(bitcast);
                    input = call.Arguments[IR.Tensors.Bitcast.Input.Index];
                    return true;
                case IR.Distributed.Boxing boxing:
                    transform = Boxing(boxing);
                    input = call.Arguments[IR.Distributed.Boxing.Input.Index];
                    return true;
                default:
                    transform = null!;
                    input = null!;
                    return false;
            }
        }

        public static bool TryCreateInputTransform(Call call, out BoundaryTransform transform, out BaseExpr input)
        {
            if (TryCreate(call, out transform, out input)
                && (transform.Kind is BoundaryTransformKind.Pack
                    or BoundaryTransformKind.Unpack
                    or BoundaryTransformKind.Transpose
                    || (transform.Kind is BoundaryTransformKind.Boxing && transform.NewIRType is DistributedType)))
            {
                return true;
            }

            transform = null!;
            input = null!;
            return false;
        }

        public static bool TryCreateOutputTransform(Call call, out BoundaryTransform transform, out BaseExpr input)
        {
            if (TryCreate(call, out transform, out input)
                && transform.Kind is BoundaryTransformKind.Unpack or BoundaryTransformKind.Bitcast or BoundaryTransformKind.Transpose or BoundaryTransformKind.Boxing)
            {
                return true;
            }

            transform = null!;
            input = null!;
            return false;
        }

        public BaseExpr Apply(
            Expr input,
            BoundaryReshardContext? reshardContext,
            DistributedReshardUsageKind usageKind)
        {
            if (Kind is BoundaryTransformKind.Boxing)
            {
                var targetType = NewIRType ?? throw new InvalidOperationException("Boxing transform is missing target IR type.");
                if (Equals(input.CheckedType, targetType))
                {
                    return input;
                }

                if (reshardContext is { } context && targetType is DistributedType distributedType)
                {
                    var sourceKind = input switch
                    {
                        TensorConst => DistributedReshardSourceKind.Constant,
                        Var => DistributedReshardSourceKind.FunctionParameter,
                        _ => DistributedReshardSourceKind.Internal,
                    };
                    var realization = DistributedReshardRealizationPolicy.Get(context.TargetOptions).Classify(
                        new DistributedReshardRealizationContext(
                            context.TargetOptions,
                            context.ModuleKind,
                            input.CheckedType,
                            distributedType,
                            sourceKind,
                            usageKind));
                    switch (realization)
                    {
                        case DistributedReshardRealization.ShardedView:
                            return IR.F.Distributed.ShardedView(input, distributedType);
                        case DistributedReshardRealization.Boxing:
                            break;
                        case DistributedReshardRealization.Unsupported:
                            throw new InvalidOperationException(
                                $"Target {context.TargetOptions.GetType().Name} cannot realize function-boundary reshard " +
                                $"{input.CheckedType} -> {distributedType}.");
                        default:
                            throw new InvalidOperationException($"Unknown distributed reshard realization {realization}.");
                    }
                }
            }

            return Kind switch
            {
                BoundaryTransformKind.Pack => IR.F.Tensors.Pack(input, Lanes, Axes),
                BoundaryTransformKind.Unpack => IR.F.Tensors.Unpack(input, Lanes, Axes),
                BoundaryTransformKind.Transpose => IR.F.Tensors.Transpose(input, Perm),
                BoundaryTransformKind.Bitcast => IR.F.Tensors.Bitcast(input, NewType ?? throw new InvalidOperationException("Bitcast transform is missing new dtype.")),
                BoundaryTransformKind.Boxing => IR.F.Distributed.Boxing(input, NewIRType ?? throw new InvalidOperationException("Boxing transform is missing target IR type.")),
                _ => throw new InvalidOperationException($"Unsupported boundary transform kind: {Kind}."),
            };
        }

        public BaseExpr ApplyInverse(Expr input, IRType originalType)
        {
            return Kind switch
            {
                BoundaryTransformKind.Pack => IR.F.Tensors.Unpack(input, Lanes, Axes),
                BoundaryTransformKind.Unpack => IR.F.Tensors.Pack(input, Lanes, Axes),
                BoundaryTransformKind.Transpose => IR.F.Tensors.Transpose(input, InvertPerm(Perm)),
                BoundaryTransformKind.Bitcast => IR.F.Tensors.Bitcast(input, GetDType(originalType, "bitcast inverse transform")),
                BoundaryTransformKind.Boxing => IR.F.Distributed.Boxing(input, originalType),
                _ => throw new InvalidOperationException($"Unsupported boundary transform kind: {Kind}."),
            };
        }

        public bool TryStrip(BaseExpr expr, out BaseExpr input)
        {
            if (expr is Call call
                && TryCreate(call, out var transform, out input)
                && Equals(transform))
            {
                return true;
            }

            input = null!;
            return false;
        }

        public bool TryStripInverse(BaseExpr expr, out BaseExpr input)
        {
            if (expr is not Call call || !TryCreate(call, out var transform, out input))
            {
                input = null!;
                return false;
            }

            if (Kind == BoundaryTransformKind.Boxing
                && transform.Kind == BoundaryTransformKind.Boxing
                && Equals(input.CheckedType, NewIRType))
            {
                return true;
            }

            var inverse = Kind switch
            {
                BoundaryTransformKind.Pack when transform.Kind == BoundaryTransformKind.Unpack => new BoundaryTransform(BoundaryTransformKind.Pack, lanes: transform.Lanes, axes: transform.Axes),
                BoundaryTransformKind.Unpack when transform.Kind == BoundaryTransformKind.Pack => new BoundaryTransform(BoundaryTransformKind.Unpack, lanes: transform.Lanes, axes: transform.Axes),
                BoundaryTransformKind.Transpose when transform.Kind == BoundaryTransformKind.Transpose => Transpose(InvertPerm(transform.Perm)),
                _ => null,
            };

            if (Equals(inverse))
            {
                return true;
            }

            input = null!;
            return false;
        }

        public bool TryCreateRestoreTransform(IRType originalType, out BoundaryTransform restoreTransform)
        {
            restoreTransform = Kind switch
            {
                BoundaryTransformKind.Pack => new BoundaryTransform(BoundaryTransformKind.Unpack, lanes: Lanes, axes: Axes),
                BoundaryTransformKind.Unpack => new BoundaryTransform(BoundaryTransformKind.Pack, lanes: Lanes, axes: Axes),
                BoundaryTransformKind.Transpose => Transpose(InvertPerm(Perm)),
                BoundaryTransformKind.Bitcast => BitcastToType(originalType),
                BoundaryTransformKind.Boxing when CanProduceBoxingTarget(originalType) => BoxingToType(originalType),
                _ => null!,
            };

            return restoreTransform is not null;
        }

        public bool IsMaterializingCollective(IRType sourceType)
        {
            return Kind is BoundaryTransformKind.Boxing
                && sourceType is DistributedType sourceDistributed
                && HasPartial(sourceDistributed)
                && NewIRType is { } targetType
                && CanProduceBoxingTarget(targetType);
        }

        public bool Equals(BoundaryTransform? other)
        {
            return other is not null
                && Kind == other.Kind
                && Lanes.SequenceEqual(other.Lanes)
                && Axes.SequenceEqual(other.Axes)
                && Perm.SequenceEqual(other.Perm)
                && Equals(NewType, other.NewType)
                && Equals(NewIRType, other.NewIRType);
        }

        public override bool Equals(object? obj) => Equals(obj as BoundaryTransform);

        public override int GetHashCode()
        {
            var hash = default(HashCode);
            hash.Add(Kind);
            foreach (var lane in Lanes)
            {
                hash.Add(lane);
            }

            hash.Add(-1);
            foreach (var axis in Axes)
            {
                hash.Add(axis);
            }

            hash.Add(-2);
            foreach (var item in Perm)
            {
                hash.Add(item);
            }

            hash.Add(NewType);
            hash.Add(NewIRType);
            return hash.ToHashCode();
        }

        public override string ToString()
        {
            return Kind switch
            {
                BoundaryTransformKind.Pack => $"Pack([{string.Join(",", Lanes)}], [{string.Join(",", Axes)}])",
                BoundaryTransformKind.Unpack => $"Unpack([{string.Join(",", Lanes)}], [{string.Join(",", Axes)}])",
                BoundaryTransformKind.Transpose => $"Transpose([{string.Join(",", Perm)}])",
                BoundaryTransformKind.Bitcast => $"Bitcast({NewType})",
                BoundaryTransformKind.Boxing => $"Boxing({NewIRType})",
                _ => Kind.ToString(),
            };
        }

        private static bool CanProduceBoxingTarget(IRType type)
            => type switch
            {
                TensorType => true,
                DistributedType distributed => distributed.Partial is null &&
                    distributed.AxisPolicies.All(policy => policy is not SBPPartial),
                TupleType tuple => tuple.Fields.All(CanProduceBoxingTarget),
                _ => false,
            };

        private static bool HasPartial(DistributedType type)
            => type.Partial is not null || type.AxisPolicies.Any(policy => policy is SBPPartial);

        private static bool TryGetFixedPerm(BaseExpr expr, out int[] perm)
        {
            if (expr is Shape { IsFixed: true } shape)
            {
                perm = shape.ToValueArray().Select(value => checked((int)value)).ToArray();
                return true;
            }

            perm = Array.Empty<int>();
            return false;
        }

        private static int[] InvertPerm(IReadOnlyList<int> perm)
        {
            var inverse = new int[perm.Count];
            for (int i = 0; i < perm.Count; i++)
            {
                inverse[perm[i]] = i;
            }

            return inverse;
        }

        private static DataType GetDType(IRType type, string context)
        {
            return type switch
            {
                TensorType tensor => tensor.DType,
                DistributedType distributed => distributed.TensorType.DType,
                _ => throw new InvalidOperationException($"Cannot get tensor dtype for {context} from {type}."),
            };
        }

        private static BoundaryTransform BitcastToType(IRType type)
        {
            return new BoundaryTransform(BoundaryTransformKind.Bitcast, newType: GetDType(type, "bitcast restore transform"));
        }

        private static BoundaryTransform BoxingToType(IRType type)
        {
            return new BoundaryTransform(BoundaryTransformKind.Boxing, newIRType: type);
        }
    }

    private sealed record BoundaryReshardContext(
        INTTTargetOptions TargetOptions,
        string ModuleKind);
}
