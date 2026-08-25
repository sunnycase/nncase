// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Sinks partial-to-broadcast normalization-statistics boxing operations from
/// function call sites into the single <see cref="NormApply"/> consumer in the callee.
/// </summary>
/// <remarks>
/// AutoDistributed establishes the distributed types first. This pass only
/// relocates an already-selected P(Sum)-to-B reshard so TIR selection can place
/// the collective and NormApply in the same PrimFunction. For the first call in
/// a recurrent chain, <see cref="BindNormStats"/> supplies the semantic relation
/// needed to compute partial statistics from the callee-selected local view.
/// Call sites with different partial signatures use distinct callee variants;
/// every variant retains an exact distributed input type and materializes the
/// statistics inside its own body.
/// The function result ABI and the entry ABI remain unchanged.
/// </remarks>
public sealed class SinkNormStatsBoxingAcrossFunctionBoundariesPass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        if (input.Entry is null)
        {
            return Task.FromResult(input);
        }

        var functions = input.Functions.OfType<Function>().ToArray();
        var callSites = FunctionCallGraphUtility.CollectCallSites(functions);
        var plans = new Dictionary<Function, FunctionPlan>(ReferenceEqualityComparer.Instance);
        foreach (var function in functions)
        {
            if (ReferenceEquals(function, input.Entry)
                || !callSites.TryGetValue(function, out var calls)
                || calls.Count == 0)
            {
                continue;
            }

            if (CreateFunctionPlan(function, calls) is { } plan)
            {
                plans.Add(function, plan);
            }
        }

        if (plans.Count == 0)
        {
            return Task.FromResult(input);
        }

        var plannedCalls = new Dictionary<Call, PlannedCall>(ReferenceEqualityComparer.Instance);
        foreach (var plan in plans.Values)
        {
            foreach (var variant in plan.Variants)
            {
                foreach (var (call, arguments) in variant.CallArguments)
                {
                    plannedCalls.Add(call, new PlannedCall(variant, arguments));
                }
            }
        }

        var replacements = new Dictionary<Function, Function>(ReferenceEqualityComparer.Instance);
        var additionalVariants = new Dictionary<Function, IReadOnlyList<Function>>(ReferenceEqualityComparer.Instance);
        foreach (var function in FunctionCallGraphUtility.GetCalleeFirstOrder(functions))
        {
            plans.TryGetValue(function, out var plan);
            var variants = new List<Function>();
            var nextVariantIndex = 1;
            Function? primary = null;
            if (plan is null || plan.PlannedCallCount != callSites[function].Count)
            {
                primary = CreateFunctionVariant(
                    function,
                    function.Name,
                    new Dictionary<int, ParameterPlan>(),
                    replacements,
                    plannedCalls);
            }

            if (plan is not null)
            {
                foreach (var variant in plan.Variants)
                {
                    var name = primary is null
                        ? function.Name
                        : $"{function.Name}_norm_stats_layout_{nextVariantIndex}";
                    var specialized = CreateFunctionVariant(
                        function,
                        name,
                        variant.Parameters,
                        replacements,
                        plannedCalls);
                    variant.Function = specialized;
                    if (primary is null)
                    {
                        primary = specialized;
                    }
                    else
                    {
                        variants.Add(specialized);
                        nextVariantIndex++;
                    }
                }
            }

            replacements.Add(function, primary ?? throw new InvalidOperationException($"No replacement was created for {function.Name}."));
            additionalVariants.Add(function, variants);
        }

        var result = new IRModule();
        foreach (var function in input.Functions)
        {
            if (function is not Function highLevel)
            {
                result.Add(function);
                continue;
            }

            result.Add(replacements[highLevel]);
            foreach (var variant in additionalVariants[highLevel])
            {
                result.Add(variant);
            }
        }

        result.Entry = input.Entry is Function entry ? replacements[entry] : input.Entry;
        return Task.FromResult(result);
    }

    private static FunctionPlan? CreateFunctionPlan(
        Function function,
        IReadOnlyList<Call> callSites)
    {
        var bodyNodes = new HashSet<BaseExpr>(
            ExprCollector.Collect(function.Body).Append(function.Body),
            ReferenceEqualityComparer.Instance);
        var parameters = function.Parameters.ToArray();
        var candidates = new Dictionary<int, ParameterCandidate>();
        for (int parameterIndex = 0; parameterIndex < parameters.Length; parameterIndex++)
        {
            if (parameters[parameterIndex] is not Var parameter
                || parameter.CheckedType is not DistributedType materializedType
                || !TryGetNormStatsConsumer(function, parameter, bodyNodes, out var consumer))
            {
                continue;
            }

            if (consumer.Binding is not null)
            {
                var localStatsType = NormStatsEvaluator.InferType(
                    new NormStats(consumer.Axis, consumer.UseMean),
                    consumer.LocalInput.CheckedType);
                if (localStatsType is not DistributedType distributedStatsType
                    || !IsSupportedPartialType(distributedStatsType, materializedType))
                {
                    continue;
                }
            }

            candidates.Add(parameterIndex, new ParameterCandidate(materializedType, consumer));
        }

        if (candidates.Count == 0)
        {
            return null;
        }

        var variants = new List<FunctionVariantPlan>();
        foreach (var callSite in callSites)
        {
            if (callSite.Arguments.Length != parameters.Length)
            {
                continue;
            }

            var parameterPlans = new Dictionary<int, ParameterPlan>();
            var callArguments = new Dictionary<int, Expr>();
            foreach (var (parameterIndex, candidate) in candidates)
            {
                if (!TryGetPartialCallArgument(
                        callSite,
                        parameterIndex,
                        candidate.MaterializedType,
                        candidate.Consumer,
                        out var partialStats,
                        out var partialType))
                {
                    continue;
                }

                parameterPlans.Add(
                    parameterIndex,
                    new ParameterPlan(partialType, candidate.MaterializedType));
                callArguments.Add(parameterIndex, partialStats);
            }

            if (parameterPlans.Count == 0)
            {
                continue;
            }

            var variant = variants.FirstOrDefault(candidate => ParameterPlansEqual(candidate.Parameters, parameterPlans));
            if (variant is null)
            {
                variant = new FunctionVariantPlan(parameterPlans);
                variants.Add(variant);
            }

            variant.CallArguments.Add(callSite, callArguments);
        }

        return variants.Count == 0 ? null : new FunctionPlan(variants);
    }

    private static bool ParameterPlansEqual(
        IReadOnlyDictionary<int, ParameterPlan> left,
        IReadOnlyDictionary<int, ParameterPlan> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }

        foreach (var (index, leftPlan) in left)
        {
            if (!right.TryGetValue(index, out var rightPlan) || leftPlan != rightPlan)
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryGetNormStatsConsumer(
        Function function,
        Var parameter,
        IReadOnlySet<BaseExpr> bodyNodes,
        out NormStatsConsumer consumer)
    {
        consumer = null!;
        var users = parameter.Users.Where(bodyNodes.Contains).ToArray();
        if (users.Length != 1 || users[0] is not Call firstUser)
        {
            return false;
        }

        if (firstUser.Target is NormApply directNormApply
            && ReferenceEquals(firstUser[NormApply.Stats], parameter)
            && CountArgumentUses(firstUser, parameter) == 1)
        {
            consumer = new NormStatsConsumer(
                null,
                null,
                null,
                (Expr)firstUser[NormApply.Input],
                directNormApply.Axis,
                directNormApply.UseMean);
            return true;
        }

        if (firstUser.Target is not BindNormStats binding
            || !ReferenceEquals(firstUser[BindNormStats.Stats], parameter)
            || CountArgumentUses(firstUser, parameter) != 1
            || firstUser[BindNormStats.Input] is not Var sourceParameter)
        {
            return false;
        }

        var sourceParameterIndex = Array.FindIndex(
            function.Parameters.ToArray(),
            candidate => ReferenceEquals(candidate, sourceParameter));
        if (sourceParameterIndex < 0)
        {
            return false;
        }

        var bindingUsers = firstUser.Users.Where(bodyNodes.Contains).ToArray();
        if (bindingUsers.Length != 1
            || bindingUsers[0] is not Call { Target: NormApply normApply } normApplyCall
            || !ReferenceEquals(normApplyCall[NormApply.Stats], firstUser)
            || CountArgumentUses(normApplyCall, firstUser) != 1
            || binding.Axis != normApply.Axis
            || binding.UseMean != normApply.UseMean
            || normApplyCall[NormApply.Input] is not Expr localInput
            || !IsShardedViewOf(localInput, sourceParameter))
        {
            return false;
        }

        consumer = new NormStatsConsumer(
            firstUser,
            sourceParameter,
            sourceParameterIndex,
            localInput,
            binding.Axis,
            binding.UseMean);
        return true;
    }

    private static bool TryGetPartialCallArgument(
        Call callSite,
        int parameterIndex,
        DistributedType materializedType,
        NormStatsConsumer consumer,
        out Expr partialStats,
        out DistributedType partialType)
    {
        if (TryMatchStatsBoxing(
                callSite.Arguments[parameterIndex],
                materializedType,
                out partialStats,
                out partialType))
        {
            return true;
        }

        partialStats = null!;
        partialType = null!;
        if (consumer.Binding is not Call
            || consumer.SourceParameter is not Var sourceParameter
            || consumer.SourceParameterIndex is not int sourceParameterIndex
            || callSite.Arguments[parameterIndex] is not Call { Target: NormStats normStats } seedStats
            || !Equals(seedStats.CheckedType, materializedType)
            || normStats.Axis != consumer.Axis
            || normStats.UseMean != consumer.UseMean
            || !ReferenceEquals(
                seedStats[NormStats.Input],
                callSite.Arguments[sourceParameterIndex])
            || callSite.Arguments[sourceParameterIndex] is not Expr sourceArgument
            || !TryInstantiateShardedView(
                consumer.LocalInput,
                sourceParameter,
                sourceArgument,
                out var localInput))
        {
            return false;
        }

        var localStats = IR.F.NN.NormStats(consumer.Axis, localInput, consumer.UseMean);
        Infer(localStats, "constructing partial normalization-statistics seed");
        if (localStats.CheckedType is not DistributedType localStatsType
            || !IsSupportedPartialType(localStatsType, materializedType))
        {
            return false;
        }

        partialStats = localStats;
        partialType = localStatsType;
        return true;
    }

    private static bool IsShardedViewOf(Expr expression, Var source)
    {
        while (expression is Call { Target: ShardedView } view)
        {
            expression = (Expr)view[ShardedView.Input];
        }

        return ReferenceEquals(expression, source);
    }

    private static bool TryInstantiateShardedView(
        Expr template,
        Var source,
        Expr replacement,
        out Expr result)
    {
        if (ReferenceEquals(template, source))
        {
            result = replacement;
            return true;
        }

        if (template is not Call viewCall
            || viewCall.Target is not ShardedView view
            || viewCall[ShardedView.Input] is not Expr input
            || !TryInstantiateShardedView(input, source, replacement, out var rewrittenInput))
        {
            result = null!;
            return false;
        }

        var rewrittenView = IR.F.Distributed.ShardedView(rewrittenInput, view.NewType);
        Infer(rewrittenView, "instantiating a normalization input sharded view at a call site");
        result = rewrittenView;
        return true;
    }

    private static int CountArgumentUses(Call call, BaseExpr expression)
        => call.Arguments.ToArray().Count(argument => ReferenceEquals(argument, expression));

    private static bool TryMatchStatsBoxing(
        BaseExpr expression,
        DistributedType materializedType,
        out Expr partialStats,
        out DistributedType partialType)
    {
        partialStats = null!;
        partialType = null!;
        if (expression is not Call { Target: Boxing boxing } boxingCall
            || !Equals(boxing.NewType, materializedType)
            || !Equals(boxingCall.CheckedType, materializedType)
            || boxingCall[Boxing.Input] is not Expr input
            || input.CheckedType is not DistributedType inputType
            || !IsSupportedPartialType(inputType, materializedType))
        {
            return false;
        }

        partialStats = input;
        partialType = inputType;
        return true;
    }

    private static bool IsSupportedPartialType(
        DistributedType partialType,
        DistributedType materializedType)
        => partialType.Partial is { Op: ReduceOp.Sum } partial
            && partial.Axes.Count != 0
            && partial.Axes.Distinct().Count() == partial.Axes.Count
            && partial.Axes.All(axis => axis >= 0 && axis < partialType.Placement.Hierarchy.Count)
            && partialType.AxisPolicies.All(policy => policy is SBPBroadCast)
            && materializedType.Partial is null
            && materializedType.AxisPolicies.All(policy => policy is SBPBroadCast)
            && Equals(partialType.TensorType, materializedType.TensorType)
            && Equals(partialType.Placement, materializedType.Placement);

    private static Function CreateFunctionVariant(
        Function function,
        string name,
        IReadOnlyDictionary<int, ParameterPlan> parameterPlans,
        IReadOnlyDictionary<Function, Function> replacements,
        IReadOnlyDictionary<Call, PlannedCall> plannedCalls)
    {
        var parameters = function.Parameters.ToArray();
        var rewrittenParameters = new IVar[parameters.Length];
        var mappedParameters = new Dictionary<Var, Var>(ReferenceEqualityComparer.Instance);
        var substitutions = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        for (int index = 0; index < parameters.Length; index++)
        {
            if (parameters[index] is not Var parameter)
            {
                if (parameterPlans.ContainsKey(index))
                {
                    throw new InvalidOperationException(
                        $"NormStats boxing plan for {function.Name} references non-tensor parameter {index}.");
                }

                rewrittenParameters[index] = parameters[index];
                substitutions.Add((BaseExpr)parameters[index], (BaseExpr)parameters[index]);
                continue;
            }

            if (parameterPlans.TryGetValue(index, out var parameterPlan))
            {
                var rewrittenParameter = parameter.With(typeAnnotation: parameterPlan.PartialType);
                var materializedStats = IR.F.Distributed.Boxing(
                    rewrittenParameter,
                    parameterPlan.MaterializedType);
                Infer(materializedStats, $"sinking normalization-statistics boxing into {name}");

                rewrittenParameters[index] = rewrittenParameter;
                mappedParameters.Add(parameter, rewrittenParameter);
                substitutions.Add(parameter, materializedStats);
            }
            else
            {
                rewrittenParameters[index] = parameter;
                mappedParameters.Add(parameter, parameter);
                substitutions.Add(parameter, parameter);
            }
        }

        var cloner = new FunctionBodyCloner(replacements, plannedCalls);
        foreach (var (source, replacement) in substitutions)
        {
            cloner.ExprMemo.Add(source, replacement);
        }

        var body = cloner.Clone(function.Body, Unit.Default);
        if (parameterPlans.Count == 0
            && name == function.Name
            && ReferenceEquals(body, function.Body))
        {
            return function;
        }

        var variant = new Function(
            name,
            function.ModuleKind,
            body,
            rewrittenParameters,
            RewriteVarMap(function.VarMap, mappedParameters))
        {
            Role = function.Role,
            Metadata = function.Metadata.Clone(),
        };
        Infer(variant, $"rewriting normalization-statistics boundary of {function.Name} as {name}");
        return variant;
    }

    private static Dictionary<IVar, Dimension[]> RewriteVarMap(
        Dictionary<IVar, Dimension[]>? source,
        IReadOnlyDictionary<Var, Var> mappedParameters)
    {
        var result = new Dictionary<IVar, Dimension[]>();
        if (source is null)
        {
            return result;
        }

        foreach (var (key, value) in source)
        {
            result[key is Var var && mappedParameters.TryGetValue(var, out var mapped) ? mapped : key] = value;
        }

        return result;
    }

    private static void Infer(BaseExpr expression, string context)
    {
        if (!CompilerServices.InferenceType(expression))
        {
            throw new InvalidOperationException($"Type inference failed while {context}.");
        }

        if (expression.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException($"Type inference failed while {context}: {invalid}.");
        }
    }

    private sealed record NormStatsConsumer(
        Call? Binding,
        Var? SourceParameter,
        int? SourceParameterIndex,
        Expr LocalInput,
        int Axis,
        bool UseMean);

    private sealed record ParameterCandidate(
        DistributedType MaterializedType,
        NormStatsConsumer Consumer);

    private sealed record ParameterPlan(
        DistributedType PartialType,
        DistributedType MaterializedType);

    private sealed class FunctionVariantPlan
    {
        public FunctionVariantPlan(IReadOnlyDictionary<int, ParameterPlan> parameters)
        {
            Parameters = parameters;
        }

        public IReadOnlyDictionary<int, ParameterPlan> Parameters { get; }

        public Dictionary<Call, IReadOnlyDictionary<int, Expr>> CallArguments { get; } = new(ReferenceEqualityComparer.Instance);

        public Function? Function { get; set; }
    }

    private sealed class FunctionPlan
    {
        public FunctionPlan(IReadOnlyList<FunctionVariantPlan> variants)
        {
            Variants = variants;
        }

        public IReadOnlyList<FunctionVariantPlan> Variants { get; }

        public int PlannedCallCount => Variants.Sum(variant => variant.CallArguments.Count);
    }

    private sealed record PlannedCall(
        FunctionVariantPlan Variant,
        IReadOnlyDictionary<int, Expr> Arguments);

    private sealed class FunctionBodyCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Function, Function> _replacements;
        private readonly IReadOnlyDictionary<Call, PlannedCall> _plannedCalls;

        public FunctionBodyCloner(
            IReadOnlyDictionary<Function, Function> replacements,
            IReadOnlyDictionary<Call, PlannedCall> plannedCalls)
            : base(cloneOtherFunctions: false)
        {
            _replacements = replacements;
            _plannedCalls = plannedCalls;
            CloneUnmutated = false;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (expr is Function function && _replacements.TryGetValue(function, out var replacement))
            {
                return replacement;
            }

            return base.DispatchVisit(expr, context);
        }

        protected override BaseExpr VisitLeafCall(Call expr, Unit context)
        {
            var arguments = new BaseExpr[expr.Arguments.Length];
            Expr target;
            if (expr.Target is Function originalTarget
                && _plannedCalls.TryGetValue(expr, out var plannedCall))
            {
                target = plannedCall.Variant.Function
                    ?? throw new InvalidOperationException(
                        $"Normalization-statistics callee variant for {originalTarget.Name} was not materialized before its caller.");
                for (int index = 0; index < arguments.Length; index++)
                {
                    if (plannedCall.Arguments.TryGetValue(index, out var partialStats))
                    {
                        if (!plannedCall.Variant.Parameters.TryGetValue(index, out var parameterPlan)
                            || !Equals(partialStats.CheckedType, parameterPlan.PartialType))
                        {
                            throw new InvalidOperationException(
                                $"Normalization-statistics boundary plan for {originalTarget.Name} is out of sync at parameter {index}.");
                        }

                        arguments[index] = Clone(partialStats, context);
                    }
                    else
                    {
                        arguments[index] = Clone(expr.Arguments[index], context);
                    }
                }
            }
            else
            {
                target = (Expr)Clone(expr.Target, context);
                arguments = CloneArray(expr.Arguments, context);
            }

            if (ReferenceEquals(target, expr.Target)
                && arguments.Zip(expr.Arguments.ToArray()).All(pair => ReferenceEquals(pair.First, pair.Second)))
            {
                return expr;
            }

            var rewritten = expr.With(target: target, arguments: arguments);
            if (target is Function)
            {
                Infer(rewritten, $"rewriting call to {target}");
            }

            return rewritten;
        }
    }
}
