// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Threads normalization statistics through repeated internal function calls.
/// </summary>
/// <remarks>
/// A function that computes <see cref="NormStats"/> directly from one of its
/// tensor parameters can consume those statistics as an explicit parameter and
/// append the statistics of its corresponding state output. This keeps the
/// public entry ABI unchanged while exposing a producer-local NormStats to
/// target-dependent fusion passes.
/// </remarks>
public sealed class ThreadNormStatsAcrossFunctionBoundariesPass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        if (input.Entry is null)
        {
            return Task.FromResult(input);
        }

        var functions = input.Functions.OfType<Function>().ToArray();
        var moduleFunctions = new HashSet<Function>(functions, ReferenceEqualityComparer.Instance);
        var callSites = CollectCallSites(functions);
        var plans = new Dictionary<Function, ThreadingPlan>(ReferenceEqualityComparer.Instance);
        foreach (var function in functions)
        {
            if (!ReferenceEquals(function, input.Entry)
                && TryCreatePlan(function, callSites, out var plan))
            {
                plans.Add(function, plan);
            }
        }

        if (plans.Count == 0)
        {
            return Task.FromResult(input);
        }

        var replacements = new Dictionary<Function, Function>(ReferenceEqualityComparer.Instance);
        var plansByReplacement = new Dictionary<Function, ThreadingPlan>(ReferenceEqualityComparer.Instance);
        foreach (var function in GetCalleeFirstOrder(functions, moduleFunctions))
        {
            plans.TryGetValue(function, out var plan);
            Var? statsParameter = null;
            IVar[] parameters;
            if (plan is not null)
            {
                statsParameter = new Var(
                    $"{((Var)function.Parameters[plan.InputParameterIndex]).Name}_norm_stats",
                    plan.StatsType);
                parameters = function.Parameters.ToArray().Append<IVar>(statsParameter).ToArray();
            }
            else
            {
                parameters = function.Parameters.ToArray();
            }

            var cloner = new FunctionBodyCloner(
                replacements,
                plans,
                plansByReplacement,
                plan,
                statsParameter);
            foreach (var parameter in function.Parameters)
            {
                cloner.ExprMemo.Add((BaseExpr)parameter, (BaseExpr)parameter);
            }

            var body = cloner.Clone(function.Body, Unit.Default);
            if (plan is not null)
            {
                body = AppendOutputStats(body, plan);
            }

            if (plan is null && ReferenceEquals(body, function.Body))
            {
                replacements.Add(function, function);
                continue;
            }

            var replacement = new Function(
                function.Name,
                function.ModuleKind,
                body,
                parameters,
                function.VarMap)
            {
                Role = function.Role,
                Metadata = function.Metadata.Clone(),
            };
            if (!CompilerServices.InferenceType(replacement))
            {
                throw new InvalidOperationException(
                    $"Type inference failed for NormStats-threaded function {function.Name}.");
            }

            if (replacement.CheckedType is InvalidType invalidType)
            {
                throw new InvalidOperationException(
                    $"NormStats threading produced an invalid function {function.Name}: {invalidType}.");
            }

            replacements.Add(function, replacement);
            if (plan is not null)
            {
                plansByReplacement.Add(replacement, plan);
            }
        }

        var result = new IRModule();
        foreach (var function in input.Functions)
        {
            result.Add(function is Function highLevel ? replacements[highLevel] : function);
        }

        result.Entry = input.Entry is Function entry ? replacements[entry] : input.Entry;
        return Task.FromResult(result);
    }

    private static Dictionary<Function, List<Call>> CollectCallSites(IEnumerable<Function> functions)
    {
        var callSites = new Dictionary<Function, List<Call>>(ReferenceEqualityComparer.Instance);
        foreach (var function in functions)
        {
            foreach (var call in ExprCollector.Collect(function.Body).OfType<Call>())
            {
                if (call.Target is not Function target)
                {
                    continue;
                }

                if (!callSites.TryGetValue(target, out var calls))
                {
                    calls = new List<Call>();
                    callSites.Add(target, calls);
                }

                calls.Add(call);
            }
        }

        return callSites;
    }

    private static bool TryCreatePlan(
        Function function,
        IReadOnlyDictionary<Function, List<Call>> callSites,
        out ThreadingPlan plan)
    {
        plan = null!;
        if (!callSites.TryGetValue(function, out var calls) || calls.Count == 0)
        {
            return false;
        }

        var parameterIndices = new Dictionary<Var, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < function.Parameters.Length; i++)
        {
            if (function.Parameters[i] is Var parameterVar)
            {
                parameterIndices.Add(parameterVar, i);
            }
        }

        var candidates = (from call in ExprCollector.Collect(function.Body).OfType<Call>()
                          where call.Target is NormStats
                          let inputExpr = call[NormStats.Input]
                          where inputExpr is Var input && parameterIndices.ContainsKey(input)
                          let normStats = (NormStats)call.Target
                          select new
                          {
                              Call = call,
                              Input = (Var)inputExpr,
                              InputIndex = parameterIndices[(Var)inputExpr],
                              Axis = NormalizeAxis(normStats.Axis, inputExpr),
                              normStats.UseMean,
                          })
            .GroupBy(candidate => (candidate.InputIndex, candidate.Axis, candidate.UseMean))
            .ToArray();
        if (candidates.Length != 1)
        {
            return false;
        }

        var candidate = candidates[0];
        var parameter = candidate.First().Input;
        var statsType = candidate.First().Call.CheckedType;
        if (statsType is not TensorType || candidate.First().Call.CheckedDataType != DataTypes.Float32)
        {
            throw new InvalidOperationException(
                $"NormStats on parameter {parameter.Name} of {function.Name} must produce an FP32 tensor, got {statsType}.");
        }

        var outputCount = function.Body is IR.Tuple tuple ? tuple.Count : 1;
        if (!TryFindStateOutputIndex(function, calls, candidate.Key.InputIndex, candidate.Key.Axis, candidate.Key.UseMean, outputCount, out var outputIndex))
        {
            return false;
        }

        if (function.Body is IR.Tuple && !CallsUseTupleFieldsOnly(calls, outputCount))
        {
            return false;
        }

        var output = function.Body is IR.Tuple outputTuple ? outputTuple[outputIndex] : function.Body;
        if (output is not Expr outputExpr || output.CheckedType is not TensorType)
        {
            return false;
        }

        var outputStats = IR.F.NN.NormStats(candidate.Key.Axis, outputExpr, candidate.Key.UseMean);
        if (!CompilerServices.InferenceType(outputStats) || !Equals(outputStats.CheckedType, statsType))
        {
            return false;
        }

        plan = new ThreadingPlan(
            candidate.Key.InputIndex,
            outputIndex,
            outputCount,
            candidate.Key.Axis,
            candidate.Key.UseMean,
            statsType,
            new HashSet<Call>(candidate.Select(item => item.Call), ReferenceEqualityComparer.Instance));
        return true;
    }

    private static bool TryFindStateOutputIndex(
        Function function,
        IEnumerable<Call> calls,
        int inputParameterIndex,
        int axis,
        bool useMean,
        int outputCount,
        out int outputIndex)
    {
        if (outputCount == 1)
        {
            outputIndex = 0;
            return true;
        }

        var evidence = new HashSet<int>();
        foreach (var call in calls)
        {
            foreach (var getItem in call.Users.OfType<Call>().Where(user => user.Target is GetItem))
            {
                if (getItem[GetItem.Input] is not Call inputCall
                    || !ReferenceEquals(inputCall, call)
                    || getItem[GetItem.Index] is not DimConst index
                    || index.Value < 0
                    || index.Value >= outputCount)
                {
                    continue;
                }

                foreach (var user in getItem.Users.OfType<Call>())
                {
                    if (user.Target is Function target
                        && ReferenceEquals(target, function)
                        && ReferenceEquals(user.Arguments[inputParameterIndex], getItem))
                    {
                        evidence.Add((int)index.Value);
                    }
                    else if (user.Target is NormStats normStats
                             && ReferenceEquals(user[NormStats.Input], getItem)
                             && normStats.UseMean == useMean
                             && NormalizeAxis(normStats.Axis, getItem) == axis)
                    {
                        evidence.Add((int)index.Value);
                    }
                }
            }
        }

        if (evidence.Count == 1)
        {
            outputIndex = evidence.Single();
            return true;
        }

        var inputType = function.Parameters[inputParameterIndex].CheckedType;
        var fields = ((IR.Tuple)function.Body).Fields.ToArray();
        var typeMatches = Enumerable.Range(0, fields.Length)
            .Where(index => Equals(fields[index].CheckedType, inputType))
            .ToArray();
        if (evidence.Count == 0 && typeMatches.Length == 1)
        {
            outputIndex = typeMatches[0];
            return true;
        }

        outputIndex = -1;
        return false;
    }

    private static bool CallsUseTupleFieldsOnly(IEnumerable<Call> calls, int outputCount)
    {
        foreach (var call in calls)
        {
            var users = call.Users.ToArray();
            if (users.Length == 0
                || users.Any(user => user is not Call getItem
                    || getItem.Target is not GetItem
                    || !ReferenceEquals(getItem[GetItem.Input], call)
                    || getItem[GetItem.Index] is not DimConst index
                    || index.Value < 0
                    || index.Value >= outputCount))
            {
                return false;
            }
        }

        return true;
    }

    private static BaseExpr AppendOutputStats(BaseExpr body, ThreadingPlan plan)
    {
        var fields = body is IR.Tuple tuple ? tuple.Fields.ToArray() : new[] { body };
        if (fields.Length != plan.OutputCount || fields[plan.OutputFieldIndex] is not Expr output)
        {
            throw new InvalidOperationException("NormStats threading output plan is out of sync with the cloned function body.");
        }

        var stats = IR.F.NN.NormStats(plan.Axis, output, plan.UseMean);
        if (!CompilerServices.InferenceType(stats) || !Equals(stats.CheckedType, plan.StatsType))
        {
            throw new InvalidOperationException(
                $"Threaded output NormStats produced {stats.CheckedType}, expected {plan.StatsType}.");
        }

        return new IR.Tuple(fields.Append<BaseExpr>(stats).ToArray());
    }

    private static Function[] GetCalleeFirstOrder(
        IEnumerable<Function> functions,
        IReadOnlySet<Function> moduleFunctions)
    {
        var visited = new HashSet<Function>(ReferenceEqualityComparer.Instance);
        var active = new HashSet<Function>(ReferenceEqualityComparer.Instance);
        var order = new List<Function>();

        void Visit(Function function)
        {
            if (visited.Contains(function))
            {
                return;
            }

            if (!active.Add(function))
            {
                throw new InvalidOperationException(
                    $"Recursive function calls are not supported by NormStats threading: {function.Name}.");
            }

            foreach (var callee in ExprCollector.Collect(function.Body)
                         .OfType<Call>()
                         .Select(call => call.Target)
                         .OfType<Function>()
                         .Where(callee => moduleFunctions.Contains(callee)))
            {
                Visit(callee);
            }

            active.Remove(function);
            visited.Add(function);
            order.Add(function);
        }

        foreach (var function in functions)
        {
            Visit(function);
        }

        return order.ToArray();
    }

    private static int NormalizeAxis(int axis, BaseExpr input)
    {
        if (input.CheckedShape.IsUnranked)
        {
            return axis;
        }

        return axis < 0 ? axis + input.CheckedShape.Rank : axis;
    }

    private sealed record ThreadingPlan(
        int InputParameterIndex,
        int OutputFieldIndex,
        int OutputCount,
        int Axis,
        bool UseMean,
        IRType StatsType,
        IReadOnlySet<Call> InputStatsCalls)
    {
        public int StatsOutputIndex => OutputCount;
    }

    private sealed class FunctionBodyCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Function, Function> _replacements;
        private readonly IReadOnlyDictionary<Function, ThreadingPlan> _plans;
        private readonly IReadOnlyDictionary<Function, ThreadingPlan> _plansByReplacement;
        private readonly ThreadingPlan? _currentPlan;
        private readonly Var? _statsParameter;

        public FunctionBodyCloner(
            IReadOnlyDictionary<Function, Function> replacements,
            IReadOnlyDictionary<Function, ThreadingPlan> plans,
            IReadOnlyDictionary<Function, ThreadingPlan> plansByReplacement,
            ThreadingPlan? currentPlan,
            Var? statsParameter)
            : base(cloneOtherFunctions: false)
        {
            _replacements = replacements;
            _plans = plans;
            _plansByReplacement = plansByReplacement;
            _currentPlan = currentPlan;
            _statsParameter = statsParameter;
            CloneUnmutated = false;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (_currentPlan is not null
                && _currentPlan.InputStatsCalls.Contains(expr))
            {
                var normStatsCall = (Call)expr;
                var input = Clone(normStatsCall[NormStats.Input], context) as Expr
                    ?? throw new InvalidOperationException("NormStats binding input did not clone to an Expr.");
                var stats = _statsParameter
                    ?? throw new InvalidOperationException("NormStats threading plan has no statistics parameter.");
                var binding = IR.F.NN.BindNormStats(
                    _currentPlan.Axis,
                    input,
                    stats,
                    _currentPlan.UseMean);
                Infer(binding, "materialized NormStats binding");
                return binding;
            }

            if (expr is Function function && _replacements.TryGetValue(function, out var replacement))
            {
                return replacement;
            }

            return base.DispatchVisit(expr, context);
        }

        protected override BaseExpr VisitLeafCall(Call expr, Unit context)
        {
            var target = Clone(expr.Target, context);
            var arguments = CloneArray(expr.Arguments, context);

            if (expr.Target is Function originalTarget
                && _plans.TryGetValue(originalTarget, out var calleePlan))
            {
                var replacement = (Function)target;
                var input = arguments[calleePlan.InputParameterIndex] as Expr
                    ?? throw new InvalidOperationException(
                        $"NormStats-threaded input {calleePlan.InputParameterIndex} of {originalTarget.Name} is not a tensor expression.");
                var stats = TryGetThreadedStats(input, calleePlan.Axis, calleePlan.UseMean, out var threadedStats)
                    ? threadedStats
                    : IR.F.NN.NormStats(calleePlan.Axis, input, calleePlan.UseMean);
                var rawCall = expr.With(
                    target: replacement,
                    arguments: arguments.Append<BaseExpr>(stats).ToArray());
                Infer(rawCall, $"call to NormStats-threaded function {replacement.Name}");
                return rawCall;
            }

            if (expr.Target is NormStats normStats
                && arguments[NormStats.Input.Index] is Expr normInput
                && TryGetThreadedStats(
                    normInput,
                    NormalizeAxis(normStats.Axis, normInput),
                    normStats.UseMean,
                    out var availableStats))
            {
                return availableStats;
            }

            return ReferenceEquals(target, expr.Target)
                   && arguments.Zip(expr.Arguments.ToArray()).All(pair => ReferenceEquals(pair.First, pair.Second))
                ? expr
                : expr.With(target: target, arguments: arguments);
        }

        private static void Infer(BaseExpr expr, string context)
        {
            if (!CompilerServices.InferenceType(expr))
            {
                throw new InvalidOperationException($"Type inference failed for {context}.");
            }

            if (expr.CheckedType is InvalidType invalidType)
            {
                throw new InvalidOperationException($"Type inference failed for {context}: {invalidType}.");
            }
        }

        private bool TryGetThreadedStats(
            Expr value,
            int axis,
            bool useMean,
            out Expr stats)
        {
            stats = null!;
            if (value is not Call { Target: GetItem } getItem
                || getItem[GetItem.Input] is not Call producer
                || producer.Target is not Function producerTarget
                || !_plansByReplacement.TryGetValue(producerTarget, out var producerPlan)
                || getItem[GetItem.Index] is not DimConst index
                || index.Value != producerPlan.OutputFieldIndex
                || axis != producerPlan.Axis
                || useMean != producerPlan.UseMean)
            {
                return false;
            }

            stats = IR.F.Tensors.GetItem(producer, producerPlan.StatsOutputIndex);
            Infer(stats, $"threaded NormStats output of {producerTarget.Name}");
            return true;
        }
    }
}
