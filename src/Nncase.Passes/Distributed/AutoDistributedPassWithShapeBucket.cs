// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

public sealed partial class AutoDistributedWithShapeBucketPass : ModulePass
{
    private const int LargeTensorSizeThreshold = 1000; // Threshold for large tensors in bytes

    private readonly CompileOptions _compileOptions;

    private readonly bool _bidirectional;

    private readonly string _moduleKind;

    public AutoDistributedWithShapeBucketPass(bool bidirectional, string moduleKind, CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
        _bidirectional = bidirectional;
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        if (input.Entry is not Function function || function.Metadata is AutoDistributedMetaData { Skip: true })
        {
            return Task.FromResult(input);
        }

        if (_compileOptions.TargetOptions is INTTTargetOptions targetOptions)
        {
            var newFunction = Distribute(function, targetOptions);
            input.Replace(GetFunctionIndex(input, function), newFunction);
        }

        return Task.FromResult(input);
    }

    private static int GetFunctionIndex(IRModule module, BaseFunction function)
    {
        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (ReferenceEquals(module.Functions[i], function))
            {
                return i;
            }
        }

        throw new InvalidOperationException($"Function {function.Name} is not in the current module.");
    }

    private static void ValidateSegmentFunction(Function inputFunction, Function sourceSegmentFunction, Function segmentFunction, IReadOnlySet<BaseFunction> forbiddenFunctions)
    {
        var directReferences = FunctionReferenceValidator.GetDirectFunctionReferences(segmentFunction);
        foreach (var referencedFunction in directReferences)
        {
            if (ReferenceEquals(referencedFunction, inputFunction)
                || ReferenceEquals(referencedFunction, sourceSegmentFunction)
                || forbiddenFunctions.Contains(referencedFunction))
            {
                throw new InvalidOperationException($"Shape bucket segment {segmentFunction.Name} illegally references source function {referencedFunction.Name}.");
            }
        }

        FunctionReferenceValidator.ValidateAcyclic(segmentFunction);
    }

    private static void ValidateShapeBucketFunctionGraph(BaseFunction function)
    {
        FunctionReferenceValidator.ValidateAcyclic(function);
    }

    private BaseFunction Distribute(Function function, INTTTargetOptions targetOptions)
    {
        var shapeBucketOptions = _compileOptions.ShapeBucketOptions;
        var dimVars = IRHelpers.GetDynamicDimVars();
        var segmentsCount = GetShapeBucketSegmentsCount();
        if (segmentsCount == 0 || dimVars.Count == 0)
        {
            // If no segments or no dynamic dim vars, distribute the entry-reachable
            // function graph as a single segment instead of running per-function
            // independently through the pass manager.
            return DistributeFunctionGraph(function, targetOptions, AutoDistributedPhase.Final);
        }
        else
        {
            var functionForSegments = function;
            if (!AutoDistributedRewriter.SupportsConstShardedView(targetOptions))
            {
                var distributedConsts = SearchDistributedConstants(function, targetOptions);
                functionForSegments = (Function)new DistributeConstCloner(distributedConsts).Clone(function, Unit.Default);

                var dumpper = DumpScope.Current;
                if (dumpper.IsEnabled(DumpFlags.PassIR))
                {
                    dumpper.DumpIR(functionForSegments, "FunctionWithDistributedConsts");
                }
            }

            var segmentStates = new SegmentAutoDistributedState[segmentsCount];
            ValidateManualSegmentRanges(segmentsCount);
            for (int segmentIndex = 0; segmentIndex < segmentsCount; segmentIndex++)
            {
                var newDimVars = (from dimVar in dimVars
                                  let newRange = GetDimSegmentRange(dimVar, segmentIndex, segmentsCount)
                                  select new KeyValuePair<DimVar, DimVar>(
                                      dimVar,
                                      dimVar.With(range: newRange))).ToDictionary(kvp => kvp.Key, kvp => kvp.Value, (IEqualityComparer<DimVar>)ReferenceEqualityComparer.Instance);
                var segmentFunction = ((Function)new SegmentFunctionCloner(newDimVars, $"_segment_{segmentIndex}").Clone(functionForSegments, Unit.Default))
                    .With(role: FunctionRole.Dispatch);
                segmentStates[segmentIndex] = new SegmentAutoDistributedState(
                    segmentFunction,
                    newDimVars);
            }

            var segmentFunctions = new List<(Function SegmentFunction, Dictionary<DimVar, DimVar> DimVars)>(segmentStates.Length);
            var forbiddenFunctions = GetReachableBaseFunctions(functionForSegments)
                .Concat(segmentStates.SelectMany(state => GetReachableBaseFunctions(state.SegmentFunction)))
                .ToHashSet((IEqualityComparer<BaseFunction>)ReferenceEqualityComparer.Instance);
            foreach (var segmentState in segmentStates)
            {
                using var segmentDumpScope = new DumpScope(segmentState.SegmentFunction.Name);
                var rewritten = DistributeFunctionGraph(segmentState.SegmentFunction, targetOptions, AutoDistributedPhase.Final);
                ValidateSegmentFunction(function, segmentState.SegmentFunction, rewritten, forbiddenFunctions);
                segmentFunctions.Add((rewritten, segmentState.DimVars));
            }

            return BuildMainFunction(function, segmentFunctions);
        }

        int GetShapeBucketSegmentsCount()
        {
            if (shapeBucketOptions.SegmentRanges.Count == 0)
            {
                return shapeBucketOptions.SegmentsCount;
            }

            var segmentCounts = shapeBucketOptions.SegmentRanges
                .Select(pair => pair.Value.Length)
                .Distinct()
                .ToArray();
            if (segmentCounts.Length != 1)
            {
                throw new InvalidOperationException($"Shape bucket manual segment ranges must have the same number of bounds, got [{string.Join(", ", segmentCounts)}].");
            }

            return segmentCounts[0];
        }

        void ValidateManualSegmentRanges(int expectedSegmentsCount)
        {
            foreach (var dimVar in dimVars)
            {
                if (!shapeBucketOptions.SegmentRanges.TryGetValue(dimVar.Name, out var upperBounds))
                {
                    continue;
                }

                if (upperBounds.Length != expectedSegmentsCount)
                {
                    throw new InvalidOperationException($"Shape bucket manual segment range for {dimVar.Name} has {upperBounds.Length} bounds, expected {expectedSegmentsCount}.");
                }

                var fullRange = dimVar.Metadata.Range!.Value;
                var min = (int)fullRange.Min;
                var max = (int)fullRange.Max;
                var previous = min - 1;
                foreach (var upperBound in upperBounds)
                {
                    if (upperBound <= previous)
                    {
                        throw new InvalidOperationException($"Shape bucket manual segment range for {dimVar.Name} must be strictly increasing, got [{string.Join(", ", upperBounds)}].");
                    }

                    if (upperBound < min || upperBound > max)
                    {
                        throw new InvalidOperationException($"Shape bucket manual segment range for {dimVar.Name} must stay within [{min}, {max}], got [{string.Join(", ", upperBounds)}].");
                    }

                    previous = upperBound;
                }

                if (upperBounds[^1] != max)
                {
                    throw new InvalidOperationException($"Shape bucket manual segment range for {dimVar.Name} must end at range max {max}, got {upperBounds[^1]}.");
                }
            }
        }

        ValueRange<double> GetDimSegmentRange(DimVar dimVar, int segmentIndex, int expectedSegmentsCount)
        {
            var fullRange = dimVar.Metadata.Range!.Value;
            if (!shapeBucketOptions.SegmentRanges.TryGetValue(dimVar.Name, out var upperBounds))
            {
                return ShapeUtility.GetDimSegmentRange(fullRange, segmentIndex, expectedSegmentsCount);
            }

            var segmentStart = segmentIndex == 0 ? fullRange.Min : upperBounds[segmentIndex - 1] + 1;
            var segmentEnd = upperBounds[segmentIndex];
            return new(segmentStart, segmentEnd);
        }
    }

    private Dictionary<TensorConst, TensorConst> SearchDistributedConstants(Function function, INTTTargetOptions targetOptions)
    {
        var distributedConsts = new Dictionary<TensorConst, TensorConst>(ReferenceEqualityComparer.Instance);
        var reachableFunctions = GetReachableFunctionsInCalleeFirstOrder(function);
        var rewriter = new AutoDistributedRewriter(_compileOptions, targetOptions, AutoDistributedPhase.SearchConstant, _moduleKind, _bidirectional);
        _ = rewriter.RewriteProgram(function, reachableFunctions);
        foreach (var (source, distributed) in rewriter.DistributedConsts)
        {
            distributedConsts[source] = distributed;
        }

        return distributedConsts;
    }

    private Function DistributeFunctionGraph(Function rootFunction, INTTTargetOptions targetOptions, AutoDistributedPhase phase)
    {
        var reachableFunctions = GetReachableFunctionsInCalleeFirstOrder(rootFunction);
        var rewriter = new AutoDistributedRewriter(_compileOptions, targetOptions, phase, _moduleKind, _bidirectional);
        var root = rewriter.RewriteProgram(rootFunction, reachableFunctions);
        foreach (var function in reachableFunctions)
        {
            if (!ReferenceEquals(function, rootFunction))
            {
                continue;
            }

            function.ReplaceAllUsesWith(root);
        }

        ValidateShapeBucketFunctionGraph(root);
        return root;
    }

    private IReadOnlyList<Function> GetReachableFunctionsInCalleeFirstOrder(BaseFunction root)
    {
        var result = new List<Function>();
        var visited = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
        var active = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
        var path = new List<BaseFunction>();

        void Visit(BaseFunction function)
        {
            if (active.Contains(function))
            {
                var cycleStart = path.FindIndex(item => ReferenceEquals(item, function));
                var cycle = path.Skip(cycleStart).Append(function).Select(item => item.Name);
                throw new InvalidOperationException($"Function reference graph contains a cycle: {string.Join(" -> ", cycle)}.");
            }

            if (!visited.Add(function))
            {
                return;
            }

            active.Add(function);
            path.Add(function);
            foreach (var referencedFunction in FunctionReferenceValidator.GetDirectFunctionReferences(function))
            {
                Visit(referencedFunction);
            }

            path.RemoveAt(path.Count - 1);
            active.Remove(function);

            if (function is Function highLevelFunction)
            {
                result.Add(highLevelFunction);
            }
        }

        Visit(root);
        return result;
    }

    private IReadOnlySet<BaseFunction> GetReachableBaseFunctions(BaseFunction root)
    {
        var result = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);

        void Visit(BaseFunction function)
        {
            if (!result.Add(function))
            {
                return;
            }

            foreach (var referencedFunction in FunctionReferenceValidator.GetDirectFunctionReferences(function))
            {
                Visit(referencedFunction);
            }
        }

        Visit(root);
        return result;
    }

    private Function BuildMainFunction(Function inputFunction, List<(Function SegmentFunction, Dictionary<DimVar, DimVar> DimVars)> segmentFunctions)
    {
        if (segmentFunctions.Count == 0)
        {
            throw new InvalidOperationException("Shape bucket dispatcher requires at least one segment function.");
        }

        BaseFunction elseFunction = segmentFunctions[^1].SegmentFunction;
        for (int i = segmentFunctions.Count - 2; i >= 1; i--)
        {
            elseFunction = CreateShapeBucketSelectorFunction(
                $"{inputFunction.Name}_shape_bucket_selector_{i}",
                inputFunction,
                segmentFunctions[i].SegmentFunction,
                elseFunction,
                segmentFunctions[i].DimVars,
                CloneParameters(inputFunction.Parameters.ToArray()));
        }

        var parameters = inputFunction.Parameters.ToArray();
        var mainBody = segmentFunctions.Count == 1
            ? new Call(segmentFunctions[0].SegmentFunction, parameters.AsValueEnumerable().Select(x => (BaseExpr)x).ToArray())
            : CreateShapeBucketIf(segmentFunctions[0].SegmentFunction, elseFunction, segmentFunctions[0].DimVars, parameters);
        var mainFunction = new Function(inputFunction.Name, inputFunction.ModuleKind, mainBody, parameters, inputFunction.VarMap)
        {
            Metadata = inputFunction.Metadata,
            Role = FunctionRole.Dispatch,
        };
        if (!CompilerServices.InferenceType(mainFunction))
        {
            throw new InvalidOperationException($"Type inference failed for shape bucket dispatcher {mainFunction.Name}.");
        }

        ValidateShapeBucketFunctionGraph(mainFunction);
        return mainFunction;
    }

    private Function CreateShapeBucketSelectorFunction(
        string name,
        Function inputFunction,
        BaseFunction thenFunction,
        BaseFunction elseFunction,
        IReadOnlyDictionary<DimVar, DimVar> dimVars,
        IVar[] parameters)
    {
        var body = CreateShapeBucketIf(thenFunction, elseFunction, dimVars, parameters);
        var selector = new Function(
            name,
            inputFunction.ModuleKind,
            body,
            parameters,
            RemapVarMap(inputFunction.VarMap, inputFunction.Parameters.ToArray(), parameters))
        {
            Metadata = inputFunction.Metadata,
            Role = FunctionRole.Dispatch,
        };
        if (!CompilerServices.InferenceType(selector))
        {
            throw new InvalidOperationException($"Type inference failed for shape bucket selector {selector.Name}.");
        }

        return selector;
    }

    private BaseExpr CreateShapeBucketIf(
        BaseFunction thenFunction,
        BaseFunction elseFunction,
        IReadOnlyDictionary<DimVar, DimVar> dimVars,
        IReadOnlyList<IVar> parameters)
    {
        var condition = dimVars
            .Select(dimVarPair => dimVarPair.Key <= (long)dimVarPair.Value.Metadata.Range!.Value.Max)
            .Aggregate(IR.F.Math.LogicalAnd);
        var arguments = parameters.Select(x => (BaseExpr)x).ToArray();
        return new If(condition, thenFunction, elseFunction, arguments);
    }

    private IVar[] CloneParameters(IReadOnlyList<IVar> parameters)
        => parameters.Select(parameter => parameter.With(parameter.Name)).ToArray();

    private Dictionary<IVar, Dimension[]>? RemapVarMap(
        Dictionary<IVar, Dimension[]>? source,
        IReadOnlyList<IVar> sourceParameters,
        IReadOnlyList<IVar> targetParameters)
    {
        if (source is null)
        {
            return null;
        }

        var parameterMap = sourceParameters
            .Zip(targetParameters)
            .ToDictionary(pair => pair.First, pair => pair.Second, (IEqualityComparer<IVar>)ReferenceEqualityComparer.Instance);
        return source.ToDictionary(
            kvp => parameterMap.TryGetValue(kvp.Key, out var mapped) ? mapped : kvp.Key,
            kvp => kvp.Value,
            (IEqualityComparer<IVar>)ReferenceEqualityComparer.Instance);
    }

    private sealed class SegmentAutoDistributedState
    {
        public SegmentAutoDistributedState(
            Function segmentFunction,
            Dictionary<DimVar, DimVar> dimVars)
        {
            SegmentFunction = segmentFunction;
            DimVars = dimVars;
        }

        public Function SegmentFunction { get; }

        public Dictionary<DimVar, DimVar> DimVars { get; }
    }

    private sealed class DistributeConstCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<TensorConst, TensorConst> _distributedConsts;

        public DistributeConstCloner(IReadOnlyDictionary<TensorConst, TensorConst> distributedConsts)
            : base(cloneOtherFunctions: true)
        {
            _distributedConsts = distributedConsts;
        }

        protected override BaseExpr VisitLeafTensorConst(TensorConst expr, Unit context)
        {
            if (_distributedConsts.TryGetValue(expr, out var distConst) &&
                expr.Value.Length * expr.Value.ElementType.SizeInBytes > LargeTensorSizeThreshold)
            {
                // If the tensor is large, we do not remove boxing
                return IR.F.Distributed.Boxing(distConst, expr.CheckedTensorType);
            }

            return expr;
        }
    }

    private sealed class SegmentFunctionCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<DimVar, DimVar> _newDimVars;
        private readonly string _nameSuffix;

        public SegmentFunctionCloner(IReadOnlyDictionary<DimVar, DimVar> newDimVars, string nameSuffix)
            : base(cloneOtherFunctions: true)
        {
            _newDimVars = newDimVars;
            _nameSuffix = nameSuffix;
        }

        protected override BaseExpr VisitLeafFunction(Function expr, Unit context)
        {
            var cloned = (Function)base.VisitLeafFunction(expr, context);
            return cloned.With(name: $"{expr.Name}{_nameSuffix}");
        }

        protected override BaseExpr VisitLeafVar(Var expr, Unit context)
        {
            bool IsOperandsMutated()
            {
                return IsMutatedType(expr.TypeAnnotation, context);
            }

            if (CloneUnmutated || IsOperandsMutated())
            {
                return expr.With(
                    typeAnnotation: CloneType(expr.TypeAnnotation, context));
            }

            return expr;
        }

        protected override BaseExpr VisitLeafDimVar(DimVar expr, Unit context)
        {
            return _newDimVars.GetValueOrDefault(expr, expr);
        }
    }

    private sealed class FunctionReferenceValidator
    {
        public static IReadOnlyList<BaseFunction> GetDirectFunctionReferences(BaseFunction function)
        {
            return function switch
            {
                Function f => GetDirectFunctionReferences(f.Body),
                Fusion f => GetDirectFunctionReferences(f.Body),
                FunctionWrapper wrapper => [wrapper.Target],
                PrimFunctionWrapper wrapper => [wrapper.Target],
                PrimFunction f => GetDirectFunctionReferences(f.Body),
                _ => throw new NotSupportedException($"Unsupported function type {function.GetType().Name}."),
            };
        }

        public static void ValidateAcyclic(BaseFunction root)
        {
            var visited = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
            var active = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
            var path = new List<BaseFunction>();
            Visit(root, visited, active, path);
        }

        private static IReadOnlyList<BaseFunction> GetDirectFunctionReferences(BaseExpr root)
        {
            var refs = new List<BaseFunction>();
            var seenRefs = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
            var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            var stack = new Stack<BaseExpr>();
            stack.Push(root);

            while (stack.Count != 0)
            {
                var expr = stack.Pop();
                if (!visited.Add(expr))
                {
                    continue;
                }

                if (expr is BaseFunction function)
                {
                    if (seenRefs.Add(function))
                    {
                        refs.Add(function);
                    }

                    continue;
                }

                var operands = expr.Operands;
                for (int i = operands.Length - 1; i >= 0; i--)
                {
                    stack.Push(operands[i]);
                }
            }

            return refs;
        }

        private static void Visit(BaseFunction function, HashSet<BaseFunction> visited, HashSet<BaseFunction> active, List<BaseFunction> path)
        {
            if (active.Contains(function))
            {
                var cycleStart = path.FindIndex(f => ReferenceEquals(f, function));
                var cycle = path.Skip(cycleStart).Append(function).Select(f => f.Name);
                throw new InvalidOperationException($"Function reference graph contains a cycle: {string.Join(" -> ", cycle)}.");
            }

            if (!visited.Add(function))
            {
                return;
            }

            active.Add(function);
            path.Add(function);
            foreach (var referencedFunction in GetDirectFunctionReferences(function))
            {
                Visit(referencedFunction, visited, active, path);
            }

            path.RemoveAt(path.Count - 1);
            active.Remove(function);
        }
    }
}
