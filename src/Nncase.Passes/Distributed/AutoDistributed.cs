// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reactive;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Google.OrTools.Sat;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Targets;
using Nncase.Utilities;
using QuikGraph;
using QuikGraph.Graphviz;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes.Distributed;

public enum AutoDistributedPhase
{
    SearchConstant,
    Final,
}

internal enum SearchGraphKind : int
{
    Root,
    DistributedCluster,
    StandaloneCluster,
    Bucket,
}

internal enum SearchableNodeKind : int
{
    Normal,
    FunctionParameter,
    FunctionResult,
    FunctionCall,
    FunctionBoundaryAdapter,
    TypeAdapter,
}

public sealed class AutoDistributedMetaData : IRMetadata
{
    public bool Skip { get; set; }
}

internal static class DistributedFunctionGraphUtility
{
    public static IReadOnlyList<Function> GetReachableFunctionsInCalleeFirstOrder(BaseFunction root)
    {
        var result = new List<Function>();
        var visited = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
        var active = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
        var path = new List<BaseFunction>();

        void Visit(BaseFunction function)
        {
            if (active.Contains(function))
            {
                var cycleStart = path.FindIndex(x => ReferenceEquals(x, function));
                var cycle = path.Skip(Math.Max(cycleStart, 0)).Append(function).Select(x => x.Name);
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

    public static IReadOnlyList<BaseFunction> GetDirectFunctionReferences(BaseExpr root)
    {
        var refs = new List<BaseFunction>();
        var seenRefs = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
        var seenExprs = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<BaseExpr>();
        stack.Push(root);
        while (stack.Count != 0)
        {
            var expr = stack.Pop();
            if (!seenExprs.Add(expr))
            {
                continue;
            }

            if (expr is BaseFunction function && !ReferenceEquals(function, root) && seenRefs.Add(function))
            {
                refs.Add(function);
                continue;
            }

            foreach (var operand in expr.Operands)
            {
                stack.Push(operand);
            }
        }

        return refs;
    }

    public static IReadOnlyDictionary<Function, long> GetStaticInvocationCounts(
        Function root,
        IReadOnlyList<Function> reachableFunctions)
    {
        var functionOrder = new Dictionary<Function, int>(ReferenceEqualityComparer.Instance);
        for (var index = 0; index < reachableFunctions.Count; index++)
        {
            var function = reachableFunctions[index];
            if (!functionOrder.TryAdd(function, index))
            {
                throw new InvalidOperationException(
                    $"AutoDistributed reachable function list contains duplicate function {function.Name}.");
            }
        }

        if (!functionOrder.ContainsKey(root))
        {
            throw new InvalidOperationException(
                $"AutoDistributed reachable function list does not contain root function {root.Name}.");
        }

        var invocationCounts = new Dictionary<Function, long>(ReferenceEqualityComparer.Instance)
        {
            [root] = 1,
        };

        for (var callerIndex = reachableFunctions.Count - 1; callerIndex >= 0; callerIndex--)
        {
            var caller = reachableFunctions[callerIndex];
            if (!invocationCounts.TryGetValue(caller, out var callerCount))
            {
                continue;
            }

            foreach (var (callee, callsPerInvocation) in GetDirectFunctionCallCounts(caller.Body))
            {
                if (!functionOrder.TryGetValue(callee, out var calleeIndex))
                {
                    throw new InvalidOperationException(
                        $"AutoDistributed reachable function list does not contain callee {callee.Name} referenced by {caller.Name}.");
                }

                if (calleeIndex >= callerIndex)
                {
                    throw new InvalidOperationException(
                        $"AutoDistributed requires callee-first function order, but {callee.Name} is not before caller {caller.Name}.");
                }

                var addedCount = checked(callerCount * callsPerInvocation);
                invocationCounts[callee] = checked(
                    invocationCounts.GetValueOrDefault(callee) + addedCount);
            }
        }

        foreach (var function in reachableFunctions)
        {
            if (!invocationCounts.ContainsKey(function))
            {
                throw new InvalidOperationException(
                    $"Reachable function {function.Name} has no static call path from entry function {root.Name}.");
            }
        }

        return invocationCounts;
    }

    private static IReadOnlyDictionary<Function, long> GetDirectFunctionCallCounts(BaseExpr root)
    {
        var counts = new Dictionary<Function, long>(ReferenceEqualityComparer.Instance);
        var seenExprs = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<BaseExpr>();
        stack.Push(root);
        while (stack.Count != 0)
        {
            var expr = stack.Pop();
            if (!seenExprs.Add(expr))
            {
                continue;
            }

            if (expr is Call { Target: Function callee } call)
            {
                counts[callee] = checked(counts.GetValueOrDefault(callee) + 1);
                foreach (var argument in call.Arguments)
                {
                    stack.Push(argument);
                }

                continue;
            }

            if (expr is BaseFunction && !ReferenceEquals(expr, root))
            {
                continue;
            }

            foreach (var operand in expr.Operands)
            {
                stack.Push(operand);
            }
        }

        return counts;
    }
}

/// <summary>
/// auto distributed the function.
/// </summary>
public sealed partial class AutoDistributedPass : FunctionPass
{
    private readonly CompileOptions _compileOptions;

    private readonly bool _bidirectional;

    private readonly string _moduleKind;

    public AutoDistributedPass(bool bidirectional, string moduleKind, CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
        _bidirectional = bidirectional;
        _moduleKind = moduleKind;
    }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function || input.Metadata is AutoDistributedMetaData { Skip: true })
        {
            return Task.FromResult(input);
        }

        if (_compileOptions.TargetOptions is INTTTargetOptions targetOptions)
        {
            var rewriter = new AutoDistributedRewriter(_compileOptions, targetOptions, AutoDistributedPhase.Final, _moduleKind, _bidirectional);
            return Task.FromResult((BaseFunction)rewriter.Rewrite(function));
        }

        return Task.FromResult(input);
    }
}

internal static class UserRebuilder
{
    public static void Rebuild(BaseExpr root)
    {
        var order = new List<BaseExpr>(256);
        var seen = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
        DfsIter(root, order, seen);

        foreach (var n in order)
        {
            var users = n.Users.ToArray();
            for (int i = 0; i < users.Length; i++)
            {
                n.RemoveUser(users[i]);
            }
        }

        foreach (var n in order)
        {
            var ops = n.Operands;
            for (int i = 0; i < ops.Length; i++)
            {
                ops[i].AddUser(n);
            }
        }
    }

    private static void DfsIter(BaseExpr root, List<BaseExpr> order, HashSet<BaseExpr> seen)
    {
        var stack = new Stack<BaseExpr>();
        stack.Push(root);

        while (stack.Count > 0)
        {
            var n = stack.Pop();
            if (!seen.Add(n))
            {
                continue;
            }

            order.Add(n);

            var ops = n.Operands;
            for (int i = ops.Length - 1; i >= 0; i--)
            {
                stack.Push(ops[i]);
            }
        }
    }
}

internal sealed class SearchableNode
{
    public SearchableNode(
        BaseExpr expr,
        IRType type,
        bool isBidirect = false,
        SearchableNodeKind kind = SearchableNodeKind.Normal,
        DistributedReshardSourceKind? sourceKind = null,
        DistributedReshardUsageKind? reshardUsageKind = null,
        IVar? originParameter = null,
        BaseExpr? metadataSource = null)
    {
        Expr = expr;
        IRType = type;
        IsBidirect = isBidirect;
        Kind = kind;
        ReshardUsageKind = reshardUsageKind;
        SourceKind = sourceKind ??
            (kind == SearchableNodeKind.FunctionParameter
                ? DistributedReshardSourceKind.FunctionParameter
                : expr is TensorConst
                    ? DistributedReshardSourceKind.Constant
                    : DistributedReshardSourceKind.Internal);
        OriginParameter = originParameter ??
            (SourceKind == DistributedReshardSourceKind.FunctionParameter && expr is IVar parameter
                ? parameter
                : null);
        MetadataSource = metadataSource;
    }

    public BaseExpr Expr { get; }

    public IRType IRType { get; }

    public bool IsBidirect { get; }

    public SearchableNodeKind Kind { get; }

    public DistributedReshardSourceKind SourceKind { get; }

    public DistributedReshardUsageKind? ReshardUsageKind { get; }

    public IVar? OriginParameter { get; }

    /// <summary>
    /// Gets the source expression whose stable semantic metadata belongs to an
    /// executable candidate. Search-only operations such as type adapters and
    /// reshard realizations intentionally leave this null.
    /// </summary>
    public BaseExpr? MetadataSource { get; }
}

internal sealed record CrossEdge : IEdge<SearchableNode>
{
    public CrossEdge(SearchableNode root, SearchableNode input, int inputIndex, DistributedSearchGraph inputGraph)
    {
        Root = root;
        Input = input;
        InputIndex = inputIndex;
        InputGraph = inputGraph;
    }

    public SearchableNode Root { get; }

    public SearchableNode Input { get; }

    public int InputIndex { get; }

    public DistributedSearchGraph InputGraph { get; }

    public SearchableNode Source => Root;

    public SearchableNode Target => Input;
}

internal sealed class DistributedSearchGraph : TieredAdjacencyGraph<SearchableNode, CrossEdge>
{
    public DistributedSearchGraph([NotNull] AdjacencyGraph<SearchableNode, CrossEdge> wrappedGraph, SearchGraphKind kind)
    : base(wrappedGraph)
    {
        Kind = kind;
    }

    public DistributedSearchGraph([NotNull] TieredAdjacencyGraph<SearchableNode, CrossEdge> parentGraph, SearchGraphKind kind)
        : base(parentGraph)
    {
        Kind = kind;
    }

    public SearchGraphKind Kind { get; }
}

internal sealed record CandidateDiagnosticKey(
    string Target,
    string Stage,
    string Status,
    string ResultType,
    string Reason,
    string Arguments);

internal sealed record BoxingTypeKey(IRType InputType, IRType OutputType, bool IsReshape);

internal sealed record LeafCandidateKey(TensorType TensorType);

internal sealed record ReshardPlanKey(
    IRType SourceType,
    IRType TargetType,
    DistributedReshardSourceKind SourceKind,
    DistributedReshardUsageKind UsageKind,
    int MaxHops);

internal sealed record ReshardCandidateKey(
    DistributedSearchGraph OwnerCluster,
    DistributedSearchGraph? OutputBucket,
    DistributedSearchGraph InputBucket,
    SearchableNode InputNode,
    IRType TargetType,
    DistributedReshardUsageKind UsageKind,
    SearchableNodeKind Kind,
    bool IsBidirect,
    DistributedSearchGraph? DependencyBucket,
    SearchableNode? DependencyNode);

internal sealed record ProviderInputTypeKey(
    DistributedSearchGraph InputCluster,
    IRType TargetType);

internal sealed record ProviderInputChoiceKey(
    DistributedSearchGraph InputCluster,
    IRType TargetType,
    Function OwnerFunction);

internal sealed class AutoDistributedProfiler
{
    private static readonly bool IsEnabled = string.Equals(
        Environment.GetEnvironmentVariable("NNCASE_PROFILE_AUTO_DIST"),
        "1",
        StringComparison.OrdinalIgnoreCase);

    private readonly string _moduleKind;
    private readonly AutoDistributedPhase _phase;
    private readonly Stopwatch _activeTotal = new();
    private readonly Dictionary<string, double> _timingsMs = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _counts = new(StringComparer.Ordinal);
    private string _functionName = string.Empty;

    public AutoDistributedProfiler(string moduleKind, AutoDistributedPhase phase)
    {
        _moduleKind = moduleKind;
        _phase = phase;
    }

    public void SetFunction(string functionName)
    {
        _functionName = functionName;
    }

    public void Count(string name, long value = 1)
    {
        if (!IsEnabled)
        {
            return;
        }

        _counts[name] = _counts.TryGetValue(name, out var current) ? current + value : value;
    }

    public T TimeActive<T>(Func<T> action)
    {
        if (!IsEnabled)
        {
            return action();
        }

        _activeTotal.Start();
        try
        {
            return action();
        }
        finally
        {
            _activeTotal.Stop();
        }
    }

    public void TimeActive(Action action)
    {
        if (!IsEnabled)
        {
            action();
            return;
        }

        _activeTotal.Start();
        try
        {
            action();
        }
        finally
        {
            _activeTotal.Stop();
        }
    }

    public T Time<T>(string name, Func<T> action)
    {
        if (!IsEnabled)
        {
            return action();
        }

        var sw = Stopwatch.StartNew();
        try
        {
            return action();
        }
        finally
        {
            AddTiming(name, sw.Elapsed.TotalMilliseconds);
        }
    }

    public void Time(string name, Action action)
    {
        if (!IsEnabled)
        {
            action();
            return;
        }

        var sw = Stopwatch.StartNew();
        try
        {
            action();
        }
        finally
        {
            AddTiming(name, sw.Elapsed.TotalMilliseconds);
        }
    }

    public void Write(DistributedSearchGraph rootSearchGraph, int candidateDiagnosticTotal)
    {
        if (!IsEnabled)
        {
            return;
        }

        _counts["candidate_diagnostics_total"] = candidateDiagnosticTotal;
        _counts["graph_vertices"] = rootSearchGraph.VertexCount;
        _counts["graph_edges"] = rootSearchGraph.EdgeCount;
        _counts["graph_clusters"] = rootSearchGraph.Clusters.Cast<object>().Count();
        _counts["graph_buckets"] = rootSearchGraph.Clusters.OfType<DistributedSearchGraph>().SelectMany(g => g.Clusters.OfType<DistributedSearchGraph>()).Count();
        foreach (var group in rootSearchGraph.Clusters.OfType<DistributedSearchGraph>().GroupBy(g => g.Kind))
        {
            _counts[$"graph_clusters_{group.Key}"] = group.Count();
        }

        using var stream = Diagnostics.DumpScope.Current.OpenFile("AutoDistributedProfile.json");
        JsonSerializer.Serialize(
            stream,
            new
            {
                function = _functionName,
                module_kind = _moduleKind,
                phase = _phase.ToString(),
                total_ms = _activeTotal.Elapsed.TotalMilliseconds,
                timings_ms = _timingsMs.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value),
                counts = _counts.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key, kv => kv.Value),
            },
            new JsonSerializerOptions { WriteIndented = true });
    }

    private void AddTiming(string name, double elapsedMs)
    {
        _timingsMs[name] = _timingsMs.TryGetValue(name, out var current) ? current + elapsedMs : elapsedMs;
    }
}

internal sealed class TypeInferenceCacheKey : IEquatable<TypeInferenceCacheKey>
{
    private readonly BaseExpr _target;
    private readonly IRType[] _argumentTypes;
    private readonly BaseExpr?[] _attributeArguments;
    private readonly int _hashCode;

    public TypeInferenceCacheKey(Call call)
    {
        _target = call.Target;
        _argumentTypes = call.Arguments.AsValueEnumerable().Select(arg => arg.CheckedType).ToArray();
        _attributeArguments = call.Arguments.AsValueEnumerable()
            .Select((arg, index) => IsAttributeArgument(call, index) ? arg : null)
            .ToArray();

        HashCode hash = default;
        hash.Add(RuntimeHelpers.GetHashCode(_target));
        for (int i = 0; i < _argumentTypes.Length; i++)
        {
            hash.Add(_argumentTypes[i]);
            hash.Add(_attributeArguments[i] is { } attr ? RuntimeHelpers.GetHashCode(attr) : 0);
        }

        _hashCode = hash.ToHashCode();
    }

    public bool Equals(TypeInferenceCacheKey? other)
    {
        if (other is null || !ReferenceEquals(_target, other._target) || _argumentTypes.Length != other._argumentTypes.Length)
        {
            return false;
        }

        for (int i = 0; i < _argumentTypes.Length; i++)
        {
            if (!EqualityComparer<IRType>.Default.Equals(_argumentTypes[i], other._argumentTypes[i])
                || !ReferenceEquals(_attributeArguments[i], other._attributeArguments[i]))
            {
                return false;
            }
        }

        return true;
    }

    public override bool Equals(object? obj) => Equals(obj as TypeInferenceCacheKey);

    public override int GetHashCode() => _hashCode;

    private static bool IsAttributeArgument(Call call, int index)
        => call.Target is Op op
            && op.Parameters.AsValueEnumerable().Any(parameter => parameter.Index == index && parameter.ParameterKind == ParameterKind.Attribute);
}

internal sealed class CandidateDominanceKey : IEquatable<CandidateDominanceKey>
{
    private readonly IRType _resultType;
    private readonly (int InputIndex, DistributedSearchGraph InputGraph)[] _inputs;
    private readonly int _hashCode;

    public CandidateDominanceKey(SearchableNode node, IReadOnlyList<CrossEdge> inputs)
    {
        _resultType = node.IRType;
        _inputs = inputs.Select(edge => (edge.InputIndex, edge.InputGraph)).ToArray();

        HashCode hash = default;
        hash.Add(_resultType);
        foreach (var (inputIndex, inputGraph) in _inputs)
        {
            hash.Add(inputIndex);
            hash.Add(RuntimeHelpers.GetHashCode(inputGraph));
        }

        _hashCode = hash.ToHashCode();
    }

    public bool Equals(CandidateDominanceKey? other)
    {
        if (other is null
            || !EqualityComparer<IRType>.Default.Equals(_resultType, other._resultType)
            || _inputs.Length != other._inputs.Length)
        {
            return false;
        }

        for (int i = 0; i < _inputs.Length; i++)
        {
            if (_inputs[i].InputIndex != other._inputs[i].InputIndex || !ReferenceEquals(_inputs[i].InputGraph, other._inputs[i].InputGraph))
            {
                return false;
            }
        }

        return true;
    }

    public override bool Equals(object? obj) => Equals(obj as CandidateDominanceKey);

    public override int GetHashCode() => _hashCode;
}

internal sealed class AutoDistributedRewriter : ExprVisitor<Unit, Unit>
{
    private const int MaxProviderReturnCandidateTypes = 4096;
    private const int HiddenFunctionDependencyIndex = -1;

    private readonly record struct CandidateInvocation(
        DistributedSearchGraph[] Buckets,
        Expr Target,
        IRType? ExpectedReturnType,
        bool AllowsPartialInputs);

    private readonly record struct ProviderCandidateResult(
        IReadOnlyList<CandidateInvocation> Invocations,
        bool IsExhaustive);

    private sealed class CandidateInvocationIdentity : IEquatable<CandidateInvocationIdentity>
    {
        private readonly Expr _target;
        private readonly IRType? _expectedReturnType;
        private readonly DistributedSearchGraph[] _buckets;
        private readonly int _hashCode;

        public CandidateInvocationIdentity(CandidateInvocation invocation)
        {
            _target = invocation.Target;
            _expectedReturnType = invocation.ExpectedReturnType;
            _buckets = invocation.Buckets;

            HashCode hash = default;
            hash.Add(RuntimeHelpers.GetHashCode(_target));
            hash.Add(_expectedReturnType);
            foreach (var bucket in _buckets)
            {
                hash.Add(RuntimeHelpers.GetHashCode(bucket));
            }

            _hashCode = hash.ToHashCode();
        }

        public bool Equals(CandidateInvocationIdentity? other)
        {
            if (other is null
                || !ReferenceEquals(_target, other._target)
                || !EqualityComparer<IRType?>.Default.Equals(_expectedReturnType, other._expectedReturnType)
                || _buckets.Length != other._buckets.Length)
            {
                return false;
            }

            for (var index = 0; index < _buckets.Length; index++)
            {
                if (!ReferenceEquals(_buckets[index], other._buckets[index]))
                {
                    return false;
                }
            }

            return true;
        }

        public override bool Equals(object? obj) => Equals(obj as CandidateInvocationIdentity);

        public override int GetHashCode() => _hashCode;
    }

    private sealed class BucketCombinationIdentity : IEquatable<BucketCombinationIdentity>
    {
        private readonly DistributedSearchGraph[] _buckets;
        private readonly int _hashCode;

        public BucketCombinationIdentity(IReadOnlyList<DistributedSearchGraph> buckets)
        {
            _buckets = buckets.ToArray();
            HashCode hash = default;
            foreach (var bucket in _buckets)
            {
                hash.Add(RuntimeHelpers.GetHashCode(bucket));
            }

            _hashCode = hash.ToHashCode();
        }

        public bool Equals(BucketCombinationIdentity? other)
        {
            if (other is null || _buckets.Length != other._buckets.Length)
            {
                return false;
            }

            for (var index = 0; index < _buckets.Length; index++)
            {
                if (!ReferenceEquals(_buckets[index], other._buckets[index]))
                {
                    return false;
                }
            }

            return true;
        }

        public override bool Equals(object? obj) => Equals(obj as BucketCombinationIdentity);

        public override int GetHashCode() => _hashCode;
    }

    private sealed class OperationCandidateSite
    {
        public OperationCandidateSite(
            Call sourceCall,
            bool isSupported,
            IReadOnlyList<DistributedSearchGraph> argumentClusters,
            DistributedSearchGraph callCluster,
            Function function)
        {
            SourceCall = sourceCall;
            IsSupported = isSupported;
            ArgumentClusters = argumentClusters;
            CallCluster = callCluster;
            Function = function;
        }

        public Call SourceCall { get; }

        public bool IsSupported { get; }

        public IReadOnlyList<DistributedSearchGraph> ArgumentClusters { get; }

        public DistributedSearchGraph CallCluster { get; }

        public Function Function { get; }

        public Dictionary<IRType, DistributedSearchGraph> OutputBuckets { get; } = new();

        public List<DistributedSearchGraph> DirectOutputBuckets { get; } = new();

        public HashSet<CandidateInvocationIdentity> Invocations { get; } = new();

        public HashSet<IRType> DemandedReturnTypes { get; } = new();
    }

    private sealed class FunctionResultSite
    {
        public FunctionResultSite(
            Function function,
            BaseExpr result,
            DistributedSearchGraph resultCluster,
            IReadOnlyList<DistributedSearchGraph>? fieldClusters = null)
        {
            Function = function;
            Result = result;
            ResultCluster = resultCluster;
            FieldClusters = fieldClusters;
        }

        public Function Function { get; }

        public BaseExpr Result { get; }

        public DistributedSearchGraph ResultCluster { get; }

        public IReadOnlyList<DistributedSearchGraph>? FieldClusters { get; }

        public HashSet<DistributedSearchGraph> SourceBuckets { get; } = new(ReferenceEqualityComparer.Instance);

        public HashSet<BucketCombinationIdentity> FieldCombinations { get; } = new();
    }

    private sealed class FunctionCallCandidateSite
    {
        public FunctionCallCandidateSite(
            Call sourceCall,
            Function callee,
            DistributedSearchGraph calleeReturnCluster,
            IReadOnlyList<DistributedSearchGraph> boundaryClusters,
            IReadOnlyList<FunctionBoundarySite?> boundarySites,
            DistributedSearchGraph callCluster,
            Function function)
        {
            SourceCall = sourceCall;
            Callee = callee;
            CalleeReturnCluster = calleeReturnCluster;
            BoundaryClusters = boundaryClusters;
            BoundarySites = boundarySites;
            CallCluster = callCluster;
            Function = function;
        }

        public Call SourceCall { get; }

        public Function Callee { get; }

        public DistributedSearchGraph CalleeReturnCluster { get; }

        public IReadOnlyList<DistributedSearchGraph> BoundaryClusters { get; }

        public IReadOnlyList<FunctionBoundarySite?> BoundarySites { get; }

        public DistributedSearchGraph CallCluster { get; }

        public Function Function { get; }

        public List<DistributedSearchGraph> DirectOutputBuckets { get; } = new();

        public HashSet<DistributedSearchGraph> ReturnBuckets { get; } = new(ReferenceEqualityComparer.Instance);
    }

    private sealed class OutputReshardClosureState
    {
        public OutputReshardClosureState(DistributedReshardUsageKind usageKind)
        {
            UsageKind = usageKind;
        }

        public DistributedReshardUsageKind UsageKind { get; }

        public HashSet<DistributedSearchGraph> Sources { get; } = new(ReferenceEqualityComparer.Instance);

        public Dictionary<IRType, DistributedSearchGraph> TargetBuckets { get; } = new();

        public DistributedSearchGraph? PathCluster { get; set; }
    }

    private sealed class ProviderInputChoiceState
    {
        public ProviderInputChoiceState(DistributedSearchGraph choiceBucket)
        {
            ChoiceBucket = choiceBucket;
        }

        public DistributedSearchGraph ChoiceBucket { get; }

        public HashSet<DistributedSearchGraph> SourceBuckets { get; } = new(ReferenceEqualityComparer.Instance);
    }

    private sealed class FunctionBoundarySite
    {
        public FunctionBoundarySite(
            Call sourceCall,
            Function callee,
            IVar parameter,
            int argumentIndex,
            BaseExpr actual,
            DistributedSearchGraph actualCluster,
            DistributedSearchGraph boundaryCluster)
        {
            SourceCall = sourceCall;
            Callee = callee;
            Parameter = parameter;
            ArgumentIndex = argumentIndex;
            Actual = actual;
            ActualCluster = actualCluster;
            BoundaryCluster = boundaryCluster;
        }

        public Call SourceCall { get; }

        public Function Callee { get; }

        public IVar Parameter { get; }

        public int ArgumentIndex { get; }

        public BaseExpr Actual { get; }

        public DistributedSearchGraph ActualCluster { get; }

        public DistributedSearchGraph BoundaryCluster { get; }

        public HashSet<DistributedSearchGraph> DirectActualBuckets { get; } = new(ReferenceEqualityComparer.Instance);

        public List<(SearchableNode CallNode, int InputIndex)> CallConsumers { get; } = new();
    }

    private readonly Dictionary<BaseExpr, DistributedSearchGraph> _reshardMemo;

    private readonly Dictionary<BaseExpr, DistributedSearchGraph> _inferedMemo;

    private readonly AdjacencyGraph<SearchableNode, CrossEdge> _rootGraph;

    private readonly DistributedSearchGraph _rootSearchGraph;

    private readonly string _moduleKind;

    private readonly bool _bidirectional;

    private readonly AutoDistributedPhase _phase;

    private readonly IDistributedCandidateProviderResolver? _candidateProviderResolver;

    private readonly IDistributedReshardRealizationPolicy _reshardRealizationPolicy;

    private readonly Dictionary<Type, ITypeInferencer> _inferencer_cache = new Dictionary<Type, ITypeInferencer>();

    private readonly Dictionary<CandidateDiagnosticKey, int> _candidateDiagnostics = new();

    private readonly AutoDistributedProfiler _profiler;

    private readonly bool _recordCandidateDiagnostics;

    private readonly Dictionary<LeafCandidateKey, IReadOnlyList<DistributedType>> _leafCandidateMemo = new();

    private readonly Dictionary<ReshardPlanKey, IReadOnlyList<DistributedReshardPlan>> _reshardPlanMemo = new();

    private readonly Dictionary<BoxingTypeKey, IRType> _boxingTypeMemo = new();

    private readonly Dictionary<ReshardCandidateKey, (DistributedSearchGraph Bucket, SearchableNode Node)> _reshardCandidateMemo = new();

    private readonly Dictionary<ProviderInputTypeKey, IReadOnlyList<DistributedSearchGraph>> _providerInputTypeMemo = new();

    private readonly Dictionary<ProviderInputChoiceKey, ProviderInputChoiceState> _providerInputChoiceMemo = new();

    private readonly Dictionary<TensorConst, DistributedSearchGraph> _constantStorageBuckets = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<DistributedSearchGraph, OutputReshardClosureState> _outputReshardClosureStates = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<TypeInferenceCacheKey, (bool Success, IRType CheckedType)> _typeInferenceMemo = new();

    private readonly Dictionary<Function, DistributedSearchGraph> _functionReturnClusters = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Function, DistributedSearchGraph> _functionRootClusters = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Function, Dictionary<IVar, DistributedSearchGraph>> _functionParameterClusters = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Function, Dictionary<IVar, DistributedSearchGraph>> _functionParameterUseClusters = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<IVar, List<FunctionBoundarySite>> _functionBoundarySites = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<IVar, Function> _functionParameterOwners = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<IVar, Dictionary<IRType, (DistributedSearchGraph Bucket, SearchableNode Node)>> _functionParameterSignatureCandidates = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<IVar, Dictionary<IRType, HashSet<DistributedSearchGraph>>> _functionParameterDerivedUseBuckets = new(ReferenceEqualityComparer.Instance);

    private readonly List<OperationCandidateSite> _operationCandidateSites = new();

    private readonly Dictionary<Call, OperationCandidateSite> _operationCandidateSitesByCall = new(ReferenceEqualityComparer.Instance);

    private readonly List<FunctionResultSite> _functionResultSites = new();

    private readonly List<FunctionCallCandidateSite> _functionCallCandidateSites = new();

    private readonly List<FunctionBoundarySite> _allFunctionBoundarySites = new();

    private readonly Dictionary<BaseExpr, IReadOnlyList<DistributedSearchGraph>> _directValueBuckets = new(ReferenceEqualityComparer.Instance);

    private readonly HashSet<DistributedSearchGraph> _singleChoiceClusters = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Function, long> _functionInvocationCounts = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<SearchableNode, Function> _nodeOwnerFunctions = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// The original tensor consts that are distributed.
    /// </summary>
    private readonly Dictionary<TensorConst, TensorConst> _distributedConstSources = new(ReferenceEqualityComparer.Instance);

    private int _candidateDemandVersion;

    private Function? _currentFunction;

    private bool _currentFunctionIsEntry;

    private Dictionary<SearchableNode, bool>? _lastPicks;

    private int _candidateDiagnosticTotal;

    public AutoDistributedRewriter(
        CompileOptions compileOptions,
        INTTTargetOptions targetOptions,
        AutoDistributedPhase phase,
        string moduleKind = "cpu",
        bool bidirectional = false)
    {
        Placements = targetOptions.Hierarchies.Select(h => new Placement(h, targetOptions.HierarchyNames, targetOptions.HierarchyLevels)).ToArray();
        Bidirectional = bidirectional;
        CompileOptions = compileOptions;
        TargetOptions = targetOptions;
        _candidateProviderResolver = CompilerServices.GetService<IDistributedCandidateProviderResolver>();
        _reshardRealizationPolicy = DistributedReshardRealizationPolicy.Get(targetOptions);
        _moduleKind = moduleKind;
        _phase = phase;
        if (Path.Exists(TargetOptions.DistributedScheme) && System.Text.Json.JsonSerializer.Deserialize<DistributedSchema>(File.ReadAllText(TargetOptions.DistributedScheme)) is DistributedSchema scheme)
        {
            Scheme = scheme.Outputs.ToDictionary(n => n.Name, n => (new IRArray<SBP>(n.NdSBP), new Placement(n.Hierarchy, n.HierarchyName, n.HierarchyLevels)));
        }
        else
        {
            Scheme = new Dictionary<string, (IRArray<SBP> NdSBP, Placement Placement)>();
        }

        _reshardMemo = new(ReferenceEqualityComparer.Instance);
        _inferedMemo = new(ReferenceEqualityComparer.Instance);
        _rootGraph = new(true);
        _rootSearchGraph = new(_rootGraph, SearchGraphKind.Root);
        _moduleKind = moduleKind;
        _bidirectional = bidirectional;
        _profiler = new AutoDistributedProfiler(moduleKind, phase);
        _recordCandidateDiagnostics = string.Equals(Environment.GetEnvironmentVariable("NNCASE_DUMP_AD_CANDIDATES"), "1", StringComparison.OrdinalIgnoreCase);
    }

    public IRArray<Placement> Placements { get; }

    public bool Bidirectional { get; }

    public CompileOptions CompileOptions { get; }

    public INTTTargetOptions TargetOptions { get; }

    public IReadOnlyDictionary<string, (IRArray<SBP> Policies, Placement Placement)> Scheme { get; }

    /// <summary>
    /// Gets the final distributed consts that are used in the function.
    /// </summary>
    public Dictionary<TensorConst, TensorConst> DistributedConsts { get; } = new(ReferenceEqualityComparer.Instance);

    public static void MemoryExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var consts = vars.Keys.Where(k => k.Expr is Call { Target: IR.Distributed.Boxing { NewType: DistributedType } } call && call.Arguments[0] is TensorConst tc && tc.Value.Length >= 8).ToArray();
        model.Add(LinearExpr.WeightedSum(consts.Select(k => vars[k]), consts.Select(k =>
        {
            var type = DistributedUtility.GetDividedTensorType((DistributedType)k.Expr.CheckedType);
            var maxShape = CompilerServices.GetMaxShape(type.Shape);
            return TensorUtilities.GetProduct(maxShape) * type.DType.SizeInBytes;
        })) < (2L * 512L * 1024L * 1024L));
    }

    public static bool SingleNodeMemoryCheck(DistributedType distributedType, string moduleKind, INTTTargetOptions targetOptions)
    {
        if (moduleKind == "xpu")
        {
            var type = DistributedUtility.GetDividedTensorType(distributedType);
            var maxShape = CompilerServices.GetMaxShape(type.Shape);
            var size = TensorUtilities.GetProduct(maxShape) * type.DType.SizeInBytes;

            return size < GetSingleBlockMemorySize(distributedType, targetOptions);
        }

        return true;
    }

    public static bool SupportsConstShardedView(INTTTargetOptions targetOptions)
        => DistributedReshardRealizationPolicy.Get(targetOptions).UsesShardedViewsForConstants(targetOptions);

    private static bool IsDistributableTensorType(TensorType tensorType)
        => tensorType.DType is not ReferenceType;

    private static bool ContainsDistributableTensorType(IRType type) => type switch
    {
        DistributedType => true,
        TensorType tensorType => IsDistributableTensorType(tensorType),
        TupleType tupleType => tupleType.Fields.Any(ContainsDistributableTensorType),
        _ => false,
    };

    public static IReadOnlyList<DistributedType> GetLeafCandidateDistTypes(TensorType tensorType, IEnumerable<Placement> placements, string moduleKind, INTTTargetOptions targetOptions)
    {
        if (!IsDistributableTensorType(tensorType))
        {
            return Array.Empty<DistributedType>();
        }

        return placements.Select(
            placement =>
            DistributedUtility.GetLeafCandidatePolicies(tensorType, placement, targetOptions.DistributedSplitCandidateProvider)
            .Where(p => SingleNodeMemoryCheck(new(tensorType, p, placement), moduleKind, targetOptions))
            .Select(ndsbp => new DistributedType(tensorType, ndsbp, placement)))
            .SelectMany(e => e).ToArray();
    }

    public void SingleNodeMemoryExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var distTypes = vars.Keys.Where(k => k.Expr.CheckedType is DistributedType dt).ToArray();
        foreach (var k in distTypes)
        {
            if (TargetOptions.HierarchySizes.Length > 1)
            {
                var type = DistributedUtility.GetDividedTensorType((DistributedType)k.Expr.CheckedType);
                var maxShape = CompilerServices.GetMaxShape(type.Shape);
                var size = TensorUtilities.GetProduct(maxShape) * type.DType.SizeInBytes;

                if (k.Expr is Call call)
                {
                    for (var i = 0; i < call.Arguments.Length; i++)
                    {
                        if (call.Arguments[i].CheckedType is DistributedType inType)
                        {
                            type = DistributedUtility.GetDividedTensorType(inType);
                            size += TensorUtilities.GetProduct(type.Shape.ToValueArray()) * type.DType.SizeInBytes;
                        }
                    }
                }

                model.Add(vars[k] * size < GetSingleBlockMemorySize((DistributedType)k.Expr.CheckedType, TargetOptions));
            }
        }
    }

    public void FilterByScheme(BaseExpr expr, DistributedSearchGraph cluster)
    {
        bool Matched(SearchableNode node, (IRArray<SBP> Policies, Placement Placement) tp)
        {
            return node.IRType is DistributedType dtype && DistributedUtility.AreSamePolicies(dtype.AxisPolicies, tp.Policies, false) && dtype.Placement == tp.Placement;
        }

        foreach (var name in expr.Metadata.OutputNames ?? Array.Empty<string>())
        {
            if (Scheme.TryGetValue(name, out var tp))
            {
                if (cluster.Kind is SearchGraphKind.DistributedCluster)
                {
                    if (!cluster.Clusters.OfType<DistributedSearchGraph>().Any(b => Matched(b.Vertices.First(), tp)))
                    {
                        return;
                    }

                    var removes = new List<DistributedSearchGraph>();
                    foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
                    {
                        bucket.RemoveVertexIf(v => !Matched(v, tp));
                        if (bucket.VertexCount == 0)
                        {
                            removes.Add(bucket);
                        }
                    }

                    foreach (var r in removes)
                    {
                        cluster.RemoveCluster(r);
                    }

                    foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>().Where(b => Matched(b.Vertices.First(), tp)))
                    {
                        bucket.RemoveVertexIf(v => _rootSearchGraph.TryGetOutEdges(v, out var edges) && !edges.Any());
                    }
                }
            }
        }
    }

    public Function Rewrite(Function function)
        => RewriteProgram(function, DistributedFunctionGraphUtility.GetReachableFunctionsInCalleeFirstOrder(function));

    public Function RewriteProgram(Function rootFunction, IReadOnlyList<Function> reachableFunctions)
    {
        if (!reachableFunctions.Contains(rootFunction, ReferenceEqualityComparer.Instance))
        {
            throw new InvalidOperationException($"AutoDistributed reachable function list does not contain root function {rootFunction.Name}.");
        }

        foreach (var (function, invocationCount) in DistributedFunctionGraphUtility.GetStaticInvocationCounts(rootFunction, reachableFunctions))
        {
            if (!_functionInvocationCounts.TryAdd(function, invocationCount) &&
                _functionInvocationCounts[function] != invocationCount)
            {
                throw new InvalidOperationException(
                    $"Function {function.Name} invocation count changed from {_functionInvocationCounts[function]} to {invocationCount} within one AutoDistributed search.");
            }
        }

        _profiler.SetFunction(rootFunction.Name);
        DistributedSearchGraph root = null!;
        using (Nncase.IR.UserTrackingScope.Suppress())
        {
            _profiler.Time("build_search_graph", () =>
            {
                foreach (var function in reachableFunctions)
                {
                    using var functionDumpScope = new DumpScope(function.Name);
                    var isEntry = ReferenceEquals(function, rootFunction);
                    var functionRoot = BuildFunctionSearchGraph(function, isEntry);
                    if (isEntry)
                    {
                        root = functionRoot;
                    }
                }

                PropagateCandidateClosure();
            });

            if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.EGraphCost))
            {
                _profiler.Time("dump_search_graph_dot", () =>
                {
                    using var stream = Diagnostics.DumpScope.Current.OpenFile("DistributedSearchGraph.dot");
                    Dump(stream, new Dictionary<SearchableNode, bool>() { }, new Dictionary<SearchableNode, CostModel.Cost>() { }, new Dictionary<SearchableNode, UInt128>() { });
                });
            }
        }

        if (root is null)
        {
            throw new InvalidOperationException($"AutoDistributed failed to build root search graph for {rootFunction.Name}.");
        }

        _profiler.TimeActive(() =>
        {
            using (Nncase.IR.UserTrackingScope.Suppress())
            {
                _ = _profiler.Time("solve_total", () => Solve(root));
            }
        });

        if (_lastPicks is null)
        {
            throw new InvalidOperationException("AutoDistributed solver finished without selected picks.");
        }

        var materializer = new DistributedProgramMaterializer(_rootSearchGraph, _lastPicks);
        var rewritten = materializer.Materialize(rootFunction, reachableFunctions, _functionRootClusters, _functionParameterClusters);
        foreach (var function in rewritten.Values)
        {
            _profiler.Time("rebuild_users", () => UserRebuilder.Rebuild(function));
        }

        _profiler.Write(_rootSearchGraph, _candidateDiagnosticTotal);
        return rewritten[rootFunction];
    }

    private void PropagateCandidateClosure()
    {
        var iterations = 0;
        bool changed;
        do
        {
            iterations++;
            changed = false;
            var demandVersion = _candidateDemandVersion;
            foreach (var site in _allFunctionBoundarySites)
            {
                var formalCluster = GetFunctionParameterClusters(site.Callee)[site.Parameter];
                var tensorType = site.Parameter.CheckedType as TensorType
                    ?? throw new InvalidOperationException(
                        $"Function {site.Callee.Name} parameter {site.Parameter.Name} must remain a TensorType during candidate propagation.");
                changed |= ExpandFunctionBoundarySite(site, formalCluster, tensorType);
            }

            foreach (var site in _operationCandidateSites)
            {
                changed |= ExpandOperationCandidateSite(site);
            }

            foreach (var site in _functionResultSites)
            {
                changed |= ExpandFunctionResultSite(site);
            }

            foreach (var site in _functionCallCandidateSites)
            {
                changed |= ExpandFunctionCallCandidateSite(site);
            }

            changed |= demandVersion != _candidateDemandVersion;
        }
        while (changed);

        _profiler.Count("candidate_closure_iterations", iterations);
    }

    public DistributedSearchGraph BuildSearchGraph(Function function)
    {
        _profiler.SetFunction(function.Name);
        _functionInvocationCounts.TryAdd(function, 1);
        return _profiler.TimeActive(() =>
        {
            DistributedSearchGraph root = null!;
            var existingNodes = new HashSet<SearchableNode>(
                _rootSearchGraph.Vertices,
                ReferenceEqualityComparer.Instance);
            _currentFunction = function;
            _currentFunctionIsEntry = true;
            using (Nncase.IR.UserTrackingScope.Suppress())
            {
                try
                {
                    _profiler.Time("build_search_graph", () =>
                    {
                        Visit(function.Body);
                        root = TryInstertTerminator(function.Body);
                    });

                    if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.EGraphCost))
                    {
                        _profiler.Time("dump_search_graph_dot", () =>
                        {
                            using var stream = Diagnostics.DumpScope.Current.OpenFile("DistributedSearchGraph.dot");
                            Dump(stream, new Dictionary<SearchableNode, bool>() { }, new Dictionary<SearchableNode, CostModel.Cost>() { }, new Dictionary<SearchableNode, UInt128>() { });
                        });
                    }
                }
                finally
                {
                    RecordNewFunctionNodes(function, existingNodes);
                    _currentFunction = null;
                    _currentFunctionIsEntry = false;
                }
            }

            return root;
        });
    }

    private DistributedSearchGraph BuildFunctionSearchGraph(Function function, bool isEntry)
    {
        _functionInvocationCounts.TryAdd(function, 1);
        var existingNodes = new HashSet<SearchableNode>(
            _rootSearchGraph.Vertices,
            ReferenceEqualityComparer.Instance);
        _currentFunction = function;
        _currentFunctionIsEntry = isEntry;
        try
        {
            Visit(function.Body);
            var root = isEntry ? TryInstertTerminator(function.Body) : CreateFunctionResultCluster(function.Body);
            _functionRootClusters[function] = root;
            if (!isEntry)
            {
                _functionReturnClusters[function] = root;
                _singleChoiceClusters.Add(root);
            }

            return root;
        }
        finally
        {
            RecordNewFunctionNodes(function, existingNodes);
            _currentFunction = null;
            _currentFunctionIsEntry = false;
        }
    }

    private void RecordNewFunctionNodes(Function function, IReadOnlySet<SearchableNode> existingNodes)
    {
        foreach (var node in _rootSearchGraph.Vertices.Where(node => !existingNodes.Contains(node)))
        {
            if (_nodeOwnerFunctions.ContainsKey(node))
            {
                // A caller can contribute a new signature candidate to a callee's
                // parameter cluster. Nodes created while wiring that candidate into
                // the callee are explicitly assigned to the callee before this scan.
                continue;
            }

            _nodeOwnerFunctions[node] = function;
        }
    }

    private void RegisterNodeOwner(SearchableNode node, Function function)
    {
        if (_nodeOwnerFunctions.TryGetValue(node, out var owner))
        {
            if (!ReferenceEquals(owner, function))
            {
                throw new InvalidOperationException(
                    $"AutoDistributed search node is owned by both {owner.Name} and {function.Name}.");
            }

            return;
        }

        _nodeOwnerFunctions.Add(node, function);
    }

    private DistributedSearchGraph CreateFunctionResultCluster(BaseExpr result)
    {
        var function = _currentFunction
            ?? throw new InvalidOperationException("Function result candidates must be created while building a function search graph.");
        if (result is IR.Tuple tuple)
        {
            var fieldClusters = tuple.Fields
                .AsValueEnumerable()
                .Select(CreateFunctionResultCluster)
                .ToArray();
            var tupleCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var tupleSite = new FunctionResultSite(function, result, tupleCluster, fieldClusters);
            _functionResultSites.Add(tupleSite);
            ExpandFunctionResultSite(tupleSite);
            if (tupleCluster.VertexCount == 0)
            {
                throw new InvalidOperationException(
                    $"Function result tuple in {function.Name} has no direct producer candidates.");
            }

            return tupleCluster;
        }

        var valueCluster = TryAddOriginator(result);
        var sourceBuckets = _directValueBuckets.TryGetValue(result, out var directBuckets)
            ? directBuckets
            : valueCluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
        if (sourceBuckets.Count == 0)
        {
            throw new InvalidOperationException(
                $"Function result {result.GetType().Name} has no direct producer candidates.");
        }

        var resultCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(valueCluster.Kind);
        var resultSite = new FunctionResultSite(function, result, resultCluster);
        _functionResultSites.Add(resultSite);
        ExpandFunctionResultSite(resultSite);

        return resultCluster;
    }

    private bool ExpandFunctionResultSite(FunctionResultSite site)
    {
        if (site.FieldClusters is { } fieldClusters)
        {
            var fieldBuckets = fieldClusters
                .Select((cluster, fieldIndex) =>
                {
                    var buckets = cluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
                    if (buckets.Length == 0)
                    {
                        throw new InvalidOperationException(
                            $"Function {site.Function.Name} result tuple field {fieldIndex} has no direct producer candidates.");
                    }

                    return buckets;
                })
                .ToArray();
            var changed = false;
            foreach (var combination in fieldBuckets.Select(buckets => buckets.AsEnumerable()).CartesianProduct())
            {
                var selectedBuckets = combination.ToArray();
                if (!site.FieldCombinations.Add(new BucketCombinationIdentity(selectedBuckets)))
                {
                    continue;
                }

                var tupleNode = new SearchableNode(
                    new IR.Tuple(),
                    new TupleType(selectedBuckets.Select(GetBucketType).ToArray()),
                    kind: SearchableNodeKind.FunctionResult);
                var tupleBucket = site.ResultCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                tupleBucket.AddVertex(tupleNode);
                RegisterNodeOwner(tupleNode, site.Function);
                for (var index = 0; index < selectedBuckets.Length; index++)
                {
                    var fieldBucket = selectedBuckets[index];
                    _rootSearchGraph.AddEdge(new(tupleNode, fieldBucket.Vertices.First(), index, fieldBucket));
                }

                changed = true;
            }

            return changed;
        }

        var valueCluster = TryAddOriginator(site.Result);
        var sourceBuckets = _directValueBuckets.TryGetValue(site.Result, out var directBuckets)
            ? directBuckets
            : valueCluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
        var expanded = false;
        foreach (var sourceBucket in sourceBuckets)
        {
            if (!site.SourceBuckets.Add(sourceBucket))
            {
                continue;
            }

            var sourceNode = sourceBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException("Function result source bucket is empty.");
            var resultBucket = site.ResultCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var resultNode = new SearchableNode(
                site.Result,
                sourceNode.IRType,
                kind: SearchableNodeKind.FunctionResult,
                sourceKind: sourceNode.SourceKind);
            resultBucket.AddVertex(resultNode);
            RegisterNodeOwner(resultNode, site.Function);
            _rootSearchGraph.AddEdge(new(resultNode, sourceNode, 0, sourceBucket));
            expanded = true;
        }

        return expanded;

        static IRType GetBucketType(DistributedSearchGraph bucket)
            => bucket.Vertices.FirstOrDefault()?.IRType
                ?? throw new InvalidOperationException("Function result bucket is empty.");
    }

    public Function SolveAndExtract(Function function, DistributedSearchGraph root)
    {
        var post = _profiler.TimeActive(() =>
        {
            BaseExpr result;
            using (Nncase.IR.UserTrackingScope.Suppress())
            {
                result = _profiler.Time("solve_and_extract_total", () =>
                {
                    var picks = Solve(root);
                    return ExtractSelectedExpression(root, picks);
                });
            }

            _profiler.Time("rebuild_users", () => UserRebuilder.Rebuild(result));
            return result;
        });
        _profiler.Write(_rootSearchGraph, _candidateDiagnosticTotal);

        return function.With(body: post);
    }

    protected override Unit DefaultVisitLeaf(BaseExpr expr)
    {
        return default;
    }

    protected override Unit VisitLeafCall(Call expr)
    {
        _profiler.Count("calls");

        if (expr.Target is Function callee && _functionReturnClusters.ContainsKey(callee))
        {
            return VisitLeafFunctionCall(expr, callee);
        }

        string DescribeType(IRType type) => type switch
        {
            DistributedType dt => dt.ToString(),
            TensorType t => t.ToString(),
            _ => type.ToString(),
        };

        string DescribeNode(SearchableNode node) => $"{node.Expr.GetType().Name}:{DescribeType(node.IRType)}";

        string DescribeSbp(IRType? type)
        {
            return type switch
            {
                DistributedType dist => $"Placement={dist.Placement}, SBP=[{string.Join(", ", dist.AxisPolicies.Select(p => p.ToString()))}] Tensor={dist.TensorType}",
                TensorType tensor => tensor.ToString(),
                null => "Empty",
                _ => type.ToString(),
            };
        }

        bool isSupported;
        bool isSparseExperts = false;
        var argClusters = new DistributedSearchGraph[expr.Arguments.Length];
        if (expr.Target is not Op op)
        {
            isSupported = false;
            foreach (var (param, i) in expr.Arguments.AsValueEnumerable().Select((p, i) => (p, i)))
            {
                argClusters[i] = VisitLeafArgument(ParameterKind.Input, expr.Arguments[i], isSupported);
            }
        }
        else
        {
            isSupported = expr.Target is AsTensor or IR.Tensors.Range ? false : true;
            isSparseExperts = expr.Target.GetType().FullName?.Contains("CustomNTT.SparseExperts", StringComparison.Ordinal) == true;
            foreach (var param in op.Parameters)
            {
                argClusters[param.Index] = VisitLeafArgument(param.ParameterKind, expr.Arguments[param.Index], isSupported);
            }
        }

        if (isSparseExperts)
        {
            var broadcastList = new List<int> { 1, 2, 3, 5, 6, 8, 9, 11, 12 };
            for (var index = 0; index < argClusters.Length; index++)
            {
                var input = argClusters[index];
                if (broadcastList.Contains(index))
                {
                    var bucketsToRemove = new List<DistributedSearchGraph>();
                    foreach (var bucket in input.Clusters.OfType<DistributedSearchGraph>())
                    {
                        bucket.RemoveVertexIf(v => !(v.IRType is not DistributedType dist ||
                            dist.AxisPolicies.All(policy => policy is SBPBroadCast)));

                        if (bucket.VertexCount == 0)
                        {
                            bucketsToRemove.Add(bucket);
                        }
                    }

                    foreach (var bucket in bucketsToRemove)
                    {
                        argClusters[index].RemoveCluster(bucket);
                    }

                    if (index < 3)
                    {
                        var buckets = input.Clusters.OfType<DistributedSearchGraph>().ToArray();
                        foreach (var bucket in buckets)
                        {
                            bucket.RemoveVertexIf(v => _rootSearchGraph.TryGetOutEdges(v, out var edges) && !edges.Any());
                            if (bucket.VertexCount == 0)
                            {
                                argClusters[index].RemoveCluster(bucket);
                            }
                        }
                    }
                }
            }

            {
                var index = 0;
                var input = argClusters[index];

                var bucketsToRemove = new List<DistributedSearchGraph>();
                foreach (var bucket in input.Clusters.OfType<DistributedSearchGraph>())
                {
                    bucket.RemoveVertexIf(v => !(v.IRType is not DistributedType dt || (dt.AxisPolicies is { Count: > 0 } policies
                        && policies[0] is SBPSplit { HierarchyAxes: [1, 3] }
                        && policies[1] is SBPSplit { HierarchyAxes: [2] })));

                    if (bucket.VertexCount == 0)
                    {
                        bucketsToRemove.Add(bucket);
                    }
                }

                foreach (var bucket in bucketsToRemove)
                {
                    argClusters[index].RemoveCluster(bucket);
                }

                var buckets = input.Clusters.OfType<DistributedSearchGraph>().ToArray();
                foreach (var bucket in buckets)
                {
                    bucket.RemoveVertexIf(v => _rootSearchGraph.TryGetOutEdges(v, out var edges) && !edges.Any());
                    if (bucket.VertexCount == 0)
                    {
                        argClusters[index].RemoveCluster(bucket);
                    }
                }
            }

            List<int> broadcastList2 = new() { 4, 7, 10 }; // expert 0维度为B
            foreach (var index in broadcastList2)
            {
                var input = argClusters[index];
                if (broadcastList2.Contains(index))
                {
                    var bucketsToRemove = new List<DistributedSearchGraph>();
                    foreach (var bucket in input.Clusters.OfType<DistributedSearchGraph>())
                    {
                        bucket.RemoveVertexIf(v => !(v.IRType is not DistributedType dt || (dt.AxisPolicies is { Count: > 0 } policies
                        && policies[0] is SBPBroadCast
                        && policies[1] is SBPSplit { HierarchyAxes: [2] }
                        && policies[2] is SBPSplit { HierarchyAxes: [1, 3] })));

                        if (bucket.VertexCount == 0)
                        {
                            bucketsToRemove.Add(bucket);
                        }
                    }

                    foreach (var bucket in bucketsToRemove)
                    {
                        argClusters[index].RemoveCluster(bucket);
                    }
                }
            }

            // 打印当前arg的节点信息
            for (var index = 0; index < argClusters.Length; index++)
            {
                var input = argClusters[index];
                Console.WriteLine($"[AutoDistributed][SparseExperts] Arg {index} Nodes:");
                foreach (var v in input.Vertices)
                {
                    Console.WriteLine($"\t{DescribeNode(v)}");
                }
            }
        }

        bool isStandalone = expr.Target is IR.NN.UpdatePagedAttentionKVCache;
        var callCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(!isSupported || isStandalone ? SearchGraphKind.StandaloneCluster : SearchGraphKind.DistributedCluster);
        var function = _currentFunction
            ?? throw new InvalidOperationException("Operation candidates must be created while building a function search graph.");
        var candidateSite = new OperationCandidateSite(expr, isSupported, argClusters, callCluster, function);
        _operationCandidateSites.Add(candidateSite);
        if (!_operationCandidateSitesByCall.TryAdd(expr, candidateSite))
        {
            throw new InvalidOperationException("An operation call cannot own more than one distributed candidate site.");
        }

        ExpandOperationCandidateSite(candidateSite, isSparseExperts);

        if (callCluster.VertexCount == 0)
        {
            if (isSparseExperts)
            {
                Console.WriteLine("[AutoDistributed][SparseExperts] No valid candidate survived. Current arg clusters:");
                for (var index = 0; index < argClusters.Length; index++)
                {
                    var input = argClusters[index];
                    foreach (var bucket in input.Clusters.OfType<DistributedSearchGraph>())
                    {
                        var vertex = bucket.Vertices.FirstOrDefault();
                        Console.WriteLine($"\tArg {index}: {DescribeSbp(vertex?.IRType)}");
                    }
                }
            }

            var failureMessage = BuildCandidateFailureMessage(expr, argClusters);
            System.Console.WriteLine(failureMessage);
            throw new InvalidOperationException(failureMessage);
        }

        RecordDirectValueBuckets(expr, candidateSite.DirectOutputBuckets);
        _inferedMemo.Add(expr, callCluster);

        if (!isSupported || isStandalone)
        {
            return default;
        }

        // 3. add bidirectional connections. Function signatures are projected
        // from direct producer buckets separately, so these reshard candidates
        // remain consumer-side alternatives even when the value is returned.
        if (Bidirectional)
        {
            CompleteBidirectionalCandidateClosure(candidateSite);
        }

        // 4. Add tuple partial-materialization alternatives before field consumers split the
        // tuple into independent search clusters.
        if (expr.CheckedType is TupleType)
        {
            CompleteTupleOutputPartialReshardClosure(
                callCluster,
                candidateSite.DirectOutputBuckets,
                DistributedReshardUsageKind.Internal);
            return default;
        }

        // 5. add not infered type in search space.
        if (expr.CheckedType is not TensorType tensorType || !IsDistributableTensorType(tensorType))
        {
            return default;
        }

        CompleteOutputReshardClosure(
            callCluster,
            tensorType,
            candidateSite.DirectOutputBuckets,
            DistributedReshardUsageKind.Internal,
            function);

        // 6. filter
        FilterByScheme(expr, callCluster);
        return default;
    }

    private bool ExpandOperationCandidateSite(OperationCandidateSite site, bool isSparseExperts = false)
    {
        var newOutputBuckets = new List<DistributedSearchGraph>();
        foreach (var candidate in EnumerateCandidateBucketArrays(
                     site.SourceCall,
                     site.IsSupported,
                     site.ArgumentClusters,
                     site.Function,
                     site.DemandedReturnTypes))
        {
            if (!site.Invocations.Add(new CandidateInvocationIdentity(candidate)))
            {
                continue;
            }

            var bucketArray = candidate.Buckets;
            _profiler.Count("candidate_arg_combinations");
            if (isSparseExperts)
            {
                Console.WriteLine("[AutoDistributed][SparseExperts] Candidate SBP combination:");
                foreach (var (bucket, index) in bucketArray.Select((bucket, index) => (bucket, index)))
                {
                    Console.WriteLine($"\tArg {index}: {FormatType(bucket.Vertices.FirstOrDefault()?.IRType)}");
                }
            }

            var tempArgs = bucketArray.Select<DistributedSearchGraph, BaseExpr>(bucket => bucket.Vertices.First() switch
            {
                SearchableNode { Expr: Dimension attr } => attr,
                SearchableNode { Expr: Shape attr } => attr,
                SearchableNode { Expr: Padding attr } => attr,
                SearchableNode { Expr: Paddings attr } => attr,
                SearchableNode { Expr: Const attr } => attr,
                SearchableNode { Expr: Call { Target: AsTensor } attr } => attr,
                SearchableNode n => new Var(n.IRType),
            }).ToArray();
            var newExprs = _profiler.Time(
                "build_equivalent_calls",
                () => BuildEquivalentCalls(candidate.Target, tempArgs).ToArray());
            _profiler.Count("candidate_equivalent_calls", newExprs.Length);
            foreach (var (newExpr, used) in newExprs)
            {
                _profiler.Count("candidate_exprs");
                if (site.SourceCall.Target is not Boxing
                    && !candidate.AllowsPartialInputs
                    && ((Call)newExpr).Arguments.AsValueEnumerable().Any(
                        argument => argument.CheckedType is DistributedType { Partial: not null }))
                {
                    RecordCandidateDiagnostic(
                        site.SourceCall,
                        bucketArray,
                        "infer",
                        "rejected",
                        null,
                        "partial argument is not allowed before boxing");
                    continue;
                }

                if (!InferCandidateType(newExpr))
                {
                    RecordCandidateDiagnostic(
                        site.SourceCall,
                        bucketArray,
                        "infer",
                        "rejected",
                        newExpr.CheckedType,
                        "type inference returned false");
                    continue;
                }

                if (newExpr.CheckedType is InvalidType invalidType)
                {
                    RecordCandidateDiagnostic(
                        site.SourceCall,
                        bucketArray,
                        "infer",
                        "rejected",
                        invalidType,
                        invalidType.Reason);
                    continue;
                }

                if (candidate.ExpectedReturnType is { } expectedReturnType
                    && newExpr.CheckedType != expectedReturnType)
                {
                    RecordCandidateDiagnostic(
                        site.SourceCall,
                        bucketArray,
                        "infer",
                        "rejected",
                        newExpr.CheckedType,
                        $"candidate provider expected {expectedReturnType}, got {newExpr.CheckedType}");
                    continue;
                }

                var checkType = newExpr.CheckedType;
                RecordCandidateDiagnostic(site.SourceCall, bucketArray, "infer", "accepted", checkType, string.Empty);
                if (!site.OutputBuckets.TryGetValue(checkType, out var outputBucket))
                {
                    outputBucket = site.CallCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    site.OutputBuckets.Add(checkType, outputBucket);
                    site.DirectOutputBuckets.Add(outputBucket);
                    newOutputBuckets.Add(outputBucket);
                }

                var node = new SearchableNode(
                    site.IsSupported && newExpr is Call newCall ? newCall.Target : newExpr,
                    checkType,
                    metadataSource: site.SourceCall);
                outputBucket.AddVertex(node);
                RegisterNodeOwner(node, site.Function);
                foreach (var ((argument, _), index) in bucketArray.Zip(used).Where(pair => pair.Second).Select((pair, index) => (pair, index)))
                {
                    _rootSearchGraph.AddEdge(new(node, argument.Vertices.First(), index, argument));
                }
            }
        }

        if (newOutputBuckets.Count == 0)
        {
            return false;
        }

        AppendDirectValueBuckets(site.SourceCall, newOutputBuckets);
        if (site.IsSupported
            && site.SourceCall.Target is not IR.NN.UpdatePagedAttentionKVCache
            && site.SourceCall.CheckedType is TensorType tensorType
            && IsDistributableTensorType(tensorType))
        {
            if (Bidirectional)
            {
                CompleteBidirectionalCandidateClosure(site);
            }

            CompleteOutputReshardClosure(
                site.CallCluster,
                tensorType,
                newOutputBuckets,
                DistributedReshardUsageKind.Internal,
                site.Function);
        }

        return true;
    }

    private void CompleteBidirectionalCandidateClosure(OperationCandidateSite site)
    {
        foreach (var (leftType, leftBucket) in site.OutputBuckets.Where(pair => pair.Key is DistributedType))
        {
            foreach (var (rightType, rightBucket) in site.OutputBuckets.Where(
                         pair => pair.Key is DistributedType && pair.Key != leftType))
            {
                if (CheckBoxingTypeCached(leftType, rightType) is InvalidType)
                {
                    continue;
                }

                GetOrCreateReshardCandidate(
                    site.CallCluster,
                    leftBucket,
                    leftBucket.Vertices.First(),
                    rightType,
                    usageKind: DistributedReshardUsageKind.Internal,
                    isBidirect: true,
                    outputBucket: rightBucket,
                    addDataEdgeToOwnerCluster: true,
                    ownerFunction: site.Function);
            }
        }
    }

    private static UInt128 GetLocalTensorBytes(DistributedType distributedType)
    {
        var type = DistributedUtility.GetDividedTensorType(distributedType, DistributedUtility.DivideFlags.MaxShape);
        var maxShape = CompilerServices.GetMaxShape(type.Shape);
        return (UInt128)(TensorUtilities.GetProduct(maxShape) * type.DType.SizeInBytes);
    }

    private static long GetSingleBlockMemorySize(DistributedType distributedType, INTTTargetOptions targetOptions)
    {
        if (targetOptions.HierarchySizes.Length < 2)
        {
            return long.MaxValue;
        }

        var blockCount = Math.Max(1, distributedType.Placement.GetPhysicalLevelSize('b'));
        return targetOptions.HierarchySizes[^2] / blockCount;
    }

    private IReadOnlyList<DistributedType> GetLeafCandidateDistTypes(TensorType tensorType)
    {
        var key = new LeafCandidateKey(tensorType);
        if (_leafCandidateMemo.TryGetValue(key, out var cached))
        {
            _profiler.Count("leaf_candidate_cache_hit");
            return cached;
        }

        _profiler.Count("leaf_candidate_cache_miss");
        var candidates = GetLeafCandidateDistTypes(tensorType, Placements, _moduleKind, TargetOptions);
        _leafCandidateMemo.Add(key, candidates);
        return candidates;
    }

    private IReadOnlyList<DistributedReshardPlan> GetReshardPlans(
        IRType sourceType,
        IRType targetType,
        DistributedReshardSourceKind sourceKind,
        DistributedReshardUsageKind usageKind,
        int maxHops = DistributedReshardPlanner.DefaultMaxHops)
    {
        var key = new ReshardPlanKey(sourceType, targetType, sourceKind, usageKind, maxHops);
        if (_reshardPlanMemo.TryGetValue(key, out var cached))
        {
            _profiler.Count("reshard_plan_cache_hit");
            return cached;
        }

        _profiler.Count("reshard_plan_cache_miss");
        var plans = DistributedReshardPlanner.Plan(sourceType, targetType, CanRealize, maxHops);
        _reshardPlanMemo.Add(key, plans);
        return plans;

        bool CanRealize(IRType edgeSourceType, IRType edgeTargetType)
        {
            if (!CanBoxingType(edgeSourceType, edgeTargetType))
            {
                return false;
            }

            var edgeSourceKind = EqualityComparer<IRType>.Default.Equals(edgeSourceType, sourceType)
                ? sourceKind
                : DistributedReshardSourceKind.Internal;
            var edgeUsageKind = EqualityComparer<IRType>.Default.Equals(edgeTargetType, targetType)
                ? usageKind
                : DistributedReshardUsageKind.Internal;
            return ClassifyReshardRealization(
                edgeSourceType,
                edgeTargetType,
                edgeSourceKind,
                edgeUsageKind) != DistributedReshardRealization.Unsupported;
        }
    }

    private IRType CheckBoxingTypeCached(IRType inType, IRType outType, bool isReshape = false)
    {
        var key = new BoxingTypeKey(inType, outType, isReshape);
        if (_boxingTypeMemo.TryGetValue(key, out var cached))
        {
            _profiler.Count("boxing_type_cache_hit");
            return cached;
        }

        _profiler.Count("boxing_type_cache_miss");
        var result = CheckBoxingType(inType, outType, isReshape);
        _boxingTypeMemo.Add(key, result);
        return result;
    }

    private bool InferCandidateType(Expr candidate)
    {
        if (candidate is not Call call)
        {
            return _profiler.Time("type_inference", () => candidate.InferenceType(_inferencer_cache));
        }

        var key = new TypeInferenceCacheKey(call);
        if (_typeInferenceMemo.TryGetValue(key, out var cached))
        {
            _profiler.Count("type_inference_cache_hit");

            // Candidate calls are ephemeral search values. Restoring a cached
            // type must not invalidate users of a structurally reused call.
            IRHelpers.SetRawCheckedType(call, cached.CheckedType);
            return cached.Success;
        }

        _profiler.Count("type_inference_cache_miss");
        var success = _profiler.Time("type_inference", () => call.InferenceType(_inferencer_cache));
        _typeInferenceMemo.Add(key, (success, call.CheckedType));
        return success;
    }

    private IEnumerable<CandidateInvocation> EnumerateCandidateBucketArrays(
        Call expr,
        bool isSupported,
        IReadOnlyList<DistributedSearchGraph> argClusters,
        Function ownerFunction,
        IReadOnlySet<IRType> demandedReturnTypes)
    {
        var providerResult = TryBuildProviderCandidateBucketArrays(
            expr,
            isSupported,
            argClusters,
            ownerFunction,
            demandedReturnTypes);
        if (providerResult.Invocations.Count > 0)
        {
            foreach (var invocation in providerResult.Invocations)
            {
                yield return invocation;
            }

            yield break;
        }

        if (providerResult.IsExhaustive)
        {
            yield break;
        }

        var allowsPartialInputs = expr.Target is Op op &&
            _candidateProviderResolver is not null &&
            _candidateProviderResolver.TryGetProvider(op, out var fallbackProvider) &&
            fallbackProvider.AllowsPartialInputs;

        foreach (var combBuckets in argClusters.Select(c => c.Clusters.OfType<DistributedSearchGraph>()).CartesianProduct())
        {
            yield return new CandidateInvocation(
                combBuckets.ToArray(),
                expr.Target,
                null,
                allowsPartialInputs);
        }
    }

    private ProviderCandidateResult TryBuildProviderCandidateBucketArrays(
        Call expr,
        bool isSupported,
        IReadOnlyList<DistributedSearchGraph> argClusters,
        Function ownerFunction,
        IReadOnlySet<IRType> demandedReturnTypes)
    {
        if (!isSupported || expr.Target is not Op op || _candidateProviderResolver is null || !_candidateProviderResolver.TryGetProvider(op, out var provider))
        {
            return new(Array.Empty<CandidateInvocation>(), false);
        }

        _profiler.Count("candidate_provider_queries");
        var candidatesByInput = argClusters
            .Select(cluster => cluster.Clusters
                .OfType<DistributedSearchGraph>()
                .Select(bucket => bucket.Vertices.FirstOrDefault() is { } node ? (Type: node.IRType, Bucket: bucket) : (Type: (IRType?)null, Bucket: (DistributedSearchGraph?)null))
                .Where(candidate => candidate.Type is not null && candidate.Bucket is not null)
                .Select(candidate => (Type: candidate.Type!, Bucket: candidate.Bucket!))
                .ToArray())
            .ToArray();
        if (candidatesByInput.Any(candidates => candidates.Length == 0))
        {
            _profiler.Count("candidate_provider_empty_input");
            return new(Array.Empty<CandidateInvocation>(), provider.IsExhaustive);
        }

        var availableInputTypes = candidatesByInput
            .Select(candidates => (IReadOnlyList<IRType>)candidates.Select(candidate => candidate.Type).Distinct().ToArray())
            .ToArray();
        var context = new DistributedCandidateContext(CompileOptions, TargetOptions, _moduleKind, expr, availableInputTypes);
        var defaultReturnTypes = GetProviderReturnCandidateTypes(expr.CheckedType)
            .Concat(demandedReturnTypes)
            .Distinct()
            .ToArray();
        var returnTypes = provider
            .GetReturnCandidateTypes(context, op, defaultReturnTypes)
            .Distinct()
            .ToArray();
        if (returnTypes.Length == 0)
        {
            _profiler.Count("candidate_provider_no_return_types");
            return new(Array.Empty<CandidateInvocation>(), provider.IsExhaustive);
        }

        if (returnTypes.Length > MaxProviderReturnCandidateTypes)
        {
            throw new InvalidOperationException(
                $"Distributed candidate provider {provider.GetType().Name} returned {returnTypes.Length} output types " +
                $"for {op.GetType().Name}; the limit is {MaxProviderReturnCandidateTypes}.");
        }

        var bucketsByInputType = candidatesByInput
            .Select(candidates => candidates
                .GroupBy(candidate => candidate.Type)
                .ToDictionary(group => group.Key, group => group.Select(candidate => candidate.Bucket).ToArray()))
            .ToArray();
        var result = new List<CandidateInvocation>();
        var tupleCount = 0;
        foreach (var returnType in returnTypes)
        {
            if (!provider.TryGetInputTypeTuples(context, op, returnType, out var tuples) || tuples.Count == 0)
            {
                continue;
            }

            tupleCount += tuples.Count;
            var candidateTarget = provider.CreateCandidateTarget(context, op, returnType);
            foreach (var tuple in tuples)
            {
                ExpandProviderTuple(
                    tuple,
                    expr.Arguments.ToArray(),
                    argClusters,
                    bucketsByInputType,
                    result,
                    candidateTarget,
                    returnType,
                    provider.AllowsPartialInputs,
                    ownerFunction,
                    demandedReturnTypes.Contains(returnType));
            }
        }

        _profiler.Count("candidate_provider_return_types", returnTypes.Length);
        _profiler.Count("candidate_provider_tuples", tupleCount);
        _profiler.Count("candidate_provider_bucket_arrays", result.Count);
        if (result.Count == 0)
        {
            _profiler.Count("candidate_provider_fallback");
            return new(Array.Empty<CandidateInvocation>(), provider.IsExhaustive);
        }

        _profiler.Count("candidate_provider_hit");
        return new(result, provider.IsExhaustive);
    }

    private void ExpandProviderTuple(
        DistributedCandidateTuple tuple,
        IReadOnlyList<BaseExpr> inputExpressions,
        IReadOnlyList<DistributedSearchGraph> inputClusters,
        IReadOnlyList<Dictionary<IRType, DistributedSearchGraph[]>> bucketsByInputType,
        List<CandidateInvocation> result,
        Expr candidateTarget,
        IRType expectedReturnType,
        bool allowsPartialInputs,
        Function ownerFunction,
        bool propagateInputDemands)
    {
        if (tuple.InputTypes.Count != bucketsByInputType.Count)
        {
            return;
        }

        var selectedBuckets = new DistributedSearchGraph[tuple.InputTypes.Count];
        for (int i = 0; i < tuple.InputTypes.Count; i++)
        {
            if (propagateInputDemands)
            {
                RegisterOperationReturnDemand(inputExpressions[i], tuple.InputTypes[i]);
            }

            if (!bucketsByInputType[i].TryGetValue(tuple.InputTypes[i], out var buckets))
            {
                buckets = GetOrCreateProviderInputTypeBuckets(
                    inputExpressions[i],
                    inputClusters[i],
                    tuple.InputTypes[i],
                    ownerFunction).ToArray();
                if (buckets.Length > 0)
                {
                    bucketsByInputType[i].Add(tuple.InputTypes[i], buckets);
                }
            }

            if (buckets.Length == 0)
            {
                return;
            }

            selectedBuckets[i] = GetOrCreateProviderInputChoiceBucket(
                inputExpressions[i],
                inputClusters[i],
                tuple.InputTypes[i],
                buckets,
                ownerFunction);
        }

        result.Add(new CandidateInvocation(
            selectedBuckets,
            candidateTarget,
            expectedReturnType,
            allowsPartialInputs));
    }

    private DistributedSearchGraph GetOrCreateProviderInputChoiceBucket(
        BaseExpr inputExpression,
        DistributedSearchGraph inputCluster,
        IRType targetType,
        IReadOnlyList<DistributedSearchGraph> sourceBuckets,
        Function ownerFunction)
    {
        if (sourceBuckets.Count == 1)
        {
            return sourceBuckets[0];
        }

        var key = new ProviderInputChoiceKey(inputCluster, targetType, ownerFunction);
        if (!_providerInputChoiceMemo.TryGetValue(key, out var state))
        {
            var choiceCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var choiceBucket = choiceCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            state = new ProviderInputChoiceState(choiceBucket);
            _providerInputChoiceMemo.Add(key, state);
        }

        foreach (var sourceBucket in sourceBuckets)
        {
            if (!state.SourceBuckets.Add(sourceBucket))
            {
                continue;
            }

            var sourceNode = sourceBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException("A provider input realization bucket cannot be empty.");
            if (!EqualityComparer<IRType>.Default.Equals(sourceNode.IRType, targetType))
            {
                throw new InvalidOperationException(
                    $"Provider input realization type {sourceNode.IRType} does not match requested type {targetType}.");
            }

            var choiceNode = new SearchableNode(
                inputExpression,
                targetType,
                kind: SearchableNodeKind.TypeAdapter,
                sourceKind: sourceNode.SourceKind,
                originParameter: sourceNode.OriginParameter);
            state.ChoiceBucket.AddVertex(choiceNode);
            RegisterNodeOwner(choiceNode, ownerFunction);
            _rootSearchGraph.AddEdge(new(choiceNode, sourceNode, 0, sourceBucket));
            _profiler.Count("candidate_provider_factored_input_realizations");
        }

        return state.ChoiceBucket;
    }

    private static bool HaveSameLogicalTensorType(IRType sourceType, IRType demandedType)
    {
        var sourceTensorType = sourceType switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => null,
        };
        var demandedTensorType = demandedType switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => null,
        };
        return sourceTensorType is not null && sourceTensorType == demandedTensorType;
    }

    private bool RegisterOperationReturnDemand(BaseExpr expression, IRType demandedType)
    {
        if (expression is not Call call ||
            !_operationCandidateSitesByCall.TryGetValue(call, out var site) ||
            !HaveSameLogicalTensorType(call.CheckedType, demandedType) ||
            !site.DemandedReturnTypes.Add(demandedType))
        {
            return false;
        }

        _candidateDemandVersion++;
        _profiler.Count("operation_demanded_return_types");
        return true;
    }

    private IReadOnlyList<DistributedSearchGraph> GetOrCreateProviderInputTypeBuckets(
        BaseExpr inputExpression,
        DistributedSearchGraph inputCluster,
        IRType targetType,
        Function ownerFunction)
    {
        var key = new ProviderInputTypeKey(inputCluster, targetType);
        if (_providerInputTypeMemo.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var existingBuckets = inputCluster.Clusters
            .OfType<DistributedSearchGraph>()
            .Where(bucket => bucket.Vertices.FirstOrDefault()?.IRType == targetType)
            .ToArray();

        if (inputCluster.Kind != SearchGraphKind.DistributedCluster ||
            !ProviderInputTypeMemoryCheck(targetType))
        {
            _providerInputTypeMemo.Add(key, existingBuckets);
            return existingBuckets;
        }

        if (targetType is DistributedType distributedTarget &&
            TryCreateDirectConstantProviderInputBucket(
                inputExpression,
                inputCluster,
                distributedTarget,
                ownerFunction) is { } directConstantBucket)
        {
            IReadOnlyList<DistributedSearchGraph> direct = [directConstantBucket];
            _providerInputTypeMemo.Add(key, direct);
            _profiler.Count("candidate_provider_direct_constant_sharded_views");
            return direct;
        }

        var paths = new List<(DistributedSearchGraph SourceBucket, SearchableNode SourceNode, IReadOnlyList<IRType> Steps)>();
        foreach (var sourceBucket in inputCluster.Clusters.OfType<DistributedSearchGraph>().ToArray())
        {
            foreach (var sourceNode in sourceBucket.Vertices.ToArray())
            {
                foreach (var plan in GetReshardPlans(
                             sourceNode.IRType,
                             targetType,
                             GetReshardSourceKind(sourceNode),
                             DistributedReshardUsageKind.Internal))
                {
                    if (plan.StepTypes.Count == 0 || plan.StepTypes[^1] != targetType)
                    {
                        throw new InvalidOperationException(
                            $"Reshard planner returned an invalid provider-input path from {sourceNode.IRType} to {targetType}.");
                    }

                    paths.Add((sourceBucket, sourceNode, plan.StepTypes));
                }
            }
        }

        if (paths.Count == 0)
        {
            _providerInputTypeMemo.Add(key, existingBuckets);
            return existingBuckets;
        }

        var adaptationCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        var targetBucket = adaptationCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        DistributedSearchGraph? pathCluster = null;
        foreach (var path in paths)
        {
            pathCluster = AddOutputReshardPath(
                adaptationCluster,
                pathCluster,
                path.SourceBucket,
                path.SourceNode,
                targetBucket,
                path.Steps,
                DistributedReshardUsageKind.Internal,
                ownerFunction);
        }

        IReadOnlyList<DistributedSearchGraph> created = [.. existingBuckets, targetBucket];
        _providerInputTypeMemo.Add(key, created);
        _profiler.Count("candidate_provider_input_types_materialized");
        return created;
    }

    private DistributedSearchGraph? TryCreateDirectConstantProviderInputBucket(
        BaseExpr inputExpression,
        DistributedSearchGraph inputCluster,
        DistributedType targetType,
        Function ownerFunction)
    {
        if (inputExpression is not TensorConst { ValueType: TensorType sourceType } source ||
            ClassifyReshardRealization(
                sourceType,
                targetType,
                DistributedReshardSourceKind.Constant,
                DistributedReshardUsageKind.Internal) != DistributedReshardRealization.ShardedView)
        {
            return null;
        }

        var sourceBucket = GetOrCreateConstantStorageBucket(source);
        var sourceNode = sourceBucket.Vertices.Single();
        var (bucket, _) = GetOrCreateReshardCandidate(
            inputCluster,
            sourceBucket,
            sourceNode,
            targetType,
            usageKind: DistributedReshardUsageKind.Internal,
            ownerFunction: ownerFunction);
        return bucket;
    }

    private bool ProviderInputTypeMemoryCheck(IRType type) => type switch
    {
        DistributedType distributedType => SingleNodeMemoryCheck(distributedType, _moduleKind, TargetOptions),
        TupleType tupleType => tupleType.Fields.All(ProviderInputTypeMemoryCheck),
        _ => true,
    };

    private IReadOnlyList<IRType> GetProviderReturnCandidateTypes(IRType type)
    {
        return type switch
        {
            DistributedType distributedType => [distributedType],
            TensorType tensorType when IsDistributableTensorType(tensorType) => GetLeafCandidateDistTypes(tensorType).Cast<IRType>().ToArray(),
            TensorType tensorType => [tensorType],
            TupleType tupleType => GetProviderTupleReturnCandidateTypes(tupleType),
            _ => Array.Empty<IRType>(),
        };
    }

    private IReadOnlyList<IRType> GetProviderTupleReturnCandidateTypes(TupleType tupleType)
    {
        var fieldCandidates = tupleType.Fields.Select(GetProviderReturnCandidateTypes).ToArray();
        if (fieldCandidates.Any(candidates => candidates.Count == 0))
        {
            return Array.Empty<IRType>();
        }

        long count = 1;
        foreach (var candidates in fieldCandidates)
        {
            count *= candidates.Count;
            if (count > MaxProviderReturnCandidateTypes)
            {
                return Array.Empty<IRType>();
            }
        }

        return fieldCandidates
            .Select(candidates => candidates.AsEnumerable())
            .CartesianProduct()
            .Select(fields => (IRType)new TupleType(fields.ToArray()))
            .ToArray();
    }

    private void RecordCandidateDiagnostic(
        Call sourceCall,
        IReadOnlyList<DistributedSearchGraph> argBuckets,
        string stage,
        string status,
        IRType? resultType,
        string reason)
    {
        _candidateDiagnosticTotal++;
        _profiler.Count($"candidate_{status}");
        if (!_recordCandidateDiagnostics)
        {
            return;
        }

        var arguments = string.Join(" | ", argBuckets.Select((bucket, index) =>
        {
            var type = bucket.Vertices.FirstOrDefault()?.IRType;
            return $"P{index}:{FormatType(type)}";
        }));
        var key = new CandidateDiagnosticKey(
            GetExprLabel(sourceCall.Target),
            stage,
            status,
            FormatType(resultType),
            string.IsNullOrWhiteSpace(reason) ? "-" : GetOneLine(reason),
            arguments);
        _candidateDiagnostics[key] = _candidateDiagnostics.TryGetValue(key, out var count) ? count + 1 : 1;
    }

    private void CompleteOutputReshardClosure(
        DistributedSearchGraph callCluster,
        TensorType tensorType,
        IReadOnlyList<DistributedSearchGraph> directOutputBuckets,
        DistributedReshardUsageKind usageKind,
        Function? ownerFunction = null)
    {
        var targetTypes = _profiler.Time(
            "output_reshard_get_target_types",
            () => GetLeafCandidateDistTypes(tensorType).Distinct().Cast<IRType>().ToArray());
        CompleteOutputReshardClosure(
            callCluster,
            targetTypes,
            directOutputBuckets,
            usageKind,
            ownerFunction: ownerFunction);
    }

    private void CompleteTupleOutputPartialReshardClosure(
        DistributedSearchGraph callCluster,
        IReadOnlyList<DistributedSearchGraph> directOutputBuckets,
        DistributedReshardUsageKind usageKind)
    {
        var sources = directOutputBuckets
            .Where(bucket => bucket.Vertices.FirstOrDefault()?.IRType is TupleType tupleType && ContainsPartial(tupleType))
            .ToArray();
        var targetTypes = directOutputBuckets
            .Select(bucket => bucket.Vertices.FirstOrDefault()?.IRType)
            .OfType<TupleType>()
            .Where(tupleType => !ContainsPartial(tupleType))
            .Distinct()
            .Cast<IRType>()
            .ToArray();
        if (sources.Length == 0 || targetTypes.Length == 0)
        {
            return;
        }

        CompleteOutputReshardClosure(
            callCluster,
            targetTypes,
            sources,
            usageKind,
            IsTuplePartialMaterialization);
    }

    private void CompleteOutputReshardClosure(
        DistributedSearchGraph callCluster,
        IReadOnlyList<IRType> targetTypes,
        IReadOnlyList<DistributedSearchGraph> directOutputBuckets,
        DistributedReshardUsageKind usageKind,
        Func<IRType, IRType, bool>? canReshard = null,
        Function? ownerFunction = null)
    {
        if (!_outputReshardClosureStates.TryGetValue(callCluster, out var state))
        {
            state = new OutputReshardClosureState(usageKind);
            _outputReshardClosureStates.Add(callCluster, state);
        }
        else if (state.UsageKind != usageKind)
        {
            throw new InvalidOperationException(
                $"Output reshard closure for one cluster changed usage from {state.UsageKind} to {usageKind}.");
        }

        var sources = directOutputBuckets
            .Where(state.Sources.Add)
            .Select(bucket => (
                Bucket: bucket,
                Node: bucket.Vertices.FirstOrDefault()
                    ?? throw new InvalidOperationException("An inferred output bucket cannot be empty when completing its reshard closure.")))
            .ToArray();
        _profiler.Count("output_reshard_target_types", targetTypes.Count);
        _profiler.Count("output_reshard_source_buckets", sources.Length);

        foreach (var targetType in targetTypes)
        {
            // Keep reshard endpoints out of directly inferred buckets to preserve a DAG,
            // while coalescing equivalent target states into one consumer-visible bucket.
            state.TargetBuckets.TryGetValue(targetType, out var targetBucket);
            foreach (var source in sources)
            {
                if (canReshard is not null && !canReshard(source.Node.IRType, targetType))
                {
                    continue;
                }

                var plans = _profiler.Time(
                    "output_reshard_plan",
                    () => GetReshardPlans(
                        source.Node.IRType,
                        targetType,
                        GetReshardSourceKind(source.Node),
                        usageKind));
                _profiler.Count("output_reshard_plans", plans.Count);
                foreach (var plan in plans)
                {
                    if (plan.StepTypes.Count == 0
                        || !EqualityComparer<IRType>.Default.Equals(plan.StepTypes[^1], targetType))
                    {
                        throw new InvalidOperationException(
                            $"Reshard planner returned an invalid path from {source.Node.IRType} to {targetType}.");
                    }

                    if (targetBucket is null)
                    {
                        targetBucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                        state.TargetBuckets.Add(targetType, targetBucket);
                    }

                    _profiler.Count("output_reshard_paths");
                    _profiler.Count("output_reshard_steps", plan.StepTypes.Count);
                    state.PathCluster = AddOutputReshardPath(
                        callCluster,
                        state.PathCluster,
                        source.Bucket,
                        source.Node,
                        targetBucket,
                        plan.StepTypes,
                        usageKind,
                        ownerFunction);
                }
            }

            if (targetBucket is not null)
            {
                _profiler.Count("output_reshard_target_buckets");
            }
        }
    }

    private static bool IsTuplePartialMaterialization(IRType sourceType, IRType targetType)
    {
        var hasPartialMaterialization = false;
        return Visit(sourceType, targetType) && hasPartialMaterialization;

        bool Visit(IRType source, IRType target)
        {
            if (source == target)
            {
                return true;
            }

            if (source is TupleType sourceTuple &&
                target is TupleType targetTuple &&
                sourceTuple.Count == targetTuple.Count &&
                sourceTuple.IsVariadic == targetTuple.IsVariadic)
            {
                for (int i = 0; i < sourceTuple.Count; i++)
                {
                    if (!Visit(sourceTuple[i], targetTuple[i]))
                    {
                        return false;
                    }
                }

                return true;
            }

            if (source is DistributedType { Partial: not null } &&
                target is DistributedType { Partial: null })
            {
                hasPartialMaterialization = true;
                return true;
            }

            return false;
        }
    }

    private DistributedSearchGraph? AddOutputReshardPath(
        DistributedSearchGraph callCluster,
        DistributedSearchGraph? pathCluster,
        DistributedSearchGraph sourceBucket,
        SearchableNode sourceNode,
        DistributedSearchGraph targetBucket,
        IReadOnlyList<IRType> stepTypes,
        DistributedReshardUsageKind usageKind,
        Function? ownerFunction = null)
    {
        var inputBucket = sourceBucket;
        var inputNode = sourceNode;
        for (int i = 0; i < stepTypes.Count; i++)
        {
            var isFinalStep = i == stepTypes.Count - 1;
            var stepUsageKind = isFinalStep
                ? usageKind
                : DistributedReshardUsageKind.Internal;
            var ownerCluster = isFinalStep
                ? callCluster
                : pathCluster ??= _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var (bucket, node) = GetOrCreateReshardCandidate(
                ownerCluster,
                inputBucket,
                inputNode,
                stepTypes[i],
                usageKind: stepUsageKind,
                outputBucket: isFinalStep ? targetBucket : null,
                ownerFunction: ownerFunction);
            inputBucket = bucket;
            inputNode = node;
        }

        return pathCluster;
    }

    private bool CanBoxingType(IRType inputType, IRType outputType) => CheckBoxingTypeCached(inputType, outputType) is not InvalidType;

    private (DistributedSearchGraph Bucket, SearchableNode Node) GetOrCreateReshardCandidate(
        DistributedSearchGraph ownerCluster,
        DistributedSearchGraph inputBucket,
        SearchableNode inputNode,
        IRType targetType,
        SearchableNodeKind kind = SearchableNodeKind.Normal,
        DistributedReshardUsageKind? usageKind = null,
        bool isBidirect = false,
        DistributedSearchGraph? outputBucket = null,
        DistributedSearchGraph? dependencyBucket = null,
        SearchableNode? dependencyNode = null,
        bool addDataEdgeToOwnerCluster = false,
        Function? ownerFunction = null)
    {
        if ((dependencyBucket is null) != (dependencyNode is null))
        {
            throw new InvalidOperationException("A reshard candidate dependency must provide both bucket and node.");
        }

        var resolvedUsageKind = usageKind ?? GetReshardUsageKind(inputNode, kind);
        var key = new ReshardCandidateKey(
            ownerCluster,
            outputBucket,
            inputBucket,
            inputNode,
            targetType,
            resolvedUsageKind,
            kind,
            isBidirect,
            dependencyBucket,
            dependencyNode);
        if (_reshardCandidateMemo.TryGetValue(key, out var existing))
        {
            if (ownerFunction is not null)
            {
                RegisterNodeOwner(existing.Node, ownerFunction);
            }

            return existing;
        }

        var bucket = outputBucket ?? ownerCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        if (kind is SearchableNodeKind.FunctionBoundaryAdapter
            && EqualityComparer<IRType>.Default.Equals(inputNode.IRType, targetType))
        {
            var identityNode = new SearchableNode(
                inputNode.Expr,
                targetType,
                isBidirect,
                kind,
                inputNode.SourceKind,
                resolvedUsageKind,
                inputNode.OriginParameter);
            bucket.AddVertex(identityNode);
            var identityDataEdge = new CrossEdge(identityNode, inputNode, 0, inputBucket);
            if (addDataEdgeToOwnerCluster)
            {
                ownerCluster.AddEdge(identityDataEdge);
            }
            else
            {
                _rootSearchGraph.AddEdge(identityDataEdge);
            }

            if (dependencyBucket is not null && dependencyNode is not null)
            {
                _rootSearchGraph.AddEdge(new(identityNode, dependencyNode, HiddenFunctionDependencyIndex, dependencyBucket));
            }

            if (ownerFunction is not null)
            {
                RegisterNodeOwner(identityNode, ownerFunction);
            }

            var identity = (bucket, identityNode);
            _reshardCandidateMemo.Add(key, identity);
            _profiler.Count("function_boundary_identity_candidates");
            return identity;
        }

        var realization = ClassifyReshardRealization(
            inputNode.IRType,
            targetType,
            GetReshardSourceKind(inputNode),
            resolvedUsageKind);
        var op = realization switch
        {
            DistributedReshardRealization.Boxing => (Op)new Boxing(targetType),
            DistributedReshardRealization.ShardedView when targetType is DistributedType distributedType
                => new IR.Distributed.ShardedView(distributedType),
            DistributedReshardRealization.ShardedView
                => throw new InvalidOperationException(
                    $"ShardedView realization requires a DistributedType target, got {targetType}."),
            DistributedReshardRealization.Unsupported
                => throw new InvalidOperationException(
                    $"Target {TargetOptions.GetType().Name} cannot realize reshard {inputNode.IRType} -> {targetType} " +
                    $"from {GetReshardSourceKind(inputNode)} storage for {resolvedUsageKind} use."),
            _ => throw new InvalidOperationException($"Unknown distributed reshard realization {realization}."),
        };
        var outputSourceKind = realization == DistributedReshardRealization.Boxing
            ? DistributedReshardSourceKind.Internal
            : inputNode.SourceKind;
        var node = new SearchableNode(
            op,
            targetType,
            isBidirect,
            kind,
            outputSourceKind,
            resolvedUsageKind,
            inputNode.OriginParameter);
        bucket.AddVertex(node);
        var dataEdge = new CrossEdge(node, inputNode, 0, inputBucket);
        if (addDataEdgeToOwnerCluster)
        {
            ownerCluster.AddEdge(dataEdge);
        }
        else
        {
            _rootSearchGraph.AddEdge(dataEdge);
        }

        if (dependencyBucket is not null && dependencyNode is not null)
        {
            _rootSearchGraph.AddEdge(new(node, dependencyNode, HiddenFunctionDependencyIndex, dependencyBucket));
        }

        if (ownerFunction is not null)
        {
            RegisterNodeOwner(node, ownerFunction);
        }

        if (kind == SearchableNodeKind.Normal
            && resolvedUsageKind == DistributedReshardUsageKind.Internal
            && node.OriginParameter is { } originParameter)
        {
            RegisterFunctionParameterDerivedUseBucket(originParameter, targetType, bucket);
        }

        var created = (bucket, node);
        _reshardCandidateMemo.Add(key, created);
        _profiler.Count($"reshard_candidate_created:{realization}");
        return created;
    }

    private DistributedReshardRealization ClassifyReshardRealization(
        IRType sourceType,
        IRType targetType,
        DistributedReshardSourceKind sourceKind,
        DistributedReshardUsageKind usageKind)
        => _reshardRealizationPolicy.Classify(
            new DistributedReshardRealizationContext(
                TargetOptions,
                _moduleKind,
                sourceType,
                targetType,
                sourceKind,
                usageKind));

    private DistributedReshardSourceKind GetReshardSourceKind(SearchableNode inputNode)
        => inputNode.SourceKind;

    private DistributedReshardUsageKind GetReshardUsageKind(
        SearchableNode inputNode,
        SearchableNodeKind candidateKind)
        => candidateKind == SearchableNodeKind.FunctionBoundaryAdapter ||
            inputNode.Kind == SearchableNodeKind.FunctionBoundaryAdapter
            ? DistributedReshardUsageKind.FunctionBoundary
            : DistributedReshardUsageKind.Internal;

    private Unit VisitLeafFunctionCall(Call expr, Function callee)
    {
        _profiler.Count("function_calls");
        var calleeReturnCluster = _functionReturnClusters[callee];
        var actualClusters = new DistributedSearchGraph[expr.Arguments.Length];
        var formalClusters = GetFunctionParameterClusters(callee);
        var boundaryClusters = new DistributedSearchGraph[expr.Arguments.Length];
        var boundarySites = new FunctionBoundarySite?[expr.Arguments.Length];
        var calleeParameters = callee.Parameters.ToArray();
        if (calleeParameters.Length != expr.Arguments.Length)
        {
            throw new InvalidOperationException($"Function call argument count mismatch for {callee.Name}: expected {calleeParameters.Length}, got {expr.Arguments.Length}.");
        }

        for (int i = 0; i < expr.Arguments.Length; i++)
        {
            var parameter = calleeParameters[i];
            var actual = expr.Arguments[i];
            if (formalClusters.TryGetValue(parameter, out var formalCluster))
            {
                actualClusters[i] = VisitLeafArgument(ParameterKind.Input, actual, isSupported: true);
                boundarySites[i] = CreateFunctionBoundaryArgumentCluster(
                    expr,
                    callee,
                    parameter,
                    i,
                    actualClusters[i],
                    formalCluster);
                boundaryClusters[i] = boundarySites[i]!.BoundaryCluster;
            }
            else
            {
                actualClusters[i] = VisitLeafArgument(ParameterKind.Input, actual, isSupported: false);
                boundaryClusters[i] = actualClusters[i];
            }
        }

        var callCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        var function = _currentFunction
            ?? throw new InvalidOperationException("Function-call candidates must be created while building a function search graph.");
        var candidateSite = new FunctionCallCandidateSite(
            expr,
            callee,
            calleeReturnCluster,
            boundaryClusters,
            boundarySites,
            callCluster,
            function);
        _functionCallCandidateSites.Add(candidateSite);
        ExpandFunctionCallCandidateSite(candidateSite);
        RecordDirectValueBuckets(expr, candidateSite.DirectOutputBuckets);

        _inferedMemo.Add(expr, callCluster);
        FilterByScheme(expr, callCluster);
        return default;
    }

    private bool ExpandFunctionCallCandidateSite(FunctionCallCandidateSite site)
    {
        var newOutputBuckets = new List<DistributedSearchGraph>();
        foreach (var returnBucket in site.CalleeReturnCluster.Clusters.OfType<DistributedSearchGraph>())
        {
            if (!site.ReturnBuckets.Add(returnBucket))
            {
                continue;
            }

            var returnNode = returnBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException($"Function {site.Callee.Name} has an empty return candidate bucket.");
            var outputBucket = site.CallCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            site.DirectOutputBuckets.Add(outputBucket);
            newOutputBuckets.Add(outputBucket);
            var callNode = new SearchableNode(
                site.Callee,
                returnNode.IRType,
                kind: SearchableNodeKind.FunctionCall,
                metadataSource: site.SourceCall);
            outputBucket.AddVertex(callNode);
            RegisterNodeOwner(callNode, site.Function);
            _rootSearchGraph.AddEdge(new(
                callNode,
                returnNode,
                HiddenFunctionDependencyIndex,
                returnBucket));
            for (var index = 0; index < site.BoundaryClusters.Count; index++)
            {
                if (site.BoundarySites[index] is { } boundarySite)
                {
                    RegisterFunctionBoundaryCallConsumer(boundarySite, callNode, index);
                    continue;
                }

                foreach (var boundaryBucket in site.BoundaryClusters[index].Clusters.OfType<DistributedSearchGraph>())
                {
                    var boundaryNode = boundaryBucket.Vertices.FirstOrDefault()
                        ?? throw new InvalidOperationException(
                            $"Function {site.Callee.Name} call boundary argument {index} has an empty candidate bucket.");
                    _rootSearchGraph.AddEdge(new(callNode, boundaryNode, index, boundaryBucket));
                }
            }
        }

        if (newOutputBuckets.Count == 0)
        {
            return false;
        }

        AppendDirectValueBuckets(site.SourceCall, newOutputBuckets);
        if (site.SourceCall.CheckedType is TensorType tensorType && IsDistributableTensorType(tensorType))
        {
            CompleteOutputReshardClosure(
                site.CallCluster,
                tensorType,
                newOutputBuckets,
                DistributedReshardUsageKind.Internal,
                site.Function);
        }

        return true;
    }

    private FunctionBoundarySite CreateFunctionBoundaryArgumentCluster(
        Call call,
        Function callee,
        IVar parameter,
        int argumentIndex,
        DistributedSearchGraph actualCluster,
        DistributedSearchGraph formalCluster)
    {
        var boundaryCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        if (!_functionBoundarySites.TryGetValue(parameter, out var sites))
        {
            sites = new List<FunctionBoundarySite>();
            _functionBoundarySites.Add(parameter, sites);
        }

        if (parameter.CheckedType is not TensorType parameterTensorType)
        {
            throw new InvalidOperationException(
                $"Function call {callee.Name} distributed boundary parameter {argumentIndex} must be a TensorType, got {parameter.CheckedType}.");
        }

        var site = new FunctionBoundarySite(
            call,
            callee,
            parameter,
            argumentIndex,
            call.Arguments[argumentIndex],
            actualCluster,
            boundaryCluster);
        sites.Add(site);
        _allFunctionBoundarySites.Add(site);
        ExpandFunctionBoundarySite(site, formalCluster, parameterTensorType);

        if (boundaryCluster.VertexCount == 0)
        {
            throw new InvalidOperationException($"Function call {callee.Name} argument {argumentIndex} has no legal actual/formal distributed boundary plan.");
        }

        return site;
    }

    private bool ExpandFunctionBoundarySite(
        FunctionBoundarySite site,
        DistributedSearchGraph formalCluster,
        TensorType parameterTensorType)
    {
        var directBuckets = _directValueBuckets.TryGetValue(site.Actual, out var recordedDirectBuckets)
            ? recordedDirectBuckets
            : site.ActualCluster.Clusters
                .OfType<DistributedSearchGraph>()
                .Where(cluster => cluster.Kind is SearchGraphKind.Bucket)
                .ToArray();
        var changed = false;
        foreach (var directBucket in directBuckets)
        {
            if (!site.DirectActualBuckets.Add(directBucket))
            {
                continue;
            }

            changed = true;
            if (directBucket.Vertices.FirstOrDefault()?.IRType is DistributedType demandedType
                && EqualityComparer<TensorType>.Default.Equals(demandedType.TensorType, parameterTensorType))
            {
                EnsureFunctionParameterSignatureCandidate(
                    site.Callee,
                    site.Parameter,
                    formalCluster,
                    demandedType);
            }
        }

        foreach (var formalBucket in formalCluster.Clusters.OfType<DistributedSearchGraph>())
        {
            var formalNode = formalBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException(
                    $"Function {site.Callee.Name} parameter {site.Parameter.Name} has an empty signature bucket.");
            changed |= RegisterOperationReturnDemand(site.Actual, formalNode.IRType);
            AddFunctionBoundaryCandidates(
                site.SourceCall,
                site.ArgumentIndex,
                site,
                formalBucket);
        }

        return changed;
    }

    private void EnsureFunctionParameterSignatureCandidate(
        Function function,
        IVar parameter,
        DistributedSearchGraph formalCluster,
        DistributedType demandedType)
    {
        if (formalCluster.Clusters.OfType<DistributedSearchGraph>()
            .Any(bucket => EqualityComparer<IRType>.Default.Equals(bucket.Vertices.FirstOrDefault()?.IRType, demandedType)))
        {
            return;
        }

        if (!SingleNodeMemoryCheck(demandedType, _moduleKind, TargetOptions))
        {
            return;
        }

        var formalBucket = formalCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var formalNode = new SearchableNode(
            (BaseExpr)parameter,
            demandedType,
            kind: SearchableNodeKind.FunctionParameter);
        formalBucket.AddVertex(formalNode);
        RegisterFunctionParameterSignatureCandidate(
            function,
            parameter,
            formalBucket,
            formalNode);
        ConnectFunctionParameterSignatureToUses(function, parameter, formalBucket, formalNode);

        if (_functionBoundarySites.TryGetValue(parameter, out var sites))
        {
            foreach (var site in sites)
            {
                AddFunctionBoundaryCandidates(null, -1, site, formalBucket);
            }
        }

        _profiler.Count("function_parameter_demanded_signature_candidates");
    }

    private void ConnectFunctionParameterSignatureToUses(
        Function function,
        IVar parameter,
        DistributedSearchGraph formalBucket,
        SearchableNode formalNode)
    {
        var useClusters = GetFunctionParameterUseClusters(function);
        if (!useClusters.TryGetValue(parameter, out var useCluster))
        {
            throw new InvalidOperationException(
                $"Function {function.Name} parameter {parameter.Name} has a signature cluster but no use cluster.");
        }

        var parameterExpression = (BaseExpr)parameter;
        var directBuckets = _directValueBuckets.TryGetValue(parameterExpression, out var recordedDirectBuckets)
            ? recordedDirectBuckets
            : Array.Empty<DistributedSearchGraph>();
        var useBucket = directBuckets.FirstOrDefault(bucket =>
            EqualityComparer<IRType>.Default.Equals(bucket.Vertices.FirstOrDefault()?.IRType, formalNode.IRType));
        if (useBucket is null)
        {
            useBucket = useCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var useNode = new SearchableNode(
                (BaseExpr)parameter,
                formalNode.IRType,
                kind: SearchableNodeKind.TypeAdapter,
                sourceKind: DistributedReshardSourceKind.FunctionParameter);
            useBucket.AddVertex(useNode);
            _rootSearchGraph.AddEdge(new(useNode, formalNode, 0, formalBucket));
            RegisterNodeOwner(useNode, function);
            AppendDirectValueBuckets(parameterExpression, [useBucket]);
        }

        if (formalNode.IRType is DistributedType { TensorType: var tensorType })
        {
            CompleteOutputReshardClosure(
                useCluster,
                tensorType,
                [useBucket],
                DistributedReshardUsageKind.Internal,
                function);
        }
    }

    private void RegisterFunctionParameterSignatureCandidate(
        Function function,
        IVar parameter,
        DistributedSearchGraph formalBucket,
        SearchableNode formalNode)
    {
        if (_functionParameterOwners.TryGetValue(parameter, out var owner))
        {
            if (!ReferenceEquals(owner, function))
            {
                throw new InvalidOperationException(
                    $"Function parameter {parameter.Name} is owned by both {owner.Name} and {function.Name}.");
            }
        }
        else
        {
            _functionParameterOwners.Add(parameter, function);
        }

        if (!_functionParameterSignatureCandidates.TryGetValue(parameter, out var candidates))
        {
            candidates = new Dictionary<IRType, (DistributedSearchGraph, SearchableNode)>();
            _functionParameterSignatureCandidates.Add(parameter, candidates);
        }

        if (!candidates.TryAdd(formalNode.IRType, (formalBucket, formalNode)))
        {
            throw new InvalidOperationException(
                $"Function {function.Name} parameter {parameter.Name} has duplicate signature candidate {formalNode.IRType}.");
        }

        RegisterNodeOwner(formalNode, function);
        if (_functionParameterDerivedUseBuckets.TryGetValue(parameter, out var bucketsByType)
            && bucketsByType.TryGetValue(formalNode.IRType, out var derivedBuckets))
        {
            foreach (var derivedBucket in derivedBuckets)
            {
                ConnectFunctionParameterSignatureToDerivedUse(
                    function,
                    parameter,
                    formalBucket,
                    formalNode,
                    derivedBucket);
            }
        }
    }

    private void RegisterFunctionParameterDerivedUseBucket(
        IVar parameter,
        IRType type,
        DistributedSearchGraph derivedBucket)
    {
        if (!_functionParameterOwners.TryGetValue(parameter, out var function))
        {
            throw new InvalidOperationException(
                $"Derived use for function parameter {parameter.Name} was registered before its owner.");
        }

        if (!_functionParameterDerivedUseBuckets.TryGetValue(parameter, out var bucketsByType))
        {
            bucketsByType = new Dictionary<IRType, HashSet<DistributedSearchGraph>>();
            _functionParameterDerivedUseBuckets.Add(parameter, bucketsByType);
        }

        if (!bucketsByType.TryGetValue(type, out var buckets))
        {
            buckets = new HashSet<DistributedSearchGraph>(ReferenceEqualityComparer.Instance);
            bucketsByType.Add(type, buckets);
        }

        if (!buckets.Add(derivedBucket))
        {
            return;
        }

        if (_functionParameterSignatureCandidates.TryGetValue(parameter, out var candidates)
            && candidates.TryGetValue(type, out var signature))
        {
            ConnectFunctionParameterSignatureToDerivedUse(
                function,
                parameter,
                signature.Bucket,
                signature.Node,
                derivedBucket);
        }
    }

    private void ConnectFunctionParameterSignatureToDerivedUse(
        Function function,
        IVar parameter,
        DistributedSearchGraph formalBucket,
        SearchableNode formalNode,
        DistributedSearchGraph derivedBucket)
    {
        if (derivedBucket.Vertices.Any(candidate =>
            candidate.Kind == SearchableNodeKind.TypeAdapter
            && _rootSearchGraph.TryGetOutEdges(candidate, out var edges)
            && edges.Any(edge => ReferenceEquals(edge.Target, formalNode))))
        {
            return;
        }

        var adapter = new SearchableNode(
            (BaseExpr)parameter,
            formalNode.IRType,
            kind: SearchableNodeKind.TypeAdapter,
            sourceKind: DistributedReshardSourceKind.FunctionParameter,
            originParameter: parameter);
        derivedBucket.AddVertex(adapter);
        _rootSearchGraph.AddEdge(new(adapter, formalNode, 0, formalBucket));
        RegisterNodeOwner(adapter, function);
        _profiler.Count("function_parameter_equivalent_derived_use_candidates");
    }

    private void AddFunctionBoundaryCandidates(
        Call? call,
        int argumentIndex,
        FunctionBoundarySite site,
        DistributedSearchGraph formalBucket)
    {
        var callTargetName = call?.Target is Callable callable
            ? callable.Name
            : call?.Target.GetType().Name ?? "function";
        DistributedSearchGraph? pathCluster = null;
        var formalNode = formalBucket.Vertices.FirstOrDefault()
            ?? throw new InvalidOperationException($"Function call {callTargetName} formal argument {argumentIndex} has an empty candidate bucket.");
        foreach (var actualBucket in site.DirectActualBuckets)
        {
            var actualNode = actualBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException($"Function call {callTargetName} actual argument {argumentIndex} has an empty candidate bucket.");
            foreach (var plan in GetFunctionBoundaryReshardPlans(
                         actualNode.IRType,
                         formalNode.IRType,
                         GetReshardSourceKind(actualNode)))
            {
                var finalBucket = plan.StepTypes.Count == 0
                    ? actualBucket
                    : AddFunctionBoundaryReshardPath(
                        pathCluster ??= _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster),
                        actualBucket,
                        actualNode,
                        plan.StepTypes);
                var finalNode = finalBucket.Vertices.First();
                var (adapterBucket, adapterNode) = GetOrCreateReshardCandidate(
                    site.BoundaryCluster,
                    finalBucket,
                    finalNode,
                    formalNode.IRType,
                    kind: SearchableNodeKind.FunctionBoundaryAdapter,
                    dependencyBucket: formalBucket,
                    dependencyNode: formalNode);
                ConnectFunctionBoundaryCandidate(site, adapterBucket, adapterNode);
            }
        }
    }

    private void RegisterFunctionBoundaryCallConsumer(
        FunctionBoundarySite site,
        SearchableNode callNode,
        int inputIndex)
    {
        site.CallConsumers.Add((callNode, inputIndex));
        foreach (var boundaryBucket in site.BoundaryCluster.Clusters.OfType<DistributedSearchGraph>())
        {
            var boundaryNode = boundaryBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException("Function call boundary has an empty candidate bucket.");
            ConnectFunctionBoundaryCandidate(site, boundaryBucket, boundaryNode, callNode, inputIndex);
        }
    }

    private void ConnectFunctionBoundaryCandidate(
        FunctionBoundarySite site,
        DistributedSearchGraph boundaryBucket,
        SearchableNode boundaryNode)
    {
        foreach (var (callNode, inputIndex) in site.CallConsumers)
        {
            ConnectFunctionBoundaryCandidate(site, boundaryBucket, boundaryNode, callNode, inputIndex);
        }
    }

    private void ConnectFunctionBoundaryCandidate(
        FunctionBoundarySite site,
        DistributedSearchGraph boundaryBucket,
        SearchableNode boundaryNode,
        SearchableNode callNode,
        int inputIndex)
    {
        if (_rootSearchGraph.TryGetOutEdges(callNode, out var edges)
            && edges.Any(edge => edge.InputIndex == inputIndex
                && ReferenceEquals(edge.InputGraph, boundaryBucket)
                && ReferenceEquals(edge.Target, boundaryNode)))
        {
            return;
        }

        if (!site.CallConsumers.Any(consumer => ReferenceEquals(consumer.CallNode, callNode)
            && consumer.InputIndex == inputIndex))
        {
            throw new InvalidOperationException("Function boundary candidate is not owned by the call consumer being connected.");
        }

        _rootSearchGraph.AddEdge(new(callNode, boundaryNode, inputIndex, boundaryBucket));
    }

    private IEnumerable<DistributedReshardPlan> GetFunctionBoundaryReshardPlans(
        IRType sourceType,
        IRType targetType,
        DistributedReshardSourceKind sourceKind)
    {
        if (EqualityComparer<IRType>.Default.Equals(sourceType, targetType))
        {
            return new[] { new DistributedReshardPlan(Array.Empty<IRType>()) };
        }

        return GetReshardPlans(
            sourceType,
            targetType,
            sourceKind,
            DistributedReshardUsageKind.FunctionBoundary);
    }

    private DistributedSearchGraph AddFunctionBoundaryReshardPath(
        DistributedSearchGraph boundaryCluster,
        DistributedSearchGraph sourceBucket,
        SearchableNode sourceNode,
        IReadOnlyList<IRType> stepTypes)
    {
        var inputBucket = sourceBucket;
        var inputNode = sourceNode;
        for (var index = 0; index < stepTypes.Count; index++)
        {
            var stepUsageKind = index == stepTypes.Count - 1
                ? DistributedReshardUsageKind.FunctionBoundary
                : DistributedReshardUsageKind.Internal;
            var (bucket, node) = GetOrCreateReshardCandidate(
                boundaryCluster,
                inputBucket,
                inputNode,
                stepTypes[index],
                usageKind: stepUsageKind);
            inputBucket = bucket;
            inputNode = node;
        }

        return inputBucket;
    }

    /// <summary>
    /// some times we didn't use all args.
    /// </summary>
    private IEnumerable<(Expr Call, bool[] Used)> BuildEquivalentCalls(Expr target, BaseExpr[] tempArgs)
    {
        IEnumerable<(Expr Call, bool[] Used)> calls = [(new Call(target, tempArgs), Enumerable.Repeat(true, tempArgs.Length).ToArray())];
        if (target is Boxing { NewType: TensorType } && tempArgs[0] is TensorConst tc && tc.ValueType is DistributedType distributedType)
        {
            calls = [((Expr)tc, new[] { true })];
        }
        else if (target is GetPositionIds)
        {
            var tensorType = (TensorType)calls.First().Call.CheckedType;
            calls = calls.Where(call => call.Call.CheckedType is DistributedType).Concat(GetLeafCandidateDistTypes(tensorType)
                .Select(dt => ((Expr)IR.F.NN.GetPositionIds((Dimension)tempArgs[0], (Expr)tempArgs[1], dt.AxisPolicies, dt.Placement), new[] { true, true })));
        }

        return calls;
    }

    private IReadOnlyList<IRArray<SBP>> GetDiverseCandidateSBPs(DistributedType distributedType, IEnumerable<Placement> placements)
    {
        return placements.Select(
            placement =>
                DistributedUtility.GetLeafCandidatePolicies(distributedType.TensorType, placement, TargetOptions.DistributedSplitCandidateProvider).
                Where(p => SingleNodeMemoryCheck(new(distributedType.TensorType, p, placement), _moduleKind, TargetOptions)).
                Where(ndsbp => ndsbp != distributedType.AxisPolicies)).
            SelectMany(e => e).ToArray();
    }

    private DistributedSearchGraph VisitLeafArgument(ParameterKind parameterKind, BaseExpr expr, bool isSupported)
    {
        DistributedSearchGraph argCluster;
        switch (parameterKind, expr)
        {
            case (_, None e):
                argCluster = TryInstertTerminator(e);
                break;
            case (ParameterKind.Input, BaseExpr e):
                if (isSupported)
                {
                    argCluster = TryAddOriginator(e);
                }
                else
                {
                    argCluster = TryInstertTerminator(e);
                }

                break;
            case (ParameterKind.Attribute, BaseExpr e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Dimension e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Shape e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Padding e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Paddings e):
                argCluster = TryInstertTerminator(e);
                break;
            default:
                throw new InvalidOperationException();
        }

        FilterByScheme(expr, argCluster);
        return argCluster ?? throw new InvalidOperationException("the argument cluster can't be null.");
    }

    private bool IsDistributed(IRType type) => type switch
    {
        DistributedType => true,
        TupleType t => t.All(IsDistributed),
        _ => false,
    };

    private Dictionary<IVar, DistributedSearchGraph> GetFunctionParameterClusters(Function function)
    {
        if (!_functionParameterClusters.TryGetValue(function, out var clusters))
        {
            clusters = new Dictionary<IVar, DistributedSearchGraph>(ReferenceEqualityComparer.Instance);
            _functionParameterClusters.Add(function, clusters);
        }

        return clusters;
    }

    private Dictionary<IVar, DistributedSearchGraph> GetFunctionParameterUseClusters(Function function)
    {
        if (!_functionParameterUseClusters.TryGetValue(function, out var clusters))
        {
            clusters = new Dictionary<IVar, DistributedSearchGraph>(ReferenceEqualityComparer.Instance);
            _functionParameterUseClusters.Add(function, clusters);
        }

        return clusters;
    }

    private bool TryGetCurrentInternalTensorParameter(Var var, [NotNullWhen(true)] out Function? function)
    {
        function = null;
        if (_currentFunction is null || _currentFunctionIsEntry || var.CheckedType is not TensorType tensorType || !IsDistributableTensorType(tensorType))
        {
            return false;
        }

        foreach (var parameter in _currentFunction.Parameters)
        {
            if (ReferenceEquals(parameter, var))
            {
                function = _currentFunction;
                return true;
            }
        }

        return false;
    }

    private DistributedSearchGraph CreateFunctionParameterCluster(Function function, Var parameter)
    {
        var clusters = GetFunctionParameterClusters(function);
        if (clusters.TryGetValue(parameter, out var existing))
        {
            return existing;
        }

        if (parameter.CheckedType is not TensorType tensorType || !IsDistributableTensorType(tensorType))
        {
            throw new InvalidOperationException($"AutoDistributed function parameter signature only supports distributable tensor parameters, got {parameter.CheckedType}.");
        }

        var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        var tensorBucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var tensorNode = new SearchableNode(parameter, tensorType, kind: SearchableNodeKind.FunctionParameter);
        tensorBucket.AddVertex(tensorNode);
        RegisterFunctionParameterSignatureCandidate(function, parameter, tensorBucket, tensorNode);
        foreach (var dType in GetLeafCandidateDistTypes(tensorType))
        {
            var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var node = new SearchableNode(parameter, dType, kind: SearchableNodeKind.FunctionParameter);
            bucket.AddVertex(node);
            RegisterFunctionParameterSignatureCandidate(function, parameter, bucket, node);
        }

        clusters.Add(parameter, distCluster);
        _singleChoiceClusters.Add(distCluster);
        return distCluster;
    }

    private DistributedSearchGraph CreateFunctionParameterUseCluster(
        Function function,
        Var parameter,
        DistributedSearchGraph signatureCluster)
    {
        var clusters = GetFunctionParameterUseClusters(function);
        if (clusters.TryGetValue(parameter, out var existing))
        {
            return existing;
        }

        if (parameter.CheckedType is not TensorType tensorType || !IsDistributableTensorType(tensorType))
        {
            throw new InvalidOperationException(
                $"AutoDistributed function parameter uses only support distributable tensor parameters, got {parameter.CheckedType}.");
        }

        var useCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        var directUseBuckets = new List<DistributedSearchGraph>();
        foreach (var signatureBucket in signatureCluster.Clusters.OfType<DistributedSearchGraph>())
        {
            var signatureNode = signatureBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException(
                    $"Function {function.Name} parameter {parameter.Name} has an empty signature bucket.");
            if (signatureNode.IRType is not DistributedType)
            {
                continue;
            }

            var useBucket = useCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var useNode = new SearchableNode(
                parameter,
                signatureNode.IRType,
                kind: SearchableNodeKind.TypeAdapter,
                sourceKind: DistributedReshardSourceKind.FunctionParameter);
            useBucket.AddVertex(useNode);
            _rootSearchGraph.AddEdge(new(useNode, signatureNode, 0, signatureBucket));
            directUseBuckets.Add(useBucket);
        }

        if (directUseBuckets.Count == 0)
        {
            throw new InvalidOperationException(
                $"Function {function.Name} parameter {parameter.Name} has no distributed signature candidates.");
        }

        CompleteOutputReshardClosure(
            useCluster,
            tensorType,
            directUseBuckets,
            DistributedReshardUsageKind.Internal,
            function);
        RecordDirectValueBuckets(parameter, directUseBuckets);
        clusters.Add(parameter, useCluster);
        return useCluster;
    }

    private void RecordDirectValueBuckets(BaseExpr expression, IReadOnlyList<DistributedSearchGraph> buckets)
    {
        if (buckets.Count == 0)
        {
            throw new InvalidOperationException(
                $"Cannot record direct producer candidates for {expression.GetType().Name}: the bucket list is empty.");
        }

        if (_directValueBuckets.TryGetValue(expression, out var existing))
        {
            if (existing.Count != buckets.Count ||
                existing.Where((bucket, index) => !ReferenceEquals(bucket, buckets[index])).Any())
            {
                throw new InvalidOperationException(
                    $"Direct producer candidates for {expression.GetType().Name} were recorded inconsistently.");
            }

            return;
        }

        _directValueBuckets.Add(expression, buckets.ToArray());
    }

    private void AppendDirectValueBuckets(
        BaseExpr expression,
        IReadOnlyList<DistributedSearchGraph> buckets)
    {
        if (buckets.Count == 0 || !_directValueBuckets.TryGetValue(expression, out var existing))
        {
            return;
        }

        var additions = buckets
            .Where(bucket => !existing.Any(current => ReferenceEquals(current, bucket)))
            .ToArray();
        if (additions.Length == 0)
        {
            return;
        }

        _directValueBuckets[expression] = existing.Concat(additions).ToArray();
    }

    private DistributedSearchGraph CreateOriginatorCluster(BaseExpr expr, bool init)
    {
        if (expr is IR.Tuple tp)
        {
            var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var buckets = new List<DistributedSearchGraph>[tp.Fields.Length];
            foreach (var (f, fGraph, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, Visit(f), i)))
            {
                buckets[i] = TryAddOriginator(f).Clusters.OfType<DistributedSearchGraph>().ToList();
            }

            var combBuckets = buckets.CartesianProduct();
            foreach (var comb in combBuckets)
            {
                var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(comb.Select(g => g.Vertices.First().IRType).ToArray()));
                var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucket.AddVertex(tpnode);
                for (int i = 0; i < tp.Fields.Length; i++)
                {
                    _rootSearchGraph.AddEdge(new(tpnode, comb.ElementAt(i).Vertices.First(), i, comb.ElementAt(i)));
                }
            }

            return distCluster;
        }
        else if (expr is Call { Target: Boxing { NewType: TensorType } } call && call[Boxing.Input] is TensorConst tc && tc.ValueType is DistributedType distributedType)
        {
            var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var dnode = new SearchableNode(tc, distributedType);
            bucket.AddVertex(dnode);

            return distCluster;
        }
        else if (expr is TensorConst tc2)
        {
            if (tc2.ValueType is TensorType tensorType && IsDistributableTensorType(tensorType))
            {
                var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
                foreach (var dType in GetLeafCandidateDistTypes(tensorType))
                {
                    var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    var realization = _reshardRealizationPolicy.Classify(
                        new DistributedReshardRealizationContext(
                            TargetOptions,
                            _moduleKind,
                            tensorType,
                            dType,
                            DistributedReshardSourceKind.Constant,
                            DistributedReshardUsageKind.Internal));
                    switch (realization)
                    {
                        case DistributedReshardRealization.Boxing:
                            var distConst = new TensorConst(tc2.Value, dType.AxisPolicies, dType.Placement);
                            if (_phase == AutoDistributedPhase.SearchConstant)
                            {
                                _distributedConstSources.Add(distConst, tc2);
                            }

                            bucket.AddVertex(new SearchableNode(distConst, dType));
                            break;
                        case DistributedReshardRealization.ShardedView:
                            var shardedViewNode = new SearchableNode(new IR.Distributed.ShardedView(dType), dType);
                            bucket.AddVertex(shardedViewNode);
                            var shardedViewInputBucket = GetOrCreateConstantStorageBucket(tc2);
                            _rootSearchGraph.AddEdge(new(shardedViewNode, shardedViewInputBucket.Vertices.First(), 0, shardedViewInputBucket));
                            break;
                        case DistributedReshardRealization.Unsupported:
                            throw new InvalidOperationException(
                                $"Target {TargetOptions.GetType().Name} cannot realize constant reshard {tensorType} -> {dType}.");
                        default:
                            throw new InvalidOperationException($"Unknown distributed reshard realization {realization}.");
                    }

                    _profiler.Count($"constant_reshard_realization:{realization}");
                }

                return distCluster;
            }
            else if (tc2.ValueType is TensorType)
            {
                var standCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);
                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucket.AddVertex(new SearchableNode(tc2, tc2.CheckedType));
                return standCluster;
            }
            else if (tc2.ValueType is DistributedType distributedType2)
            {
                var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
                var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var dnode = new SearchableNode(tc2, distributedType2);
                bucket.AddVertex(dnode);

                return distCluster;
            }
            else
            {
                throw new InvalidOperationException($"Unsupported TensorConst type: {tc2.ValueType}");
            }
        }
        else
        {
            if (init && expr is Var var && TryGetCurrentInternalTensorParameter(var, out var function))
            {
                return CreateFunctionParameterCluster(function, var);
            }

            if (init)
            {
                var standCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);
                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var node = new SearchableNode(expr, expr.CheckedType);
                bucket.AddVertex(node);
                return standCluster;
            }
            else
            {
                if (expr.CheckedType is TupleType)
                {
                    return CreateTuplePassThroughOriginatorCluster(expr);
                }

                var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
                var inferCluster = _inferedMemo[expr];
                var sourceType = inferCluster.Vertices.First().IRType;
                if (sourceType is TensorType sourceTensorType && !IsDistributableTensorType(sourceTensorType))
                {
                    return inferCluster;
                }

                if (sourceType is not TensorType tensorType)
                {
                    throw new InvalidOperationException($"AutoDistributed can only create tensor originator candidates from TensorType, but got {sourceType} for {expr.GetType().Name}.");
                }

                foreach (var dType in GetLeafCandidateDistTypes(tensorType))
                {
                    var inputBucket = inferCluster.Clusters.OfType<DistributedSearchGraph>().First();
                    var inputNode = inputBucket.Vertices.First();
                    GetOrCreateReshardCandidate(
                        distCluster,
                        inputBucket,
                        inputNode,
                        dType,
                        usageKind: DistributedReshardUsageKind.Internal);
                }

                return distCluster;
            }
        }
    }

    private DistributedSearchGraph CreateTuplePassThroughOriginatorCluster(BaseExpr expr)
    {
        if (!_inferedMemo.TryGetValue(expr, out var inferCluster))
        {
            throw new InvalidOperationException($"Tuple originator {expr.GetType().Name} must be inferred before resharding.");
        }

        var sourceBuckets = inferCluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
        if (sourceBuckets.Length == 0)
        {
            throw new InvalidOperationException($"Tuple originator {expr.GetType().Name} has no inferred candidate buckets.");
        }

        var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        foreach (var sourceBucket in sourceBuckets)
        {
            var sourceNode = sourceBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException($"Tuple originator {expr.GetType().Name} has an empty inferred bucket.");
            if (sourceNode.IRType is not TupleType)
            {
                throw new InvalidOperationException($"Tuple originator {expr.GetType().Name} expected TupleType source, but got {sourceNode.IRType}.");
            }

            var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var node = new SearchableNode(expr, sourceNode.IRType);
            bucket.AddVertex(node);
            _rootSearchGraph.AddEdge(new(node, sourceNode, 0, sourceBucket));
        }

        return distCluster;
    }

    private DistributedSearchGraph GetOrCreateConstantStorageBucket(TensorConst source)
    {
        if (_constantStorageBuckets.TryGetValue(source, out var existing))
        {
            return existing;
        }

        var sourceCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);
        var sourceBucket = sourceCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        sourceBucket.AddVertex(new SearchableNode(source, source.CheckedType));
        _constantStorageBuckets.Add(source, sourceBucket);
        return sourceBucket;
    }

    private DistributedSearchGraph TryAddOriginator(BaseExpr expr)
    {
        if (!_inferedMemo.TryGetValue(expr, out var inferCluster))
        {
            inferCluster = CreateOriginatorCluster(expr, true);
            _inferedMemo.Add(expr, inferCluster);
        }

        if (inferCluster.Kind is SearchGraphKind.DistributedCluster)
        {
            if (expr is Var parameter &&
                TryGetCurrentInternalTensorParameter(parameter, out var function))
            {
                return CreateFunctionParameterUseCluster(function, parameter, inferCluster);
            }

            return inferCluster;
        }

        if (!ContainsDistributableTensorType(expr.CheckedType))
        {
            return inferCluster;
        }

        // unshard to standalone
        if (!_reshardMemo.TryGetValue(expr, out var distCluster))
        {
            distCluster = CreateOriginatorCluster(expr, false);
            _reshardMemo.Add(expr, distCluster);
        }

        if (distCluster.Kind != SearchGraphKind.DistributedCluster)
        {
            throw new InvalidOperationException("The inference and reshard cluster cannot be distributed either.");
        }

        return distCluster;
    }

    private DistributedSearchGraph CreateTerminatorCluster(BaseExpr expr, bool init)
    {
        var standCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);

        if (expr is IR.Tuple tp)
        {
            var fieldClusters = new DistributedSearchGraph[tp.Fields.Length];
            foreach (var (field, index) in tp.Fields.AsValueEnumerable().Select((field, index) => (field, index)))
            {
                Visit(field);
                fieldClusters[index] = TryInstertTerminator(field);
            }

            AddTupleTerminatorCandidates(standCluster, fieldClusters);
        }
        else if (expr.CheckedType is TupleType tupleType)
        {
            if (init)
            {
                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var node = new SearchableNode(expr, expr.CheckedType);
                bucket.AddVertex(node);
            }
            else
            {
                var fieldClusters = new DistributedSearchGraph[tupleType.Fields.Count];
                for (int i = 0; i < tupleType.Fields.Count; i++)
                {
                    var field = IR.F.Tensors.GetItem(expr, i);
                    Visit(field);
                    fieldClusters[i] = TryInstertTerminator(field);
                }

                AddTupleTerminatorCandidates(standCluster, fieldClusters);
            }
        }
        else if (expr is TensorConst tc && tc.ValueType is TensorType tensorType)
        {
            var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var node = new SearchableNode(expr, expr.CheckedType);
            bucket.AddVertex(node);
        }
        else if (expr is Shape or Padding or Paddings or Dimension or None)
        {
            var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var node = new SearchableNode(expr, expr.CheckedType);
            bucket.AddVertex(node);
        }
        else
        {
            if (init)
            {
                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var node = new SearchableNode(expr, expr.CheckedType);
                bucket.AddVertex(node);
            }
            else
            {
                if (!ContainsDistributableTensorType(expr.CheckedType))
                {
                    var passthroughInputBuckets = _inferedMemo[expr].Clusters.OfType<DistributedSearchGraph>().ToArray();
                    var passthroughBucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    var passthroughNode = new SearchableNode(expr, expr.CheckedType, kind: SearchableNodeKind.TypeAdapter);
                    passthroughBucket.AddVertex(passthroughNode);
                    foreach (var inputBucket in passthroughInputBuckets)
                    {
                        var inputNode = inputBucket.Vertices.FirstOrDefault();
                        if (inputNode is not null && EqualityComparer<IRType>.Default.Equals(inputNode.IRType, passthroughNode.IRType))
                        {
                            _rootSearchGraph.AddEdge(new(passthroughNode, inputNode, 0, inputBucket));
                        }
                    }

                    if (!_rootSearchGraph.TryGetOutEdges(passthroughNode, out var edges) || !edges.Any())
                    {
                        throw new InvalidOperationException($"AutoDistributed cannot create standalone passthrough for non-distributable tensor {expr.CheckedType}.");
                    }

                    return standCluster;
                }

                var onode = new SearchableNode(
                    new Boxing(expr.CheckedType),
                    expr.CheckedType,
                    reshardUsageKind: DistributedReshardUsageKind.ProgramOutput);
                var inputBuckets = _inferedMemo[expr].Clusters.OfType<DistributedSearchGraph>().ToArray();

                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucket.AddVertex(onode);
                foreach (var inputBucket in inputBuckets)
                {
                    if (inputBucket.Vertices.Any() && CheckBoxingTypeCached(inputBucket.Vertices.First().IRType, onode.IRType) is not InvalidType)
                    {
                        _rootSearchGraph.AddEdge(new(onode, inputBucket.Vertices.First(), 0, inputBucket));
                    }
                }

                AddProgramOutputShardedViewCandidates(standCluster, inputBuckets);
            }
        }

        return standCluster;
    }

    private void AddTupleTerminatorCandidates(
        DistributedSearchGraph terminatorCluster,
        IReadOnlyList<DistributedSearchGraph> fieldClusters)
    {
        var fieldBuckets = fieldClusters
            .Select((cluster, fieldIndex) =>
            {
                var buckets = cluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
                if (buckets.Length == 0)
                {
                    throw new InvalidOperationException(
                        $"Tuple terminator field {fieldIndex} has no output candidate buckets.");
                }

                return buckets;
            })
            .ToArray();
        var selectedBuckets = new DistributedSearchGraph[fieldBuckets.Length];
        AddCombinations(0);

        void AddCombinations(int fieldIndex)
        {
            if (fieldIndex < fieldBuckets.Length)
            {
                foreach (var fieldBucket in fieldBuckets[fieldIndex])
                {
                    selectedBuckets[fieldIndex] = fieldBucket;
                    AddCombinations(fieldIndex + 1);
                }

                return;
            }

            var tupleNode = new SearchableNode(
                new IR.Tuple(),
                new TupleType(selectedBuckets.Select(GetBucketType).ToArray()));
            var tupleBucket = terminatorCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            tupleBucket.AddVertex(tupleNode);
            for (var index = 0; index < selectedBuckets.Length; index++)
            {
                var fieldBucket = selectedBuckets[index];
                _rootSearchGraph.AddEdge(new(tupleNode, fieldBucket.Vertices.First(), index, fieldBucket));
            }
        }

        static IRType GetBucketType(DistributedSearchGraph bucket)
            => bucket.Vertices.FirstOrDefault()?.IRType ??
                throw new InvalidOperationException("Tuple terminator candidate bucket is empty.");
    }

    private void AddProgramOutputShardedViewCandidates(
        DistributedSearchGraph terminatorCluster,
        IReadOnlyList<DistributedSearchGraph> inputBuckets)
    {
        var outputBuckets = new Dictionary<DistributedType, DistributedSearchGraph>();
        foreach (var inputBucket in inputBuckets)
        {
            var inputNode = inputBucket.Vertices.FirstOrDefault();
            if (inputNode?.IRType is not DistributedType sourceType)
            {
                continue;
            }

            var outputType = new DistributedType(
                sourceType.TensorType,
                Enumerable.Repeat<SBP>(SBP.B, sourceType.AxisPolicies.Count).ToArray(),
                sourceType.Placement);
            if (ClassifyReshardRealization(
                    sourceType,
                    outputType,
                    GetReshardSourceKind(inputNode),
                    DistributedReshardUsageKind.ProgramOutput) != DistributedReshardRealization.ShardedView)
            {
                continue;
            }

            if (!outputBuckets.TryGetValue(outputType, out var outputBucket))
            {
                outputBucket = terminatorCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                outputBuckets.Add(outputType, outputBucket);
            }

            GetOrCreateReshardCandidate(
                terminatorCluster,
                inputBucket,
                inputNode,
                outputType,
                usageKind: DistributedReshardUsageKind.ProgramOutput,
                outputBucket: outputBucket);
        }
    }

    private IRType CheckBoxingType(IRType inType, IRType outType, bool isReshape = false)
    {
        IRType VisitD2D(DistributedType inv, DistributedType outv)
        {
            if (inv.TensorType != outv.TensorType)
            {
                return new InvalidType($"D2D boxing requires the same tensor type, but got {inv.TensorType} -> {outv.TensorType}");
            }

            if (inv.Placement != outv.Placement)
            {
                return new InvalidType($"D2D boxing requires the same placement, but got {inv.Placement} -> {outv.Placement}");
            }

            if (inv.Partial == outv.Partial && DistributedUtility.AreSamePolicies(inv.AxisPolicies, outv.AxisPolicies))
            {
                return new InvalidType("Same DistributedType");
            }

            if (inv.AxisPolicies.Any(sbp => sbp is SBPPartial) || outv.AxisPolicies.Any(sbp => sbp is SBPPartial))
            {
                return new InvalidType("Not Support Partial in Policeis.");
            }

            var partialDims = new List<int>();
            if (inv.Partial is not null)
            {
                for (int i = 0; i < inv.AxisPolicies.Count; i++)
                {
                    if (inv.AxisPolicies[i] is SBPSplit && outv.AxisPolicies[i] is SBPBroadCast)
                    {
                        return new InvalidType("Not supported input is BroadCast output is Split");
                    }

                    if (outv.AxisPolicies[i] is SBPSplit s)
                    {
                        if (inv.AxisPolicies[i] is SBPSplit splitIn)
                        {
                            if (!DistributedUtility.IsSamePolicy(splitIn, s, checkGranularity: false))
                            {
                                return new InvalidType("Not Supported Split-> Split.");
                            }
                        }

                        if (s.HierarchyAxes.Any(inv.Partial.Axes.Contains))
                        {
                            partialDims.Add(i);
                        }
                    }
                }

                var ndspsIn = DistributedUtility.GetHierarchyAxisPolicies(inv.AxisPolicies, inv.Placement.Rank);
                var ndspsOut = DistributedUtility.GetHierarchyAxisPolicies(outv.AxisPolicies, outv.Placement.Rank);
                if (Enumerable.Range(0, ndspsIn.Count).Any(i =>
                    ndspsIn[i] is HierarchyAxisSplit splitIn &&
                    (ndspsOut[i] is HierarchyAxisBroadcast ||
                     (ndspsOut[i] is HierarchyAxisSplit splitOut && splitOut != splitIn))))
                {
                    return new InvalidType("Not Supported Split-> Broadcast.");
                }
            }

            if (partialDims.Count > 0 && !Enumerable.Range(0, inv.AxisPolicies.Count).Except(partialDims.ToArray()).All(i => DistributedUtility.IsSamePolicy(inv.AxisPolicies[i], outv.AxisPolicies[i])))
            {
                return new InvalidType("Not Supported Partial.");
            }

            return outv;
        }

        IRType VisitD2T(DistributedType inv, TensorType outv)
        {
            if (inv.AxisPolicies.Any(s => s is SBPPartial) || inv.Partial is not null)
            {
                return new InvalidType("Not supported input is Partial output is Unshard");
            }

            return outv;
        }

        IRType VisitT2D(TensorType inv, DistributedType outv)
        {
            if (outv.AxisPolicies.Any(s => s is SBPPartial) || outv.Partial is not null)
            {
                return new InvalidType("Not supported input is Unshard output is Partial");
            }

            return outv;
        }

        IRType VisitTuple(TupleType inv, TupleType outv)
        {
            if (inv.Count != outv.Count)
            {
                return new InvalidType($"Tuple boxing field count mismatch: {inv.Count} -> {outv.Count}");
            }

            var changed = false;
            for (int i = 0; i < inv.Count; i++)
            {
                if (EqualityComparer<IRType>.Default.Equals(inv[i], outv[i]))
                {
                    continue;
                }

                var fieldResult = CheckBoxingTypeCached(inv[i], outv[i], isReshape);
                if (fieldResult is InvalidType invalidType)
                {
                    return new InvalidType($"Tuple boxing field {i} is invalid: {invalidType.Reason}");
                }

                changed = true;
            }

            return changed ? outv : new InvalidType("Same TupleType");
        }

        return (inType, outType) switch
        {
            (InvalidType inv, _) => inv,
            (_, InvalidType inv) => inv,
            (TupleType inv, TupleType outv) => VisitTuple(inv, outv),
            (DistributedType d, DistributedType d1) => VisitD2D(d, d1),
            (TensorType t, DistributedType d) => VisitT2D(t, d),
            (DistributedType d, TensorType t) => VisitD2T(d, t),
            _ => new InvalidType($"not support boxing {inType} to {outType}"),
        };
    }

    private DistributedSearchGraph TryInstertTerminator(BaseExpr expr)
    {
        if (!_inferedMemo.TryGetValue(expr, out var inferCluster))
        {
            inferCluster = CreateTerminatorCluster(expr, true);
            _inferedMemo.Add(expr, inferCluster);
            return inferCluster;
        }

        if (inferCluster.Kind is SearchGraphKind.StandaloneCluster)
        {
            return inferCluster;
        }

        // unshard to standalone
        if (!_reshardMemo.TryGetValue(expr, out var standCluster))
        {
            standCluster = CreateTerminatorCluster(expr, false);
            _reshardMemo.Add(expr, standCluster);
            return standCluster;
        }

        if (standCluster.Kind != SearchGraphKind.StandaloneCluster)
        {
            throw new InvalidOperationException("The inference and reshard cluster cannot be distributed either.");
        }

        return standCluster;
    }

    private void Dump(Stream stream, IReadOnlyDictionary<SearchableNode, bool> pickMemo, IReadOnlyDictionary<SearchableNode, CostModel.Cost> costMemo, IReadOnlyDictionary<SearchableNode, UInt128> costScoreMemo)
    {
        using var writer = new StreamWriter(stream);
        writer.Write(_rootSearchGraph.ToGraphviz(alg =>
        {
            alg.GraphFormat.RankDirection = QuikGraph.Graphviz.Dot.GraphvizRankDirection.LR;
            alg.FormatCluster += (_, arg) =>
            {
                if (arg.Cluster is DistributedSearchGraph tg)
                {
                    arg.GraphFormat.LabelLocation = QuikGraph.Graphviz.Dot.GraphvizLabelLocation.T;
                    arg.GraphFormat.LabelJustification = QuikGraph.Graphviz.Dot.GraphvizLabelJustification.L;
                    arg.GraphFormat.Label = tg.Kind.ToString();
                    if (tg.Kind is SearchGraphKind.Bucket && tg.Vertices.Any())
                    {
                        arg.GraphFormat.Label += ": " + tg.Vertices.First().IRType.ToString();
                    }
                }
            };

            alg.FormatVertex += (_, arg) =>
            {
                var row0 = new QuikGraph.Graphviz.Dot.GraphvizRecordCell();
                var col1 = new QuikGraph.Graphviz.Dot.GraphvizRecordCell();
                row0.Cells.Add(col1);

                col1.Cells.Add(new() { Text = arg.Vertex.Expr.GetType().ToString() });
                if (arg.Vertex.Expr is IR.Tuple && arg.Vertex.IRType is TupleType tpTuple)
                {
                    for (int i = 0; i < tpTuple.Fields.Count; i++)
                    {
                        col1.Cells.Add(new() { Text = i.ToString(), Port = $"P{i}" });
                    }
                }
                else if (arg.Vertex.Expr is Op op)
                {
                    for (int i = 0; i < op.Parameters.Count; i++)
                    {
                        col1.Cells.Add(new() { Text = i.ToString(), Port = $"P{i}" });
                    }
                }

                arg.VertexFormat.Record.Cells.Add(row0);
                arg.VertexFormat.Shape = QuikGraph.Graphviz.Dot.GraphvizVertexShape.Record;
                arg.VertexFormat.Style = QuikGraph.Graphviz.Dot.GraphvizVertexStyle.Filled;
                if (costMemo.TryGetValue(arg.Vertex, out var cost))
                {
                    var row1 = new QuikGraph.Graphviz.Dot.GraphvizRecordCell();
                    foreach (var (k, v) in cost.Factors)
                    {
                        row1.Cells.Add(new() { Text = $"{k}: {v}" });
                    }

                    row1.Cells.Add(new() { Text = $"Local score: {costScoreMemo[arg.Vertex]}" });
                    row1.Cells.Add(new() { Text = $"Invocation count: {GetInvocationCount(arg.Vertex)}" });
                    row1.Cells.Add(new() { Text = $"Objective score: {GetObjectiveScore(costScoreMemo, arg.Vertex)}" });
                    col1.Cells.Add(row1);
                }

                if (pickMemo.TryGetValue(arg.Vertex, out var picked) && picked == true)
                {
                    arg.VertexFormat.FillColor = QuikGraph.Graphviz.Dot.GraphvizColor.SkyBlue;
                }
            };

            alg.FormatEdge += (_, arg) =>
            {
                arg.EdgeFormat.Direction = QuikGraph.Graphviz.Dot.GraphvizEdgeDirection.Back;
                arg.EdgeFormat.TailPort = $"P{arg.Edge.InputIndex}";
            };
        }));
    }

    private void DumpCostSummary(Stream stream, DistributedSearchGraph rootCluster, IReadOnlyDictionary<SearchableNode, bool> pickMemo, IReadOnlyDictionary<SearchableNode, CostModel.Cost> costMemo, IReadOnlyDictionary<SearchableNode, UInt128> costScoreMemo)
    {
        using var writer = new StreamWriter(stream);
        var dump = BuildCostDumpContext(rootCluster, pickMemo, costMemo, costScoreMemo);
        var topK = GetCostDumpTopK();
        var focusTerms = GetCostDumpFocusTerms();

        writer.WriteLine("# AutoDistributed Cost Pick Summary");
        writer.WriteLine($"top_k: {topK}");
        writer.WriteLine($"focus_terms: {string.Join(", ", focusTerms)}");
        writer.WriteLine($"selected_local_score_sum: {pickMemo.Where(kv => kv.Value).Aggregate((UInt128)0, (sum, kv) => sum + GetScore(costScoreMemo, kv.Key))}");
        writer.WriteLine($"selected_objective_score_sum: {pickMemo.Where(kv => kv.Value).Aggregate((UInt128)0, (sum, kv) => checked(sum + GetObjectiveScore(costScoreMemo, kv.Key)))}");
        var selectedAggregateCost = pickMemo.Where(kv => kv.Value).Aggregate(CostModel.Cost.Zero, (sum, kv) => sum + (costMemo.TryGetValue(kv.Key, out var cost) ? cost : CostModel.Cost.Zero));
        var selectedExecutionWeightedCost = pickMemo.Where(kv => kv.Value).Aggregate(
            CostModel.Cost.Zero,
            (sum, kv) => sum + ((costMemo.TryGetValue(kv.Key, out var cost) ? cost : CostModel.Cost.Zero) * (UInt128)GetInvocationCount(kv.Key)));
        writer.WriteLine($"selected_local_aggregate_cost: {FormatCost(selectedAggregateCost)}");
        writer.WriteLine($"selected_local_aggregate_latency: {FormatLatencyBreakdown(dump.TargetCostModel, selectedAggregateCost, null)}");
        writer.WriteLine($"selected_execution_weighted_cost: {FormatCost(selectedExecutionWeightedCost)}");
        writer.WriteLine($"root_cluster: {dump.GetGraphName(rootCluster)}");
        writer.WriteLine();

        foreach (var cluster in dump.Clusters)
        {
            var selectedNodes = cluster.Clusters.OfType<DistributedSearchGraph>()
                .SelectMany(bucket => bucket.Vertices)
                .Where(dump.IsPicked)
                .ToArray();

            writer.WriteLine($"## {dump.GetGraphName(cluster)} {cluster.Kind}");
            if (selectedNodes.Length > 0)
            {
                writer.WriteLine($"selected: {string.Join(", ", selectedNodes.Select(dump.GetNodeName))}");
            }

            foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
            {
                DumpBucketSummary(writer, bucket, dump, topK, focusTerms);
            }

            writer.WriteLine();
        }
    }

    private void DumpSelectedTree(Stream stream, DistributedSearchGraph rootCluster, IReadOnlyDictionary<SearchableNode, bool> pickMemo, IReadOnlyDictionary<SearchableNode, CostModel.Cost> costMemo, IReadOnlyDictionary<SearchableNode, UInt128> costScoreMemo)
    {
        using var writer = new StreamWriter(stream);
        var dump = BuildCostDumpContext(rootCluster, pickMemo, costMemo, costScoreMemo);
        var rootNode = rootCluster.Vertices.FirstOrDefault(dump.IsPicked);
        var maxDepth = GetSelectedTreeMaxDepth();
        var maxNodes = GetSelectedTreeMaxNodes();
        writer.WriteLine("# AutoDistributed Selected Tree");
        writer.WriteLine($"root_cluster: {dump.GetGraphName(rootCluster)}");
        writer.WriteLine($"max_depth: {maxDepth}");
        writer.WriteLine($"max_nodes: {maxNodes}");
        if (rootNode is null)
        {
            writer.WriteLine("root_selected: <none>");
            return;
        }

        var active = new HashSet<SearchableNode>();
        var emitted = new HashSet<SearchableNode>();
        var references = 0;
        var truncated = false;
        DumpSelectedNode(writer, rootNode, dump, active, emitted, ref references, ref truncated, 0, maxDepth, maxNodes);
        writer.WriteLine();
        writer.WriteLine($"emitted_nodes: {emitted.Count}");
        writer.WriteLine($"references: {references}");
        writer.WriteLine($"truncated: {truncated}");
    }

    private void DumpCandidateDiagnostics(Stream stream)
    {
        using var writer = new StreamWriter(stream);
        var focusTerms = GetCostDumpFocusTerms();
        writer.WriteLine("# AutoDistributed Candidate Diagnostics");
        writer.WriteLine($"total_records: {_candidateDiagnosticTotal}");
        writer.WriteLine($"distinct_records: {_candidateDiagnostics.Count}");
        writer.WriteLine($"focus_terms: {string.Join(", ", focusTerms)}");
        if (!_recordCandidateDiagnostics)
        {
            writer.WriteLine("detail: disabled");
            writer.WriteLine("enable_with: NNCASE_DUMP_AD_CANDIDATES=1");
            return;
        }

        writer.WriteLine();

        foreach (var entry in _candidateDiagnostics
            .Where(entry => focusTerms.Count == 0 || MatchesFocusText(entry.Key.ToString(), focusTerms))
            .OrderBy(entry => entry.Key.Target)
            .ThenBy(entry => entry.Key.Status)
            .ThenByDescending(entry => entry.Value)
            .ThenBy(entry => entry.Key.Reason)
            .ThenBy(entry => entry.Key.ResultType))
        {
            writer.WriteLine($"## {entry.Key.Target}");
            writer.WriteLine($"count: {entry.Value}");
            writer.WriteLine($"stage: {entry.Key.Stage}");
            writer.WriteLine($"status: {entry.Key.Status}");
            writer.WriteLine($"reason: {entry.Key.Reason}");
            writer.WriteLine($"result: {entry.Key.ResultType}");
            writer.WriteLine($"args: {entry.Key.Arguments}");
            writer.WriteLine();
        }
    }

    private void DumpBucketSummary(StreamWriter writer, DistributedSearchGraph bucket, CostDumpContext dump, int topK, IReadOnlyList<string> focusTerms)
    {
        var bucketName = dump.GetGraphName(bucket);
        var selected = bucket.Vertices.Where(dump.IsPicked).ToArray();
        writer.WriteLine($"### {bucketName} {bucket.Kind}");
        writer.WriteLine($"type: {GetOneLine(bucket.Vertices.FirstOrDefault()?.IRType.ToString() ?? string.Empty)}");
        writer.WriteLine($"root_reachable: {dump.IsRootReachable(bucket)} selected_tree: {dump.IsSelectedDependency(bucket)}");
        if (selected.Length > 0)
        {
            writer.WriteLine($"picked: {string.Join(", ", selected.Select(dump.GetNodeName))}");
        }

        var ranked = bucket.Vertices
            .OrderBy(node => GetScore(dump.CostScoreMemo, node))
            .ThenBy(dump.GetNodeName)
            .ToArray();
        var printed = new HashSet<SearchableNode>();
        for (int i = 0; i < ranked.Length; i++)
        {
            var node = ranked[i];
            if (i < topK || dump.IsPicked(node) || MatchesFocus(node, focusTerms) || ContainsPartial(node.IRType))
            {
                printed.Add(node);
                DumpCandidate(writer, node, dump, indent: "  ", rank: i);
            }
        }

        if (ranked.Length > printed.Count)
        {
            writer.WriteLine($"  ... {ranked.Length - printed.Count} candidates omitted");
        }

        if (dump.ConsumerEdgesByInputGraph.TryGetValue(bucket, out var consumers))
        {
            writer.WriteLine("  consumers:");
            foreach (var edge in consumers
                .OrderBy(e => dump.GetNodeName(e.Root))
                .ThenBy(e => e.InputIndex))
            {
                writer.WriteLine($"    <- P{edge.InputIndex} {dump.GetNodeName(edge.Root)} {GetNodeLabel(edge.Root)} picked={dump.IsPicked(edge.Root)} local_score={GetScore(dump.CostScoreMemo, edge.Root)} objective_score={GetObjectiveScore(dump.CostScoreMemo, edge.Root)} bucket={dump.GetGraphName(dump.GetBucket(edge.Root))}");
            }
        }
    }

    private void DumpCandidate(StreamWriter writer, SearchableNode node, CostDumpContext dump, string indent, int rank)
    {
        writer.WriteLine($"{indent}[{rank}] {dump.GetNodeName(node)} picked={dump.IsPicked(node)} local_score={GetScore(dump.CostScoreMemo, node)} invocation_count={GetInvocationCount(node)} objective_score={GetObjectiveScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
        writer.WriteLine($"{indent}    type: {GetOneLine(node.IRType.ToString() ?? string.Empty)}");
        var cost = dump.CostMemo.TryGetValue(node, out var nodeCost) ? nodeCost : CostModel.Cost.Zero;
        writer.WriteLine($"{indent}    cost: {FormatCost(cost)}");
        writer.WriteLine($"{indent}    latency: {FormatLatencyBreakdown(dump.TargetCostModel, cost, node.IRType)}");
        DumpCandidateInputs(writer, node, dump, indent + "    ");
    }

    private void DumpCandidateInputs(StreamWriter writer, SearchableNode node, CostDumpContext dump, string indent)
    {
        if (!_rootSearchGraph.TryGetOutEdges(node, out var edges))
        {
            return;
        }

        var orderedEdges = edges.OrderBy(e => e.InputIndex).ToArray();
        if (orderedEdges.Length == 0)
        {
            return;
        }

        writer.WriteLine($"{indent}inputs:");
        foreach (var edge in orderedEdges)
        {
            var selected = edge.InputGraph.Vertices.FirstOrDefault(dump.IsPicked);
            var selectedText = selected is null
                ? "<none>"
                : $"{dump.GetNodeName(selected)} {GetNodeLabel(selected)} local_score={GetScore(dump.CostScoreMemo, selected)} objective_score={GetObjectiveScore(dump.CostScoreMemo, selected)}";
            var best = dump.GetBestNode(edge.InputGraph);
            var bestText = best is null
                ? "<none>"
                : $"{dump.GetNodeName(best)} {GetNodeLabel(best)} local_score={GetScore(dump.CostScoreMemo, best)} objective_score={GetObjectiveScore(dump.CostScoreMemo, best)}";
            writer.WriteLine($"{indent}  P{edge.InputIndex} -> {dump.GetGraphName(edge.InputGraph)} root_reachable={dump.IsRootReachable(edge.InputGraph)} selected_tree={dump.IsSelectedDependency(edge.InputGraph)} selected={selectedText} best={bestText}");
        }
    }

    private void DumpSelectedNode(
        StreamWriter writer,
        SearchableNode node,
        CostDumpContext dump,
        HashSet<SearchableNode> active,
        HashSet<SearchableNode> emitted,
        ref int references,
        ref bool truncated,
        int depth,
        int maxDepth,
        int maxNodes)
    {
        var indent = new string(' ', depth * 2);
        if (depth > maxDepth)
        {
            writer.WriteLine($"{indent}<max-depth node={dump.GetNodeName(node)} bucket={dump.GetGraphName(dump.GetBucket(node))}>");
            truncated = true;
            return;
        }

        if (active.Contains(node))
        {
            writer.WriteLine($"{indent}{dump.GetNodeName(node)} <cycle> bucket={dump.GetGraphName(dump.GetBucket(node))} local_score={GetScore(dump.CostScoreMemo, node)} objective_score={GetObjectiveScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
            truncated = true;
            return;
        }

        if (!emitted.Add(node))
        {
            references++;
            writer.WriteLine($"{indent}{dump.GetNodeName(node)} <ref> bucket={dump.GetGraphName(dump.GetBucket(node))} local_score={GetScore(dump.CostScoreMemo, node)} objective_score={GetObjectiveScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
            return;
        }

        if (emitted.Count > maxNodes)
        {
            writer.WriteLine($"{indent}<max-nodes node={dump.GetNodeName(node)} bucket={dump.GetGraphName(dump.GetBucket(node))}>");
            truncated = true;
            return;
        }

        writer.WriteLine($"{indent}{dump.GetNodeName(node)} bucket={dump.GetGraphName(dump.GetBucket(node))} local_score={GetScore(dump.CostScoreMemo, node)} invocation_count={GetInvocationCount(node)} objective_score={GetObjectiveScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
        writer.WriteLine($"{indent}  type: {GetOneLine(node.IRType.ToString() ?? string.Empty)}");
        var cost = dump.CostMemo.TryGetValue(node, out var nodeCost) ? nodeCost : CostModel.Cost.Zero;
        writer.WriteLine($"{indent}  cost: {FormatCost(cost)}");
        writer.WriteLine($"{indent}  latency: {FormatLatencyBreakdown(dump.TargetCostModel, cost, node.IRType)}");

        active.Add(node);
        if (_rootSearchGraph.TryGetOutEdges(node, out var edges))
        {
            foreach (var edge in edges.OrderBy(e => e.InputIndex))
            {
                var selected = edge.InputGraph.Vertices.FirstOrDefault(dump.IsPicked);
                writer.WriteLine($"{indent}  P{edge.InputIndex} -> {dump.GetGraphName(edge.InputGraph)}");
                if (selected is null)
                {
                    writer.WriteLine($"{indent}    <none>");
                }
                else
                {
                    DumpSelectedNode(writer, selected, dump, active, emitted, ref references, ref truncated, depth + 2, maxDepth, maxNodes);
                }
            }
        }

        active.Remove(node);
    }

    private CostDumpContext BuildCostDumpContext(DistributedSearchGraph rootCluster, IReadOnlyDictionary<SearchableNode, bool> pickMemo, IReadOnlyDictionary<SearchableNode, CostModel.Cost> costMemo, IReadOnlyDictionary<SearchableNode, UInt128> costScoreMemo)
    {
        var clusters = _rootSearchGraph.Clusters.OfType<DistributedSearchGraph>().ToArray();
        var graphNames = new Dictionary<DistributedSearchGraph, string>();
        var bucketByNode = new Dictionary<SearchableNode, DistributedSearchGraph>();
        var nodeNames = _rootSearchGraph.Vertices
            .Select((node, index) => (node, name: $"N{index}"))
            .ToDictionary(pair => pair.node, pair => pair.name);

        for (int clusterIndex = 0; clusterIndex < clusters.Length; clusterIndex++)
        {
            var cluster = clusters[clusterIndex];
            var clusterName = $"C{clusterIndex}";
            graphNames[cluster] = clusterName;
            var buckets = cluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
            for (int bucketIndex = 0; bucketIndex < buckets.Length; bucketIndex++)
            {
                var bucket = buckets[bucketIndex];
                graphNames[bucket] = $"{clusterName}.B{bucketIndex}";
                foreach (var node in bucket.Vertices)
                {
                    bucketByNode[node] = bucket;
                }
            }
        }

        var consumerEdgesByInputGraph = _rootSearchGraph.Edges
            .GroupBy(edge => edge.InputGraph)
            .ToDictionary(group => group.Key, group => group.ToArray());
        var bestNodeByBucket = graphNames.Keys
            .Where(graph => graph.Kind is SearchGraphKind.Bucket)
            .ToDictionary(
                graph => graph,
                graph => graph.Vertices
                    .OrderBy(node => GetScore(costScoreMemo, node))
                    .ThenBy(node => nodeNames.TryGetValue(node, out var name) ? name : string.Empty)
                    .FirstOrDefault());
        var rootReachableBuckets = GetDependencyBuckets(rootCluster, pickMemo: null);
        var selectedDependencyBuckets = GetDependencyBuckets(rootCluster, pickMemo);
        var targetCostModel = CostModel.TargetOpCostModelUtility.GetTargetCostModel(CompileOptions);

        return new CostDumpContext(clusters, graphNames, nodeNames, bucketByNode, consumerEdgesByInputGraph, bestNodeByBucket, rootReachableBuckets, selectedDependencyBuckets, pickMemo, costMemo, costScoreMemo, targetCostModel);
    }

    private HashSet<DistributedSearchGraph> GetDependencyBuckets(DistributedSearchGraph rootCluster, IReadOnlyDictionary<SearchableNode, bool>? pickMemo)
    {
        bool IsAllowedNode(SearchableNode node) => pickMemo is null || (pickMemo.TryGetValue(node, out var picked) && picked);

        var visited = new HashSet<DistributedSearchGraph>();
        var queue = new Queue<DistributedSearchGraph>();
        foreach (var rootBucket in rootCluster.Clusters.OfType<DistributedSearchGraph>())
        {
            visited.Add(rootBucket);
            queue.Enqueue(rootBucket);
        }

        while (queue.Count > 0)
        {
            var bucket = queue.Dequeue();
            foreach (var node in bucket.Vertices.Where(IsAllowedNode))
            {
                if (!_rootSearchGraph.TryGetOutEdges(node, out var edges))
                {
                    continue;
                }

                foreach (var edge in edges)
                {
                    if (pickMemo is not null && !edge.InputGraph.Vertices.Any(IsAllowedNode))
                    {
                        continue;
                    }

                    if (visited.Add(edge.InputGraph))
                    {
                        queue.Enqueue(edge.InputGraph);
                    }
                }
            }
        }

        return visited;
    }

    private int GetCostDumpTopK()
    {
        if (int.TryParse(Environment.GetEnvironmentVariable("NNCASE_DUMP_AD_COST_TOPK"), out var topK))
        {
            return Math.Max(1, topK);
        }

        return 6;
    }

    private IReadOnlyList<string> GetCostDumpFocusTerms()
    {
        var text = Environment.GetEnvironmentVariable("NNCASE_DUMP_AD_COST_FILTER");
        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<string>();
        }

        return text.Split(new[] { ';', ',', '|' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    private bool ShouldDumpSelectedTree()
        => string.Equals(Environment.GetEnvironmentVariable("NNCASE_DUMP_AD_SELECTED_TREE"), "1", StringComparison.OrdinalIgnoreCase);

    private int GetSelectedTreeMaxDepth()
    {
        if (int.TryParse(Environment.GetEnvironmentVariable("NNCASE_DUMP_AD_SELECTED_TREE_MAX_DEPTH"), out var maxDepth))
        {
            return Math.Max(1, maxDepth);
        }

        return 2048;
    }

    private int GetSelectedTreeMaxNodes()
    {
        if (int.TryParse(Environment.GetEnvironmentVariable("NNCASE_DUMP_AD_SELECTED_TREE_MAX_NODES"), out var maxNodes))
        {
            return Math.Max(1, maxNodes);
        }

        return 200000;
    }

    private bool MatchesFocus(SearchableNode node, IReadOnlyList<string> focusTerms)
    {
        if (focusTerms.Count == 0)
        {
            return false;
        }

        var label = $"{GetNodeLabel(node)} {node.IRType}";
        return MatchesFocusText(label, focusTerms);
    }

    private bool MatchesFocusText(string text, IReadOnlyList<string> focusTerms)
    {
        return focusTerms.Any(term => text.Contains(term, StringComparison.OrdinalIgnoreCase));
    }

    private bool ContainsPartial(IRType type) => type switch
    {
        DistributedType distributedType => distributedType.Partial is not null || distributedType.AxisPolicies.Any(policy => policy is SBPPartial),
        TupleType tupleType => tupleType.Fields.Any(ContainsPartial),
        _ => false,
    };

    private UInt128 GetScore(IReadOnlyDictionary<SearchableNode, UInt128> costScoreMemo, SearchableNode node)
        => costScoreMemo.TryGetValue(node, out var score) ? score : 0;

    private long GetInvocationCount(SearchableNode node)
    {
        if (!_nodeOwnerFunctions.TryGetValue(node, out var function))
        {
            return 1;
        }

        if (!_functionInvocationCounts.TryGetValue(function, out var invocationCount))
        {
            throw new InvalidOperationException(
                $"Function {function.Name} has no static invocation count.");
        }

        if (invocationCount <= 0)
        {
            throw new InvalidOperationException(
                $"Function {function.Name} has invalid static invocation count {invocationCount}.");
        }

        return invocationCount;
    }

    private UInt128 GetObjectiveScore(
        IReadOnlyDictionary<SearchableNode, UInt128> costScoreMemo,
        SearchableNode node)
        => checked(GetScore(costScoreMemo, node) * (UInt128)GetInvocationCount(node));

    private string FormatCost(CostModel.Cost cost)
        => cost.Factors.Count == 0
            ? "{}"
            : "{ " + string.Join(", ", cost.Factors.Select(kv => $"{kv.Key}={kv.Value}")) + " }";

    private string FormatLatencyBreakdown(CostModel.ITargetOpCostModel targetCostModel, CostModel.Cost cost, IRType? resultType)
    {
        var breakdown = CostModel.TargetOpCostModelUtility.GetCostLatencyBreakdown(targetCostModel, cost, resultType);
        var blockLocalBytes = GetCostFactor(cost, CostModel.CostFactorNames.BlockLocalMemoryLoadBytes) + GetCostFactor(cost, CostModel.CostFactorNames.BlockLocalMemoryStoreBytes);
        var explicitChipGlobalBytes = GetCostFactor(cost, CostModel.CostFactorNames.ChipGlobalMemoryLoadBytes) + GetCostFactor(cost, CostModel.CostFactorNames.ChipGlobalMemoryStoreBytes);
        var effectiveChipGlobalBytes = ToDouble(blockLocalBytes + explicitChipGlobalBytes) * Math.Max(1L, breakdown.ActiveBlockCount);
        return "{" +
            $" active_blocks={breakdown.ActiveBlockCount}," +
            $" local_bytes={blockLocalBytes}," +
            $" explicit_chipglobal_bytes={explicitChipGlobalBytes}," +
            $" effective_chipglobal_bytes={FormatDouble(effectiveChipGlobalBytes)}," +
            $" cpu={FormatDouble(breakdown.CPUCycles)}," +
            $" blocklocal={FormatDouble(breakdown.BlockLocalMemoryCycles)}," +
            $" chipglobal={FormatDouble(breakdown.ChipGlobalMemoryCycles)}," +
            $" overlap={FormatDouble(breakdown.OverlappedCycles)}," +
            $" block_sync={FormatDouble(breakdown.BlockSynchronizationCycles)}," +
            $" grid_sync={FormatDouble(breakdown.GridSynchronizationCycles)}," +
            $" comm={FormatDouble(breakdown.CommCycles)}," +
            $" other={FormatDouble(breakdown.OtherCycles)}," +
            $" latency={breakdown.Latency}" +
            " }";
    }

    private string FormatDouble(double value)
        => value.ToString("0.###", CultureInfo.InvariantCulture);

    private UInt128 GetCostFactor(CostModel.Cost cost, string name)
        => cost.Factors.TryGetValue(name, out var value) ? value : 0;

    private double ToDouble(UInt128 value)
        => value > ulong.MaxValue ? ulong.MaxValue : (ulong)value;

    private string GetNodeLabel(SearchableNode node)
        => GetExprLabel(node.Expr);

    private string GetExprLabel(BaseExpr expr)
    {
        if (expr is Op op)
        {
            var property = op.DisplayProperty();
            return string.IsNullOrWhiteSpace(property)
                ? op.GetType().FullName ?? op.GetType().Name
                : $"{op.GetType().FullName}({property})";
        }

        return expr.GetType().FullName ?? expr.GetType().Name;
    }

    private string FormatType(IRType? type)
    {
        return GetOneLine(type?.ToString() ?? "<none>");
    }

    private string GetOneLine(string text)
        => text.Replace("\r", " ", StringComparison.Ordinal).Replace("\n", " ", StringComparison.Ordinal);

    private string BuildCandidateFailureMessage(Call expr, IReadOnlyList<DistributedSearchGraph> argClusters)
    {
        const int maxBucketsPerArg = 16;

        var builder = new StringBuilder();
        builder.AppendLine($"[AutoDistributed] Type infer failed for {GetExprLabel(expr.Target)}.");
        builder.AppendLine($"Source: {GetOneLine(expr.ToString())}");
        builder.AppendLine($"Source checked type: {FormatType(expr.CheckedType)}");
        for (var i = 0; i < expr.Arguments.Length; i++)
        {
            builder.AppendLine($"Source arg {i}: {GetExprLabel(expr.Arguments[i])}, checked type: {FormatType(expr.Arguments[i].CheckedType)}");
        }

        for (var i = 0; i < argClusters.Count; i++)
        {
            var buckets = argClusters[i].Clusters.OfType<DistributedSearchGraph>().ToArray();
            builder.AppendLine($"Arg {i}: buckets={buckets.Length}");
            foreach (var (bucket, bucketIndex) in buckets.Take(maxBucketsPerArg).Select((bucket, index) => (bucket, index)))
            {
                var vertex = bucket.Vertices.FirstOrDefault();
                builder.AppendLine($"  [{bucketIndex}] {FormatType(vertex?.IRType)}");
            }

            if (buckets.Length > maxBucketsPerArg)
            {
                builder.AppendLine($"  ... {buckets.Length - maxBucketsPerArg} more bucket(s)");
            }
        }

        return builder.ToString();
    }

    private void PruneDominatedCandidates(Dictionary<SearchableNode, UInt128> costScoreMemo, Dictionary<SearchableNode, CostModel.Cost> costMemo)
    {
        var replacements = new Dictionary<SearchableNode, SearchableNode>(ReferenceEqualityComparer.Instance);
        var buckets = _rootSearchGraph.Clusters
            .OfType<DistributedSearchGraph>()
            .SelectMany(cluster => cluster.Clusters.OfType<DistributedSearchGraph>())
            .Where(bucket => bucket.Kind is SearchGraphKind.Bucket)
            .ToArray();

        foreach (var bucket in buckets)
        {
            var vertices = bucket.Vertices.ToArray();
            if (vertices.Length <= 1)
            {
                continue;
            }

            var groups = vertices
                .GroupBy(node => new CandidateDominanceKey(node, GetOrderedOutEdges(node)))
                .Where(group => group.Count() > 1);
            foreach (var group in groups)
            {
                var keep = group
                    .OrderBy(node => GetScore(costScoreMemo, node))
                    .ThenBy(node => IsReshardRealization(node.Expr) ? 1 : 0)
                    .ThenBy(node => Array.IndexOf(vertices, node))
                    .First();

                foreach (var node in group)
                {
                    if (ReferenceEquals(node, keep))
                    {
                        continue;
                    }

                    replacements.Add(node, keep);
                }
            }
        }

        if (replacements.Count == 0)
        {
            _profiler.Count("pruned_dominated_candidates", 0);
            return;
        }

        var redirectedEdges = _rootSearchGraph.Edges
            .Where(edge => replacements.ContainsKey(edge.Target))
            .Select(edge => new CrossEdge(
                edge.Root,
                replacements[edge.Target],
                edge.InputIndex,
                edge.InputGraph))
            .ToArray();
        _rootSearchGraph.AddEdgeRange(redirectedEdges);
        var removedCount = _rootSearchGraph.RemoveVertexRange(replacements.Keys);
        if (removedCount != replacements.Count)
        {
            throw new InvalidOperationException(
                $"Dominated-candidate pruning planned {replacements.Count} removals, " +
                $"but the search graph removed {removedCount} vertices.");
        }

        foreach (var node in replacements.Keys)
        {
            costMemo.Remove(node);
            costScoreMemo.Remove(node);
        }

        _profiler.Count("pruned_dominated_candidates", removedCount);
    }

    private IReadOnlyList<CrossEdge> GetOrderedOutEdges(SearchableNode node)
        => _rootSearchGraph.TryGetOutEdges(node, out var edges)
            ? edges.OrderBy(edge => edge.InputIndex).ToArray()
            : Array.Empty<CrossEdge>();

    private Dictionary<SearchableNode, bool> Solve(DistributedSearchGraph rootCluster)
    {
        // 0. create bool var for all node.
        var cpmodel = new CpModel();
        var varMemo = new Dictionary<SearchableNode, BoolVar>();
        var clusterVarMemo = new Dictionary<DistributedSearchGraph, List<BoolVar>>();
        var costMemo = new Dictionary<SearchableNode, CostModel.Cost>();
        var costScoreMemo = new Dictionary<SearchableNode, UInt128>();
        var targetCostModel = CostModel.TargetOpCostModelUtility.GetTargetCostModel(CompileOptions);
        _profiler.Time("sat_build_costs_and_vars", () =>
        {
            foreach (var cluster in _rootSearchGraph.Clusters.OfType<DistributedSearchGraph>())
            {
                foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
                {
                    foreach (var enode in bucket.Vertices)
                    {
                        CostModel.Cost cost;
                        if (enode is
                            {
                                Expr: IR.Distributed.ShardedView,
                                ReshardUsageKind: DistributedReshardUsageKind.ProgramOutput,
                            })
                        {
                            // Kernel completion publishes caller-allocated outputs. Unlike an
                            // internal widened view, this terminal alias has no grid consumer.
                            cost = CostModel.Cost.Zero;
                        }
                        else if (enode.Kind is SearchableNodeKind.FunctionResult or SearchableNodeKind.FunctionBoundaryAdapter or SearchableNodeKind.TypeAdapter)
                        {
                            cost = new CostModel.Cost() { [CostModel.CostFactorNames.CPUCycles] = 0 };
                        }
                        else
                        {
                            switch (enode.Expr)
                            {
                                case TensorConst { ValueType: DistributedType distributedType }:
                                    cost = new CostModel.Cost()
                                    {
                                        [CostModel.CostFactorNames.BlockLocalMemoryStoreBytes] = GetLocalTensorBytes(distributedType),
                                    };
                                    break;
                                case Const or Var or If or IR.Tuple or BaseFunction or Shape or Padding or Paddings or Dimension or None or Call:
                                    cost = new CostModel.Cost() { [CostModel.CostFactorNames.CPUCycles] = 1 };
                                    break;
                                case Op op:
                                    {
                                        _profiler.Count("cost_evaluate_ops");
                                        _profiler.Count($"cost_evaluate_op:{op.GetType().Name}");
                                        if (!_rootSearchGraph.TryGetOutEdges(enode, out var edges))
                                        {
                                            throw new NotSupportedException("graph doesn't contain the vertex.");
                                        }

                                        var tempArgs = edges.Where(e => e.InputIndex >= 0).OrderBy(e => e.InputIndex).Select<CrossEdge, BaseExpr>(e => e.Target switch
                                        {
                                            SearchableNode { Expr: Dimension attr } => attr,
                                            SearchableNode { Expr: Shape attr } => attr,
                                            SearchableNode { Expr: Padding attr } => attr,
                                            SearchableNode { Expr: Paddings attr } => attr,
                                            SearchableNode { Expr: Const attr } => attr,
                                            SearchableNode n => new Var(n.IRType),
                                        }).ToArray();

                                        var context = new DistributedCostEvaluateContext(op, enode.IRType, tempArgs, CompileOptions);
                                        cost = _profiler.Time("cost_evaluate", () => CompilerServices.EvaluateOpCost(op, context));
                                    }

                                    break;
                                default:
                                    throw new NotSupportedException($"extract not support {enode.Expr.GetType()}");
                            }
                        }

                        costMemo.Add(enode, cost);
                        costScoreMemo.Add(enode, CostModel.TargetOpCostModelUtility.GetCostLatency(targetCostModel, cost, enode.IRType));
                    }
                }
            }

            _profiler.Time("prune_dominated_candidates", () => PruneDominatedCandidates(costScoreMemo, costMemo));

            foreach (var cluster in _rootSearchGraph.Clusters.OfType<DistributedSearchGraph>())
            {
                clusterVarMemo.Add(cluster, new());
                foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
                {
                    foreach (var enode in bucket.Vertices)
                    {
                        var boolVar = cpmodel.NewBoolVar(string.Empty);
                        varMemo.Add(enode, boolVar);
                        if (_singleChoiceClusters.Contains(cluster)
                            || IsExecutableOperationCandidate(enode))
                        {
                            clusterVarMemo[cluster].Add(boolVar);
                        }
                    }
                }
            }
        });
        _profiler.Count("sat_vars", varMemo.Count);
        _profiler.Count("sat_cost_nodes", costMemo.Count);

        // 1. must pick one in root enode.
        _profiler.Time("sat_add_constraints", () =>
        {
            cpmodel.AddExactlyOne(rootCluster.Vertices.Select(n => varMemo[n]).ToArray());

            // 2. pick only one in each cluster.
            foreach (var (cluster, vars) in clusterVarMemo)
            {
                if (vars.Count > 0)
                {
                    cpmodel.AddExactlyOne(vars.ToArray());
                }
            }

            // 3. when pick node, must pick one child node.
            foreach (var n in _rootSearchGraph.Vertices)
            {
                if (_rootSearchGraph.TryGetOutEdges(n, out var allEdges))
                {
                    foreach (var argEdges in allEdges.GroupBy(g => g.InputIndex))
                    {
                        var cns = argEdges.SelectMany(e => e.InputGraph.Vertices).Select(cn => varMemo[cn]).ToList();
                        if (cns.Count > 0)
                        {
                            cpmodel.Add(LinearExpr.Sum(cns) == 1).OnlyEnforceIf(varMemo[n]);
                        }
                    }
                }
            }
        });

#if false
        // 4. no cycle
        foreach (var cluster in _rootSearchGraph.Clusters.OfType<DistributedSearchGraph>())
        {
            foreach (var sourceBucket in cluster.Clusters.OfType<DistributedSearchGraph>())
            {
                foreach (var destBucket in cluster.Clusters.OfType<DistributedSearchGraph>().Where(b => !ReferenceEquals(b, sourceBucket)))
                {
                    foreach (var (src, dest) in sourceBucket.Vertices.Where(v => v.IsBidirect).Zip(destBucket.Vertices.Where(v => v.IsBidirect)))
                    {
                        cpmodel.AddBoolAnd([varMemo[src].Not(), varMemo[dest].Not()]);
                    }
                }
            }
        }
#endif

        // 5. add pick weights for all enode.
        _profiler.Time("sat_set_objective", () =>
            cpmodel.Minimize(LinearExpr.WeightedSum(
                _rootSearchGraph.Vertices.Select(n => varMemo[n]),
                _rootSearchGraph.Vertices.Select(n => checked((long)GetObjectiveScore(costScoreMemo, n))))));

        var validation = _profiler.Time("sat_validate", () => cpmodel.Validate());
        if (validation.Any())
        {
            throw new InvalidDataException("the sat model invalid: " + validation);
        }

        var solver = new CpSolver();
        int max_time = 120;
        if (System.Environment.GetEnvironmentVariable("SOLVE_MAX_TIME") is string s_solve_max_time)
        {
            try
            {
                var solve_max_time = int.Parse(s_solve_max_time);
                max_time = solve_max_time;
            }
            catch (System.Exception)
            {
            }
        }

        int processorCount = Math.Max(System.Environment.ProcessorCount / 2, 1);
        if (System.Environment.GetEnvironmentVariable("SOLVE_PROCESSOR_COUNT") is string s_solve_processor_count)
        {
            try
            {
                var solve_processor_count = int.Parse(s_solve_processor_count);
                processorCount = solve_processor_count;
            }
            catch (System.Exception)
            {
            }
        }

        solver.StringParameters = $"max_time_in_seconds:{max_time},num_workers:{processorCount}";

        var enableDump = Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Compile)
            || Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.EGraphCost)
            || string.Equals(Environment.GetEnvironmentVariable("NNCASE_DUMP_AD_COSTS"), "1", StringComparison.Ordinal);
        CpSolverStatus status;
        using (var dumpStream = Diagnostics.DumpScope.Current.OpenFile("Costs/Solve.txt"))
        {
            using var writer = new StreamWriter(dumpStream);
            var cb = new PrintCostCallBack(varMemo, costMemo, targetCostModel, writer, enableDump);
            status = _profiler.Time("sat_solve", () => solver.Solve(cpmodel, cb));
            writer.WriteLine($"Status : {status}");
        }

        if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Feasible))
        {
            throw new InvalidProgramException("SatExtract Failed!");
        }

        var picks = _profiler.Time("sat_read_picks", () => _rootSearchGraph.Vertices.ToDictionary(e => e, e => solver.BooleanValue(varMemo[e])));
        _lastPicks = picks;
        _profiler.Count("sat_picked_nodes", picks.Count(kv => kv.Value));
        _profiler.Time("dump_pick_dot", () =>
        {
            using var stream = enableDump ? Diagnostics.DumpScope.Current.OpenFile("Costs/Pick.dot") : Stream.Null;
            Dump(stream, picks, costMemo, costScoreMemo);
        });

        if (enableDump)
        {
            _profiler.Time("dump_pick_txt", () =>
            {
                using var stream = Diagnostics.DumpScope.Current.OpenFile("Costs/Pick.txt");
                DumpCostSummary(stream, rootCluster, picks, costMemo, costScoreMemo);
            });

            if (ShouldDumpSelectedTree())
            {
                _profiler.Time("dump_selected_tree", () =>
                {
                    using var stream = Diagnostics.DumpScope.Current.OpenFile("Costs/SelectedTree.txt");
                    DumpSelectedTree(stream, rootCluster, picks, costMemo, costScoreMemo);
                });
            }

            _profiler.Time("dump_candidate_diagnostics", () =>
            {
                using var stream = Diagnostics.DumpScope.Current.OpenFile("Costs/CandidateDiagnostics.txt");
                DumpCandidateDiagnostics(stream);
            });
        }

        if (_phase == AutoDistributedPhase.SearchConstant)
        {
            foreach (var pick in picks)
            {
                if (pick.Value && pick.Key.Expr is TensorConst { ValueType: DistributedType } distConst
                    && _distributedConstSources.TryGetValue(distConst, out var source))
                {
                    DistributedConsts.Add(source, distConst);
                }
            }
        }

        return picks;
    }

    private bool IsReshardRealization(BaseExpr expr)
        => expr is IR.Distributed.Boxing or IR.Distributed.ShardedView;

    private bool IsExecutableOperationCandidate(SearchableNode node)
        => node is { Kind: SearchableNodeKind.Normal, Expr: Op op }
            && !IsReshardRealization(op);

    private BaseExpr ExtractSelectedExpression(DistributedSearchGraph rootCluster, Dictionary<SearchableNode, bool> picks)
        => _profiler.Time("extract_expr", () => new ExprBuildVisitor(_rootSearchGraph, picks).Visit(rootCluster.Clusters.OfType<DistributedSearchGraph>()));

    private HyperGraph<DistributedSearchGraph, SearchableNode> ToHyperGraph(DistributedSearchGraph root, DistributedSearchGraph rootCluster)
    {
        var hgraph = new HyperGraph<DistributedSearchGraph, SearchableNode>();
        var visited = new HashSet<DistributedSearchGraph>();
        var queue = new Queue<DistributedSearchGraph>();
        var rootBuckets = rootCluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
        if (rootBuckets.Length != 1)
        {
            throw new InvalidOperationException("The root Cluster should contains only one bucket!");
        }

        queue.Enqueue(rootBuckets[0]);
        visited.Add(rootBuckets[0]);
        while (queue.Any())
        {
            var front = queue.Dequeue();
            foreach (var node in front.Vertices)
            {
                root.TryGetOutEdges(node, out var edges);
                foreach (var edge in edges)
                {
                    var canonical = edge.InputGraph;
                    hgraph.Connect(front, canonical, node);
                    if (!visited.Contains(canonical))
                    {
                        visited.Add(canonical);
                        queue.Enqueue(canonical);
                    }
                }
            }
        }

        return hgraph;
    }

    private sealed record CostDumpContext(
        IReadOnlyList<DistributedSearchGraph> Clusters,
        IReadOnlyDictionary<DistributedSearchGraph, string> GraphNames,
        IReadOnlyDictionary<SearchableNode, string> NodeNames,
        IReadOnlyDictionary<SearchableNode, DistributedSearchGraph> BucketByNode,
        IReadOnlyDictionary<DistributedSearchGraph, CrossEdge[]> ConsumerEdgesByInputGraph,
        IReadOnlyDictionary<DistributedSearchGraph, SearchableNode?> BestNodeByBucket,
        IReadOnlySet<DistributedSearchGraph> RootReachableBuckets,
        IReadOnlySet<DistributedSearchGraph> SelectedDependencyBuckets,
        IReadOnlyDictionary<SearchableNode, bool> PickMemo,
        IReadOnlyDictionary<SearchableNode, CostModel.Cost> CostMemo,
        IReadOnlyDictionary<SearchableNode, UInt128> CostScoreMemo,
        CostModel.ITargetOpCostModel TargetCostModel)
    {
        public bool IsPicked(SearchableNode node) => PickMemo.TryGetValue(node, out var picked) && picked;

        public string GetGraphName(DistributedSearchGraph graph) => GraphNames.TryGetValue(graph, out var name) ? name : "<unknown>";

        public string GetNodeName(SearchableNode node) => NodeNames.TryGetValue(node, out var name) ? name : "<unknown>";

        public DistributedSearchGraph GetBucket(SearchableNode node) => BucketByNode.TryGetValue(node, out var bucket) ? bucket : null!;

        public SearchableNode? GetBestNode(DistributedSearchGraph bucket) => BestNodeByBucket.TryGetValue(bucket, out var node) ? node : null;

        public bool IsRootReachable(DistributedSearchGraph bucket) => RootReachableBuckets.Contains(bucket);

        public bool IsSelectedDependency(DistributedSearchGraph bucket) => SelectedDependencyBuckets.Contains(bucket);
    }
}

internal sealed class DistributedProgramMaterializer
{
    private readonly DistributedSearchGraph _rootSearchGraph;
    private readonly Dictionary<SearchableNode, bool> _picks;

    public DistributedProgramMaterializer(DistributedSearchGraph rootSearchGraph, Dictionary<SearchableNode, bool> picks)
    {
        _rootSearchGraph = rootSearchGraph;
        _picks = picks;
    }

    public IReadOnlyDictionary<Function, Function> Materialize(
        Function rootFunction,
        IReadOnlyList<Function> reachableFunctions,
        IReadOnlyDictionary<Function, DistributedSearchGraph> functionRootClusters,
        IReadOnlyDictionary<Function, Dictionary<IVar, DistributedSearchGraph>> functionParameterClusters)
    {
        var rewritten = new Dictionary<Function, Function>(ReferenceEqualityComparer.Instance);
        foreach (var function in reachableFunctions)
        {
            var isEntry = ReferenceEquals(function, rootFunction);
            var parameterMap = BuildParameterMap(function, isEntry, functionParameterClusters);
            var rootCluster = functionRootClusters.TryGetValue(function, out var cluster)
                ? cluster
                : throw new InvalidOperationException($"AutoDistributed has no root cluster for function {function.Name}.");
            var body = new ExprBuildVisitor(_rootSearchGraph, _picks, parameterMap, rewritten).Visit(rootCluster.Clusters.OfType<DistributedSearchGraph>());
            var parameters = function.Parameters.ToArray()
                .Select(parameter => parameterMap.TryGetValue(parameter, out var mapped) ? mapped : parameter)
                .ToArray();
            var newVarMap = RemapVarMap(function, parameterMap);
            var newFunction = new Function(function.Name, function.ModuleKind, body, parameters, newVarMap)
            {
                Metadata = function.Metadata,
                Role = function.Role,
            };
            if (!CompilerServices.InferenceType(newFunction) || newFunction.CheckedType is InvalidType)
            {
                throw new InvalidOperationException($"AutoDistributed materialized function {function.Name} produced invalid type: {newFunction.CheckedType}.");
            }

            rewritten.Add(function, newFunction);
        }

        return rewritten;
    }

    private Dictionary<IVar, IVar> BuildParameterMap(
        Function function,
        bool isEntry,
        IReadOnlyDictionary<Function, Dictionary<IVar, DistributedSearchGraph>> functionParameterClusters)
    {
        var result = new Dictionary<IVar, IVar>(ReferenceEqualityComparer.Instance);
        if (isEntry || !functionParameterClusters.TryGetValue(function, out var parameterClusters))
        {
            return result;
        }

        foreach (var parameter in function.Parameters)
        {
            if (!parameterClusters.TryGetValue(parameter, out var cluster))
            {
                continue;
            }

            var selected = GetSelectedNode(cluster);
            result.Add(parameter, parameter switch
            {
                Var var => var.With(typeAnnotation: selected.IRType),
                _ => throw new InvalidOperationException($"AutoDistributed can only materialize tensor function parameter signatures for Var, got {parameter.GetType().Name}."),
            });
        }

        return result;
    }

    private Dictionary<IVar, Dimension[]>? RemapVarMap(Function function, IReadOnlyDictionary<IVar, IVar> parameterMap)
    {
        if (function.VarMap is null)
        {
            return null;
        }

        return function.VarMap.ToDictionary(
            kvp => parameterMap.TryGetValue(kvp.Key, out var mapped) ? mapped : kvp.Key,
            kvp => kvp.Value,
            (IEqualityComparer<IVar>)ReferenceEqualityComparer.Instance);
    }

    private SearchableNode GetSelectedNode(DistributedSearchGraph cluster)
    {
        var selected = cluster.Clusters.OfType<DistributedSearchGraph>()
            .SelectMany(bucket => bucket.Vertices)
            .Where(node => _picks.TryGetValue(node, out var picked) && picked)
            .ToArray();
        if (selected.Length != 1)
        {
            throw new InvalidOperationException($"AutoDistributed expected one selected signature node in cluster, got {selected.Length}.");
        }

        return selected[0];
    }
}

internal sealed class ExprBuildVisitor
{
    private readonly Dictionary<SearchableNode, bool> _picks;
    private readonly DistributedSearchGraph _rootSearchGraph;
    private readonly Dictionary<SearchableNode, BaseExpr> _memo;
    private readonly Dictionary<BaseExpr, Dictionary<ReshardMaterializationKey, BaseExpr>> _materializedReshards;
    private readonly IReadOnlyDictionary<IVar, IVar> _parameterMap;
    private readonly IReadOnlyDictionary<Function, Function> _functionMap;

    public ExprBuildVisitor(
        DistributedSearchGraph rootSearchGraph,
        Dictionary<SearchableNode, bool> picks,
        IReadOnlyDictionary<IVar, IVar>? parameterMap = null,
        IReadOnlyDictionary<Function, Function>? functionMap = null)
    {
        _rootSearchGraph = rootSearchGraph;
        _picks = picks;
        _memo = new();
        _materializedReshards = new(ReferenceEqualityComparer.Instance);
        _parameterMap = parameterMap ?? new Dictionary<IVar, IVar>(ReferenceEqualityComparer.Instance);
        _functionMap = functionMap ?? new Dictionary<Function, Function>(ReferenceEqualityComparer.Instance);
    }

    public BaseExpr Visit(IEnumerable<DistributedSearchGraph> rootBuckets)
    {
        var rootPicks = rootBuckets.SelectMany(b => b.Vertices).Where(v => _picks.TryGetValue(v, out var pick) && pick).ToArray();
        if (rootPicks.Length != 1)
        {
            throw new InvalidProgramException("the one cluster only can pick one vertex!");
        }

        var root = rootPicks[0];
        if (!_memo.TryGetValue(root, out var expr))
        {
            _rootSearchGraph.TryGetOutEdges(root, out var edges);
            var children = edges
                .Where(e => e.InputIndex >= 0)
                .GroupBy(e => e.InputIndex)
                .OrderBy(g => g.Key)
                .Select(g => Visit(g.Select(e => e.InputGraph)))
                .ToArray();
            switch (root.Kind, root.Expr)
            {
                case (SearchableNodeKind.FunctionResult, IR.Tuple tuple):
                    expr = tuple.With(fields: children);
                    break;
                case (SearchableNodeKind.FunctionResult, _):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"{root.Kind} expects one direct producer, got {children.Length}.");
                    }

                    var resultType = EnsureMaterializedType(children[0], $"{root.Kind} input");
                    if (!EqualityComparer<IRType>.Default.Equals(resultType, root.IRType))
                    {
                        throw new InvalidOperationException(
                            $"{root.Kind} cannot change direct producer type {resultType} to signature type {root.IRType}.");
                    }

                    expr = children[0];
                    break;
                case (SearchableNodeKind.FunctionBoundaryAdapter, Op realization):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"{root.Kind} expects one data input, got {children.Length}.");
                    }

                    expr = MaterializeReshard(children[0], realization, root.IRType, $"{root.Kind} node");
                    break;
                case (SearchableNodeKind.FunctionBoundaryAdapter, _):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"{root.Kind} expects one data input, got {children.Length}.");
                    }

                    var boundaryInputType = EnsureMaterializedType(children[0], $"{root.Kind} input");
                    if (!EqualityComparer<IRType>.Default.Equals(boundaryInputType, root.IRType))
                    {
                        throw new InvalidOperationException(
                            $"{root.Kind} identity cannot change {boundaryInputType} to {root.IRType}.");
                    }

                    expr = children[0];
                    break;
                case (SearchableNodeKind.TypeAdapter, _):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"{root.Kind} expects one data input, got {children.Length}.");
                    }

                    var childType = EnsureMaterializedType(children[0], $"{root.Kind} input");
                    if (!EqualityComparer<IRType>.Default.Equals(childType, root.IRType))
                    {
                        throw new InvalidOperationException(
                            $"{root.Kind} cannot change {childType} to {root.IRType} without an explicit reshard node.");
                    }

                    expr = children[0];
                    break;
                case (_, Var var):
                    expr = _parameterMap.TryGetValue(var, out var mapped) ? (BaseExpr)mapped : var;
                    break;
                case (_, TensorConst or TupleConst or None or Shape or Padding or Paddings or Dimension):
                    expr = root.Expr;
                    break;
                case (_, Call { Target: Boxing boxing } call):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"Cannot rebuild boxing call: expected one argument, got {children.Length}.");
                    }

                    expr = MaterializeReshard(children[0], boxing, boxing.NewType, "selected boxing call");
                    break;
                case (_, Call { Target: IR.Distributed.ShardedView shardedView }):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"Cannot rebuild sharded view call: expected one argument, got {children.Length}.");
                    }

                    expr = MaterializeReshard(children[0], shardedView, shardedView.NewType, "selected sharded view call");
                    break;
                case (_, Call call):
                    if (children.Length == call.Arguments.Length)
                    {
                        expr = InheritSelectedMetadata(call.With(arguments: children), root);
                    }
                    else if (children.Length == 1 && EqualityComparer<IRType>.Default.Equals(children[0].CheckedType, root.IRType))
                    {
                        expr = children[0];
                    }
                    else
                    {
                        throw new InvalidOperationException($"Cannot rebuild call {call.Target.GetType().Name}: expected {call.Arguments.Length} arguments, got {children.Length}.");
                    }

                    break;
                case (_, Fusion fusion):
                    expr = fusion;
                    break;
                case (SearchableNodeKind.FunctionCall, Function func):
                    {
                        var target = _functionMap.TryGetValue(func, out var rewritten) ? rewritten : func;
                        expr = InheritSelectedMetadata(
                            new Call(target: target, arguments: BuildFunctionCallArguments(target, children)),
                            root);
                    }

                    break;
                case (_, BaseFunction func):
                    expr = InheritSelectedMetadata(new Call(target: func, arguments: children), root);
                    break;
                case (_, Boxing boxing):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"Cannot rebuild boxing op: expected one argument, got {children.Length}.");
                    }

                    expr = MaterializeReshard(children[0], boxing, boxing.NewType, "selected boxing op");
                    break;
                case (_, IR.Distributed.ShardedView shardedView):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"Cannot rebuild sharded view op: expected one argument, got {children.Length}.");
                    }

                    expr = MaterializeReshard(children[0], shardedView, shardedView.NewType, "selected sharded view op");
                    break;
                case (_, Op op):
                    expr = InheritSelectedMetadata(new Call(target: op, arguments: children), root);
                    break;
                case (_, IR.Tuple tp):
                    expr = (BaseExpr)tp.With(fields: children);
                    break;
                case (_, IR.If @if):
                    expr = @if.With(condition: (Expr)children[^3], then: (BaseFunction)children[^2], @else: (BaseFunction)children[^1], arguments: children[..^3].ToArray());
                    break;
                default:
                    throw new NotSupportedException(root.Expr.GetType().Name);
            }

            _ = EnsureMaterializedType(expr, $"selected {root.Expr.GetType().Name}");
            _memo.Add(root, expr);
        }

        return expr;
    }

    private static BaseExpr InheritSelectedMetadata(BaseExpr expression, SearchableNode node)
        => node.MetadataSource is { } source
            ? expression.InheritMetaData(source)
            : expression;

    private BaseExpr[] BuildFunctionCallArguments(Function target, BaseExpr[] children)
    {
        var parameters = target.Parameters.ToArray();
        if (parameters.Length != children.Length)
        {
            throw new InvalidOperationException($"Cannot rebuild function call {target.Name}: expected {parameters.Length} arguments, got {children.Length}.");
        }

        var arguments = new BaseExpr[children.Length];
        for (int i = 0; i < children.Length; i++)
        {
            var parameterType = parameters[i].CheckedType;
            if (RequiresExactFunctionArgumentType(parameterType))
            {
                var argumentType = EnsureMaterializedType(children[i], $"function {target.Name} argument {i}");
                if (!EqualityComparer<IRType>.Default.Equals(argumentType, parameterType))
                {
                    throw new InvalidOperationException(
                        $"Function {target.Name} argument {i} requires {parameterType}, but the selected boundary produced " +
                        $"{argumentType} without an explicit reshard node.");
                }
            }

            arguments[i] = children[i];
        }

        return arguments;
    }

    private bool RequiresExactFunctionArgumentType(IRType targetType) => targetType switch
    {
        TensorType or DistributedType => true,
        TupleType tupleType => tupleType.Fields.Any(RequiresExactFunctionArgumentType),
        _ => false,
    };

    private BaseExpr MaterializeReshard(BaseExpr value, Op realization, IRType targetType, string context)
    {
        var valueType = EnsureMaterializedType(value, context);
        if (EqualityComparer<IRType>.Default.Equals(valueType, targetType))
        {
            return value;
        }

        if (!_materializedReshards.TryGetValue(value, out var byTargetType))
        {
            byTargetType = new Dictionary<ReshardMaterializationKey, BaseExpr>();
            _materializedReshards.Add(value, byTargetType);
        }

        var realizationKind = realization switch
        {
            Boxing => DistributedReshardRealization.Boxing,
            IR.Distributed.ShardedView => DistributedReshardRealization.ShardedView,
            _ => throw new InvalidOperationException(
                $"AutoDistributed cannot materialize {context} with non-reshard op {realization.GetType().Name}."),
        };
        var key = new ReshardMaterializationKey(targetType, realizationKind);
        if (byTargetType.TryGetValue(key, out var existing))
        {
            return existing;
        }

        var resharded = new Call(realization, value);
        if (!CompilerServices.InferenceType(resharded) || resharded.CheckedType is InvalidType)
        {
            throw new InvalidOperationException($"AutoDistributed cannot materialize {context}: cannot convert {value.CheckedType} to {targetType}.");
        }

        byTargetType.Add(key, resharded);
        return resharded;
    }

    private IRType EnsureMaterializedType(BaseExpr value, string context)
    {
        var rawType = IRHelpers.GetRawCheckedType(value);
        if (rawType is not null and not InvalidType)
        {
            return rawType;
        }

        if (rawType is InvalidType)
        {
            ClearDerivedCheckedTypes(value, new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance));
        }

        if (!CompilerServices.InferenceType(value) || value.CheckedType is InvalidType)
        {
            throw new InvalidOperationException($"AutoDistributed cannot infer materialized {context}: {DescribeMaterializedExpr(value)}.");
        }

        return value.CheckedType;
    }

    private void ClearDerivedCheckedTypes(BaseExpr value, HashSet<BaseExpr> visited)
    {
        if (!visited.Add(value))
        {
            return;
        }

        if (value is Call or IR.Tuple or IR.If)
        {
            IRHelpers.SetRawCheckedType(value, null);
        }

        var operands = value is Call call ? call.Arguments.ToArray() : value.Operands;
        foreach (var operand in operands)
        {
            ClearDerivedCheckedTypes(operand, visited);
        }
    }

    private string DescribeMaterializedExpr(BaseExpr value)
    {
        var builder = new StringBuilder();
        builder.Append($"{GetExprLabel(value)} checked_type={FormatType(IRHelpers.GetRawCheckedType(value))}");
        if (value.CheckedType is InvalidType invalidType)
        {
            builder.Append($" reason={FormatOneLine(invalidType.Reason ?? string.Empty)}");
        }

        if (value is Call call)
        {
            builder.Append($" target={GetExprLabel(call.Target)} target_type={FormatType(IRHelpers.GetRawCheckedType(call.Target))}");
            for (var i = 0; i < call.Arguments.Length; i++)
            {
                builder.Append($" arg{i}={GetExprLabel(call.Arguments[i])}:{FormatType(IRHelpers.GetRawCheckedType(call.Arguments[i]))}");
            }
        }

        return builder.ToString();
    }

    private string GetExprLabel(BaseExpr expr)
    {
        if (expr is Op op)
        {
            var property = op.DisplayProperty();
            return string.IsNullOrWhiteSpace(property)
                ? op.GetType().FullName ?? op.GetType().Name
                : $"{op.GetType().FullName}({property})";
        }

        return expr.GetType().FullName ?? expr.GetType().Name;
    }

    private string FormatType(IRType? type)
        => FormatOneLine(type?.ToString() ?? "<none>");

    private string FormatOneLine(string text)
        => text.Replace("\r", " ", StringComparison.Ordinal).Replace("\n", " ", StringComparison.Ordinal);

    private readonly record struct ReshardMaterializationKey(
        IRType TargetType,
        DistributedReshardRealization Realization);
}

internal sealed class DistributedCostEvaluateContext : Evaluator.ICostEvaluateContext
{
    public DistributedCostEvaluateContext(Op op, IRType returnType, BaseExpr[] args, CompileOptions compileOptions)
    {
        Op = op;
        ReturnType = returnType;
        Args = args;
        CompileOptions = compileOptions;
        TargetCostModel = CostModel.TargetOpCostModelUtility.GetTargetCostModel(compileOptions);
    }

    public Op Op { get; }

    public IRType ReturnType { get; }

    public BaseExpr[] Args { get; }

    public CompileOptions CompileOptions { get; }

    public CostModel.ITargetOpCostModel TargetCostModel { get; }

    public T GetArgument<T>(Op op, ParameterInfo parameter)
        where T : BaseFunction
    {
        throw new NotSupportedException();
    }

    public T GetArgumentType<T>(Op op, ParameterInfo parameter)
        where T : IRType
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return (T?)Args[parameter.Index].CheckedType ?? throw new InvalidOperationException("Run type infer first.");
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public T GetReturnType<T>()
         where T : IRType
    {
        return (T)ReturnType;
    }
}

internal sealed class PrintCostCallBack : CpSolverSolutionCallback
{
    private readonly IReadOnlyDictionary<SearchableNode, BoolVar> _vars;
    private readonly Dictionary<SearchableNode, CostModel.Cost> _costModel;
    private readonly CostModel.ITargetOpCostModel _targetCostModel;
    private readonly StreamWriter _dumpWriter;
    private readonly bool _enableDump;
    private int _count;

    public PrintCostCallBack(IReadOnlyDictionary<SearchableNode, BoolVar> vars, Dictionary<SearchableNode, CostModel.Cost> costModel, CostModel.ITargetOpCostModel targetCostModel, StreamWriter writer, bool enableDump)
    {
        _vars = vars;
        _costModel = costModel;
        _targetCostModel = targetCostModel;
        _dumpWriter = writer;
        _enableDump = enableDump;
    }

    public override void OnSolutionCallback()
    {
        if (_enableDump)
        {
            var cost = CostModel.Cost.Zero;
            foreach (var (n, v) in _vars)
            {
                if (_costModel[n] != CostModel.Cost.Zero && BooleanValue(v))
                {
                    cost += _costModel[n];
                }
            }

            _dumpWriter.WriteLine($"Solution {_count++} @ {WallTime()}:");
            _dumpWriter.WriteLine(cost.ToString());
            var breakdown = CostModel.TargetOpCostModelUtility.GetCostLatencyBreakdown(_targetCostModel, cost, null);
            _dumpWriter.WriteLine($"Latency: {breakdown.Latency}");
            _dumpWriter.WriteLine($"LatencyBreakdown: {FormatLatencyBreakdown(breakdown)}");
            _dumpWriter.Flush();
        }
    }

    private static string FormatLatencyBreakdown(CostModel.TargetCostLatencyBreakdown breakdown)
    {
        return "{" +
            $" active_blocks={breakdown.ActiveBlockCount}," +
            $" cpu={FormatDouble(breakdown.CPUCycles)}," +
            $" blocklocal={FormatDouble(breakdown.BlockLocalMemoryCycles)}," +
            $" chipglobal={FormatDouble(breakdown.ChipGlobalMemoryCycles)}," +
            $" overlap={FormatDouble(breakdown.OverlappedCycles)}," +
            $" block_sync={FormatDouble(breakdown.BlockSynchronizationCycles)}," +
            $" grid_sync={FormatDouble(breakdown.GridSynchronizationCycles)}," +
            $" comm={FormatDouble(breakdown.CommCycles)}," +
            $" other={FormatDouble(breakdown.OtherCycles)}," +
            $" latency={breakdown.Latency}" +
            " }";
    }

    private static string FormatDouble(double value)
        => value.ToString("0.###", CultureInfo.InvariantCulture);
}
