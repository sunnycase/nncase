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
using Nncase.IR.Affine;
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
    public SearchableNode(BaseExpr expr, IRType type, bool isBidirect = false, SearchableNodeKind kind = SearchableNodeKind.Normal)
    {
        Expr = expr;
        IRType = type;
        IsBidirect = isBidirect;
        Kind = kind;
    }

    public BaseExpr Expr { get; }

    public IRType IRType { get; }

    public bool IsBidirect { get; }

    public SearchableNodeKind Kind { get; }
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

internal sealed record ReshardPlanKey(IRType SourceType, IRType TargetType, int MaxHops);

internal sealed record BoxingCandidateKey(
    DistributedSearchGraph OwnerCluster,
    DistributedSearchGraph? OutputBucket,
    DistributedSearchGraph InputBucket,
    SearchableNode InputNode,
    IRType TargetType,
    SearchableNodeKind Kind,
    bool IsBidirect,
    DistributedSearchGraph? DependencyBucket,
    SearchableNode? DependencyNode);

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

    private readonly Dictionary<BaseExpr, DistributedSearchGraph> _reshardMemo;

    private readonly Dictionary<BaseExpr, DistributedSearchGraph> _inferedMemo;

    private readonly AdjacencyGraph<SearchableNode, CrossEdge> _rootGraph;

    private readonly DistributedSearchGraph _rootSearchGraph;

    private readonly string _moduleKind;

    private readonly bool _bidirectional;

    private readonly AutoDistributedPhase _phase;

    private readonly IDistributedCandidateProviderResolver? _candidateProviderResolver;

    private readonly Dictionary<Type, ITypeInferencer> _inferencer_cache = new Dictionary<Type, ITypeInferencer>();

    private readonly Dictionary<CandidateDiagnosticKey, int> _candidateDiagnostics = new();

    private readonly AutoDistributedProfiler _profiler;

    private readonly bool _recordCandidateDiagnostics;

    private readonly Dictionary<LeafCandidateKey, IReadOnlyList<DistributedType>> _leafCandidateMemo = new();

    private readonly Dictionary<ReshardPlanKey, IReadOnlyList<DistributedReshardPlan>> _reshardPlanMemo = new();

    private readonly Dictionary<BoxingTypeKey, IRType> _boxingTypeMemo = new();

    private readonly Dictionary<BoxingCandidateKey, (DistributedSearchGraph Bucket, SearchableNode Node)> _boxingCandidateMemo = new();

    private readonly Dictionary<TypeInferenceCacheKey, (bool Success, IRType CheckedType)> _typeInferenceMemo = new();

    private readonly Dictionary<Function, DistributedSearchGraph> _functionReturnClusters = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Function, DistributedSearchGraph> _functionRootClusters = new(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Function, Dictionary<IVar, DistributedSearchGraph>> _functionParameterClusters = new(ReferenceEqualityComparer.Instance);

    private readonly HashSet<DistributedSearchGraph> _singleChoiceClusters = new(ReferenceEqualityComparer.Instance);

    private Function? _currentFunction;

    private bool _currentFunctionIsEntry;

    private Dictionary<SearchableNode, bool>? _lastPicks;

    /// <summary>
    /// The original tensor consts that are distributed.
    /// </summary>
    private readonly Dictionary<TensorConst, TensorConst> _distributedConstSources = new(ReferenceEqualityComparer.Instance);

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

    public static bool SupportsConstAffineView(INTTTargetOptions targetOptions)
        => targetOptions.UnifiedMemoryArch && targetOptions.MemoryAccessArch == MemoryAccessArchitecture.UMA;

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
            DistributedUtility.GetLeafCandidatePolicies(tensorType, placement)
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

    public DistributedSearchGraph BuildSearchGraph(Function function)
    {
        _profiler.SetFunction(function.Name);
        return _profiler.TimeActive(() =>
        {
            DistributedSearchGraph root = null!;
            using (Nncase.IR.UserTrackingScope.Suppress())
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

            return root;
        });
    }

    private DistributedSearchGraph BuildFunctionSearchGraph(Function function, bool isEntry)
    {
        _currentFunction = function;
        _currentFunctionIsEntry = isEntry;
        try
        {
            Visit(function.Body);
            var root = isEntry ? TryInstertTerminator(function.Body) : TryAddOriginator(function.Body);
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
            _currentFunction = null;
            _currentFunctionIsEntry = false;
        }
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
                        && policies[0] is SBPSplit { Axes: [1, 3] }
                        && policies[1] is SBPSplit { Axes: [2] })));

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
                        && policies[1] is SBPSplit { Axes: [2] }
                        && policies[2] is SBPSplit { Axes: [1, 3] })));

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

        // 1. inference
        var bucketMemo = new Dictionary<IRType, DistributedSearchGraph>();
        foreach (var bucketArray in EnumerateCandidateBucketArrays(expr, isSupported, argClusters))
        {
            _profiler.Count("candidate_arg_combinations");

            string[]? candidateDesc = null;
            if (isSparseExperts)
            {
                candidateDesc = bucketArray.Select((bucket, idx) =>
                {
                    var vertex = bucket.Vertices.FirstOrDefault();
                    return vertex is null ? $"Arg {idx}: Empty" : $"Arg {idx}: {DescribeSbp(vertex.IRType)}";
                }).ToArray();

                Console.WriteLine("[AutoDistributed][SparseExperts] Candidate SBP combination:");
                foreach (var desc in candidateDesc)
                {
                    Console.WriteLine($"\t{desc}");
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
            var newExprs = _profiler.Time("build_equivalent_calls", () => BuildEquivalentCalls(expr.Target, tempArgs).ToArray());
            _profiler.Count("candidate_equivalent_calls", newExprs.Length);
            foreach (var (newExpr, used) in newExprs)
            {
                _profiler.Count("candidate_exprs");
                if (expr.Target is not Boxing && ((Call)newExpr).Arguments.AsValueEnumerable().Any(a => a.CheckedType is DistributedType dt && dt.Partial is not null))
                {
                    RecordCandidateDiagnostic(expr, bucketArray, "infer", "rejected", null, "partial argument is not allowed before boxing");
                    continue;
                }

                if (!InferCandidateType(newExpr))
                {
                    RecordCandidateDiagnostic(expr, bucketArray, "infer", "rejected", newExpr.CheckedType, "type inference returned false");
                    continue;
                }

                if (newExpr.CheckedType is InvalidType invalidType)
                {
                    RecordCandidateDiagnostic(expr, bucketArray, "infer", "rejected", invalidType, invalidType.Reason);
                    continue;
                }

                var checkType = newExpr.CheckedType;
                RecordCandidateDiagnostic(expr, bucketArray, "infer", "accepted", checkType, string.Empty);
                if (!bucketMemo.TryGetValue(checkType, out var dbucket))
                {
                    dbucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    bucketMemo.Add(checkType, dbucket);
                }

                var dnode = new SearchableNode(isSupported && newExpr is Call newCall ? newCall.Target : newExpr, checkType);
                dbucket.AddVertex(dnode);

                foreach (var ((arg, _), i) in bucketArray.Zip(used).Where(p => p.Second is true).Select((arg, i) => (arg, i)))
                {
                    _rootSearchGraph.AddEdge(new(dnode, arg.Vertices.First(), i, arg));
                }
            }
        }

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

        _inferedMemo.Add(expr, callCluster);

        if (!isSupported || isStandalone)
        {
            return default;
        }

        // 3. add bidirectional connections.
        if (Bidirectional)
        {
            foreach (var (lType, lBucket) in bucketMemo.Where(kv => kv.Key is DistributedType))
            {
                foreach (var (rType, rBucket) in bucketMemo.Where(kv => kv.Key is DistributedType distributedType && distributedType != lType))
                {
                    if (CheckBoxingTypeCached(lType, rType) is not InvalidType)
                    {
                        GetOrCreateBoxingCandidate(
                            callCluster,
                            lBucket,
                            lBucket.Vertices.First(),
                            rType,
                            isBidirect: true,
                            outputBucket: rBucket,
                            addDataEdgeToOwnerCluster: true);
                    }
                }
            }
        }

        // 4. add not infered type in search space.
        var addedBuckets = bucketMemo.Values.ToArray();

        if (expr.CheckedType is not TensorType tensorType || !IsDistributableTensorType(tensorType))
        {
            return default;
        }

        AddSupplementalReshardCandidates(expr, callCluster, addedBuckets);

        // 5. filter
        FilterByScheme(expr, callCluster);
        return default;
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

    private IReadOnlyList<DistributedReshardPlan> GetReshardPlans(IRType sourceType, IRType targetType, int maxHops = DistributedReshardPlanner.DefaultMaxHops)
    {
        var key = new ReshardPlanKey(sourceType, targetType, maxHops);
        if (_reshardPlanMemo.TryGetValue(key, out var cached))
        {
            _profiler.Count("reshard_plan_cache_hit");
            return cached;
        }

        _profiler.Count("reshard_plan_cache_miss");
        var plans = DistributedReshardPlanner.Plan(sourceType, targetType, CanBoxingType, maxHops);
        _reshardPlanMemo.Add(key, plans);
        return plans;
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
            call.CheckedType = cached.CheckedType;
            return cached.Success;
        }

        _profiler.Count("type_inference_cache_miss");
        var success = _profiler.Time("type_inference", () => call.InferenceType(_inferencer_cache));
        _typeInferenceMemo.Add(key, (success, call.CheckedType));
        return success;
    }

    private IEnumerable<DistributedSearchGraph[]> EnumerateCandidateBucketArrays(Call expr, bool isSupported, IReadOnlyList<DistributedSearchGraph> argClusters)
    {
        var providerBuckets = TryBuildProviderCandidateBucketArrays(expr, isSupported, argClusters);
        if (providerBuckets.Count > 0)
        {
            foreach (var bucketArray in providerBuckets)
            {
                yield return bucketArray;
            }

            yield break;
        }

        foreach (var combBuckets in argClusters.Select(c => c.Clusters.OfType<DistributedSearchGraph>()).CartesianProduct())
        {
            yield return combBuckets.ToArray();
        }
    }

    private IReadOnlyList<DistributedSearchGraph[]> TryBuildProviderCandidateBucketArrays(Call expr, bool isSupported, IReadOnlyList<DistributedSearchGraph> argClusters)
    {
        if (!isSupported || expr.Target is not Op op || _candidateProviderResolver is null || !_candidateProviderResolver.TryGetProvider(op, out var provider))
        {
            return Array.Empty<DistributedSearchGraph[]>();
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
            return Array.Empty<DistributedSearchGraph[]>();
        }

        var availableInputTypes = candidatesByInput
            .Select(candidates => (IReadOnlyList<IRType>)candidates.Select(candidate => candidate.Type).Distinct().ToArray())
            .ToArray();
        var returnTypes = GetProviderReturnCandidateTypes(expr.CheckedType);
        if (returnTypes.Count == 0)
        {
            _profiler.Count("candidate_provider_no_return_types");
            return Array.Empty<DistributedSearchGraph[]>();
        }

        var context = new DistributedCandidateContext(CompileOptions, TargetOptions, _moduleKind, expr, availableInputTypes);
        var bucketsByInputType = candidatesByInput
            .Select(candidates => candidates
                .GroupBy(candidate => candidate.Type)
                .ToDictionary(group => group.Key, group => group.Select(candidate => candidate.Bucket).ToArray()))
            .ToArray();
        var result = new List<DistributedSearchGraph[]>();
        var tupleCount = 0;
        foreach (var returnType in returnTypes)
        {
            if (!provider.TryGetInputTypeTuples(context, op, returnType, out var tuples) || tuples.Count == 0)
            {
                continue;
            }

            tupleCount += tuples.Count;
            foreach (var tuple in tuples)
            {
                ExpandProviderTuple(tuple, bucketsByInputType, result);
            }
        }

        _profiler.Count("candidate_provider_return_types", returnTypes.Count);
        _profiler.Count("candidate_provider_tuples", tupleCount);
        _profiler.Count("candidate_provider_bucket_arrays", result.Count);
        if (result.Count == 0)
        {
            _profiler.Count("candidate_provider_fallback");
            return Array.Empty<DistributedSearchGraph[]>();
        }

        _profiler.Count("candidate_provider_hit");
        return result;
    }

    private void ExpandProviderTuple(
        DistributedCandidateTuple tuple,
        IReadOnlyList<Dictionary<IRType, DistributedSearchGraph[]>> bucketsByInputType,
        List<DistributedSearchGraph[]> result)
    {
        if (tuple.InputTypes.Count != bucketsByInputType.Count)
        {
            return;
        }

        var bucketChoices = new DistributedSearchGraph[tuple.InputTypes.Count][];
        for (int i = 0; i < tuple.InputTypes.Count; i++)
        {
            if (!bucketsByInputType[i].TryGetValue(tuple.InputTypes[i], out var buckets) || buckets.Length == 0)
            {
                return;
            }

            bucketChoices[i] = buckets;
        }

        foreach (var combBuckets in bucketChoices.CartesianProduct())
        {
            result.Add(combBuckets.ToArray());
        }
    }

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

    private void AddSupplementalReshardCandidates(
        Call expr,
        DistributedSearchGraph callCluster,
        IReadOnlyList<DistributedSearchGraph> inferredBuckets)
    {
        var sourceBuckets = ShouldLinkAllSupplementalReshardSources(expr, inferredBuckets)
            ? inferredBuckets
            : inferredBuckets.Take(1).ToArray();

        var targetTypes = _profiler.Time("supplemental_get_target_types", () => GetLeafCandidateDistTypes(expr.CheckedTensorType));
        _profiler.Count("supplemental_target_types", targetTypes.Count);
        _profiler.Count("supplemental_source_buckets", sourceBuckets.Count);

        foreach (var targetType in targetTypes)
        {
            foreach (var sourceBucket in sourceBuckets)
            {
                var sourceNode = sourceBucket.Vertices.FirstOrDefault();
                if (sourceNode is null)
                {
                    continue;
                }

                var plans = _profiler.Time("supplemental_plan_reshard", () => GetReshardPlans(sourceNode.IRType, targetType).ToArray());
                _profiler.Count("supplemental_reshard_plans", plans.Length);
                foreach (var plan in plans)
                {
                    _profiler.Count("supplemental_reshard_steps", plan.StepTypes.Count);
                    AddSupplementalReshardPath(callCluster, sourceBucket, sourceNode, plan.StepTypes);
                }
            }
        }
    }

    private void AddSupplementalReshardPath(
        DistributedSearchGraph callCluster,
        DistributedSearchGraph sourceBucket,
        SearchableNode sourceNode,
        IReadOnlyList<IRType> stepTypes)
    {
        var inputBucket = sourceBucket;
        var inputNode = sourceNode;
        foreach (var stepType in stepTypes)
        {
            var (bucket, node) = GetOrCreateBoxingCandidate(
                callCluster,
                inputBucket,
                inputNode,
                stepType,
                addDataEdgeToOwnerCluster: true);
            inputBucket = bucket;
            inputNode = node;
        }
    }

    private bool ShouldLinkAllSupplementalReshardSources(Call expr, IReadOnlyList<DistributedSearchGraph> inferredBuckets)
        => inferredBuckets.Any(bucket => bucket.Vertices.Any(node => ContainsPartial(node.IRType)))
            || expr.Users.Any(u => u is Call call && call.Target.GetType().FullName!.Contains("CustomNTT", StringComparison.Ordinal))
            || expr.Target.GetType().FullName!.Contains("CustomNTT", StringComparison.Ordinal)
            || expr.Target.GetType().FullName!.Contains("VectorizedRoPE", StringComparison.Ordinal)
            || expr.Target.GetType().FullName!.Contains("Matmul", StringComparison.InvariantCultureIgnoreCase)
            || expr.Target is PagedAttention
            || expr.Target is Gather;

    private bool CanBoxingType(IRType inputType, IRType outputType) => CheckBoxingTypeCached(inputType, outputType) is not InvalidType;

    private (DistributedSearchGraph Bucket, SearchableNode Node) GetOrCreateBoxingCandidate(
        DistributedSearchGraph ownerCluster,
        DistributedSearchGraph inputBucket,
        SearchableNode inputNode,
        IRType targetType,
        SearchableNodeKind kind = SearchableNodeKind.Normal,
        bool isBidirect = false,
        DistributedSearchGraph? outputBucket = null,
        DistributedSearchGraph? dependencyBucket = null,
        SearchableNode? dependencyNode = null,
        bool addDataEdgeToOwnerCluster = false)
    {
        if ((dependencyBucket is null) != (dependencyNode is null))
        {
            throw new InvalidOperationException("A boxing candidate dependency must provide both bucket and node.");
        }

        var key = new BoxingCandidateKey(
            ownerCluster,
            outputBucket,
            inputBucket,
            inputNode,
            targetType,
            kind,
            isBidirect,
            dependencyBucket,
            dependencyNode);
        if (_boxingCandidateMemo.TryGetValue(key, out var existing))
        {
            return existing;
        }

        var bucket = outputBucket ?? ownerCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var node = new SearchableNode(new Boxing(targetType), targetType, isBidirect, kind);
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

        var created = (bucket, node);
        _boxingCandidateMemo.Add(key, created);
        _profiler.Count("boxing_candidate_created");
        return created;
    }

    private Unit VisitLeafFunctionCall(Call expr, Function callee)
    {
        _profiler.Count("function_calls");
        var calleeReturnCluster = _functionReturnClusters[callee];
        var actualClusters = new DistributedSearchGraph[expr.Arguments.Length];
        var formalClusters = GetFunctionParameterClusters(callee);
        var boundaryClusters = new DistributedSearchGraph[expr.Arguments.Length];
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
                boundaryClusters[i] = CreateFunctionBoundaryArgumentCluster(expr, i, actualClusters[i], formalCluster);
            }
            else
            {
                actualClusters[i] = VisitLeafArgument(ParameterKind.Input, actual, isSupported: false);
                boundaryClusters[i] = actualClusters[i];
            }
        }

        var callCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        foreach (var returnBucket in calleeReturnCluster.Clusters.OfType<DistributedSearchGraph>())
        {
            var returnNode = returnBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException($"Function {callee.Name} has an empty return candidate bucket.");
            var bucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var callNode = new SearchableNode(callee, returnNode.IRType, kind: SearchableNodeKind.FunctionCall);
            bucket.AddVertex(callNode);
            _rootSearchGraph.AddEdge(new(callNode, returnNode, HiddenFunctionDependencyIndex, returnBucket));
            for (int i = 0; i < boundaryClusters.Length; i++)
            {
                foreach (var boundaryBucket in boundaryClusters[i].Clusters.OfType<DistributedSearchGraph>())
                {
                    var boundaryNode = boundaryBucket.Vertices.FirstOrDefault()
                        ?? throw new InvalidOperationException($"Function {callee.Name} call boundary argument {i} has an empty candidate bucket.");
                    _rootSearchGraph.AddEdge(new(callNode, boundaryNode, i, boundaryBucket));
                }
            }
        }

        _inferedMemo.Add(expr, callCluster);
        FilterByScheme(expr, callCluster);
        return default;
    }

    private DistributedSearchGraph CreateFunctionBoundaryArgumentCluster(Call call, int argumentIndex, DistributedSearchGraph actualCluster, DistributedSearchGraph formalCluster)
    {
        var callTargetName = call.Target is Callable callable ? callable.Name : call.Target.GetType().Name;
        var boundaryCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        var formalBuckets = formalCluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
        var actualBuckets = actualCluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
        foreach (var formalBucket in formalBuckets)
        {
            var formalNode = formalBucket.Vertices.FirstOrDefault()
                ?? throw new InvalidOperationException($"Function call {callTargetName} formal argument {argumentIndex} has an empty candidate bucket.");
            foreach (var actualBucket in actualBuckets)
            {
                var actualNode = actualBucket.Vertices.FirstOrDefault()
                    ?? throw new InvalidOperationException($"Function call {callTargetName} actual argument {argumentIndex} has an empty candidate bucket.");
                foreach (var plan in GetFunctionBoundaryReshardPlans(actualNode.IRType, formalNode.IRType))
                {
                    var finalBucket = AddFunctionBoundaryReshardPath(boundaryCluster, actualBucket, actualNode, plan.StepTypes);
                    var finalNode = finalBucket.Vertices.First();
                    GetOrCreateBoxingCandidate(
                        boundaryCluster,
                        finalBucket,
                        finalNode,
                        formalNode.IRType,
                        kind: SearchableNodeKind.FunctionBoundaryAdapter,
                        dependencyBucket: formalBucket,
                        dependencyNode: formalNode);
                }
            }
        }

        if (boundaryCluster.VertexCount == 0)
        {
            throw new InvalidOperationException($"Function call {callTargetName} argument {argumentIndex} has no legal actual/formal distributed boundary plan.");
        }

        return boundaryCluster;
    }

    private IEnumerable<DistributedReshardPlan> GetFunctionBoundaryReshardPlans(IRType sourceType, IRType targetType)
    {
        if (EqualityComparer<IRType>.Default.Equals(sourceType, targetType))
        {
            return new[] { new DistributedReshardPlan(Array.Empty<IRType>()) };
        }

        return GetReshardPlans(sourceType, targetType);
    }

    private DistributedSearchGraph AddFunctionBoundaryReshardPath(
        DistributedSearchGraph boundaryCluster,
        DistributedSearchGraph sourceBucket,
        SearchableNode sourceNode,
        IReadOnlyList<IRType> stepTypes)
    {
        var inputBucket = sourceBucket;
        var inputNode = sourceNode;
        foreach (var stepType in stepTypes)
        {
            var (bucket, node) = GetOrCreateBoxingCandidate(
                boundaryCluster,
                inputBucket,
                inputNode,
                stepType);
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
                DistributedUtility.GetLeafCandidatePolicies(distributedType.TensorType, placement).
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
        tensorBucket.AddVertex(new SearchableNode(parameter, tensorType, kind: SearchableNodeKind.FunctionParameter));
        foreach (var dType in GetLeafCandidateDistTypes(tensorType))
        {
            var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            bucket.AddVertex(new SearchableNode(parameter, dType, kind: SearchableNodeKind.FunctionParameter));
        }

        clusters.Add(parameter, distCluster);
        _singleChoiceClusters.Add(distCluster);
        return distCluster;
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
                DistributedSearchGraph? affineViewInputBucket = null;
                foreach (var dType in GetLeafCandidateDistTypes(tensorType))
                {
                    var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    var distConst = new TensorConst(tc2.Value, dType.AxisPolicies, dType.Placement);
                    if (_phase == AutoDistributedPhase.SearchConstant)
                    {
                        _distributedConstSources.Add(distConst, tc2);
                    }

                    var dnode = new SearchableNode(distConst, dType);
                    bucket.AddVertex(dnode);

                    if (SupportsConstAffineView(TargetOptions))
                    {
                        var affineViewBucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                        var affineViewNode = new SearchableNode(
                            new AffineView(dType, AffineViewTransform.Identity(tensorType.Shape)),
                            dType);
                        affineViewBucket.AddVertex(affineViewNode);
                        affineViewInputBucket ??= CreateAffineViewInputBucket(tc2);
                        _rootSearchGraph.AddEdge(new(affineViewNode, affineViewInputBucket.Vertices.First(), 0, affineViewInputBucket));
                    }
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
                    GetOrCreateBoxingCandidate(
                        distCluster,
                        inputBucket,
                        inputNode,
                        dType);
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

    private DistributedSearchGraph CreateAffineViewInputBucket(TensorConst source)
    {
        var sourceCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);
        var sourceBucket = sourceCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        sourceBucket.AddVertex(new SearchableNode(source, source.CheckedType));
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
            var buckets = new DistributedSearchGraph[tp.Fields.Length];
            foreach (var (f, fGraph, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, Visit(f), i)))
            {
                buckets[i] = TryInstertTerminator(f).Clusters.OfType<DistributedSearchGraph>().First();
            }

            var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(buckets.Select(g => g.Vertices.First().IRType).ToArray()));
            var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            bucket.AddVertex(tpnode);
            for (int i = 0; i < tp.Fields.Length; i++)
            {
                _rootSearchGraph.AddEdge(new(tpnode, buckets[i].Vertices.First(), i, buckets[i]));
            }
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
                var buckets = new DistributedSearchGraph[tupleType.Fields.Count];
                for (int i = 0; i < tupleType.Fields.Count; i++)
                {
                    var field = IR.F.Tensors.GetItem(expr, i);
                    buckets[i] = TryInstertTerminator(field).Clusters.OfType<DistributedSearchGraph>().First();
                }

                var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(buckets.Select(g => g.Vertices.First().IRType).ToArray()));
                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucket.AddVertex(tpnode);
                for (int i = 0; i < tupleType.Fields.Count; i++)
                {
                    _rootSearchGraph.AddEdge(new(tpnode, buckets[i].Vertices.First(), i, buckets[i]));
                }
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

                var onode = new SearchableNode(new Boxing(expr.CheckedType), expr.CheckedType);
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
            }
        }

        return standCluster;
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
                            if (splitIn.Axes.Except(s.Axes).Any())
                            {
                                return new InvalidType("Not Supported Split-> Split.");
                            }
                        }

                        if (s.Axes.Except(inv.Partial.Axes).ToArray() != s.Axes)
                        {
                            partialDims.Add(i);
                        }
                    }
                }

                var ndspsIn = DistributedUtility.AxisPolicesToNDSBP(inv.AxisPolicies, inv.Placement.Rank);
                var ndspsOut = DistributedUtility.AxisPolicesToNDSBP(outv.AxisPolicies, outv.Placement.Rank);
                if (Enumerable.Range(0, ndspsIn.Count).Any(i => ndspsIn[i] is SBPSplit si && (ndspsOut[i] is SBPBroadCast || (ndspsOut[i] is SBPSplit so && so.Axes[0] != si.Axes[0]))))
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

                    row1.Cells.Add(new() { Text = $"Score: {costScoreMemo[arg.Vertex]}" });
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
        writer.WriteLine($"selected_score_sum: {pickMemo.Where(kv => kv.Value).Aggregate((UInt128)0, (sum, kv) => sum + GetScore(costScoreMemo, kv.Key))}");
        var selectedAggregateCost = pickMemo.Where(kv => kv.Value).Aggregate(CostModel.Cost.Zero, (sum, kv) => sum + (costMemo.TryGetValue(kv.Key, out var cost) ? cost : CostModel.Cost.Zero));
        writer.WriteLine($"selected_aggregate_cost: {FormatCost(selectedAggregateCost)}");
        writer.WriteLine($"selected_aggregate_latency: {FormatLatencyBreakdown(dump.TargetCostModel, selectedAggregateCost, null)}");
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
                writer.WriteLine($"    <- P{edge.InputIndex} {dump.GetNodeName(edge.Root)} {GetNodeLabel(edge.Root)} picked={dump.IsPicked(edge.Root)} score={GetScore(dump.CostScoreMemo, edge.Root)} bucket={dump.GetGraphName(dump.GetBucket(edge.Root))}");
            }
        }
    }

    private void DumpCandidate(StreamWriter writer, SearchableNode node, CostDumpContext dump, string indent, int rank)
    {
        writer.WriteLine($"{indent}[{rank}] {dump.GetNodeName(node)} picked={dump.IsPicked(node)} score={GetScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
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
                : $"{dump.GetNodeName(selected)} {GetNodeLabel(selected)} score={GetScore(dump.CostScoreMemo, selected)}";
            var best = dump.GetBestNode(edge.InputGraph);
            var bestText = best is null
                ? "<none>"
                : $"{dump.GetNodeName(best)} {GetNodeLabel(best)} score={GetScore(dump.CostScoreMemo, best)}";
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
            writer.WriteLine($"{indent}{dump.GetNodeName(node)} <cycle> bucket={dump.GetGraphName(dump.GetBucket(node))} score={GetScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
            truncated = true;
            return;
        }

        if (!emitted.Add(node))
        {
            references++;
            writer.WriteLine($"{indent}{dump.GetNodeName(node)} <ref> bucket={dump.GetGraphName(dump.GetBucket(node))} score={GetScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
            return;
        }

        if (emitted.Count > maxNodes)
        {
            writer.WriteLine($"{indent}<max-nodes node={dump.GetNodeName(node)} bucket={dump.GetGraphName(dump.GetBucket(node))}>");
            truncated = true;
            return;
        }

        writer.WriteLine($"{indent}{dump.GetNodeName(node)} bucket={dump.GetGraphName(dump.GetBucket(node))} score={GetScore(dump.CostScoreMemo, node)} expr={GetNodeLabel(node)}");
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

        return text.Split([';', ',', '|'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
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

    private bool ContainsPartial(IRType type)
    {
        var text = type.ToString();
        return text?.Contains("Partial: P", StringComparison.OrdinalIgnoreCase) == true
            || text?.Contains("SBPPartial", StringComparison.OrdinalIgnoreCase) == true;
    }

    private UInt128 GetScore(IReadOnlyDictionary<SearchableNode, UInt128> costScoreMemo, SearchableNode node)
        => costScoreMemo.TryGetValue(node, out var score) ? score : 0;

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
        var removed = new HashSet<SearchableNode>();
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
                    .ThenBy(node => node.Expr is IR.Distributed.Boxing ? 1 : 0)
                    .ThenBy(node => Array.IndexOf(vertices, node))
                    .First();

                foreach (var node in group)
                {
                    if (ReferenceEquals(node, keep))
                    {
                        continue;
                    }

                    RedirectIncomingEdges(node, keep);
                    removed.Add(node);
                }
            }

            if (removed.Count > 0)
            {
                bucket.RemoveVertexIf(removed.Contains);
            }
        }

        foreach (var node in removed)
        {
            costMemo.Remove(node);
            costScoreMemo.Remove(node);
        }

        _profiler.Count("pruned_dominated_candidates", removed.Count);
    }

    private IReadOnlyList<CrossEdge> GetOrderedOutEdges(SearchableNode node)
        => _rootSearchGraph.TryGetOutEdges(node, out var edges)
            ? edges.OrderBy(edge => edge.InputIndex).ToArray()
            : Array.Empty<CrossEdge>();

    private void RedirectIncomingEdges(SearchableNode removed, SearchableNode replacement)
    {
        var incomingEdges = _rootSearchGraph.Edges.Where(edge => ReferenceEquals(edge.Target, removed)).ToArray();
        foreach (var edge in incomingEdges)
        {
            _rootSearchGraph.AddEdge(new CrossEdge(edge.Root, replacement, edge.InputIndex, edge.InputGraph));
        }
    }

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
                        if (enode.Kind is SearchableNodeKind.FunctionBoundaryAdapter or SearchableNodeKind.TypeAdapter)
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
                            || enode.Expr is Op o && o is not Boxing)
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
            cpmodel.Minimize(LinearExpr.WeightedSum(_rootSearchGraph.Vertices.Select(n => varMemo[n]), _rootSearchGraph.Vertices.Select(n => checked((long)costScoreMemo[n])))));

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
            var newFunction = new Function(function.Name, function.ModuleKind, body, parameters, newVarMap) { Metadata = function.Metadata };
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
    private readonly Dictionary<BaseExpr, Dictionary<IRType, BaseExpr>> _materializedBoxings;
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
        _materializedBoxings = new(ReferenceEqualityComparer.Instance);
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
                case (SearchableNodeKind.FunctionBoundaryAdapter or SearchableNodeKind.TypeAdapter, _):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"{root.Kind} expects one data input, got {children.Length}.");
                    }

                    expr = MaterializeBoxing(children[0], root.IRType, $"{root.Kind} node");
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

                    expr = MaterializeBoxing(children[0], boxing.NewType, "selected boxing call");
                    break;
                case (_, Call call):
                    if (children.Length == call.Arguments.Length)
                    {
                        expr = call.With(arguments: children);
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
                        expr = new Call(target: target, arguments: BuildFunctionCallArguments(target, children));
                    }

                    break;
                case (_, BaseFunction func):
                    expr = new Call(target: func, arguments: children);
                    break;
                case (_, Boxing boxing):
                    if (children.Length != 1)
                    {
                        throw new InvalidOperationException($"Cannot rebuild boxing op: expected one argument, got {children.Length}.");
                    }

                    expr = MaterializeBoxing(children[0], boxing.NewType, "selected boxing op");
                    break;
                case (_, Op op):
                    expr = new Call(target: op, arguments: children);
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
            arguments[i] = RequiresExactFunctionArgumentType(parameterType)
                ? EnsureType(children[i], parameterType, $"function {target.Name} argument {i}")
                : children[i];
        }

        return arguments;
    }

    private bool RequiresExactFunctionArgumentType(IRType targetType) => targetType switch
    {
        TensorType or DistributedType => true,
        TupleType tupleType => tupleType.Fields.Any(RequiresExactFunctionArgumentType),
        _ => false,
    };

    private BaseExpr EnsureType(BaseExpr value, IRType targetType, string context)
    {
        return MaterializeBoxing(value, targetType, context);
    }

    private BaseExpr MaterializeBoxing(BaseExpr value, IRType targetType, string context)
    {
        var valueType = EnsureMaterializedType(value, context);
        if (EqualityComparer<IRType>.Default.Equals(valueType, targetType))
        {
            return value;
        }

        if (!_materializedBoxings.TryGetValue(value, out var byTargetType))
        {
            byTargetType = new Dictionary<IRType, BaseExpr>();
            _materializedBoxings.Add(value, byTargetType);
        }

        if (byTargetType.TryGetValue(targetType, out var existing))
        {
            return existing;
        }

        var boxed = new Call(new Boxing(targetType), value);
        if (!CompilerServices.InferenceType(boxed) || boxed.CheckedType is InvalidType)
        {
            throw new InvalidOperationException($"AutoDistributed cannot materialize {context}: cannot convert {value.CheckedType} to {targetType}.");
        }

        byTargetType.Add(targetType, boxed);
        return boxed;
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
