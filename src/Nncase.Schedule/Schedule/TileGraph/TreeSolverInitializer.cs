// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Graphviz;
using static Nncase.TIR.TIRExtensions;

namespace Nncase.Schedule.TileGraph;

public sealed class TreeSolverInitializer : TreeSolverBase<IntExpr>, ITreeNodeVisitor<TreeSolverInitializer.Context, TreeSolverInitializer.InitResult>
{
    private readonly Dictionary<IntExpr, Dictionary<long, TailGeometry>> _tailGeometryMemo = new(new ReferenceEqualityComparer<IntExpr>());
    private int _tailGeometryId;

    public TreeSolverInitializer(Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, int levelCount, Solver solver, Dictionary<OpNode, OpNodeInfo<IntExpr>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<IntExpr>> levelBufferInfos, Dictionary<ITileable, DomainInfo<IntExpr>> domainDimInfos, INTTTargetOptions targetOptions)
        : base(solver, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
        BufferGraphMemo = bufferGraphMemo;
        TopLevel = levelCount;
    }

    public int TimeStamp { get; private set; }

    public IReadOnlyDictionary<TieredTileGraph, BufferGraph> BufferGraphMemo { get; }

    public int TopLevel { get; }

    public Dictionary<OpNode, Constraint[]> CoverageConstraints { get; } = new();

    public Dictionary<OpNode, TileLifetime> OpLifetimes { get; } = new();

    public static void Init(TileNode tree, Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, int levelCount, INTTTargetOptions options, out Solver solver, out Dictionary<OpNode, OpNodeInfo<IntExpr>> opNodeMemo, out Dictionary<TileNode, TileNodeInfo<IntExpr>> tileNodeMemo, out Dictionary<ITileable, DomainInfo<IntExpr>> tileableNodeMemo, out Dictionary<OpNode, Constraint[]> coverageConstraints, out Dictionary<OpNode, TileLifetime> opLifetimes)
    {
        solver = new Solver("GraphSolver");
        opNodeMemo = new Dictionary<OpNode, OpNodeInfo<IntExpr>>();
        tileNodeMemo = new Dictionary<TileNode, TileNodeInfo<IntExpr>>();
        tileableNodeMemo = new Dictionary<ITileable, DomainInfo<IntExpr>>();
        var initializer = new TreeSolverInitializer(bufferGraphMemo, levelCount, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, options);
        initializer.Visit(tree, Context.Default);
        coverageConstraints = initializer.CoverageConstraints;
        opLifetimes = initializer.OpLifetimes;
    }

    public InitResult Visit(TileNode value, Context context)
    {
        if (value.ScopeKind == TileScopeKind.Sequential)
        {
            return VisitSequentialScope(value, context);
        }

        var (pid, pvars, parentTailStates) = context;
        var loopOrder = value.LoopOrder;
        var dimsMap = GetDimsMap(value);
        if (!pvars.Any())
        {
            dimsMap.Clear();
        }

        var domainBounds = GetDomainBounds(value);
        if (domainBounds.Length != value.DomainRelation.Map.Results.Length)
        {
            throw new InvalidOperationException(
                $"Tile node Op{value.OpId}@{value.Level} has {value.DomainRelation.Map.Results.Length} domain results, but {domainBounds.Length} domain bounds.");
        }

        var reductionAxes = value.DomainAxisSemantics.ReductionAxes;
        var fixedToOne = reductionAxes
            .Select(isReduction => isReduction && value.Level > 0)
            .ToArray();

        // The backend-private accumulator is scoped by the first L0 reduction
        // loop. A spatial loop may be lexically inside that scope only when it
        // has one trip; otherwise one accumulator would alias multiple output
        // tiles. Backends can relax this contract when they expose indexed
        // accumulator state explicitly.
        if (value.Level == 0)
        {
            var firstReductionPosition = -1;
            for (int position = 0; position < loopOrder.Length; position++)
            {
                if (reductionAxes[loopOrder[position]])
                {
                    firstReductionPosition = position;
                    break;
                }
            }

            if (firstReductionPosition >= 0)
            {
                for (int position = firstReductionPosition + 1; position < loopOrder.Length; position++)
                {
                    var axis = loopOrder[position];
                    if (reductionAxes[axis])
                    {
                        continue;
                    }

                    fixedToOne[axis] = true;
                }
            }
        }

        // TileNode variables group child tiles at one hierarchy level. The
        // first structurally variable factor on an axis owns the tail; fixed
        // factors remain one and cannot silently turn into extra loop trips.
        var tileVars = domainBounds
            .Select((bound, axis) => fixedToOne[axis] || bound == 1
                ? Solver.MakeIntConst(1).Var()
                : Solver.MakeIntVar(
                    1,
                    bound,
                    TileSemanticNaming.GetTileVariableName(value, axis, TargetOptions.TargetMachineModel)))
            .ToArray();

        var tailStates = new TailCoverageState?[tileVars.Length];
        foreach (var (currentAxis, parentAxis) in dimsMap)
        {
            if ((uint)parentAxis < (uint)parentTailStates.Count)
            {
                tailStates[currentAxis] = parentTailStates[parentAxis];
            }
        }

        for (int axis = 0; axis < tileVars.Length; axis++)
        {
            if (tailStates[axis] is { } inherited)
            {
                tailStates[axis] = inherited with
                {
                    InnerFactorProduct = inherited.InnerFactorProduct * tileVars[axis],
                };
            }
            else if (!fixedToOne[axis] && domainBounds[axis] > 1)
            {
                tailStates[axis] = new(tileVars[axis], Solver.MakeIntConst(1));
            }
        }

        var forwardExtents = tileVars.Cast<IntExpr>().ToArray();
        if (!TileableNodeMemo.TryGetValue(value, out var dimInfo))
        {
            foreach (var (k, v) in dimsMap)
            {
                forwardExtents[k] *= pvars[v];
            }

            for (int i = 0; i < forwardExtents.Length; i++)
            {
                forwardExtents[i].SetRange(1, GetMaximumNominalCoverage(domainBounds[i]));
            }

            TileableNodeMemo.Add(value, new(tileVars, forwardExtents, dimsMap));
        }

        InitResult childResult;
        {
            var childContext = context with
            {
                ParentOpId = value.OpId,
                ForwardExtents = forwardExtents,
                TailCoverageStates = tailStates,
            };

            var results = new List<BufferResult>();
            var names = new List<Dictionary<int, int>>();
            var extents = new List<IntExpr[]>();
            var childDefUseMap = new BiDictionary<BufferIdentity, BufferIdentity>();
            foreach (var child in value.Children)
            {
                var res = child.Accept(this, childContext);
                results.AddRange(res.BufferResults);
                extents.AddRange(res.BackWardExtents);
                names.AddRange(res.DimsMaps);
                foreach (var (k, v) in res.DefUseMap)
                {
                    childDefUseMap.Add(k, v);
                }
            }

            childResult = new(results.ToArray(), childDefUseMap, names.ToArray(), extents.ToArray());
        }

        var backWardExtents = GetBackWardExtents(tileVars, loopOrder, childResult.DimsMaps, childResult.BackWardExtents, domainBounds);
        var tripCounts = new IntExpr[tileVars.Length + 1];
        for (int entry = 0; entry < tripCounts.Length; entry++)
        {
            tripCounts[entry] = Enumerable.Range(0, tileVars.Length)
                .Select(axis => GetTailGeometry(domainBounds[axis], backWardExtents[entry][axis]).TotalTrips)
                .Aggregate((IntExpr)Solver.MakeIntConst(1), Solver.MakeProd);
            tripCounts[entry].SetRange(1, GetDomainVolume(domainBounds, value));
        }

        // {def bid : use bid}
        var currentBufferGraph = BufferGraphMemo[value.Wrapped];
        var defUseMap = childResult.DefUseMap;
        foreach (var edge in currentBufferGraph.GetOwnedInterEdges())
        {
            defUseMap.Add(edge.Source, edge.Target);
        }

        var bufferResults = new List<BufferResult>();

        // Gather the inclusive lifetime of this lexical tile scope.
        TileLifetime? nodeLifetime = null;
        for (int i = 0; i < childResult.BufferResults.Length; i++)
        {
            var lifetime = childResult.BufferResults[i].Lifetime;
            nodeLifetime = nodeLifetime is { } existing ? existing.Union(lifetime) : lifetime;
        }

        if (nodeLifetime is null)
        {
            throw new InvalidOperationException($"Tile node Op{value.OpId}@L{value.Level} has no executable buffer lifetime.");
        }

        // each tile node have buffer place vars.
        if (!TileNodeMemo.TryGetValue(value, out var info))
        {
            var bufferInfoMap = new Dictionary<BufferIdentity, TileNodeBufferInfo<IntExpr>>();
            for (int i = 0; i < childResult.BufferResults.Length; i++)
            {
                var result = childResult.BufferResults[i];
                var curId = result.Bid;
                var isUseLocalMaterialization = RequiresUseLocalMaterialization(value, curId, defUseMap);
                if (defUseMap.ContainsValue(curId) && !isUseLocalMaterialization)
                {
                    continue;
                }

                AffineMap currentAccessMap = result.AccessMap;
                var currentLifetime = result.Lifetime;
                if (defUseMap.TryGetByKey(curId, out var sinkBIds))
                {
                    foreach (var sinkBid in sinkBIds)
                    {
                        if (Array.FindIndex(childResult.BufferResults, r => r.Bid == sinkBid) is var sinkIndex && sinkIndex != -1)
                        {
                            currentAccessMap = childResult.BufferResults[sinkIndex].AccessMap;
                            currentLifetime = currentLifetime.Union(childResult.BufferResults[sinkIndex].Lifetime);
                        }
                    }
                }

                if (!bufferInfoMap.TryGetValue(curId, out var bufferInfo))
                {
                    bufferInfo = GetBufferInfo(value, curId, currentAccessMap, nodeLifetime.Value, currentLifetime, backWardExtents, domainBounds, result.ElemSize);
                    bufferInfoMap.Add(curId, bufferInfo);
                    var isInternalDefinition = defUseMap.ContainsKey(curId);
                    var isVisibleToParent = value.Parent is TileNode;
                    if (!isUseLocalMaterialization && (!isInternalDefinition || isVisibleToParent))
                    {
                        bufferResults.Add(new(curId, currentLifetime, value.DomainRelation.Map * currentAccessMap, result.ElemSize));
                    }
                }
            }

            TileNodeMemo.Add(value, new(tripCounts, backWardExtents, defUseMap, bufferInfoMap));
        }

        return new(bufferResults.ToArray(), defUseMap, new[] { dimsMap }, new[] { backWardExtents[0] });
    }

    public InitResult Visit(OpNode value, Context context)
    {
        var (pid, pvars, parentTailStates) = context;
        var dimsMap = GetDimsMap(value);
        if (value.TileAxisPolicies.Length != value.DomainBounds.Length)
        {
            throw new InvalidOperationException(
                $"Grid for {value.Op.GetType().Name} exposes {value.TileAxisPolicies.Length} tile-axis policies for a rank-{value.DomainBounds.Length} domain.");
        }

        var tileVars = new IntVar[value.DomainBounds.Length];
        for (int i = 0; i < tileVars.Length; i++)
        {
            var domainBound = value.DomainBounds[i];
            var policy = value.TileAxisPolicies[i];
            switch (policy.ExtentKind)
            {
                case GridTileExtentKind.Search:
                    var powerOfTwoExtents = GetPowerOfTwoExtents(1, domainBound)
                        .Where(extent => extent % policy.Alignment == 0)
                        .ToArray();
                    if (powerOfTwoExtents.Length == 0)
                    {
                        throw new InvalidOperationException(
                            $"Grid Op{value.OpId} axis {i} has no power-of-two tile extent up to {domainBound} aligned to {policy.Alignment}.");
                    }

                    tileVars[i] = Solver.MakeIntVar(
                        powerOfTwoExtents[0],
                        powerOfTwoExtents[^1],
                        TileSemanticNaming.GetTileVariableName(value, i, TargetOptions.TargetMachineModel));
                    Solver.Add(Solver.MakeMemberCt(tileVars[i], powerOfTwoExtents));
                    break;
                case GridTileExtentKind.FullExtent:
                    tileVars[i] = Solver.MakeIntConst(domainBound).Var();
                    break;
                case GridTileExtentKind.Fixed:
                    if (policy.Extent > domainBound)
                    {
                        throw new InvalidOperationException(
                            $"Grid Op{value.OpId} axis {i} fixes tile extent {policy.Extent}, larger than domain bound {domainBound}.");
                    }

                    tileVars[i] = Solver.MakeIntConst(policy.Extent).Var();
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(value), policy, "Unsupported grid tile-axis policy.");
            }
        }

        // cache the primitive buffer shape and sizes.
        var accessMaps = new AffineMap[value.BufferShapes.Length];
        var elemSizes = new IntExpr[value.BufferShapes.Length];
        if (!OpNodeMemo.TryGetValue(value, out var opInfo))
        {
            var shapes = new IntExpr[value.BufferShapes.Length][];
            var sizes = new IntExpr[value.BufferShapes.Length];
            for (int a = 0; a < value.BufferShapes.Length; a++)
            {
                shapes[a] = new IntExpr[value.BufferShapes[a].Length];
                elemSizes[a] = sizes[a] = Solver.MakeIntConst(value.GetBufferElemSize(a));
                accessMaps[a] = value.AccessMaps[a];
                var converter = new AffineExprToIntExprConverter(Solver, tileVars);
                var accessMap = value.DomainRelation.Map * value.AccessMaps[a];
                for (int i = 0; i < shapes[a].Length; i++)
                {
                    // note themory solution, the custom affine map is not suitable for compute buffer size.
                    if (accessMap.Results[i].Offset is AffineConstant c)
                    {
                        shapes[a][i] = converter.Visit(accessMap.Results[i].Offset) + converter.Visit(accessMap.Results[i].Extent);
                    }
                    else
                    {
                        shapes[a][i] = converter.Visit(accessMap.Results[i].Offset);
                    }

                    sizes[a] *= shapes[a][i];
                }
            }

            opInfo = new(accessMaps, shapes, sizes);
            OpNodeMemo.Add(value, opInfo);
        }

        if (!TileableNodeMemo.TryGetValue(value, out var dimInfo))
        {
            var forwardExtents = tileVars.Cast<IntExpr>().ToArray();
            foreach (var (i, j) in dimsMap)
            {
                forwardExtents[i] *= pvars[j];
            }

            TileableNodeMemo.Add(value, new(tileVars, forwardExtents, dimsMap));

            var tailStates = new TailCoverageState?[tileVars.Length];
            foreach (var (currentAxis, parentAxis) in dimsMap)
            {
                if ((uint)parentAxis < (uint)parentTailStates.Count)
                {
                    tailStates[currentAxis] = parentTailStates[parentAxis];
                }
            }

            var constraints = new Constraint[tileVars.Length][];
            for (int axis = 0; axis < tileVars.Length; axis++)
            {
                var extent = value.DomainBounds[axis];
                if (tailStates[axis] is { } state)
                {
                    var childSpan = state.InnerFactorProduct * tileVars[axis];
                    var nominalCoverage = state.OwnerFactor * childSpan;
                    var requiredOwnerFactor = Solver.MakeDiv(Solver.MakeIntConst(extent - 1), childSpan) + 1;
                    constraints[axis] =
                    [
                        AddNamedConstraint(
                            Solver.MakeEquality(state.OwnerFactor, requiredOwnerFactor),
                            $"tail_owner_ceil[op{value.OpId},d{axis}]"),
                        AddNamedConstraint(
                            Solver.MakeLessOrEqual(childSpan, extent),
                            $"tail_child_span_le[op{value.OpId},d{axis}]"),
                        AddNamedConstraint(
                            Solver.MakeGreaterOrEqual(nominalCoverage, extent),
                            $"tail_coverage_ge[op{value.OpId},d{axis}]"),
                        AddNamedConstraint(
                            Solver.MakeLess(nominalCoverage - childSpan, extent),
                            $"tail_minimal_coverage[op{value.OpId},d{axis}]"),
                    ];
                }
                else
                {
                    constraints[axis] =
                    [
                        AddNamedConstraint(
                            Solver.MakeEquality(forwardExtents[axis], extent),
                            $"exact_leaf_coverage[op{value.OpId},d{axis}]"),
                    ];
                }
            }

            CoverageConstraints.Add(value, constraints.SelectMany(axisConstraints => axisConstraints).ToArray());
        }

        // Prepare return information and retain the operation lifetime for
        // backend-private resource contention modeling.
        OpLifetimes.Add(value, new TileLifetime(TimeStamp, TimeStamp + 2));
        var bufferResults = new List<BufferResult>();
        for (int i = 0; i < value.Grid.Accesses.Length; i++)
        {
            var access = value.Grid.Accesses[i];
            if (access.IsRead)
            {
                BufferIdentity inputBid = new(value.Wrapped, i, BufferEndpoint.Input);
                bufferResults.Add(new(inputBid, new TileLifetime(TimeStamp, TimeStamp + 1), value.DomainRelation.Map * accessMaps[i], elemSizes[i]));
            }

            if (access.IsWrite)
            {
                BufferIdentity outputBid = new(value.Wrapped, i, BufferEndpoint.Output);
                var outputElemSize = value.TryGetAliasSourceAccess(i, out _)
                    ? Solver.MakeIntConst(0)
                    : elemSizes[i];
                bufferResults.Add(new(outputBid, new TileLifetime(TimeStamp + 1, TimeStamp + 2), value.DomainRelation.Map * accessMaps[i], outputElemSize));
            }
        }

        TimeStamp += 2;

        // todo backward extents should times primtives.
        return new(bufferResults.ToArray(), new(), new[] { dimsMap }, new IntExpr[][] { tileVars.Cast<IntExpr>().ToArray() });
    }

    private InitResult VisitSequentialScope(TileNode value, Context context)
    {
        if (context.ForwardExtents.Count != 0)
        {
            throw new InvalidOperationException(
                $"Sequential tile scope {value} cannot be nested in an iteration domain. " +
                "Chip phases must own the outermost executable schedule component.");
        }

        var bufferGraph = BufferGraphMemo[value.Wrapped];
        var nonRootEdges = bufferGraph.GetOwnedInterEdges().ToArray();
        if (nonRootEdges.Length != 0)
        {
            throw new InvalidOperationException(
                $"Sequential tile scope {value} contains non-chip inter-phase values: " +
                string.Join(", ", nonRootEdges.Select(edge => $"{edge.Source}->{edge.Target}")) + ".");
        }

        if (!bufferGraph.GetOwnedRootMaterializationEdges().Any())
        {
            throw new InvalidOperationException(
                $"Sequential tile scope {value} has no chip-visible phase boundary.");
        }

        var results = new List<BufferResult>();
        var defUseMap = new BiDictionary<BufferIdentity, BufferIdentity>();
        foreach (var child in value.Children)
        {
            if (child is not TileNode phase ||
                phase.ScopeKind != TileScopeKind.Iteration ||
                phase.Level != value.Level)
            {
                throw new InvalidOperationException(
                    $"Sequential tile scope {value} requires independent iteration children at L{value.Level}, got {child}.");
            }

            var childResult = child.Accept(this, Context.Default);
            results.AddRange(childResult.BufferResults);
            foreach (var (source, target) in childResult.DefUseMap)
            {
                defUseMap.Add(source, target);
            }
        }

        var one = Solver.MakeIntConst(1);
        var emptyExtents = Array.Empty<IntExpr>();
        TileableNodeMemo.Add(
            value,
            new DomainInfo<IntExpr>(Array.Empty<IntExpr>(), Array.Empty<IntExpr>(), new Dictionary<int, int>()));
        TileNodeMemo.Add(
            value,
            new TileNodeInfo<IntExpr>(
                new[] { one },
                new[] { emptyExtents },
                defUseMap,
                new Dictionary<BufferIdentity, TileNodeBufferInfo<IntExpr>>()));

        return new(
            results.ToArray(),
            defUseMap,
            new[] { new Dictionary<int, int>() },
            new[] { emptyExtents });
    }

    private bool RequiresUseLocalMaterialization(
        TileNode tileNode,
        BufferIdentity use,
        BiDictionary<BufferIdentity, BufferIdentity> defUseMap)
    {
        if (use.IsOutput ||
            tileNode.Wrapped.IsPureBufferViewScope() ||
            !TargetOptions.TargetMachineModel.RequiresExplicitTransfer(tileNode.Level) ||
            !defUseMap.TryGetByValue(use, out var definition) ||
            !TileBufferAliasAnalysis.IsPureAliasEndpoint(definition))
        {
            return false;
        }

        var effect = use.Node.LocalAccessEffects[use.Index];
        return use.Access.BindingMode == GridBindingMode.Subview &&
            effect.Scope != MemoryAccessScope.Chip &&
            effect.Mode.HasFlag(MemoryAccessMode.Read);
    }

    private static long[] GetPowerOfTwoExtents(long min, long max)
    {
        if (min <= 0 || max < min)
        {
            throw new InvalidOperationException($"Invalid tile extent range [{min}, {max}].");
        }

        var values = new List<long>();
        var value = 1L;
        while (value <= max)
        {
            if (value >= min)
            {
                values.Add(value);
            }

            if (value > long.MaxValue / 2)
            {
                break;
            }

            value *= 2;
        }

        if (values.Count == 0)
        {
            throw new InvalidOperationException($"Tile extent range [{min}, {max}] contains no power-of-two candidate.");
        }

        return values.ToArray();
    }

    private static long[] GetDomainBounds(TileNode tileNode)
    {
        var bounds = CompilerServices.GetMaxShape(new RankedShape(tileNode.DomainBoundExprs.ToArray()))
            .Select(Convert.ToInt64)
            .ToArray();
        if (bounds.Any(bound => bound <= 0))
        {
            throw new InvalidOperationException(
                $"Tile node Op{tileNode.OpId}@{tileNode.Level} has a non-positive domain bound [{string.Join(", ", bounds)}].");
        }

        return bounds;
    }

    private static long GetDomainVolume(IReadOnlyList<long> bounds, TileNode tileNode)
    {
        try
        {
            return bounds.Aggregate(1L, checked((volume, bound) => volume * bound));
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException(
                $"Tile node Op{tileNode.OpId}@{tileNode.Level} domain volume exceeds the 64-bit solver limit.",
                ex);
        }
    }

    /// <summary>
    /// Get the backward accumulated domain extents.
    /// backWardExtents[i] contains a extents[domain rank], note the extents[0:i] is not accumulated, extents[i:] is accumulated.
    /// for example. backWardExtents[2] contains extents[3], this extents[0],extents[1] is not accumulated, extents[2] is accumulated.
    /// so backWardExtents[0] means extents[0:domain rank] is accumulated.
    /// </summary>
    private IntExpr[][] GetBackWardExtents(
        IntVar[] tileVars,
        IReadOnlyList<int> loopOrder,
        Dictionary<int, int>[] childDimsMaps,
        IntExpr[][] childBackWardExtents,
        IReadOnlyList<long> domainBounds)
    {
        var backWardExtents = new IntExpr[tileVars.Length + 1][];
        IntExpr GetSharedChildExtent(int axis)
        {
            var childExtents = new List<IntExpr>();
            for (int cid = 0; cid < childDimsMaps.Length; cid++)
            {
                var cmap = childDimsMaps[cid];
                var cextents = childBackWardExtents[cid];
                foreach (var (k, v) in cmap)
                {
                    if (axis == v)
                    {
                        childExtents.Add(cextents[k]);
                    }
                }
            }

            if (childExtents.Count == 0)
            {
                throw new InvalidOperationException($"Cannot find a child tile extent for canonical axis d{axis}.");
            }

            var canonicalExtent = childExtents[0];
            for (int childIndex = 1; childIndex < childExtents.Count; childIndex++)
            {
                AddNamedConstraint(
                    Solver.MakeEquality(canonicalExtent, childExtents[childIndex]),
                    $"shared_child_span_eq[d{axis},c{childIndex}]");
            }

            return canonicalExtent;
        }

        for (int entry = 0; entry < tileVars.Length + 1; entry++)
        {
            var extents = backWardExtents[entry] = new IntExpr[tileVars.Length];
            for (int position = 0; position < loopOrder.Count; position++)
            {
                var axis = loopOrder[position];
                var childExtent = GetSharedChildExtent(axis);
                extents[axis] = position >= entry
                    ? tileVars[axis] * childExtent
                    : childExtent;
            }

            for (int j = 0; j < extents.Length; j++)
            {
                extents[j].SetRange(1, GetMaximumNominalCoverage(domainBounds[j]));
            }
        }

        return backWardExtents;
    }

    private TileNodeBufferInfo<IntExpr> GetBufferInfo(TileNode tileNode, BufferIdentity bid, AffineMap accessMap, TileLifetime nodeLifetime, TileLifetime currentLifetime, IntExpr[][] backWardExtents, IReadOnlyList<long> domainBounds, IntExpr elemSize)
    {
        var rank = tileNode.DomainRelation.Map.Results.Length;
        var fullPos = rank + 1;
        var levelCount = tileNode.Level + 1;
        var bufferPlaces = Enumerable.Range(0, fullPos).Select(i => Array.Empty<IntExpr>()).ToArray();
        var bufferShapes = Enumerable.Range(0, fullPos).Select(i => Array.Empty<IntExpr>()).ToArray();
        var bufferSizes = new IntExpr[fullPos];
        var bufferTrips = new IntExpr[fullPos];
        var bufferTransferBytes = new IntExpr[fullPos];
        var bufferLifetimes = new TileLifetime[fullPos];
        bufferLifetimes[^1] = currentLifetime;
        for (int i = 0; i < fullPos - 1; i++)
        {
            bufferLifetimes[i] = nodeLifetime;
        }

        LoopMask bufferMask = new(0);

        var relatedAxes = Enumerable.Range(0, rank)
            .Where(axis => AccessMapDependsOnAxis(accessMap, axis))
            .ToHashSet();
        for (int position = 0; position < tileNode.LoopOrder.Length; position++)
        {
            if (relatedAxes.Contains(tileNode.LoopOrder[position]))
            {
                bufferMask.SetRelated(position);
            }
        }

        for (int pos = 0; pos < fullPos; pos++)
        {
            var subLevelPlace = bufferPlaces[pos] = new IntVar[levelCount];
            for (int sl = 0; sl < subLevelPlace.Length; sl++)
            {
                subLevelPlace[sl] = Solver.MakeBoolVar($"p[cl{tileNode.Level}, op{bid.Node.OpId}, b{bid.Index}_{bid.Endpoint}, ci{pos}, sl{sl}]");
            }

            var capacityDomainExtents = backWardExtents[pos]
                .Select((extent, axis) => Solver.MakeMin(extent, domainBounds[axis]))
                .ToArray();
            var subDomainShapes = bufferShapes[pos] = GetBufferShapes(bid, accessMap, capacityDomainExtents);

            var sizeExpr = subDomainShapes.Aggregate(elemSize, Solver.MakeProd);
            var maxBufferSize = GetMaxBufferSize(bid, elemSize);
            bufferSizes[pos] = Solver.MakeMin(sizeExpr, maxBufferSize);
            bufferSizes[pos].SetName($"size[cl{tileNode.Level}, op{bid.Node.OpId}, b{bid.Index}_{bid.Endpoint}, ci{pos}]");

            var enteredAxes = tileNode.LoopOrder
                .Take(pos)
                .Distinct()
                .ToArray();
            var enteredRelatedAxes = enteredAxes
                .Where(relatedAxes.Contains)
                .ToArray();
            var enteredRepeatedAxes = enteredAxes
                .Where(axis => !relatedAxes.Contains(axis))
                .ToArray();
            bufferTrips[pos] = enteredAxes
                .Select(axis => GetTailGeometry(domainBounds[axis], backWardExtents[pos][axis]).TotalTrips)
                .Aggregate((IntExpr)Solver.MakeIntConst(1), Solver.MakeProd);
            bufferTransferBytes[pos] = GetTailAwareTransferBytes(
                bid,
                accessMap,
                elemSize,
                domainBounds,
                backWardExtents[pos],
                enteredRelatedAxes,
                enteredRepeatedAxes);

            // note update writes in second visitor.
        }

        var bufferInfo = new TileNodeBufferInfo<IntExpr>(bufferLifetimes, accessMap, bufferPlaces, bufferShapes, bufferSizes, bufferTrips, bufferTransferBytes, bufferMask);
        return bufferInfo;
    }

    private IntExpr[] GetBufferShapes(BufferIdentity bid, AffineMap accessMap, IReadOnlyList<IntExpr> domainExtents)
    {
        var maximumShape = bid.Node.BufferShapes[bid.Index];
        if (accessMap.Results.Length != maximumShape.Length)
        {
            throw new InvalidOperationException(
                $"Tile buffer {bid} access rank {accessMap.Results.Length} does not match its physical rank {maximumShape.Length}.");
        }

        var converter = new AffineExprToIntExprConverter(Solver, domainExtents.ToArray());
        var shapes = new IntExpr[accessMap.Results.Length];
        for (int axis = 0; axis < shapes.Length; axis++)
        {
            var range = accessMap.Results[axis];
            var shape = range.Offset is AffineConstant
                ? converter.Visit(range.Offset) + converter.Visit(range.Extent)
                : converter.Visit(range.Offset);
            shapes[axis] = Solver.MakeMin(shape, maximumShape[axis]);
            shapes[axis].SetRange(0, maximumShape[axis]);
        }

        return shapes;
    }

    private IntExpr GetTailAwareTransferBytes(
        BufferIdentity bid,
        AffineMap accessMap,
        IntExpr elemSize,
        IReadOnlyList<long> domainBounds,
        IReadOnlyList<IntExpr> nominalDomainExtents,
        IReadOnlyList<int> tiledAxes,
        IReadOnlyList<int> repeatedAxes)
    {
        var regionExtents = nominalDomainExtents
            .Select((extent, axis) => Solver.MakeMin(extent, domainBounds[axis]))
            .ToArray();
        var terms = new List<IntExpr>();
        var repeatedTrips = repeatedAxes
            .Select(axis => GetTailGeometry(domainBounds[axis], nominalDomainExtents[axis]).TotalTrips)
            .Aggregate((IntExpr)Solver.MakeIntConst(1), Solver.MakeProd);

        void Enumerate(int axisIndex, IntExpr multiplicity)
        {
            if (axisIndex == tiledAxes.Count)
            {
                var regionSize = GetBufferShapes(bid, accessMap, regionExtents)
                    .Aggregate(elemSize, Solver.MakeProd);
                terms.Add(multiplicity * Solver.MakeMin(regionSize, GetMaxBufferSize(bid, elemSize)));
                return;
            }

            var axis = tiledAxes[axisIndex];
            var nominalExtent = nominalDomainExtents[axis];
            var geometry = GetTailGeometry(domainBounds[axis], nominalExtent);
            var savedExtent = regionExtents[axis];

            regionExtents[axis] = nominalExtent;
            Enumerate(axisIndex + 1, multiplicity * geometry.FullTrips);

            regionExtents[axis] = geometry.TailExtent;
            Enumerate(axisIndex + 1, multiplicity * geometry.HasTail);

            regionExtents[axis] = savedExtent;
        }

        Enumerate(0, repeatedTrips);
        return terms.Count == 1 ? terms[0] : Solver.MakeSum(terms.ToArray());
    }

    private TailGeometry GetTailGeometry(long logicalExtent, IntExpr nominalExtent)
    {
        if (!_tailGeometryMemo.TryGetValue(nominalExtent, out var byLogicalExtent))
        {
            byLogicalExtent = new();
            _tailGeometryMemo.Add(nominalExtent, byLogicalExtent);
        }

        if (byLogicalExtent.TryGetValue(logicalExtent, out var geometry))
        {
            return geometry;
        }

        var geometryId = _tailGeometryId++;
        var logical = Solver.MakeIntConst(logicalExtent);
        var fullTrips = Solver.MakeDiv(logical, nominalExtent).Var();
        fullTrips.SetName($"tail_full_trips[{geometryId}]");
        fullTrips.SetRange(0, logicalExtent);
        var tailExtent = (logical - (fullTrips * nominalExtent)).Var();
        tailExtent.SetName($"tail_extent[{geometryId}]");
        tailExtent.SetRange(0, logicalExtent);
        var hasTail = Solver.MakeIsGreaterCstVar(tailExtent, 0);
        hasTail.SetName($"tail_present[{geometryId}]");
        var totalTrips = (fullTrips + hasTail).Var();
        totalTrips.SetName($"tail_total_trips[{geometryId}]");
        totalTrips.SetRange(1, logicalExtent);
        geometry = new(fullTrips, tailExtent, hasTail, totalTrips);
        byLogicalExtent.Add(logicalExtent, geometry);
        return geometry;
    }

    private static bool AccessMapDependsOnAxis(AffineMap accessMap, int axis)
        => accessMap.Results.ToArray().Any(range =>
            AffineExprDependsOnAxis(range.Offset, axis) ||
            AffineExprDependsOnAxis(range.Extent, axis));

    private static bool AffineExprDependsOnAxis(AffineExpr expression, int axis)
        => expression switch
        {
            AffineDim dim => dim.Position == axis,
            AffineExtent extent => extent.Position == axis,
            AffineAddBinary add => AffineExprDependsOnAxis(add.Lhs, axis) || AffineExprDependsOnAxis(add.Rhs, axis),
            AffineMulBinary mul => AffineExprDependsOnAxis(mul.Lhs, axis) || AffineExprDependsOnAxis(mul.Rhs, axis),
            AffineDivBinary div => AffineExprDependsOnAxis(div.Lhs, axis) || AffineExprDependsOnAxis(div.Rhs, axis),
            AffineConstant or AffineSymbol => false,
            _ => throw new NotSupportedException($"Unsupported affine expression {expression.GetType().Name}."),
        };

    private Constraint AddNamedConstraint(Constraint constraint, string name)
    {
        constraint.SetName(name);
        Solver.Add(constraint);
        return constraint;
    }

    private static long GetMaximumNominalCoverage(long logicalExtent)
        => logicalExtent > (long.MaxValue / 2)
            ? long.MaxValue
            : checked((logicalExtent * 2) - 1);

    private static long GetMaxBufferSize(BufferIdentity bid, IntExpr elemSize)
    {
        var elemSizeVar = elemSize.Var();
        if (elemSizeVar.Min() != elemSizeVar.Max())
        {
            throw new InvalidOperationException($"Tile buffer {bid} element size must be a compile-time constant.");
        }

        try
        {
            return bid.Node.BufferShapes[bid.Index]
                .Aggregate(elemSizeVar.Max(), checked((size, extent) => size * extent));
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException(
                $"Tile buffer {bid} maximum byte size exceeds the 64-bit solver limit.",
                ex);
        }
    }

    /// <summary>
    /// each buffer with each access Maps, note the access map domain is this node's domain. extents also mapping to current node's domain.
    /// </summary>
    /// <param name="BufferResults">buffer info.</param>
    /// <param name="DefUseMap">the defuse map is used to record cache buffer in the top memory level. </param>
    /// <param name="DimsMaps">dims map.</param>
    /// <param name="BackWardExtents"> backward extents for cout the buffer size. </param>
    public sealed record InitResult(BufferResult[] BufferResults, BiDictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<int, int>[] DimsMaps, IntExpr[][] BackWardExtents)
    {
    }

    /// <summary>
    /// buffer init result.
    /// </summary>
    /// <param name="Bid">buffer id.</param>
    /// <param name="Lifetime">Buffer's inclusive execution lifetime.</param>
    /// <param name="AccessMap">access buffer relation from current node's domain, e.g. node.DomainRelation * buffer.AccessMap.</param>
    /// <param name="ElemSize">buffer size.</param>
    public sealed record BufferResult(BufferIdentity Bid, TileLifetime Lifetime, AffineMap AccessMap, IntExpr ElemSize)
    {
    }

    public sealed record TailCoverageState(IntExpr OwnerFactor, IntExpr InnerFactorProduct);

    public sealed record Context(int ParentOpId, IReadOnlyList<IntExpr> ForwardExtents, IReadOnlyList<TailCoverageState?> TailCoverageStates)
    {
        public static Context Default => new(-1, Array.Empty<IntVar>(), Array.Empty<TailCoverageState?>());
    }

    private sealed record TailGeometry(IntExpr FullTrips, IntExpr TailExtent, IntExpr HasTail, IntExpr TotalTrips);
}
