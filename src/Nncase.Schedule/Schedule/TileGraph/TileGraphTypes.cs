// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;
using QuikGraph;
using QuikGraph.Collections;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule.TileGraph;

[Flags]
public enum TileGridAttribute : uint
{
    None = 0,

    /// <summary>
    /// The grid is required by outside, e.g., the output of the function.
    /// </summary>
    LiveOut = 1 << 1,
}

/// <summary>
/// Execution semantics of one hierarchical schedule scope.
/// </summary>
public enum TileScopeKind
{
    /// <summary>
    /// A normal tiling scope whose domain is decomposed into loops.
    /// </summary>
    Iteration,

    /// <summary>
    /// A zero-dimensional container whose child scopes execute sequentially
    /// and retain their own independent iteration domains.
    /// </summary>
    Sequential,
}

public interface ITileable
{
    int Level { get; }

    int OpId { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// todo using isl map as domain relation, so we can get the dynamic property directly.
    /// </summary>
    DomainRelation DomainRelation { get; set; }

    /// <summary>
    /// Gets the dynamic property of the domain.
    /// note it's temporary solution.
    /// </summary>
    ImmutableArray<bool> DomainDynamic { get; }

    /// <summary>
    /// Gets the domain bounds.
    /// todo using isl map to get when use it. and we actually need to.
    /// </summary>
    ImmutableArray<Dimension> DomainBoundExprs { get; }
}

public sealed record DomainRelation(int DomainOp, int RangeOp, AffineMap Map)
{
    public DomainRelation ApplyRange(DomainRelation other)
    {
        if (RangeOp != other.DomainOp)
        {
            throw new InvalidOperationException(string.Empty);
        }

        return new DomainRelation(DomainOp, other.RangeOp, Map * other.Map);
    }

    public Isl.map ToMap()
    {
        var domains = string.Join(", ", Enumerable.Range(0, Map.Domains.Length).Select(i => $"d{i}"));
        if (Map.Symbols.Length > 0)
        {
            throw new NotSupportedException("Isl map does not support symbols yet.");
        }

        var constraints = new List<string>();
        string ConvertRange(AffineRange range, int index)
        {
            var offset = range.Offset.GetDisplayString(Map.Symbols);
            var pointMultiplicity = Map.GetPointMultiplicity(index);

            // A rectangular range map describes how a whole input tile maps
            // to an output tile. ISL applies relations to individual points:
            // (scale*d, scale*t) therefore maps one point to scale adjacent
            // points, while identity and downscale maps remain single-valued.
            if (pointMultiplicity <= 1)
            {
                return offset;
            }

            var result = $"r{index}";
            constraints.Add($"({offset}) <= {result}");
            constraints.Add($"{result} < ({offset}) + {pointMultiplicity}");
            return result;
        }

        var results = StringUtility.Join(", ", Map.Results.ToArray().Select(ConvertRange));

        var constraintClause = constraints.Count == 0 ? string.Empty : $" : {string.Join(" and ", constraints)}";
        return new Isl.map(Isl.ctx.Current, $"{{ [{domains}] -> [{results}]{constraintClause} }}");
    }

    public override string ToString() => $"Op{DomainOp} -> Op{RangeOp}: {Map}";
}

public sealed class TileGrid : ITileable
{
    public TileGrid(Grid grid, Op op, int opId, int regionOpId, IEnumerable<long> domainBounds, DomainRelation relation, Dimension[] domainBoundsExpr, IEnumerable<bool> domainDynamic, IEnumerable<IEnumerable<long>> bufferShapes, TileGridAttribute attribute)
    {
        Level = -1;
        Grid = grid;
        Op = op;
        OpId = opId;
        RegionOpId = regionOpId;
        DomainDynamic = ImmutableArray.CreateRange(domainDynamic);
        DomainRelation = relation;
        Attribute = attribute;
        DomainBounds = ImmutableArray.CreateRange(domainBounds);
        DomainBoundExprs = ImmutableArray.CreateRange(domainBoundsExpr);
        BufferShapes = ImmutableArray.CreateRange(bufferShapes.Select(x => ImmutableArray.CreateRange(x)));
        BufferDataTypes = ImmutableArray.CreateRange(grid.Accesses.ToArray().Select(access => access.Buffer.CheckedDataType));
        var domainRank = grid.Accesses.ToArray().First(access => access.IsAffine).AffineMap.Domains.Length;
        var accessMaps = new AffineMap[grid.Accesses.Length];
        for (var index = 0; index < grid.Accesses.Length; index++)
        {
            var access = grid.Accesses[index];
            if (!access.IsAffine)
            {
                accessMaps[index] = AffineMap.FromCallable((_, _) => Array.Empty<AffineRange>(), domainRank, 0);
                continue;
            }

            accessMaps[index] = AffineUtility.RestrictAccessMapToShape(access.AffineMap, BufferShapes[index].AsSpan());
        }

        AccessMaps = ImmutableArray.CreateRange(accessMaps);
        for (var index = 0; index < grid.Accesses.Length; index++)
        {
            if (grid.Accesses[index].IsAffine && AccessMaps[index].Results.Length != BufferShapes[index].Length)
            {
                throw new InvalidOperationException(
                    $"Grid {op.GetType().Name} access {index} resolves to a rank-{AccessMaps[index].Results.Length} storage map " +
                    $"for a rank-{BufferShapes[index].Length} backing buffer: {AccessMaps[index]}.");
            }
        }

        TileAxisPolicies = ImmutableArray.CreateRange(grid.TileAxisPolicies);
        DomainAxisSemantics = TileDomainAxisSemantics.FromGrid(RegionOpId, TileAxisPolicies);
        var bodyAnalysis = new GridMemoryEffectAnalysis().Analyze(grid);
        LocalAccessEffects = bodyAnalysis.Effects;
        BufferAliases = bodyAnalysis.BufferAliases;
        ReadAccessIndices = ImmutableArray.CreateRange(Enumerable.Range(0, grid.Accesses.Length).Where(index => grid.Accesses[index].IsRead));
        WriteAccessIndices = ImmutableArray.CreateRange(Enumerable.Range(0, grid.Accesses.Length).Where(index => grid.Accesses[index].IsWrite));
    }

    public int Level { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the immutable source-region identity. Schedule-only clones have a
    /// unique <see cref="OpId"/> but retain their source operation identity.
    /// </summary>
    public int RegionOpId { get; }

    public DomainRelation DomainRelation { get; set; }

    public TileGridAttribute Attribute { get; }

    public Grid Grid { get; }

    public Op Op { get; }

    public ImmutableArray<long> DomainBounds { get; }

    public ImmutableArray<Dimension> DomainBoundExprs { get; }

    public ImmutableArray<bool> DomainDynamic { get; }

    public ImmutableArray<ImmutableArray<long>> BufferShapes { get; }

    public ImmutableArray<DataType> BufferDataTypes { get; }

    public ImmutableArray<AffineMap> AccessMaps { get; }

    public ImmutableArray<GridTileAxisPolicy> TileAxisPolicies { get; }

    public TileDomainAxisSemantics DomainAxisSemantics { get; }

    public ImmutableArray<MemoryEffect> LocalAccessEffects { get; }

    public ImmutableArray<GridBufferAlias> BufferAliases { get; }

    public bool IsPureBufferView => BufferAliases.Length > 0 &&
        LocalAccessEffects.All(effect => effect.Mode == MemoryAccessMode.None);

    public ImmutableArray<int> ReadAccessIndices { get; }

    public ImmutableArray<int> WriteAccessIndices { get; }

    public bool TryGetAliasSourceAccess(int resultAccessIndex, out int sourceAccessIndex)
    {
        foreach (var alias in BufferAliases)
        {
            if (alias.ResultAccessIndex == resultAccessIndex)
            {
                sourceAccessIndex = alias.SourceAccessIndex;
                return true;
            }
        }

        sourceAccessIndex = -1;
        return false;
    }

    public AffineMap GetAccessMap(int accessIndex) => AccessMaps[accessIndex];

    public AffineMap GetWriteAccess(int outputIndex) => AccessMaps[GetWriteAccessIndex(outputIndex)];

    public int GetWriteAccessIndex(int outputIndex) => WriteAccessIndices[outputIndex];

    public long GetBufferElemSize(int i) => Grid.Accesses[i].Buffer.CheckedDataType is ReferenceType
        ? 0
        : Grid.Accesses[i].Buffer.CheckedDataType.SizeInBytes;

    public TileWorkload GetTileWorkload() => CompilerServices.GetTileWorkload(Op, new(Op, BufferShapes, BufferDataTypes));

    public override string ToString()
    {
        return $"Op{OpId}";
    }
}

[DebuggerDisplay("Op{OpId}@{Level} VertexCount = {VertexCount}, EdgeCount = {EdgeCount}")]
public sealed class TieredTileGraph : TieredAdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>>, ITileable
{
    public TieredTileGraph([NotNull] AdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>> wrappedGraph)
        : base(wrappedGraph)
    {
        OpId = -1;
        Level = -1;
        ScopeKind = TileScopeKind.Iteration;
        DomainRelation = new(-1, -1, IR.Affine.AffineMap.Identity(0));
        DomainDynamic = ImmutableArray<bool>.Empty;
        DomainBoundExprs = ImmutableArray<Dimension>.Empty;
        LoopOrder = ImmutableArray<int>.Empty;
    }

    public TieredTileGraph(
        [NotNull] TieredTileGraph parentGraph,
        int level,
        int opid,
        DomainRelation relation,
        Dimension[] domainBoundsExpr,
        IEnumerable<bool> domainDynamic,
        IEnumerable<int> loopOrder,
        TileScopeKind scopeKind)
        : base(parentGraph)
    {
        OpId = opid;
        Level = level;
        ScopeKind = scopeKind;
        DomainRelation = relation;
        DomainDynamic = ImmutableArray.CreateRange(domainDynamic);
        DomainBoundExprs = ImmutableArray.CreateRange(domainBoundsExpr);
        LoopOrder = ImmutableArray.CreateRange(loopOrder);
        if (DomainRelation.Map.Results.Length != DomainBoundExprs.Length)
        {
            throw new ArgumentException(
                $"Tile scope Op{opid}@L{level} has a rank-{DomainRelation.Map.Results.Length} domain relation " +
                $"but {DomainBoundExprs.Length} domain bounds.",
                nameof(relation));
        }

        if (DomainDynamic.Length != DomainBoundExprs.Length)
        {
            throw new ArgumentException(
                $"Tile scope Op{opid}@L{level} has {DomainBoundExprs.Length} domain bounds " +
                $"but {DomainDynamic.Length} dynamic-axis flags.",
                nameof(domainDynamic));
        }

        ValidateLoopOrder(LoopOrder, DomainBoundExprs.Length, $"Op{opid}@L{level}");

        if (ScopeKind == TileScopeKind.Sequential &&
            (DomainRelation.Map.Domains.Length != 0 ||
             DomainRelation.Map.Results.Length != 0 ||
             DomainBoundExprs.Length != 0 ||
             LoopOrder.Length != 0))
        {
            throw new ArgumentException(
                $"Sequential tile scope Op{opid}@L{level} must have a zero-dimensional domain.",
                nameof(scopeKind));
        }
    }

    public int Level { get; }

    public int OpId { get; }

    public TileScopeKind ScopeKind { get; }

    public DomainRelation DomainRelation { get; set; }

    public ImmutableArray<bool> DomainDynamic { get; }

    public ImmutableArray<Dimension> DomainBoundExprs { get; }

    /// <summary>
    /// Gets the permutation from lexical loop position to canonical domain axis.
    /// Affine relations always use canonical domain axes and are never rewritten
    /// when this order changes.
    /// </summary>
    public ImmutableArray<int> LoopOrder { get; private set; }

    public void SetLoopOrder(IEnumerable<int> loopOrder)
    {
        var order = ImmutableArray.CreateRange(loopOrder);
        ValidateLoopOrder(order, DomainBoundExprs.Length, ToString());
        LoopOrder = order;
    }

    public override string ToString() => ScopeKind == TileScopeKind.Sequential
        ? $"Sequence{OpId}@{Level}"
        : $"Op{OpId}@{Level}";

    private static void ValidateLoopOrder(ImmutableArray<int> loopOrder, int rank, string owner)
    {
        if (loopOrder.Length != rank || !loopOrder.Order().SequenceEqual(Enumerable.Range(0, rank)))
        {
            throw new ArgumentException(
                $"Tile scope {owner} requires a permutation of canonical axes [0, {rank}), got [{string.Join(", ", loopOrder)}].",
                nameof(loopOrder));
        }
    }
}
