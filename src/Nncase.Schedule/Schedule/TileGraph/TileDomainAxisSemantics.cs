// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Semantic uses of one canonical tile-scope axis. Different fused source
/// regions may legitimately use the same scope axis in different ways.
/// </summary>
[Flags]
public enum TileAxisRole
{
    None = 0,
    Parallel = 1,
    Reduction = 2,
}

/// <summary>
/// Projection of one immutable source Grid's domain semantics onto a tile
/// scope. Reduction-axis mappings retain source semantic order and support an
/// eliminated or split axis by mapping it to zero or multiple scope axes.
/// </summary>
public sealed class TileRegionAxisProjection
{
    internal TileRegionAxisProjection(
        int regionOpId,
        ImmutableArray<TileAxisRole> axisRoles,
        ImmutableArray<ImmutableArray<int>> reductionAxisMappings)
    {
        RegionOpId = regionOpId;
        AxisRoles = axisRoles;
        ReductionAxisMappings = reductionAxisMappings;
        Validate();
    }

    public int RegionOpId { get; }

    public ImmutableArray<TileAxisRole> AxisRoles { get; }

    public ImmutableArray<ImmutableArray<int>> ReductionAxisMappings { get; }

    internal bool IsEquivalentTo(TileRegionAxisProjection other)
        => RegionOpId == other.RegionOpId &&
           AxisRoles.SequenceEqual(other.AxisRoles) &&
           ReductionAxisMappings.Length == other.ReductionAxisMappings.Length &&
           ReductionAxisMappings.Zip(other.ReductionAxisMappings).All(pair => pair.First.SequenceEqual(pair.Second));

    private void Validate()
    {
        for (var axis = 0; axis < AxisRoles.Length; axis++)
        {
            var role = AxisRoles[axis];
            if (role == TileAxisRole.None)
            {
                continue;
            }

            if ((role & ~(TileAxisRole.Parallel | TileAxisRole.Reduction)) != 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(AxisRoles),
                    role,
                    $"Region Op{RegionOpId} axis {axis} has an unknown tile-axis role.");
            }

            if (role == (TileAxisRole.Parallel | TileAxisRole.Reduction))
            {
                throw new InvalidOperationException(
                    $"Region Op{RegionOpId} projects parallel and reduction iteration onto the same scope axis {axis}.");
            }
        }

        foreach (var mapping in ReductionAxisMappings)
        {
            if (!mapping.SequenceEqual(mapping.Distinct().OrderBy(axis => axis)))
            {
                throw new InvalidOperationException(
                    $"Region Op{RegionOpId} reduction-axis mapping [{string.Join(",", mapping)}] must be sorted and unique.");
            }

            foreach (var axis in mapping)
            {
                if ((uint)axis >= (uint)AxisRoles.Length)
                {
                    throw new InvalidOperationException(
                        $"Region Op{RegionOpId} reduction-axis mapping references axis {axis} in a rank-{AxisRoles.Length} scope.");
                }

                if (!AxisRoles[axis].HasFlag(TileAxisRole.Reduction))
                {
                    throw new InvalidOperationException(
                        $"Region Op{RegionOpId} reduction-axis mapping references non-reduction scope axis {axis}.");
                }
            }
        }
    }
}

/// <summary>
/// Immutable domain semantics of one finalized tile-tree node. Leaf roles are
/// declared by Grid construction; parent roles are composed exactly once when
/// the mutable schedule graph is frozen into a tile tree.
/// </summary>
public sealed class TileDomainAxisSemantics
{
    private readonly ImmutableDictionary<int, TileRegionAxisProjection> _regionProjections;

    private TileDomainAxisSemantics(
        int rank,
        ImmutableDictionary<int, TileRegionAxisProjection> regionProjections)
    {
        Rank = rank;
        _regionProjections = regionProjections;
        var axisRoles = new TileAxisRole[rank];
        foreach (var projection in regionProjections.Values)
        {
            if (projection.AxisRoles.Length != rank)
            {
                throw new InvalidOperationException(
                    $"Region Op{projection.RegionOpId} has rank-{projection.AxisRoles.Length} axis semantics in a rank-{rank} scope.");
            }

            for (var axis = 0; axis < rank; axis++)
            {
                axisRoles[axis] |= projection.AxisRoles[axis];
            }
        }

        AxisRoles = ImmutableArray.CreateRange(axisRoles);
        ReductionAxes = ImmutableArray.CreateRange(
            AxisRoles.Select(role => (role & TileAxisRole.Reduction) != 0));
    }

    public int Rank { get; }

    /// <summary>
    /// Gets the union of all source-region roles on each scope axis. A mixed
    /// parallel/reduction role is valid across different fused regions.
    /// </summary>
    public ImmutableArray<TileAxisRole> AxisRoles { get; }

    public ImmutableArray<bool> ReductionAxes { get; }

    public IReadOnlyDictionary<int, TileRegionAxisProjection> RegionProjections => _regionProjections;

    public TileRegionAxisProjection GetRegionProjection(int regionOpId)
        => _regionProjections.TryGetValue(regionOpId, out var projection)
            ? projection
            : throw new KeyNotFoundException($"Tile scope does not contain source region Op{regionOpId}.");

    internal static TileDomainAxisSemantics Empty(int rank)
    {
        if (rank < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), rank, "Tile-domain rank must not be negative.");
        }

        return new(rank, ImmutableDictionary<int, TileRegionAxisProjection>.Empty);
    }

    internal static TileDomainAxisSemantics FromGrid(
        int regionOpId,
        IReadOnlyList<GridTileAxisPolicy> tileAxisPolicies)
    {
        var roles = tileAxisPolicies
            .Select(policy => policy.AxisKind switch
            {
                GridAxisKind.Parallel => TileAxisRole.Parallel,
                GridAxisKind.Reduction => TileAxisRole.Reduction,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(tileAxisPolicies),
                    policy.AxisKind,
                    $"Grid region Op{regionOpId} has an unknown axis kind."),
            })
            .ToImmutableArray();
        var reductionMappings = roles
            .Select((role, axis) => (role, axis))
            .Where(item => item.role == TileAxisRole.Reduction)
            .Select(item => ImmutableArray.Create(item.axis))
            .ToImmutableArray();
        var projection = new TileRegionAxisProjection(regionOpId, roles, reductionMappings);
        return new(
            roles.Length,
            ImmutableDictionary<int, TileRegionAxisProjection>.Empty.Add(regionOpId, projection));
    }

    internal static TileDomainAxisSemantics Compose(
        string owner,
        int rank,
        IEnumerable<(DomainRelation Relation, TileDomainAxisSemantics Semantics)> children)
    {
        var projections = ImmutableDictionary.CreateBuilder<int, TileRegionAxisProjection>();
        foreach (var (relation, semantics) in children)
        {
            if (relation.Map.Domains.Length != rank || relation.Map.Results.Length != semantics.Rank)
            {
                throw new InvalidOperationException(
                    $"Tile scope {owner} cannot compose relation {relation}: expected rank-{rank} domains and " +
                    $"rank-{semantics.Rank} results, got {relation.Map.Domains.Length} and {relation.Map.Results.Length}.");
            }

            var dependencies = relation.Map.Results
                .ToArray()
                .Select(range => GetDomainDependencies(range, rank))
                .ToArray();
            foreach (var childProjection in semantics.RegionProjections.Values)
            {
                var projected = Project(childProjection, dependencies, rank);
                if (projections.TryGetValue(projected.RegionOpId, out var existing))
                {
                    if (!existing.IsEquivalentTo(projected))
                    {
                        throw new InvalidOperationException(
                            $"Schedule clones of source region Op{projected.RegionOpId} have inconsistent axis projections in {owner}.");
                    }

                    continue;
                }

                projections.Add(projected.RegionOpId, projected);
            }
        }

        return new(rank, projections.ToImmutable());
    }

    private static TileRegionAxisProjection Project(
        TileRegionAxisProjection child,
        IReadOnlyList<ImmutableArray<int>> dependencies,
        int parentRank)
    {
        if (child.AxisRoles.Length != dependencies.Count)
        {
            throw new InvalidOperationException(
                $"Region Op{child.RegionOpId} has rank-{child.AxisRoles.Length} semantics but " +
                $"the parent relation exposes {dependencies.Count} result axes.");
        }

        var roles = new TileAxisRole[parentRank];
        for (var childAxis = 0; childAxis < child.AxisRoles.Length; childAxis++)
        {
            foreach (var parentAxis in dependencies[childAxis])
            {
                roles[parentAxis] |= child.AxisRoles[childAxis];
            }
        }

        var reductionMappings = child.ReductionAxisMappings
            .Select(mapping => mapping
                .SelectMany(childAxis => dependencies[childAxis])
                .Distinct()
                .OrderBy(axis => axis)
                .ToImmutableArray())
            .ToImmutableArray();
        return new TileRegionAxisProjection(
            child.RegionOpId,
            ImmutableArray.CreateRange(roles),
            reductionMappings);
    }

    private static ImmutableArray<int> GetDomainDependencies(AffineRange range, int domainRank)
    {
        var dependencies = new bool[domainRank];
        MarkDomainDependencies(range.Offset, dependencies);
        MarkDomainDependencies(range.Extent, dependencies);
        return dependencies
            .Select((used, axis) => (used, axis))
            .Where(item => item.used)
            .Select(item => item.axis)
            .ToImmutableArray();
    }

    private static void MarkDomainDependencies(AffineExpr expression, bool[] dependencies)
    {
        switch (expression)
        {
            case AffineDim dim:
                MarkDomainAxis(dim.Position, dependencies);
                break;
            case AffineExtent extent:
                MarkDomainAxis(extent.Position, dependencies);
                break;
            case AffineAddBinary add:
                MarkDomainDependencies(add.Lhs, dependencies);
                MarkDomainDependencies(add.Rhs, dependencies);
                break;
            case AffineMulBinary multiply:
                MarkDomainDependencies(multiply.Lhs, dependencies);
                MarkDomainDependencies(multiply.Rhs, dependencies);
                break;
            case AffineDivBinary divide:
                MarkDomainDependencies(divide.Lhs, dependencies);
                MarkDomainDependencies(divide.Rhs, dependencies);
                break;
            case AffineConstant or AffineSymbol:
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(expression),
                    expression,
                    "Unsupported affine tile-axis expression.");
        }
    }

    private static void MarkDomainAxis(int axis, bool[] dependencies)
    {
        if ((uint)axis >= (uint)dependencies.Length)
        {
            throw new InvalidOperationException(
                $"Affine tile-axis expression references domain axis {axis} in a rank-{dependencies.Length} domain.");
        }

        dependencies[axis] = true;
    }
}
