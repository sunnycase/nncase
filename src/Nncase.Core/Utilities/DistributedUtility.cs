// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Utilities;

public static class DistributedUtility
{
    [Flags]
    public enum DivideFlags
    {
        None = 0,
        MaxShape = 1 << 1,
        FloorDiv = 1 << 2,
    }

    public static List<List<int>> GetHierarchyCombinations(int rank)
    {
        var allCombinations = new List<List<int>>(rank);
        for (int length = 1; length <= rank; length++)
        {
            GetCombinations(Enumerable.Range(0, rank).ToArray(), length, 0, new List<int>(), allCombinations);
        }

        return allCombinations;
    }

    public static void GetCombinations(int[] array, int length, int startIndex, List<int> current, List<List<int>> result)
    {
        if (current.Count == length)
        {
            result.Add([.. current]);
            return;
        }

        for (int i = startIndex; i < array.Length; i++)
        {
            current.Add(array[i]);
            GetCombinations(array, length, i + 1, current, result);
            current.RemoveAt(current.Count - 1);
        }
    }

    public static IReadOnlyList<IRArray<SBP>> GetLeafCandidatePolicies(
        TensorType tensorType,
        Placement placement,
        IDistributedSplitCandidateProvider splitCandidateProvider)
    {
        ArgumentNullException.ThrowIfNull(splitCandidateProvider);
        var maxShape = CompilerServices.GetMaxShape(tensorType.Shape);
        var splitsAxes = GetHierarchyCombinations(placement.Rank);
        var policies = new List<List<SBP>>();
        for (int di = 0; di < tensorType.Shape.Rank; di++)
        {
            var policy = new List<SBP>();
            for (int ti = 0; ti < splitsAxes.Count; ti++)
            {
                var axis = splitsAxes[ti];
                var divisor = axis.Select(a => placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                var dim = tensorType.Shape[di];
                if (axis.All(a => placement.Hierarchy[a] > 1) && divisor > 1 && IsDivideBy(maxShape[di], divisor, dim.IsFixed))
                {
                    var context = new DistributedSplitCandidateContext(
                        tensorType,
                        di,
                        placement,
                        axis.ToArray(),
                        GetSplitGranularity(dim, maxShape[di], divisor),
                        maxShape[di]);
                    policy.AddRange(splitCandidateProvider.GetCandidates(context));
                }
            }

            policy.Add(SBP.B);
            policies.Add(policy);
        }

        var candidates = policies.CartesianProduct().Select(policy => policy.ToArray()).Where(policy => IsDistributable(tensorType, policy, placement)).Select(policy => new IRArray<SBP>(policy)).ToArray();
        return candidates;
    }

    public static IReadOnlyList<IRArray<SBP>> GetPartialCandidateNDSBPs(DistributedType distributedType)
    {
        IRArray<SBP> ndsbp = distributedType.AxisPolicies;
        TensorType tensorType = distributedType.TensorType;
        var maxShape = CompilerServices.GetMaxShape(tensorType.Shape);
        Placement placement = distributedType.Placement;
        if (!ndsbp.Any(sbp => sbp is SBPPartial))
        {
            return Array.Empty<IRArray<SBP>>();
        }

        var candidateNdsbps = new List<SBP>[placement.Rank];
        for (int i = 0; i < placement.Rank; i++)
        {
            candidateNdsbps[i] = new List<SBP>();

            // var innerSplitedAxes = distributedType.NdSBP.Skip(i + 1).OfType<SBPSplit>().Select(sbp => sbp.Axis).ToList();
            if (ndsbp[i] is SBPPartial)
            {
                candidateNdsbps[i].Add(SBP.B);

                // note separate reduce boxing and reshard boxing.
                // for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
                // {
                //     if (placement.Hierarchy[i] > 1 && IsDivideBy(maxShape[axis], placement.Hierarchy[i]) && !innerSplitedAxes.Contains(axis))
                //     {
                //         candidateNdsbps[i].Add(SBP.SContiguous(axis));
                //     }
                // }
            }
            else
            {
                candidateNdsbps[i].Add(ndsbp[i]);
            }
        }

        return candidateNdsbps.CartesianProduct().Select(ndsbp => ndsbp.ToArray()).Where(ndsbp => IsDistributable(tensorType, ndsbp, placement)).Select(ndsbp => new IRArray<SBP>(ndsbp)).ToArray();
    }

    public static bool IsDistributable(TensorType tensorType, ReadOnlySpan<SBP> polices, Placement placement)
    {
        if (!tensorType.Shape.IsRanked)
        {
            return false;
        }

        // 1. S on different dim must have different topology axis.
        if (!IsDistributable(polices))
        {
            return false;
        }

        // 2. All shapes are divisible by the mesh.
        var maxShape = CompilerServices.GetMaxShape(tensorType.Shape);
        var divisors = GetDivisors(new DistributedType(tensorType, polices.ToArray(), placement));
        return divisors.Select((d, axis) => (d, axis)).All(p => p.d == 0 ? true : IsDivideBy(maxShape[p.axis], p.d, tensorType.Shape[p.axis].IsFixed));
    }

    public static bool IsDistributable(ReadOnlySpan<SBP> polices)
    {
        var splits = polices.ToArray().OfType<SBPSplit>().ToArray();
        if (splits.Length == 0)
        {
            return true;
        }

        if (splits.Any(split => split.HierarchyAxes.Distinct().Count() != split.HierarchyAxes.Count))
        {
            return false;
        }

        for (int i = 0; i < splits.Length - 1; i++)
        {
            for (int j = i + 1; j < splits.Length; j++)
            {
                if (splits[i].HierarchyAxes.Intersect(splits[j].HierarchyAxes).Any())
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Gets whether every placement hierarchy axis participates in a tensor
    /// split. Such a value has one distinct local component per placement
    /// owner and can therefore use compact-per-owner physical storage.
    /// </summary>
    public static bool IsFullyShardedAcrossPlacement(DistributedType distributedType)
    {
        if (distributedType.Placement.Rank == 0)
        {
            return false;
        }

        var splitHierarchyAxes = distributedType.AxisPolicies
            .OfType<SBPSplit>()
            .SelectMany(split => split.HierarchyAxes)
            .Distinct()
            .Count();
        return splitHierarchyAxes == distributedType.Placement.Rank;
    }

    public static long GetDivisor(SBP policy, Placement placement)
    {
        if (policy is SBPSplit split)
        {
            return split.HierarchyAxes.Select(a => placement.Hierarchy[a]).Aggregate(1L, (a, b) => a * b);
        }

        return 1;
    }

    public static Dimension GetLocalCapacity(
        Dimension globalExtent,
        SBPSplit split,
        Placement placement,
        DivideFlags divideFlags = DivideFlags.None)
        => GetAxisLocalCapacity(globalExtent, split, placement, divideFlags);

    public static bool TryScaleSplitUnits(
        SBPSplit split,
        long numerator,
        long denominator,
        [MaybeNullWhen(false)] out SBPSplit result)
    {
        result = null;
        if (numerator <= 0 || denominator <= 0)
        {
            return false;
        }

        var commonDivisor = GreatestCommonDivisor(numerator, denominator);
        numerator /= commonDivisor;
        denominator /= commonDivisor;
        var stages = new SplitStage[split.Stages.Count];
        for (var index = 0; index < split.Stages.Count; index++)
        {
            var stage = split.Stages[index];
            SplitDistribution distribution;
            switch (stage.Distribution)
            {
                case ContiguousSplit { Granularity: null }:
                    distribution = new ContiguousSplit();
                    break;
                case ContiguousSplit { Granularity: { } granularity }:
                    var scaledGranularity = granularity * numerator;
                    if (!Dimension.TryDivExactly(scaledGranularity, denominator, out scaledGranularity))
                    {
                        return false;
                    }

                    distribution = new ContiguousSplit(scaledGranularity);
                    break;
                case BlockCyclicSplit blockCyclic:
                    var scaledBlockSize = checked(blockCyclic.BlockSize * numerator);
                    if (scaledBlockSize % denominator != 0)
                    {
                        return false;
                    }

                    distribution = new BlockCyclicSplit(scaledBlockSize / denominator);
                    break;
                default:
                    throw new NotSupportedException(
                        $"Unsupported split distribution {stage.Distribution.GetType().Name}.");
            }

            stages[index] = new SplitStage(stage.HierarchyAxes, distribution);
        }

        result = SBP.S(stages);
        return true;
    }

    public static IReadOnlyList<int> GetDivisors(DistributedType distributedType)
    {
        var rank = distributedType.TensorType.Shape.Rank;
        var divisors = Enumerable.Repeat(0, rank).ToArray();
        for (int i = 0; i < distributedType.AxisPolicies.Count; i++)
        {
            if (distributedType.AxisPolicies[i] is SBPSplit split)
            {
                foreach (var a in split.HierarchyAxes)
                {
                    if (divisors[i] == 0)
                    {
                        divisors[i] = 1;
                    }

                    divisors[i] *= distributedType.Placement.Hierarchy[a];
                }
            }
        }

        return divisors;
    }

    public static bool TryGetDividedTensorType(DistributedType distributedType, [MaybeNullWhen(false)] out TensorType tensorType)
    {
        tensorType = null;
        var divisors = GetDivisors(distributedType);
        var maxShape = CompilerServices.GetMaxShape(distributedType.TensorType.Shape);
        tensorType = new TensorType(
            distributedType.TensorType.DType,
            maxShape.Zip(divisors).Select(p => p.Second == 0 ? p.First : Dimension.CeilDiv(p.First, p.Second)).ToArray());
        return true;
    }

    public static IRArray<HierarchyAxisPolicy> GetHierarchyAxisPolicies(
        IRArray<SBP> axisPolicies,
        int hierarchyRank)
    {
        var hierarchyPolicies = Enumerable.Repeat(
            (HierarchyAxisPolicy)HierarchyAxisBroadcast.Instance,
            hierarchyRank).ToArray();
        for (var tensorAxis = 0; tensorAxis < axisPolicies.Count; tensorAxis++)
        {
            var policy = axisPolicies[tensorAxis];
            if (policy is SBPSplit split)
            {
                for (var stageIndex = 0; stageIndex < split.Stages.Count; stageIndex++)
                {
                    var stage = split.Stages[stageIndex];
                    for (var stageAxisIndex = 0; stageAxisIndex < stage.HierarchyAxes.Count; stageAxisIndex++)
                    {
                        var hierarchyAxis = stage.HierarchyAxes[stageAxisIndex];
                        if (hierarchyAxis < 0 || hierarchyAxis >= hierarchyRank)
                        {
                            throw new ArgumentOutOfRangeException(
                                nameof(axisPolicies),
                                $"Split hierarchy axis {hierarchyAxis} is outside rank {hierarchyRank}.");
                        }

                        if (hierarchyPolicies[hierarchyAxis] is not HierarchyAxisBroadcast)
                        {
                            throw new InvalidOperationException(
                                $"Hierarchy axis {hierarchyAxis} is owned by more than one tensor-axis policy.");
                        }

                        hierarchyPolicies[hierarchyAxis] = new HierarchyAxisSplit(
                            tensorAxis,
                            stageIndex,
                            stageAxisIndex,
                            stage.Distribution);
                    }
                }
            }
            else if (policy is SBPPartial partial)
            {
                foreach (var hierarchyAxis in partial.Axes)
                {
                    if (hierarchyAxis < 0 || hierarchyAxis >= hierarchyRank)
                    {
                        throw new ArgumentOutOfRangeException(
                            nameof(axisPolicies),
                            $"Partial hierarchy axis {hierarchyAxis} is outside rank {hierarchyRank}.");
                    }

                    hierarchyPolicies[hierarchyAxis] = hierarchyPolicies[hierarchyAxis] switch
                    {
                        HierarchyAxisBroadcast => new HierarchyAxisPartial([tensorAxis], partial.Op),
                        HierarchyAxisPartial existing when existing.Op == partial.Op =>
                            existing with { TensorAxes = existing.TensorAxes.Append(tensorAxis).ToArray() },
                        _ => throw new InvalidOperationException(
                            $"Hierarchy axis {hierarchyAxis} has incompatible split/partial ownership."),
                    };
                }
            }
        }

        return hierarchyPolicies;
    }

    public static IRArray<SBP> ToTensorAxisPolicies(
        IRArray<HierarchyAxisPolicy> hierarchyPolicies,
        int tensorRank)
    {
        var policies = Enumerable.Repeat(SBP.B, tensorRank).Select(policy => (SBP)policy).ToArray();
        for (var tensorAxis = 0; tensorAxis < tensorRank; tensorAxis++)
        {
            var splitAxes = Enumerable.Range(0, hierarchyPolicies.Count)
                .Select(hierarchyAxis => (HierarchyAxis: hierarchyAxis, Policy: hierarchyPolicies[hierarchyAxis] as HierarchyAxisSplit))
                .Where(item => item.Policy?.TensorAxis == tensorAxis)
                .ToArray();
            var partialAxes = Enumerable.Range(0, hierarchyPolicies.Count)
                .Where(hierarchyAxis => hierarchyPolicies[hierarchyAxis] is HierarchyAxisPartial partial && partial.TensorAxes.Contains(tensorAxis))
                .ToArray();
            if (splitAxes.Length > 0 && partialAxes.Length > 0)
            {
                throw new InvalidOperationException(
                    $"Tensor axis {tensorAxis} cannot be both split and partial.");
            }

            if (splitAxes.Any())
            {
                var splitStages = splitAxes
                    .GroupBy(item => item.Policy!.StageIndex)
                    .OrderBy(group => group.Key)
                    .Select((group, expectedStageIndex) =>
                {
                    if (group.Key != expectedStageIndex)
                    {
                        throw new InvalidOperationException(
                            $"Tensor axis {tensorAxis} has non-contiguous split-stage indexes.");
                    }

                    var ordered = group.OrderBy(item => item.Policy!.StageAxisIndex).ToArray();
                    if (ordered.Select((item, index) => item.Policy!.StageAxisIndex == index).Any(valid => !valid))
                    {
                        throw new InvalidOperationException(
                            $"Tensor axis {tensorAxis} stage {group.Key} has non-contiguous hierarchy-axis indexes.");
                    }

                    var distribution = ordered[0].Policy!.Distribution;
                    if (ordered.Any(item => item.Policy!.Distribution != distribution))
                    {
                        throw new InvalidOperationException(
                            $"Tensor axis {tensorAxis} stage {group.Key} has inconsistent distributions.");
                    }

                    return new SplitStage(
                        ordered.Select(item => item.HierarchyAxis).ToArray(),
                        distribution);
                }).ToArray();
                policies[tensorAxis] = SBP.S(splitStages);
            }

            if (partialAxes.Any())
            {
                var operation = ((HierarchyAxisPartial)hierarchyPolicies[partialAxes[0]]).Op;
                if (partialAxes.Any(axis => ((HierarchyAxisPartial)hierarchyPolicies[axis]).Op != operation))
                {
                    throw new InvalidOperationException(
                        $"Tensor axis {tensorAxis} has inconsistent partial reduction operations.");
                }

                policies[tensorAxis] = SBP.P(partialAxes, operation);
            }
        }

        return policies;
    }

    public static List<long[]> TryGetNonUniformDividedSlice(DistributedType distributedType)
    {
        var maxShape = CompilerServices.GetMaxShape(distributedType.TensorType.Shape);
        var hierarchies = Enumerable.Range(0, maxShape.Length).Select(i => new List<int>()).ToArray();
        for (int i = 0; i < distributedType.AxisPolicies.Count; i++)
        {
            if (distributedType.AxisPolicies[i] is SBPSplit split)
            {
                hierarchies[i].AddRange(split.HierarchyAxes);
            }
        }

        var spliList = hierarchies.Select<List<int>, long[]>((divs, axis) =>
        {
            long[] dim;
            if (divs.Any())
            {
                var divsor = (int)TensorUtilities.GetProduct(divs.Select(h => distributedType.Placement.Hierarchy[h]).ToArray());
                var (res, rem) = Math.DivRem(maxShape[axis], divsor);
                if (rem == 0)
                {
                    return new[] { res };
                }

                dim = new[] { res, res + rem };
            }
            else
            {
                dim = maxShape.Skip(axis).Take(1).ToArray();
            }

            return dim;
        }).ToList();

        IEnumerable<long[]> ret = new[] { Array.Empty<long>() };
        foreach (long[] array in spliList)
        {
            ret = from seq in ret
                  from item in array
                  select seq.Concat(new[] { item }).ToArray();
        }

        return ret.ToList();
    }

    public static bool IsDivideBy(long input, int divisor, bool isFixed)
    {
        if (!isFixed || input >= divisor)
        {
            return true;
        }

        return false;
    }

    public static bool IsDivideExactly(long input, int divisor, bool isFixed = true)
    {
        if (!isFixed || (input >= divisor && input % divisor == 0))
        {
            return true;
        }

        return false;
    }

    public static bool AreSamePolicies(IRArray<SBP>? a, IRArray<SBP>? b, bool checkGranularity = true)
    {
        if (a == null && b == null)
        {
            return true;
        }

        if (a == null || b == null || a.Value.Count != b.Value.Count)
        {
            return false;
        }

        for (int i = 0; i < a.Value.Count; i++)
        {
            if (!IsSamePolicy(a.Value[i], b.Value[i], checkGranularity))
            {
                return false;
            }
        }

        return true;
    }

    public static bool IsSamePolicy(SBP a, SBP b, bool checkGranularity = true)
    {
        if (a == null || b == null)
        {
            return false;
        }

        if (a is SBPSplit splitA && b is SBPSplit splitB)
        {
            if (checkGranularity)
            {
                return a == b;
            }
            else
            {
                if (splitA.Stages.Count != splitB.Stages.Count)
                {
                    return false;
                }

                for (var index = 0; index < splitA.Stages.Count; index++)
                {
                    var stageA = splitA.Stages[index];
                    var stageB = splitB.Stages[index];
                    if (stageA.HierarchyAxes != stageB.HierarchyAxes ||
                        (stageA.Distribution, stageB.Distribution) switch
                        {
                            (ContiguousSplit, ContiguousSplit) => false,
                            (BlockCyclicSplit blockA, BlockCyclicSplit blockB) when blockA.BlockSize == blockB.BlockSize => false,
                            _ => true,
                        })
                    {
                        return false;
                    }
                }

                return true;
            }
        }
        else
        {
            return a == b;
        }
    }

    /// <summary>
    /// Validates the target-independent semantic contract of a distributed alias view.
    /// </summary>
    public static bool TryValidateShardedView(
        IRType sourceType,
        DistributedType targetType,
        [NotNullWhen(false)] out string? reason)
    {
        reason = null;
        TensorType sourceTensorType;
        switch (sourceType)
        {
            case TensorType tensorType:
                sourceTensorType = tensorType;
                break;
            case DistributedType distributedType:
                sourceTensorType = distributedType.TensorType;
                if (distributedType.Placement != targetType.Placement)
                {
                    reason = $"ShardedView requires the same placement, but got {distributedType.Placement} -> {targetType.Placement}.";
                    return false;
                }

                if (HasPartial(distributedType))
                {
                    reason = "ShardedView source cannot contain a partial value.";
                    return false;
                }

                if (!HasOnlySplitOrBroadcast(distributedType))
                {
                    reason = "ShardedView source policies must contain only Split or Broadcast.";
                    return false;
                }

                if (distributedType == targetType)
                {
                    reason = "ShardedView source and target distributed types must differ.";
                    return false;
                }

                break;
            default:
                reason = $"ShardedView expects a tensor or distributed tensor input, got {sourceType}.";
                return false;
        }

        if (sourceTensorType != targetType.TensorType)
        {
            reason = $"ShardedView input tensor type {sourceTensorType} does not match target tensor type {targetType.TensorType}.";
            return false;
        }

        if (HasPartial(targetType))
        {
            reason = "ShardedView target cannot contain a partial value.";
            return false;
        }

        if (!HasOnlySplitOrBroadcast(targetType))
        {
            reason = "ShardedView target policies must contain only Split or Broadcast.";
            return false;
        }

        return true;

        static bool HasPartial(DistributedType type)
            => type.Partial is not null || type.AxisPolicies.Any(policy => policy is SBPPartial);

        static bool HasOnlySplitOrBroadcast(DistributedType type)
            => type.AxisPolicies.All(policy => policy is SBPSplit or SBPBroadCast);
    }

    /// <summary>
    /// Returns whether every target local shard is provably a subview of the source
    /// local shard at the same placement coordinate.
    /// </summary>
    /// <remarks>
    /// This is intentionally structural: an existing split must remain identical,
    /// while a broadcast axis may become split. More general split refinements can
    /// depend on granularity arithmetic and are not considered aliases unless that
    /// containment can be proven by construction.
    /// </remarks>
    public static bool IsLocalShardSubview(DistributedType sourceType, DistributedType targetType)
    {
        if (sourceType.TensorType != targetType.TensorType ||
            sourceType.Placement != targetType.Placement ||
            sourceType.Partial is not null ||
            targetType.Partial is not null ||
            sourceType.AxisPolicies.Count != targetType.AxisPolicies.Count)
        {
            return false;
        }

        for (var tensorAxis = 0; tensorAxis < sourceType.AxisPolicies.Count; tensorAxis++)
        {
            var sourcePolicy = sourceType.AxisPolicies[tensorAxis];
            var targetPolicy = targetType.AxisPolicies[tensorAxis];
            switch (sourcePolicy, targetPolicy)
            {
                case (SBPBroadCast, SBPBroadCast or SBPSplit):
                    break;
                case (SBPSplit sourceSplit, SBPSplit targetSplit) when sourceSplit == targetSplit:
                    break;
                default:
                    return false;
            }
        }

        return true;
    }

    public static float GetDividedTensorEfficiency(DistributedType distributedType, int burstLength)
    {
        var (tiles, shape) = GetDividedTile(distributedType);
        if (tiles.Contains(0))
        {
            return 1f;
        }

        return Enumerable.Range(0, tiles.Rank).Select(i => ((int)tiles[i].FixedValue).Ranges(0, (int)shape[i].FixedValue)).CartesianProduct().Select(rgs =>
        {
            var slice = rgs.ToArray();
            var iscontiguous = TensorUtilities.IsContiguousSlice(shape.ToValueArray(), slice, out var contiguousStart);
            var size = TensorUtilities.GetProduct(tiles.ToValueArray(), contiguousStart) * distributedType.TensorType.DType.SizeInBytes;
            var (div, rem) = Math.DivRem(size, burstLength);
            return ((div * 1.0f) + ((float)rem / burstLength)) / (div + 1);
        }).Average();
    }

    public static TensorType GetDividedTensorType(DistributedType distributedType, DivideFlags divideFlags = DivideFlags.None)
    {
        var (tiles, _) = GetDividedTile(distributedType, divideFlags);
        return distributedType.TensorType with { Shape = tiles };
    }

    public static int[] GetUnraveledIndex(int index, int[] hierarchies)
    {
        int remain = index;
        var unraveledIndex = new int[hierarchies.Length];
        for (int i = unraveledIndex.Length - 1; i >= 0; i--)
        {
            var hierarchy = hierarchies[i];
            unraveledIndex[i] = remain % hierarchy;
            remain = remain / hierarchy;
        }

        return unraveledIndex;
    }

    public static (Dimension[] Offset, Dimension[] Shape) GetLocalOffsetAndShape(DistributedType distributedType, int[] shardIndex, DivideFlags divideFlags = DivideFlags.None)
        => GetLocalOffsetAndShape(distributedType, shardIndex.Select(index => (Dimension)index).ToArray(), divideFlags);

    public static (Dimension[] Offset, Dimension[] Shape) GetLocalOffsetAndShape(DistributedType distributedType, Dimension[] shardIndex, DivideFlags divideFlags = DivideFlags.None)
    {
        var descriptor = GetLocalShardDescriptor(distributedType, shardIndex, divideFlags);
        if (!descriptor.TryGetContiguousRegion(out var offset, out var shape))
        {
            throw new InvalidOperationException(
                $"Distributed type {distributedType} has a non-contiguous shard mapping. " +
                $"Use {nameof(GetLocalShardDescriptor)} instead of requesting one rectangular offset/shape region.");
        }

        return (offset, shape);
    }

    public static LocalShardDescriptor GetLocalShardDescriptor(
        DistributedType distributedType,
        int[] shardIndex,
        DivideFlags divideFlags = DivideFlags.None)
        => GetLocalShardDescriptor(
            distributedType,
            shardIndex.Select(index => (Dimension)index).ToArray(),
            divideFlags);

    public static LocalShardDescriptor GetLocalShardDescriptor(
        DistributedType distributedType,
        Dimension[] shardIndex,
        DivideFlags divideFlags = DivideFlags.None)
    {
        if (shardIndex.Length != distributedType.Placement.Rank)
        {
            throw new ArgumentException(
                $"Shard coordinate rank {shardIndex.Length} does not match placement rank {distributedType.Placement.Rank}.",
                nameof(shardIndex));
        }

        var globalShape = divideFlags.HasFlag(DivideFlags.MaxShape)
            ? CompilerServices.GetMaxShape(distributedType.TensorType.Shape).Select(dim => (Dimension)dim).ToArray()
            : distributedType.TensorType.Shape.ToArray();
        var axes = new LocalShardAxisDescriptor[distributedType.TensorType.Shape.Rank];
        var usedHierarchyAxes = new HashSet<int>();
        for (int axis = 0; axis < axes.Length; axis++)
        {
            var policy = distributedType.AxisPolicies[axis];
            if (policy is not SBPSplit split)
            {
                axes[axis] = new LocalShardAxisDescriptor(
                    globalShape[axis],
                    globalShape[axis],
                    globalShape[axis],
                    Array.Empty<LocalShardStageDescriptor>());
                continue;
            }

            var parentExtent = globalShape[axis];
            var localCapacity = parentExtent;
            var activeExtent = parentExtent;
            var stages = new LocalShardStageDescriptor[split.Stages.Count];
            for (var stageIndex = 0; stageIndex < split.Stages.Count; stageIndex++)
            {
                var stage = split.Stages[stageIndex];
                foreach (var hierarchyAxis in stage.HierarchyAxes)
                {
                    if ((uint)hierarchyAxis >= (uint)distributedType.Placement.Rank)
                    {
                        throw new InvalidOperationException(
                            $"Split stage hierarchy axis {hierarchyAxis} is outside placement rank {distributedType.Placement.Rank}.");
                    }

                    if (!usedHierarchyAxes.Add(hierarchyAxis))
                    {
                        throw new InvalidOperationException(
                            $"Placement hierarchy axis {hierarchyAxis} is assigned to more than one tensor split policy.");
                    }
                }

                var stageHierarchy = stage.HierarchyAxes
                    .Select(hierarchyAxis => distributedType.Placement.Hierarchy[hierarchyAxis])
                    .ToArray();
                var stageHierarchyStrides = TensorUtilities.GetDefaultStrides(stageHierarchy)
                    .Select(stride => (Dimension)stride)
                    .ToArray();
                var linearShardIndex = TensorUtilities.GetLinearOffset(
                    stageHierarchyStrides,
                    stage.HierarchyAxes.Select(hierarchyAxis => shardIndex[hierarchyAxis]).ToArray());
                var shardCount = checked((int)TensorUtilities.GetProduct(stageHierarchy));
                localCapacity = LocalShardStageDescriptor.GetLocalCapacity(
                    parentExtent,
                    shardCount,
                    stage.Distribution,
                    divideFlags);
                activeExtent = LocalShardStageDescriptor.GetActiveExtent(
                    parentExtent,
                    localCapacity,
                    linearShardIndex,
                    shardCount,
                    stage.Distribution);
                stages[stageIndex] = new LocalShardStageDescriptor(
                    stage,
                    parentExtent,
                    localCapacity,
                    activeExtent,
                    linearShardIndex,
                    shardCount);
                parentExtent = activeExtent;
            }

            axes[axis] = new LocalShardAxisDescriptor(
                globalShape[axis],
                GetAxisLocalCapacity(globalShape[axis], split, distributedType.Placement, divideFlags),
                activeExtent,
                stages);
        }

        return new LocalShardDescriptor(distributedType, axes);
    }

    internal static Dimension GetMaxDimension(Dimension dimension)
    {
        if (dimension.IsFixed)
        {
            return dimension;
        }

        if (dimension.Metadata.Range is { } range &&
            double.IsFinite(range.Max) &&
            range.Max >= long.MinValue &&
            range.Max <= long.MaxValue)
        {
            return checked((long)Math.Ceiling(range.Max));
        }

        return dimension;
    }

    private static Dimension GetSplitGranularity(Dimension dim, long maxDim, int divisor)
        => dim.IsFixed ? (Dimension)(int)MathUtility.CeilDiv(maxDim, divisor) : Dimension.CeilDiv(dim, divisor);

    private static bool CanUseFullLocalDim(Dimension globalDim, Dimension localDim, int shardCount)
    {
        if (!globalDim.IsFixed || !localDim.IsFixed)
        {
            return false;
        }

        var globalValue = globalDim.FixedValue;
        var localValue = localDim.FixedValue;
        return localValue > 0 && globalValue >= localValue * shardCount && globalValue % localValue == 0;
    }

    private static (RankedShape Tile, RankedShape Shape) GetDividedTile(DistributedType distributedType, DivideFlags divideFlags = DivideFlags.None)
    {
        Dimension[] shape = divideFlags.HasFlag(DivideFlags.MaxShape) ? CompilerServices.GetMaxShape(distributedType.TensorType.Shape).Select(i => (Dimension)i).ToArray() : distributedType.TensorType.Shape.ToArray();
        Dimension[] tiles = divideFlags.HasFlag(DivideFlags.MaxShape) ? CompilerServices.GetMaxShape(distributedType.TensorType.Shape).Select(i => (Dimension)i).ToArray() : distributedType.TensorType.Shape.ToArray();
        for (var d = 0; d < shape.Length; d++)
        {
            if (distributedType.AxisPolicies.Count > d && distributedType.AxisPolicies[d] is SBPSplit split)
            {
                tiles[d] = GetAxisLocalCapacity(shape[d], split, distributedType.Placement, divideFlags);
            }
        }

        return (new(tiles), new(shape));
    }

    private static Dimension GetAxisLocalCapacity(
        Dimension globalExtent,
        SBPSplit split,
        Placement placement,
        DivideFlags divideFlags)
    {
        var capacity = globalExtent;
        foreach (var stage in split.Stages)
        {
            var shardCount = stage.HierarchyAxes.Aggregate(
                1,
                (product, hierarchyAxis) => checked(product * placement.Hierarchy[hierarchyAxis]));
            capacity = LocalShardStageDescriptor.GetLocalCapacity(
                capacity,
                shardCount,
                stage.Distribution,
                divideFlags);
        }

        return capacity;
    }

    private static long GreatestCommonDivisor(long lhs, long rhs)
    {
        while (rhs != 0)
        {
            (lhs, rhs) = (rhs, lhs % rhs);
        }

        return lhs;
    }
}
