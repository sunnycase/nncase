// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;

namespace Nncase.Schedule;

/// <summary>
/// Target-owned model for legal physical storage encodings. AutoTiling selects
/// one candidate jointly with tile sizes, placement, and block microkernels.
/// </summary>
public interface ITargetStorageEncodingModelProvider
{
    IReadOnlyList<TargetStorageEncodingCandidate> GetCandidates(TargetStorageEncodingModelContext context);
}

/// <summary>
/// Stable target-defined identity for one physical buffer storage encoding.
/// This is distinct from a logical tensor layout and from backend-private
/// warp/thread/register layouts.
/// </summary>
public readonly record struct TargetStorageEncodingId(string Value)
{
    public override string ToString() => Value;
}

/// <summary>
/// Well-known target-neutral storage encodings.
/// </summary>
public static class TargetStorageEncodingIds
{
    public static readonly TargetStorageEncodingId Linear = new("linear");
}

/// <summary>
/// Symbolic staged-allocation ownership for one physical placement. Stage
/// count is the exact one-hot expression selected by the owning lexical loop.
/// </summary>
public sealed record StagedAllocationContext
{
    public StagedAllocationContext(string channelId, IntExpr stageCount)
    {
        if (string.IsNullOrWhiteSpace(channelId))
        {
            throw new ArgumentException("Staged allocation channel identity must not be empty.", nameof(channelId));
        }

        ArgumentNullException.ThrowIfNull(stageCount);
        var stageCountVar = stageCount.Var();
        if (stageCountVar.Min() < 1 || stageCountVar.Max() > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(stageCount), $"Stage-count domain must be in [1,{int.MaxValue}], got [{stageCountVar.Min()},{stageCountVar.Max()}].");
        }

        ChannelId = channelId;
        StageCount = stageCount;
    }

    public string ChannelId { get; }

    public IntExpr StageCount { get; }
}

/// <summary>
/// Symbolic information available when the target enumerates physical storage
/// encodings for one candidate buffer materialization.
/// </summary>
public sealed record TargetStorageEncodingModelContext(
    TargetMemorySpaceSpec MemorySpace,
    DataType DataType,
    ImmutableArray<IntExpr> LogicalShape,
    IntExpr LogicalBytes,
    TargetMachineModel Machine,
    Solver Solver)
{
    /// <summary>
    /// Gets loop-selected staging information when this logical
    /// materialization crosses from a producer phase to a consumer phase.
    /// <see langword="null"/> denotes an ordinary allocation.
    /// </summary>
    public StagedAllocationContext? StagedAllocation { get; init; }
}

/// <summary>
/// One symbolic parameter of a target storage encoding.
/// </summary>
public sealed record TargetStorageEncodingParameter(string Name, IntExpr Value);

/// <summary>
/// One legal physical representation for a candidate buffer materialization.
/// </summary>
public sealed record TargetStorageEncodingCandidate(
    TargetStorageEncodingId Id,
    IntExpr IsLegal,
    IntExpr PhysicalBytes,
    int AlignmentBytes,
    IntExpr EstimatedCycles,
    ImmutableArray<TargetStorageEncodingParameter> Parameters)
{
    /// <summary>
    /// Gets the target-owned deterministic tie-break priority. Lower values
    /// are preferred only when complete schedules have identical predicted
    /// latency.
    /// </summary>
    public int SelectionPriority { get; init; }

    /// <summary>
    /// Gets the encoded byte stride between adjacent physical stages. A target
    /// must provide this expression when the model context has
    /// <see cref="TargetStorageEncodingModelContext.StagedAllocation"/>; it remains
    /// <see langword="null"/> for ordinary allocations.
    /// </summary>
    public IntExpr? StageStrideBytes { get; init; }
}

/// <summary>
/// Concrete physical storage encoding carried by the selected TIR buffer.
/// </summary>
public sealed class TargetStorageEncodingSelection : IEquatable<TargetStorageEncodingSelection>
{
    public TargetStorageEncodingSelection(
        TargetStorageEncodingId id,
        long physicalBytes,
        int alignmentBytes,
        IEnumerable<KeyValuePair<string, long>> parameters)
    {
        if (string.IsNullOrWhiteSpace(id.Value))
        {
            throw new ArgumentException("Storage encoding identity must not be empty.", nameof(id));
        }

        if (physicalBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(physicalBytes), physicalBytes, "Physical storage size must not be negative.");
        }

        if (alignmentBytes <= 0 || !System.Numerics.BitOperations.IsPow2((uint)alignmentBytes))
        {
            throw new ArgumentOutOfRangeException(nameof(alignmentBytes), alignmentBytes, "Storage alignment must be a positive power of two.");
        }

        Id = id;
        PhysicalBytes = physicalBytes;
        AlignmentBytes = alignmentBytes;
        Parameters = parameters.ToImmutableSortedDictionary(
            pair => pair.Key,
            pair => pair.Value,
            StringComparer.Ordinal);
    }

    public TargetStorageEncodingId Id { get; }

    public long PhysicalBytes { get; }

    public int AlignmentBytes { get; }

    public ImmutableSortedDictionary<string, long> Parameters { get; }

    /// <summary>
    /// Creates the complete physical staged layout using this encoding as the
    /// representation of one logical stage.
    /// </summary>
    public StagedBufferLayout CreateStagedBufferLayout(int stageCount, long stageStrideBytes)
    {
        if ((stageStrideBytes % AlignmentBytes) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stageStrideBytes),
                stageStrideBytes,
                $"Stage stride must be aligned to {AlignmentBytes} bytes for encoding {Id}.");
        }

        return new StagedBufferLayout(stageCount, PhysicalBytes, stageStrideBytes);
    }

    public bool Equals(TargetStorageEncodingSelection? other)
        => other is not null
        && Id == other.Id
        && PhysicalBytes == other.PhysicalBytes
        && AlignmentBytes == other.AlignmentBytes
        && Parameters.SequenceEqual(other.Parameters);

    public override bool Equals(object? obj) => obj is TargetStorageEncodingSelection other && Equals(other);

    public override int GetHashCode()
    {
        HashCode hash = default;
        hash.Add(Id);
        hash.Add(PhysicalBytes);
        hash.Add(AlignmentBytes);
        foreach (var parameter in Parameters)
        {
            hash.Add(parameter.Key, StringComparer.Ordinal);
            hash.Add(parameter.Value);
        }

        return hash.ToHashCode();
    }

    public override string ToString()
        => Parameters.Count == 0
            ? Id.ToString()
            : $"{Id}<{string.Join(",", Parameters.Select(pair => $"{pair.Key}={pair.Value}"))}>";
}
