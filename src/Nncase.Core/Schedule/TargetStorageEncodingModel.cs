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
/// Symbolic information available when the target enumerates physical storage
/// encodings for one candidate buffer materialization.
/// </summary>
public sealed record TargetStorageEncodingModelContext(
    TargetMemorySpaceSpec MemorySpace,
    DataType DataType,
    ImmutableArray<IntExpr> LogicalShape,
    IntExpr LogicalBytes,
    TargetMachineModel Machine,
    Solver Solver);

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
    ImmutableArray<TargetStorageEncodingParameter> Parameters);

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
