// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR;

/// <summary>
/// Stable semantic identity of one concrete pipeline-owning TIR loop.
/// This is intentionally distinct from the reusable pipeline schedule
/// identity carried by <see cref="Schedule.PipelineRegionPlan"/>. The identity
/// includes the owning scheduled <see cref="PrimFunction"/> so it remains
/// globally unambiguous after scheduled functions are inlined into a caller.
/// </summary>
public sealed record PipelineRegionId
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PipelineRegionId"/> class.
    /// </summary>
    /// <param name="owningFunctionId">Stable identity of the scheduled function that created the loop.</param>
    /// <param name="localId">Stable function-local identity derived from scheduling provenance.</param>
    public PipelineRegionId(string owningFunctionId, string localId)
    {
        if (string.IsNullOrWhiteSpace(owningFunctionId))
        {
            throw new ArgumentException("Pipeline region owning-function identity must not be empty.", nameof(owningFunctionId));
        }

        if (owningFunctionId.Contains("/", StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "Pipeline region owning-function identity must be one path segment and cannot contain '/'.",
                nameof(owningFunctionId));
        }

        if (string.IsNullOrWhiteSpace(localId) ||
            localId.StartsWith("/", StringComparison.Ordinal) ||
            localId.EndsWith("/", StringComparison.Ordinal) ||
            localId.Contains("//", StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "Pipeline region local identity must be a non-empty path with no empty segments.",
                nameof(localId));
        }

        OwningFunctionId = owningFunctionId;
        LocalId = localId;
    }

    /// <summary>
    /// Gets the stable identity of the scheduled function that created the loop.
    /// </summary>
    public string OwningFunctionId { get; }

    /// <summary>
    /// Gets the stable function-local identity derived from scheduling provenance.
    /// </summary>
    public string LocalId { get; }

    /// <summary>
    /// Gets the globally stable manifest identity.
    /// </summary>
    public string Value => $"{OwningFunctionId}/{LocalId}";

    /// <summary>
    /// Derives the stable identity of a structured full/tail child loop.
    /// </summary>
    /// <param name="partition">The semantic loop partition.</param>
    /// <returns>The current identity for an unpartitioned loop, otherwise a partition child.</returns>
    public PipelineRegionId ForPartition(LoopPartition partition)
        => partition switch
        {
            LoopPartition.Unpartitioned => this,
            LoopPartition.Full => new(OwningFunctionId, $"{LocalId}/full"),
            LoopPartition.Tail => new(OwningFunctionId, $"{LocalId}/tail"),
            _ => throw new ArgumentOutOfRangeException(nameof(partition), partition, "Unsupported pipeline loop partition."),
        };

    /// <summary>
    /// Derives a stable descendant identity under a structured ancestor-loop boundary.
    /// </summary>
    /// <param name="boundaryOwner">The loop whose full/tail boundary owns the descendant instance.</param>
    /// <returns>An identity in the same scheduled-function namespace.</returns>
    public PipelineRegionId ForBoundary(PipelineRegionId boundaryOwner)
    {
        ArgumentNullException.ThrowIfNull(boundaryOwner);
        if (!string.Equals(OwningFunctionId, boundaryOwner.OwningFunctionId, StringComparison.Ordinal))
        {
            throw new ArgumentException(
                $"Pipeline boundary owner {boundaryOwner.Value} belongs to scheduled function " +
                $"{boundaryOwner.OwningFunctionId}, but descendant {Value} belongs to {OwningFunctionId}.",
                nameof(boundaryOwner));
        }

        return ForBoundary(boundaryOwner.LocalId);
    }

    /// <summary>
    /// Derives a stable descendant identity under an ancestor-loop boundary
    /// that does not itself own a pipeline region.
    /// </summary>
    /// <param name="stableBoundaryPath">Stable semantic path of the ancestor boundary.</param>
    /// <returns>An identity in the current scheduled-function namespace.</returns>
    public PipelineRegionId ForBoundary(string stableBoundaryPath)
    {
        if (string.IsNullOrWhiteSpace(stableBoundaryPath))
        {
            throw new ArgumentException(
                "Pipeline boundary semantic path must not be empty.",
                nameof(stableBoundaryPath));
        }

        return new(
            OwningFunctionId,
            $"{LocalId}/boundary/{Uri.EscapeDataString(stableBoundaryPath)}");
    }

    /// <inheritdoc/>
    public override string ToString() => Value;
}
