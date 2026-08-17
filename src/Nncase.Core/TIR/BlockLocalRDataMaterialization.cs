// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Shapes;

namespace Nncase.TIR;

/// <summary>
/// A canonical tensor constant together with the distribution used to derive
/// one block-local readonly shard.
/// </summary>
public sealed record DistributedTensorRDataSource
{
    public DistributedTensorRDataSource(
        TensorConst tensor,
        DistributedType distributedType)
    {
        if (tensor.CheckedTensorType != distributedType.TensorType)
        {
            throw new ArgumentException(
                $"Derived block-local rdata source type {tensor.CheckedTensorType} does not match " +
                $"its distributed type {distributedType.TensorType}.",
                nameof(distributedType));
        }

        Tensor = tensor;
        DistributedType = distributedType;
    }

    public TensorConst Tensor { get; }

    public DistributedType DistributedType { get; }
}

/// <summary>
/// Describes a block-local readonly tensor materialized from canonical
/// distributed constants. The descriptor is storage metadata, not executable
/// TIR; code generators serialize one materialization for each physical owner.
/// </summary>
public abstract class BlockLocalRDataMaterialization
{
    protected BlockLocalRDataMaterialization(TensorType localTensorType)
    {
        LocalTensorType = localTensorType;
    }

    /// <summary>
    /// Gets the fixed-capacity local tensor type stored in each owner's rdata pool.
    /// </summary>
    public TensorType LocalTensorType { get; }
}

/// <summary>
/// Materializes local shards from multiple distributed constants and
/// concatenates their fixed-capacity local tensors along one physical axis.
/// Inactive tail elements remain zero-filled.
/// </summary>
public sealed class ConcatenatedDistributedTensorRDataMaterialization : BlockLocalRDataMaterialization
{
    public ConcatenatedDistributedTensorRDataMaterialization(
        TensorType localTensorType,
        IReadOnlyList<DistributedTensorRDataSource> sources,
        int axis)
        : base(localTensorType)
    {
        if (sources.Count < 2)
        {
            throw new ArgumentException(
                "Concatenated block-local rdata requires at least two sources.",
                nameof(sources));
        }

        if (localTensorType.Shape is not RankedShape shape || axis < 0 || axis >= shape.Rank)
        {
            throw new ArgumentOutOfRangeException(
                nameof(axis),
                $"Concatenation axis {axis} is invalid for local type {localTensorType}.");
        }

        Sources = sources.ToArray();
        Axis = axis;
    }

    public IReadOnlyList<DistributedTensorRDataSource> Sources { get; }

    public int Axis { get; }
}
