// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// the memory type.
/// </summary>
[Flags]
public enum MemoryLocation
{
    /// <summary>
    /// input.
    /// </summary>
    Input = 1 << 1,

    /// <summary>
    /// output.
    /// </summary>
    Output = 1 << 2,

    /// <summary>
    /// constant data.
    /// </summary>
    Rdata = 1 << 3,

    /// <summary>
    /// block local constant data.
    /// </summary>
    BlockLocalRdata = 1 << 6,

    /// <summary>
    /// chip local constant data.
    /// </summary>
    ChipLocalRdata = 1 << 12,

    /// <summary>
    /// compute temp data.
    /// </summary>
    Data = 1 << 7,

    /// <summary>
    /// block local data.
    /// </summary>
    BlockLocalData = 1 << 9,

    /// <summary>
    /// chip local mutable data.
    /// </summary>
    ChipLocalData = 1 << 13,

    /// <summary>
    /// cache.
    /// </summary>
    Cache = 1 << 10,

    /// <summary>
    /// explicitly managed register-backed tile storage.
    /// </summary>
    Register = 1 << 14,

    /// <summary>
    /// explicitly managed block shared-memory tile storage.
    /// </summary>
    Shared = 1 << 15,

    /// <summary>
    /// base addr.
    /// </summary>
    PrivateBase = 1 << 11,
}

public sealed class PhysicalBuffer : BaseExpr
{
    public PhysicalBuffer(
        int alignment,
        Dimension size,
        MemoryLocation location,
        int hierarchy = 0,
        BlockLocalRDataMaterialization? blockLocalRDataMaterialization = null)
        : base([None.Default, size])
    {
        Alignment = alignment;
        Location = location;
        Hierarchy = hierarchy;
        BlockLocalRDataMaterialization = blockLocalRDataMaterialization;
        ValidateBlockLocalRDataMaterialization(location, blockLocalRDataMaterialization);
    }

    public PhysicalBuffer(
        int alignment,
        Expr start,
        Dimension size,
        MemoryLocation location,
        int hierarchy = 0,
        BlockLocalRDataMaterialization? blockLocalRDataMaterialization = null)
        : base([start, size])
    {
        Alignment = alignment;
        Location = location;
        Hierarchy = hierarchy;
        BlockLocalRDataMaterialization = blockLocalRDataMaterialization;
        ValidateBlockLocalRDataMaterialization(location, blockLocalRDataMaterialization);
    }

    /// <summary>
    /// Gets the start.
    /// </summary>
    public Expr Start => (Expr)Operands[0];

    /// <summary>
    /// Gets the size of bytes.
    /// </summary>
    public Dimension Size => (Dimension)Operands[1];

    /// <summary>
    /// Gets the alignment.
    /// </summary>
    public int Alignment { get; }

    /// <summary>
    /// Gets the memory location.
    /// </summary>
    public MemoryLocation Location { get; }

    /// <summary>
    /// Gets the memory hierarchy.
    /// </summary>
    public int Hierarchy { get; }

    /// <summary>
    /// Gets the optional compiler-owned recipe used to serialize this
    /// block-local readonly allocation.
    /// </summary>
    public BlockLocalRDataMaterialization? BlockLocalRDataMaterialization { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPhysicalBuffer(this, context);

    public PhysicalBuffer With(
        int? alignment = null,
        Expr? start = null,
        Dimension? size = null,
        MemoryLocation? location = null,
        int? hierarchy = null,
        BlockLocalRDataMaterialization? blockLocalRDataMaterialization = null) =>
        new(
            alignment ?? Alignment,
            start ?? Start,
            size ?? Size,
            location ?? Location,
            hierarchy ?? Hierarchy,
            blockLocalRDataMaterialization ?? BlockLocalRDataMaterialization);

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is PhysicalBuffer other &&
            GetHashCode() == other.GetHashCode() &&
            Location == other.Location &&
            ReferenceEquals(BlockLocalRDataMaterialization, other.BlockLocalRDataMaterialization) &&
            Operands.SequenceEqual(other.Operands);
    }

    protected override int GetHashCodeCore() => HashCode.Combine(Location, BlockLocalRDataMaterialization, base.GetHashCodeCore());

    private static void ValidateBlockLocalRDataMaterialization(
        MemoryLocation location,
        BlockLocalRDataMaterialization? materialization)
    {
        if (materialization is not null && location != MemoryLocation.BlockLocalRdata)
        {
            throw new ArgumentException(
                $"A block-local rdata materialization cannot back {location} storage.",
                nameof(location));
        }
    }
}
