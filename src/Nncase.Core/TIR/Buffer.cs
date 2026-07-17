// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// buffer.
/// </summary>
public sealed class Buffer : Expr
{
    public Buffer(
        string name,
        DataType elemType,
        MemSpan memSpan,
        Dimension[] dimensions,
        Dimension[] strides,
        DistributedType? distributedType,
        TargetStorageEncodingSelection? storageEncoding = null)
        : base(new BaseExpr[] { memSpan }.Concat(dimensions).Concat(strides))
    {
        Name = name;
        ElemType = elemType;
        Rank = dimensions.Length;
        DistributedType = distributedType;
        StorageEncoding = storageEncoding;
    }

    public string Name { get; }

    public DataType ElemType { get; }

    /// <summary>
    /// Gets rank of the tensor: number of dimensions.
    /// </summary>
    public int Rank { get; }

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public MemSpan MemSpan => (MemSpan)Operands[0];

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public ReadOnlySpan<Dimension> Dimensions => SpanUtility.UnsafeCast<BaseExpr, Dimension>(Operands[1..(1 + Rank)]);

    /// <summary>
    /// Gets the strides.
    /// <remarks>
    /// This Strides is by elements not by bytes!
    /// </remarks>
    /// </summary>
    public ReadOnlySpan<Dimension> Strides => SpanUtility.UnsafeCast<BaseExpr, Dimension>(Operands[(1 + Rank)..(1 + Rank + Rank)]);

    public DistributedType? DistributedType { get; }

    /// <summary>
    /// Gets the selected physical storage encoding. Logical dimensions and
    /// strides remain represented independently by <see cref="Dimensions"/>
    /// and <see cref="Strides"/>.
    /// </summary>
    public TargetStorageEncodingSelection? StorageEncoding { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitBuffer(this, context);

    public Buffer With(string? name = null, DataType? elemType = null, MemSpan? memSpan = null, Dimension[]? dimensions = null, Dimension[]? strides = null, Expr[]? globalShape = null, DistributedType? distributedType = null, TargetStorageEncodingSelection? storageEncoding = null)
        => new Buffer(
            name ?? Name,
            elemType ?? ElemType,
            memSpan ?? MemSpan,
            dimensions ?? Dimensions.ToArray(),
            strides ?? Strides.ToArray(),
            distributedType ?? DistributedType,
            storageEncoding ?? StorageEncoding);

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is TIR.Buffer other
            && GetHashCode() == other.GetHashCode()
            && Name == other.Name
            && ElemType == other.ElemType
            && Rank == other.Rank
            && Equals(StorageEncoding, other.StorageEncoding)
            && Operands.SequenceEqual(other.Operands);
    }

    protected override int GetHashCodeCore() => HashCode.Combine(Name, ElemType, Rank, StorageEncoding, base.GetHashCodeCore());
}
