// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

public partial class Tensor<T>
{
    /// <summary>
    /// Cast and copy to array.
    /// </summary>
    /// <returns>Casted array.</returns>
    public T[] ToArray()
    {
        if (IsContiguous)
        {
            var array = new T[Length];
            Buffer.CopyTo(array);
            return array;
        }
        else
        {
            var array = new T[TensorUtilities.GetProduct(Dimensions)];
            CopyTo(new Tensor<T>(array, Dimensions));
            return array;
        }
    }

    /// <summary>
    /// Cast to a scalar.
    /// </summary>
    /// <returns>Casted scalar.</returns>
    public T ToScalar()
    {
        if (Length != 1)
        {
            throw new InvalidOperationException("This tensor is not a scalar.");
        }

        return Buffer.Span[0];
    }

    /// <summary>
    /// Create the view from a <see cref="Tensor{T}"/>.
    /// </summary>
    /// <returns> tensor view. </returns>
    public override Tensor<T> View(ReadOnlySpan<long> starts, ReadOnlySpan<long> shape)
    {
        if (starts.Length != shape.Length || starts.Length != Dimensions.Length)
        {
            throw new ArgumentOutOfRangeException("starts", "the starts and shape must be equal to this tensor rank.");
        }

        var start = (int)TensorUtilities.GetLinearOffset(Strides, starts);
        var size = (int)TensorUtilities.GetSize(shape, Strides, 1);
        size = Math.Min(size, Buffer.Length - start);
        var subBuffer = Buffer.Slice(start, size);
        return new Tensor<T>(subBuffer, shape, Strides);
    }

    public override Tensor Transpose(ReadOnlySpan<long> perm)
    {
        var permArr = perm.ToInts();
        if (permArr.Length != Rank || permArr.Any(x => x < 0 || x >= Rank) || permArr.Distinct().Count() != Rank)
        {
            throw new ArgumentException("Permutation must contain every tensor axis exactly once", nameof(perm));
        }

        var destDimensions = Enumerable.Range(0, Rank).Select(i => Dimensions[permArr[i]]).ToArray();
        var destStrides = TensorUtilities.GetDefaultStrides(destDimensions);

        if (permArr.SequenceEqual(Enumerable.Range(0, Rank)) && IsContiguous)
        {
            return Clone();
        }

        var newBuffer = new T[checked((int)TensorUtilities.GetProduct(destDimensions))];
        if (newBuffer.Length == 0)
        {
            return new Tensor<T>(newBuffer, destDimensions, destStrides);
        }

        var sourceStrides = permArr.Select(axis => Strides[axis]).ToArray();
        var elementsPerRange = Math.Max(64 * 1024, (4 * 1024 * 1024) / Unsafe.SizeOf<T>());
        if (newBuffer.Length <= elementsPerRange || Environment.ProcessorCount == 1)
        {
            TransposeRange(Buffer, newBuffer, destDimensions, sourceStrides, 0, newBuffer.Length);
        }
        else
        {
            Parallel.ForEach(
                Partitioner.Create(0, newBuffer.Length, elementsPerRange),
                range => TransposeRange(Buffer, newBuffer, destDimensions, sourceStrides, range.Item1, range.Item2));
        }

        return new Tensor<T>(newBuffer, destDimensions, destStrides);
    }

    public override Tensor<T> Squeeze(params long[] axes)
    {
        var dimensions = Enumerable.Range(0, Rank).Where(i =>
        {
            if (axes.Contains(i))
            {
                if (Dimensions[i] != 1)
                {
                    throw new ArgumentOutOfRangeException("axes", "the axes dimension must be 1.");
                }

                return false;
            }

            return true;
        }).Select(i => Dimensions[i]).ToArray();
        var strides = Enumerable.Range(0, Rank).Where(i => !axes.Contains(i)).Select(i => Strides[i]).ToArray();
        return new Tensor<T>(Buffer, dimensions, strides);
    }

    public override void CopyTo(Tensor dest)
    {
        CopyTo(this, dest.Cast<T>());
    }

    public override Tensor<T> AsContiguous(bool force = false)
    {
        if (!force && TensorUtilities.IsContiguous(Dimensions, Strides))
        {
            return this;
        }

        var dest = Tensor.Zeros(ElementType, Dimensions).Cast<T>();
        CopyTo(dest);
        return dest;
    }

    private static void TransposeRange(
        Memory<T> sourceMemory,
        T[] destinationArray,
        long[] dimensions,
        long[] sourceStrides,
        int start,
        int end)
    {
        var rank = dimensions.Length;
        var coordinates = new long[rank];
        var remainder = (long)start;
        long sourceOffset = 0;
        for (var axis = rank - 1; axis >= 0; axis--)
        {
            coordinates[axis] = remainder % dimensions[axis];
            remainder /= dimensions[axis];
            sourceOffset += coordinates[axis] * sourceStrides[axis];
        }

        var source = sourceMemory.Span;
        var destination = destinationArray.AsSpan();
        for (var outputOffset = start; outputOffset < end; outputOffset++)
        {
            destination[outputOffset] = source[checked((int)sourceOffset)];
            for (var axis = rank - 1; axis >= 0; axis--)
            {
                coordinates[axis]++;
                sourceOffset += sourceStrides[axis];
                if (coordinates[axis] < dimensions[axis])
                {
                    break;
                }

                coordinates[axis] = 0;
                sourceOffset -= dimensions[axis] * sourceStrides[axis];
            }
        }
    }

    private static void CopyTo(Tensor<T> src, Tensor<T> dest)
    {
        if (!src.Dimensions.SequenceEqual(dest.Dimensions))
        {
            throw new ArgumentException("the dest tensor shape must be equal to this tensor shape.", "dest");
        }

        var conti_dims = Math.Min(
            TensorUtilities.GetContiguousDims(src.Dimensions, src.Strides),
            TensorUtilities.GetContiguousDims(dest.Dimensions, dest.Strides));

        void Apply(int axis, long[] index)
        {
            if (axis >= (src.Rank - conti_dims))
            {
                var size = TensorUtilities.GetProduct(src.Dimensions, axis);

                var srcSpan = src.Buffer.Slice((int)TensorUtilities.GetLinearOffset(src.Strides, index), (int)size);
                var destSpan = dest.Buffer.Slice((int)TensorUtilities.GetLinearOffset(dest.Strides, index), (int)size);
                srcSpan.CopyTo(destSpan);
            }
            else
            {
                var dim = src.Dimensions[axis];
                for (index[axis] = 0; index[axis] < dim; index[axis]++)
                {
                    Apply(axis + 1, index);
                }
            }
        }

        long[] index = new long[src.Dimensions.Length];
        Apply(0, index);
    }
}
