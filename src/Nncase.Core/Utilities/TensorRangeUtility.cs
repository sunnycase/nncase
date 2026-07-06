// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nncase.Utilities;

public static class TensorRangeUtility
{
    public static bool TryGetValueRange(Tensor tensor, out ValueRange<double> range)
    {
        switch (tensor)
        {
            case Tensor<bool> value:
                range = GetBooleanRange(value.Buffer.Span);
                return true;
            case Tensor<Utf8Char> value:
                range = GetUtf8CharRange(value.Buffer.Span);
                return true;
            case Tensor<sbyte> value:
                range = GetComparableRange<sbyte>(value.Buffer.Span);
                return true;
            case Tensor<byte> value:
                range = GetComparableRange<byte>(value.Buffer.Span);
                return true;
            case Tensor<short> value:
                range = GetComparableRange<short>(value.Buffer.Span);
                return true;
            case Tensor<ushort> value:
                range = GetComparableRange<ushort>(value.Buffer.Span);
                return true;
            case Tensor<int> value:
                range = GetComparableRange<int>(value.Buffer.Span);
                return true;
            case Tensor<uint> value:
                range = GetComparableRange<uint>(value.Buffer.Span);
                return true;
            case Tensor<long> value:
                range = GetComparableRange<long>(value.Buffer.Span);
                return true;
            case Tensor<ulong> value:
                range = GetComparableRange<ulong>(value.Buffer.Span);
                return true;
            case Tensor<Half> value:
                range = GetHalfRange(value.Buffer.Span);
                return true;
            case Tensor<BFloat16> value:
                range = GetBFloat16Range(value.Buffer.Span);
                return true;
            case Tensor<float> value:
                range = GetSingleRange(value.Buffer.Span);
                return true;
            case Tensor<double> value:
                range = GetDoubleRange(value.Buffer.Span);
                return true;
            case Tensor<Float8E4M3> value:
                range = GetFloat8E4M3Range(value.Buffer.Span);
                return true;
            case Tensor<Float8E5M2> value:
                range = GetFloat8E5M2Range(value.Buffer.Span);
                return true;
            default:
                range = ValueRange<double>.Full;
                return false;
        }
    }

    private static ValueRange<double> GetBooleanRange(ReadOnlySpan<bool> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        var hasFalse = false;
        var hasTrue = false;
        foreach (var value in values)
        {
            hasTrue |= value;
            hasFalse |= !value;
            if (hasTrue && hasFalse)
            {
                return new(0, 1);
            }
        }

        return hasTrue ? new(1, 1) : new(0, 0);
    }

    private static ValueRange<double> GetUtf8CharRange(ReadOnlySpan<Utf8Char> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        var min = (byte)values[0];
        var max = min;
        for (var i = 1; i < values.Length; i++)
        {
            var value = (byte)values[i];
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }
        }

        return new(min, max);
    }

    private static ValueRange<double> GetComparableRange<T>(ReadOnlySpan<T> values)
        where T : struct, IComparable<T>
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        var min = values[0];
        var max = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            var value = values[i];
            if (value.CompareTo(min) < 0)
            {
                min = value;
            }

            if (value.CompareTo(max) > 0)
            {
                max = value;
            }
        }

        return new(Convert.ToDouble(min), Convert.ToDouble(max));
    }

    private static ValueRange<double> GetSingleRange(ReadOnlySpan<float> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        if (!Vector.IsHardwareAccelerated || values.Length < Vector<float>.Count)
        {
            return GetSingleScalarRange(values);
        }

        var vectorLength = values.Length / Vector<float>.Count * Vector<float>.Count;
        var vectors = MemoryMarshal.Cast<float, Vector<float>>(values[..vectorLength]);
        var minVector = vectors[0];
        var maxVector = vectors[0];
        for (var i = 1; i < vectors.Length; i++)
        {
            minVector = Vector.Min(minVector, vectors[i]);
            maxVector = Vector.Max(maxVector, vectors[i]);
        }

        var min = minVector[0];
        var max = maxVector[0];
        for (var i = 1; i < Vector<float>.Count; i++)
        {
            if (minVector[i] < min)
            {
                min = minVector[i];
            }

            if (maxVector[i] > max)
            {
                max = maxVector[i];
            }
        }

        for (var i = vectorLength; i < values.Length; i++)
        {
            var value = values[i];
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }
        }

        return new(min, max);
    }

    private static ValueRange<double> GetSingleScalarRange(ReadOnlySpan<float> values)
    {
        var min = values[0];
        var max = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            var value = values[i];
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }
        }

        return new(min, max);
    }

    private static ValueRange<double> GetDoubleRange(ReadOnlySpan<double> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        if (!Vector.IsHardwareAccelerated || values.Length < Vector<double>.Count)
        {
            return GetDoubleScalarRange(values);
        }

        var vectorLength = values.Length / Vector<double>.Count * Vector<double>.Count;
        var vectors = MemoryMarshal.Cast<double, Vector<double>>(values[..vectorLength]);
        var minVector = vectors[0];
        var maxVector = vectors[0];
        for (var i = 1; i < vectors.Length; i++)
        {
            minVector = Vector.Min(minVector, vectors[i]);
            maxVector = Vector.Max(maxVector, vectors[i]);
        }

        var min = minVector[0];
        var max = maxVector[0];
        for (var i = 1; i < Vector<double>.Count; i++)
        {
            if (minVector[i] < min)
            {
                min = minVector[i];
            }

            if (maxVector[i] > max)
            {
                max = maxVector[i];
            }
        }

        for (var i = vectorLength; i < values.Length; i++)
        {
            var value = values[i];
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }
        }

        return new(min, max);
    }

    private static ValueRange<double> GetDoubleScalarRange(ReadOnlySpan<double> values)
    {
        var min = values[0];
        var max = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            var value = values[i];
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }
        }

        return new(min, max);
    }

    private static ValueRange<double> GetHalfRange(ReadOnlySpan<Half> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        var (minBits, maxBits) = GetFloat16BitRange(MemoryMarshal.Cast<Half, ushort>(values));
        return new((double)BitConverter.UInt16BitsToHalf(minBits), (double)BitConverter.UInt16BitsToHalf(maxBits));
    }

    private static ValueRange<double> GetBFloat16Range(ReadOnlySpan<BFloat16> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        var (minBits, maxBits) = GetFloat16BitRange(MemoryMarshal.Cast<BFloat16, ushort>(values));
        return new((float)BFloat16.FromRaw(minBits), (float)BFloat16.FromRaw(maxBits));
    }

    private static ValueRange<double> GetFloat8E4M3Range(ReadOnlySpan<Float8E4M3> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        var min = (float)values[0];
        var max = min;
        for (var i = 1; i < values.Length; i++)
        {
            var value = (float)values[i];
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }
        }

        return new(min, max);
    }

    private static ValueRange<double> GetFloat8E5M2Range(ReadOnlySpan<Float8E5M2> values)
    {
        if (values.Length == 0)
        {
            return new(0, 0);
        }

        var min = (float)values[0];
        var max = min;
        for (var i = 1; i < values.Length; i++)
        {
            var value = (float)values[i];
            if (value < min)
            {
                min = value;
            }

            if (value > max)
            {
                max = value;
            }
        }

        return new(min, max);
    }

    private static (ushort MinBits, ushort MaxBits) GetFloat16BitRange(ReadOnlySpan<ushort> values)
    {
        if (!Vector.IsHardwareAccelerated || values.Length < Vector<ushort>.Count)
        {
            return GetFloat16BitScalarRange(values);
        }

        var vectorLength = values.Length / Vector<ushort>.Count * Vector<ushort>.Count;
        var vectors = MemoryMarshal.Cast<ushort, Vector<ushort>>(values[..vectorLength]);
        var minKeyVector = GetOrderedFloat16KeyVector(vectors[0]);
        var maxKeyVector = minKeyVector;
        for (var i = 1; i < vectors.Length; i++)
        {
            var keyVector = GetOrderedFloat16KeyVector(vectors[i]);
            minKeyVector = Vector.Min(minKeyVector, keyVector);
            maxKeyVector = Vector.Max(maxKeyVector, keyVector);
        }

        var minKey = minKeyVector[0];
        var maxKey = maxKeyVector[0];
        for (var i = 1; i < Vector<ushort>.Count; i++)
        {
            if (minKeyVector[i] < minKey)
            {
                minKey = minKeyVector[i];
            }

            if (maxKeyVector[i] > maxKey)
            {
                maxKey = maxKeyVector[i];
            }
        }

        for (var i = vectorLength; i < values.Length; i++)
        {
            var key = GetOrderedFloat16Key(values[i]);
            if (key < minKey)
            {
                minKey = key;
            }

            if (key > maxKey)
            {
                maxKey = key;
            }
        }

        return (GetFloat16BitsFromOrderedKey(minKey), GetFloat16BitsFromOrderedKey(maxKey));
    }

    private static (ushort MinBits, ushort MaxBits) GetFloat16BitScalarRange(ReadOnlySpan<ushort> values)
    {
        var minKey = GetOrderedFloat16Key(values[0]);
        var maxKey = minKey;
        for (var i = 1; i < values.Length; i++)
        {
            var key = GetOrderedFloat16Key(values[i]);
            if (key < minKey)
            {
                minKey = key;
            }

            if (key > maxKey)
            {
                maxKey = key;
            }
        }

        return (GetFloat16BitsFromOrderedKey(minKey), GetFloat16BitsFromOrderedKey(maxKey));
    }

    private static Vector<ushort> GetOrderedFloat16KeyVector(Vector<ushort> bits)
    {
        var signMask = new Vector<ushort>(0x8000);
        var negativeMask = Vector.Equals(bits & signMask, signMask);
        var positiveKeys = bits | signMask;
        var negativeKeys = bits ^ new Vector<ushort>(ushort.MaxValue);
        return Vector.ConditionalSelect(negativeMask, negativeKeys, positiveKeys);
    }

    private static ushort GetOrderedFloat16Key(ushort bits)
    {
        return (ushort)((bits & 0x8000) == 0 ? bits | 0x8000 : ~bits & 0xffff);
    }

    private static ushort GetFloat16BitsFromOrderedKey(ushort key)
    {
        return (ushort)((key & 0x8000) != 0 ? key & 0x7fff : ~key & 0xffff);
    }
}
