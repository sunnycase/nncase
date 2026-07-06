// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Numerics;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensorRangeUtility
{
    [Fact]
    public void TestGetSingleRange()
    {
        var data = new float[(Vector<float>.Count * 2) + 3];
        data.AsSpan().Fill(1.25f);
        data[1] = -7.5f;
        data[^2] = 9.25f;

        Assert.True(TensorRangeUtility.TryGetValueRange(Tensor.From<float>(data, [data.Length]), out var range));
        Assert.Equal(-7.5, range.Min);
        Assert.Equal(9.25, range.Max);
    }

    [Fact]
    public void TestGetBFloat16Range()
    {
        var min = (BFloat16)(-3.25f);
        var max = (BFloat16)2.5f;
        var data = new BFloat16[(Vector<ushort>.Count * 2) + 3];
        data.AsSpan().Fill((BFloat16)1.5f);
        data[2] = min;
        data[^2] = max;

        Assert.True(TensorRangeUtility.TryGetValueRange(Tensor.From<BFloat16>(data, [data.Length]), out var range));
        Assert.Equal((float)min, range.Min);
        Assert.Equal((float)max, range.Max);
    }

    [Fact]
    public void TestGetHalfRange()
    {
        var min = (Half)(-4.5f);
        var max = (Half)7.75f;
        var data = new Half[(Vector<ushort>.Count * 2) + 5];
        data.AsSpan().Fill((Half)0.25f);
        data[3] = min;
        data[^1] = max;

        Assert.True(TensorRangeUtility.TryGetValueRange(Tensor.From<Half>(data, [data.Length]), out var range));
        Assert.Equal((double)min, range.Min);
        Assert.Equal((double)max, range.Max);
    }

    [Fact]
    public void TestGetBooleanRange()
    {
        Assert.True(TensorRangeUtility.TryGetValueRange(Tensor.From<bool>([true, true], [2]), out var trueRange));
        Assert.Equal(1, trueRange.Min);
        Assert.Equal(1, trueRange.Max);

        Assert.True(TensorRangeUtility.TryGetValueRange(Tensor.From<bool>([true, false], [2]), out var mixedRange));
        Assert.Equal(0, mixedRange.Min);
        Assert.Equal(1, mixedRange.Max);
    }
}
