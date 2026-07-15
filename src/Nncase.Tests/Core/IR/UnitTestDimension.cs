// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestDimension
{
    [Fact]
    public void TestValue()
    {
        long v1 = 1;
        var d1 = new DimConst(v1);
        Assert.Equal(v1, d1.Value);
        Assert.Equal(v1, d1.FixedValue);

        long v2 = -1;
        var d2 = new DimConst(v2);
        Assert.Equal(v2, d2.Value);
        Assert.Equal(v2, d2.FixedValue);
    }

    [Fact]
    public void TestKind()
    {
        long v1 = 1;
        var d1 = new DimConst(v1);
        Assert.Equal(DimensionKind.Fixed, d1.Kind);
        Assert.False(d1.IsUnknown);
        Assert.True(d1.IsFixed);

        var d2 = Dimension.Unknown;
        Assert.Equal(DimensionKind.Unknown, d2.Kind);
        Assert.True(d2.IsUnknown);
        Assert.False(d2.IsFixed);
    }

    [Fact]
    public void TestOperatorEqual()
    {
        Dimension d1 = 1;
        Dimension d2 = 1;
        Dimension d3 = 3;
        Assert.True(d1 == d2);
        Assert.False(d1 == d3);
    }

    [Fact]
    public void TestOperatorNotEqual()
    {
        Dimension d1 = 1;
        Dimension d2 = 1;
        Dimension d3 = 3;
        Assert.False(d1 != d2);
        Assert.True(d1 != d3);
    }

    [Fact]
    public void TestOperatorAdd()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 + d2;
        Assert.Equal(v1 + v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 + d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);

        d4 = d1 + v2;
        Assert.Equal(v1 + v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);
    }

    [Fact]
    public void TestOperatorSubtract()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 - d2;
        Assert.Equal(v1 - v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 - d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);
    }

    [Fact]
    public void TestOperatorMul()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 * d2;
        Assert.Equal(v1 * v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 * d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);
    }

    [Fact]
    public void TestOperatorDiv()
    {
        long v1 = 2;
        long v2 = 1;
        Dimension d1 = v1;
        Dimension d2 = v2;
        var d3 = Dimension.Unknown;

        var d4 = d1 / d2;
        Assert.Equal(v1 / v2, d4);
        Assert.Equal(DimensionKind.Fixed, d4.Kind);

        d4 = d1 / d3;
        Assert.Equal(DimensionKind.Unknown, d4.Kind);
    }

    [Fact]
    public void TestDimensionSum()
    {
        var dv = new DimVar("x");
        var negdv = -dv;
        var zero = dv + negdv;
        Assert.Equal(zero, new DimConst(0));
        var padding = new IR.Shapes.Padding(0, 128 - dv);
        var paded = dv + padding.Sum();
        Assert.Equal(paded, new DimConst(128));
        var paded2 = padding.Sum() + dv;
        Assert.Equal(paded2, new DimConst(128));
    }

    [Fact]
    public void TestAlignUp()
    {
        var x = new DimVar("x")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };
        var alignUp = (x + 8 - 1) / 8;
        var frac = Assert.IsType<DimFraction>(alignUp);
        var sum = Assert.IsType<DimSum>(frac.Numerator);
        Assert.Equal(7, sum.Bias);
    }

    [Fact]
    public void TestDimensionSumSimplify()
    {
        var dv = new DimVar("x");
        var ceilDiv = Dimension.CeilDiv(new DimProduct([dv], 1), 8);
        var mul = 8 * ceilDiv;
        var neg = -dv;
        var sum = mul + neg;
        sum = dv + sum;
        Assert.Equal(sum, mul);
    }

    [Fact]
    public void TestDimensionProductDistributesSumBias()
    {
        var value = new DimVar("value");
        var expression = 256 + (4 * value);
        var negated = Assert.IsType<DimSum>(-expression);
        var difference = expression - expression;

        Assert.Equal(-256, negated.Bias);
        Assert.Equal(Dimension.Zero, difference.Simplify());
        Assert.NotEqual(new DimSum([value], 1), new DimSum([value], 2));
        Assert.NotEqual(
            new DimFraction(DimDivideMode.FloorDiv, value, 8),
            new DimFraction(DimDivideMode.CeilDiv, value, 8));
    }

    [Fact]
    public void TestDimensionMinMaxInferRange()
    {
        var lhs = new DimVar("lhs")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };
        var rhs = new DimVar("rhs")
        {
            Metadata = new()
            {
                Range = new(4, 64),
            },
        };

        Assert.Equal(new ValueRange<double>(1, 64), Dimension.Min(lhs, rhs).Metadata.Range);
        Assert.Equal(new ValueRange<double>(4, 128), Dimension.Max(lhs, rhs).Metadata.Range);

        var offset = new DimVar("offset")
        {
            Metadata = new()
            {
                Range = new(0, 1),
            },
        };
        var boundedTile = Dimension.Min(1L, 1L, Dimension.Max(0L, 1L - offset));
        Assert.Equal(new ValueRange<double>(0, 1), boundedTile.Metadata.Range);
    }

    [Fact]
    public void TestDimensionExtremumCanonicalization()
    {
        var bounded = new DimVar("bounded")
        {
            Metadata = new()
            {
                Range = new(8, 16),
            },
        };

        var canonicalizer = new DimensionCanonicalizer();
        Assert.Same(bounded, canonicalizer.Canonicalize(new DimMin(32, new DimMin(bounded, 32), bounded)));
        Assert.Same(bounded, canonicalizer.Canonicalize(new DimMax(0, new DimMax(bounded, 0), bounded)));
    }

    [Fact]
    public void TestDimensionCanonicalizerIsIdempotent()
    {
        var value = new DimVar("value")
        {
            Metadata = new()
            {
                Range = new(0, 256),
            },
        };
        var expression = new DimMin(128, 128, new DimMax(0, value));
        var canonicalizer = new DimensionCanonicalizer();

        var first = canonicalizer.Canonicalize(expression);
        var second = new DimensionCanonicalizer().Canonicalize(first);

        Assert.Same(first, second);
        var min = Assert.IsType<DimMin>(first);
        Assert.Equal(2, min.Operands.Length);
        Assert.DoesNotContain(min.Operands.ToArray(), operand => operand is DimMax);
    }
}
