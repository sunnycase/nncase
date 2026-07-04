// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.DistributedTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestDistributedTypeInfer : TestClassBase
{
    public static TheoryData<DistributedType, long[], IRType> ReshapeTypeInferData { get; } = new()
    {
        {
            // split on not related axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on splited-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.S(new[] { 0 }) }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.B, SBP.S(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on splited-by-reshape axis, but less than split factor.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.S(new[] { 0 }) }, new(new[] { 128 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new InvalidType("not support")
        },
        {
            // split on sequeezed axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.S(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on right unsequeeze axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 1, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1, 64, 16 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.B, SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on left unsequeeze axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 384, 128 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 1, 384, 128 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 1, 384, 128 }), new SBP[] { SBP.B, SBP.B, SBP.S(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.B, SBP.S(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.S(new[] { 0 }) }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on merged-by-reshape axis, but not support.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.B, SBP.B, SBP.S(new[] { 0 }) }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 1024 },
            new InvalidType("not support")
        },
        {
            // mesh dim 0 split on first merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.S(new[] { 1 }), SBP.B }, new(new[] { 4, 8 }, "yx", "bb")),
            new long[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.S(new[] { 1 }) }, new(new[] { 4, 8 }, "yx", "bb"))
        },
        {
            // mesh dim 1 split on first merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.S(new[] { 1 }), SBP.S(new[] { 0 }), SBP.B }, new(new[] { 4, 8 }, "yx", "bb")),
            new long[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.S(new[] { 1 }), SBP.S(new[] { 0 }) }, new(new[] { 4, 8 }, "yx", "bb"))
        },
        {
            // split on second merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.S(new[] { 0 }), SBP.B, SBP.S(new[] { 1 }) }, new(new[] { 4, 8 }, "yx", "bb")),
            new long[] { 1, 48, 1024 },
            new InvalidType("not support")
        },
        {
            // split and merge axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 16, 48, 1024 }), new SBP[] { SBP.S(new[] { 0 }), SBP.B, SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 8, 2, 64, 768 },
            new DistributedType(new(DataTypes.Float32, new long[] { 8, 2, 64, 768 }), new SBP[] { SBP.S(new[] { 0 }), SBP.B, SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // unmapable reshape
            new DistributedType(new(DataTypes.Float32, new long[] { 2, 30 }), new SBP[] { SBP.S(new[] { 0 }), SBP.B }, new(new[] { 6 }, "b", "b")),
            new long[] { 3, 20 },
            new InvalidType("unmapable")
        },
    };

    [Fact]
    public void TestGenerateReduceGroups()
    {
        Assert.Single(LinqUtility.Combination(1));
        Assert.Equal(3, LinqUtility.Combination(2).Count());
    }

    [Fact]
    public void TestMatMulGluRejectsPartialProjectionSharding()
    {
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 128) } };
        var placement = new Placement(new[] { 4 }, "b", "b");
        var input = new Var("input", new DistributedType(new TensorType(DataTypes.BFloat16, new Dimension[] { sequenceLength, 64 }), new SBP[] { SBP.B, SBP.S([0]) }, placement));
        var gateWeight = new Var("gate_weight", new DistributedType(new TensorType(DataTypes.BFloat16, new long[] { 64, 128 }), new SBP[] { SBP.S([0]), SBP.B }, placement));
        var upWeight = new Var("up_weight", gateWeight.CheckedType);

        var expr = IR.F.NN.MatMulGlu(
            input,
            gateWeight,
            upWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);

        var invalid = Assert.IsType<InvalidType>(expr.CheckedType);
        Assert.Contains("nonlinear", invalid.Reason, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackedMatMulGluRejectsPartialProjectionSharding()
    {
        var placement = new Placement(new[] { 4 }, "b", "b");
        var packedBFloat16 = new VectorType(DataTypes.BFloat16, [4, 8]);
        var input = new Var("input", new DistributedType(new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }), new SBP[] { SBP.B, SBP.S([0]) }, placement));
        var gateWeight = new Var("gate_weight", new DistributedType(new TensorType(packedBFloat16, new long[] { 4, 64 }), new SBP[] { SBP.B, SBP.S([0]) }, placement));
        var upWeight = new Var("up_weight", gateWeight.CheckedType);

        var expr = IR.F.NTT.PackedMatMulGlu(
            input,
            gateWeight,
            upWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);

        var invalid = Assert.IsType<InvalidType>(expr.CheckedType);
        Assert.Contains("nonlinear", invalid.Reason, System.StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(ReshapeTypeInferData))]
    public void TestReshapeTypeInfer(DistributedType inType, long[] newShape, IRType except)
    {
        var reshape = IR.F.Tensors.Reshape(new Var(inType), newShape);
        if (except is InvalidType)
        {
            Assert.IsType<InvalidType>(reshape.CheckedType);
        }
        else
        {
            Assert.Equal(except, reshape.CheckedType);
        }
    }
}
