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
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on splited-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }) }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on splited-by-reshape axis, but less than split factor.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }) }, new(new[] { 128 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new InvalidType("not support")
        },
        {
            // split on sequeezed axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on right unsequeeze axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 1, 64, 16 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1, 64, 16 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B, SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on left unsequeeze axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 384, 128 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 1, 384, 128 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 1, 384, 128 }), new SBP[] { SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }) }, new(new[] { 8 }, "b", "b"))
        },
        {
            // split on merged-by-reshape axis, but not support.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.B, SBP.B, SBP.SContiguous(new[] { 0 }) }, new(new[] { 8 }, "b", "b")),
            new long[] { 1, 48, 1024 },
            new InvalidType("not support")
        },
        {
            // mesh dim 0 split on first merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.SContiguous(new[] { 1 }), SBP.B }, new(new[] { 4, 8 }, "yx", "bb")),
            new long[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.SContiguous(new[] { 1 }) }, new(new[] { 4, 8 }, "yx", "bb"))
        },
        {
            // mesh dim 1 split on first merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 1 }), SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 4, 8 }, "yx", "bb")),
            new long[] { 1, 48, 1024 },
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 1024 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 1 }), SBP.SContiguous(new[] { 0 }) }, new(new[] { 4, 8 }, "yx", "bb"))
        },
        {
            // split on second merged-by-reshape axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 1, 48, 64, 16 }), new SBP[] { SBP.B, SBP.SContiguous(new[] { 0 }), SBP.B, SBP.SContiguous(new[] { 1 }) }, new(new[] { 4, 8 }, "yx", "bb")),
            new long[] { 1, 48, 1024 },
            new InvalidType("not support")
        },
        {
            // split and merge axis.
            new DistributedType(new(DataTypes.Float32, new long[] { 16, 48, 1024 }), new SBP[] { SBP.SContiguous(new[] { 0 }), SBP.B, SBP.B }, new(new[] { 8 }, "b", "b")),
            new long[] { 8, 2, 64, 768 },
            new DistributedType(new(DataTypes.Float32, new long[] { 8, 2, 64, 768 }), new SBP[] { SBP.SContiguous(new[] { 0 }), SBP.B, SBP.B, SBP.B }, new(new[] { 8 }, "b", "b"))
        },
        {
            // unmapable reshape
            new DistributedType(new(DataTypes.Float32, new long[] { 2, 30 }), new SBP[] { SBP.SContiguous(new[] { 0 }), SBP.B }, new(new[] { 6 }, "b", "b")),
            new long[] { 3, 20 },
            new InvalidType("unmapable")
        },
    };

    [Fact]
    public void TestD2DBoxingRejectsDifferentTensorType()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var sourceType = new DistributedType(new TensorType(DataTypes.Float32, [16, 32]), new SBP[] { SBP.SContiguous([0]), SBP.SContiguous([1]) }, placement);
        var targetType = new DistributedType(new TensorType(DataTypes.Float32, [16, 8, 4]), new SBP[] { SBP.SContiguous([0]), SBP.SContiguous([1]), SBP.B }, placement);
        var input = new Var("input", sourceType);

        var boxing = IR.F.Distributed.Boxing(input, targetType);

        Assert.IsType<InvalidType>(boxing.CheckedType);
    }

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
        var input = new Var("input", new DistributedType(new TensorType(DataTypes.BFloat16, new Dimension[] { sequenceLength, 64 }), new SBP[] { SBP.B, SBP.SContiguous([0]) }, placement));
        var gateWeight = new Var("gate_weight", new DistributedType(new TensorType(DataTypes.BFloat16, new long[] { 64, 128 }), new SBP[] { SBP.SContiguous([0]), SBP.B }, placement));
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
    public void TestMatMulGluRejectsMismatchedProjectionSharding()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var input = new Var(
            "input",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 32, 1024 }),
                new SBP[] { SBP.B, SBP.B },
                placement));
        var gateWeight = new Var(
            "gate_weight",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 1024, 3072 }),
                new SBP[] { SBP.B, SBP.SContiguous([0, 1], 96) },
                placement));
        var upWeight = new Var(
            "up_weight",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 1024, 3072 }),
                new SBP[] { SBP.B, SBP.SContiguous([1], 384) },
                placement));

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
        Assert.Contains("same distributed type", invalid.Reason, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackedMatMulGluRejectsPartialProjectionSharding()
    {
        var placement = new Placement(new[] { 4 }, "b", "b");
        var packedBFloat16 = new VectorType(DataTypes.BFloat16, [4, 8]);
        var input = new Var("input", new DistributedType(new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }), new SBP[] { SBP.B, SBP.SContiguous([0]) }, placement));
        var gateWeight = new Var("gate_weight", new DistributedType(new TensorType(packedBFloat16, new long[] { 4, 64 }), new SBP[] { SBP.B, SBP.SContiguous([0]) }, placement));
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

    [Fact]
    public void TestPackedMatMulGluRejectsMismatchedProjectionSharding()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var packedBFloat16 = new VectorType(DataTypes.BFloat16, [4, 8]);
        var input = new Var(
            "input",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 32, 1024 }),
                new SBP[] { SBP.B, SBP.B },
                placement));
        var gateWeight = new Var(
            "gate_weight",
            new DistributedType(
                new TensorType(packedBFloat16, new long[] { 96, 1024 }),
                new SBP[] { SBP.SContiguous([0, 1], 3), SBP.B },
                placement));
        var upWeight = new Var(
            "up_weight",
            new DistributedType(
                new TensorType(packedBFloat16, new long[] { 96, 1024 }),
                new SBP[] { SBP.SContiguous([1], 12), SBP.B },
                placement));

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
        Assert.Contains("same distributed type", invalid.Reason, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackRejectsSplitThatCutsRepeatedAxisVectorGroup()
    {
        var placement = new Placement(new[] { 4 }, "b", "b");
        var input = new Var(
            "input",
            new DistributedType(
                new TensorType(DataTypes.Float32, new long[] { 1, 128 }),
                new SBP[] { SBP.B, SBP.SContiguous([0], 4) },
                placement));

        var packed = IR.F.Tensors.Pack(input, [2, 8], [1, 1]);

        var invalid = Assert.IsType<InvalidType>(packed.CheckedType);
        Assert.Contains("cuts vector lane group 16", invalid.Reason, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackAndUnpackScaleRepeatedAxisSplitGranularityByLaneProduct()
    {
        var placement = new Placement(new[] { 4 }, "b", "b");
        var inputType = new DistributedType(
            new TensorType(DataTypes.Float32, new long[] { 1, 128 }),
            new SBP[] { SBP.B, SBP.SContiguous([0], 32) },
            placement);
        var input = new Var("input", inputType);

        var packed = IR.F.Tensors.Pack(input, [2, 8], [1, 1]);
        var packedType = Assert.IsType<DistributedType>(packed.CheckedType);
        Assert.Equal(new TensorType(new VectorType(DataTypes.Float32, [2, 8]), new long[] { 1, 8 }), packedType.TensorType);
        Assert.Equal(new SBP[] { SBP.B, SBP.SContiguous([0], 2) }, packedType.AxisPolicies.ToArray());

        var unpacked = IR.F.Tensors.Unpack(packed, [2, 8], [1, 1]);
        Assert.Equal(inputType, unpacked.CheckedType);
    }

    [Fact]
    public void TestPackAndUnpackPreserveNonUniformPhysicalShardBoundaries()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var packedType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 64, 18992 }),
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 594) },
            placement);
        var packed = new Var("packed", packedType);

        var unpacked = IR.F.Tensors.Unpack(packed, [8], [1]);
        var unpackedType = Assert.IsType<DistributedType>(unpacked.CheckedType);
        Assert.Equal(new TensorType(DataTypes.BFloat16, new long[] { 64, 151936 }), unpackedType.TensorType);
        Assert.Equal(new SBP[] { SBP.B, SBP.SContiguous([0, 1], 4752) }, unpackedType.AxisPolicies.ToArray());

        var repacked = IR.F.Tensors.Pack(unpacked, [8], [1]);
        Assert.Equal(packedType, repacked.CheckedType);
    }

    [Fact]
    public void TestVectorizedCastPreservesNonUniformPhysicalShardBoundaries()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 1, 18992 }),
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 594) },
            placement);
        var input = new Var("input", inputType);

        var widened = IR.F.NTT.VectorizedCast(
            input,
            new VectorType(DataTypes.Float32, [4]),
            CastMode.KDefault,
            [1],
            None.Default);
        var widenedType = Assert.IsType<DistributedType>(widened.CheckedType);
        Assert.Equal(
            new TensorType(new VectorType(DataTypes.Float32, [4]), new long[] { 1, 37984 }),
            widenedType.TensorType);
        Assert.Equal(new SBP[] { SBP.B, SBP.SContiguous([0, 1], 1188) }, widenedType.AxisPolicies.ToArray());

        for (var linearIndex = 0; linearIndex < 32; linearIndex++)
        {
            var shardIndex = DistributedUtility.GetUnraveledIndex(linearIndex, placement.Hierarchy.ToArray());
            var inputRegion = DistributedUtility.GetLocalOffsetAndShape(inputType, shardIndex);
            var outputRegion = DistributedUtility.GetLocalOffsetAndShape(widenedType, shardIndex);
            Assert.Equal(inputRegion.Offset[1].FixedValue * 2, outputRegion.Offset[1].FixedValue);
            Assert.Equal(inputRegion.Shape[1].FixedValue * 2, outputRegion.Shape[1].FixedValue);
        }

        var scalar = IR.F.Tensors.Bitcast(widened, DataTypes.Float32);
        var scalarType = Assert.IsType<DistributedType>(scalar.CheckedType);
        Assert.Equal(new SBP[] { SBP.B, SBP.SContiguous([0, 1], 4752) }, scalarType.AxisPolicies.ToArray());

        var narrowed = IR.F.NTT.VectorizedCast(
            new Var("widened", widenedType),
            new VectorType(DataTypes.BFloat16, [8]),
            CastMode.KDefault,
            [1],
            None.Default);
        Assert.Equal(inputType, narrowed.CheckedType);
    }

    [Fact]
    public void TestBitcastPreservesSplitGranularityOnUnchangedAxes()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var input = new Var(
            "input",
            new DistributedType(
                new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 96, 16, 16 }),
                new SBP[] { SBP.SBlockCyclic([0], 8), SBP.SBlockCyclic([1], 2), SBP.B },
                placement));

        var bitcast = IR.F.Tensors.Bitcast(input, DataTypes.BFloat16);

        var outputType = Assert.IsType<DistributedType>(bitcast.CheckedType);
        Assert.Equal(new TensorType(DataTypes.BFloat16, new long[] { 96, 16, 128 }), outputType.TensorType);
        Assert.Equal(
            new SBP[] { SBP.SBlockCyclic([0], 8), SBP.SBlockCyclic([1], 2), SBP.B },
            outputType.AxisPolicies.ToArray());
    }

    [Fact]
    public void TestVectorizedCastRejectsMisalignedSplitGranularity()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var input = new Var(
            "input",
            new DistributedType(
                new TensorType(new VectorType(DataTypes.Float32, [4]), new long[] { 1, 37984 }),
                new SBP[] { SBP.B, SBP.SContiguous([0, 1], 1187) },
                placement));

        var cast = IR.F.NTT.VectorizedCast(
            input,
            new VectorType(DataTypes.BFloat16, [8]),
            CastMode.KDefault,
            [1],
            None.Default);

        var invalid = Assert.IsType<InvalidType>(cast.CheckedType);
        Assert.Contains("split granularity", invalid.Reason, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestVectorizedCastShapeFollowsLaneLayout()
    {
        var input = new Var(
            "input",
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 3 }));

        var cast = IR.F.NTT.VectorizedCast(
            input,
            new VectorType(DataTypes.Float32, [2]),
            CastMode.KDefault,
            [0],
            None.Default);

        Assert.Equal(
            new TensorType(new VectorType(DataTypes.Float32, [2]), new long[] { 12 }),
            cast.CheckedType);
    }

    [Fact]
    public void TestKMajorPackedMatMulAcceptsNonUniformOutputSharding()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var lhs = new Var(
            "lhs",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 1, 1024 }),
                new SBP[] { SBP.B, SBP.B },
                placement));
        var rhs = new Var(
            "rhs",
            new DistributedType(
                new TensorType(new VectorType(DataTypes.BFloat16, [8, 2, 8]), new long[] { 64, 18992 }),
                new SBP[] { SBP.B, SBP.SContiguous([0, 1], 594) },
                placement));

        var matmul = IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: IR.NTT.PackedMatMulRhsLayout.KMajor);

        var outputType = Assert.IsType<DistributedType>(matmul.CheckedType);
        Assert.Equal(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 1, 18992 }),
            outputType.TensorType);
        Assert.Equal(new SBP[] { SBP.B, SBP.SContiguous([0, 1], 594) }, outputType.AxisPolicies.ToArray());
    }

    [Fact]
    public void TestFixedMatMulReductionSplitProducesPartialOutput()
    {
        var placement = new Placement(new[] { 8, 16 }, "yx", "bb");
        var lhs = new DistributedType(
            new TensorType(DataTypes.BFloat16, new long[] { 1, 2048 }),
            new SBP[] { SBP.B, SBP.SContiguous([0]) },
            placement);
        var rhs = new DistributedType(
            new TensorType(DataTypes.BFloat16, new long[] { 2048, 4096 }),
            new SBP[] { SBP.SContiguous([0]), SBP.SContiguous([1]) },
            placement);

        var output = Assert.IsType<DistributedType>(
            Nncase.Evaluator.Math.MatMulEvaluator.VisitDistributedType(lhs, rhs, NoneType.Default));

        Assert.Equal(new TensorType(DataTypes.BFloat16, new long[] { 1, 4096 }), output.TensorType);
        Assert.Equal(new SBP[] { SBP.B, SBP.SContiguous([1]) }, output.AxisPolicies.ToArray());
        Assert.Equal(SBP.P([0], ReduceOp.Sum), output.Partial);
    }

    [Fact]
    public void TestNormStatsProducesPartialAndNormApplyRequiresBroadcastStats()
    {
        var placement = new Placement(new[] { 4 }, "b", "b");
        var inputType = new DistributedType(
            new TensorType(DataTypes.Float32, new long[] { 2, 8 }),
            new SBP[] { SBP.B, SBP.SContiguous([0]) },
            placement);
        var input = new Var("input", inputType);
        var stats = IR.F.NN.NormStats(1, input, useMean: true);

        var statsType = Assert.IsType<DistributedType>(stats.CheckedType);
        Assert.Equal(new TensorType(DataTypes.Float32, new long[] { 2, 2, 1 }), statsType.TensorType);
        Assert.Equal(new SBP[] { SBP.B, SBP.B, SBP.B }, statsType.AxisPolicies.ToArray());
        Assert.Equal(SBP.P([0], ReduceOp.Sum), statsType.Partial);

        var scale = new Var(
            "scale",
            new DistributedType(new TensorType(DataTypes.Float32, new long[] { 8 }), new SBP[] { SBP.SContiguous([0]) }, placement));
        var bias = new Var("bias", scale.CheckedType);
        var invalidApply = IR.F.NN.NormApply(1, 1e-5f, input, new Var("partial_stats", statsType), scale, bias, useMean: true);
        Assert.IsType<InvalidType>(invalidApply.CheckedType);

        var broadcastStats = new Var(
            "broadcast_stats",
            new DistributedType(statsType.TensorType, statsType.AxisPolicies, statsType.Placement));
        var apply = IR.F.NN.NormApply(1, 1e-5f, input, broadcastStats, scale, bias, useMean: true);
        Assert.Equal(inputType, apply.CheckedType);
    }

    [Fact]
    public void TestVectorizedNormStatsProducesScalarFp32Stats()
    {
        var placement = new Placement(new[] { 4 }, "b", "b");
        var inputType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 2, 16 }),
            new SBP[] { SBP.B, SBP.SContiguous([0]) },
            placement);
        var input = new Var("input", inputType);
        var stats = IR.F.NN.NormStats(1, input, useMean: false);

        var statsType = Assert.IsType<DistributedType>(stats.CheckedType);
        Assert.Equal(new TensorType(DataTypes.Float32, new long[] { 1, 2, 1 }), statsType.TensorType);
        Assert.Equal(new SBP[] { SBP.B, SBP.B, SBP.B }, statsType.AxisPolicies.ToArray());
        Assert.Equal(SBP.P([0], ReduceOp.Sum), statsType.Partial);
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

    [Fact]
    public void TestReshapeFlattensOrderedBlockCyclicSplits()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), [16, 16, 1]),
            [
                SBP.SBlockCyclic([1], 2),
                SBP.SBlockCyclic([0], 4),
                SBP.B,
            ],
            placement);

        var reshape = IR.F.Tensors.Reshape(new Var("input", inputType), [1, 256]);
        var expectedReshapeType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), [1, 256]),
            [
                SBP.B,
                SBP.S(
                    SplitStage.BlockCyclic([1], 32),
                    SplitStage.BlockCyclic([0], 4)),
            ],
            placement);
        Assert.Equal(expectedReshapeType, reshape.CheckedType);

        var bitcast = IR.F.Tensors.Bitcast(reshape, DataTypes.BFloat16);
        var expectedBitcastType = new DistributedType(
            new TensorType(DataTypes.BFloat16, [1, 2048]),
            [
                SBP.B,
                SBP.S(
                    SplitStage.BlockCyclic([1], 256),
                    SplitStage.BlockCyclic([0], 32)),
            ],
            placement);
        Assert.Equal(expectedBitcastType, bitcast.CheckedType);
    }
}
