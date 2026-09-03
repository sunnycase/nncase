// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFusePackedMatMulNormStatsPass : TestClassBase
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TestFuseLocalStatsAndKeepReductionExplicit(bool useMean)
    {
        var (lhs, rhs, addend, packedMatMul) = CreateSplitNPackedMatMul();
        var packedType = Assert.IsType<DistributedType>(packedMatMul.CheckedType);
        var broadcastType = new DistributedType(
            packedType.TensorType,
            Enumerable.Repeat<SBP>(SBP.B, packedType.TensorType.Shape.Rank).ToArray(),
            packedType.Placement);
        var broadcastView = IR.F.Distributed.ShardedView(packedMatMul, broadcastType);
        var stats = IR.F.NN.NormStats(-1, broadcastView, useMean);
        var body = new IR.Tuple(broadcastView, stats);
        var function = new Function(
            "main",
            string.Empty,
            body,
            new IVar[] { lhs, rhs, addend });
        Assert.True(function.InferenceType());
        var originalType = function.CheckedType;

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulNormStatsPass(true, false, false).RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var fusedCall = Assert.Single(
            calls.Where(call => call.Target is PackedMatMulNormStats));
        var fused = Assert.IsType<PackedMatMulNormStats>(fusedCall.Target);
        var fusedType = Assert.IsType<TupleType>(fusedCall.CheckedType);
        var localStatsType = Assert.IsType<DistributedType>(fusedType.Fields[1]);
        var partial = Assert.IsType<SBPPartial>(localStatsType.Partial);

        Assert.Equal(packedMatMul.CheckedType, fusedType.Fields[0]);
        Assert.Equal(1, fused.Axis);
        Assert.Equal(useMean, fused.UseMean);
        Assert.Equal(ReduceOp.Sum, partial.Op);
        Assert.Equal(new[] { 0, 1 }, partial.Axes.ToArray());
        Assert.DoesNotContain(calls, call => call.Target is PackedMatMul or NormStats);

        var viewCall = Assert.Single(calls.Where(call => call.Target is ShardedView));
        var viewInput = Assert.IsType<Call>(viewCall[ShardedView.Input]);
        Assert.IsType<IR.Tensors.GetItem>(viewInput.Target);
        var boxingCall = Assert.Single(calls.Where(call => call.Target is Boxing));
        Assert.Equal(stats.CheckedType, boxingCall.CheckedType);
        Assert.True(rewritten.InferenceType());
        Assert.Equal(originalType, rewritten.CheckedType);
    }

    [Fact]
    public async Task TestRejectPartialPackedMatMulOutput()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var lhs = new Var(
            "lhs",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 1, 2048 }),
                new SBP[] { SBP.B, SBP.SBlockCyclic([1], 256) },
                placement));
        var rhs = new Var(
            "rhs",
            new DistributedType(
                new TensorType(
                    new VectorType(DataTypes.BFloat16, [8, 2, 8]),
                    new long[] { 128, 256 }),
                new SBP[] { SBP.SBlockCyclic([1], 16), SBP.SBlockCyclic([0], 8) },
                placement));
        var packed = IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor);
        var packedType = Assert.IsType<DistributedType>(packed.CheckedType);
        Assert.NotNull(packedType.Partial);
        var function = new Function(
            "main",
            string.Empty,
            IR.F.NN.NormStats(-1, packed, false),
            new IVar[] { lhs, rhs });

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulNormStatsPass(true, false, false).RunAsync(function, new()));
        Assert.Same(function, rewritten);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is PackedMatMulNormStats);
    }

    [Fact]
    public async Task TestRewriteNestedFusedProducerWithoutDuplicatingMatMul()
    {
        var (lhs, rhs, addend, firstPacked) = CreateSplitNPackedMatMul();
        var firstStats = IR.F.NN.NormStats(-1, firstPacked, false);
        var secondPacked = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor,
            addend: firstPacked));
        var secondStats = IR.F.NN.NormStats(-1, secondPacked, false);
        var function = new Function(
            "main",
            string.Empty,
            new IR.Tuple(firstStats, secondPacked, secondStats),
            new IVar[] { lhs, rhs, addend });
        Assert.True(function.InferenceType());

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulNormStatsPass(true, false, false).RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var fusedCalls = calls
            .Where(call => call.Target is PackedMatMulNormStats)
            .ToArray();

        Assert.Equal(2, fusedCalls.Length);
        Assert.DoesNotContain(calls, call => call.Target is PackedMatMul or NormStats);
        Assert.Contains(
            fusedCalls,
            call => call[PackedMatMulNormStats.Addend] is Call { Target: IR.Tensors.GetItem });
        Assert.True(rewritten.InferenceType());
    }

    [Fact]
    public async Task TestBlockScaledOnlyModeLeavesPackedMatMulForLaterFusion()
    {
        var (lhs, rhs, addend, packedMatMul) = CreateSplitNPackedMatMul();
        var function = new Function(
            "main",
            string.Empty,
            new IR.Tuple(packedMatMul, IR.F.NN.NormStats(-1, packedMatMul, false)),
            new IVar[] { lhs, rhs, addend });
        Assert.True(function.InferenceType());

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulNormStatsPass(false, true, false).RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();

        Assert.Contains(calls, call => call.Target is PackedMatMul);
        Assert.Contains(calls, call => call.Target is NormStats);
        Assert.DoesNotContain(calls, call => call.Target is PackedMatMulNormStats);
        Assert.True(rewritten.InferenceType());
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public async Task TestBlockScaledFusionRequiresTargetCapability(
        bool enableBlockScaledMatMul,
        bool expectFusion)
    {
        var lhs = new Var(
            "lhs",
            new TensorType(DataTypes.BFloat16, new long[] { 1, 128 }));
        var rhs = new Var(
            "rhs",
            new TensorType(
                new VectorType(DataTypes.Float8E4M3, [2, 16]),
                new long[] { 256, 4 }));
        var rhsScale = new Var(
            "rhs_scale",
            new TensorType(DataTypes.BFloat16, new long[] { 2, 1 }));
        var packedWithoutAddend = Assert.IsType<Call>(IR.F.NTT.PackedBlockScaledMatMul(
            lhs,
            rhs,
            rhsScale,
            DataTypes.BFloat16,
            128,
            128,
            PackedMatMulRhsLayout.NMajorKPacked,
            8));
        Assert.True(packedWithoutAddend.InferenceType());
        var addend = new Var("addend", packedWithoutAddend.CheckedType);
        var packed = IR.F.NTT.PackedBlockScaledMatMul(
            lhs,
            rhs,
            rhsScale,
            DataTypes.BFloat16,
            128,
            128,
            PackedMatMulRhsLayout.NMajorKPacked,
            8,
            addend);
        var function = new Function(
            "main",
            string.Empty,
            new IR.Tuple(packed, IR.F.NN.NormStats(-1, packed, false)),
            new IVar[] { lhs, rhs, rhsScale, addend });
        Assert.True(function.InferenceType());

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulNormStatsPass(false, enableBlockScaledMatMul, false)
                .RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();

        Assert.Equal(
            expectFusion,
            calls.Any(call => call.Target is PackedBlockScaledMatMulNormStats));
        Assert.Equal(
            !expectFusion,
            calls.Any(call => call.Target is PackedBlockScaledMatMul));
        Assert.Equal(
            !expectFusion,
            calls.Any(call => call.Target is NormStats));
        Assert.True(rewritten.InferenceType());
        Assert.Equal(function.CheckedType, rewritten.CheckedType);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public async Task TestNVFP4FusionRequiresTargetCapability(
        bool enableNVFP4MatMul,
        bool expectFusion)
    {
        var lhs = new Var(
            "lhs",
            new TensorType(
                new VectorType(DataTypes.BFloat16, [8]),
                new RankedShape(1, 640)));
        var rhsPacked = new Var(
            "rhs_packed",
            new TensorType(
                new VectorType(DataTypes.UInt8, [2, 16]),
                new RankedShape(640, 80)));
        var rhsScale = new Var(
            "rhs_scale",
            new TensorType(DataTypes.Float8E4M3, new RankedShape(640, 320)));
        var lhsGlobalScale = new Var(
            "lhs_global_scale",
            new TensorType(DataTypes.Float32, new RankedShape(1)));
        var rhsGlobalScale = new Var(
            "rhs_global_scale",
            new TensorType(DataTypes.Float32, new RankedShape(1)));
        var packedWithoutAddend = Assert.IsType<Call>(IR.F.NTT.PackedNVFP4MatMul(
            lhs,
            rhsPacked,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            DataTypes.BFloat16,
            16,
            8,
            2,
            16,
            8));
        Assert.True(packedWithoutAddend.InferenceType());
        var addend = new Var("addend", packedWithoutAddend.CheckedType);
        var packed = IR.F.NTT.PackedNVFP4MatMul(
            lhs,
            rhsPacked,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            DataTypes.BFloat16,
            16,
            8,
            2,
            16,
            8,
            addend);
        var function = new Function(
            "main",
            string.Empty,
            new IR.Tuple(packed, IR.F.NN.NormStats(-1, packed, false)),
            new IVar[]
            {
                lhs,
                rhsPacked,
                rhsScale,
                lhsGlobalScale,
                rhsGlobalScale,
                addend,
            });
        Assert.True(function.InferenceType());

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulNormStatsPass(false, false, enableNVFP4MatMul)
                .RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();

        Assert.Equal(
            expectFusion,
            calls.Any(call => call.Target is PackedNVFP4MatMulNormStats));
        Assert.Equal(
            !expectFusion,
            calls.Any(call => call.Target is PackedNVFP4MatMul));
        Assert.Equal(
            !expectFusion,
            calls.Any(call => call.Target is NormStats));
        if (expectFusion)
        {
            var fusedCall = Assert.Single(
                calls.Where(call => call.Target is PackedNVFP4MatMulNormStats));
            Assert.Same(addend, fusedCall[PackedNVFP4MatMulNormStats.Addend]);
        }

        Assert.True(rewritten.InferenceType());
        Assert.Equal(function.CheckedType, rewritten.CheckedType);
    }

    [Fact]
    public async Task TestFuseNVFP4StatsThroughLogicalViewsAndPipelineYield()
    {
        var lhs = new Var(
            "lhs",
            new TensorType(
                new VectorType(DataTypes.BFloat16, [8]),
                new RankedShape(1, 640)));
        var rhsPacked = new Var(
            "rhs_packed",
            new TensorType(
                new VectorType(DataTypes.UInt8, [2, 16]),
                new RankedShape(640, 80)));
        var rhsScale = new Var(
            "rhs_scale",
            new TensorType(DataTypes.Float8E4M3, new RankedShape(640, 320)));
        var lhsGlobalScale = new Var(
            "lhs_global_scale",
            new TensorType(DataTypes.Float32, new RankedShape(1)));
        var rhsGlobalScale = new Var(
            "rhs_global_scale",
            new TensorType(DataTypes.Float32, new RankedShape(1)));
        var dependency = new Var("dependency", NoneType.Default);
        var packed = IR.F.NTT.PackedNVFP4MatMul(
            lhs,
            rhsPacked,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            DataTypes.BFloat16,
            16,
            8,
            2,
            16,
            8);
        var scalarView = IR.F.Tensors.Bitcast(packed, DataTypes.BFloat16);
        var yielded = IR.F.Heterogeneous.PipelineYield(scalarView, dependency);
        var packedView = IR.F.Tensors.Pack(yielded, [8], [1]);
        var stats = IR.F.NN.NormStats(-1, packedView, false);
        var function = new Function(
            "main",
            string.Empty,
            new IR.Tuple(yielded, stats),
            new IVar[]
            {
                lhs,
                rhsPacked,
                rhsScale,
                lhsGlobalScale,
                rhsGlobalScale,
                dependency,
            });
        Assert.True(function.InferenceType());

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulNormStatsPass(false, false, true)
                .RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();

        Assert.Single(calls.Where(call => call.Target is PackedNVFP4MatMulNormStats));
        Assert.DoesNotContain(calls, call => call.Target is PackedNVFP4MatMul or NormStats);
        Assert.Equal(2, calls.Count(call => call.Target is IR.Heterogeneous.PipelineYield));
        Assert.True(rewritten.InferenceType());
        Assert.Equal(function.CheckedType, rewritten.CheckedType);
    }

    private static (Var Lhs, Var Rhs, Var Addend, Call PackedMatMul) CreateSplitNPackedMatMul()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var lhs = new Var(
            "lhs",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }),
                new SBP[] { SBP.B, SBP.B },
                placement));
        var rhs = new Var(
            "rhs",
            new DistributedType(
                new TensorType(
                    new VectorType(DataTypes.BFloat16, [8, 2, 8]),
                    new long[] { 4, 16 }),
                new SBP[] { SBP.B, SBP.SContiguous([0, 1], 1) },
                placement));
        var outputWithoutAddend = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor));
        var addend = new Var("addend", outputWithoutAddend.CheckedType);
        var packed = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor,
            addend: addend));
        return (lhs, rhs, addend, packed);
    }
}
