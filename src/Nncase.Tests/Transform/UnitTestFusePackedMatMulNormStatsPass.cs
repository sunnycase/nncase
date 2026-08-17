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
            await new FusePackedMatMulNormStatsPass().RunAsync(function, new()));
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
            await new FusePackedMatMulNormStatsPass().RunAsync(function, new()));
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
            await new FusePackedMatMulNormStatsPass().RunAsync(function, new()));
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
