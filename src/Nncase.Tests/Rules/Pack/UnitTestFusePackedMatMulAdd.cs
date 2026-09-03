// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFusePackedMatMulAdd : TransformTestBase
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TestFuseExactDistributedAddend(bool matmulOnRight)
    {
        var (packedMatMul, addend) = CreateDistributedPackedMatMul();
        var semanticRegion = new SemanticRegion(
            SemanticRegionKinds.Attention,
            "model.layers.0.linear_attn");
        packedMatMul.Metadata.SemanticRegion = semanticRegion;
        var expression = matmulOnRight
            ? IR.F.Math.Add(addend, packedMatMul)
            : IR.F.Math.Add(packedMatMul, addend);
        CompilerServices.InferenceType(expression);

        var rewritten = Rewrite(expression);
        var call = Assert.IsType<Call>(rewritten);
        Assert.IsType<IR.NTT.PackedMatMul>(call.Target);
        Assert.Same(addend, call[IR.NTT.PackedMatMul.Addend]);
        Assert.Equal(expression.CheckedType, call.CheckedType);
        Assert.Same(semanticRegion, call.Metadata.SemanticRegion);
    }

    [Fact]
    public void TestRejectBroadcastAddend()
    {
        var (packedMatMul, _) = CreateTensorPackedMatMul();
        var broadcastAddend = new Var(
            "broadcast_addend",
            new TensorType(
                new VectorType(DataTypes.BFloat16, [8]),
                new RankedShape(1, 1)));
        var expression = IR.F.Math.Add(packedMatMul, broadcastAddend);
        CompilerServices.InferenceType(expression);

        var rewritten = Rewrite(expression);
        Assert.IsType<IR.Math.Binary>(Assert.IsType<Call>(rewritten).Target);
    }

    [Fact]
    public void TestRejectFusedReduce()
    {
        var (packedMatMul, addend) = CreateTensorPackedMatMul(fusedReduce: true);
        var expression = IR.F.Math.Add(packedMatMul, addend);
        CompilerServices.InferenceType(expression);

        var rewritten = Rewrite(expression);
        Assert.IsType<IR.Math.Binary>(Assert.IsType<Call>(rewritten).Target);
    }

    [Fact]
    public void TestRejectMultiUsePackedMatMul()
    {
        var (packedMatMul, addend) = CreateTensorPackedMatMul();
        var sum = IR.F.Math.Add(packedMatMul, addend);
        var expression = new IR.Tuple(packedMatMul, sum);
        CompilerServices.InferenceType(expression);

        var rewritten = Rewrite(expression);
        var calls = ExprCollector.Collect(rewritten).OfType<Call>().ToArray();
        Assert.Contains(calls, call => call.Target is IR.Math.Binary);
        Assert.All(
            calls.Where(call => call.Target is IR.NTT.PackedMatMul),
            call => Assert.IsType<None>(call[IR.NTT.PackedMatMul.Addend]));
    }

    [Fact]
    public void TestFuseAddendThroughShardedView()
    {
        var (packedMatMul, _) = CreateDistributedPackedMatMul();
        var packedType = Assert.IsType<DistributedType>(packedMatMul.CheckedType);
        var broadcastType = new DistributedType(
            packedType.TensorType,
            Enumerable.Repeat<SBP>(SBP.B, packedType.TensorType.Shape.Rank).ToArray(),
            packedType.Placement);
        var viewed = IR.F.Distributed.ShardedView(packedMatMul, broadcastType);
        var addend = new Var("broadcast_addend", broadcastType);
        var expression = IR.F.Math.Add(viewed, addend);

        var rewritten = CompilerServices.Rewrite(
            expression,
            [new FusePackedMatMulAddThroughShardedView()],
            new Nncase.Passes.RunPassContext());
        Assert.True(CompilerServices.InferenceType(rewritten));
        Assert.Equal(expression.CheckedType, rewritten.CheckedType);

        var outputView = Assert.IsType<Call>(rewritten);
        Assert.IsType<IR.Distributed.ShardedView>(outputView.Target);
        var fused = Assert.IsType<Call>(outputView[IR.Distributed.ShardedView.Input]);
        Assert.IsType<IR.NTT.PackedMatMul>(fused.Target);
        var localAddend = Assert.IsType<Call>(fused[IR.NTT.PackedMatMul.Addend]);
        Assert.IsType<IR.Distributed.ShardedView>(localAddend.Target);
        Assert.Same(addend, localAddend[IR.Distributed.ShardedView.Input]);
        Assert.Equal(packedType, localAddend.CheckedType);
    }

    [Fact]
    public void TestPackedMatMulRejectsMismatchedAddendType()
    {
        var (packedMatMul, _) = CreateTensorPackedMatMul();
        var op = Assert.IsType<IR.NTT.PackedMatMul>(packedMatMul.Target);
        var invalid = IR.F.NTT.PackedMatMul(
            (Expr)packedMatMul[IR.NTT.PackedMatMul.Lhs],
            (Expr)packedMatMul[IR.NTT.PackedMatMul.Rhs],
            op.FusedReduce,
            op.OutputDataType,
            (Expr)packedMatMul[IR.NTT.PackedMatMul.Scale],
            op.RhsLayout,
            new Var(
                "invalid_addend",
                new TensorType(
                    new VectorType(DataTypes.BFloat16, [8]),
                    new RankedShape(1, 1))));

        Assert.IsType<InvalidType>(invalid.CheckedType);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TestFusePackedNVFP4MatMulExactAddend(bool matmulOnRight)
    {
        var (packedMatMul, addend) = CreateTensorPackedNVFP4MatMul();
        var expression = matmulOnRight
            ? IR.F.Math.Add(addend, packedMatMul)
            : IR.F.Math.Add(packedMatMul, addend);
        Assert.True(expression.InferenceType());

        var rewritten = CompilerServices.Rewrite(
            expression,
            [new FusePackedNVFP4MatMulAdd()],
            new Nncase.Passes.RunPassContext());
        Assert.True(rewritten.InferenceType());

        var call = Assert.IsType<Call>(rewritten);
        Assert.IsType<IR.NTT.PackedNVFP4MatMul>(call.Target);
        Assert.Same(addend, call[IR.NTT.PackedNVFP4MatMul.Addend]);
        Assert.Equal(expression.CheckedType, call.CheckedType);
    }

    [Fact]
    public void TestPackedNVFP4MatMulRejectsMismatchedAddendType()
    {
        var (packedMatMul, _) = CreateTensorPackedNVFP4MatMul();
        var op = Assert.IsType<IR.NTT.PackedNVFP4MatMul>(packedMatMul.Target);
        var invalid = IR.F.NTT.PackedNVFP4MatMul(
            (Expr)packedMatMul[IR.NTT.PackedNVFP4MatMul.Lhs],
            (Expr)packedMatMul[IR.NTT.PackedNVFP4MatMul.RhsPacked],
            (Expr)packedMatMul[IR.NTT.PackedNVFP4MatMul.RhsScale],
            (Expr)packedMatMul[IR.NTT.PackedNVFP4MatMul.LhsGlobalScale],
            (Expr)packedMatMul[IR.NTT.PackedNVFP4MatMul.RhsGlobalScale],
            op.OutputDataType,
            op.GroupSize,
            op.InputKVectorLaneCount,
            op.RhsKPackLaneCount,
            op.RhsKVectorLaneCount,
            op.OutputNVectorLaneCount,
            new Var(
                "invalid_addend",
                new TensorType(
                    new VectorType(DataTypes.BFloat16, [8]),
                    new RankedShape(1, 1))));

        Assert.IsType<InvalidType>(invalid.CheckedType);
    }

    private static BaseExpr Rewrite(BaseExpr expression)
    {
        var rewritten = CompilerServices.Rewrite(
            expression,
            [new FusePackedMatMulAdd()],
            new Nncase.Passes.RunPassContext());
        CompilerServices.InferenceType(rewritten);
        return rewritten;
    }

    private static (Call PackedMatMul, Var Addend) CreateTensorPackedMatMul(
        bool fusedReduce = false)
    {
        var lhs = new Var(
            "lhs",
            new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var rhs = new Var(
            "rhs",
            new TensorType(
                new VectorType(DataTypes.BFloat16, [8, 2, 8]),
                new RankedShape(4, 16)));
        var packedMatMul = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            fusedReduce,
            DataTypes.BFloat16,
            rhsLayout: IR.NTT.PackedMatMulRhsLayout.KMajor));
        var addend = new Var("addend", packedMatMul.CheckedType);
        return (packedMatMul, addend);
    }

    private static (Call PackedMatMul, Var Addend) CreateDistributedPackedMatMul()
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
        var packedMatMul = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: IR.NTT.PackedMatMulRhsLayout.KMajor));
        var addend = new Var("addend", packedMatMul.CheckedType);
        return (packedMatMul, addend);
    }

    private static (Call PackedMatMul, Var Addend) CreateTensorPackedNVFP4MatMul()
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
        var packedMatMul = Assert.IsType<Call>(IR.F.NTT.PackedNVFP4MatMul(
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
        Assert.True(packedMatMul.InferenceType());
        var addend = new Var("addend", packedMatMul.CheckedType);
        return (packedMatMul, addend);
    }
}
