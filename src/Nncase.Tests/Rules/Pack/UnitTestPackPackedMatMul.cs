// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.NTT;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestVectorizeVectorizedMatMul : TransformTestBase
{
    [Fact]
    public void TestPackMatMulByNDoesNotRequireOuterUnpack()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(8, 32)));
        var rhs = new Var("rhs", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(32, 8)));
        var expr = VectorizedMatMul(lhs, rhs, [], new[] { 1 }, outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = (Expr)CompilerServices.Rewrite(expr, [new PackMatMulByN(4)], context);
        CompilerServices.InferenceType(post);
        var printed = CompilerServices.Print(post);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Contains("PackedMatMul", printed, System.StringComparison.Ordinal);
        Assert.Contains("Unpack(Lanes: {4}", printed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("Unpack(Lanes: {8}", printed, System.StringComparison.Ordinal);
        var packedMatMul = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(call => call.Target is IR.NTT.PackedMatMul));
        Assert.Equal(
            IR.NTT.PackedMatMulRhsLayout.NMajor,
            Assert.IsType<IR.NTT.PackedMatMul>(packedMatMul.Target).RhsLayout);
    }

    [Fact]
    public void TestPackMatMulRhsKMajorPreservesVectorizedResultType()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var rhs = new Var("rhs", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(64, 16)));
        var expr = VectorizedMatMul(lhs, rhs, [], new[] { 1 }, outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = (Expr)CompilerServices.Rewrite(expr, [new PackMatMulRhsKMajor(16, 2)], context);
        CompilerServices.InferenceType(post);
        var packedMatMul = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(call => call.Target is IR.NTT.PackedMatMul));
        var packedOp = Assert.IsType<IR.NTT.PackedMatMul>(packedMatMul.Target);
        var packedRhs = (Expr)packedMatMul.Arguments[IR.NTT.PackedMatMul.Rhs.Index];

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, packedOp.RhsLayout);
        Assert.Equal(new RankedShape(4, 16), packedRhs.CheckedShape);
        Assert.Equal(new VectorType(DataTypes.BFloat16, [8, 2, 8]), packedRhs.CheckedDataType);
    }

    [Fact]
    public void TestPackQKVParallelLinearByNUsesSinglePackedQKVOp()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var qWeight = new Var("q_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 64)));
        var kWeight = new Var("k_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var vWeight = new Var("v_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var expr = QKVParallelLinear(
            input,
            qWeight,
            kWeight,
            vWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            4,
            2,
            DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(expr, [new PackQKVParallelLinearByN(4, 16)], context);
        CompilerServices.InferenceType(post);
        var printed = CompilerServices.Print(post);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Contains("PackedQKVParallelLinear", printed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("PackedMatMul", printed, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackQKVParallelLinearRhsKMajorMatchesPackedMatMulLayout()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var qWeight = new Var("q_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 64)));
        var kWeight = new Var("k_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var vWeight = new Var("v_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var qBias = new Var("q_bias", new TensorType(DataTypes.BFloat16, new RankedShape(64)));
        var kBias = new Var("k_bias", new TensorType(DataTypes.BFloat16, new RankedShape(32)));
        var vBias = new Var("v_bias", new TensorType(DataTypes.BFloat16, new RankedShape(32)));
        var expr = QKVParallelLinear(
            input,
            qWeight,
            kWeight,
            vWeight,
            qBias,
            kBias,
            vBias,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            4,
            2,
            DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(
            expr,
            [new PackQKVParallelLinearRhsKMajor(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedQKVParallelLinear));
        var packedOp = Assert.IsType<IR.NTT.PackedQKVParallelLinear>(packedCall.Target);
        var expectedWeightType = new VectorType(DataTypes.BFloat16, [8, 2, 8]);
        var expectedOutputType = new VectorType(DataTypes.BFloat16, [8]);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, packedOp.RhsLayout);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.QWeight.Index, new RankedShape(4, 8), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.KWeight.Index, new RankedShape(4, 4), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.VWeight.Index, new RankedShape(4, 4), expectedWeightType);
        Assert.Equal(
            new TupleType(
            [
                new TensorType(expectedOutputType, new RankedShape(1, 8)),
                new TensorType(expectedOutputType, new RankedShape(1, 4)),
                new TensorType(expectedOutputType, new RankedShape(1, 4)),
            ]),
            packedCall.CheckedType);

        void AssertPackedBuffer(int index, RankedShape shape, DataType dataType)
        {
            var argument = (Expr)packedCall.Arguments[index];
            Assert.Equal(shape, argument.CheckedShape);
            Assert.Equal(dataType, argument.CheckedDataType);
        }
    }

    [Fact]
    public void TestPackMatMulGluByNUsesSinglePackedMatMulGluOp()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var gateWeight = new Var("gate_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var upWeight = new Var("up_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var expr = MatMulGlu(
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
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(expr, [new PackMatMulGluByN(4, 16)], context);
        CompilerServices.InferenceType(post);
        var printed = CompilerServices.Print(post);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Contains("PackedMatMulGlu", printed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("PackedMatMul(", printed, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackMatMulGluRhsKMajorMatchesPackedMatMulLayout()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var gateWeight = new Var("gate_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var upWeight = new Var("up_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var gateBias = new Var("gate_bias", new TensorType(DataTypes.BFloat16, new RankedShape(128)));
        var upBias = new Var("up_bias", new TensorType(DataTypes.BFloat16, new RankedShape(128)));
        var expr = MatMulGlu(
            input,
            gateWeight,
            upWeight,
            gateBias,
            upBias,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(
            expr,
            [new PackMatMulGluRhsKMajor(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedMatMulGlu));
        var packedOp = Assert.IsType<IR.NTT.PackedMatMulGlu>(packedCall.Target);
        var expectedWeightType = new VectorType(DataTypes.BFloat16, [8, 2, 8]);
        var expectedOutputType = new VectorType(DataTypes.BFloat16, [8]);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, packedOp.RhsLayout);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.GateWeight.Index, new RankedShape(4, 16), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.UpWeight.Index, new RankedShape(4, 16), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.GateBias.Index, new RankedShape(16), expectedOutputType);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.UpBias.Index, new RankedShape(16), expectedOutputType);
        Assert.Equal(new TensorType(expectedOutputType, new RankedShape(1, 16)), packedCall.CheckedType);

        void AssertPackedBuffer(int index, RankedShape shape, DataType dataType)
        {
            var argument = (Expr)packedCall.Arguments[index];
            Assert.Equal(shape, argument.CheckedShape);
            Assert.Equal(dataType, argument.CheckedDataType);
        }
    }

    [Fact]
    public void TestVectorizedMatMulDevectorizePropagation()
    {
        var lhs = Pack(Testing.Rand<float>(3, 24), [8], [1]).Evaluate().AsTensor();
        var lhsVar = new Var(new TensorType(lhs.ElementType, lhs.Shape));
        var rhs = Pack(Testing.Rand<float>(24, 24), [8], [1]).Evaluate().AsTensor();
        var expr = Unpack(lhsVar, [8], [1]);
        expr = VectorizedMatMul(expr, rhs, [], new int[] { 1 });
        expr = Unpack(expr, [8], [1]);
        TestMatched<VectorizedMatMulDevectorizePropagation>(
            expr,
            new Dictionary<IVar, IValue> {
                { lhsVar, Value.FromTensor(lhs) },
            });
    }
}
