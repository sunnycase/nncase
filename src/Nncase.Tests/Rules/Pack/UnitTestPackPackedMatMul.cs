// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
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
