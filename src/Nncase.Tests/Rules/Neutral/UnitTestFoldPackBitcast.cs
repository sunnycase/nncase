// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldPackBitcast : TransformTestBase
{
    [Fact]
    public void TestFoldPackAfterBitcastToScalar()
    {
        var input = new Var("input", new TensorType(new VectorType(DataTypes.BFloat16, [4, 8]), new RankedShape(2, 64)));
        var vectorView = Bitcast(input, new VectorType(DataTypes.BFloat16, [8]));
        var scalarView = Bitcast(vectorView, DataTypes.BFloat16);
        var repacked = Pack(scalarView, [8], [1]);
        CompilerServices.InferenceType(repacked);

        var post = (Expr)CompilerServices.Rewrite(repacked, [new FoldPackBitcast()], new());
        CompilerServices.InferenceType(post);

        Assert.Equal(vectorView.CheckedType, post.CheckedType);
        Assert.IsType<IR.Tensors.Bitcast>(((Call)post).Target);
    }

    [Fact]
    public void TestFoldPackAfterTupleGetItemBitcastToScalar()
    {
        var input = new Var("input", new TensorType(new VectorType(DataTypes.BFloat16, [4, 8]), new RankedShape(2, 64)));
        var vectorView = Bitcast(input, new VectorType(DataTypes.BFloat16, [8]));
        var scalarView = Bitcast(vectorView, DataTypes.BFloat16);
        var tuple = new IR.Tuple(scalarView, scalarView, scalarView);
        var repacked = Pack(GetItem(tuple, 1), [8], [1]);
        CompilerServices.InferenceType(repacked);

        var post = (Expr)CompilerServices.Rewrite(repacked, [new FoldGetItemTuple()], new());
        post = (Expr)CompilerServices.Rewrite(post, [new FoldPackBitcast()], new());
        CompilerServices.InferenceType(post);

        Assert.Equal(vectorView.CheckedType, post.CheckedType);
        Assert.IsType<IR.Tensors.Bitcast>(((Call)post).Target);
    }

    [Fact]
    public void TestFoldPackAfterReshapeBitcastToScalar()
    {
        var input = new Var("input", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(2, 128)));
        var scalarView = Bitcast(input, DataTypes.BFloat16);
        var reshaped = Reshape(scalarView, new RankedShape(2, 8, 128));
        var repacked = Pack(reshaped, [8], [2]);
        CompilerServices.InferenceType(repacked);

        var post = (Expr)CompilerServices.Rewrite(repacked, [new FoldPackReshapeBitcast()], new());
        CompilerServices.InferenceType(post);

        Assert.Equal(repacked.CheckedType, post.CheckedType);
        Assert.IsType<IR.Tensors.Reshape>(((Call)post).Target);
        Assert.True(ReferenceEquals(input, ((Call)post).Arguments[IR.Tensors.Reshape.Input.Index]));
    }
}
