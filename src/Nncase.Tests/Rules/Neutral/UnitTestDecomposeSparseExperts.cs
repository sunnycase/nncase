// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestDecomposeSparseExperts : TransformTestBase
{
    [Fact]
    public void TestDecomposeIntoMaterializedGateUpAndDownStages()
    {
        const int hidden = 8;
        const int intermediate = 4;
        const int experts = 2;
        const int topK = 2;
        var qValue = Testing.Rand<float>(1, hidden);
        var gateValue = Testing.Rand<float>(experts, intermediate, hidden);
        var downValue = Testing.Rand<float>(experts, hidden, intermediate);
        var upValue = Testing.Rand<float>(experts, intermediate, hidden);
        var q = new Var("q", new TensorType(qValue.ElementType, qValue.Shape));
        var gate = new Var("gate", new TensorType(gateValue.ElementType, gateValue.Shape));
        var down = new Var("down", new TensorType(downValue.ElementType, downValue.Shape));
        var up = new Var("up", new TensorType(upValue.ElementType, upValue.Shape));
        var ids = Tensor.From(new long[] { 0, 1 }, [1, topK]);
        var routerWeights = Tensor.From(new float[] { 0.4F, 0.6F }, [1, topK]);
        var scales = Tensor.Ones(DataTypes.Float32, [experts, 1]);
        var expr = IR.F.NN.SparseExperts(
            q,
            ids,
            routerWeights,
            scales,
            gate,
            scales,
            scales,
            down,
            scales,
            scales,
            up,
            scales,
            hidden,
            intermediate,
            experts,
            topK,
            1);

        var post = TestMatched<DecomposeSparseExperts>(
            expr,
            new Dictionary<IVar, IValue>
            {
                { q, Value.FromTensor(qValue) },
                { gate, Value.FromTensor(gateValue) },
                { down, Value.FromTensor(downValue) },
                { up, Value.FromTensor(upValue) },
            });
        var downCall = Assert.IsType<Call>(post);
        Assert.IsType<IR.NN.SparseExpertsDown>(downCall.Target);
        var gateUpCall = Assert.IsType<Call>(downCall[IR.NN.SparseExpertsDown.Activations]);
        Assert.IsType<IR.NN.SparseExpertsGateUp>(gateUpCall.Target);
        Assert.Equal(new long[] { 1, topK, intermediate }, gateUpCall.CheckedShape.ToValueArray());
        Assert.Equal(new long[] { 1, hidden }, downCall.CheckedShape.ToValueArray());
    }
}
