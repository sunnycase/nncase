// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

/// <inheritdoc />
[AutoSetupTestMethod(InitSession = true)]
public class UnitTestDecomposeLayerNorm : TransformTestBase
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TestDecomposeLayerNormWithDynamicParameters(bool useMean)
    {
        var input = Testing.Rand<float>(2, 3, 4);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var scale = Testing.Rand<float>(3, 4);
        var scaleVar = new Var(new TensorType(scale.ElementType, scale.Shape));
        var bias = Testing.Rand<float>(3, 4);
        var biasVar = new Var(new TensorType(bias.ElementType, bias.Shape));

        var expr = IR.F.NN.LayerNorm(1, 1e-5f, inputVar, scaleVar, biasVar, useMean);

        var post = TestMatched<DecomposeLayerNorm>(
            expr,
            new Dictionary<IVar, IValue>
            {
                { inputVar, Value.FromTensor(input) },
                { scaleVar, Value.FromTensor(scale) },
                { biasVar, Value.FromTensor(bias) },
            });
        var apply = Assert.IsType<Call>(post);
        Assert.IsType<IR.NN.NormApply>(apply.Target);
        Assert.Contains(apply.Arguments.ToArray(), arg => arg is Call { Target: IR.NN.NormStats });
    }
}
