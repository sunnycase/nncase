// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFoldTopKSoftmaxNormalization : TestClassBase
{
    [Fact]
    public async Task TestFoldPreservesNormalizedTopKValuesAndIndices()
    {
        var logits = new Var("logits", new TensorType(DataTypes.Float32, new[] { 2, 8 }));
        var function = CreateFunction(logits, observeUnnormalizedValues: false);
        Assert.True(function.InferenceType());
        var input = Tensor.From(
            new float[]
            {
                1.5f, -0.25f, 3.0f, 0.75f, -2.0f, 2.25f, 0.0f, 1.0f,
                -1.0f, 0.5f, 0.25f, 4.0f, 3.5f, -0.5f, 2.0f, 1.25f,
            },
            [2, 8]);
        var arguments = new Dictionary<IVar, IValue>
        {
            { logits, Value.FromTensor(input) },
        };
        var expected = function.Body.Evaluate(arguments).AsTensors();

        var rewritten = Assert.IsType<Function>(
            await new FoldTopKSoftmaxNormalizationPass().RunAsync(function, new()));
        var actual = rewritten.Body.Evaluate(arguments).AsTensors();

        Assert.Equal(expected[1].ToArray<long>(), actual[1].ToArray<long>());
        Assert.Equal(expected[0].Shape, actual[0].Shape);
        foreach (var (expectedValue, actualValue) in expected[0].ToArray<float>().Zip(actual[0].ToArray<float>()))
        {
            Assert.InRange(MathF.Abs(expectedValue - actualValue), 0.0f, 1e-6f);
        }

        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var topK = Assert.Single(calls.Where(call => call.Target is TopK));
        Assert.Same(logits, topK[TopK.X]);
        var selectedSoftmax = Assert.Single(calls.Where(call => call.Target is Softmax));
        var selectedValues = Assert.IsType<Call>(selectedSoftmax[Softmax.Input]);
        Assert.IsType<GetItem>(selectedValues.Target);
        Assert.Same(topK, selectedValues[GetItem.Input]);
        Assert.DoesNotContain(calls, call => call.Target is Reduce or Binary { BinaryOp: BinaryOp.Div });
        Assert.Equal(function.CheckedType, rewritten.CheckedType);
    }

    [Fact]
    public async Task TestKeepOriginalGraphWhenTopKValuesHaveAnotherObserver()
    {
        var logits = new Var("logits", new TensorType(DataTypes.Float32, new[] { 2, 8 }));
        var function = CreateFunction(logits, observeUnnormalizedValues: true);
        Assert.True(function.InferenceType());

        var rewritten = await new FoldTopKSoftmaxNormalizationPass().RunAsync(function, new());

        Assert.Same(function, rewritten);
        var topK = Assert.Single(
            ExprCollector.Collect(function.Body).OfType<Call>().Where(call => call.Target is TopK));
        var softmax = Assert.IsType<Call>(topK[TopK.X]);
        Assert.IsType<Softmax>(softmax.Target);
        Assert.Same(logits, softmax[Softmax.Input]);
    }

    [Fact]
    public async Task TestSharesSelectedSoftmaxAcrossEquivalentNormalizations()
    {
        var logits = new Var("logits", new TensorType(DataTypes.Float32, new[] { 2, 8 }));
        var probabilities = IR.F.NN.Softmax(logits, -1);
        var topK = IR.F.Tensors.TopK(
            probabilities,
            Tensor.FromScalar(DataTypes.Int64, 4, [1]),
            -1,
            true,
            true);
        var values = topK[0];
        var firstSum = IR.F.Tensors.Reduce(ReduceOp.Sum, values, new long[] { -1 }, 0.0f, true);
        var secondSum = IR.F.Tensors.Reduce(ReduceOp.Sum, values, new long[] { -1 }, 0.0f, true);
        var function = new Function(
            "router",
            string.Empty,
            new IR.Tuple(values / firstSum, values / secondSum, topK[1]),
            new[] { logits });
        Assert.True(function.InferenceType());

        var rewritten = Assert.IsType<Function>(
            await new FoldTopKSoftmaxNormalizationPass().RunAsync(function, new()));
        var output = Assert.IsType<IR.Tuple>(rewritten.Body);

        Assert.Same(output[0], output[1]);
        Assert.Single(ExprCollector.Collect(rewritten.Body).OfType<Call>(), call => call.Target is Softmax);
    }

    private static Function CreateFunction(Var logits, bool observeUnnormalizedValues)
    {
        var probabilities = IR.F.NN.Softmax(logits, -1);
        var topK = IR.F.Tensors.TopK(
            probabilities,
            Tensor.FromScalar(DataTypes.Int64, 4, [1]),
            -1,
            true,
            true);
        var values = topK[0];
        var indices = topK[1];
        var sum = IR.F.Tensors.Reduce(ReduceOp.Sum, values, new long[] { -1 }, 0.0f, true);
        var normalized = values / sum;
        var output = observeUnnormalizedValues
            ? new IR.Tuple(normalized, indices, values)
            : new IR.Tuple(normalized, indices);
        return new Function("router", string.Empty, output, new[] { logits });
    }
}
