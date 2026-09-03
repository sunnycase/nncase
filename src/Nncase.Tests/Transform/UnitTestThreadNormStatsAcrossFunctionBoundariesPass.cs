// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestThreadNormStatsAcrossFunctionBoundariesPass : TestClassBase
{
    [Fact]
    public async Task TestThreadStatsThroughRepeatedTensorReturningFunction()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new long[] { 2, 8 }));
        var scale = (Expr)Tensor.From(Enumerable.Repeat(1.25f, 8).ToArray(), [8]);
        var bias = (Expr)Tensor.From(Enumerable.Repeat(0.125f, 8).ToArray(), [8]);
        var inputStats = IR.F.NN.NormStats(1, layerInput, useMean: false);
        var normalized = IR.F.NN.NormApply(1, 1e-6f, layerInput, inputStats, scale, bias, useMean: false);
        var layer = new Function("decoder_layer", IR.F.Math.Add(layerInput, normalized), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new long[] { 2, 8 }));
        var call0 = new Call(layer, input);
        var call1 = new Call(layer, call0);
        var finalStats = IR.F.NN.NormStats(1, call1, useMean: false);
        var main = new Function(
            "main",
            IR.F.NN.NormApply(1, 1e-6f, call1, finalStats, scale, bias, useMean: false),
            input);
        Assert.True(main.InferenceType());
        var feeds = new Dictionary<IVar, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(Tensor.From(Enumerable.Range(0, 16).Select(value => value / 8.0f).ToArray(), [2, 8])) },
        };
        var expected = CompilerServices.Evaluate(main.Body, feeds);

        var module = new IRModule(main);
        module.Add(layer);
        var rewritten = await new ThreadNormStatsAcrossFunctionBoundariesPass().RunAsync(module, new());
        var rewrittenMain = Assert.IsType<Function>(rewritten.Entry);
        var rewrittenLayer = Assert.Single(rewritten.Functions.OfType<Function>().Where(function => function.Name == layer.Name));
        var layerBody = Assert.IsType<IR.Tuple>(rewrittenLayer.Body);
        Assert.Equal(2, layerBody.Count);
        Assert.Equal(2, rewrittenLayer.Parameters.Length);

        var layerCalls = ExprCollector.Collect(rewrittenMain.Body)
            .OfType<Call>()
            .Where(call => ReferenceEquals(call.Target, rewrittenLayer))
            .ToArray();
        Assert.Equal(2, layerCalls.Length);
        Assert.All(layerCalls, call => Assert.Contains(call.Users, user => user is Call { Target: GetItem }));
        Assert.Equal(expected, CompilerServices.Evaluate(rewrittenMain.Body, feeds));
    }

    [Fact]
    public async Task TestThreadStatsThroughRepeatedTupleReturningFunction()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new long[] { 2, 8 }));
        var layerToken = new Var("layer_token", new TensorType(DataTypes.Int32, new long[] { 1 }));
        var scale = (Expr)Tensor.From(Enumerable.Repeat(1.25f, 8).ToArray(), [8]);
        var bias = (Expr)Tensor.From(Enumerable.Repeat(0.125f, 8).ToArray(), [8]);
        var inputStats = IR.F.NN.NormStats(1, layerInput, useMean: false);
        var normalized = IR.F.NN.NormApply(1, 1e-6f, layerInput, inputStats, scale, bias, useMean: false);
        var layerOutput = IR.F.Math.Add(layerInput, normalized);
        var layer = new Function(
            "decoder_layer",
            new IR.Tuple(layerOutput, layerToken),
            layerInput,
            layerToken);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new long[] { 2, 8 }));
        var token = new Var("token", new TensorType(DataTypes.Int32, new long[] { 1 }));
        var call0 = new Call(layer, input, token);
        var call1 = new Call(layer, IR.F.Tensors.GetItem(call0, 0), IR.F.Tensors.GetItem(call0, 1));
        var finalValue = IR.F.Tensors.GetItem(call1, 0);
        var finalStats = IR.F.NN.NormStats(1, finalValue, useMean: false);
        var finalNormalized = IR.F.NN.NormApply(1, 1e-6f, finalValue, finalStats, scale, bias, useMean: false);
        var main = new Function(
            "main",
            new IR.Tuple(finalNormalized, IR.F.Tensors.GetItem(call1, 1)),
            input,
            token);
        Assert.True(main.InferenceType());
        var originalEntryType = main.CheckedType;

        var feeds = new Dictionary<IVar, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(Tensor.From(Enumerable.Range(0, 16).Select(value => value / 8.0f).ToArray(), [2, 8])) },
            { token, Value.FromTensor(Tensor.From(new[] { 7 }, [1])) },
        };
        var expected = CompilerServices.Evaluate(main.Body, feeds);

        var module = new IRModule(main);
        module.Add(layer);
        var rewritten = await new ThreadNormStatsAcrossFunctionBoundariesPass().RunAsync(module, new());
        var rewrittenMain = Assert.IsType<Function>(rewritten.Entry);
        var rewrittenLayer = Assert.Single(rewritten.Functions.OfType<Function>().Where(function => function.Name == layer.Name));

        Assert.Equal(originalEntryType, rewrittenMain.CheckedType);
        Assert.Equal(3, rewrittenLayer.Parameters.Length);
        var layerBody = Assert.IsType<IR.Tuple>(rewrittenLayer.Body);
        Assert.Equal(3, layerBody.Count);
        Assert.IsType<TensorType>(rewrittenLayer.Parameters[2].CheckedType);
        Assert.Equal(DataTypes.Float32, rewrittenLayer.Parameters[2].CheckedDataType);

        var layerStatsCalls = ExprCollector.Collect(rewrittenLayer.Body)
            .OfType<Call>()
            .Where(call => call.Target is NormStats)
            .ToArray();
        var outputStatsCall = Assert.Single(layerStatsCalls);
        Assert.Same(layerBody[0], outputStatsCall[NormStats.Input]);
        var statsBinding = Assert.Single(
            ExprCollector.Collect(rewrittenLayer.Body)
                .OfType<Call>()
                .Where(call => call.Target is BindNormStats));
        Assert.Same(rewrittenLayer.Parameters[0], statsBinding[BindNormStats.Input]);
        Assert.Same(rewrittenLayer.Parameters[2], statsBinding[BindNormStats.Stats]);

        var mainCalls = ExprCollector.Collect(rewrittenMain.Body).OfType<Call>().ToArray();
        var layerCalls = mainCalls.Where(call => ReferenceEquals(call.Target, rewrittenLayer)).ToArray();
        Assert.Equal(2, layerCalls.Length);
        Assert.All(layerCalls, call => Assert.Equal(3, call.Arguments.Length));
        var seedStats = Assert.IsType<Call>(layerCalls[0].Arguments[2]);
        Assert.IsType<NormStats>(seedStats.Target);
        var threadedStats = Assert.IsType<Call>(layerCalls[1].Arguments[2]);
        Assert.IsType<GetItem>(threadedStats.Target);
        Assert.Same(layerCalls[0], threadedStats[GetItem.Input]);
        Assert.Equal(2, Assert.IsType<DimConst>(threadedStats[GetItem.Index]).Value);
        Assert.Single(mainCalls.Where(call => call.Target is NormStats));

        var actual = CompilerServices.Evaluate(rewrittenMain.Body, feeds);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public async Task TestThreadStatsThroughPackedViewsAndDifferentFunctions()
    {
        Function MakeLayer(string name, float residualScale)
        {
            var layerInput = new Var(
                $"{name}_input",
                new TensorType(DataTypes.Float32, new long[] { 2, 8 }));
            var packedInput = IR.F.Tensors.Pack(layerInput, [4], [1]);
            var inputStats = IR.F.NN.NormStats(1, packedInput, useMean: false);
            var statsSum = IR.F.Tensors.Reduce(
                ReduceOp.Sum,
                inputStats,
                new long[] { 0, 1, 2 },
                0f,
                keepDims: false);
            var statsBroadcast = IR.F.Tensors.Broadcast(statsSum, new long[] { 2, 8 });
            var output = IR.F.Math.Add(
                layerInput,
                IR.F.Math.Mul(statsBroadcast, residualScale));
            var layer = new Function(
                name,
                output,
                layerInput);
            Assert.True(layer.InferenceType());
            return layer;
        }

        var linearLayer = MakeLayer("linear_decoder_layer", 0.25f);
        var fullLayer = MakeLayer("full_decoder_layer", 0.5f);
        var input = new Var("input", new TensorType(DataTypes.Float32, new long[] { 2, 8 }));
        var call0 = new Call(linearLayer, input);
        var call1 = new Call(linearLayer, call0);
        var call2 = new Call(fullLayer, call1);
        var packedFinal = IR.F.Tensors.Pack(call2, [4], [1]);
        var finalStats = IR.F.NN.NormStats(1, packedFinal, useMean: false);
        var finalOutput = IR.F.Tensors.Reduce(
            ReduceOp.Sum,
            finalStats,
            new long[] { 0, 1, 2 },
            0f,
            keepDims: false);
        var main = new Function(
            "main",
            finalOutput,
            input);
        Assert.True(main.InferenceType());

        var feeds = new Dictionary<IVar, IValue>(ReferenceEqualityComparer.Instance)
        {
            { input, Value.FromTensor(Tensor.From(Enumerable.Range(0, 16).Select(value => value / 8.0f).ToArray(), [2, 8])) },
        };
        var expected = CompilerServices.Evaluate(main.Body, feeds);

        var module = new IRModule(main);
        module.Add(linearLayer);
        module.Add(fullLayer);
        var rewritten = await new ThreadNormStatsAcrossFunctionBoundariesPass().RunAsync(module, new());
        var rewrittenMain = Assert.IsType<Function>(rewritten.Entry);
        var rewrittenLayers = rewritten.Functions
            .OfType<Function>()
            .Where(function => function.Name != main.Name)
            .ToArray();
        Assert.Equal(2, rewrittenLayers.Length);
        Assert.All(rewrittenLayers, layer =>
        {
            Assert.Equal(2, layer.Parameters.Length);
            Assert.Equal(2, Assert.IsType<IR.Tuple>(layer.Body).Count);
            Assert.Contains(
                ExprCollector.Collect(layer.Body).OfType<Call>(),
                call => call.Target is BindNormStats
                    && call[BindNormStats.Input] is Call { Target: Pack });
        });

        var rewrittenCalls = ExprCollector.Collect(rewrittenMain.Body)
            .OfType<Call>()
            .Where(call => call.Target is Function function && function.Name != main.Name)
            .ToArray();
        Assert.Equal(3, rewrittenCalls.Length);
        Assert.IsType<NormStats>(Assert.IsType<Call>(rewrittenCalls[0].Arguments[1]).Target);
        Assert.All(rewrittenCalls.Skip(1), call =>
        {
            var stats = Assert.IsType<Call>(call.Arguments[1]);
            Assert.IsType<GetItem>(stats.Target);
        });

        var actual = CompilerServices.Evaluate(rewrittenMain.Body, feeds);
        Assert.Equal(expected, actual);
    }
}
