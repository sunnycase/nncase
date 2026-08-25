// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestSinkNormStatsBoxingAcrossFunctionBoundariesPass : TestClassBase
{
    [Fact]
    public async Task TestConvertInitialMaterializedSeedAndRepeatedPartialStats()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var inputTensorType = new TensorType(DataTypes.Float32, new long[] { 2, 8 });
        var statsTensorType = new TensorType(DataTypes.Float32, new long[] { 1, 2, 1 });
        var parameterTensorType = new TensorType(DataTypes.Float32, new long[] { 8 });
        var inputType = new DistributedType(inputTensorType, [SBP.B, SBP.B], placement);
        var localInputType = new DistributedType(
            inputTensorType,
            [SBP.B, SBP.SContiguous([0, 1])],
            placement);
        var materializedType = new DistributedType(
            statsTensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var partialType = materializedType with { Partial = SBP.P([0, 1], ReduceOp.Sum) };
        var parameterType = new DistributedType(
            parameterTensorType,
            [SBP.SContiguous([0, 1])],
            placement);

        var layerInput = new Var("layer_input", inputType);
        var layerStats = new Var("layer_stats", materializedType);
        var layerScale = new Var("layer_scale", parameterType);
        var layerBias = new Var("layer_bias", parameterType);
        var localInput = IR.F.Distributed.ShardedView(layerInput, localInputType);
        var boundStats = IR.F.NN.BindNormStats(1, layerInput, layerStats, useMean: false);
        var normalized = IR.F.NN.NormApply(
            1,
            1e-6f,
            localInput,
            boundStats,
            layerScale,
            layerBias,
            useMean: false);
        var layer = new Function(
            "layer",
            "pyntt",
            normalized,
            new IVar[] { layerInput, layerStats, layerScale, layerBias });
        Assert.True(layer.InferenceType());

        var input0 = new Var("input0", inputType);
        var input1 = new Var("input1", inputType);
        var partial = new Var("partial", partialType);
        var scale = new Var("scale", parameterType);
        var bias = new Var("bias", parameterType);
        var seed = IR.F.NN.NormStats(1, input0, useMean: false);
        var call0 = new Call(layer, input0, seed, scale, bias);
        var call1 = new Call(layer, input1, IR.F.Distributed.Boxing(partial, materializedType), scale, bias);
        var main = new Function(
            "main",
            "pyntt",
            new IR.Tuple(call0, call1),
            new IVar[] { input0, input1, partial, scale, bias });
        Assert.True(main.InferenceType());
        var module = new IRModule(main);
        module.Add(layer);

        var rewritten = await new SinkNormStatsBoxingAcrossFunctionBoundariesPass().RunAsync(module, new());

        var rewrittenLayer = Assert.Single(
            rewritten.Functions.OfType<Function>().Where(function => function.Name == "layer"));
        Assert.Equal(partialType, rewrittenLayer.Parameters[1].CheckedType);
        var binding = Assert.Single(
            ExprCollector.Collect(rewrittenLayer.Body)
                .OfType<Call>()
                .Where(call => call.Target is BindNormStats));
        var sunkBoxing = Assert.IsType<Call>(binding[BindNormStats.Stats]);
        Assert.IsType<Boxing>(sunkBoxing.Target);
        Assert.Equal(materializedType, sunkBoxing.CheckedType);
        Assert.Same(rewrittenLayer.Parameters[1], sunkBoxing[Boxing.Input]);

        var rewrittenMain = Assert.IsType<Function>(rewritten.Entry);
        var calls = ExprCollector.Collect(rewrittenMain.Body)
            .OfType<Call>()
            .Where(call => ReferenceEquals(call.Target, rewrittenLayer))
            .ToArray();
        Assert.Equal(2, calls.Length);
        Assert.All(calls, call => Assert.Equal(partialType, call.Arguments[1].CheckedType));
        var partialSeed = Assert.IsType<Call>(calls[0].Arguments[1]);
        Assert.IsType<NormStats>(partialSeed.Target);
        Assert.IsType<ShardedView>(Assert.IsType<Call>(partialSeed[NormStats.Input]).Target);
        Assert.Same(partial, calls[1].Arguments[1]);
    }

    [Fact]
    public async Task TestSpecializeCalleeForDifferentInitialAndRecurrentPartialAxes()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var inputTensorType = new TensorType(DataTypes.Float32, new long[] { 2, 8 });
        var statsTensorType = new TensorType(DataTypes.Float32, new long[] { 1, 2, 1 });
        var parameterTensorType = new TensorType(DataTypes.Float32, new long[] { 8 });
        var inputType = new DistributedType(inputTensorType, [SBP.B, SBP.B], placement);
        var localInputType = new DistributedType(
            inputTensorType,
            [SBP.B, SBP.SContiguous([0])],
            placement);
        var materializedType = new DistributedType(
            statsTensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var initialPartialType = materializedType with { Partial = SBP.P([0], ReduceOp.Sum) };
        var recurrentPartialType = materializedType with { Partial = SBP.P([0, 1], ReduceOp.Sum) };
        var parameterType = new DistributedType(
            parameterTensorType,
            [SBP.SContiguous([0])],
            placement);

        var layerInput = new Var("layer_input", inputType);
        var layerStats = new Var("layer_stats", materializedType);
        var layerScale = new Var("layer_scale", parameterType);
        var layerBias = new Var("layer_bias", parameterType);
        var localInput = IR.F.Distributed.ShardedView(layerInput, localInputType);
        var boundStats = IR.F.NN.BindNormStats(1, layerInput, layerStats, useMean: false);
        var normalized = IR.F.NN.NormApply(
            1,
            1e-6f,
            localInput,
            boundStats,
            layerScale,
            layerBias,
            useMean: false);
        var layer = new Function(
            "layer",
            "pyntt",
            normalized,
            new IVar[] { layerInput, layerStats, layerScale, layerBias });
        Assert.True(layer.InferenceType());

        var input0 = new Var("input0", inputType);
        var input1 = new Var("input1", inputType);
        var recurrentPartial = new Var("recurrent_partial", recurrentPartialType);
        var scale = new Var("scale", parameterType);
        var bias = new Var("bias", parameterType);
        var seed = IR.F.NN.NormStats(1, input0, useMean: false);
        var call0 = new Call(layer, input0, seed, scale, bias);
        var call1 = new Call(
            layer,
            input1,
            IR.F.Distributed.Boxing(recurrentPartial, materializedType),
            scale,
            bias);
        var main = new Function(
            "main",
            "pyntt",
            new IR.Tuple(call0, call1),
            new IVar[] { input0, input1, recurrentPartial, scale, bias });
        Assert.True(main.InferenceType());
        var module = new IRModule(main);
        module.Add(layer);

        var rewritten = await new SinkNormStatsBoxingAcrossFunctionBoundariesPass().RunAsync(module, new());

        var layerVariants = rewritten.Functions.OfType<Function>()
            .Where(function => function.Name.StartsWith("layer", System.StringComparison.Ordinal))
            .ToArray();
        Assert.Equal(2, layerVariants.Length);
        Assert.Equal(
            new IRType[] { initialPartialType, recurrentPartialType },
            layerVariants.Select(function => function.Parameters[1].CheckedType).ToArray());
        Assert.All(
            layerVariants,
            variant =>
            {
                var boxing = Assert.Single(
                    ExprCollector.Collect(variant.Body)
                        .OfType<Call>()
                        .Where(call => call.Target is Boxing));
                Assert.Equal(materializedType, boxing.CheckedType);
                Assert.Same(variant.Parameters[1], boxing[Boxing.Input]);
            });

        var rewrittenMain = Assert.IsType<Function>(rewritten.Entry);
        var calls = ExprCollector.Collect(rewrittenMain.Body)
            .OfType<Call>()
            .Where(call => call.Target is Function function && layerVariants.Contains(function))
            .ToArray();
        Assert.Equal(2, calls.Length);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewrittenMain.Body).OfType<Call>(),
            call => call.Target is Boxing && Equals(call.CheckedType, materializedType));
        Assert.Equal(initialPartialType, calls[0].Arguments[1].CheckedType);
        Assert.IsType<NormStats>(Assert.IsType<Call>(calls[0].Arguments[1]).Target);
        Assert.Equal(recurrentPartialType, calls[1].Arguments[1].CheckedType);
        Assert.Same(recurrentPartial, calls[1].Arguments[1]);
    }

    [Fact]
    public async Task TestSinkConsistentPartialStatsIntoRepeatedCallee()
    {
        var fixture = CreateModule(additionalStatsConsumer: false, inconsistentPartialTypes: false);

        var rewritten = await new SinkNormStatsBoxingAcrossFunctionBoundariesPass().RunAsync(
            fixture.Module,
            new());

        var layer = Assert.Single(rewritten.Functions.OfType<Function>().Where(function => function.Name == "layer"));
        Assert.Equal(fixture.FirstPartialType, layer.Parameters[1].CheckedType);
        var normApplyCall = Assert.Single(
            ExprCollector.Collect(layer.Body)
                .OfType<Call>()
                .Where(call => call.Target is NormApply));
        var sunkBoxingCall = Assert.IsType<Call>(normApplyCall[NormApply.Stats]);
        var sunkBoxing = Assert.IsType<Boxing>(sunkBoxingCall.Target);
        Assert.Equal(fixture.MaterializedType, sunkBoxing.NewType);
        Assert.Same(layer.Parameters[1], sunkBoxingCall[Boxing.Input]);

        var main = Assert.IsType<Function>(rewritten.Entry);
        var layerCalls = ExprCollector.Collect(main.Body)
            .OfType<Call>()
            .Where(call => ReferenceEquals(call.Target, layer))
            .ToArray();
        Assert.Equal(2, layerCalls.Length);
        Assert.All(layerCalls, call => Assert.Equal(fixture.FirstPartialType, call.Arguments[1].CheckedType));
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing && Equals(call.CheckedType, fixture.MaterializedType));
        Assert.Equal(fixture.EntryType, main.CheckedType);
    }

    [Fact]
    public async Task TestKeepBoundaryWhenStatsHasAnotherConsumer()
    {
        var fixture = CreateModule(additionalStatsConsumer: true, inconsistentPartialTypes: false);

        var rewritten = await new SinkNormStatsBoxingAcrossFunctionBoundariesPass().RunAsync(
            fixture.Module,
            new());

        var layer = Assert.Single(rewritten.Functions.OfType<Function>().Where(function => function.Name == "layer"));
        Assert.Equal(fixture.MaterializedType, layer.Parameters[1].CheckedType);
        Assert.All(
            ExprCollector.Collect(Assert.IsType<Function>(rewritten.Entry).Body)
                .OfType<Call>()
                .Where(call => ReferenceEquals(call.Target, layer)),
            call => Assert.IsType<Boxing>(Assert.IsType<Call>(call.Arguments[1]).Target));
    }

    [Fact]
    public async Task TestSpecializeCalleeWhenCallSitesHaveDifferentPartialTypes()
    {
        var fixture = CreateModule(additionalStatsConsumer: false, inconsistentPartialTypes: true);

        var rewritten = await new SinkNormStatsBoxingAcrossFunctionBoundariesPass().RunAsync(
            fixture.Module,
            new());

        var layerVariants = rewritten.Functions.OfType<Function>()
            .Where(function => function.Name.StartsWith("layer", System.StringComparison.Ordinal))
            .ToArray();
        Assert.Equal(2, layerVariants.Length);
        Assert.Equal(
            new IRType[] { fixture.FirstPartialType, fixture.SecondPartialType },
            layerVariants.Select(function => function.Parameters[1].CheckedType).ToArray());
        Assert.All(
            layerVariants,
            variant =>
            {
                var normApply = Assert.Single(
                    ExprCollector.Collect(variant.Body)
                        .OfType<Call>()
                        .Where(call => call.Target is NormApply));
                Assert.IsType<Boxing>(Assert.IsType<Call>(normApply[NormApply.Stats]).Target);
            });

        var main = Assert.IsType<Function>(rewritten.Entry);
        var calls = ExprCollector.Collect(main.Body)
            .OfType<Call>()
            .Where(call => call.Target is Function function && layerVariants.Contains(function))
            .ToArray();
        Assert.Equal(2, calls.Length);
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing && Equals(call.CheckedType, fixture.MaterializedType));
    }

    private static Fixture CreateModule(bool additionalStatsConsumer, bool inconsistentPartialTypes)
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var inputTensorType = new TensorType(DataTypes.Float32, new long[] { 2, 8 });
        var statsTensorType = new TensorType(DataTypes.Float32, new long[] { 1, 2, 1 });
        var parameterTensorType = new TensorType(DataTypes.Float32, new long[] { 8 });
        var inputType = new DistributedType(inputTensorType, [SBP.B, SBP.B], placement);
        var materializedType = new DistributedType(statsTensorType, [SBP.B, SBP.B, SBP.B], placement);
        var firstPartialType = materializedType with { Partial = SBP.P([0, 1], ReduceOp.Sum) };
        var secondPartialType = inconsistentPartialTypes
            ? materializedType with { Partial = SBP.P([0], ReduceOp.Sum) }
            : firstPartialType;
        var parameterType = new DistributedType(parameterTensorType, [SBP.B], placement);

        var layerInput = new Var("layer_input", inputType);
        var layerStats = new Var("layer_stats", materializedType);
        var layerScale = new Var("layer_scale", parameterType);
        var layerBias = new Var("layer_bias", parameterType);
        var normalized = IR.F.NN.NormApply(
            1,
            1e-6f,
            layerInput,
            layerStats,
            layerScale,
            layerBias,
            useMean: false);
        BaseExpr layerBody = additionalStatsConsumer
            ? new IR.Tuple(normalized, layerStats)
            : normalized;
        var layer = new Function(
            "layer",
            "pyntt",
            layerBody,
            new IVar[] { layerInput, layerStats, layerScale, layerBias });
        Assert.True(layer.InferenceType());

        var input0 = new Var("input0", inputType);
        var input1 = new Var("input1", inputType);
        var partial0 = new Var("partial0", firstPartialType);
        var partial1 = new Var("partial1", secondPartialType);
        var scale = new Var("scale", parameterType);
        var bias = new Var("bias", parameterType);
        var call0 = new Call(layer, input0, IR.F.Distributed.Boxing(partial0, materializedType), scale, bias);
        var call1 = new Call(layer, input1, IR.F.Distributed.Boxing(partial1, materializedType), scale, bias);
        var main = new Function(
            "main",
            "pyntt",
            new IR.Tuple(call0, call1),
            new IVar[] { input0, input1, partial0, partial1, scale, bias });
        Assert.True(main.InferenceType());
        var entryType = main.CheckedType;

        var module = new IRModule(main);
        module.Add(layer);
        return new Fixture(module, materializedType, firstPartialType, secondPartialType, entryType);
    }

    private sealed record Fixture(
        IRModule Module,
        DistributedType MaterializedType,
        DistributedType FirstPartialType,
        DistributedType SecondPartialType,
        IRType EntryType);
}
