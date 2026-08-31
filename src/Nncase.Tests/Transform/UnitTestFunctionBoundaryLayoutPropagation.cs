// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.Passes;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFunctionBoundaryLayoutPropagation : TestClassBase
{
    [Fact]
    public async Task TestReuseSingleSpecializationForRepeatedInternalFunction()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var layer = MakePackUnpackLayer("layer", layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var call0 = new Call(layer, input);
        var call1 = new Call(layer, call0);
        var main = new Function("main", call1, input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");

        var mainBody = CompilerServices.Print(main.Body);
        Assert.Equal(1, Count(mainBody, "Pack("));
        Assert.Equal(1, Count(mainBody, "Unpack("));

        var specializedBody = CompilerServices.Print(specialized.Body);
        Assert.DoesNotContain("Pack(", specializedBody, StringComparison.Ordinal);
        Assert.DoesNotContain("Unpack(", specializedBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestTupleOutputGetItemFeedsPackedConsumerWithoutRepack()
    {
        var producerInput = new Var("producer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var packed = Pack(producerInput, [4], [1]);
        var producer = new Function(
            "producer",
            new IR.Tuple(
                Unpack(packed, [4], [1]),
                Unpack(packed, [4], [1])),
            producerInput);
        Assert.True(producer.InferenceType());

        var consumerInput = new Var("consumer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var consumer = MakePackUnpackLayer("consumer", consumerInput);
        Assert.True(consumer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var producerCall = new Call(producer, input);
        var consumerCall = new Call(consumer, GetItem(producerCall, 0));
        var main = new Function("main", consumerCall, input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(producer);
        module.Add(consumer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        var mainBody = CompilerServices.Print(main.Body);
        Assert.Equal(1, Count(mainBody, "Pack("));
        Assert.Equal(1, Count(mainBody, "Unpack("));
    }

    [Fact]
    public async Task TestCallerOutputPackDemandSpecializesCalleeOutput()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var layer = new Function("layer", IR.F.Math.Unary(UnaryOp.Abs, layerInput), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var main = new Function("main", Pack(new Call(layer, input), [4], [1]), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        var specializedBody = CompilerServices.Print(specialized.Body);
        Assert.Contains("Pack(", specializedBody, StringComparison.Ordinal);

        var mainCall = Assert.IsType<Call>(main.Body);
        var target = Assert.IsType<Function>(mainCall.Target);
        Assert.Equal("layer", target.Name);
        var mainType = Assert.IsType<TensorType>(mainCall.CheckedType);
        Assert.IsType<VectorType>(mainType.DType);
    }

    [Fact]
    public async Task TestCallerOutputReshardDemandBecomesInternalShardedView()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 64));
        var placement = new Placement(new[] { 2, 4 }, "yx", "bb");
        var sourceType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1]) },
            placement);
        var targetType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0]) },
            placement);
        var layerInput = new Var("layer_input", sourceType);
        var layer = new Function("layer", IR.F.Math.Abs(layerInput), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", sourceType);
        var layerCall = new Call(layer, input);
        var resharded = IR.F.Distributed.Boxing(layerCall, targetType);
        var main = new Function("main", IR.F.Math.Abs(resharded), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        var passManager = CompileSession.CreatePassManager("PostDistributedBoundaryOutputDemand");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("FoldBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.FoldBoxingBoxing>();
            p.Add<Passes.Rules.FoldBoxingShardedView>();
        });
        await passManager.RunAsync(module);

        var specialized = GetFunction(module, "layer");
        var shardedView = Assert.Single(
            ExprCollector.Collect(specialized.Body)
                .OfType<Call>()
                .Where(call => call.Target is ShardedView));
        Assert.Equal(targetType, shardedView.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(specialized.Body).OfType<Call>(),
            call => call.Target is Boxing);
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing);
        var specializedCall = Assert.Single(
            ExprCollector.Collect(main.Body)
                .OfType<Call>()
                .Where(call => ReferenceEquals(call.Target, specialized)));
        Assert.Equal(targetType, specializedCall.CheckedType);
    }

    [Fact]
    public async Task TestMixedCallerOutputReshardDemandUsesPerCallShardedViews()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 64));
        var placement = new Placement(new[] { 2, 4 }, "yx", "bb");
        var sourceType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1]) },
            placement);
        var targetType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var layerInput = new Var("layer_input", sourceType);
        var layer = new Function("layer", IR.F.Math.Abs(layerInput), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", sourceType);
        var demandedCall = new Call(layer, input);
        var sourceLayoutCall = new Call(layer, input);
        var main = new Function(
            "main",
            new IR.Tuple(
                IR.F.Distributed.Boxing(demandedCall, targetType),
                sourceLayoutCall),
            input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        var passManager = CompileSession.CreatePassManager("MixedPostDistributedBoundaryOutputDemand");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("FoldBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.FoldBoxingBoxing>();
            p.Add<Passes.Rules.FoldBoxingShardedView>();
        });
        await passManager.RunAsync(module);

        var specialized = GetFunction(module, "layer");
        var internalView = Assert.Single(
            ExprCollector.Collect(specialized.Body)
                .OfType<Call>()
                .Where(call => call.Target is ShardedView));
        Assert.Equal(targetType, internalView.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing);

        var mainTuple = Assert.IsType<IR.Tuple>(main.Body);
        var directCall = Assert.IsType<Call>(mainTuple[0]);
        Assert.Same(specialized, directCall.Target);
        Assert.Equal(targetType, directCall.CheckedType);
        var restoredView = Assert.IsType<Call>(mainTuple[1]);
        Assert.IsType<ShardedView>(restoredView.Target);
        Assert.Equal(sourceType, restoredView.CheckedType);
        var restoredCall = Assert.IsType<Call>(restoredView[ShardedView.Input]);
        Assert.Same(specialized, restoredCall.Target);
        Assert.Equal(targetType, restoredCall.CheckedType);
    }

    [Fact]
    public async Task TestTupleCallerOutputReshardDemandFoldsRestoredView()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 64));
        var placement = new Placement(new[] { 2, 4 }, "yx", "bb");
        var sourceType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1]) },
            placement);
        var targetType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var layerInput = new Var("layer_input", sourceType);
        var layer = new Function(
            "layer",
            new IR.Tuple(IR.F.Math.Abs(layerInput), IR.F.Math.Neg(layerInput)),
            layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", sourceType);
        var layerCall = new Call(layer, input);
        var demandedOutput = IR.F.Distributed.Boxing(GetItem(layerCall, 0), targetType);
        var main = new Function("main", demandedOutput, input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        var passManager = CompileSession.CreatePassManager("TuplePostDistributedBoundaryOutputDemand");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("FoldBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldGetItemTuple>();
            p.Add<Passes.Rules.FoldBoxingBoxing>();
            p.Add<Passes.Rules.FoldBoxingShardedView>();
        });
        await passManager.RunAsync(module);

        var specialized = GetFunction(module, "layer");
        var outputType = Assert.IsType<TupleType>(specialized.Body.CheckedType);
        Assert.Equal(targetType, outputType[0]);
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing);
        var projectedCall = Assert.IsType<Call>(main.Body);
        Assert.IsType<IR.Tensors.GetItem>(projectedCall.Target);
        var rawCall = Assert.IsType<Call>(projectedCall[IR.Tensors.GetItem.Input]);
        Assert.Same(specialized, rawCall.Target);
    }

    [Fact]
    public async Task TestCallerOutputDemandKeepsUndemandedCollectiveInCallee()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var placement = new Placement(new[] { 2, 4 }, "yx", "bb");
        var hiddenTensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 64));
        var hiddenType = new DistributedType(
            hiddenTensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var demandedHiddenType = new DistributedType(
            hiddenTensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1]) },
            placement);
        var statsTensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 1, 1));
        var materializedStatsType = new DistributedType(
            statsTensorType,
            new SBP[] { SBP.B, SBP.B, SBP.B },
            placement);
        var partialStatsType = materializedStatsType with
        {
            Partial = SBP.P([0, 1], ReduceOp.Sum),
        };

        var layerInput = new Var("layer_input", hiddenType);
        var layerStats = new Var("layer_stats", partialStatsType);
        var layer = new Function(
            "layer",
            new IR.Tuple(
                IR.F.Math.Abs(layerInput),
                IR.F.Distributed.Boxing(layerStats, materializedStatsType)),
            layerInput,
            layerStats);
        Assert.True(layer.InferenceType());

        var input = new Var("input", hiddenType);
        var stats = new Var("stats", partialStatsType);
        var layerCall = new Call(layer, input, stats);
        var main = new Function(
            "main",
            new IR.Tuple(
                IR.F.Distributed.Boxing(GetItem(layerCall, 0), demandedHiddenType),
                GetItem(layerCall, 1)),
            input,
            stats);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        var passManager = CompileSession.CreatePassManager("DemandedOutputsOnlyBoundaryLayout");
        passManager.AddWithName<FunctionBoundaryLayoutPropagationPass>(
            "DemandedOutputsOnlyBoundaryLayout",
            true,
            false);
        passManager.AddWithName<DataflowPass>("FoldBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldGetItemTuple>();
            p.Add<Passes.Rules.FoldBoxingBoxing>();
            p.Add<Passes.Rules.FoldBoxingShardedView>();
        });
        await passManager.RunAsync(module);

        var specialized = GetFunction(module, "layer");
        var bodyCalls = ExprCollector.Collect(specialized.Body).OfType<Call>().ToArray();
        Assert.Single(bodyCalls.Where(call => call.Target is ShardedView));
        var statsBoxing = Assert.Single(bodyCalls.Where(call => call.Target is Boxing));
        Assert.Equal(materializedStatsType, statsBoxing.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing);
    }

    [Fact]
    public async Task TestCallerOutputDemandAbsorbsPartialReductionBoxing()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var placement = new Placement(new[] { 2, 4 }, "yx", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 1, 1));
        var materializedType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B, SBP.B },
            placement);
        var partialType = materializedType with
        {
            Partial = SBP.P([0, 1], ReduceOp.Sum),
        };

        var layerInput = new Var("layer_input", partialType);
        var layer = new Function("layer", layerInput, layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", partialType);
        var layerCall = new Call(layer, input);
        var main = new Function(
            "main",
            IR.F.Distributed.Boxing(layerCall, materializedType),
            input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass(
            enableCallerOutputDemandLayouts: true,
            enableInternalOutputLayouts: false).RunAsync(module, new());

        var specialized = GetFunction(module, "layer");
        Assert.Equal(materializedType, specialized.Body.CheckedType);
        var specializedMaterialize = Assert.IsType<Call>(specialized.Body);
        Assert.IsType<Boxing>(specializedMaterialize.Target);
        Assert.Equal(partialType, specializedMaterialize[Boxing.Input].CheckedType);

        var specializedCall = Assert.IsType<Call>(main.Body);
        Assert.Same(specialized, specializedCall.Target);
        Assert.Equal(materializedType, specializedCall.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing);
    }

    [Fact]
    public async Task TestCallerOutputDemandKeepsPartialAbiForMixedConsumers()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var placement = new Placement(new[] { 2, 4 }, "yx", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 1, 1));
        var materializedType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B, SBP.B },
            placement);
        var partialType = materializedType with
        {
            Partial = SBP.P([0, 1], ReduceOp.Sum),
        };

        var layerInput = new Var("layer_input", partialType);
        var layer = new Function("layer", layerInput, layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", partialType);
        var layerCall = new Call(layer, input);
        var main = new Function(
            "main",
            new IR.Tuple(
                IR.F.Distributed.Boxing(layerCall, materializedType),
                layerCall),
            input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass(
            enableCallerOutputDemandLayouts: true,
            enableInternalOutputLayouts: false).RunAsync(module, new());

        var unchangedLayer = GetFunction(module, "layer");
        Assert.Equal(partialType, unchangedLayer.Body.CheckedType);
        var output = Assert.IsType<IR.Tuple>(main.Body);
        var materialize = Assert.IsType<Call>(output.Fields[0]);
        Assert.IsType<Boxing>(materialize.Target);
        Assert.Equal(partialType, materialize[Boxing.Input].CheckedType);
        Assert.Equal(materializedType, materialize.CheckedType);
        Assert.Equal(partialType, output.Fields[1].CheckedType);
    }

    [Fact]
    public async Task TestWrappedCallerOutputReshardDemandIsDiscovered()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 64));
        var placement = new Placement(new[] { 2, 4 }, "yx", "bb");
        var sourceType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1]) },
            placement);
        var targetType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var layerInput = new Var("layer_input", sourceType);
        var layer = new Function("layer", IR.F.Math.Abs(layerInput), layerInput);
        var wrapper = new FunctionWrapper("layer_wrapper", "pyntt", layer, returnOutput: true);
        Assert.True(layer.InferenceType());
        Assert.True(wrapper.InferenceType());

        var input = new Var("input", sourceType);
        var wrappedCall = new Call(wrapper, input);
        var main = new Function(
            "main",
            IR.F.Distributed.Boxing(wrappedCall, targetType),
            input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        module.Add(wrapper);
        var passManager = CompileSession.CreatePassManager("WrappedPostDistributedBoundaryOutputDemand");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("FoldBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.FoldBoxingBoxing>();
            p.Add<Passes.Rules.FoldBoxingShardedView>();
        });
        await passManager.RunAsync(module);

        var specialized = GetFunction(module, "layer");
        Assert.Equal(targetType, specialized.Body.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(main.Body).OfType<Call>(),
            call => call.Target is Boxing);
        var finalCall = Assert.IsType<Call>(main.Body);
        var finalWrapper = Assert.IsType<FunctionWrapper>(finalCall.Target);
        Assert.Same(specialized, finalWrapper.Target);
    }

    [Fact]
    public async Task TestDynamicDimensionIdentityIsPreservedInSpecializedBody()
    {
        var n = new DimVar("n");
        n.Metadata.Range = new(1, 128);
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(n, 16)));
        var packed = Pack(layerInput, [4], [1]);
        var reshaped = Reshape(packed, new RankedShape(n, 4));
        var sum = IR.F.Math.Add(packed, reshaped);
        var layer = new Function("layer", Unpack(sum, [4], [1]), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(n, 16)));
        var main = new Function("main", new Call(layer, input), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        var outputType = Assert.IsType<TensorType>(specialized.Body.CheckedType);
        var outputShape = Assert.IsType<RankedShape>(outputType.Shape);
        Assert.Equal(n, outputShape[0]);
        Assert.DoesNotContain("max(n, n)", CompilerServices.Print(specialized.Body), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestInputPackIsHoistedWhenParameterAlsoHasRawUse()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var packed = Pack(layerInput, [4], [1]);
        var rawPacked = Pack(IR.F.Math.Unary(UnaryOp.Abs, layerInput), [4], [1]);
        var sum = IR.F.Math.Add(packed, rawPacked);
        var layer = new Function("layer", Unpack(sum, [4], [1]), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var main = new Function("main", new Call(layer, input), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        var parameter = Assert.IsType<Var>(Assert.Single(specialized.Parameters.ToArray()));
        var parameterType = Assert.IsType<TensorType>(parameter.CheckedType);
        Assert.IsType<VectorType>(parameterType.DType);

        var outerUnpack = Assert.IsType<Call>(main.Body);
        Assert.IsType<Nncase.IR.Tensors.Unpack>(outerUnpack.Target);
        var specializedCall = Assert.IsType<Call>(outerUnpack.Arguments[Nncase.IR.Tensors.Unpack.Input.Index]);
        var boundaryPack = Assert.IsType<Call>(specializedCall.Arguments[0]);
        Assert.IsType<Nncase.IR.Tensors.Pack>(boundaryPack.Target);

        var specializedBody = CompilerServices.Print(specialized.Body);
        Assert.Contains("Unpack(", specializedBody, StringComparison.Ordinal);
        Assert.Contains("Pack(", specializedBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestNestedInputPackIsHoistedByFixedPoint()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var pack4 = Pack(layerInput, [4], [1]);
        var pack2 = Pack(pack4, [2], [1]);
        var unpack2 = Unpack(pack2, [2], [1]);
        var layer = new Function("layer", Unpack(unpack2, [4], [1]), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var main = new Function("main", new Call(layer, input), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);

        var outputUnpack4 = Assert.IsType<Call>(main.Body);
        Assert.IsType<Nncase.IR.Tensors.Unpack>(outputUnpack4.Target);
        var outputUnpack2 = Assert.IsType<Call>(outputUnpack4.Arguments[Nncase.IR.Tensors.Unpack.Input.Index]);
        Assert.IsType<Nncase.IR.Tensors.Unpack>(outputUnpack2.Target);
        var specializedCall = Assert.IsType<Call>(outputUnpack2.Arguments[Nncase.IR.Tensors.Unpack.Input.Index]);
        var boundaryPack2 = Assert.IsType<Call>(specializedCall.Arguments[0]);
        Assert.IsType<Nncase.IR.Tensors.Pack>(boundaryPack2.Target);
        var boundaryPack4 = Assert.IsType<Call>(boundaryPack2.Arguments[Nncase.IR.Tensors.Pack.Input.Index]);
        Assert.IsType<Nncase.IR.Tensors.Pack>(boundaryPack4.Target);

        var finalSpecialized = Assert.IsType<Function>(specializedCall.Target);
        Assert.Equal("layer", finalSpecialized.Name);
        var specializedBody = CompilerServices.Print(finalSpecialized.Body);
        Assert.DoesNotContain("Pack(", specializedBody, StringComparison.Ordinal);
        Assert.DoesNotContain("Unpack(", specializedBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestInputRelayoutChainIsHoistedByFixedPoint()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(8, 16)));
        var oldLayout = Pack(layerInput, [4], [1]);
        var scalar = Unpack(oldLayout, [4], [1]);
        var newLayout = Pack(scalar, [2], [0]);
        var layer = new Function("layer", IR.F.Math.Unary(UnaryOp.Abs, newLayout), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(8, 16)));
        var main = new Function("main", new Call(layer, input), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        var parameter = Assert.IsType<Var>(Assert.Single(specialized.Parameters.ToArray()));
        Assert.Equal(
            new TensorType(new VectorType(DataTypes.Float32, [2]), new RankedShape(4, 16)),
            parameter.CheckedType);

        var specializedBody = CompilerServices.Print(specialized.Body);
        Assert.DoesNotContain("Pack(", specializedBody, StringComparison.Ordinal);
        Assert.DoesNotContain("Unpack(", specializedBody, StringComparison.Ordinal);

        var specializedCall = Assert.IsType<Call>(main.Body);
        var boundaryPack = Assert.IsType<Call>(specializedCall.Arguments[0]);
        var pack = Assert.IsType<Nncase.IR.Tensors.Pack>(boundaryPack.Target);
        Assert.Equal(new[] { 2 }, pack.Lanes.ToArray());
        Assert.Equal(new[] { 0 }, pack.Axes.ToArray());
    }

    [Fact]
    public async Task TestCompilerInsertedRestoreDoesNotReverseInputPropagation()
    {
        var packedType = new TensorType(
            new VectorType(DataTypes.Float32, [4]),
            new RankedShape(4, 4));
        var scalarType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var layerInput = new Var("layer_input", packedType);
        var scalar = Unpack(layerInput, [4], [1]);
        var layer = new Function(
            "layer",
            new IR.Tuple(layerInput, IR.F.Math.Unary(UnaryOp.Abs, scalar)),
            layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", packedType);
        var main = new Function("main", new Call(layer, input), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        var parameter = Assert.IsType<Var>(Assert.Single(specialized.Parameters.ToArray()));
        Assert.Equal(scalarType, parameter.CheckedType);

        var specializedBody = CompilerServices.Print(specialized.Body);
        Assert.Equal(1, Count(specializedBody, "Pack("));
        Assert.DoesNotContain("Unpack(", specializedBody, StringComparison.Ordinal);

        var specializedCall = Assert.IsType<Call>(main.Body);
        var boundaryUnpack = Assert.IsType<Call>(specializedCall.Arguments[0]);
        Assert.IsType<Nncase.IR.Tensors.Unpack>(boundaryUnpack.Target);
    }

    [Fact]
    public async Task TestNonTensorParametersArePreserved()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var layerId = new DimVar("layer_id");
        var layer = MakePackUnpackLayer("layer", layerInput, layerId);
        Assert.True(layer.InferenceType());
        Assert.True(layer.Clone().InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var main = new Function("main", new Call(layer, input, new DimConst(0)), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        await new FunctionBoundaryLayoutPropagationPass().RunAsync(module, new());

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        Assert.Equal(2, specialized.Parameters.Length);
        Assert.IsType<Var>(specialized.Parameters[0]);
        Assert.Same(layerId, specialized.Parameters[1]);
    }

    [Fact]
    public async Task TestBoxingIsHoistedAcrossFunctionBoundary()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.B }, new Placement(new[] { 2 }, "b", "b"));
        var layerInput = new Var("layer_input", tensorType);
        var layerDistributed = IR.F.Distributed.Boxing(layerInput, distributedType);
        var layer = new Function("layer", IR.F.Distributed.Boxing(layerDistributed, tensorType), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", tensorType);
        var main = new Function("main", IR.F.Distributed.Boxing(new Call(layer, input), distributedType), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        var passManager = CompileSession.CreatePassManager("BoundaryBoxingPropagation");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("FoldBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.FoldBoxingBoxing>();
        });

        await passManager.RunAsync(module);

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        var parameter = Assert.IsType<Var>(Assert.Single(specialized.Parameters.ToArray()));
        Assert.Equal(distributedType, parameter.CheckedType);
        Assert.Equal(distributedType, specialized.Body.CheckedType);
        Assert.DoesNotContain("Boxing(", CompilerServices.Print(specialized.Body), StringComparison.Ordinal);

        var mainBody = CompilerServices.Print(main.Body);
        Assert.Equal(1, Count(mainBody, "Boxing("));
        Assert.DoesNotContain("Boxing(Boxing(", mainBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestMultipleDistributedBoxingInputsUseDistributedAbi()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var placement = new Placement(new[] { 2 }, "b", "b");
        var broadcastType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.B }, placement);
        var splitType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.SContiguous([0]) }, placement);
        var layerInput = new Var("layer_input", tensorType);
        var broadcast = IR.F.Distributed.Boxing(layerInput, broadcastType);
        var split = IR.F.Distributed.Boxing(layerInput, splitType);
        var splitToBroadcast = IR.F.Distributed.Boxing(split, broadcastType);
        var layer = new Function("layer", IR.F.Distributed.Boxing(IR.F.Math.Add(broadcast, splitToBroadcast), tensorType), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", tensorType);
        var main = new Function("main", new Call(layer, input), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        var passManager = CompileSession.CreatePassManager("BoundaryMultipleDistributedBoxing");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("FoldBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.FoldBoxingBoxing>();
        });

        await passManager.RunAsync(module);

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        var parameter = Assert.IsType<Var>(Assert.Single(specialized.Parameters.ToArray()));
        Assert.IsType<DistributedType>(parameter.CheckedType);
        var specializedBody = CompilerServices.Print(specialized.Body);
        Assert.DoesNotContain($"NewType: {tensorType}", specializedBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestDistributedPlainDistributedBoxingFoldsToDirectReshard()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var placement = new Placement(new[] { 2 }, "b", "b");
        var sourceType = new DistributedType(tensorType, new SBP[] { SBP.SContiguous([0]) }, placement);
        var targetType = new DistributedType(tensorType, new SBP[] { SBP.B }, placement);
        var input = new Var("input", sourceType);
        var body = IR.F.Distributed.Boxing(IR.F.Distributed.Boxing(input, tensorType), targetType);
        var main = new Function("main", body, input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        var passManager = CompileSession.CreatePassManager("FoldDistributedPlainDistributedBoxing");
        passManager.AddWithName<DataflowPass>("FoldBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.FoldBoxingBoxing>();
        });

        await passManager.RunAsync(module);

        var mainBody = CompilerServices.Print(main.Body);
        Assert.Equal(1, Count(mainBody, "Boxing("));
        Assert.DoesNotContain("Boxing(Boxing(", mainBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestTupleBoundaryDistributedOutputFoldsWithoutPlainBoxing()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var placement = new Placement(new[] { 2 }, "b", "b");
        var sourceType = new DistributedType(tensorType, new SBP[] { SBP.SContiguous([0]) }, placement);
        var targetType = new DistributedType(tensorType, new SBP[] { SBP.B }, placement);
        var input = new Var("input", sourceType);
        var tuple = new IR.Tuple(IR.F.Distributed.Boxing(input, tensorType), input);
        var body = IR.F.Distributed.Boxing(GetItem(tuple, 0), targetType);
        var main = new Function("main", body, input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        var passManager = CompileSession.CreatePassManager("FoldTupleBoundaryDistributedBoxing");
        passManager.AddWithName<DataflowPass>("FoldTupleBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldGetItemTuple>();
            p.Add<Passes.Rules.FoldBoxingBoxing>();
        });

        await passManager.RunAsync(module);

        var mainBody = CompilerServices.Print(main.Body);
        Assert.Equal(1, Count(mainBody, "Boxing("));
        Assert.DoesNotContain("GetItem(", mainBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestConstWeightDistributedBoundaryUsesShardedView()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.NTTTargetOptions
        {
            MemoryAccessArch = MemoryAccessArchitecture.UMA,
            UnifiedMemoryArch = true,
        };
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.SContiguous([0]) }, new Placement(new[] { 2 }, "b", "b"));
        var weight = new Var("weight", tensorType);
        var distributedWeight = IR.F.Distributed.Boxing(weight, distributedType);
        var layer = new Function("layer", IR.F.Distributed.Boxing(distributedWeight, tensorType), weight);
        Assert.True(layer.InferenceType());

        var weightConst = Tensor.From(Enumerable.Range(0, 64).Select(x => (float)x).ToArray(), [4, 16]);
        var main = new Function("main", new Call(layer, Const.FromTensor(weightConst)));
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);
        var passManager = CompileSession.CreatePassManager("BoundaryConstShardedView");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        await passManager.RunAsync(module);

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        Assert.DoesNotContain("Boxing(", CompilerServices.Print(specialized.Body), StringComparison.Ordinal);
        var mainCalls = ExprCollector.Collect(main.Body).OfType<Call>().ToArray();
        Assert.Contains(mainCalls, call => call.Target is IR.Distributed.ShardedView && call.Arguments[IR.Distributed.ShardedView.Input.Index] is TensorConst);
        Assert.DoesNotContain(mainCalls, call => call.Target is IR.Distributed.Boxing { NewType: DistributedType } && call.Arguments[IR.Distributed.Boxing.Input.Index] is TensorConst);
    }

    [Fact]
    public async Task TestTIRSelectionReusesCallerAllocatedTupleOutputs()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var calleeInput = new Var("callee_input", tensorType);
        var callee = new Function(
            "callee",
            new IR.Tuple(
                IR.F.Math.Unary(UnaryOp.Abs, calleeInput),
                IR.F.Math.Unary(UnaryOp.Neg, calleeInput)),
            calleeInput);
        Assert.True(callee.InferenceType());

        var input = new Var("input", tensorType);
        var calleeCall = new Call(callee, input);
        var main = new Function(
            "main",
            IR.F.Math.Add(GetItem(calleeCall, 0), GetItem(calleeCall, 1)),
            input);
        Assert.True(main.InferenceType());

        // Entry-first module order exercises the required callee-first TIR
        // selection traversal.
        var module = new IRModule(main);
        module.Add(callee);

        var passManager = CompileSession.CreatePassManager("TIRSelectionCallerAllocatedTupleOutputs");
        passManager.Add<NTTTIRSelectionPass>();
        await passManager.RunAsync(module);

        var calleeWrapper = Assert.Single(module.Functions.OfType<PrimFunctionWrapper>());
        var mainPrim = Assert.IsType<PrimFunction>(module.Entry);
        var calleeCalls = ExprCollector.Collect(mainPrim.Body)
            .OfType<Call>()
            .Where(call => ReferenceEquals(call.Target, calleeWrapper.Target))
            .ToArray();
        var selectedCall = Assert.Single(calleeCalls);
        var abi = calleeWrapper.Target.GetAbiView();
        Assert.Equal(abi.Inputs.Count + abi.OutputParameters.Count, selectedCall.Arguments.Length);
    }

    [Fact]
    public async Task TestTIRSelectionPromotesPartialTupleOutputAsCompactPerOwnerStorage()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var placement = new Placement([4, 8], "yx", "bb");
        var lhsType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }),
            [SBP.B, SBP.B],
            placement);
        var rhsType = new DistributedType(
            new TensorType(
                new VectorType(DataTypes.BFloat16, [8, 2, 8]),
                new long[] { 4, 16 }),
            [SBP.B, SBP.SContiguous([0, 1], 1)],
            placement);
        var lhs = new Var("lhs", lhsType);
        var rhs = new Var("rhs", rhsType);
        var fused = IR.F.NTT.PackedMatMulNormStats(
            lhs,
            rhs,
            DataTypes.BFloat16,
            IR.NTT.PackedMatMulRhsLayout.KMajor,
            axis: 1,
            useMean: false);
        var value = GetItem(fused, 0);
        var stats = GetItem(fused, 1);
        var statsType = Assert.IsType<DistributedType>(stats.CheckedType);
        Assert.NotNull(statsType.Partial);
        var callee = new Function("callee", new IR.Tuple(value, stats), lhs, rhs);
        Assert.True(callee.InferenceType());

        var mainLhs = new Var("main_lhs", lhsType);
        var mainRhs = new Var("main_rhs", rhsType);
        var calleeCall = new Call(callee, mainLhs, mainRhs);
        var broadcastStatsType = new DistributedType(
            statsType.TensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var reducedStats = IR.F.Distributed.Boxing(GetItem(calleeCall, 1), broadcastStatsType);
        var main = new Function(
            "main",
            IR.F.Distributed.Boxing(reducedStats, statsType.TensorType),
            mainLhs,
            mainRhs);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(callee);
        var passManager = CompileSession.CreatePassManager("TIRSelectionPartialTupleOutput");
        passManager.Add<NTTTIRSelectionPass>();
        await passManager.RunAsync(module);

        var calleeWrapper = Assert.Single(module.Functions.OfType<PrimFunctionWrapper>());
        var calleeAbi = calleeWrapper.Target.GetAbiView();
        var valueResult = calleeAbi.Results[0];
        var valueOutput = Assert.IsType<BufferVar>(valueResult.Storage);
        Assert.Equal(
            DistributedBufferStorageKind.CompactPerOwner,
            valueOutput.LayoutAnnotation.DistributedStorageKind);
        var valueView = Assert.IsType<TIR.Buffer>(valueResult.Value);
        Assert.Equal(DistributedBufferStorageKind.CompactPerOwner, valueView.DistributedStorageKind);
        Assert.Equal(MemoryLocation.Output, valueView.MemSpan.Buffer.Location);
        Assert.Equal(
            valueView.MemSpan.Size.FixedValue * 32,
            valueView.MemSpan.Buffer.Size.FixedValue);

        var statsResult = calleeAbi.Results[1];
        var statsOutput = Assert.IsType<BufferVar>(statsResult.Storage);
        Assert.Equal(
            DistributedBufferStorageKind.CompactPerOwner,
            statsOutput.LayoutAnnotation.DistributedStorageKind);
        var statsView = Assert.IsType<TIR.Buffer>(statsResult.Value);
        Assert.Equal(DistributedBufferStorageKind.CompactPerOwner, statsView.DistributedStorageKind);
        Assert.Equal(MemoryLocation.Output, statsView.MemSpan.Buffer.Location);
        Assert.Equal(
            statsView.MemSpan.Size.FixedValue * 32,
            statsView.MemSpan.Buffer.Size.FixedValue);

        var mainPrim = Assert.IsType<PrimFunction>(module.Entry);
        var selectedCall = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => ReferenceEquals(call.Target, calleeWrapper.Target)));
        var valueParameterIndex = Array.FindIndex(
            calleeWrapper.Target.Parameters.ToArray(),
            parameter => ReferenceEquals(parameter, valueOutput));
        var callerValue = Assert.IsType<TIR.Buffer>(selectedCall.Arguments[valueParameterIndex]);
        Assert.Equal(MemoryLocation.ChipLocalData, callerValue.MemSpan.Buffer.Location);
        Assert.Equal(DistributedBufferStorageKind.CompactPerOwner, callerValue.DistributedStorageKind);
        Assert.Equal(
            callerValue.MemSpan.Size.FixedValue * 32,
            callerValue.MemSpan.Buffer.Size.FixedValue);

        var statsParameterIndex = Array.FindIndex(
            calleeWrapper.Target.Parameters.ToArray(),
            parameter => ReferenceEquals(parameter, statsOutput));
        var callerStats = Assert.IsType<TIR.Buffer>(selectedCall.Arguments[statsParameterIndex]);
        Assert.Equal(MemoryLocation.ChipLocalData, callerStats.MemSpan.Buffer.Location);
        Assert.Equal(DistributedBufferStorageKind.CompactPerOwner, callerStats.DistributedStorageKind);
        Assert.Equal(
            callerStats.MemSpan.Size.FixedValue * 32,
            callerStats.MemSpan.Buffer.Size.FixedValue);
    }

    [Fact]
    public async Task TestTIRSelectionPromotesFullyShardedOutputAsCompactPerOwnerStorage()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var placement = new Placement([4, 8], "yx", "bb");
        var distributedType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }),
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var calleeInput = new Var("callee_input", distributedType);
        var callee = new Function(
            "callee",
            IR.F.Math.Unary(
                UnaryOp.Neg,
                IR.F.Math.Unary(UnaryOp.Abs, calleeInput)),
            calleeInput);
        Assert.True(callee.InferenceType());

        var mainInput = new Var("main_input", distributedType);
        var calleeCall = new Call(callee, mainInput);
        var main = new Function(
            "main",
            IR.F.Distributed.Boxing(calleeCall, distributedType.TensorType),
            mainInput);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(callee);
        var passManager = CompileSession.CreatePassManager("TIRSelectionFullyShardedOutput");
        passManager.Add<NTTTIRSelectionPass>();
        await passManager.RunAsync(module);

        var calleeWrapper = Assert.Single(module.Functions.OfType<PrimFunctionWrapper>());
        var calleeAbi = calleeWrapper.Target.GetAbiView();
        var input = Assert.Single(calleeAbi.Inputs.OfType<BufferVar>());
        Assert.Equal(
            DistributedBufferStorageKind.CanonicalGlobal,
            input.LayoutAnnotation.DistributedStorageKind);
        var output = Assert.Single(calleeAbi.OutputParameters);
        Assert.Equal(
            DistributedBufferStorageKind.CompactPerOwner,
            output.LayoutAnnotation.DistributedStorageKind);

        var mainPrim = Assert.IsType<PrimFunction>(module.Entry);
        var selectedCall = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => ReferenceEquals(call.Target, calleeWrapper.Target)));
        var outputParameterIndex = Array.FindIndex(
            calleeWrapper.Target.Parameters.ToArray(),
            parameter => ReferenceEquals(parameter, output));
        var callerOutput = Assert.IsType<TIR.Buffer>(selectedCall.Arguments[outputParameterIndex]);
        Assert.Equal(MemoryLocation.ChipLocalData, callerOutput.MemSpan.Buffer.Location);
        Assert.Equal(DistributedBufferStorageKind.CompactPerOwner, callerOutput.DistributedStorageKind);
        Assert.Equal(
            callerOutput.MemSpan.Size.FixedValue * 32,
            callerOutput.MemSpan.Buffer.Size.FixedValue);

        var ordinaryIntermediates = ExprCollector.Collect(calleeWrapper.Target.Body)
            .OfType<TIR.Buffer>()
            .Where(buffer => buffer.MemSpan.Buffer.Location == MemoryLocation.Data)
            .ToArray();
        Assert.NotEmpty(ordinaryIntermediates);
        Assert.All(
            ordinaryIntermediates,
            buffer => Assert.Equal(
                DistributedBufferStorageKind.CompactLocal,
                buffer.DistributedStorageKind));
    }

    [Fact]
    public async Task TestTIRLayoutPropagationUnifiesRepeatedFullyShardedCallStorage()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var placement = new Placement([4, 8], "yx", "bb");
        var distributedType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }),
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var calleeInput = new Var("callee_input", distributedType);
        var callee = new Function(
            "callee",
            IR.F.Math.Unary(UnaryOp.Neg, calleeInput),
            calleeInput);
        Assert.True(callee.InferenceType());

        var mainInput = new Var("main_input", distributedType);
        var seed = IR.F.Math.Unary(UnaryOp.Abs, mainInput);
        var call0 = new Call(callee, seed);
        var call1 = new Call(callee, call0);
        var main = new Function(
            "main",
            IR.F.Distributed.Boxing(call1, distributedType.TensorType),
            mainInput);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(callee);
        var passManager = CompileSession.CreatePassManager("TIRLayoutPropagationFullyShardedCalls");
        passManager.Add<NTTTIRSelectionPass>();
        passManager.Add<AddFunctionToModule>();
        passManager.Add<RemoveFunctionWrapperPass>();
        passManager.Add<PropagatePrimFunctionBufferLayoutsPass>();
        passManager.Add<SpecializePrimFunctionBufferLayoutsPass>();
        await passManager.RunAsync(module);

        var mainPrim = Assert.IsType<PrimFunction>(module.Entry);
        var calleePrim = Assert.Single(
            module.Functions.OfType<PrimFunction>().Where(function =>
                !ReferenceEquals(function, mainPrim)));
        Assert.DoesNotContain("_layout_", calleePrim.Name, StringComparison.Ordinal);
        var calls = ExprCollector.Collect(mainPrim.Body)
            .OfType<Call>()
            .Where(call => ReferenceEquals(call.Target, calleePrim))
            .ToArray();
        Assert.Equal(2, calls.Length);

        var firstInput = Assert.IsType<TIR.Buffer>(calls[0].Arguments[0]);
        Assert.Equal(MemoryLocation.ChipLocalData, firstInput.MemSpan.Buffer.Location);
        Assert.Equal(
            DistributedBufferStorageKind.CompactPerOwner,
            firstInput.DistributedStorageKind);
        Assert.Equal(
            firstInput.MemSpan.Size.FixedValue * 32,
            firstInput.MemSpan.Buffer.Size.FixedValue);
        Assert.All(
            calls,
            call => Assert.Equal(
                DistributedBufferStorageKind.CompactPerOwner,
                Assert.IsType<TIR.Buffer>(call.Arguments[0]).DistributedStorageKind));
    }

    [Fact]
    public async Task TestTIRSelectionUsesMaterializedStorageForTupleBoxingOutputAtCaller()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.PyNTTTargetOptions();
        var placement = new Placement([4, 8], "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(DataTypes.Float32, new long[] { 1, 128 }),
            [SBP.B, SBP.SContiguous([0, 1], 4)],
            placement);
        var input = new Var("input", inputType);
        var stats = IR.F.NN.NormStats(1, input, useMean: false);
        var partialStatsType = Assert.IsType<DistributedType>(stats.CheckedType);
        var broadcastStatsType = partialStatsType with { Partial = null };
        var boxedTuple = IR.F.Distributed.Boxing(
            new IR.Tuple(input, stats),
            new TupleType(new IRType[] { inputType, broadcastStatsType }));
        var callee = new Function(
            "callee",
            new IR.Tuple(GetItem(boxedTuple, 0), GetItem(boxedTuple, 1)),
            input);
        Assert.True(callee.InferenceType());

        var mainInput = new Var("main_input", inputType);
        var calleeCall = new Call(callee, mainInput);
        var main = new Function(
            "main",
            IR.F.Distributed.Boxing(GetItem(calleeCall, 1), partialStatsType.TensorType),
            mainInput);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(callee);
        var passManager = CompileSession.CreatePassManager("TIRSelectionCompactTupleBoxingOutput");
        passManager.Add<NTTTIRSelectionPass>();
        await passManager.RunAsync(module);

        var calleeWrapper = Assert.Single(module.Functions.OfType<PrimFunctionWrapper>());
        var calleeAbi = calleeWrapper.Target.GetAbiView();
        var statsOutput = Assert.Single(calleeAbi.OutputParameters);
        Assert.Equal(
            DistributedBufferStorageKind.CompactLocal,
            statsOutput.LayoutAnnotation.DistributedStorageKind);

        var mainPrim = Assert.IsType<PrimFunction>(module.Entry);
        var selectedCall = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => ReferenceEquals(call.Target, calleeWrapper.Target)));
        var statsParameterIndex = Array.FindIndex(
            calleeWrapper.Target.Parameters.ToArray(),
            parameter => ReferenceEquals(parameter, statsOutput));
        var callerStats = Assert.IsType<TIR.Buffer>(selectedCall.Arguments[statsParameterIndex]);
        Assert.Equal(MemoryLocation.Data, callerStats.MemSpan.Buffer.Location);
        Assert.Equal(DistributedBufferStorageKind.CompactLocal, callerStats.DistributedStorageKind);
        Assert.Equal(callerStats.MemSpan.Size.FixedValue, callerStats.MemSpan.Buffer.Size.FixedValue);
    }

    [Fact]
    public async Task TestTIRSelectionPreservesInputAliasInTupleOutput()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var calleeInput = new Var("callee_input", tensorType);
        var callee = new Function(
            "callee",
            new IR.Tuple(
                IR.F.Math.Unary(UnaryOp.Abs, calleeInput),
                calleeInput),
            calleeInput);
        Assert.True(callee.InferenceType());

        var input = new Var("input", tensorType);
        var calleeCall = new Call(callee, input);
        var main = new Function(
            "main",
            IR.F.Math.Add(GetItem(calleeCall, 0), GetItem(calleeCall, 1)),
            input);
        Assert.True(main.InferenceType());

        // The alias result must not become a second caller-allocated output
        // before the callee ABI is known.
        var module = new IRModule(main);
        module.Add(callee);

        var passManager = CompileSession.CreatePassManager("TIRSelectionInputAliasTupleOutput");
        passManager.Add<NTTTIRSelectionPass>();
        await passManager.RunAsync(module);

        var calleeWrapper = Assert.Single(module.Functions.OfType<PrimFunctionWrapper>());
        var calleeAbi = calleeWrapper.Target.GetAbiView();
        var inputParameter = Assert.IsType<BufferVar>(Assert.Single(calleeAbi.Inputs));
        Assert.Equal(BufferVarRole.Input, inputParameter.Role);
        Assert.Single(calleeAbi.OutputParameters);
        Assert.Equal(2, calleeAbi.Results.Count);
        Assert.Same(inputParameter, calleeAbi.Results[1].Storage);
        Assert.DoesNotContain(
            ExprCollector.Collect(calleeWrapper.Target.Body).OfType<Call>(),
            call => call.Target is Memcopy);

        var mainPrim = Assert.IsType<PrimFunction>(module.Entry);
        var selectedCall = Assert.Single(ExprCollector.Collect(mainPrim.Body)
            .OfType<Call>()
            .Where(call => ReferenceEquals(call.Target, calleeWrapper.Target)));
        Assert.Equal(calleeWrapper.Target.Parameters.Length, selectedCall.Arguments.Length);

        var add = Assert.Single(ExprCollector.Collect(mainPrim.Body)
            .OfType<Call>()
            .Where(call => call.Target is TIR.NTT.VectorizedBinary binary && binary.BinaryOp == BinaryOp.Add));
        Assert.Same(
            Assert.IsType<TIR.Buffer>(selectedCall.Arguments[1]).MemSpan.Buffer,
            Assert.IsType<TIR.Buffer>(add.Arguments[0]).MemSpan.Buffer);
        Assert.Same(
            Assert.IsType<TIR.Buffer>(selectedCall.Arguments[0]).MemSpan.Buffer,
            Assert.IsType<TIR.Buffer>(add.Arguments[1]).MemSpan.Buffer);
    }

    [Fact]
    public async Task TestPostBoundaryPackPropagationPushesCallerPackThroughUnary()
    {
        var layerInput = new Var("layer_input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var layer = MakePackUnpackLayer("layer", layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 16)));
        var main = new Function("main", new Call(layer, IR.F.Math.Unary(UnaryOp.Cos, input)), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);

        var passManager = CompileSession.CreatePassManager("BoundaryPackPropagation");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<EGraphRulesPass>("PostFunctionBoundaryPackPropagation").Configure(p =>
        {
            new Nncase.Targets.CPUTarget().RegisterPackPropagationRules(p, CompileOptions);
        });

        await passManager.RunAsync(module);

        var postMain = Assert.IsType<Function>(module.Entry);
        var unpackCall = Assert.IsType<Call>(postMain.Body);
        Assert.IsType<Nncase.IR.Tensors.Unpack>(unpackCall.Target);
        var layerCall = Assert.IsType<Call>(unpackCall.Arguments[Nncase.IR.Tensors.Unpack.Input.Index]);
        var unaryCall = Assert.IsType<Call>(layerCall.Arguments[0]);
        var unary = Assert.IsType<IR.Math.Unary>(unaryCall.Target);
        Assert.Equal(UnaryOp.Cos, unary.UnaryOp);
        var packCall = Assert.IsType<Call>(unaryCall.Arguments[IR.Math.Unary.Input.Index]);
        Assert.IsType<Nncase.IR.Tensors.Pack>(packCall.Target);
    }

    [Fact]
    public async Task TestInputTransposeConstIsHoistedAndFolded()
    {
        var activation = new Var("activation", new TensorType(DataTypes.Float32, new RankedShape(3, 2)));
        var weight = new Var("weight", new TensorType(DataTypes.Float32, new RankedShape(2, 3)));
        var layer = new Function("layer", IR.F.Math.Add(activation, Transpose(weight, new[] { 1, 0 })), activation, weight);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(3, 2)));
        var weightConst = Tensor.From(new float[] { 0, 1, 2, 3, 4, 5 }, [2, 3]);
        var main = new Function("main", new Call(layer, input, Const.FromValue(Value.FromTensor(weightConst))), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);

        var passManager = CompileSession.CreatePassManager("BoundaryTransposeConstFold");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("PostBoundaryFoldConst").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
        });

        await passManager.RunAsync(module);

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        Assert.DoesNotContain("Transpose(", CompilerServices.Print(specialized.Body), StringComparison.Ordinal);
        Assert.DoesNotContain("Transpose(", CompilerServices.Print(main.Body), StringComparison.Ordinal);
        Assert.Contains("f32[3,2]", CompilerServices.Print(main.Body), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestOutputBitcastIsHoistedAndCallerPackFolds()
    {
        var vectorType = new VectorType(DataTypes.Float32, [4]);
        var layerInput = new Var("layer_input", new TensorType(vectorType, new RankedShape(2, 4)));
        var layer = new Function("layer", Bitcast(layerInput, DataTypes.Float32), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", new TensorType(vectorType, new RankedShape(2, 4)));
        var main = new Function("main", Pack(new Call(layer, input), [4], [1]), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        module.Add(layer);

        var passManager = CompileSession.CreatePassManager("BoundaryOutputBitcastFold");
        passManager.Add<FunctionBoundaryLayoutPropagationPass>();
        passManager.AddWithName<DataflowPass>("PostBoundaryFoldPackBitcast").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldPackBitcast>();
        });

        await passManager.RunAsync(module);

        AssertNoLayoutFunctions(module);
        var specialized = GetFunction(module, "layer");
        Assert.DoesNotContain("Bitcast(", CompilerServices.Print(specialized.Body), StringComparison.Ordinal);
        var mainBody = CompilerServices.Print(main.Body);
        Assert.DoesNotContain("Pack(", mainBody, StringComparison.Ordinal);
        Assert.DoesNotContain("Bitcast(", mainBody, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestNestedBitcastFoldsToDirectBitcast()
    {
        var packedType = new VectorType(DataTypes.Float32, [2, 2]);
        var vectorType = new VectorType(DataTypes.Float32, [4]);
        var input = new Var("input", new TensorType(packedType, new RankedShape(2, 4)));
        var main = new Function("main", Bitcast(Bitcast(input, vectorType), DataTypes.Float32), input);
        Assert.True(main.InferenceType());

        var module = new IRModule(main);
        var passManager = CompileSession.CreatePassManager("FoldBitcastBitcast");
        passManager.AddWithName<DataflowPass>("FoldBitcastBitcast").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldBitcastBitcast>();
        });

        await passManager.RunAsync(module);

        Assert.Equal(1, Count(CompilerServices.Print(main.Body), "Bitcast("));
    }

    [Fact]
    public void TestVectorUnaryCostIsLowerThanScalarUnaryCost()
    {
        var costModel = DefaultTargetOpCostModel.Instance;
        var scalar = new TargetCostTensor(DataTypes.Float32, new RankedShape(4, 16));
        var vector = new TargetCostTensor(new VectorType(DataTypes.Float32, [4]), new RankedShape(4, 4));

        Assert.True(costModel.TryGetUnaryCost(new(UnaryOp.Cos, scalar, scalar), out var scalarCost));
        Assert.True(costModel.TryGetUnaryCost(new(UnaryOp.Cos, vector, vector), out var vectorCost));
        Assert.True(costModel.GetLatency(vectorCost) < costModel.GetLatency(scalarCost));
    }

    private static Function MakePackUnpackLayer(string name, Var input)
    {
        var packed = Pack(input, [4], [1]);
        var output = Unpack(packed, [4], [1]);
        return new Function(name, output, input);
    }

    private static Function MakePackUnpackLayer(string name, Var input, params IVar[] extraParameters)
    {
        var packed = Pack(input, [4], [1]);
        var output = Unpack(packed, [4], [1]);
        return new Function(name, output, new IVar[] { input }.Concat(extraParameters).ToArray());
    }

    private static Function GetFunction(IRModule module, string name)
    {
        return Assert.Single(module.Functions.OfType<Function>().Where(x => x.Name == name));
    }

    private static void AssertNoLayoutFunctions(IRModule module)
    {
        Assert.DoesNotContain(module.Functions.OfType<Function>(), x => x.Name.Contains("__layout_", StringComparison.Ordinal));
    }

    private static int Count(string text, string value)
    {
        var count = 0;
        var start = 0;
        while (true)
        {
            var index = text.IndexOf(value, start, StringComparison.Ordinal);
            if (index < 0)
            {
                return count;
            }

            count++;
            start = index + value.Length;
        }
    }
}
