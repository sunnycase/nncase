// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.Passes;
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
        var splitType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([0]) }, placement);
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
        var sourceType = new DistributedType(tensorType, new SBP[] { SBP.S([0]) }, placement);
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
        var sourceType = new DistributedType(tensorType, new SBP[] { SBP.S([0]) }, placement);
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
            ConstShardedView = true,
            MemoryAccessArch = MemoryAccessArchitecture.UMA,
            UnifiedMemoryArch = true,
        };
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4, 16));
        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([0]) }, new Placement(new[] { 2 }, "b", "b"));
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

        var module = new IRModule(callee);
        module.Add(main);
        module.Entry = main;

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
        Assert.Equal(abi.Inputs.Count + abi.Outputs.Count, selectedCall.Arguments.Length);
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

        var module = new IRModule(callee);
        module.Add(main);
        module.Entry = main;

        var passManager = CompileSession.CreatePassManager("TIRSelectionInputAliasTupleOutput");
        passManager.Add<NTTTIRSelectionPass>();
        await passManager.RunAsync(module);

        var calleeWrapper = Assert.Single(module.Functions.OfType<PrimFunctionWrapper>());
        var calleeAbi = calleeWrapper.Target.GetAbiView();
        var inOut = Assert.IsType<BufferVar>(Assert.Single(calleeAbi.Inputs));
        Assert.Equal(BufferVarRole.InOut, inOut.Role);
        Assert.Equal(2, calleeAbi.Outputs.Count);
        Assert.Same(inOut, calleeAbi.Outputs[0]);
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
        Assert.Same(selectedCall.Arguments[1], add.Arguments[0]);
        Assert.Same(selectedCall.Arguments[0], add.Arguments[1]);
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
