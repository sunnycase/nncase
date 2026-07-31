// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Transforms;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TransformTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestWeightPipelineRegions : TestClassBase
{
    [Fact]
    public async Task TestOverlappingPipelineStagesDrainPreviousOwner()
    {
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var shared = CreateBuffer("shared", MemoryLocation.Shared, 0);
        var first = CreateWeightPipelineCall(source, shared);
        var second = CreateWeightPipelineCall(source, shared);
        var module = CreateModule(new Sequential(first, second));

        await RunPass(module);

        var region = GetRegion(Assert.IsType<PrimFunction>(module.Entry));
        Assert.Equal(
            ["main_weight_stage_0"],
            region.ConsumeBody.Fields.ToArray().OfType<PipelineDrain>().Select(drain => drain.StageId));
        Assert.Equal(
            ["main_weight_stage_0"],
            region.ProduceBody.Fields.ToArray().OfType<PipelineDrain>().Select(drain => drain.StageId));
    }

    [Fact]
    public async Task TestDisjointPipelineStagesDoNotSynchronize()
    {
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var firstShared = CreateBuffer("first_shared", MemoryLocation.Shared, 0);
        var secondShared = CreateBuffer("second_shared", MemoryLocation.Shared, 512);
        var module = CreateModule(
            new Sequential(
                CreateWeightPipelineCall(source, firstShared),
                CreateWeightPipelineCall(source, secondShared)));

        await RunPass(module);

        var region = GetRegion(Assert.IsType<PrimFunction>(module.Entry));
        Assert.Empty(ExprCollector.Collect(region).OfType<PipelineDrain>());
        Assert.Empty(ExprCollector.Collect(region).OfType<PipelineHandoff>());
    }

    [Fact]
    public async Task TestOrdinarySharedOwnerHandsStorageToProducer()
    {
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var destination = CreateBuffer("destination", MemoryLocation.Data, 8192);
        var shared = CreateBuffer("shared", MemoryLocation.Shared, 0);
        var module = CreateModule(
            new Sequential(
                T.Memcopy(destination, shared),
                CreateWeightPipelineCall(source, shared)));

        await RunPass(module);

        var region = GetRegion(Assert.IsType<PrimFunction>(module.Entry));
        var consumerHandoff = Assert.Single(
            region.ConsumeBody.Fields.ToArray().OfType<PipelineHandoff>());
        var producerHandoff = Assert.Single(
            region.ProduceBody.Fields.ToArray().OfType<PipelineHandoff>());
        Assert.Equal(consumerHandoff.HandoffId, producerHandoff.HandoffId);
        Assert.Equal(0, consumerHandoff.SharedOffsetBytes);
        Assert.Equal(consumerHandoff.SharedOffsetBytes, producerHandoff.SharedOffsetBytes);
        Assert.True(
            Array.IndexOf(region.ConsumeBody.Fields.ToArray(), consumerHandoff) <
            Array.FindIndex(
                region.ConsumeBody.Fields.ToArray(),
                expression => expression is PipelineStage));
        Assert.True(
            Array.IndexOf(region.ProduceBody.Fields.ToArray(), producerHandoff) <
            Array.FindIndex(
                region.ProduceBody.Fields.ToArray(),
                expression => expression is PipelineStage));
    }

    [Fact]
    public async Task TestStructuredOrdinarySharedOwnerHandsStorageToProducer()
    {
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var destination = CreateBuffer("destination", MemoryLocation.Data, 8192);
        var shared = CreateBuffer("shared", MemoryLocation.Shared, 0);
        var conditionalOwner = new IfThenElse(
            IR.F.Math.Equal(1, 1),
            new Sequential(T.Memcopy(destination, shared)));
        var module = CreateModule(
            new Sequential(
                conditionalOwner,
                CreateWeightPipelineCall(source, shared)));

        await RunPass(module);

        var region = GetRegion(Assert.IsType<PrimFunction>(module.Entry));
        var consumerFields = region.ConsumeBody.Fields.ToArray();
        var producerFields = region.ProduceBody.Fields.ToArray();
        var consumerHandoff = Assert.Single(consumerFields.OfType<PipelineHandoff>());
        var producerHandoff = Assert.Single(producerFields.OfType<PipelineHandoff>());
        Assert.Equal(consumerHandoff.HandoffId, producerHandoff.HandoffId);
        Assert.Equal(
            Array.IndexOf(consumerFields, conditionalOwner) + 1,
            Array.IndexOf(consumerFields, consumerHandoff));
        Assert.True(
            Array.IndexOf(producerFields, producerHandoff) <
            Array.FindIndex(
                producerFields,
                expression => expression is PipelineStage));
    }

    [Fact]
    public async Task TestPipelineOwnsOnlyDeclaredSharedWorkspace()
    {
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var destination = CreateBuffer("destination", MemoryLocation.Data, 8192);
        var consumerShared = CreateBuffer("consumer_shared", MemoryLocation.Shared, 0);
        var pipelineShared = CreateBuffer("pipeline_shared", MemoryLocation.Shared, 512);
        var module = CreateModule(
            new Sequential(
                T.Memcopy(destination, consumerShared),
                CreateWeightPipelineCall(
                    source,
                    pipelineShared,
                    lhs: consumerShared,
                    output: destination)));

        await RunPass(module);

        var region = GetRegion(Assert.IsType<PrimFunction>(module.Entry));
        Assert.Empty(ExprCollector.Collect(region).OfType<PipelineHandoff>());
        Assert.Empty(ExprCollector.Collect(region).OfType<PipelineDrain>());
    }

    [Fact]
    public async Task TestUnusedSharedInputDoesNotOwnStorage()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 64 });
        var formalUnused = new BufferVar(
            "formal_unused",
            tensorType,
            BufferVarRole.Input,
            MemoryLocation.Shared);
        var callee = new PrimFunction(
            "unused_shared_input",
            PyNTTTarget.Kind,
            new Sequential(),
            new IVar[] { formalUnused });
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var shared = CreateBuffer("shared", MemoryLocation.Shared, 0);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(callee, shared),
                CreateWeightPipelineCall(source, shared)),
            Array.Empty<IVar>());
        var module = new IRModule(main);
        module.Add(callee);

        await RunPass(module);

        var region = GetRegion(Assert.IsType<PrimFunction>(module.Entry));
        Assert.Empty(ExprCollector.Collect(region).OfType<PipelineHandoff>());
        Assert.Empty(ExprCollector.Collect(region).OfType<PipelineDrain>());
    }

    [Fact]
    public async Task TestPipelineUnderConditionalFailsFast()
    {
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var shared = CreateBuffer("shared", MemoryLocation.Shared, 0);
        var conditional = new IfThenElse(
            IR.F.Math.Equal(1, 1),
            new Sequential(CreateWeightPipelineCall(source, shared)));
        var module = CreateModule(new Sequential(conditional));

        var exception = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunPass(module));

        Assert.Contains(nameof(IfThenElse), exception.Message, StringComparison.Ordinal);
        Assert.Contains("straight-line", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestPipelineRequiresPhysicalSharedWorkspace()
    {
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var unbufferizedShared = new BufferVar(
            "unbufferized_shared",
            TensorType.Scalar(new PointerType(DataTypes.UInt8)),
            BufferVarRole.Workspace,
            MemoryLocation.Shared);
        var module = CreateModule(
            new Sequential(CreateWeightPipelineCall(source, unbufferizedShared)));

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => RunPass(module));

        Assert.Contains("after Bufferize", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestRepeatedPipelineCalleeUsesCallerOwnedStages()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 64 });
        var formalShared = new BufferVar(
            "formal_shared",
            TensorType.Scalar(new PointerType(DataTypes.UInt8)),
            BufferVarRole.Workspace,
            MemoryLocation.Shared);
        var formalSource = new BufferVar(
            "formal_source",
            tensorType,
            BufferVarRole.Input,
            MemoryLocation.Data);
        var calleeShared = CreateBuffer("callee_shared", MemoryLocation.Shared, 0);
        var callee = new PrimFunction(
            "pipeline_callee",
            PyNTTTarget.Kind,
            new Sequential(CreateWeightPipelineCall(formalSource, calleeShared)),
            new IVar[] { formalSource, formalShared });
        var shared = CreateBuffer("shared", MemoryLocation.Shared, 0);
        var source = CreateBuffer("source", MemoryLocation.Data, 4096);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(callee, source, shared),
                new Call(callee, source, shared)),
            Array.Empty<IVar>());
        var module = new IRModule(main);
        module.Add(callee);

        await RunPass(module);

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        var rewrittenCallee = Assert.IsType<PrimFunction>(
            module.Functions.Single(function => function.Name == callee.Name));
        var callerRegion = GetRegion(rewrittenMain);
        var calleeRegion = GetRegion(rewrittenCallee);
        var callerStages = callerRegion.ConsumeBody.Fields.ToArray()
            .OfType<PipelineStage>()
            .ToArray();
        Assert.Equal(2, callerStages.Length);
        Assert.Single(calleeRegion.ConsumeBody.Fields.ToArray().OfType<PipelineStage>());
        Assert.Single(callerRegion.ConsumeBody.Fields.ToArray().OfType<PipelineDrain>());
        Assert.All(
            callerStages,
            stage => Assert.Same(rewrittenCallee, stage.Operation.Target));
    }

    private static Task<IRModule> RunPass(IRModule module)
        => new LowerWeightPipelineRegionsPass(PyNTTTarget.Kind).RunAsync(module, new());

    private static IRModule CreateModule(Sequential body)
        => new(new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            body,
            Array.Empty<IVar>()));

    private static ProducerConsumerRegion GetRegion(PrimFunction function)
        => Assert.IsType<ProducerConsumerRegion>(Assert.Single(function.Body.Fields.ToArray()));

    private static Call CreateWeightPipelineCall(
        Expr weight,
        Expr sharedWorkspace,
        Expr? lhs = null,
        Expr? output = null)
    {
        var call = TIR.F.NTT.PackedMatMul(
            lhs ?? weight,
            weight,
            output ?? weight,
            None.Default,
            1.0f);
        var arguments = call.Arguments.ToArray();
        arguments[^1] = sharedWorkspace;
        call = call.With(arguments: arguments);
        call.Metadata.TIRMicroKernel = new TIRMicroKernelSelection(
            "triton.matmul",
            "simt_fma_smem_pipeline",
            ImmutableDictionary<string, long>.Empty,
            ImmutableArray.Create(
                new TIRSharedWorkspaceDescriptor(
                    "shared",
                    new TensorType(DataTypes.Float32, new[] { 64 }),
                    256)),
            new TIRWeightPipelineContract([1], [0]));
        return call;
    }

    private static Nncase.TIR.Buffer CreateBuffer(
        string name,
        MemoryLocation location,
        ulong offset)
    {
        const long elementCount = 64;
        var sizeBytes = elementCount * DataTypes.Float32.SizeInBytes;
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(offset, DataTypes.Float32),
            sizeBytes,
            location);
        return new Nncase.TIR.Buffer(
            name,
            DataTypes.Float32,
            new MemSpan(physical, 0, sizeBytes),
            new Dimension[] { elementCount },
            new Dimension[] { 1 },
            null);
    }
}
