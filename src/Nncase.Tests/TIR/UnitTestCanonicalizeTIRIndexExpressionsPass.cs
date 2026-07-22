// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Transforms;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestCanonicalizeTIRIndexExpressionsPass : TestClassBase
{
    private const string ModuleKind = "test_tir_index_canonicalization";

    [Fact]
    public async Task TestStaticSingleIterationCanonicalizesBufferDescriptorAndPreservesIdentity()
    {
        var loopVar = new DimVar("block_offset");
        var extent = Dimension.Min(4, 4, Dimension.Max(0, 4 - loopVar));
        var (module, function, buffer, physicalBuffer) = MakeLoopFunction(
            loopVar,
            new TIR.Range(0, 4, 4),
            extent);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.Equal(4L, buffer.Dimensions[0].FixedValue);
        Assert.Same(physicalBuffer, buffer.MemSpan.Buffer);
        var load = Assert.Single(ExprCollector.Collect(function).OfType<Call>());
        Assert.Same(buffer, load.Arguments[0]);
        Assert.Same(buffer, load.Arguments[1]);
    }

    [Fact]
    public async Task TestDynamicUnitStepCanonicalizesFullTileExtent()
    {
        var sequenceLength = MakeDynamicDimension("sequence_length", 1, 128);
        var loopVar = new DimVar("sequence_offset");
        var extent = Dimension.Min(
            1,
            1,
            Dimension.Max(0, sequenceLength - loopVar));
        var (module, _, buffer, _) = MakeLoopFunction(
            loopVar,
            new TIR.Range(0, sequenceLength, 1),
            extent,
            sequenceLength);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.Equal(1L, buffer.Dimensions[0].FixedValue);
    }

    [Fact]
    public async Task TestFullPartitionCanonicalizesDynamicFullTileExtent()
    {
        var stop = MakeDynamicDimension("full_end", 128, 1024);
        var loopVar = new DimVar("reduce_offset");
        var extent = Dimension.Min(128, stop - loopVar);
        var physicalBuffer = new PhysicalBuffer(16, 4096, MemoryLocation.Data);
        var buffer = new Nncase.TIR.Buffer(
            "full_tile",
            DataTypes.UInt8,
            new MemSpan(physicalBuffer, 0, 4096),
            [extent],
            [1],
            null);
        var load = new Call(new LoadT(), buffer, buffer);
        var loop = new For(
            loopVar,
            new TIR.Range(0, stop, 128),
            LoopMode.Reduction,
            new Sequential(load),
            LoopPartition.Full);
        var function = new PrimFunction(
            "canonicalize_full_partition",
            ModuleKind,
            new Sequential(loop),
            [stop]);
        var module = new IRModule(function);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.Equal(128L, buffer.Dimensions[0].FixedValue);
    }

    [Fact]
    public async Task TestPeeledFullPartitionCanonicalizesExtentAgainstLogicalStop()
    {
        var remaining = MakeDynamicDimension("remaining", 0, 149);
        var logicalStop = Dimension.Min(8, remaining);
        var loopVar = new DimVar("n_offset");
        var extent = Dimension.Min(2, logicalStop - loopVar);
        var fullEnd = (logicalStop / 2) * 2;
        var physicalBuffer = new PhysicalBuffer(16, 4096, MemoryLocation.Shared);
        var buffer = new Nncase.TIR.Buffer(
            "peeled_full_tile",
            DataTypes.UInt8,
            new MemSpan(physicalBuffer, 0, 4096),
            [extent, 256],
            [256, 1],
            null);
        var loop = new For(
            loopVar,
            new TIR.Range(0, fullEnd, 2),
            LoopMode.Serial,
            new Sequential(new Call(new LoadT(), buffer, buffer)),
            LoopPartition.Full);
        var function = new PrimFunction(
            "canonicalize_peeled_full_partition",
            ModuleKind,
            new Sequential(loop),
            [remaining]);
        var module = new IRModule(function);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.Equal(2L, buffer.Dimensions[0].FixedValue);
        Assert.Equal(256L, buffer.Dimensions[1].FixedValue);
    }

    [Fact]
    public async Task TestDynamicTiledLoopPreservesTailMinAndRemovesRedundantClamp()
    {
        var sequenceLength = MakeDynamicDimension("sequence_length", 1, 1024);
        var loopVar = new DimVar("sequence_offset");
        var extent = Dimension.Min(
            128,
            128,
            Dimension.Max(0, sequenceLength - loopVar));
        var (module, _, buffer, _) = MakeLoopFunction(
            loopVar,
            new TIR.Range(0, sequenceLength, 128),
            extent,
            sequenceLength);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        var tailExtent = Assert.IsType<DimMin>(buffer.Dimensions[0]);
        Assert.Equal(2, tailExtent.Operands.Length);
        Assert.Contains(tailExtent.Operands.ToArray(), operand => operand.IsFixed && operand.FixedValue == 128);
        Assert.Contains(ExprCollector.Collect(tailExtent), expression => ReferenceEquals(expression, loopVar));
        Assert.DoesNotContain(ExprCollector.Collect(tailExtent), expression => expression is DimMax);
    }

    [Fact]
    public async Task TestFixedDivisibleAsyncCopyLoopCanonicalizesFullTransportTile()
    {
        var loopVar = new DimVar("reduce_offset");
        var clippedExtent = Dimension.Select(8, loopVar + 1, 2, 4);
        var source = new Nncase.TIR.Buffer(
            "source_tile",
            DataTypes.Float32,
            new MemSpan(new PhysicalBuffer(4, 32, MemoryLocation.Data)),
            [clippedExtent],
            [1],
            null);
        var encoding = new TargetStorageEncodingSelection(
            new TargetStorageEncodingId("test.shared"),
            physicalBytes: 16,
            alignmentBytes: 4,
            Array.Empty<KeyValuePair<string, long>>());
        var layout = encoding.CreateStagedBufferLayout(stageCount: 2, stageStrideBytes: 16);
        var staged = new Nncase.TIR.Buffer(
            "staged_tile",
            DataTypes.Float32,
            new MemSpan(new PhysicalBuffer(4, layout.PhysicalBytes, MemoryLocation.Shared)),
            [clippedExtent],
            [1],
            null,
            encoding,
            layout);
        var access = new Var("staged_access");
        var allocation = IR.F.Buffer.AllocateBufferView(staged, new RankedShape(0));
        var stageBuffer = new Nncase.TIR.Buffer(
            "copy_stage_buffer",
            staged.ElemType,
            staged.MemSpan.With(size: layout.StagePhysicalBytes),
            staged.Dimensions.ToArray(),
            staged.Strides.ToArray(),
            staged.DistributedType,
            staged.StorageEncoding);
        var copyBody = TIR.T.Let(
            out var stage,
            stageBuffer,
            "copy_stage")
            .Body(new Sequential(TIR.T.TileLoad(stage, source)))
            .Build();
        var plan = MakeCpAsyncPlan();
        var loop = new PipelineFor(
            loopVar,
            new TIR.Range(0, 8, 4),
            LoopMode.Reduction,
            LoopPartition.Unpartitioned,
            new Sequential(copyBody),
            new Sequential(TIR.T.Nop()),
            plan,
            new PipelineRegionId("canonicalize_async_copy", "test/full-transport"),
            [new(
                "lhs",
                new TargetMemorySpaceId("gpu.block-global"),
                new TargetMemorySpaceId("gpu.shared"))],
            [access],
            [allocation],
            [staged]);
        var function = new PrimFunction(
            "canonicalize_async_copy",
            ModuleKind,
            new Sequential(loop),
            Array.Empty<IVar>());
        var module = new IRModule(function);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.Equal(4L, source.Dimensions[0].FixedValue);
        Assert.Equal(4L, staged.Dimensions[0].FixedValue);
        Assert.Same(staged, Assert.Single(loop.StagedBuffers.ToArray()));
    }

    [Fact]
    public async Task TestNonDivisibleUnpartitionedLoopPreservesClippedTransportTile()
    {
        var loopVar = new DimVar("reduce_offset");
        var extent = Dimension.Min(4, 10 - loopVar);
        var (module, _, buffer, _) = MakeLoopFunction(
            loopVar,
            new TIR.Range(0, 10, 4),
            extent);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.IsType<DimMin>(buffer.Dimensions[0]);
    }

    [Fact]
    public async Task TestUnrelatedNonlinearExpressionIsNotSimplifiedByLoopConstraint()
    {
        var sequenceLength = MakeDynamicDimension("sequence_length", 1, 128);
        var lhs = MakeDynamicDimension("lhs", 1, 16);
        var rhs = MakeDynamicDimension("rhs", 1, 16);
        var loopVar = new DimVar("sequence_offset");
        var extent = Dimension.Max(0, (lhs * rhs) - loopVar);
        var (module, _, buffer, _) = MakeLoopFunction(
            loopVar,
            new TIR.Range(0, sequenceLength, 1),
            extent,
            sequenceLength,
            lhs,
            rhs);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.IsType<DimMax>(buffer.Dimensions[0]);
    }

    [Fact]
    public async Task TestPassIsIdempotent()
    {
        var sequenceLength = MakeDynamicDimension("sequence_length", 1, 1024);
        var loopVar = new DimVar("sequence_offset");
        var extent = Dimension.Min(128, 128, Dimension.Max(0, sequenceLength - loopVar));
        var (module, _, buffer, _) = MakeLoopFunction(
            loopVar,
            new TIR.Range(0, sequenceLength, 128),
            extent,
            sequenceLength);
        var pass = new CanonicalizeTIRIndexExpressionsPass();

        await pass.RunAsync(module, new());
        var first = buffer.Dimensions[0];
        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.Same(first, buffer.Dimensions[0]);
    }

    [Fact]
    public async Task TestLargeLoopSpanDoesNotOverflowSingleIterationProof()
    {
        var loopVar = new DimVar("large_offset");
        var extent = new DimMin(0, loopVar);
        var (module, _, buffer, _) = MakeLoopFunction(
            loopVar,
            new TIR.Range(long.MinValue, 0, 1),
            extent);

        await new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new());

        Assert.Same(loopVar, buffer.Dimensions[0]);
    }

    [Fact]
    public async Task TestContextDependentSharedSubtreeFailsFast()
    {
        var loopVar = new DimVar("shared_offset");
        var physicalBuffer = new PhysicalBuffer(16, 4096, MemoryLocation.Data);
        var buffer = new TIR.Buffer(
            "shared_tile",
            DataTypes.UInt8,
            new MemSpan(physicalBuffer, 0, 4096),
            new[] { new DimMin(4, new DimMax(0, 4 - loopVar)) },
            new Dimension[] { 1 },
            null);
        var sharedBody = new Sequential(new Call(new LoadT(), buffer, buffer));
        var function = new PrimFunction(
            "shared_context",
            ModuleKind,
            new Sequential(
                new For(loopVar, new TIR.Range(0, 4, 4), LoopMode.Serial, sharedBody),
                new For(loopVar, new TIR.Range(1, 5, 4), LoopMode.Serial, sharedBody)),
            Array.Empty<IVar>());
        var module = new IRModule(function);

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new CanonicalizeTIRIndexExpressionsPass().RunAsync(module, new()));

        Assert.Contains("context-dependent dimension", exception.Message, StringComparison.Ordinal);
    }

    private static DimVar MakeDynamicDimension(string name, long min, long max)
        => new(name)
        {
            Metadata = new()
            {
                Range = new(min, max),
            },
        };

    private static PipelineRegionPlan MakeCpAsyncPlan()
        => new(
            "test.cp_async.n2",
            TritonLoopPipelineBackend.CpAsyncN2TemplateId,
            TritonLoopPipelineBackend.CpAsyncN2Synchronization,
            stageCount: 2,
            prefetchDistance: 1,
            PipelineTailPolicy.Serial,
            [
                new PipelineStageChannelPlan(
                    "lhs",
                    new TargetMemorySpaceId("gpu.block-global"),
                    new TargetMemorySpaceId("gpu.shared")),
            ]);

    private static (IRModule Module, PrimFunction Function, TIR.Buffer Buffer, PhysicalBuffer PhysicalBuffer) MakeLoopFunction(
        DimVar loopVar,
        TIR.Range domain,
        Dimension extent,
        params DimVar[] parameters)
    {
        var physicalBuffer = new PhysicalBuffer(16, 4096, MemoryLocation.Data);
        var buffer = new TIR.Buffer(
            "tile",
            DataTypes.UInt8,
            new MemSpan(physicalBuffer, 0, 4096),
            new[] { extent },
            new Dimension[] { 1 },
            null);
        var load = new Call(new LoadT(), buffer, buffer);
        var loop = new For(loopVar, domain, LoopMode.Serial, new Sequential(load));
        var function = new PrimFunction(
            "canonicalize_indices",
            ModuleKind,
            new Sequential(loop),
            parameters.Cast<IVar>().ToArray());
        return (new IRModule(function), function, buffer, physicalBuffer);
    }
}
