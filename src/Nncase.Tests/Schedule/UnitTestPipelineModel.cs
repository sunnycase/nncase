// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.ScheduleTest;

public sealed class UnitTestPipelineModel
{
    [Fact]
    public void TestLoopPipelineEstimateBuildsSerialAndOverlappedAlternativesFromOneRegion()
    {
        using var solver = new Solver("loop-pipeline-estimate");
        var estimate = LoopPipelineScheduleEstimate.Create(
            solver,
            iterationCount: solver.MakeIntConst(4),
            invocationCount: solver.MakeIntConst(2),
            producerCycles: solver.MakeIntConst(10),
            consumerCycles: solver.MakeIntConst(20),
            producerCommitCycles: solver.MakeIntConst(1),
            consumerWaitAcquireCycles: solver.MakeIntConst(3),
            consumerReleaseCycles: solver.MakeIntConst(5));

        AssertConstant(estimate.SerialRegionCycles, 240);
        AssertConstant(estimate.InitiationIntervalCycles, 28);
        AssertConstant(estimate.PipelinedRegionCycles, 246);
    }

    [Fact]
    public void TestLoopPipelineEstimateRejectsInvalidDomains()
    {
        using var solver = new Solver("loop-pipeline-estimate-domain");
        Assert.Throws<ArgumentOutOfRangeException>(() => LoopPipelineScheduleEstimate.Create(
            solver,
            solver.MakeIntConst(0),
            solver.MakeIntConst(1),
            solver.MakeIntConst(1),
            solver.MakeIntConst(1),
            solver.MakeIntConst(0),
            solver.MakeIntConst(0),
            solver.MakeIntConst(0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => LoopPipelineScheduleEstimate.Create(
            solver,
            solver.MakeIntConst(1),
            solver.MakeIntConst(1),
            solver.MakeIntConst(1),
            solver.MakeIntConst(1),
            solver.MakeIntVar(-1, 4, "invalid_commit"),
            solver.MakeIntConst(0),
            solver.MakeIntConst(0)));
    }

    [Fact]
    public void TestPipelinePlanOwnsExactLoopChannels()
    {
        var first = CreatePlan("test.schedule");
        var second = CreatePlan("test.schedule");

        Assert.Equal(first, second);
        Assert.Equal(first.GetHashCode(), second.GetHashCode());
        Assert.Equal(TritonLoopPipelineBackend.CpAsyncN2TemplateId, first.TemplateId);
        Assert.Equal(2, first.StageCount);
        Assert.Equal(1, first.PrefetchDistance);
        Assert.Equal(PipelineTailPolicy.Serial, first.TailPolicy);
        var channel = Assert.Single(first.Channels);
        Assert.Equal("lhs", channel.ChannelId);
        Assert.Throws<ArgumentException>(() => new PipelineStageChannelPlan(
            "invalid",
            channel.DestinationMemorySpace,
            channel.DestinationMemorySpace));
    }

    [Fact]
    public void TestPipelineRegionIdentityNamespacesLocalProvenanceByOwningFunction()
    {
        var localId = "pipeline_op3_packed_mat_mul__axis2_entry3";
        var first = new PipelineRegionId("device_op3_packed_mat_mul", localId);
        var second = new PipelineRegionId("device_op13_packed_mat_mul", localId);

        Assert.NotEqual(first, second);
        Assert.Equal(
            "device_op3_packed_mat_mul/pipeline_op3_packed_mat_mul__axis2_entry3/full",
            first.ForPartition(LoopPartition.Full).Value);
        var boundary = new PipelineRegionId("device_op3_packed_mat_mul", "pipeline_outer/full");
        Assert.Equal(
            "device_op3_packed_mat_mul/pipeline_op3_packed_mat_mul__axis2_entry3/boundary/pipeline_outer%2Ffull",
            first.ForBoundary(boundary).Value);
        Assert.Throws<ArgumentException>(() => first.ForBoundary(second));
    }

    [Fact]
    public void TestStagedBufferLayoutSeparatesOneStageEncodingFromPhysicalAllocation()
    {
        var encoding = new TargetStorageEncodingSelection(
            new TargetStorageEncodingId("test.shared"),
            physicalBytes: 128,
            alignmentBytes: 16,
            Array.Empty<KeyValuePair<string, long>>());
        var stagedLayout = encoding.CreateStagedBufferLayout(stageCount: 4, stageStrideBytes: 128);
        var buffer = new TIR.Buffer(
            "a_shared",
            DataTypes.Float16,
            new MemSpan(new PhysicalBuffer(16, stagedLayout.PhysicalBytes, MemoryLocation.Shared)),
            [8, 8],
            [8, 1],
            null,
            encoding,
            stagedLayout);

        Assert.Equal(128, buffer.StorageEncoding!.PhysicalBytes);
        Assert.Equal(4, buffer.StagedLayout!.StageCount);
        Assert.Equal(512, buffer.StagedLayout.PhysicalBytes);
        Assert.Equal(512, buffer.MemSpan.Size.FixedValue);
        Assert.Throws<ArgumentException>(() => new TIR.Buffer(
            "invalid_shared",
            DataTypes.Float16,
            new MemSpan(new PhysicalBuffer(16, 128, MemoryLocation.Shared)),
            [8, 8],
            [8, 1],
            null,
            encoding,
            stagedLayout));
    }

    [Fact]
    public void TestPipelineForRequiresExactPlanAndConcreteStagedAllocation()
    {
        var plan = CreatePlan("test.schedule");
        var access = new Var("staged_access");
        var staged = CreateStagedBuffer("staged", stageCount: 2);
        var allocation = IR.F.Buffer.AllocateBufferView(staged, new RankedShape(0));
        var loop = CreatePipelineFor(plan, access, allocation, staged);

        Assert.Same(access, Assert.Single(loop.StagedAccesses.ToArray()));
        Assert.Same(allocation, Assert.Single(loop.StagedAllocations.ToArray()));
        Assert.Same(staged, Assert.Single(loop.StagedBuffers.ToArray()));
        Assert.Equal(7, loop.Operands.Length);

        var wrongDepth = CreateStagedBuffer("wrong_depth", stageCount: 4);
        Assert.Throws<ArgumentException>(() => CreatePipelineFor(
            plan,
            access,
            IR.F.Buffer.AllocateBufferView(wrongDepth, new RankedShape(0)),
            wrongDepth));
    }

    [Fact]
    public void TestTailLoopPeelingPreservesStructuredPipelineOnBothPartitions()
    {
        var plan = CreatePlan("test.schedule");
        var access = new Var("staged_access");
        var staged = CreateStagedBuffer("staged", stageCount: 2);
        var loop = CreatePipelineFor(
            plan,
            access,
            IR.F.Buffer.AllocateBufferView(staged, new RankedShape(0)),
            staged,
            new TIR.Range(0, 5, 4),
            LoopPartition.Unpartitioned);

        var rewritten = Assert.IsType<Sequential>(new TailLoopPeeling().Rewrite(loop));
        var loops = rewritten.Fields.ToArray().Select(Assert.IsType<PipelineFor>).ToArray();

        Assert.Equal(2, loops.Length);
        Assert.Same(plan, loops[0].Plan);
        Assert.Same(plan, loops[1].Plan);
        Assert.Equal(LoopPartition.Full, loops[0].Partition);
        Assert.Equal(LoopPartition.Tail, loops[1].Partition);
        Assert.Equal("test/op7/axis0/entry1/full", loops[0].RegionId.Value);
        Assert.Equal("test/op7/axis0/entry1/tail", loops[1].RegionId.Value);
        Assert.NotSame(loops[0].LoopVar, loops[1].LoopVar);
        Assert.Single(loops[1].StagedBuffers.ToArray());
    }

    [Fact]
    public void TestTailLoopPeelingClonesNestedPipelineBinders()
    {
        var plan = CreatePlan("test.nested.schedule");
        var access = new Var("nested_staged_access");
        var staged = CreateStagedBuffer("nested_staged", stageCount: 2);
        var inner = CreatePipelineFor(
            plan,
            access,
            IR.F.Buffer.AllocateBufferView(staged, new RankedShape(0)),
            staged,
            new TIR.Range(0, 8, 4),
            LoopPartition.Unpartitioned);
        var outer = new For(
            new DimVar("outer"),
            new TIR.Range(0, 5, 4),
            LoopMode.Serial,
            new Sequential(inner));

        var rewritten = Assert.IsType<Sequential>(new TailLoopPeeling().Rewrite(outer));
        var outerLoops = rewritten.Fields.ToArray().Select(Assert.IsType<For>).ToArray();
        var fullInner = Assert.IsType<PipelineFor>(Assert.Single(outerLoops[0].Body.Fields.ToArray()));
        var tailInner = Assert.IsType<PipelineFor>(Assert.Single(outerLoops[1].Body.Fields.ToArray()));

        Assert.NotSame(fullInner.LoopVar, tailInner.LoopVar);
        Assert.NotSame(
            Assert.Single(fullInner.StagedAccesses.ToArray()),
            Assert.Single(tailInner.StagedAccesses.ToArray()));
    }

    [Fact]
    public void TestEliminateStaticallyEmptyPipelineLoop()
    {
        var plan = CreatePlan("test.schedule");
        var access = new Var("staged_access");
        var staged = CreateStagedBuffer("staged", stageCount: 2);
        var loop = CreatePipelineFor(
            plan,
            access,
            IR.F.Buffer.AllocateBufferView(staged, new RankedShape(0)),
            staged,
            new TIR.Range(2, 2, 2),
            LoopPartition.Tail);
        var rewriter = new EliminateEmptyLoops();

        var rewritten = rewriter.Rewrite(loop);

        Assert.True(rewriter.IsMutated);
        Assert.IsType<Nop>(Assert.IsType<Call>(rewritten).Target);
    }

    [Fact]
    public void TestTritonLoopBackendExposesOnlyExactCpAsyncN2Capability()
    {
        var machine = NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb);
        var backend = TritonLoopPipelineBackend.Instance;

        Assert.False(backend.SupportsStageCount(1));
        Assert.True(backend.SupportsStageCount(2));
        Assert.False(backend.SupportsStageCount(3));
        var template = backend.GetTemplate(2, machine);
        Assert.Equal("triton.loop.cp_async.n2.v1", template.Id.Value);
        Assert.Equal(TritonLoopPipelineBackend.CpAsyncN2Synchronization, template.Synchronization);

        var transfer = GetSharedLoadTransfer(machine);
        var control = template.Synchronization.GetControlCost(machine, transfer);
        Assert.Equal(1, control.ProducerCommitCycles);
        Assert.Equal(26, control.ConsumerWaitAcquireCycles);
        Assert.Equal(25, control.ConsumerReleaseCycles);

        using var solver = new Solver("triton-loop-backend-legality");
        var destination = machine.TilingMemorySpaces.Single(
            space => machine.GetMemoryResource(space).Kind == TargetMemorySpaceKind.Shared);
        var source = machine.GetTilingParentMemorySpace(destination.TilingLevel);
        var legality = backend.GetChannelLegality(
            new(
                source,
                destination,
                DataTypes.Float16,
                ImmutableArray.Create<IntExpr>(solver.MakeIntConst(16)),
                ImmutableArray.Create<IntExpr>(solver.MakeIntConst(64)),
                machine,
                solver),
            stageCount: 2);
        AssertConstant(legality, 1);
    }

    private static PipelineRegionPlan CreatePlan(string scheduleId)
    {
        var source = new TargetMemorySpaceId("gpu.block-global");
        var destination = new TargetMemorySpaceId("gpu.shared");
        return new(
            scheduleId,
            TritonLoopPipelineBackend.CpAsyncN2TemplateId,
            TritonLoopPipelineBackend.CpAsyncN2Synchronization,
            stageCount: 2,
            prefetchDistance: 1,
            PipelineTailPolicy.Serial,
            [new PipelineStageChannelPlan("lhs", source, destination)]);
    }

    private static PipelineFor CreatePipelineFor(
        PipelineRegionPlan plan,
        IVar access,
        BaseExpr allocation,
        TIR.Buffer staged,
        TIR.Range? domain = null,
        LoopPartition partition = LoopPartition.Full)
        => new(
            new DimVar("logical_sequence"),
            domain ?? new TIR.Range(0, 8, 4),
            LoopMode.Reduction,
            partition,
            new Sequential(T.Nop()),
            new Sequential(T.Nop()),
            plan,
            new PipelineRegionId("test", "op7/axis0/entry1"),
            [new PipelineBufferBindingDescriptor(
                "lhs",
                plan.Channels[0].SourceMemorySpace,
                plan.Channels[0].DestinationMemorySpace)],
            [access],
            [allocation],
            [staged]);

    private static TargetMemoryTransferSpec GetSharedLoadTransfer(TargetMachineModel machine)
    {
        var shared = machine.TilingMemorySpaces.Single(
            space => machine.GetMemoryResource(space).Kind == TargetMemorySpaceKind.Shared);
        return machine.GetTransfer(machine.GetTilingParentMemorySpace(shared.TilingLevel).Id, shared.Id);
    }

    private static TIR.Buffer CreateStagedBuffer(string name, int stageCount)
    {
        var encoding = new TargetStorageEncodingSelection(
            new TargetStorageEncodingId("test.shared"),
            physicalBytes: 64,
            alignmentBytes: 4,
            Array.Empty<KeyValuePair<string, long>>());
        var layout = encoding.CreateStagedBufferLayout(stageCount, stageStrideBytes: 64);
        return new TIR.Buffer(
            name,
            DataTypes.Float32,
            new MemSpan(new PhysicalBuffer(
                4,
                layout.PhysicalBytes,
                MemoryLocation.Shared | MemoryLocation.Data)),
            [16],
            [1],
            null,
            encoding,
            layout);
    }

    private static void AssertConstant(IntExpr expression, long expected)
    {
        var variable = expression.Var();
        Assert.Equal(expected, variable.Min());
        Assert.Equal(expected, variable.Max());
    }
}
