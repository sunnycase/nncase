// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.TransformTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestMemorySynchronization : TestClassBase
{
    [Fact]
    public void TestNoMemoryEffectIsMergeIdentity()
    {
        Assert.Equal(
            MemoryEffect.ReductionWrite,
            MemoryEffectUtility.Merge(MemoryEffect.None, MemoryEffect.ReductionWrite));
        Assert.Equal(
            MemoryEffect.ReductionReadWrite,
            MemoryEffectUtility.Merge(MemoryEffect.ReductionReadWrite, MemoryEffect.None));

        var mixed = MemoryEffectUtility.Merge(MemoryEffect.Read, MemoryEffect.ReductionWrite);
        Assert.Equal(MemoryAccessMode.ReadWrite, mixed.Mode);
        Assert.Equal(MemoryEffectKind.Direct, mixed.Kind);

        var fixedBlock = MemoryEffectUtility.Merge(
            MemoryEffect.Read.InFixedBlock(3),
            MemoryEffect.Write.InFixedBlock(3));
        Assert.Equal(MemoryAccessDomain.FixedBlock(3), fixedBlock.AccessDomain);

        var differentBlocks = MemoryEffectUtility.Merge(
            MemoryEffect.Read.InFixedBlock(3),
            MemoryEffect.Write.InFixedBlock(4));
        Assert.Equal(MemoryAccessDomain.AllBlocks, differentBlocks.AccessDomain);
    }

    [Fact]
    public void TestReductionAccumulatorReadDoesNotReachPhysicalBuffer()
    {
        Assert.Equal(
            MemoryAccessMode.Write,
            MemoryEffectUtility.GetPhysicalBufferAccessMode(MemoryEffect.ReductionReadWrite));
        Assert.Equal(
            MemoryAccessMode.Write,
            MemoryEffectUtility.GetPhysicalBufferAccessMode(MemoryEffect.ReductionWrite));
        Assert.Equal(
            MemoryAccessMode.ReadWrite,
            MemoryEffectUtility.GetPhysicalBufferAccessMode(MemoryEffect.ReadWrite));
    }

    [Fact]
    public void TestAllNTTKernelOperandsDeclareMemoryEffects()
    {
        var missing = typeof(TIR.NTT.NTTKernelOp).Assembly.GetTypes()
            .Where(type => !type.IsAbstract && type.IsAssignableTo(typeof(TIR.NTT.NTTKernelOp)))
            .SelectMany(type => type.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static))
            .Where(field => field.FieldType == typeof(ParameterInfo))
            .Select(field => (Field: field, Parameter: Assert.IsType<ParameterInfo>(field.GetValue(null))))
            .Where(item => item.Parameter.MemoryEffect is null)
            .Select(item => $"{item.Field.DeclaringType!.Name}.{item.Field.Name}")
            .ToArray();

        Assert.Empty(missing);
    }

    [Fact]
    public void TestVariadicOperandMemoryEffectsMapFinalOutput()
    {
        var input0 = new Var("input0", new TensorType(DataTypes.Float32, new[] { 2 }));
        var input1 = new Var("input1", new TensorType(DataTypes.Float32, new[] { 3 }));
        var output = new Var("output", new TensorType(DataTypes.Float32, new[] { 5 }));
        var call = Assert.IsType<Call>(TIR.F.NTT.Concat([input0, input1], output, 0));
        var concat = Assert.IsType<TIR.NTT.Concat>(call.Target);
        var parameters = new List<ParameterInfo>();
        call.ParametersForeach((_, parameter) => parameters.Add(parameter));

        Assert.Equal(
            [TIR.NTT.Concat.Input, TIR.NTT.Concat.Input, TIR.NTT.Concat.Output, concat.SharedWorkspaceParameter],
            parameters);
        Assert.Same(output, call[TIR.NTT.Concat.Output]);
        Assert.IsType<None>(call[concat.SharedWorkspaceParameter]);
    }

    [Fact]
    public async Task TestPyNTTTIRSelectionUsesOperandMemoryEffects()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4 }));
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Abs, input),
            new[] { input });

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        Assert.Equal(1, lowered.Body.Count);
        var call = Assert.IsType<Call>(lowered.Body[0]);
        Assert.IsType<TIR.NTT.Unary>(call.Target);
        Assert.Equal(MemoryEffect.Read, TIR.NTT.Unary.Input.MemoryEffect);
        Assert.Equal(MemoryEffect.Write, TIR.NTT.Unary.Output.MemoryEffect);
        Assert.Empty(ExprCollector.Collect(lowered.Body).OfType<Block>());
    }

    [Fact]
    public async Task TestPyNTTPagedAttentionCombineSynchronizesOnlySplitAxis()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions
        {
            Hierarchies = new[] { new[] { 4, 8 } },
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
        };
        var placement = new Placement([4, 8], "yx", "bb");
        var layout = new IRArray<AttentionDimKind>(new[]
        {
            AttentionDimKind.Head,
            AttentionDimKind.Dim,
            AttentionDimKind.Seq,
        });
        var statePolicies = new SBP[]
        {
            SBP.SBlockCyclic([1], 2),
            SBP.B,
            SBP.B,
        };
        var maxStateType = new DistributedType(
            new TensorType(DataTypes.Float32, new RankedShape(16, 1, 1)),
            statePolicies,
            placement,
            SBP.P([0], ReduceOp.Max));
        var sumStateType = maxStateType with { Partial = SBP.P([0], ReduceOp.Sum) };
        var accStateType = new DistributedType(
            new TensorType(DataTypes.Float32, new RankedShape(16, 128, 1)),
            statePolicies,
            placement,
            SBP.P([0], ReduceOp.Sum));
        var outputType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new RankedShape(16, 128, 1)),
            [SBP.SBlockCyclic([1], 2), SBP.SBlockCyclic([0], 32), SBP.B],
            placement);
        var maxState = new Var("max_state", maxStateType);
        var sumState = new Var("sum_state", sumStateType);
        var accState = new Var("acc_state", accStateType);
        var combine = IR.F.NTT.PagedAttentionCombine(
            maxState,
            sumState,
            accState,
            None.Default,
            layout,
            2048,
            DataTypes.BFloat16,
            outputType,
            0,
            4);
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            combine,
            [maxState, sumState, accState]);
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        var selectedBarrierCall = Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Barrier));
        var selectedBarrier = Assert.IsType<TIR.NTT.Barrier>(selectedBarrierCall.Target);
        Assert.Equal(new[] { 0 }, selectedBarrier.AxisGroupAxes.ToArray());
        var mergeCall = Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.PagedAttentionMerge));
        var selectedMaxState = Assert.IsType<TIR.Buffer>(mergeCall[TIR.NTT.PagedAttentionMerge.MaxState]);
        var selectedSumState = Assert.IsType<TIR.Buffer>(mergeCall[TIR.NTT.PagedAttentionMerge.SumState]);
        var selectedAccState = Assert.IsType<TIR.Buffer>(mergeCall[TIR.NTT.PagedAttentionMerge.AccState]);
        var seed = CreateWorkspaceBuffer("seed", DataTypes.Float32, 4096, 4, [1]);
        var module = new IRModule(lowered.With(
            body: new Sequential(
                TIR.F.NTT.Unary(UnaryOp.Abs, seed, selectedMaxState),
                TIR.F.NTT.Unary(UnaryOp.Abs, seed, selectedSumState),
                TIR.F.NTT.Unary(UnaryOp.Abs, seed, selectedAccState),
                selectedBarrierCall,
                mergeCall)));
        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var planned = Assert.IsType<PrimFunction>(module.Entry);
        var barriers = ExprCollector.Collect(planned.Body)
            .OfType<Call>()
            .Select(call => call.Target)
            .OfType<TIR.NTT.Barrier>()
            .ToArray();
        var barrier = Assert.Single(barriers);
        Assert.Equal(TIR.NTT.BarrierScope.Chip, barrier.Scope);
        Assert.Equal(new[] { 0 }, barrier.AxisGroupAxes.ToArray());
        Assert.Single(
            ExprCollector.Collect(planned.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.PagedAttentionMerge));
    }

    [Fact]
    public async Task TestPyNTTShardedViewLowersDirectlyToChipLocalAlias()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 64 });
        var placement = new Placement([4, 8], "yx", "bb");
        var splitType = new DistributedType(
            tensorType,
            [SBP.SContiguous([0]), SBP.SContiguous([1])],
            placement);
        var broadcastType = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            placement);
        var input = new Var("input", splitType);
        var producer = IR.F.Math.Unary(UnaryOp.Abs, input);
        var view = IR.F.Distributed.ShardedView(producer, broadcastType);
        var consumer = IR.F.Math.Unary(UnaryOp.Neg, view);
        var result = IR.F.Distributed.Boxing(consumer, tensorType);
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            result,
            new[] { input });
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        var unaryCalls = ExprCollector.Collect(lowered.Body)
            .OfType<Call>()
            .Where(call => call.Target is TIR.NTT.Unary)
            .ToArray();
        Assert.Equal(2, unaryCalls.Length);
        var producerOutput = Assert.IsType<TIR.Buffer>(unaryCalls[0][TIR.NTT.Unary.Output]);
        var consumerInput = Assert.IsType<TIR.Buffer>(unaryCalls[1][TIR.NTT.Unary.Input]);

        Assert.Equal(MemoryLocation.ChipLocalData, producerOutput.MemSpan.Buffer.Location);
        Assert.Same(producerOutput.MemSpan.Buffer, consumerInput.MemSpan.Buffer);
        Assert.Equal(splitType, producerOutput.DistributedType);
        Assert.Equal(broadcastType, consumerInput.DistributedType);
        Assert.Equal(new long[] { 64, 1 }, producerOutput.Strides.ToArray().Select(stride => stride.FixedValue).ToArray());
        Assert.Equal(new long[] { 64, 1 }, consumerInput.Strides.ToArray().Select(stride => stride.FixedValue).ToArray());
        Assert.Equal(8192, producerOutput.MemSpan.Buffer.Size.FixedValue);
        Assert.Equal(1824, producerOutput.MemSpan.Size.FixedValue);
        Assert.Equal(8192, consumerInput.MemSpan.Size.FixedValue);
        Assert.DoesNotContain(
            ExprCollector.Collect(lowered),
            expression => expression is IR.Distributed.ShardedView);
        Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.TensorStore));

        var module = new IRModule(lowered);
        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());
        var planned = Assert.IsType<PrimFunction>(module.Entry);
        _ = Assert.Single(
            ExprCollector.Collect(planned.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Barrier { Scope: TIR.NTT.BarrierScope.Chip }));
    }

    [Fact]
    public async Task TestPyNTTReplicatedBlockShardedViewUsesIdempotentCanonicalWrites()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 64 });
        var placement = new Placement([4, 8], "yx", "bb");
        var xSplitType = new DistributedType(
            tensorType,
            [SBP.B, SBP.SContiguous([1])],
            placement);
        var broadcastType = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            placement);
        var input = new Var("input", xSplitType);
        var producer = IR.F.Math.Unary(UnaryOp.Abs, input);
        var view = IR.F.Distributed.ShardedView(producer, broadcastType);
        var consumer = IR.F.Math.Unary(UnaryOp.Neg, view);
        var result = IR.F.Distributed.Boxing(consumer, tensorType);
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            result,
            new[] { input });
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        var unaryCalls = ExprCollector.Collect(lowered.Body)
            .OfType<Call>()
            .Where(call => call.Target is TIR.NTT.Unary)
            .ToArray();
        Assert.Equal(2, unaryCalls.Length);
        var producerOutput = Assert.IsType<TIR.Buffer>(unaryCalls[0][TIR.NTT.Unary.Output]);
        var consumerInput = Assert.IsType<TIR.Buffer>(unaryCalls[1][TIR.NTT.Unary.Input]);

        Assert.Equal(MemoryLocation.ChipLocalData, producerOutput.MemSpan.Buffer.Location);
        Assert.Same(producerOutput.MemSpan.Buffer, consumerInput.MemSpan.Buffer);
        Assert.Equal(xSplitType, producerOutput.DistributedType);
        Assert.Equal(broadcastType, consumerInput.DistributedType);
        Assert.Equal(new long[] { 32, 8 }, producerOutput.Dimensions.ToArray().Select(dim => dim.FixedValue).ToArray());
        Assert.Equal(new long[] { 64, 1 }, producerOutput.Strides.ToArray().Select(stride => stride.FixedValue).ToArray());
        var producerCoordinates = ExprCollector.Collect(producerOutput.MemSpan.Start)
            .OfType<DimVar>()
            .Select(dimVar => dimVar.Name)
            .ToArray();
        Assert.Contains("__shard_coord_1", producerCoordinates);
        Assert.DoesNotContain("__shard_coord_0", producerCoordinates);
        Assert.Empty(ExprCollector.Collect(consumerInput.MemSpan.Start).OfType<DimVar>());

        var module = new IRModule(lowered);
        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());
        var planned = Assert.IsType<PrimFunction>(module.Entry);
        var barrier = Assert.Single(
            ExprCollector.Collect(planned.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Barrier { Scope: TIR.NTT.BarrierScope.Chip }));
        Assert.Equal(
            new[] { 1 },
            Assert.IsType<TIR.NTT.Barrier>(barrier.Target).AxisGroupAxes.ToArray());
    }

    [Fact]
    public async Task TestAxisGroupBarrierCoverageDoesNotHideLaterFullMeshHazard()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 64 });
        var yxSplitType = new DistributedType(tensorType, [SBP.SContiguous([0, 1])], placement);
        var ySplitType = new DistributedType(tensorType, [SBP.SContiguous([0])], placement);
        var broadcastType = new DistributedType(tensorType, [SBP.B], placement);
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(4096, DataTypes.Float32),
            256,
            MemoryLocation.ChipLocalData);
        var producer = CreateDistributedAlias("producer_yx", physical, yxSplitType);
        var yView = CreateDistributedAlias("view_y", physical, ySplitType);
        var broadcastView = CreateDistributedAlias("view_b", physical, broadcastType);
        var input = CreateWorkspaceBuffer("input", DataTypes.Float32, 0, 256, [64]);
        var yOutput = CreateWorkspaceBuffer("y_output", DataTypes.Float32, 512, 256, [64]);
        var broadcastOutput = CreateWorkspaceBuffer("broadcast_output", DataTypes.Float32, 1024, 256, [64]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                T.Memcopy(producer, input),
                T.Memcopy(yOutput, yView),
                T.Memcopy(broadcastOutput, broadcastView)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var barriers = rewritten.Body.Fields
            .ToArray()
            .OfType<Call>()
            .Select(call => call.Target)
            .OfType<TIR.NTT.Barrier>()
            .ToArray();
        Assert.Collection(
            barriers,
            barrier => Assert.Equal(new[] { 1 }, barrier.AxisGroupAxes.ToArray()),
            barrier => Assert.Empty(barrier.AxisGroupAxes));
    }

    [Fact]
    public async Task TestNonPrefixShardedViewCoarseningUsesFullMeshBarrier()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 64 });
        var yxSplitType = new DistributedType(tensorType, [SBP.SContiguous([0, 1])], placement);
        var xSplitType = new DistributedType(tensorType, [SBP.SContiguous([1])], placement);
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(4096, DataTypes.Float32),
            256,
            MemoryLocation.ChipLocalData);
        var producer = CreateDistributedAlias("producer_yx", physical, yxSplitType);
        var xView = CreateDistributedAlias("view_x", physical, xSplitType);
        var input = CreateWorkspaceBuffer("input", DataTypes.Float32, 0, 256, [64]);
        var output = CreateWorkspaceBuffer("output", DataTypes.Float32, 512, 256, [64]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(T.Memcopy(producer, input), T.Memcopy(output, xView)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var barrier = Assert.Single(
            rewritten.Body.Fields
                .ToArray()
                .OfType<Call>()
                .Select(call => call.Target)
                .OfType<TIR.NTT.Barrier>());
        Assert.Equal(TIR.NTT.BarrierScope.Chip, barrier.Scope);
        Assert.Empty(barrier.AxisGroupAxes);
    }

    [Fact]
    public async Task TestChangedBlockCyclicOwnershipUsesFullMeshBarrier()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 2048 });
        var producerType = new DistributedType(
            tensorType,
            [SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var consumerType = new DistributedType(
            tensorType,
            [SBP.SBlockCyclic([0, 1], 64)],
            placement);
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(4096, DataTypes.Float32),
            8192,
            MemoryLocation.ChipLocalData);
        var producer = CreateDistributedAlias("producer_bc1", physical, producerType);
        var consumer = CreateDistributedAlias("consumer_bc64", physical, consumerType);
        var input = CreateWorkspaceBuffer("input", DataTypes.Float32, 0, 8192, [2048]);
        var output = CreateWorkspaceBuffer("output", DataTypes.Float32, 12288, 8192, [2048]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(T.Memcopy(producer, input), T.Memcopy(output, consumer)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var barrier = Assert.Single(
            rewritten.Body.Fields
                .ToArray()
                .OfType<Call>()
                .Select(call => call.Target)
                .OfType<TIR.NTT.Barrier>());
        Assert.Equal(TIR.NTT.BarrierScope.Chip, barrier.Scope);
        Assert.Empty(barrier.AxisGroupAxes);
    }

    [Fact]
    public async Task TestEquivalentBlockCyclicOwnershipNeedsOnlyBlockBarrier()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 2048 });
        var distributedType = new DistributedType(
            tensorType,
            [SBP.SBlockCyclic([0, 1], 64)],
            placement);
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(4096, DataTypes.Float32),
            8192,
            MemoryLocation.ChipLocalData);
        var producer = CreateDistributedAlias("producer_bc64", physical, distributedType);
        var consumer = CreateDistributedAlias("consumer_bc64", physical, distributedType);
        var input = CreateWorkspaceBuffer("input", DataTypes.Float32, 0, 8192, [2048]);
        var output = CreateWorkspaceBuffer("output", DataTypes.Float32, 12288, 8192, [2048]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(T.Memcopy(producer, input), T.Memcopy(output, consumer)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var barrier = Assert.Single(
            rewritten.Body.Fields
                .ToArray()
                .OfType<Call>()
                .Select(call => call.Target)
                .OfType<TIR.NTT.Barrier>());
        Assert.Equal(TIR.NTT.BarrierScope.Block, barrier.Scope);
    }

    [Fact]
    public async Task TestSamplingPartialReadsOnlyItsLocalVocabularyShard()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var config = new SamplerConfig(
            vocabSize: 1024,
            maxBatchSize: 1,
            maxLogprobs: 0,
            SamplerLogprobsMode.RawLogprobs);
        var packedDataType = new VectorType(DataTypes.BFloat16, [8]);
        var packedType = new DistributedType(
            new TensorType(packedDataType, new[] { 1, 128 }),
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var scalarType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new[] { 1, 1024 }),
            [SBP.B, SBP.SBlockCyclic([0, 1], 8)],
            placement);
        var physical = new PhysicalBuffer(
            packedDataType.SizeInBytes,
            Tensor.FromPointer(4096, packedDataType),
            2048,
            MemoryLocation.Data);
        var packedLogits = CreateDistributedAlias("packed_logits", physical, packedType);
        var scalarLogits = CreateDistributedAlias("scalar_logits", physical, scalarType);
        var input = CreateWorkspaceBuffer("input", packedDataType, 12288, 2048, [1, 128]);
        var processedLogits = CreateDistributedAlias(
            "processed_logits",
            new PhysicalBuffer(
                DataTypes.Float32.SizeInBytes,
                Tensor.FromPointer(12288, DataTypes.Float32),
                4096,
                MemoryLocation.Data),
            new DistributedType(
                new TensorType(DataTypes.Float32, new[] { 1, 1024 }),
                scalarType.AxisPolicies,
                placement));
        var argMaxState = CreateDistributedAlias(
            "argmax_state",
            new PhysicalBuffer(
                DataTypes.UInt64.SizeInBytes,
                Tensor.FromPointer(16384, DataTypes.UInt64),
                256,
                MemoryLocation.ChipLocalData),
            new DistributedType(
                new TensorType(DataTypes.UInt64, new[] { 1 }),
                [SBP.B],
                placement,
                SBP.P([0, 1], ReduceOp.Max)));
        var samplerStateType = new ReferenceType(new SamplerStateType { Config = config });
        var samplerState = CreateBufferView(
            "sampler_state",
            samplerStateType,
            20480,
            samplerStateType.SizeInBytes,
            MemoryLocation.Input,
            0,
            samplerStateType.SizeInBytes,
            Array.Empty<Dimension>());
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                T.Memcopy(packedLogits, input),
                TIR.F.NTT.SamplingPartial(
                    scalarLogits,
                    samplerState,
                    processedLogits,
                    argMaxState,
                    config)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var barrier = Assert.Single(
            ExprCollector.Collect(rewritten.Body)
                .OfType<Call>()
                .Select(call => call.Target)
                .OfType<TIR.NTT.Barrier>());
        Assert.Equal(TIR.NTT.BarrierScope.Block, barrier.Scope);
    }

    [Fact]
    public async Task TestPyNTTLocalShardSubviewAliasesCallerBackingWithoutGridSynchronization()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 64 });
        var placement = new Placement([4, 8], "yx", "bb");
        var broadcastType = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            placement);
        var splitType = new DistributedType(
            tensorType,
            [SBP.SContiguous([0]), SBP.SContiguous([1])],
            placement);
        var residual = new Var("residual", broadcastType);
        var projected = new Var("projected", splitType);
        var localResidual = IR.F.Distributed.ShardedView(residual, splitType);
        var localSum = IR.F.Math.Binary(BinaryOp.Add, localResidual, projected);
        var broadcastResult = IR.F.Distributed.ShardedView(localSum, broadcastType);
        var result = IR.F.Distributed.Boxing(broadcastResult, tensorType);
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            result,
            new[] { residual, projected });
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        var binaryCall = Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.VectorizedBinary));
        var residualView = Assert.IsType<TIR.Buffer>(binaryCall[TIR.NTT.VectorizedBinary.Lhs]);
        var localSumBuffer = Assert.IsType<TIR.Buffer>(binaryCall[TIR.NTT.VectorizedBinary.Output]);
        var tensorStore = Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.TensorStore));
        var broadcastResultBuffer = Assert.IsType<TIR.Buffer>(tensorStore[TIR.NTT.TensorStore.Src]);

        Assert.Equal(MemoryLocation.Input, residualView.MemSpan.Buffer.Location);
        Assert.Equal(splitType, residualView.DistributedType);
        Assert.Equal(new long[] { 8, 8 }, residualView.Dimensions.ToArray().Select(dim => dim.FixedValue).ToArray());
        Assert.Equal(new long[] { 64, 1 }, residualView.Strides.ToArray().Select(stride => stride.FixedValue).ToArray());
        Assert.Equal(1824, residualView.MemSpan.Size.FixedValue);
        Assert.Equal(MemoryLocation.ChipLocalData, localSumBuffer.MemSpan.Buffer.Location);
        Assert.Same(localSumBuffer.MemSpan.Buffer, broadcastResultBuffer.MemSpan.Buffer);
        Assert.Equal(splitType, localSumBuffer.DistributedType);
        Assert.Equal(broadcastType, broadcastResultBuffer.DistributedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(lowered),
            expression => expression is IR.Distributed.ShardedView);

        var module = new IRModule(lowered);
        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());
        var planned = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Single(
            ExprCollector.Collect(planned.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Barrier { Scope: TIR.NTT.BarrierScope.Chip }));
    }

    [Fact]
    public async Task TestPyNTTPartialReduceScatterFeedsBroadcastShardedView()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var placement = new Placement([4, 8], "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(DataTypes.Float32, new[] { 1, 128 }),
            [SBP.B, SBP.SContiguous([0, 1], 4)],
            placement);
        var input = new Var("input", inputType);
        var stats = IR.F.NN.NormStats(1, input, useMean: false);
        var partialType = Assert.IsType<DistributedType>(stats.CheckedType);
        Assert.Equal(SBP.P([0, 1], ReduceOp.Sum), partialType.Partial);
        var reduceScatterType = new DistributedType(
            partialType.TensorType,
            [SBP.SContiguous([0, 1], 1), SBP.B, SBP.B],
            placement);
        var broadcastType = new DistributedType(
            partialType.TensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var reduceScatter = IR.F.Distributed.Boxing(stats, reduceScatterType);
        var broadcastView = IR.F.Distributed.ShardedView(reduceScatter, broadcastType);
        var consumer = IR.F.Math.Unary(UnaryOp.Neg, broadcastView);
        var result = IR.F.Distributed.Boxing(consumer, partialType.TensorType);
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            result,
            new[] { input });
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        var reduceScatterCall = Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.GatherReduceScatter));
        var consumerCall = Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Unary));
        var reduceScatterOutput = Assert.IsType<TIR.Buffer>(
            reduceScatterCall[TIR.NTT.GatherReduceScatter.Output]);
        var consumerInput = Assert.IsType<TIR.Buffer>(consumerCall[TIR.NTT.Unary.Input]);

        Assert.Equal(MemoryLocation.ChipLocalData, reduceScatterOutput.MemSpan.Buffer.Location);
        Assert.Same(reduceScatterOutput.MemSpan.Buffer, consumerInput.MemSpan.Buffer);
        Assert.Equal(reduceScatterType, reduceScatterOutput.DistributedType);
        Assert.Equal(broadcastType, consumerInput.DistributedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(lowered),
            expression => expression is IR.Distributed.ShardedView);

        var module = new IRModule(lowered);
        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());
        var planned = Assert.IsType<PrimFunction>(module.Entry);
        var barriers = ExprCollector.Collect(planned.Body)
            .OfType<Call>()
            .Select(call => call.Target)
            .OfType<TIR.NTT.Barrier>()
            .Where(barrier => barrier.Scope == TIR.NTT.BarrierScope.Chip)
            .ToArray();
        Assert.Equal(2, barriers.Length);
        Assert.All(barriers, barrier => Assert.Empty(barrier.AxisGroupAxes));
    }

    [Fact]
    public async Task TestPyNTTPartialAllReduceWritesCompactLocalReplicaWithoutPostChipBarrier()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var placement = new Placement([4, 8], "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(DataTypes.Float32, new[] { 1, 128 }),
            [SBP.B, SBP.SContiguous([0, 1], 4)],
            placement);
        var input = new Var("input", inputType);
        var stats = IR.F.NN.NormStats(1, input, useMean: false);
        var partialType = Assert.IsType<DistributedType>(stats.CheckedType);
        var broadcastType = new DistributedType(
            partialType.TensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var allReduce = IR.F.Distributed.Boxing(stats, broadcastType);
        var consumer = IR.F.Math.Unary(UnaryOp.Neg, allReduce);
        var result = IR.F.Distributed.Boxing(consumer, partialType.TensorType);
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            result,
            new[] { input });
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        var allReduceCall = Assert.Single(
            ExprCollector.Collect(lowered.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.GatherReduceScatter));
        var allReduceOutput = Assert.IsType<TIR.Buffer>(
            allReduceCall[TIR.NTT.GatherReduceScatter.Output]);
        var partialInput = Assert.IsType<TIR.Buffer>(
            allReduceCall[TIR.NTT.GatherReduceScatter.Input]);

        Assert.Equal(MemoryLocation.ChipLocalData, allReduceOutput.MemSpan.Buffer.Location);
        Assert.Equal(DistributedBufferStorageKind.CompactPerOwner, allReduceOutput.DistributedStorageKind);
        Assert.Equal(MemoryLocation.ChipLocalData, partialInput.MemSpan.Buffer.Location);
        Assert.Equal(DistributedBufferStorageKind.CompactPerOwner, partialInput.DistributedStorageKind);
        Assert.Equal(
            partialInput.MemSpan.Size.FixedValue * 32,
            partialInput.MemSpan.Buffer.Size.FixedValue);
        Assert.Equal(
            allReduceOutput.MemSpan.Size.FixedValue * 32,
            allReduceOutput.MemSpan.Buffer.Size.FixedValue);

        var partialPhysical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(0, DataTypes.Float32),
            4,
            MemoryLocation.Data);
        var allReducePhysical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(256, DataTypes.Float32),
            4,
            MemoryLocation.Data);
        var consumerPhysical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(512, DataTypes.Float32),
            4,
            MemoryLocation.Data);
        var partialBuffer = CreateDistributedAlias("partial", partialPhysical, partialType);
        var broadcastBuffer = CreateDistributedAlias("broadcast", allReducePhysical, broadcastType);
        var consumerBuffer = CreateDistributedAlias("consumer", consumerPhysical, broadcastType);
        var seed = CreateWorkspaceBuffer("seed", DataTypes.Float32, 768, 4, [1, 1, 1]);
        var tir = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                T.Memcopy(partialBuffer, seed),
                TIR.F.NTT.GatherReduceScatter(partialBuffer, broadcastBuffer, partialType, broadcastType),
                TIR.F.NTT.Unary(UnaryOp.Neg, broadcastBuffer, consumerBuffer)),
            Array.Empty<IVar>());
        var module = new IRModule(tir);
        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());
        var planned = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            planned.Body.Fields.ToArray(),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Chip,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<TIR.NTT.GatherReduceScatter>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<TIR.NTT.Unary>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestCanonicalGlobalReshardOutputRequiresPostChipBarrier()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 64 });
        var inputType = new DistributedType(
            tensorType,
            [SBP.SContiguous([0, 1])],
            placement);
        var outputType = new DistributedType(tensorType, [SBP.B], placement);
        var inputPhysical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(4096, DataTypes.Float32),
            256,
            MemoryLocation.ChipLocalData);
        var input = CreateDistributedAlias("reshard_input", inputPhysical, inputType);
        var output = CreateCanonicalReplicatedBuffer(
            "reshard_output",
            DataTypes.Float32,
            8192,
            [64],
            placement);
        var consumer = CreateWorkspaceBuffer(
            "reshard_consumer",
            DataTypes.Float32,
            12288,
            256,
            [64]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                TIR.F.NTT.GatherReduceScatter(input, output, inputType, outputType),
                TIR.F.NTT.Unary(UnaryOp.Neg, output, consumer)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var planned = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            planned.Body.Fields.ToArray(),
            field => Assert.IsType<TIR.NTT.GatherReduceScatter>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Chip,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<TIR.NTT.Unary>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestFixedBlockSamplingResultMaterializationRequiresOnlyBlockBarrier()
    {
        var (combine, sampledIds, placement) = CreateSamplingCombineForSynchronizationTest();
        var output = CreateBufferView(
            "sampled_ids_output",
            DataTypes.Int32,
            49152,
            4,
            MemoryLocation.Output,
            0,
            4,
            [1, 1]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                combine,
                TIR.F.NTT.TensorStore(sampledIds, output, new[] { SBP.B, SBP.B }, placement)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var planned = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            planned.Body.Fields.ToArray(),
            field => Assert.IsType<TIR.NTT.SamplingCombine>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<TIR.NTT.TensorStore>(Assert.IsType<Call>(field).Target));
        Assert.DoesNotContain(
            ExprCollector.Collect(planned.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.Barrier { Scope: TIR.NTT.BarrierScope.Chip });
    }

    [Fact]
    public async Task TestForwardTerminalFixedBlockSamplingResultIntoCallerOutput()
    {
        var (combine, sampledIds, placement) = CreateSamplingCombineForSynchronizationTest();
        var offsetBacking = sampledIds.MemSpan.Buffer.With(size: sampledIds.MemSpan.Size + 64);
        var offsetSampledIds = sampledIds.With(
            memSpan: new MemSpan(offsetBacking, 64, sampledIds.MemSpan.Size));
        var originalCombineCall = Assert.IsType<Call>(combine);
        var combineArguments = originalCombineCall.Arguments.ToArray();
        combineArguments[TIR.NTT.SamplingCombine.SampledIds.Index] = offsetSampledIds;
        combine = originalCombineCall.With(arguments: combineArguments);
        var (outputParameter, output) = CreateCallerOutputBuffer(
            "sampled_ids_output",
            DataTypes.Int32,
            [1, 1]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                combine,
                TIR.F.NTT.TensorStore(offsetSampledIds, output, new[] { SBP.B, SBP.B }, placement)),
            new Return(new Expr[] { output }),
            new IVar[] { outputParameter });
        var module = new IRModule(main);

        await new ForwardTerminalStoreDestinationsPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.TensorStore);
        var combineCall = Assert.Single(
            ExprCollector.Collect(rewritten.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.SamplingCombine));
        var forwarded = Assert.IsType<TIR.Buffer>(combineCall[TIR.NTT.SamplingCombine.SampledIds]);
        var canonicalOutput = Assert.IsType<BufferVar>(forwarded.MemSpan.Buffer.Start);
        Assert.Equal(BufferVarRole.Output, canonicalOutput.Role);
        Assert.Equal(BufferLayoutKind.ExactStrided, canonicalOutput.LayoutAnnotation.Kind);
        Assert.Equal(
            DistributedBufferStorageKind.CompactLocal,
            canonicalOutput.LayoutAnnotation.DistributedStorageKind);
        Assert.Equal(MemoryLocation.Output, forwarded.MemSpan.Buffer.Location);
        Assert.Equal(sampledIds.DistributedType, forwarded.DistributedType);
        Assert.Equal(output.MemSpan.Start, forwarded.MemSpan.Start);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.Barrier);
    }

    [Fact]
    public async Task TestKeepTerminalStoreWhenCanonicalSourceHasAnotherConsumer()
    {
        var (combine, sampledIds, placement) = CreateSamplingCombineForSynchronizationTest();
        var (outputParameter, output) = CreateCallerOutputBuffer(
            "sampled_ids_output",
            DataTypes.Int32,
            [1, 1]);
        var consumerOutput = CreateWorkspaceBuffer(
            "sampled_ids_consumer",
            DataTypes.Int32,
            49152,
            4,
            [1, 1]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                combine,
                TIR.F.NTT.Unary(UnaryOp.Neg, sampledIds, consumerOutput),
                TIR.F.NTT.TensorStore(sampledIds, output, new[] { SBP.B, SBP.B }, placement)),
            new Return(new Expr[] { output }),
            new IVar[] { outputParameter });
        var module = new IRModule(main);

        await new ForwardTerminalStoreDestinationsPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Contains(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.TensorStore);
        var combineCall = Assert.Single(
            ExprCollector.Collect(rewritten.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.SamplingCombine));
        Assert.Same(sampledIds, combineCall[TIR.NTT.SamplingCombine.SampledIds]);
    }

    [Fact]
    public async Task TestKeepTerminalStoreWhenCallerOutputHasAnotherAlias()
    {
        var (combine, sampledIds, placement) = CreateSamplingCombineForSynchronizationTest();
        var (outputParameter, output) = CreateCallerOutputBuffer(
            "sampled_ids_output",
            DataTypes.Int32,
            [1, 1]);
        var outputAlias = output.With(name: "sampled_ids_output_alias");
        var consumerOutput = CreateWorkspaceBuffer(
            "sampled_ids_consumer",
            DataTypes.Int32,
            49152,
            4,
            [1, 1]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                combine,
                TIR.F.NTT.TensorStore(sampledIds, output, new[] { SBP.B, SBP.B }, placement),
                TIR.F.NTT.Unary(UnaryOp.Neg, outputAlias, consumerOutput)),
            new Return(new Expr[] { output }),
            new IVar[] { outputParameter });
        var module = new IRModule(main);

        await new ForwardTerminalStoreDestinationsPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Contains(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.TensorStore);
        var combineCall = Assert.Single(
            ExprCollector.Collect(rewritten.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.SamplingCombine));
        Assert.Same(sampledIds, combineCall[TIR.NTT.SamplingCombine.SampledIds]);
    }

    [Fact]
    public async Task TestFuseTerminalNormCastIntoCallerOutputPreservesSemanticType()
    {
        var placement = new Placement([2, 2], "yx", "bb");
        var semanticType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 1, 4 }),
            [SBP.B, SBP.SContiguous([0, 1])],
            placement);
        var outputType = new DistributedType(
            new TensorType(new VectorType(DataTypes.Float32, [4]), new long[] { 1, 8 }),
            [SBP.B, SBP.SContiguous([0, 1])],
            placement);
        var normOutputPhysical = new PhysicalBuffer(
            16,
            Tensor.FromPointer(49152, semanticType.TensorType.DType),
            64,
            MemoryLocation.Data);
        var normOutput = new Nncase.TIR.Buffer(
            "norm_output",
            semanticType.TensorType.DType,
            new MemSpan(normOutputPhysical, 0, 16),
            [1, 4],
            [0, 1],
            semanticType);
        var outputParameter = new BufferVar(
            "output",
            outputType.TensorType,
            BufferVarRole.Output,
            MemoryLocation.Output);
        var outputPhysical = new PhysicalBuffer(
            16,
            outputParameter,
            128,
            MemoryLocation.Output);
        var output = new Nncase.TIR.Buffer(
            "output_view",
            outputType.TensorType.DType,
            new MemSpan(outputPhysical, 0, 128),
            [1, 2],
            [0, 1],
            outputType,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
        var outputAlias = output.With(name: "output_result");
        var scratch = Enumerable.Range(0, 7)
            .Select(index => CreateWorkspaceBuffer($"norm_scratch_{index}", DataTypes.Float32, (ulong)(65536 + (index * 256)), 256, [64]))
            .ToArray();
        var gather = TIR.F.NTT.GatherReduceAddNormApply(
            scratch[0],
            scratch[1],
            scratch[2],
            scratch[3],
            scratch[4],
            scratch[5],
            scratch[6],
            normOutput,
            semanticType,
            semanticType,
            semanticType,
            1,
            1e-6f,
            false);
        var cast = TIR.F.NTT.Cast(
            normOutput,
            output,
            new VectorType(DataTypes.Float32, [4]),
            CastMode.KDefault,
            [1]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(gather, cast),
            new Return(new Expr[] { outputAlias }),
            new IVar[] { outputParameter });
        var module = new IRModule(main);

        await new ForwardTerminalStoreDestinationsPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.Cast);
        var gatherCall = Assert.Single(
            ExprCollector.Collect(rewritten.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.GatherReduceAddNormApply));
        var gatherOp = Assert.IsType<TIR.NTT.GatherReduceAddNormApply>(gatherCall.Target);
        Assert.Equal(semanticType, gatherOp.NormOutputType);
        Assert.Same(output, gatherCall[TIR.NTT.GatherReduceAddNormApply.NormOutput]);
    }

    [Fact]
    public async Task TestFixedBlockSamplingResultStillRequiresChipBarrierForAllBlockConsumer()
    {
        var (combine, sampledIds, _) = CreateSamplingCombineForSynchronizationTest();
        var consumerOutput = CreateWorkspaceBuffer(
            "sampled_ids_consumer",
            DataTypes.Int32,
            49152,
            4,
            [1, 1]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                combine,
                TIR.F.NTT.Unary(UnaryOp.Neg, sampledIds, consumerOutput)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var planned = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            planned.Body.Fields.ToArray(),
            field => Assert.IsType<TIR.NTT.SamplingCombine>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Chip,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<TIR.NTT.Unary>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestPyNTTTuplePartialAllReduceSharesOneChipBarrier()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var placement = new Placement([4, 8], "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(DataTypes.Float32, new[] { 1, 128 }),
            [SBP.B, SBP.SContiguous([0, 1], 4)],
            placement);
        var input = new Var("input", inputType);
        var partials = Enumerable.Range(0, 3)
            .Select(_ => IR.F.NN.NormStats(1, input, useMean: false))
            .ToArray();
        var partialTypes = partials.Select(partial => Assert.IsType<DistributedType>(partial.CheckedType)).ToArray();
        var outputTypes = partialTypes
            .Select(partialType => (IRType)(partialType with { Partial = null }))
            .ToArray();
        var allReduce = IR.F.Distributed.Boxing(new IR.Tuple(partials), new TupleType(outputTypes));
        var function = new Function(
            "main",
            PyNTTTarget.Kind,
            allReduce,
            new[] { input });
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(function, new()));
        var allReduceCalls = ExprCollector.Collect(lowered.Body)
            .OfType<Call>()
            .Where(call => call.Target is TIR.NTT.GatherReduceScatter)
            .ToArray();
        Assert.Equal(3, allReduceCalls.Length);

        var partialBuffers = allReduceCalls
            .Select(call => Assert.IsType<TIR.Buffer>(call[TIR.NTT.GatherReduceScatter.Input]))
            .ToArray();
        var seed = CreateWorkspaceBuffer("seed", DataTypes.Float32, 4096, 4, [1024]);
        var producer = TIR.F.NTT.PackedQKVParallelLinear(
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            seed,
            partialBuffers[0],
            partialBuffers[1],
            partialBuffers[2],
            1,
            1);

        var synchronizationBody = new Sequential(
            new Expr[] { producer }.Concat(allReduceCalls.Cast<Expr>()).ToArray());
        var module = new IRModule(lowered.With(body: synchronizationBody));
        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());
        var planned = Assert.IsType<PrimFunction>(module.Entry);
        var chipBarriers = ExprCollector.Collect(planned.Body)
            .OfType<Call>()
            .Count(call => call.Target is TIR.NTT.Barrier { Scope: TIR.NTT.BarrierScope.Chip });
        Assert.Equal(1, chipBarriers);
    }

    [Fact]
    public async Task TestInterproceduralUpdatesShareOneOuterChipBarrier()
    {
        var cacheType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var dataType = new TensorType(DataTypes.Float32, new[] { 4 });
        var calleeCache = new BufferVar("callee_cache", cacheType, BufferVarRole.InOut, MemoryLocation.Input);
        var calleeData = new BufferVar("callee_data", dataType, BufferVarRole.InOut, MemoryLocation.Data);
        var loop = new DimVar("tile");
        var update = TIR.F.NTT.UpdatePagedAttentionKVCache(
            calleeData,
            calleeCache,
            0,
            AttentionCacheKind.Key,
            new[] { AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim });
        var callee = new PrimFunction(
            "update_cache",
            PyNTTTarget.Kind,
            new Sequential(new Nncase.TIR.For(loop, new Nncase.TIR.Range(0, 4, 1), LoopMode.Serial, new Sequential(update))),
            new IVar[] { calleeCache, calleeData });

        var cache = new BufferVar("cache", cacheType, BufferVarRole.InOut, MemoryLocation.Input);
        var data = new BufferVar("data", dataType, BufferVarRole.InOut, MemoryLocation.Data);
        var consume = TIR.F.NTT.PagedAttention(
            data,
            cache,
            data,
            data,
            0,
            None.Default,
            data,
            new[] { AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim },
            4);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(callee, cache, data),
                new Call(callee, cache, data),
                consume),
            new IVar[] { cache, data });
        var module = new IRModule(main);
        module.Add(callee);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Equal(4, rewrittenMain.Body.Count);
        Assert.IsType<Call>(rewrittenMain.Body[0]);
        Assert.IsType<Call>(rewrittenMain.Body[1]);
        var barrierCall = Assert.IsType<Call>(rewrittenMain.Body[2]);
        Assert.Equal(TIR.NTT.BarrierScope.Chip, Assert.IsType<TIR.NTT.Barrier>(barrierCall.Target).Scope);
        Assert.IsType<Call>(rewrittenMain.Body[3]);
        Assert.Single(ExprCollector.Collect(rewrittenMain.Body).OfType<Call>().Where(call => call.Target is TIR.NTT.Barrier));
        Assert.Empty(ExprCollector.Collect(rewrittenMain.Body).OfType<Block>());

        var rewrittenCallee = Assert.IsType<PrimFunction>(module.Functions.Single(function => function.Name == "update_cache"));
        var tiledLoop = Assert.Single(ExprCollector.Collect(rewrittenCallee.Body).OfType<Nncase.TIR.For>());
        Assert.DoesNotContain(ExprCollector.Collect(tiledLoop.Body).OfType<Call>(), call => call.Target is TIR.NTT.Barrier { Scope: TIR.NTT.BarrierScope.Chip });
        Assert.DoesNotContain(ExprCollector.Collect(rewrittenCallee.Body).OfType<Call>(), call => call.Target is TIR.NTT.Barrier);
    }

    [Fact]
    public async Task TestInterproceduralDisjointKVCacheLayerPartitionsDoNotSynchronize()
    {
        var cacheType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var dataType = new TensorType(DataTypes.Float32, new[] { 4 });
        var calleeCache = new BufferVar("callee_cache", cacheType, BufferVarRole.InOut, MemoryLocation.Input);
        var calleeData = new BufferVar("callee_data", dataType, BufferVarRole.Input, MemoryLocation.Data);
        var calleeLayerId = new DimVar("callee_layer_id");
        var update = TIR.F.NTT.UpdatePagedAttentionKVCache(
            calleeData,
            calleeCache,
            calleeLayerId,
            AttentionCacheKind.Key,
            new[] { AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim });
        var callee = new PrimFunction(
            "update_cache_layer",
            PyNTTTarget.Kind,
            new Sequential(update),
            new IVar[] { calleeCache, calleeData, calleeLayerId });

        var cache = new BufferVar("cache", cacheType, BufferVarRole.InOut, MemoryLocation.Input);
        var layer0Data = CreateWorkspaceBuffer("layer0_data", DataTypes.Float32, 0, 16, [4]);
        var layer1Data = CreateWorkspaceBuffer("layer1_data", DataTypes.Float32, 16, 16, [4]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(callee, cache, layer0Data, new DimConst(0)),
                new Call(callee, cache, layer1Data, new DimConst(1))),
            new IVar[] { cache });
        var module = new IRModule(main);
        module.Add(callee);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Equal(2, rewrittenMain.Body.Count);
        Assert.All(rewrittenMain.Body.Fields.ToArray(), field => Assert.IsType<Call>(field));
        Assert.DoesNotContain(
            ExprCollector.Collect(rewrittenMain.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.Barrier);
    }

    [Fact]
    public async Task TestInterproceduralDynamicKVCacheLayerPartitionsRemainConservative()
    {
        var cacheType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var dataType = new TensorType(DataTypes.Float32, new[] { 4 });
        var calleeCache = new BufferVar("callee_cache", cacheType, BufferVarRole.InOut, MemoryLocation.Input);
        var calleeData = new BufferVar("callee_data", dataType, BufferVarRole.Input, MemoryLocation.Data);
        var calleeLayerId = new DimVar("callee_layer_id");
        var update = TIR.F.NTT.UpdatePagedAttentionKVCache(
            calleeData,
            calleeCache,
            calleeLayerId,
            AttentionCacheKind.Key,
            new[] { AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim });
        var callee = new PrimFunction(
            "update_dynamic_cache_layer",
            PyNTTTarget.Kind,
            new Sequential(update),
            new IVar[] { calleeCache, calleeData, calleeLayerId });

        var cache = new BufferVar("cache", cacheType, BufferVarRole.InOut, MemoryLocation.Input);
        var layer0Data = CreateWorkspaceBuffer("layer0_data", DataTypes.Float32, 0, 16, [4]);
        var layer1Data = CreateWorkspaceBuffer("layer1_data", DataTypes.Float32, 16, 16, [4]);
        var layer0 = new DimVar("layer0");
        var layer1 = new DimVar("layer1");
        var consume = TIR.F.NTT.PagedAttention(
            layer1Data,
            cache,
            layer1Data,
            layer1Data,
            layer1,
            None.Default,
            layer1Data,
            new[] { AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim },
            4);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(callee, cache, layer0Data, layer0),
                consume),
            new IVar[] { cache, layer0, layer1 });
        var module = new IRModule(main);
        module.Add(callee);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<Call>(field),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Chip,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<Call>(field));
    }

    [Fact]
    public async Task TestExplicitChipScopePropagatesAcrossDataBufferParameter()
    {
        var dataType = new TensorType(DataTypes.Float32, new[] { 4 });
        var calleeData = new BufferVar("callee_data", dataType, BufferVarRole.InOut, MemoryLocation.Data);
        var produce = CreateChipTransfer(calleeData);
        var callee = new PrimFunction(
            "produce_data",
            PyNTTTarget.Kind,
            new Sequential(produce),
            new IVar[] { calleeData });

        var data = new BufferVar("data", dataType, BufferVarRole.InOut, MemoryLocation.Data);
        var consume = CreateChipTransfer(data);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(new Call(callee, data), consume),
            new IVar[] { data });
        var module = new IRModule(main);
        module.Add(callee);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Equal(3, rewrittenMain.Body.Count);
        Assert.IsType<Call>(rewrittenMain.Body[0]);
        var barrierCall = Assert.IsType<Call>(rewrittenMain.Body[1]);
        Assert.Equal(TIR.NTT.BarrierScope.Chip, Assert.IsType<TIR.NTT.Barrier>(barrierCall.Target).Scope);
        Assert.IsType<Call>(rewrittenMain.Body[2]);
    }

    [Fact]
    public async Task TestInterproceduralWorkspaceAliasesUseByteRanges()
    {
        var dataType = new TensorType(DataTypes.Float32, new[] { 64 });
        var calleeOutput = new BufferVar("callee_output", dataType, BufferVarRole.Output, MemoryLocation.Data);
        var produce = CreateChipTransfer(calleeOutput);
        var callee = new PrimFunction(
            "produce_data",
            PyNTTTarget.Kind,
            new Sequential(produce),
            new IVar[] { calleeOutput });

        var produced = CreateWorkspaceBuffer("produced", DataTypes.Float32, 256, 256, [64]);
        var disjoint = CreateWorkspaceBuffer("disjoint", DataTypes.UInt8, 1024, 256, [256]);
        var aliasedView = CreateWorkspaceBuffer("aliased_view", DataTypes.UInt8, 256, 256, [256]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(callee, produced),
                T.Memcopy(disjoint, disjoint),
                T.Memcopy(aliasedView, aliasedView)),
            Array.Empty<IVar>());
        var module = new IRModule(main);
        module.Add(callee);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Equal(4, rewrittenMain.Body.Count);
        Assert.IsType<Call>(rewrittenMain.Body[0]);
        Assert.IsType<Call>(rewrittenMain.Body[1]);
        var barrierCall = Assert.IsType<Call>(rewrittenMain.Body[2]);
        Assert.Equal(TIR.NTT.BarrierScope.Chip, Assert.IsType<TIR.NTT.Barrier>(barrierCall.Target).Scope);
        Assert.IsType<Call>(rewrittenMain.Body[3]);
        Assert.Single(ExprCollector.Collect(rewrittenMain.Body).OfType<Call>().Where(call => call.Target is TIR.NTT.Barrier));
    }

    [Fact]
    public async Task TestInterproceduralProducerConsumerWorkspaceAliasesUseByteRanges()
    {
        var producerType = new TensorType(DataTypes.Float32, new[] { 64 });
        var producerOutput = new BufferVar("producer_output", producerType, BufferVarRole.Output, MemoryLocation.Data);
        var producer = new PrimFunction(
            "produce_data",
            PyNTTTarget.Kind,
            new Sequential(CreateChipTransfer(producerOutput)),
            new IVar[] { producerOutput });

        var consumerType = new TensorType(DataTypes.UInt8, new[] { 256 });
        var consumerInput = new BufferVar("consumer_input", consumerType, BufferVarRole.Input, MemoryLocation.Data);
        var consumerView = new Var("consumer_view");
        var consumer = new PrimFunction(
            "consume_data",
            PyNTTTarget.Kind,
            new Sequential(
                new Let(
                    consumerView,
                    IR.F.Buffer.BufferSubview(consumerInput, new Dimension[] { 0 }, new Dimension[] { 256 }),
                    new Sequential(T.Memcopy(consumerView, consumerView)))),
            new IVar[] { consumerInput });
        Assert.True(consumer.InferenceType());

        var produced = CreateWorkspaceBuffer("produced", DataTypes.Float32, 256, 256, [64]);
        var aliasedView = CreateWorkspaceBuffer("aliased_view", DataTypes.UInt8, 256, 256, [256]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(new Call(producer, produced), new Call(consumer, aliasedView)),
            Array.Empty<IVar>());
        var module = new IRModule(main);
        module.Add(producer);
        module.Add(consumer);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Equal(3, rewrittenMain.Body.Count);
        Assert.IsType<Call>(rewrittenMain.Body[0]);
        var barrierCall = Assert.IsType<Call>(rewrittenMain.Body[1]);
        Assert.Equal(TIR.NTT.BarrierScope.Chip, Assert.IsType<TIR.NTT.Barrier>(barrierCall.Target).Scope);
        Assert.IsType<Call>(rewrittenMain.Body[2]);
    }

    [Fact]
    public async Task TestInterproceduralCompactPerOwnerStoreRequiresOnlyBlockBarrier()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 8 });
        var placement = new Placement([2], "b", "b");
        var distributedType = new DistributedType(
            tensorType,
            new SBP[] { SBP.SContiguous([0], 1) },
            placement);
        var compactLayout = BufferLayoutAnnotation.ExactStrided(
            new Dimension[] { 1 },
            DistributedBufferStorageKind.CompactPerOwner);

        var producerOutputVar = new BufferVar(
            "producer_output",
            distributedType,
            BufferVarRole.Output,
            MemoryLocation.Output,
            compactLayout);
        var producerOutput = T.AttachBuffer(
            producerOutputVar,
            tensorType,
            MemoryLocation.Output,
            0,
            out _,
            "producer_output_view",
            distributedType);
        var producerSource = CreateCompactPerOwnerBuffer(
            "producer_source",
            distributedType,
            2048);
        var producer = new PrimFunction(
            "produce_compact",
            PyNTTTarget.Kind,
            new Sequential(TIR.F.NTT.TensorStore(
                producerSource,
                producerOutput,
                distributedType.AxisPolicies,
                placement)),
            new IVar[] { producerOutputVar });

        var consumerInputVar = new BufferVar(
            "consumer_input",
            distributedType,
            BufferVarRole.Input,
            MemoryLocation.Input,
            compactLayout);
        var consumerInput = T.AttachBuffer(
            consumerInputVar,
            tensorType,
            MemoryLocation.Input,
            0,
            out _,
            "consumer_input_view",
            distributedType);
        var consumerOutput = CreateCompactPerOwnerBuffer(
            "consumer_output",
            distributedType,
            4096);
        var consumer = new PrimFunction(
            "consume_compact",
            PyNTTTarget.Kind,
            new Sequential(TIR.F.NTT.Unary(UnaryOp.Abs, consumerInput, consumerOutput)),
            new IVar[] { consumerInputVar });

        var intermediate = CreateCompactPerOwnerBuffer(
            "intermediate",
            distributedType,
            0);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(producer, intermediate),
                new Call(consumer, intermediate)),
            Array.Empty<IVar>());
        var module = new IRModule(main);
        module.Add(producer);
        module.Add(consumer);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.Equal(
                "produce_compact",
                Assert.IsType<PrimFunction>(Assert.IsType<Call>(field).Target).Name),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.Equal(
                "consume_compact",
                Assert.IsType<PrimFunction>(Assert.IsType<Call>(field).Target).Name));
    }

    [Fact]
    public async Task TestPartialOwnerGroupConsumerSynchronizesOnlyPartialAxes()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var partialType = new DistributedType(
            new TensorType(DataTypes.Float32, new[] { 1, 128 }),
            new SBP[] { SBP.B, SBP.SContiguous([1], 16) },
            placement,
            SBP.P([0], ReduceOp.Sum));
        var partial = CreateCompactPerOwnerBuffer("partial", partialType, 0);
        var producerInput = CreateCompactPerOwnerBuffer("producer_input", partialType, 65536);
        var state = CreateWorkspaceBuffer("state", DataTypes.Float32, 131072, 4, [1]);
        var convWeight = CreateWorkspaceBuffer("conv_weight", DataTypes.Float32, 131076, 16, [4]);
        var output = CreateWorkspaceBuffer("output", DataTypes.Float32, 131092, 512, [1, 128]);
        var producer = TIR.F.NTT.Unary(UnaryOp.Abs, producerInput, partial);
        var consumer = TIR.F.NTT.GatedDeltaNetConvolution(
            partial,
            state,
            convWeight,
            output,
            0,
            4);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(producer, consumer),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<TIR.NTT.Unary>(Assert.IsType<Call>(field).Target),
            field =>
            {
                var barrier = Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target);
                Assert.Equal(TIR.NTT.BarrierScope.Chip, barrier.Scope);
                Assert.Equal(new[] { 0 }, barrier.AxisGroupAxes.ToArray());
            },
            field => Assert.IsType<TIR.NTT.GatedDeltaNetConvolution>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestInterproceduralBlockLocalProducerConsumerIsSynchronized()
    {
        var dataType = new TensorType(DataTypes.Float32, new[] { 64 });
        var producerInput = new BufferVar("producer_input", dataType, BufferVarRole.Input, MemoryLocation.Input);
        var producerOutput = new BufferVar("producer_output", dataType, BufferVarRole.Output, MemoryLocation.Data);
        var producer = new PrimFunction(
            "produce_data",
            PyNTTTarget.Kind,
            new Sequential(T.Memcopy(producerOutput, producerInput)),
            new IVar[] { producerInput, producerOutput });

        var consumerInput = new BufferVar("consumer_input", dataType, BufferVarRole.Input, MemoryLocation.Data);
        var consumerOutput = new BufferVar("consumer_output", dataType, BufferVarRole.Output, MemoryLocation.Output);
        var consumer = new PrimFunction(
            "consume_data",
            PyNTTTarget.Kind,
            new Sequential(T.Memcopy(consumerOutput, consumerInput)),
            new IVar[] { consumerInput, consumerOutput });

        var source = CreateWorkspaceBuffer("source", DataTypes.Float32, 512, 256, [64]);
        var intermediate = CreateWorkspaceBuffer("intermediate", DataTypes.Float32, 0, 256, [64]);
        var destination = CreateWorkspaceBuffer("destination", DataTypes.Float32, 1024, 256, [64]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Call(producer, source, intermediate),
                new Call(consumer, intermediate, destination)),
            Array.Empty<IVar>());
        var module = new IRModule(main);
        module.Add(producer);
        module.Add(consumer);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.Equal("produce_data", Assert.IsType<PrimFunction>(Assert.IsType<Call>(field).Target).Name),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.Equal("consume_data", Assert.IsType<PrimFunction>(Assert.IsType<Call>(field).Target).Name));
    }

    [Fact]
    public async Task TestPhysicalWorkspaceReuseSynchronizesDistinctLogicalWriters()
    {
        var firstSource = CreateWorkspaceBuffer("first_source", DataTypes.Float32, 512, 256, [64]);
        var secondSource = CreateWorkspaceBuffer("second_source", DataTypes.Float32, 1024, 256, [64]);
        var firstLifetime = CreateWorkspaceBuffer("first_lifetime", DataTypes.Float32, 0, 256, [64]);
        var reusedLifetime = CreateWorkspaceBuffer("reused_lifetime", DataTypes.Float32, 0, 256, [64]);
        var placement = new Placement([1], "b", "b");
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                TIR.F.NTT.TensorStore(firstSource, firstLifetime, Array.Empty<SBP>(), placement),
                TIR.F.NTT.TensorStore(secondSource, reusedLifetime, Array.Empty<SBP>(), placement)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<TIR.NTT.TensorStore>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Chip,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<TIR.NTT.TensorStore>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestDynamicSubviewOfDisjointPhysicalAllocationDoesNotRequireChipBarrier()
    {
        var shardCoordinate = new DimVar("shard_coordinate") { Metadata = { Range = new(0, 31) } };
        var producerInput = CreateBufferView(
            "producer_input",
            DataTypes.Float32,
            10240,
            4096,
            MemoryLocation.ChipLocalData,
            0,
            64,
            [16]);
        var consumerOutput = CreateBufferView(
            "consumer_output",
            DataTypes.Float32,
            14336,
            2048,
            MemoryLocation.ChipLocalData,
            shardCoordinate * 64,
            64,
            [16]);
        var intermediate = CreateWorkspaceBuffer("intermediate", DataTypes.Float32, 0, 64, [16]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                T.Memcopy(intermediate, producerInput),
                T.Memcopy(consumerOutput, intermediate)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestDynamicSubviewOfSamePhysicalAllocationRemainsConservativelyAliased()
    {
        var shardCoordinate = new DimVar("shard_coordinate") { Metadata = { Range = new(0, 31) } };
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(10240, DataTypes.Float32),
            4096,
            MemoryLocation.ChipLocalData);
        var producerInput = CreateBufferView("producer_input", DataTypes.Float32, physical, 0, 64, [16]);
        var consumerOutput = CreateBufferView(
            "consumer_output",
            DataTypes.Float32,
            physical,
            shardCoordinate * 64,
            64,
            [16]);
        var intermediate = CreateWorkspaceBuffer("intermediate", DataTypes.Float32, 0, 64, [16]);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                T.Memcopy(intermediate, producerInput),
                T.Memcopy(consumerOutput, intermediate)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Chip,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestReductionLoopProtectsReusedBlockLocalStagingBuffer()
    {
        var source = CreateWorkspaceBuffer("source", DataTypes.Float32, 0, 256, [64]);
        var destination = CreateWorkspaceBuffer("destination", DataTypes.Float32, 512, 256, [64]);
        var sharedPhysical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(0, DataTypes.Float32),
            256,
            MemoryLocation.Shared);
        var shared = new Nncase.TIR.Buffer(
            "staging",
            DataTypes.Float32,
            new MemSpan(sharedPhysical, 0, 256),
            new Dimension[] { 64 },
            new Dimension[] { 1 },
            null);
        var tile = new DimVar("k_tile");
        var loop = new Nncase.TIR.For(
            tile,
            new Nncase.TIR.Range(0, 4, 1),
            LoopMode.Reduction,
            new Sequential(
                T.Memcopy(shared, source),
                T.Memcopy(destination, shared)));
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(loop),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        var rewrittenLoop = Assert.Single(ExprCollector.Collect(rewrittenMain.Body).OfType<Nncase.TIR.For>());
        Assert.Collection(
            rewrittenLoop.Body.Fields.ToArray(),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope));
    }

    [Fact]
    public async Task TestNestedSynchronizedLoopDoesNotAddOuterExitBarrier()
    {
        var source = CreateWorkspaceBuffer("nested_source", DataTypes.Float32, 0, 256, [64]);
        var destination = CreateWorkspaceBuffer("nested_destination", DataTypes.Float32, 512, 256, [64]);
        var shared = CreateSharedBuffer("nested_staging", 0);
        var inner = new Nncase.TIR.For(
            new DimVar("nested_inner"),
            new Nncase.TIR.Range(0, 4, 1),
            LoopMode.Reduction,
            new Sequential(
                T.Memcopy(shared, source),
                T.Memcopy(destination, shared)));
        var outer = new Nncase.TIR.For(
            new DimVar("nested_outer"),
            new Nncase.TIR.Range(0, 2, 1),
            LoopMode.Serial,
            new Sequential(inner));
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(outer),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        var rewrittenOuter = Assert.IsType<Nncase.TIR.For>(Assert.Single(rewrittenMain.Body.Fields.ToArray()));
        var rewrittenInner = Assert.IsType<Nncase.TIR.For>(Assert.Single(rewrittenOuter.Body.Fields.ToArray()));
        Assert.Equal(
            2,
            ExprCollector.Collect(rewrittenInner.Body)
                .OfType<Call>()
                .Count(call => call.Target is TIR.NTT.Barrier));
    }

    [Fact]
    public async Task TestSynchronizedZeroTripLoopDoesNotDischargeEarlierEffects()
    {
        var source = CreateWorkspaceBuffer("zero_trip_source", DataTypes.Float32, 0, 256, [64]);
        var destination = CreateWorkspaceBuffer("zero_trip_destination", DataTypes.Float32, 512, 256, [64]);
        var loopSource = CreateWorkspaceBuffer("zero_trip_loop_source", DataTypes.Float32, 1024, 256, [64]);
        var loopDestination = CreateWorkspaceBuffer("zero_trip_loop_destination", DataTypes.Float32, 1536, 256, [64]);
        var shared = CreateSharedBuffer("zero_trip_pending", 0);
        var loopShared = CreateSharedBuffer("zero_trip_loop_staging", 256);
        var loop = new Nncase.TIR.For(
            new DimVar("zero_trip_loop"),
            new Nncase.TIR.Range(0, 0, 1),
            LoopMode.Reduction,
            new Sequential(
                T.Memcopy(loopShared, loopSource),
                T.Memcopy(loopDestination, loopShared)));
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                T.Memcopy(shared, source),
                loop,
                T.Memcopy(destination, shared)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target),
            field => Assert.IsType<Nncase.TIR.For>(field),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestUnsynchronizedLoopEffectStillRequiresBoundaryBarrier()
    {
        var source = CreateWorkspaceBuffer("boundary_source", DataTypes.Float32, 0, 256, [64]);
        var destination = CreateWorkspaceBuffer("boundary_destination", DataTypes.Float32, 512, 256, [64]);
        var shared = CreateSharedBuffer("boundary_staging", 0);
        var loop = new Nncase.TIR.For(
            new DimVar("boundary_loop"),
            new Nncase.TIR.Range(0, 4, 1),
            LoopMode.Serial,
            new Sequential(T.Memcopy(shared, source)));
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(loop, T.Memcopy(destination, shared)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<Nncase.TIR.For>(field),
            field => Assert.Equal(
                TIR.NTT.BarrierScope.Block,
                Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(field).Target).Scope),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestAsyncCopyPipelinePlansPhasesIndependentlyAndDischargesBlockEffects()
    {
        var source = CreateWorkspaceBuffer("pipeline_source", DataTypes.Float32, 4096, 256, [64]);
        var destination = CreateWorkspaceBuffer("pipeline_destination", DataTypes.Float32, 8192, 256, [64]);
        var boundaryDestination = CreateWorkspaceBuffer("pipeline_boundary_destination", DataTypes.Float32, 12288, 256, [64]);
        var staged = CreateStagedSharedBuffer("pipeline_staged", 0);
        var stagedAccess = new Var(
            "pipeline_staged_access",
            new TensorType(DataTypes.Float32, new[] { 64 }));
        var allocation = IR.F.Buffer.AllocateBufferView(staged, new RankedShape(0));
        var stage = CreatePipelineStageAlias(staged, "pipeline_staged_stage", 0);
        var loop = CreateAsyncCopyPipelineLoop(
            stagedAccess,
            allocation,
            staged,
            new Sequential(T.Memcopy(stage, source)),
            new Sequential(T.Memcopy(destination, stage)));
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(loop, T.Memcopy(boundaryDestination, staged)),
            Array.Empty<IVar>());
        Assert.True(main.InferenceType());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind, MemorySynchronizationScopes.All).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field =>
            {
                var rewrittenLoop = Assert.IsType<Nncase.TIR.PipelineFor>(field);
                Assert.DoesNotContain(
                    ExprCollector.Collect(rewrittenLoop.ProduceBody).OfType<Call>(),
                    call => call.Target is TIR.NTT.Barrier);
                Assert.DoesNotContain(
                    ExprCollector.Collect(rewrittenLoop.ConsumeBody).OfType<Call>(),
                    call => call.Target is TIR.NTT.Barrier);
            },
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target));
    }

    [Fact]
    public async Task TestBackendManagedBlockSynchronizationIsNotMaterialized()
    {
        var source = CreateWorkspaceBuffer("source", DataTypes.Float32, 0, 256, [64]);
        var destination = CreateWorkspaceBuffer("destination", DataTypes.Float32, 512, 256, [64]);
        var sharedPhysical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(0, DataTypes.Float32),
            256,
            MemoryLocation.Shared);
        var shared = new Nncase.TIR.Buffer(
            "staging",
            DataTypes.Float32,
            new MemSpan(sharedPhysical, 0, 256),
            new Dimension[] { 64 },
            new Dimension[] { 1 },
            null);
        var main = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                T.Memcopy(shared, source),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
                T.Memcopy(destination, shared)),
            Array.Empty<IVar>());
        var module = new IRModule(main);

        await new PlanMemorySynchronizationPass(
            PyNTTTarget.Kind,
            MemorySynchronizationScopes.Chip).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Collection(
            rewrittenMain.Body.Fields.ToArray(),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target),
            field => Assert.IsType<Memcopy>(Assert.IsType<Call>(field).Target));
        Assert.DoesNotContain(
            ExprCollector.Collect(rewrittenMain.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.Barrier);
    }

    private static Nncase.TIR.Buffer CreateWorkspaceBuffer(
        string name,
        DataType dataType,
        ulong offset,
        long sizeBytes,
        Dimension[] shape)
        => CreateBufferView(
            name,
            dataType,
            offset,
            sizeBytes,
            MemoryLocation.Data,
            0,
            sizeBytes,
            shape);

    private static (BufferVar Parameter, Nncase.TIR.Buffer Buffer) CreateCallerOutputBuffer(
        string name,
        DataType dataType,
        Dimension[] shape)
    {
        var tensorType = new TensorType(dataType, shape);
        var parameter = new BufferVar(
            name,
            tensorType,
            BufferVarRole.Output,
            MemoryLocation.Output);
        var sizeBytes = shape.Aggregate(
            (long)dataType.SizeInBytes,
            (size, dimension) => checked(size * dimension.FixedValue));
        var physical = new PhysicalBuffer(
            dataType.SizeInBytes,
            parameter,
            sizeBytes,
            MemoryLocation.Output);
        var buffer = new Nncase.TIR.Buffer(
            $"{name}_view",
            dataType,
            new MemSpan(physical, 0, sizeBytes),
            shape,
            TensorUtilities.GetDefaultStrides(shape)
                .Select(stride => (Dimension)stride)
                .ToArray(),
            distributedType: null);
        return (parameter, buffer);
    }

    private static (Expr Combine, Nncase.TIR.Buffer SampledIds, Placement Placement) CreateSamplingCombineForSynchronizationTest()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var config = new SamplerConfig(
            vocabSize: 128,
            maxBatchSize: 1,
            maxLogprobs: 0,
            SamplerLogprobsMode.RawLogprobs);
        var state = new Var(
            "sampler_state",
            TensorType.Scalar(new ReferenceType(new SamplerStateType { Config = config })));
        var logits = CreateWorkspaceBuffer("sampling_logits", DataTypes.Float32, 0, 512, [1, 128]);
        var processedLogits = CreateWorkspaceBuffer("sampling_processed", DataTypes.Float32, 1024, 512, [1, 128]);
        var argMaxState = CreateWorkspaceBuffer("sampling_argmax", DataTypes.UInt64, 2048, 256, [1, 32]);
        var summary = CreateWorkspaceBuffer("sampling_summary", DataTypes.Float32, 3072, 34816, [1, 32, 272]);
        var sampledIds = CreateCanonicalReplicatedBuffer("sampling_ids", DataTypes.Int32, 40960, [1, 1], placement);
        var logprobIds = CreateCanonicalReplicatedBuffer("sampling_logprob_ids", DataTypes.Int32, 41216, [1, 1], placement);
        var logprobs = CreateCanonicalReplicatedBuffer("sampling_logprobs", DataTypes.Float32, 41472, [1, 1], placement);
        var ranks = CreateCanonicalReplicatedBuffer("sampling_ranks", DataTypes.Int32, 41728, [1], placement);
        var counts = CreateCanonicalReplicatedBuffer("sampling_counts", DataTypes.Int32, 41984, [1], placement);
        var combine = TIR.F.NTT.SamplingCombine(
            logits,
            processedLogits,
            argMaxState,
            state,
            summary,
            sampledIds,
            logprobIds,
            logprobs,
            ranks,
            counts,
            config,
            blockCount: 32);
        return (combine, sampledIds, placement);
    }

    private static Nncase.TIR.Buffer CreateCanonicalReplicatedBuffer(
        string name,
        DataType dataType,
        ulong offset,
        Dimension[] shape,
        Placement placement)
    {
        var sizeBytes = shape.Aggregate(1L, (size, dimension) => checked(size * dimension.FixedValue)) *
            dataType.SizeInBytes;
        var physical = new PhysicalBuffer(
            dataType.SizeInBytes,
            Tensor.FromPointer(offset, dataType),
            sizeBytes,
            MemoryLocation.ChipLocalData);
        var tensorType = new TensorType(dataType, shape);
        var distributedType = new DistributedType(
            tensorType,
            Enumerable.Repeat<SBP>(SBP.B, placement.Rank).ToArray(),
            placement);
        return new Nncase.TIR.Buffer(
            name,
            dataType,
            new MemSpan(physical, 0, sizeBytes),
            shape,
            TensorUtilities.GetDefaultStrides(shape)
                .Select(stride => (Dimension)stride)
                .ToArray(),
            distributedType,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
    }

    private static Nncase.TIR.Buffer CreateBufferView(
        string name,
        DataType dataType,
        ulong allocationOffset,
        long allocationSizeBytes,
        MemoryLocation location,
        Dimension spanStart,
        Dimension spanSize,
        Dimension[] shape)
    {
        var physical = new PhysicalBuffer(
            dataType.SizeInBytes,
            Tensor.FromPointer(allocationOffset, dataType),
            allocationSizeBytes,
            location);
        return CreateBufferView(name, dataType, physical, spanStart, spanSize, shape);
    }

    private static Nncase.TIR.Buffer CreateBufferView(
        string name,
        DataType dataType,
        PhysicalBuffer physical,
        Dimension spanStart,
        Dimension spanSize,
        Dimension[] shape)
    {
        return new Nncase.TIR.Buffer(
            name,
            dataType,
            new MemSpan(physical, spanStart, spanSize),
            shape,
            TensorUtilities.GetDefaultStrides(shape).Select(stride => (Dimension)stride).ToArray(),
            null);
    }

    private static Nncase.TIR.Buffer CreateDistributedAlias(
        string name,
        PhysicalBuffer physical,
        DistributedType distributedType)
    {
        var shape = distributedType.TensorType.Shape;
        var strides = TensorUtilities.GetDefaultStrides(
            shape.ToValueArray().Select(value => checked((int)value)).ToArray());
        return new(
            name,
            distributedType.TensorType.DType,
            new MemSpan(physical, 0, physical.Size),
            shape.ToArray(),
            strides.Select(stride => (Dimension)stride).ToArray(),
            distributedType);
    }

    private static Nncase.TIR.Buffer CreateCompactPerOwnerBuffer(
        string name,
        DistributedType distributedType,
        ulong allocationOffset)
    {
        var localTensorType = DistributedUtility.GetDividedTensorType(distributedType);
        var localShape = ((RankedShape)localTensorType.Shape).Dimensions.ToArray();
        var componentSize = localShape.Aggregate(
            (long)distributedType.TensorType.DType.SizeInBytes,
            (size, dimension) => checked(size * dimension.FixedValue));
        var ownerCount = distributedType.Placement.Hierarchy.Aggregate(
            1L,
            (count, extent) => checked(count * extent));
        var physical = new PhysicalBuffer(
            distributedType.TensorType.DType.SizeInBytes,
            Tensor.FromPointer(allocationOffset, distributedType.TensorType.DType),
            checked(componentSize * ownerCount),
            MemoryLocation.ChipLocalData);
        var globalShape = ((RankedShape)distributedType.TensorType.Shape).Dimensions.ToArray();
        return new Nncase.TIR.Buffer(
            name,
            distributedType.TensorType.DType,
            new MemSpan(physical, 0, componentSize),
            globalShape,
            TensorUtilities.GetDefaultStrides(localShape)
                .Select(stride => (Dimension)stride)
                .ToArray(),
            distributedType,
            distributedStorageKind: DistributedBufferStorageKind.CompactPerOwner);
    }

    private static Nncase.TIR.Buffer CreateSharedBuffer(string name, ulong offset)
    {
        const long sizeBytes = 256;
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(offset, DataTypes.Float32),
            sizeBytes,
            MemoryLocation.Shared);
        return new Nncase.TIR.Buffer(
            name,
            DataTypes.Float32,
            new MemSpan(physical, 0, sizeBytes),
            new Dimension[] { 64 },
            new Dimension[] { 1 },
            null);
    }

    private static Nncase.TIR.Buffer CreateStagedSharedBuffer(string name, ulong offset)
    {
        const long stageBytes = 256;
        const int stageCount = 2;
        var encoding = new TargetStorageEncodingSelection(
            TargetStorageEncodingIds.Linear,
            stageBytes,
            DataTypes.Float32.SizeInBytes,
            Array.Empty<KeyValuePair<string, long>>());
        var layout = encoding.CreateStagedBufferLayout(stageCount, stageBytes);
        var physical = new PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            Tensor.FromPointer(offset, DataTypes.Float32),
            layout.PhysicalBytes,
            MemoryLocation.Shared);
        return new Nncase.TIR.Buffer(
            name,
            DataTypes.Float32,
            new MemSpan(physical, 0, layout.PhysicalBytes),
            new Dimension[] { 64 },
            new Dimension[] { 1 },
            null,
            encoding,
            layout);
    }

    private static Nncase.TIR.Buffer CreatePipelineStageAlias(
        Nncase.TIR.Buffer source,
        string name,
        Dimension byteOffset)
    {
        var layout = source.StagedLayout ?? throw new ArgumentException(
            $"Buffer {source.Name} is not a staged allocation.",
            nameof(source));
        return new Nncase.TIR.Buffer(
            name,
            source.ElemType,
            source.MemSpan.With(
                start: source.MemSpan.Start + byteOffset,
                size: layout.StagePhysicalBytes),
            source.Dimensions.ToArray(),
            source.Strides.ToArray(),
            source.DistributedType,
            source.StorageEncoding);
    }

    private static Nncase.TIR.PipelineFor CreateAsyncCopyPipelineLoop(
        IVar stagedAccess,
        Expr allocation,
        Nncase.TIR.Buffer staged,
        Sequential copyBody,
        Sequential computeBody)
    {
        var plan = new PipelineRegionPlan(
            "test.cp_async.n2",
            TritonLoopPipelineBackend.CpAsyncN2TemplateId,
            TritonLoopPipelineBackend.CpAsyncN2Synchronization,
            stageCount: 2,
            prefetchDistance: 1,
            PipelineTailPolicy.Serial,
            [
                new PipelineStageChannelPlan(
                    "tile",
                    new TargetMemorySpaceId("gpu.block-global"),
                    new TargetMemorySpaceId("gpu.shared")),
            ]);
        return new Nncase.TIR.PipelineFor(
            new DimVar("pipeline_k"),
            new Nncase.TIR.Range(0, 2, 1),
            LoopMode.Reduction,
            LoopPartition.Full,
            copyBody,
            computeBody,
            plan,
            new PipelineRegionId("test", "op0/reduction0"),
            [new(
                "tile",
                new TargetMemorySpaceId("gpu.block-global"),
                new TargetMemorySpaceId("gpu.shared"))],
            [stagedAccess],
            [allocation],
            [staged]);
    }

    private static Call CreateChipTransfer(Expr buffer)
        => TIR.F.NTT.TensorStore(buffer, buffer, Array.Empty<SBP>(), new Placement([1], "b", "b"));
}
