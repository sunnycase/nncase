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
    public async Task TestPyNTTShardedViewLowersDirectlyToChipLocalAlias()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 64 });
        var placement = new Placement([4, 8], "yx", "bb");
        var splitType = new DistributedType(
            tensorType,
            [SBP.S([0]), SBP.S([1])],
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
            [SBP.B, SBP.S([1])],
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
        var yxSplitType = new DistributedType(tensorType, [SBP.S([0, 1])], placement);
        var ySplitType = new DistributedType(tensorType, [SBP.S([0])], placement);
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
        var yxSplitType = new DistributedType(tensorType, [SBP.S([0, 1])], placement);
        var xSplitType = new DistributedType(tensorType, [SBP.S([1])], placement);
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
            [SBP.S([0]), SBP.S([1])],
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
            [SBP.B, SBP.S([0, 1], 4)],
            placement);
        var input = new Var("input", inputType);
        var stats = IR.F.NN.NormStats(1, input, useMean: false);
        var partialType = Assert.IsType<DistributedType>(stats.CheckedType);
        Assert.Equal(SBP.P([0, 1], ReduceOp.Sum), partialType.Partial);
        var reduceScatterType = new DistributedType(
            partialType.TensorType,
            [SBP.S([0, 1], 1), SBP.B, SBP.B],
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
    {
        var physical = new PhysicalBuffer(
            dataType.SizeInBytes,
            Tensor.FromPointer(offset, dataType),
            sizeBytes,
            MemoryLocation.Data);
        return new Nncase.TIR.Buffer(
            name,
            dataType,
            new MemSpan(physical, 0, sizeBytes),
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
