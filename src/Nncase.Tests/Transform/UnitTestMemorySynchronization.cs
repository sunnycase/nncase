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
using Nncase.Passes.Transforms;
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
        var parameters = new List<ParameterInfo>();
        call.ParametersForeach((_, parameter) => parameters.Add(parameter));

        Assert.Equal([TIR.NTT.Concat.Input, TIR.NTT.Concat.Input, TIR.NTT.Concat.Output], parameters);
        Assert.Same(output, call[TIR.NTT.Concat.Output]);
    }

    [Fact]
    public async Task TestPyNTTTIRSelectionUsesOperandMemoryEffects()
    {
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

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind).RunAsync(module, new());

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

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind).RunAsync(module, new());

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

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind).RunAsync(module, new());

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

        await new PlanMemorySynchronizationPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewrittenMain = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Equal(3, rewrittenMain.Body.Count);
        Assert.IsType<Call>(rewrittenMain.Body[0]);
        var barrierCall = Assert.IsType<Call>(rewrittenMain.Body[1]);
        Assert.Equal(TIR.NTT.BarrierScope.Chip, Assert.IsType<TIR.NTT.Barrier>(barrierCall.Target).Scope);
        Assert.IsType<Call>(rewrittenMain.Body[2]);
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

    private static Call CreateChipTransfer(Expr buffer)
        => TIR.F.NTT.TensorStore(buffer, buffer, Array.Empty<SBP>(), new Placement([1], "b", "b"));
}
