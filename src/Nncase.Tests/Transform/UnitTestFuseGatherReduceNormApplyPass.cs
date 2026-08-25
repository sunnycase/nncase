// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TransformTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFuseGatherReduceNormApplyPass : TestClassBase
{
    [Fact]
    public async Task TestFuseSingleUsePartialStatistics()
    {
        var (function, _, broadcastStats) = CreateFunction(
            additionalStatsConsumer: false,
            addInterveningNop: true);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var fusedCall = Assert.Single(rewritten.Body.Fields.ToArray().OfType<Call>());
        var fused = Assert.IsType<Nncase.TIR.NTT.GatherReduceNormApply>(fusedCall.Target);
        Assert.True(fused.HasBias);
        Assert.Equal(new[] { 0, 1 }, fused.InStatsType.Partial!.Axes.ToArray());
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.GatherReduceScatter or Nncase.TIR.NTT.NormApply);
        Assert.DoesNotContain(
            fusedCall.Arguments.ToArray(),
            argument => ReferenceEquals(argument, broadcastStats));
    }

    [Fact]
    public async Task TestElideConstantZeroBias()
    {
        var (function, _, _) = CreateFunction(
            additionalStatsConsumer: false,
            addInterveningNop: false,
            zeroBias: true);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyPass(PyNTTTarget.Kind).RunAsync(module, new());

        var fusedCall = Assert.Single(
            Assert.IsType<PrimFunction>(module.Entry).Body.Fields.ToArray().OfType<Call>());
        var fused = Assert.IsType<Nncase.TIR.NTT.GatherReduceNormApply>(fusedCall.Target);
        Assert.False(fused.HasBias);
    }

    [Fact]
    public async Task TestKeepSharedBroadcastStatisticsMaterialization()
    {
        var (function, _, _) = CreateFunction(
            additionalStatsConsumer: true,
            addInterveningNop: false);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Contains(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.GatherReduceScatter);
        Assert.Contains(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.NormApply);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.GatherReduceNormApply);
    }

    [Fact]
    public async Task TestFuseAliasedStatisticsAcrossPreservedCodegenScope()
    {
        var (function, _, broadcastStats) = CreateFunction(
            additionalStatsConsumer: false,
            addInterveningNop: true,
            aliasNormStats: true,
            preserveProducerScope: true,
            preserveConsumerScope: true);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var fusedCall = Assert.Single(
            calls,
            call => call.Target is Nncase.TIR.NTT.GatherReduceNormApply);
        Assert.DoesNotContain(
            calls,
            call => call.Target is Nncase.TIR.NTT.GatherReduceScatter or Nncase.TIR.NTT.NormApply);
        Assert.DoesNotContain(
            fusedCall.Arguments.ToArray(),
            argument => ReferenceEquals(argument, broadcastStats));
        Assert.Contains(
            rewritten.Body.Fields.ToArray().OfType<Sequential>(),
            sequential => sequential.TraceScopeName == "decoder_layer_callee" &&
                sequential.PreserveCodegenBoundary);
        Assert.Contains(
            rewritten.Body.Fields.ToArray().OfType<Sequential>(),
            sequential => sequential.TraceScopeName == "next_decoder_layer_callee" &&
                sequential.PreserveCodegenBoundary);
    }

    [Fact]
    public async Task TestKeepAliasedStatisticsWithAdditionalConsumer()
    {
        var (function, _, _) = CreateFunction(
            additionalStatsConsumer: true,
            addInterveningNop: false,
            aliasNormStats: true,
            preserveProducerScope: true,
            preserveConsumerScope: true);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyPass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var calls = ExprCollector.Collect(rewritten).OfType<Call>().ToArray();
        Assert.Contains(calls, call => call.Target is Nncase.TIR.NTT.GatherReduceScatter);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.NTT.NormApply);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.GatherReduceNormApply);
    }

    [Fact]
    public async Task TestFuseChainedPreservedScopes()
    {
        var (baseFunction, partialStats, broadcastStats) = CreateFunction(
            additionalStatsConsumer: false,
            addInterveningNop: false,
            aliasNormStats: true,
            preserveProducerScope: true,
            preserveConsumerScope: true);
        var partialType = partialStats.DistributedType!;
        var broadcastType = broadcastStats.DistributedType!;
        var nextPartialStats = CreateBuffer(
            "next_partial_stats",
            DataTypes.Float32,
            448,
            [1, 1, 1],
            [0, 0, 0],
            partialType,
            compactPerOwner: true);
        var nextBroadcastStats = CreateBuffer(
            "next_broadcast_stats",
            DataTypes.Float32,
            960,
            [1, 1, 1],
            [0, 0, 0],
            broadcastType,
            compactPerOwner: true);
        var nextNormStats = CreateAliasedBuffer("next_norm_stats_alias", nextBroadcastStats);
        var nextInput = CreateBuffer("next_input", DataTypes.Float32, 1472, [1, 8], [8, 1]);
        var nextScale = CreateBuffer("next_scale", DataTypes.Float32, 1504, [8], [1]);
        var nextBias = CreateBuffer("next_bias", DataTypes.Float32, 1536, [8], [1]);
        var nextOutput = CreateBuffer("next_output", DataTypes.Float32, 1568, [1, 8], [8, 1]);

        var fields = baseFunction.Body.Fields.ToArray();
        var middleScope = Assert.IsType<Sequential>(fields[1]);
        fields[1] = middleScope.With(
            fields: middleScope.Fields.ToArray().Append(
                Nncase.TIR.F.NTT.GatherReduceScatter(
                    nextPartialStats,
                    nextBroadcastStats,
                    partialType,
                    broadcastType)).ToArray());
        var chainedFunction = new PrimFunction(
            "chained_decoder_layers",
            PyNTTTarget.Kind,
            new Sequential(
                fields.Append(
                    Nncase.TIR.F.NTT.NormApply(
                        nextInput,
                        nextNormStats,
                        nextScale,
                        nextBias,
                        nextOutput,
                        1,
                        1e-6f,
                        useMean: false)).ToArray()),
            Array.Empty<IVar>());
        var module = new IRModule(chainedFunction);

        await new FuseGatherReduceNormApplyPass(PyNTTTarget.Kind).RunAsync(module, new());

        var calls = ExprCollector.Collect(module.Entry!).OfType<Call>().ToArray();
        Assert.Equal(
            2,
            calls.Count(call => call.Target is Nncase.TIR.NTT.GatherReduceNormApply));
        Assert.DoesNotContain(
            calls,
            call => call.Target is Nncase.TIR.NTT.GatherReduceScatter or Nncase.TIR.NTT.NormApply);
    }

    private static (PrimFunction Function, Nncase.TIR.Buffer PartialStats, Nncase.TIR.Buffer BroadcastStats) CreateFunction(
        bool additionalStatsConsumer,
        bool addInterveningNop,
        bool zeroBias = false,
        bool aliasNormStats = false,
        bool preserveProducerScope = false,
        bool preserveConsumerScope = false)
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var statsTensorType = new TensorType(DataTypes.Float32, new[] { 1, 1, 1 });
        var partialType = new DistributedType(
            statsTensorType,
            [SBP.B, SBP.B, SBP.B],
            placement,
            SBP.P([0, 1], ReduceOp.Sum));
        var broadcastType = new DistributedType(
            statsTensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var partialStats = CreateBuffer(
            "partial_stats",
            DataTypes.Float32,
            0,
            [1, 1, 1],
            [0, 0, 0],
            partialType,
            compactPerOwner: true);
        var broadcastStats = CreateBuffer(
            "broadcast_stats",
            DataTypes.Float32,
            128,
            [1, 1, 1],
            [0, 0, 0],
            broadcastType,
            compactPerOwner: true);
        var input = CreateBuffer("input", DataTypes.Float32, 256, [1, 8], [8, 1]);
        var scale = CreateBuffer("scale", DataTypes.Float32, 288, [8], [1]);
        Nncase.TIR.Buffer bias;
        if (zeroBias)
        {
            T.AttachBuffer(new TensorConst(Tensor.From<float>(new float[8], [8])), out bias);
        }
        else
        {
            bias = CreateBuffer("bias", DataTypes.Float32, 320, [8], [1]);
        }

        var output = CreateBuffer("output", DataTypes.Float32, 352, [1, 8], [8, 1]);
        var normStats = aliasNormStats
            ? CreateAliasedBuffer("norm_stats_alias", broadcastStats)
            : broadcastStats;
        Expr gather = Nncase.TIR.F.NTT.GatherReduceScatter(partialStats, broadcastStats, partialType, broadcastType);
        if (preserveProducerScope)
        {
            gather = new Sequential(
                [T.Nop(), gather, T.Nop()],
                traceScopeName: "decoder_layer_callee",
                preserveCodegenBoundary: true);
        }

        var fields = new List<Expr> { gather };
        if (addInterveningNop)
        {
            fields.Add(T.Nop());
        }

        Expr normApply = Nncase.TIR.F.NTT.NormApply(input, normStats, scale, bias, output, 1, 1e-6f, useMean: false);
        if (preserveConsumerScope)
        {
            normApply = new Sequential(
                [T.Nop(), normApply, T.Nop()],
                traceScopeName: "next_decoder_layer_callee",
                preserveCodegenBoundary: true);
        }

        fields.Add(normApply);
        if (additionalStatsConsumer)
        {
            var copiedStats = CreateBuffer("copied_stats", DataTypes.Float32, 384, [1, 1, 1], [0, 0, 0]);
            var copiedStatsSource = aliasNormStats
                ? CreateAliasedBuffer("copied_stats_alias", broadcastStats)
                : broadcastStats;
            fields.Add(T.Memcopy(copiedStats, copiedStatsSource));
        }

        return (
            new PrimFunction(
                "decoder_layer",
                PyNTTTarget.Kind,
                new Sequential(fields.ToArray()),
                Array.Empty<IVar>()),
            partialStats,
            broadcastStats);
    }

    private static Nncase.TIR.Buffer CreateAliasedBuffer(
        string name,
        Nncase.TIR.Buffer source)
    {
        var sourcePhysical = source.MemSpan.Buffer;
        var aliasedPhysical = sourcePhysical.With(
            start: sourcePhysical.Start,
            size: sourcePhysical.Size);
        return new Nncase.TIR.Buffer(
            name,
            source.ElemType,
            source.MemSpan.With(buffer: aliasedPhysical),
            source.Dimensions.ToArray(),
            source.Strides.ToArray(),
            source.DistributedType);
    }

    private static Nncase.TIR.Buffer CreateBuffer(
        string name,
        DataType elementType,
        long startBytes,
        long[] shape,
        long[] strides,
        DistributedType? distributedType = null,
        bool compactPerOwner = false)
    {
        var logicalBytes = checked(shape.Aggregate(1L, (product, extent) => product * extent) * elementType.SizeInBytes);
        var ownerCount = compactPerOwner
            ? distributedType!.Placement.Hierarchy.Aggregate(1L, (product, extent) => product * extent)
            : 1L;
        var location = compactPerOwner ? MemoryLocation.ChipLocalData : MemoryLocation.Data;
        var physical = new PhysicalBuffer(
            elementType.SizeInBytes,
            startBytes,
            checked(logicalBytes * ownerCount),
            location);
        var storageKind = compactPerOwner
            ? DistributedBufferStorageKind.CompactPerOwner
            : DistributedBufferStorageKind.CompactLocal;
        return new Nncase.TIR.Buffer(
            name,
            elementType,
            new MemSpan(physical, 0, logicalBytes),
            shape.Select(value => (Dimension)value).ToArray(),
            strides.Select(value => (Dimension)value).ToArray(),
            distributedType,
            distributedStorageKind: storageKind);
    }
}
