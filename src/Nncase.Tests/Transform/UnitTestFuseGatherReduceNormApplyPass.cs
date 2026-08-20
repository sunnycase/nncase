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

    private static (PrimFunction Function, Nncase.TIR.Buffer PartialStats, Nncase.TIR.Buffer BroadcastStats) CreateFunction(
        bool additionalStatsConsumer,
        bool addInterveningNop,
        bool zeroBias = false)
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
        var fields = new List<Expr>
        {
            Nncase.TIR.F.NTT.GatherReduceScatter(partialStats, broadcastStats, partialType, broadcastType),
        };
        if (addInterveningNop)
        {
            fields.Add(T.Nop());
        }

        fields.Add(Nncase.TIR.F.NTT.NormApply(input, broadcastStats, scale, bias, output, 1, 1e-6f, useMean: false));
        if (additionalStatsConsumer)
        {
            var copiedStats = CreateBuffer("copied_stats", DataTypes.Float32, 384, [1, 1, 1], [0, 0, 0]);
            fields.Add(T.Memcopy(copiedStats, broadcastStats));
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
