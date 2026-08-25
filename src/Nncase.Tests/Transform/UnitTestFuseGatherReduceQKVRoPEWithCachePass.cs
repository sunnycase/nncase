// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TransformTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFuseGatherReduceQKVRoPEWithCachePass : TestClassBase
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TestFuseSingleUsePartialQKVMaterializations(bool groupCollectives)
    {
        var (function, partials, _) = CreateFunction(
            additionalQConsumer: false,
            groupCollectives: groupCollectives);
        var module = new IRModule(function);

        await new FuseGatherReduceQKVRoPEWithCachePass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var fusedCall = Assert.Single(rewritten.Body.Fields.ToArray().OfType<Call>());
        var fused = Assert.IsType<Nncase.TIR.NTT.GatherReduceQKVRoPEWithCache>(fusedCall.Target);
        Assert.Equal(new[] { 0 }, fused.QInType.Partial!.Axes.ToArray());
        Assert.Equal(new long[] { 1, 1, 16 }, fused.QShape.Select(dimension => dimension.FixedValue).ToArray());
        Assert.Equal(new long[] { 0, 16, 1 }, fused.QStrides.Select(dimension => dimension.FixedValue).ToArray());
        Assert.Same(partials[0], fusedCall.Arguments[0]);
        Assert.Same(partials[1], fusedCall.Arguments[1]);
        Assert.Same(partials[2], fusedCall.Arguments[2]);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.GatherReduceScatter or Nncase.TIR.NTT.QKVRoPEWithCache);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TestKeepSharedQMaterialization(bool groupCollectives)
    {
        var (function, _, qCombined) = CreateFunction(
            additionalQConsumer: true,
            groupCollectives: groupCollectives);
        var module = new IRModule(function);

        await new FuseGatherReduceQKVRoPEWithCachePass(PyNTTTarget.Kind).RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        Assert.Contains(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.GatherReduceScatter);
        Assert.Contains(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.QKVRoPEWithCache);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Target is Nncase.TIR.NTT.GatherReduceQKVRoPEWithCache);
        Assert.Contains(
            ExprCollector.Collect(rewritten.Body).OfType<Call>(),
            call => call.Arguments.ToArray().Any(argument => ReferenceEquals(argument, qCombined)));
    }

    private static (PrimFunction Function, Nncase.TIR.Buffer[] Partials, Nncase.TIR.Buffer QCombined) CreateFunction(
        bool additionalQConsumer,
        bool groupCollectives = false)
    {
        var placement = new Placement([8, 16], "yx", "bb");
        var vectorType = new VectorType(DataTypes.BFloat16, [8]);
        var qPackedType = new TensorType(vectorType, [1, 256]);
        var kvPackedType = new TensorType(vectorType, [1, 128]);
        var qPartialType = new DistributedType(
            qPackedType,
            [SBP.B, SBP.SBlockCyclic([1], 8)],
            placement,
            SBP.P([0], ReduceOp.Sum));
        var qCombinedType = new DistributedType(
            qPackedType,
            [SBP.B, SBP.SBlockCyclic([1], 8)],
            placement);
        var kvPartialType = new DistributedType(
            kvPackedType,
            [SBP.B, SBP.SBlockCyclic([1], 8)],
            placement,
            SBP.P([0], ReduceOp.Sum));
        var kvCombinedType = new DistributedType(
            kvPackedType,
            [SBP.B, SBP.SBlockCyclic([1], 8)],
            placement);
        var qLogicalType = new DistributedType(
            new TensorType(vectorType, [1, 16, 16]),
            [SBP.B, SBP.SBlockCyclic([1], 1), SBP.B],
            placement);
        var kvLogicalType = new DistributedType(
            new TensorType(vectorType, [1, 8, 16]),
            [SBP.B, SBP.SBlockCyclic([0], 1), SBP.B],
            placement);

        var qPartial = CreateCompactPerOwnerBuffer("q_partial", vectorType, 0, [1, 16], [0, 1], qPartialType);
        var kPartial = CreateCompactPerOwnerBuffer("k_partial", vectorType, 32768, [1, 8], [0, 1], kvPartialType);
        var vPartial = CreateCompactPerOwnerBuffer("v_partial", vectorType, 49152, [1, 8], [0, 1], kvPartialType);
        var qCombined = CreateCanonicalBuffer("q_combined", vectorType, 65536, [1, 16], [0, 1], qCombinedType, 256);
        var kCombined = CreateCanonicalBuffer("k_combined", vectorType, 69632, [1, 8], [0, 1], kvCombinedType, 128);
        var vCombined = CreateCanonicalBuffer("v_combined", vectorType, 71680, [1, 8], [0, 1], kvCombinedType, 128);
        var qView = qCombined.With(
            name: "q_view",
            dimensions: [(Dimension)1, (Dimension)1, (Dimension)16],
            strides: [(Dimension)0, (Dimension)16, (Dimension)1],
            distributedType: qLogicalType);
        var kView = kCombined.With(
            name: "k_view",
            dimensions: [(Dimension)1, (Dimension)1, (Dimension)16],
            strides: [(Dimension)0, (Dimension)16, (Dimension)1],
            distributedType: kvLogicalType);
        var vView = vCombined.With(
            name: "v_view",
            dimensions: [(Dimension)1, (Dimension)1, (Dimension)16],
            strides: [(Dimension)0, (Dimension)16, (Dimension)1],
            distributedType: kvLogicalType);
        var parameterType = new DistributedType(
            new TensorType(vectorType, [16]),
            [SBP.B],
            placement);
        var qScale = CreateCanonicalBuffer("q_scale", vectorType, 73728, [16], [1], parameterType, 16);
        var kScale = CreateCanonicalBuffer("k_scale", vectorType, 73984, [16], [1], parameterType, 16);
        var bias = CreateCanonicalBuffer("bias", vectorType, 74240, [16], [1], parameterType, 16);
        var trigType = new DistributedType(
            new TensorType(new VectorType(DataTypes.Float32, [8]), [1, 1, 16]),
            [SBP.B, SBP.B, SBP.B],
            placement);
        var cos = CreateCanonicalBuffer("cos", new VectorType(DataTypes.Float32, [8]), 74496, [1, 1, 16], [0, 0, 1], trigType, 16);
        var sin = CreateCanonicalBuffer("sin", new VectorType(DataTypes.Float32, [8]), 75008, [1, 1, 16], [0, 0, 1], trigType, 16);
        var qOutput = CreateCanonicalBuffer("q_output", vectorType, 75520, [1, 1, 16], [0, 16, 1], qLogicalType, 256);

        var collectives = new Expr[]
        {
            Nncase.TIR.F.NTT.GatherReduceScatter(qPartial, qCombined, qPartialType, qCombinedType),
            Nncase.TIR.F.NTT.GatherReduceScatter(kPartial, kCombined, kvPartialType, kvCombinedType),
            Nncase.TIR.F.NTT.GatherReduceScatter(vPartial, vCombined, kvPartialType, kvCombinedType),
        };
        var qkvRoPE = Nncase.TIR.F.NTT.QKVRoPEWithCache(
            qView,
            kView,
            vView,
            qScale,
            kScale,
            bias,
            bias,
            cos,
            sin,
            None.Default,
            0,
            qOutput,
            2,
            1e-6f,
            false,
            2,
            1e-6f,
            false,
            [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim],
            [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim]);
        var fields = new List<Expr>();
        if (groupCollectives)
        {
            fields.Add(new Sequential(collectives));
        }
        else
        {
            fields.AddRange(collectives);
        }

        fields.Add(qkvRoPE);
        if (additionalQConsumer)
        {
            fields.Add(T.Memcopy(qCombined, qCombined));
        }

        return (
            new PrimFunction(
                "decoder_layer",
                PyNTTTarget.Kind,
                new Sequential(fields.ToArray()),
                Array.Empty<IVar>()),
            [qPartial, kPartial, vPartial],
            qCombined);
    }

    private static Nncase.TIR.Buffer CreateCompactPerOwnerBuffer(
        string name,
        DataType elementType,
        long startBytes,
        long[] shape,
        long[] strides,
        DistributedType distributedType)
    {
        var componentBytes = checked(shape.Aggregate(1L, (product, extent) => product * extent) * elementType.SizeInBytes);
        var ownerCount = distributedType.Placement.Hierarchy.Aggregate(1L, (product, extent) => product * extent);
        var physical = new PhysicalBuffer(
            elementType.SizeInBytes,
            startBytes,
            checked(componentBytes * ownerCount),
            MemoryLocation.ChipLocalData);
        return new Nncase.TIR.Buffer(
            name,
            elementType,
            new MemSpan(physical, 0, componentBytes),
            shape.Select(value => (Dimension)value).ToArray(),
            strides.Select(value => (Dimension)value).ToArray(),
            distributedType,
            distributedStorageKind: DistributedBufferStorageKind.CompactPerOwner);
    }

    private static Nncase.TIR.Buffer CreateCanonicalBuffer(
        string name,
        DataType elementType,
        long startBytes,
        long[] localShape,
        long[] strides,
        DistributedType distributedType,
        long globalPhysicalElements)
    {
        var physicalBytes = checked(globalPhysicalElements * elementType.SizeInBytes);
        var physical = new PhysicalBuffer(
            elementType.SizeInBytes,
            startBytes,
            physicalBytes,
            MemoryLocation.ChipLocalData);
        return new Nncase.TIR.Buffer(
            name,
            elementType,
            new MemSpan(physical, 0, physicalBytes),
            localShape.Select(value => (Dimension)value).ToArray(),
            strides.Select(value => (Dimension)value).ToArray(),
            distributedType,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
    }
}
