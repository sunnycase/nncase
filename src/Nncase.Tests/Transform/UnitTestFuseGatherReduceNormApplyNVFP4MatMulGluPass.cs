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
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFuseGatherReduceNormApplyNVFP4MatMulGluPass : TestClassBase
{
    [Fact]
    public async Task TestFuseThroughBroadcastLogicalAlias()
    {
        var (function, rawInput, _) = CreateFunction(additionalNormalizedConsumer: false);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyNVFP4MatMulGluPass(PyNTTTarget.Kind)
            .RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var fusedCall = Assert.Single(
            calls.Where(call =>
                call.Target is Nncase.TIR.NTT.GatherReduceNormApplyNVFP4MatMulGlu));
        var fused = Assert.IsType<Nncase.TIR.NTT.GatherReduceNormApplyNVFP4MatMulGlu>(
            fusedCall.Target);
        Assert.Same(rawInput, fusedCall[Nncase.TIR.NTT.GatherReduceNormApplyNVFP4MatMulGlu.Input]);
        Assert.Equal(GluType.SwiGLU, fused.GluType);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.GatherReduceNormApply);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.NVFP4MatMulGlu);
    }

    [Fact]
    public async Task TestKeepMaterializationWithAdditionalConsumer()
    {
        var (function, _, normalizedOutput) = CreateFunction(additionalNormalizedConsumer: true);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyNVFP4MatMulGluPass(PyNTTTarget.Kind)
            .RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        Assert.Contains(calls, call => call.Target is Nncase.TIR.NTT.GatherReduceNormApply);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.NTT.NVFP4MatMulGlu);
        Assert.DoesNotContain(
            calls,
            call => call.Target is Nncase.TIR.NTT.GatherReduceNormApplyNVFP4MatMulGlu);
        Assert.Contains(
            calls,
            call => call.Target is Memcopy && call.Arguments.ToArray().Contains(normalizedOutput));
    }

    [Fact]
    public async Task TestCanonicalizesContiguousNormScaleBacking()
    {
        var (function, _, _) = CreateFunction(
            additionalNormalizedConsumer: false,
            compactContiguousNormScale: true);
        var module = new IRModule(function);

        await new FuseGatherReduceNormApplyNVFP4MatMulGluPass(PyNTTTarget.Kind)
            .RunAsync(module, new());

        var rewritten = Assert.IsType<PrimFunction>(module.Entry);
        var fusedCall = Assert.Single(
            ExprCollector.Collect(rewritten.Body)
                .OfType<Call>()
                .Where(call =>
                    call.Target is Nncase.TIR.NTT.GatherReduceNormApplyNVFP4MatMulGlu));
        var scale = Assert.IsType<Nncase.TIR.Buffer>(
            fusedCall[Nncase.TIR.NTT.GatherReduceNormApplyNVFP4MatMulGlu.NormScale]);
        Assert.Equal(DistributedBufferStorageKind.CanonicalGlobal, scale.DistributedStorageKind);
        Assert.Equal(0, scale.MemSpan.Start.FixedValue);
        Assert.Equal(10240, scale.MemSpan.Size.FixedValue);
        Assert.Equal(10240, scale.MemSpan.Buffer.Size.FixedValue);
        Assert.Equal(new long[] { 20 }, scale.Dimensions.ToArray().Select(dim => dim.FixedValue));
        Assert.Equal(new long[] { 1 }, scale.Strides.ToArray().Select(dim => dim.FixedValue));
    }

    private static (PrimFunction Function, Nncase.TIR.Buffer RawInput, Nncase.TIR.Buffer NormalizedOutput)
        CreateFunction(
            bool additionalNormalizedConsumer,
            bool compactContiguousNormScale = false)
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var activationTensorType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(1, 640));
        var shardedActivationType = new DistributedType(
            activationTensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 8)],
            placement);
        var broadcastActivationType = new DistributedType(
            activationTensorType,
            [SBP.B, SBP.B],
            placement);
        var statsTensorType = new TensorType(DataTypes.Float32, new RankedShape(1, 1, 1));
        var partialStatsType = new DistributedType(
            statsTensorType,
            [SBP.B, SBP.B, SBP.B],
            placement,
            SBP.P([0, 1], ReduceOp.Sum));
        var broadcastStatsType = new DistributedType(
            statsTensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var rawInput = CreateBuffer(
            "raw_input",
            activationTensorType.DType,
            MemoryLocation.Data,
            0,
            384,
            [1, 640],
            [0, 1],
            shardedActivationType,
            DistributedBufferStorageKind.CompactLocal);
        var partialStats = CreateBuffer(
            "partial_stats",
            DataTypes.Float32,
            MemoryLocation.ChipLocalData,
            0,
            4,
            [1, 1, 1],
            [0, 0, 0],
            partialStatsType,
            DistributedBufferStorageKind.CompactPerOwner);
        var normParameterType = new DistributedType(
            new TensorType(activationTensorType.DType, new RankedShape(640)),
            [compactContiguousNormScale
                ? SBP.SContiguous([0, 1], 20)
                : SBP.SBlockCyclic([0, 1], 8)],
            placement);
        var scale = compactContiguousNormScale
            ? CreateCompactContiguousNormParameter("scale", activationTensorType.DType, normParameterType)
            : CreateBuffer(
                "scale",
                activationTensorType.DType,
                MemoryLocation.ChipLocalRdata,
                0,
                10240,
                [640],
                [1],
                normParameterType,
                DistributedBufferStorageKind.CanonicalGlobal);
        var bias = CreateBuffer(
            "bias",
            activationTensorType.DType,
            MemoryLocation.ChipLocalRdata,
            10240,
            10240,
            [640],
            [1],
            scale.DistributedType,
            DistributedBufferStorageKind.CanonicalGlobal);
        var normalizedStorage = new PhysicalBuffer(
            16,
            0,
            10240,
            MemoryLocation.ChipLocalData);
        var normalizedOutput = new Nncase.TIR.Buffer(
            "normalized_output",
            activationTensorType.DType,
            new MemSpan(normalizedStorage, 0, 10240),
            [1, 24],
            [0, 1],
            shardedActivationType,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);
        var broadcastInput = new Nncase.TIR.Buffer(
            "broadcast_input",
            activationTensorType.DType,
            new MemSpan(normalizedStorage, 0, 10240),
            [1, 640],
            [0, 1],
            broadcastActivationType,
            distributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal);

        var gateWeight = CreateProjectionBuffer("gate_weight", new VectorType(DataTypes.UInt8, [2, 16]), [576, 80]);
        var upWeight = CreateProjectionBuffer("up_weight", new VectorType(DataTypes.UInt8, [2, 16]), [576, 80]);
        var gateScale = CreateProjectionBuffer("gate_scale", DataTypes.Float8E4M3, [576, 320]);
        var upScale = CreateProjectionBuffer("up_scale", DataTypes.Float8E4M3, [576, 320]);
        var gateInputGlobalScale = CreateProjectionBuffer("gate_input_global_scale", DataTypes.Float32, [1]);
        var upInputGlobalScale = CreateProjectionBuffer("up_input_global_scale", DataTypes.Float32, [1]);
        var gateWeightGlobalScale = CreateProjectionBuffer("gate_weight_global_scale", DataTypes.Float32, [1]);
        var upWeightGlobalScale = CreateProjectionBuffer("up_weight_global_scale", DataTypes.Float32, [1]);
        var output = CreateProjectionBuffer(
            "output",
            new VectorType(DataTypes.BFloat16, [8]),
            [1, 72]);

        var fields = new List<Expr>
        {
            Nncase.TIR.F.NTT.GatherReduceNormApply(
                partialStats,
                rawInput,
                scale,
                bias,
                normalizedOutput,
                partialStatsType,
                broadcastStatsType,
                1,
                1e-6f,
                useMean: false,
                hasBias: false),
            T.Nop(),
            Nncase.TIR.F.NTT.NVFP4MatMulGlu(
                broadcastInput,
                gateWeight,
                upWeight,
                gateScale,
                upScale,
                gateInputGlobalScale,
                upInputGlobalScale,
                gateWeightGlobalScale,
                upWeightGlobalScale,
                output,
                GluType.SwiGLU,
                16),
        };
        if (additionalNormalizedConsumer)
        {
            var copied = CreateProjectionBuffer(
                "copied_normalized",
                activationTensorType.DType,
                [1, 640]);
            fields.Add(T.Memcopy(copied, normalizedOutput));
        }

        var function = new PrimFunction(
            "decoder_layer",
            PyNTTTarget.Kind,
            new Sequential(fields.ToArray()),
            Array.Empty<IVar>());
        return (function, rawInput, normalizedOutput);
    }

    private static Nncase.TIR.Buffer CreateCompactContiguousNormParameter(
        string name,
        DataType elementType,
        DistributedType distributedType)
    {
        var shardCoord0 = new DimVar("__shard_coord_0");
        var shardCoord1 = new DimVar("__shard_coord_1");
        var localStart = ((2560 * shardCoord0) + (320 * shardCoord1)).Simplify();
        var physical = new PhysicalBuffer(
            elementType.SizeInBytes,
            0,
            10240,
            MemoryLocation.Input);
        return new Nncase.TIR.Buffer(
            name,
            elementType,
            new MemSpan(physical, localStart, 320),
            [20],
            [1],
            distributedType,
            distributedStorageKind: DistributedBufferStorageKind.CompactLocal);
    }

    private static Nncase.TIR.Buffer CreateProjectionBuffer(
        string name,
        DataType elementType,
        long[] shape)
    {
        var size = checked(
            shape.Aggregate(1L, (product, extent) => product * extent) *
            elementType.SizeInBytes);
        return CreateBuffer(
            name,
            elementType,
            MemoryLocation.Rdata,
            0,
            size,
            shape,
            GetDefaultStrides(shape),
            null,
            DistributedBufferStorageKind.CompactLocal);
    }

    private static long[] GetDefaultStrides(long[] shape)
    {
        var strides = new long[shape.Length];
        var stride = 1L;
        for (var axis = shape.Length - 1; axis >= 0; axis--)
        {
            strides[axis] = stride;
            stride = checked(stride * shape[axis]);
        }

        return strides;
    }

    private static Nncase.TIR.Buffer CreateBuffer(
        string name,
        DataType elementType,
        MemoryLocation location,
        long startBytes,
        long sizeBytes,
        long[] shape,
        long[] strides,
        DistributedType? distributedType,
        DistributedBufferStorageKind storageKind)
    {
        var physical = new PhysicalBuffer(
            Math.Max(1, elementType.SizeInBytes),
            startBytes,
            sizeBytes,
            location);
        return new Nncase.TIR.Buffer(
            name,
            elementType,
            new MemSpan(physical, 0, sizeBytes),
            shape.Select(value => (Dimension)value).ToArray(),
            strides.Select(value => (Dimension)value).ToArray(),
            distributedType,
            distributedStorageKind: storageKind);
    }
}
