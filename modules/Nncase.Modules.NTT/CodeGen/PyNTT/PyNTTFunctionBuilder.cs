// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.CodeGen.PyNTT;

internal sealed class PyNTTFunctionBuilder
{
    private readonly uint _id;
    private readonly CompileOptions _compileOptions;

    public PyNTTFunctionBuilder(uint id, CompileOptions compileOptions)
    {
        _id = id;
        _compileOptions = compileOptions;
    }

    public PyNTTLinkableFunction Build(BaseFunction function)
    {
        var visitor = new PyNTTKernelSourceConvertVisitor(_compileOptions);
        visitor.Visit(function);
        var generatedKernelSource = visitor.GetKernelSource();

        return new PyNTTLinkableFunction(_id, function, generatedKernelSource, BuildRDataBundle(function));
    }

    private PyNTTRDataBundle BuildRDataBundle(BaseFunction function)
    {
        if (function is not TIR.PrimFunction primFunction)
        {
            return PyNTTRDataBundle.Empty;
        }

        var targetOptions = PyNTTTargetOptionsUtility.Normalize(_compileOptions);
        var rdata = SerializeRData(primFunction.SchedResult.Rdatas);
        var threadLocalRdatas = SerializeLocalRData(primFunction.SchedResult.ThreadLocalRdatas, targetOptions, "t");
        var warpLocalRdatas = SerializeLocalRData(primFunction.SchedResult.WarpLocalRdatas, targetOptions, "w");
        var blockLocalRdatas = SerializeLocalRData(primFunction.SchedResult.BlockLocalRdatas, targetOptions, "b");
        return new(
            rdata.Payload,
            rdata.Bytes,
            threadLocalRdatas.Payloads,
            threadLocalRdatas.Bytes,
            warpLocalRdatas.Payloads,
            warpLocalRdatas.Bytes,
            blockLocalRdatas.Payloads,
            blockLocalRdatas.Bytes);
    }

    private (string Payload, long Bytes) SerializeRData(IReadOnlyDictionary<Const, ValueRange<ulong>> rdatas)
    {
        var poolSize = PyNTTRDataUtility.GetPoolSizeBytes(rdatas);
        if (poolSize == 0)
        {
            return (string.Empty, 0);
        }

        using var payload = CreatePayloadStream(poolSize, "rdata");
        var stream = payload.Stream;
        stream.SetLength(checked((long)poolSize));
        foreach (var (@const, range) in rdatas)
        {
            var tensor = ((TensorConst)@const).Value;
            var size = range.Max - range.Min;
            if ((ulong)tensor.Length * (ulong)tensor.ElementType.SizeInBytes != size)
            {
                throw new InvalidDataException("The PyNTT rdata buffer size does not match the scheduled range.");
            }

            stream.Position = checked((long)range.Min);
            tensor.Serialize(stream);
        }

        return (FinalizePayload(payload), poolSize);
    }

    private (string[] Payloads, long Bytes) SerializeLocalRData(
        IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas,
        NTTTargetOptions targetOptions,
        string scopeName)
    {
        var poolSize = PyNTTRDataUtility.GetPoolSizeBytes(localRdatas);
        if (poolSize == 0)
        {
            return (Array.Empty<string>(), 0);
        }

        var shardCount = PyNTTRDataUtility.GetScopedShardCount(targetOptions, scopeName);
        var tableStride = PyNTTRDataUtility.GetLocalRDataTableStrideBytes(localRdatas, targetOptions, scopeName);
        var payloads = new string[tableStride == 0 ? 1 : shardCount];
        var payloadBySignature = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var shard = 0; shard < shardCount; shard++)
        {
            var signature = PyNTTRDataUtility.GetLocalRDataShardSignature(localRdatas, targetOptions, scopeName, shard);
            if (payloadBySignature.TryGetValue(signature, out var existingPayload))
            {
                if (shard < payloads.Length)
                {
                    payloads[shard] = existingPayload;
                }

                continue;
            }

            using var payload = CreatePayloadStream(poolSize, $"{scopeName}_{shard}");
            var stream = payload.Stream;
            stream.SetLength(checked((long)poolSize));
            foreach (var (@const, range) in localRdatas)
            {
                var tensor = ((TensorConst)@const).Value;
                var distributedType = (DistributedType)@const.CheckedType;
                var size = range.Max - range.Min;
                var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
                var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
                var shardIndex = PyNTTRDataUtility.GetScopedShardIndex(shard, targetOptions, scopeName);
                (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

                if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                {
                    throw new InvalidDataException("The PyNTT local rdata buffer size does not match the scheduled range.");
                }

                stream.Position = checked((long)range.Min);
                tensor.Serialize(stream, linearOffset, localShape, localStrides);
            }

            var finalizedPayload = FinalizePayload(payload);
            payloadBySignature[signature] = finalizedPayload;
            if (shard < payloads.Length)
            {
                payloads[shard] = finalizedPayload;
            }

            if (tableStride == 0)
            {
                break;
            }
        }

        return (payloads, poolSize);
    }

    private PayloadStream CreatePayloadStream(long poolSize, string label)
    {
        var directory = Path.Join(Path.GetTempPath(), "nncase_pyntt_rdata");
        Directory.CreateDirectory(directory);
        var path = Path.Join(directory, $"{_id}_{label}_{Guid.NewGuid():N}.bin");
        return new(new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None), path);
    }

    private static string FinalizePayload(PayloadStream payload)
    {
        payload.Stream.Flush();
        if (!string.IsNullOrEmpty(payload.Path))
        {
            return $"file:{payload.Path}";
        }

        throw new InvalidOperationException("PyNTT rdata payloads must be backed by binary files.");
    }

    private sealed record PayloadStream(Stream Stream, string? Path) : IDisposable
    {
        public void Dispose()
        {
            Stream.Dispose();
        }
    }
}
