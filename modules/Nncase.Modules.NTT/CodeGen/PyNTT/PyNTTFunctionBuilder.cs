// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Shapes;
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

        var targetOptions = PyNTTTargetOptionsUtility.Get(_compileOptions);
        var blockLocalRdatas = SerializeLocalRData(
            primFunction.SchedResult.BlockLocalRdatas,
            primFunction.SchedResult.BlockLocalRDataMaterializations,
            targetOptions,
            "b");
        return new(
            string.Empty,
            0,
            string.Empty,
            0,
            blockLocalRdatas.Payloads,
            blockLocalRdatas.Bytes);
    }

    private (string[] Payloads, long Bytes) SerializeLocalRData(
        IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas,
        IReadOnlyDictionary<Const, TIR.BlockLocalRDataMaterialization> materializations,
        NTTTargetOptions targetOptions,
        string scopeName)
    {
        var poolSize = PyNTTRDataUtility.GetPoolSizeBytes(localRdatas);
        if (poolSize == 0)
        {
            return (Array.Empty<string>(), 0);
        }

        var shardCount = PyNTTRDataUtility.GetScopedShardCount(targetOptions, scopeName);
        var tableStride = PyNTTRDataUtility.GetLocalRDataTableStrideBytes(
            localRdatas,
            materializations,
            targetOptions,
            scopeName);
        var payloads = new string[tableStride == 0 ? 1 : shardCount];
        var payloadBySignature = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var shard = 0; shard < shardCount; shard++)
        {
            var signature = PyNTTRDataUtility.GetLocalRDataShardSignature(
                localRdatas,
                materializations,
                targetOptions,
                scopeName,
                shard);
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
            var shardIndex = PyNTTRDataUtility.GetScopedShardIndex(shard, targetOptions, scopeName);
            foreach (var (@const, range) in localRdatas)
            {
                if (materializations.TryGetValue(@const, out var materialization))
                {
                    SerializeMaterialization(stream, range, materialization, shardIndex);
                    continue;
                }

                var tensor = ((TensorConst)@const).Value;
                var distributedType = (DistributedType)@const.CheckedType;
                var size = range.Max - range.Min;
                var descriptor = DistributedUtility.GetLocalShardDescriptor(
                    distributedType,
                    shardIndex,
                    DistributedUtility.DivideFlags.MaxShape);
                var localCapacity = descriptor.LocalCapacityShape.ToValueArray();
                var localStrides = TensorUtilities.GetDefaultStrides(localCapacity);
                SerializeLocalTensorShard(stream, tensor, range, descriptor, localStrides);
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

    private void SerializeMaterialization(
        Stream stream,
        ValueRange<ulong> range,
        TIR.BlockLocalRDataMaterialization materialization,
        int[] shardIndex)
    {
        switch (materialization)
        {
            case TIR.ConcatenatedDistributedTensorRDataMaterialization concatenated:
                SerializeConcatenatedDistributedTensors(stream, range, concatenated, shardIndex);
                return;
            default:
                throw new NotSupportedException(
                    $"Unsupported block-local rdata materialization {materialization.GetType().Name}.");
        }
    }

    private void SerializeConcatenatedDistributedTensors(
        Stream stream,
        ValueRange<ulong> range,
        TIR.ConcatenatedDistributedTensorRDataMaterialization materialization,
        int[] shardIndex)
    {
        var localShape = materialization.LocalTensorType.Shape.ToValueArray();
        var localStrides = TensorUtilities.GetDefaultStrides(localShape);
        var elementSize = materialization.LocalTensorType.DType.SizeInBytes;
        var expectedBytes = checked(TensorUtilities.GetProduct(localShape) * elementSize);
        var allocationBytes = checked((long)(range.Max - range.Min));
        if (allocationBytes != expectedBytes)
        {
            throw new InvalidDataException(
                $"Derived block-local rdata allocation is {allocationBytes} bytes, expected {expectedBytes} for {materialization.LocalTensorType}.");
        }

        var payload = new byte[checked((int)allocationBytes)];
        long concatenatedAxisOffset = 0;
        foreach (var source in materialization.Sources)
        {
            var tensor = source.Tensor.Value;
            if (tensor.ElementType != materialization.LocalTensorType.DType)
            {
                throw new InvalidDataException(
                    $"Derived block-local rdata source dtype {tensor.ElementType} does not match destination {materialization.LocalTensorType.DType}.");
            }

            var descriptor = DistributedUtility.GetLocalShardDescriptor(
                source.DistributedType,
                shardIndex,
                DistributedUtility.DivideFlags.MaxShape);
            var sourceCapacity = descriptor.LocalCapacityShape.ToValueArray();
            var activeShape = descriptor.ActiveShape.ToValueArray();
            if (sourceCapacity.Length != localShape.Length ||
                sourceCapacity.Where((extent, axis) =>
                    axis != materialization.Axis && extent != localShape[axis]).Any())
            {
                throw new InvalidDataException(
                    "Derived block-local rdata concatenation sources must match every non-concatenated local capacity.");
            }

            var activeElements = TensorUtilities.GetProduct(activeShape);
            var localIndex = new long[activeShape.Length];
            var sourceIndex = new long[activeShape.Length];
            var destinationIndex = new long[activeShape.Length];
            for (long linear = 0; linear < activeElements; linear++)
            {
                TensorUtilities.UnravelIndex(linear, activeShape, localIndex);
                for (var axis = 0; axis < localIndex.Length; axis++)
                {
                    var globalCoordinate = descriptor.Axes[axis].MapLocalToGlobal(localIndex[axis]);
                    if (!globalCoordinate.IsFixed)
                    {
                        throw new InvalidDataException(
                            $"Derived block-local rdata source axis {axis} did not resolve to a fixed coordinate: {globalCoordinate}.");
                    }

                    sourceIndex[axis] = globalCoordinate.FixedValue;
                    destinationIndex[axis] = localIndex[axis] +
                        (axis == materialization.Axis ? concatenatedAxisOffset : 0);
                }

                var sourceElementOffset = TensorUtilities.GetLinearOffset(tensor.Strides, sourceIndex);
                var destinationElementOffset = TensorUtilities.GetLinearOffset(localStrides, destinationIndex);
                var sourceByteOffset = checked(sourceElementOffset * elementSize);
                var destinationByteOffset = checked((int)(destinationElementOffset * elementSize));
                if (sourceByteOffset < 0 || sourceByteOffset + elementSize > tensor.ByteLength)
                {
                    throw new InvalidDataException(
                        $"Derived block-local rdata source slice is out of range: " +
                        $"source={sourceByteOffset}, element_size={elementSize}, tensor_bytes={tensor.ByteLength}, " +
                        $"source_shape=[{string.Join(',', tensor.Dimensions.ToArray())}], " +
                        $"source_strides=[{string.Join(',', tensor.Strides.ToArray())}], " +
                        $"source_index=[{string.Join(',', sourceIndex)}], source_type={source.DistributedType}.");
                }

                if (destinationByteOffset < 0 || destinationByteOffset + elementSize > payload.Length)
                {
                    throw new InvalidDataException(
                        $"Derived block-local rdata destination slice is out of range: " +
                        $"destination={destinationByteOffset}, element_size={elementSize}, payload_bytes={payload.Length}, " +
                        $"destination_shape=[{string.Join(',', localShape)}], " +
                        $"destination_index=[{string.Join(',', destinationIndex)}].");
                }

                tensor.CopyBytesTo(sourceByteOffset, payload.AsSpan(destinationByteOffset, elementSize));
            }

            concatenatedAxisOffset = checked(concatenatedAxisOffset + sourceCapacity[materialization.Axis]);
        }

        if (concatenatedAxisOffset != localShape[materialization.Axis])
        {
            throw new InvalidDataException(
                $"Derived block-local rdata concatenation produced axis extent {concatenatedAxisOffset}, expected {localShape[materialization.Axis]}.");
        }

        stream.Position = checked((long)range.Min);
        stream.Write(payload);
    }

    private void SerializeLocalTensorShard(
        Stream stream,
        Tensor tensor,
        ValueRange<ulong> range,
        LocalShardDescriptor descriptor,
        long[] localStrides)
    {
        var size = checked((long)(range.Max - range.Min));
        var elementSize = tensor.ElementType.SizeInBytes;
        var payload = new byte[checked((int)size)];
        var localShape = descriptor.ActiveShape.ToValueArray();
        var localElementCount = TensorUtilities.GetProduct(localShape);
        var localIndex = new long[localShape.Length];
        var sourceIndex = new long[localShape.Length];
        for (long linear = 0; linear < localElementCount; linear++)
        {
            TensorUtilities.UnravelIndex(linear, localShape, localIndex);
            for (int axis = 0; axis < localIndex.Length; axis++)
            {
                var sourceCoordinate = descriptor.Axes[axis].MapLocalToGlobal(localIndex[axis]);
                if (!sourceCoordinate.IsFixed)
                {
                    throw new InvalidDataException(
                        $"PyNTT local rdata axis {axis} did not resolve to a fixed global coordinate: {sourceCoordinate}.");
                }

                sourceIndex[axis] = sourceCoordinate.FixedValue;
            }

            var sourceElementOffset = TensorUtilities.GetLinearOffset(tensor.Strides, sourceIndex);
            var destinationElementOffset = TensorUtilities.GetLinearOffset(localStrides, localIndex);
            var sourceByteOffset = checked(sourceElementOffset * elementSize);
            var destinationByteOffset = checked((int)(destinationElementOffset * elementSize));

            if (sourceByteOffset < 0 || sourceByteOffset + elementSize > tensor.ByteLength)
            {
                throw new InvalidDataException($"The PyNTT local rdata source slice is out of range: source={sourceByteOffset}, element_size={elementSize}, tensor_bytes={tensor.ByteLength}.");
            }

            if (destinationByteOffset < 0 || destinationByteOffset + elementSize > payload.Length)
            {
                throw new InvalidDataException($"The PyNTT local rdata destination slice is out of range: destination={destinationByteOffset}, element_size={elementSize}, payload_bytes={payload.Length}.");
            }

            tensor.CopyBytesTo(sourceByteOffset, payload.AsSpan(destinationByteOffset, elementSize));
        }

        stream.Position = checked((long)range.Min);
        stream.Write(payload);
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
