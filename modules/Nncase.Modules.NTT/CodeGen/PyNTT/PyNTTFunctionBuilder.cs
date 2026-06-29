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

        var targetOptions = _compileOptions.TargetOptions as NTTTargetOptions ?? new NTTTargetOptions();
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
        var poolSize = GetPoolSize(rdatas);
        if (poolSize == 0)
        {
            return (string.Empty, 0);
        }

        using var stream = new MemoryStream();
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

        return (Convert.ToBase64String(stream.ToArray()), checked((long)poolSize));
    }

    private (string[] Payloads, long Bytes) SerializeLocalRData(
        IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas,
        NTTTargetOptions targetOptions,
        string scopeName)
    {
        var poolSize = GetPoolSize(localRdatas);
        if (poolSize == 0)
        {
            return (Array.Empty<string>(), 0);
        }

        var shardCount = GetScopedShardCount(targetOptions, scopeName);
        var payloads = new string[shardCount];
        for (var shard = 0; shard < shardCount; shard++)
        {
            using var stream = new MemoryStream();
            stream.SetLength(checked((long)poolSize));
            foreach (var (@const, range) in localRdatas)
            {
                var tensor = ((TensorConst)@const).Value;
                var distributedType = (DistributedType)@const.CheckedType;
                var size = range.Max - range.Min;
                var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
                var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
                var shardIndex = GetScopedShardIndex(shard, targetOptions, scopeName);
                (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

                if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                {
                    throw new InvalidDataException("The PyNTT local rdata buffer size does not match the scheduled range.");
                }

                stream.Position = checked((long)range.Min);
                tensor.Serialize(stream, linearOffset, localShape, localStrides);
            }

            payloads[shard] = Convert.ToBase64String(stream.ToArray());
        }

        return (payloads, checked((long)poolSize));
    }

    private ulong GetPoolSize(IReadOnlyDictionary<Const, ValueRange<ulong>> ranges)
    {
        return ranges.Count == 0 ? 0UL : ranges.Values.Max(range => range.Max);
    }

    private int GetScopedShardCount(NTTTargetOptions targetOptions, string scopeName)
    {
        var hierarchies = targetOptions.Hierarchies.Length == 0 ? new[] { 1 } : targetOptions.Hierarchies[0];
        var scopeIndex = targetOptions.HierarchyNames.IndexOf(scopeName, StringComparison.Ordinal);
        if (scopeIndex < 0)
        {
            return checked((int)TensorUtilities.GetProduct(hierarchies));
        }

        return checked((int)TensorUtilities.GetProduct(hierarchies.Take(scopeIndex + 1).ToArray()));
    }

    private int[] GetScopedShardIndex(int writerIndex, NTTTargetOptions targetOptions, string scopeName)
    {
        var hierarchies = targetOptions.Hierarchies.Length == 0 ? new[] { 1 } : targetOptions.Hierarchies[0];
        var scopeIndex = targetOptions.HierarchyNames.IndexOf(scopeName, StringComparison.Ordinal);
        if (scopeIndex < 0)
        {
            return DistributedUtility.GetUnraveledIndex(writerIndex, hierarchies);
        }

        var scopedHierarchies = hierarchies.Take(scopeIndex + 1).ToArray();
        return DistributedUtility.GetUnraveledIndex(writerIndex, scopedHierarchies)
            .Concat(Enumerable.Repeat(0, hierarchies.Length - scopedHierarchies.Length))
            .ToArray();
    }
}
