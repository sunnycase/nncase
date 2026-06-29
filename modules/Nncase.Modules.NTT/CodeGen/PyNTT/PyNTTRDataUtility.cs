// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.IR;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTRDataUtility
{
    public static int GetScopedShardCount(NTTTargetOptions targetOptions, string scopeName)
    {
        var hierarchies = targetOptions.Hierarchies.Length == 0 ? new[] { 1 } : targetOptions.Hierarchies[0];
        var scopeIndex = targetOptions.HierarchyNames.IndexOf(scopeName, StringComparison.Ordinal);
        if (scopeIndex < 0)
        {
            return checked((int)TensorUtilities.GetProduct(hierarchies));
        }

        return checked((int)TensorUtilities.GetProduct(hierarchies.Take(scopeIndex + 1).ToArray()));
    }

    public static int[] GetScopedShardIndex(int writerIndex, NTTTargetOptions targetOptions, string scopeName)
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

    public static long GetLocalRDataTableStrideBytes(IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas, NTTTargetOptions targetOptions, string scopeName)
    {
        var poolSize = GetPoolSizeBytes(localRdatas);
        if (poolSize == 0)
        {
            return 0;
        }

        var shardCount = GetScopedShardCount(targetOptions, scopeName);
        if (shardCount <= 1)
        {
            return poolSize;
        }

        var firstSignature = GetLocalRDataShardSignature(localRdatas, targetOptions, scopeName, 0);
        for (var shard = 1; shard < shardCount; shard++)
        {
            if (GetLocalRDataShardSignature(localRdatas, targetOptions, scopeName, shard) != firstSignature)
            {
                return poolSize;
            }
        }

        return 0;
    }

    public static string GetLocalRDataShardSignature(IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas, NTTTargetOptions targetOptions, string scopeName, int shard)
    {
        var builder = new StringBuilder();
        var shardIndex = GetScopedShardIndex(shard, targetOptions, scopeName);
        foreach (var (@const, range) in localRdatas)
        {
            var tensor = ((TensorConst)@const).Value;
            var distributedType = (DistributedType)@const.CheckedType;
            (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
            var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

            builder.Append(range.Min);
            builder.Append(':');
            builder.Append(range.Max);
            builder.Append(':');
            builder.Append(linearOffset);
            builder.Append(':');
            builder.AppendJoin(',', localShape);
            builder.Append(';');
        }

        return builder.ToString();
    }

    public static long GetPoolSizeBytes(IReadOnlyDictionary<Const, ValueRange<ulong>> ranges)
        => ranges.Count == 0 ? 0L : checked((long)ranges.Values.Max(range => range.Max));
}
