// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTRDataUtility
{
    public static int GetScopedShardCount(NTTTargetOptions targetOptions, string scopeName)
    {
        var hierarchies = targetOptions.Hierarchies.Length == 0 ? new[] { 1 } : targetOptions.Hierarchies[0];
        var scopeIndex = GetScopeIndex(targetOptions, scopeName, hierarchies.Length);
        if (scopeIndex < 0)
        {
            return checked((int)TensorUtilities.GetProduct(hierarchies));
        }

        return checked((int)TensorUtilities.GetProduct(hierarchies.Take(scopeIndex + 1).ToArray()));
    }

    public static int[] GetScopedShardIndex(int writerIndex, NTTTargetOptions targetOptions, string scopeName)
    {
        var hierarchies = targetOptions.Hierarchies.Length == 0 ? new[] { 1 } : targetOptions.Hierarchies[0];
        var scopeIndex = GetScopeIndex(targetOptions, scopeName, hierarchies.Length);
        if (scopeIndex < 0)
        {
            return DistributedUtility.GetUnraveledIndex(writerIndex, hierarchies);
        }

        var scopedHierarchies = hierarchies.Take(scopeIndex + 1).ToArray();
        return DistributedUtility.GetUnraveledIndex(writerIndex, scopedHierarchies)
            .Concat(Enumerable.Repeat(0, hierarchies.Length - scopedHierarchies.Length))
            .ToArray();
    }

    public static long GetLocalRDataTableStrideBytes(
        IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas,
        IReadOnlyDictionary<Const, TIR.BlockLocalRDataMaterialization> materializations,
        NTTTargetOptions targetOptions,
        string scopeName)
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

        var firstSignature = GetLocalRDataShardSignature(localRdatas, materializations, targetOptions, scopeName, 0);
        for (var shard = 1; shard < shardCount; shard++)
        {
            if (GetLocalRDataShardSignature(localRdatas, materializations, targetOptions, scopeName, shard) != firstSignature)
            {
                return poolSize;
            }
        }

        return 0;
    }

    public static string GetLocalRDataShardSignature(
        IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas,
        IReadOnlyDictionary<Const, TIR.BlockLocalRDataMaterialization> materializations,
        NTTTargetOptions targetOptions,
        string scopeName,
        int shard)
    {
        var builder = new StringBuilder();
        var shardIndex = GetScopedShardIndex(shard, targetOptions, scopeName);
        foreach (var (@const, range) in localRdatas)
        {
            if (materializations.TryGetValue(@const, out var materialization))
            {
                AppendMaterializationSignature(builder, materialization, range, shardIndex);
                continue;
            }

            var distributedType = (DistributedType)@const.CheckedType;
            var descriptor = DistributedUtility.GetLocalShardDescriptor(
                distributedType,
                shardIndex,
                DistributedUtility.DivideFlags.MaxShape);
            var localShape = descriptor.ActiveShape.ToValueArray();

            builder.Append(range.Min);
            builder.Append(':');
            builder.Append(range.Max);
            builder.Append(':');
            builder.AppendJoin(
                ',',
                distributedType.AxisPolicies
                    .OfType<SBPSplit>()
                    .SelectMany(split => split.HierarchyAxes)
                    .Distinct()
                    .OrderBy(axis => axis)
                    .Select(axis => shardIndex[axis]));
            builder.Append(':');
            builder.AppendJoin(',', distributedType.AxisPolicies.Select(policy => policy.ToString()));
            builder.Append(':');
            builder.AppendJoin(',', localShape);
            builder.Append(';');
        }

        return builder.ToString();
    }

    private static void AppendMaterializationSignature(
        StringBuilder builder,
        TIR.BlockLocalRDataMaterialization materialization,
        ValueRange<ulong> range,
        int[] shardIndex)
    {
        builder.Append(range.Min);
        builder.Append(':');
        builder.Append(range.Max);
        builder.Append(":derived:");
        switch (materialization)
        {
            case TIR.ConcatenatedDistributedTensorRDataMaterialization concatenated:
                builder.Append("concat@");
                builder.Append(concatenated.Axis);
                builder.Append(':');
                foreach (var source in concatenated.Sources)
                {
                    var descriptor = DistributedUtility.GetLocalShardDescriptor(
                        source.DistributedType,
                        shardIndex,
                        DistributedUtility.DivideFlags.MaxShape);
                    builder.AppendJoin(
                        ',',
                        source.DistributedType.AxisPolicies
                            .OfType<SBPSplit>()
                            .SelectMany(split => split.HierarchyAxes)
                            .Distinct()
                            .OrderBy(axis => axis)
                            .Select(axis => shardIndex[axis]));
                    builder.Append('/');
                    builder.AppendJoin(',', source.DistributedType.AxisPolicies.Select(policy => policy.ToString()));
                    builder.Append('/');
                    builder.AppendJoin(',', descriptor.ActiveShape.ToValueArray());
                    builder.Append('|');
                }

                break;
            default:
                throw new NotSupportedException(
                    $"Unsupported block-local rdata materialization {materialization.GetType().Name}.");
        }

        builder.Append(';');
    }

    public static long GetPoolSizeBytes(IReadOnlyDictionary<Const, ValueRange<ulong>> ranges)
        => ranges.Count == 0 ? 0L : checked((long)ranges.Values.Max(range => range.Max));

    private static int GetScopeIndex(NTTTargetOptions targetOptions, string scopeName, int rank)
    {
        if (scopeName.Length == 1 && scopeName[0] is 'c' or 'd' or 'b')
        {
            var levels = Placement.NormalizeHierarchyLevels(targetOptions.HierarchyLevels, targetOptions.HierarchyNames, rank);
            return levels.LastIndexOf(scopeName[0]);
        }

        return -1;
    }
}
