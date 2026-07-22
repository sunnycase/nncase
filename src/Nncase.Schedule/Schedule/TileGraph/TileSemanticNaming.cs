// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Globalization;
using System.Text;
using Nncase.TIR;

namespace Nncase.Schedule.TileGraph;

internal readonly record struct TileOperationName(int RegionOpId, string Kind)
{
    public string Symbol => $"op{RegionOpId.ToString(CultureInfo.InvariantCulture)}_{Kind}";

    public string TraceName => $"op{RegionOpId.ToString(CultureInfo.InvariantCulture)}:{Kind}";
}

internal sealed record TileFusionName(
    string Symbol,
    string TraceName,
    ImmutableArray<TileOperationName> Operations);

/// <summary>
/// Stable semantic names derived from immutable AutoTiling provenance.
/// </summary>
internal static class TileSemanticNaming
{
    private const int MaximumSymbolLength = 200;

    public static TileFusionName DescribeFusion(TileNode node)
    {
        var operations = new List<TileOperationName>();
        var visited = new HashSet<int>();
        CollectOperations(node, operations, visited);
        if (operations.Count == 0)
        {
            throw new InvalidOperationException($"Tile scope {node} contains no source operations.");
        }

        var traceName = (operations.Count == 1 ? "op" : "fusion") + "[" +
            string.Join(",", operations.Select(operation => operation.TraceName)) + "]";
        return new TileFusionName(
            ComposeSymbol(string.Empty, operations, string.Empty),
            traceName,
            operations.ToImmutableArray());
    }

    public static string GetBufferEndpointName(BufferIdentity buffer)
    {
        var operation = DescribeOperation(buffer.Node);
        var endpoint = buffer.IsOutput
            ? $"out{buffer.OutputIndex.ToString(CultureInfo.InvariantCulture)}"
            : $"in{buffer.Index.ToString(CultureInfo.InvariantCulture)}";
        return $"{operation.Symbol}_{endpoint}";
    }

    public static string GetLoopVariableName(
        TileNode scope,
        int axis,
        bool isReduction,
        TargetMachineModel targetMachine)
    {
        var fusion = DescribeFusion(scope);
        var level = GetMemoryLevelName(targetMachine, scope.Level);
        var axisKind = isReduction ? "reduce" : "spatial";
        return ComposeSymbol(
            "loop_",
            fusion.Operations,
            $"__{level}__{axisKind}_axis{axis.ToString(CultureInfo.InvariantCulture)}");
    }

    /// <summary>
    /// Builds a stable identity for one loop-owned pipeline schedule.
    /// </summary>
    public static PipelineRegionId GetPipelineRegionId(
        string owningScheduledFunctionId,
        TileNode scope,
        int loopEntry,
        int domainAxis)
    {
        if (loopEntry <= 0 || domainAxis < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(loopEntry),
                $"Pipeline loop identity requires a positive entry and nonnegative axis, got entry{loopEntry}/axis{domainAxis}.");
        }

        var fusion = DescribeFusion(scope);
        return new PipelineRegionId(
            owningScheduledFunctionId,
            ComposeSymbol(
                "pipeline_",
                fusion.Operations,
                $"__entry{loopEntry.ToString(CultureInfo.InvariantCulture)}__axis{domainAxis.ToString(CultureInfo.InvariantCulture)}"));
    }

    public static string GetTileVariableName(
        ITreeNode scope,
        int axis,
        TargetMachineModel targetMachine)
    {
        var fusion = DescribeScope(scope);
        var level = GetMemoryLevelName(targetMachine, scope.Level);
        return ComposeSymbol(
            "tile_",
            fusion.Operations,
            $"__{level}__axis{axis.ToString(CultureInfo.InvariantCulture)}");
    }

    public static string GetLocalExtentName(
        TileNode scope,
        int axis,
        TargetMachineModel targetMachine)
    {
        var fusion = DescribeFusion(scope);
        var level = GetMemoryLevelName(targetMachine, scope.Level);
        return ComposeSymbol(
            "extent_",
            fusion.Operations,
            $"__{level}__axis{axis.ToString(CultureInfo.InvariantCulture)}");
    }

    public static string GetStorageBufferName(
        BufferIdentity buffer,
        TileNode creationScope,
        int storageLevel,
        TargetMachineModel targetMachine)
    {
        var scope = DescribeFusion(creationScope);
        var level = GetMemoryLevelName(targetMachine, storageLevel);
        return ComposeSymbol(
            $"buffer_{GetBufferEndpointName(buffer)}__{level}__at_",
            scope.Operations,
            string.Empty);
    }

    public static string GetViewName(
        BufferIdentity buffer,
        TileNode scope,
        TargetMachineModel targetMachine,
        string role = "view")
    {
        var fusion = DescribeFusion(scope);
        var level = GetMemoryLevelName(targetMachine, scope.Level);
        return ComposeSymbol(
            $"{role}_{GetBufferEndpointName(buffer)}__{level}__at_",
            fusion.Operations,
            string.Empty);
    }

    public static string GetMemoryLevelName(TargetMachineModel targetMachine, int level)
    {
        if (level < 0)
        {
            return "root";
        }

        if (level >= targetMachine.TilingMemorySpaces.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(level),
                level,
                $"Target {targetMachine.Id} has {targetMachine.TilingMemorySpaces.Length} tiling memory levels.");
        }

        var memorySpace = targetMachine.TilingMemorySpaces[level];
        return $"l{level.ToString(CultureInfo.InvariantCulture)}_{SanitizeIdentifier(memorySpace.Id.Value)}";
    }

    private static TileOperationName DescribeOperation(TileGrid node)
        => new(node.RegionOpId, ToSnakeCase(node.Op.GetType().Name));

    private static TileFusionName DescribeScope(ITreeNode node)
        => node switch
        {
            TileNode scope => DescribeFusion(scope),
            OpNode operation => new TileFusionName(
                DescribeOperation(operation.Wrapped).Symbol,
                $"op[{DescribeOperation(operation.Wrapped).TraceName}]",
                ImmutableArray.Create(DescribeOperation(operation.Wrapped))),
            _ => throw new ArgumentOutOfRangeException(nameof(node), node, "Unsupported tile scope."),
        };

    private static void CollectOperations(
        ITreeNode node,
        ICollection<TileOperationName> operations,
        ISet<int> visited)
    {
        switch (node)
        {
            case OpNode operationNode when visited.Add(operationNode.Wrapped.RegionOpId):
                operations.Add(DescribeOperation(operationNode.Wrapped));
                break;
            case TileNode scope:
                foreach (var child in scope.Children)
                {
                    CollectOperations(child, operations, visited);
                }

                break;
        }
    }

    private static string ComposeSymbol(
        string prefix,
        IReadOnlyList<TileOperationName> operations,
        string suffix)
    {
        var operationSymbols = operations.Select(operation => operation.Symbol).ToArray();
        var operationSequence = BuildOperationSequence(operationSymbols);
        var fullSymbol = prefix + operationSequence + suffix;
        if (fullSymbol.Length <= MaximumSymbolLength)
        {
            return fullSymbol;
        }

        var operationBudget = MaximumSymbolLength - prefix.Length - suffix.Length;
        if (operationBudget <= 0)
        {
            throw new InvalidOperationException(
                $"Semantic symbol fixed fields exceed {MaximumSymbolLength} characters: {prefix}...{suffix}.");
        }

        var hash = StableHash(operationSequence).ToString("x8", CultureInfo.InvariantCulture);
        for (var headCount = Math.Min(3, operationSymbols.Length); headCount >= 1; headCount--)
        {
            for (var tailCount = Math.Min(2, operationSymbols.Length - headCount); tailCount >= 0; tailCount--)
            {
                var omitted = operationSymbols.Length - headCount - tailCount;
                var parts = operationSymbols.Take(headCount)
                    .Concat(new[] { $"plus{omitted.ToString(CultureInfo.InvariantCulture)}_h{hash}" })
                    .Concat(tailCount == 0 ? Array.Empty<string>() : operationSymbols.Skip(operationSymbols.Length - tailCount));
                var candidate = (operationSymbols.Length == 1 ? string.Empty : "fused_") + string.Join("__", parts);
                if (candidate.Length <= operationBudget)
                {
                    return prefix + candidate + suffix;
                }
            }
        }

        var fallback = $"fusion_h{hash}";
        if (fallback.Length > operationBudget)
        {
            throw new InvalidOperationException(
                $"Semantic symbol has only {operationBudget} characters available for fusion provenance.");
        }

        return prefix + fallback + suffix;
    }

    private static string BuildOperationSequence(IReadOnlyList<string> operationSymbols)
        => (operationSymbols.Count == 1 ? string.Empty : "fused_") + string.Join("__", operationSymbols);

    private static uint StableHash(string value)
    {
        const uint offsetBasis = 2166136261;
        const uint prime = 16777619;
        var hash = offsetBasis;
        foreach (var character in value)
        {
            hash ^= character;
            hash *= prime;
        }

        return hash;
    }

    private static string ToSnakeCase(string value)
    {
        var builder = new StringBuilder(value.Length + 8);
        for (var index = 0; index < value.Length; index++)
        {
            var character = value[index];
            if (!char.IsLetterOrDigit(character))
            {
                if (builder.Length > 0 && builder[^1] != '_')
                {
                    builder.Append('_');
                }

                continue;
            }

            if (char.IsUpper(character) && index > 0)
            {
                var previous = value[index - 1];
                var nextIsLower = index + 1 < value.Length && char.IsLower(value[index + 1]);
                if ((char.IsLower(previous) || char.IsDigit(previous) || nextIsLower) && builder[^1] != '_')
                {
                    builder.Append('_');
                }
            }

            builder.Append(char.ToLowerInvariant(character));
        }

        return SanitizeIdentifier(builder.ToString());
    }

    private static string SanitizeIdentifier(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (var character in value)
        {
            builder.Append(char.IsLetterOrDigit(character) ? char.ToLowerInvariant(character) : '_');
        }

        return builder.ToString().Trim('_');
    }
}
