// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using DryIoc.ImTools;

namespace Nncase.IR;

[JsonConverter(typeof(SBPConverter))]
public abstract record SBP
{
    public static SBPBroadCast B => SBPBroadCast.Instance;

    public static SBPPartial P(IRArray<int> axes, ReduceOp op = ReduceOp.Sum) => new SBPPartial(axes, op);

    public static SBPSplit S(params SplitStage[] stages) => new(stages);

    public static SBPSplit SContiguous(IRArray<int> hierarchyAxes, Dimension? granularity = null)
        => S(SplitStage.Contiguous(hierarchyAxes, granularity));

    public static SBPSplit SBlockCyclic(IRArray<int> hierarchyAxes, long blockSize)
        => S(SplitStage.BlockCyclic(hierarchyAxes, blockSize));
}

public abstract record SplitDistribution;

public sealed record ContiguousSplit(Dimension? Granularity = null) : SplitDistribution
{
    public override string ToString() => Granularity is null ? "C" : $"C({Granularity})";
}

public sealed record BlockCyclicSplit : SplitDistribution
{
    public BlockCyclicSplit(long blockSize)
    {
        if (blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize, "Block-cyclic split block size must be positive.");
        }

        BlockSize = blockSize;
    }

    public long BlockSize { get; }

    public override string ToString() => $"BC({BlockSize})";
}

public sealed record SplitStage
{
    public SplitStage(IRArray<int> hierarchyAxes, SplitDistribution distribution)
    {
        if (hierarchyAxes.Count == 0)
        {
            throw new ArgumentException("A split stage must own at least one hierarchy axis.", nameof(hierarchyAxes));
        }

        if (hierarchyAxes.Any(axis => axis < 0))
        {
            throw new ArgumentOutOfRangeException(nameof(hierarchyAxes), "Split-stage hierarchy axes must be non-negative.");
        }

        if (hierarchyAxes.Distinct().Count() != hierarchyAxes.Count)
        {
            throw new ArgumentException("A split stage cannot contain duplicate hierarchy axes.", nameof(hierarchyAxes));
        }

        HierarchyAxes = hierarchyAxes;
        Distribution = distribution ?? throw new ArgumentNullException(nameof(distribution));
    }

    public IRArray<int> HierarchyAxes { get; }

    public SplitDistribution Distribution { get; }

    public static SplitStage Contiguous(IRArray<int> hierarchyAxes, Dimension? granularity = null)
        => new(hierarchyAxes, new ContiguousSplit(granularity));

    public static SplitStage BlockCyclic(IRArray<int> hierarchyAxes, long blockSize)
        => new(hierarchyAxes, new BlockCyclicSplit(blockSize));

    public override string ToString() => $"{Distribution}@[{string.Join(",", HierarchyAxes)}]";
}

public sealed record SBPSplit : SBP
{
    public SBPSplit(IRArray<SplitStage> stages)
    {
        if (stages.Count == 0)
        {
            throw new ArgumentException("A split policy must contain at least one stage.", nameof(stages));
        }

        var hierarchyAxes = stages.SelectMany(stage => stage.HierarchyAxes).ToArray();
        if (hierarchyAxes.Distinct().Count() != hierarchyAxes.Length)
        {
            throw new ArgumentException("A hierarchy axis can occur in only one split stage.", nameof(stages));
        }

        Stages = stages;
        HierarchyAxes = hierarchyAxes;
    }

    public IRArray<SplitStage> Stages { get; }

    public IRArray<int> HierarchyAxes { get; }

    public bool IsContiguous => Stages.All(stage => stage.Distribution is ContiguousSplit);

    public override string ToString() => $"S({string.Join(", ", Stages)})";
}

public sealed record SBPPartial(IRArray<int> Axes, ReduceOp Op) : SBP
{
    public override string ToString() => $"P([{string.Join(",", Axes)}], {Op})";
}

public sealed record SBPBroadCast : SBP
{
    public static readonly SBPBroadCast Instance = new SBPBroadCast();

    public override string ToString() => "B";
}

public class SBPConverter : JsonConverter<SBP>
{
    public override SBP Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException("An SBP policy must be a JSON object.");
        }

        using var document = JsonDocument.ParseValue(ref reader);
        var root = document.RootElement;
        if (!root.TryGetProperty("$type", out var typeProperty))
        {
            throw new JsonException("An SBP policy requires a '$type' discriminator.");
        }

        return typeProperty.GetString() switch
        {
            "B" => ReadBroadcast(root),
            "P" => ReadPartial(root, options),
            "S" => ReadSplit(root, options),
            var discriminator => throw new JsonException($"Unknown SBP '$type' discriminator '{discriminator}'."),
        };
    }

    public override void Write(Utf8JsonWriter writer, SBP value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();

        if (value is SBPBroadCast)
        {
            writer.WriteString("$type", "B");
        }
        else if (value is SBPPartial partialValue)
        {
            writer.WriteString("$type", "P");
            writer.WritePropertyName("Axes");
            JsonSerializer.Serialize(writer, partialValue.Axes.ToArray(), options);
            writer.WriteString("Op", partialValue.Op.ToString());
        }
        else if (value is SBPSplit splitValue)
        {
            writer.WriteString("$type", "S");
            writer.WritePropertyName("Stages");
            writer.WriteStartArray();
            foreach (var stage in splitValue.Stages)
            {
                WriteStage(writer, stage, options);
            }

            writer.WriteEndArray();
        }
        else
        {
            throw new JsonException($"Unknown SBP type: {value.GetType()}");
        }

        writer.WriteEndObject();
    }

    private static SBPPartial ReadPartial(JsonElement root, JsonSerializerOptions options)
    {
        ValidateProperties(root, "$type", "Axes", "Op");
        var axes = ReadRequiredArray<int>(root, "Axes", options);
        if (!root.TryGetProperty("Op", out var opProperty) ||
            !Enum.TryParse<ReduceOp>(opProperty.GetString(), out var op))
        {
            throw new JsonException("An SBP partial policy requires a valid 'Op'.");
        }

        return SBP.P(axes, op);
    }

    private static SBPSplit ReadSplit(JsonElement root, JsonSerializerOptions options)
    {
        ValidateProperties(root, "$type", "Stages");
        if (!root.TryGetProperty("Stages", out var stagesProperty) ||
            stagesProperty.ValueKind != JsonValueKind.Array)
        {
            throw new JsonException("An SBP split policy requires a 'Stages' array.");
        }

        var stages = stagesProperty.EnumerateArray().Select(stage => ReadStage(stage, options)).ToArray();
        return SBP.S(stages);
    }

    private static SplitStage ReadStage(JsonElement element, JsonSerializerOptions options)
    {
        ValidateProperties(element, "HierarchyAxes", "Distribution");
        var axes = ReadRequiredArray<int>(element, "HierarchyAxes", options);
        if (!element.TryGetProperty("Distribution", out var distribution) ||
            !distribution.TryGetProperty("$type", out var typeProperty))
        {
            throw new JsonException("A split stage requires a typed 'Distribution'.");
        }

        var distributionType = typeProperty.GetString();
        ValidateProperties(
            distribution,
            distributionType == "Contiguous"
                ? new[] { "$type", "Granularity" }
                : new[] { "$type", "BlockSize" });
        return distributionType switch
        {
            "Contiguous" => SplitStage.Contiguous(
                axes,
                distribution.TryGetProperty("Granularity", out var granularity)
                    ? ReadFixedDimension(granularity, "Granularity")
                    : null),
            "BlockCyclic" => SplitStage.BlockCyclic(
                axes,
                distribution.TryGetProperty("BlockSize", out var blockSize)
                    ? blockSize.GetInt64()
                    : throw new JsonException("A block-cyclic split distribution requires 'BlockSize'.")),
            var discriminator => throw new JsonException($"Unknown split distribution '$type' discriminator '{discriminator}'."),
        };
    }

    private static void WriteStage(Utf8JsonWriter writer, SplitStage stage, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WritePropertyName("HierarchyAxes");
        JsonSerializer.Serialize(writer, stage.HierarchyAxes.ToArray(), options);
        writer.WritePropertyName("Distribution");
        writer.WriteStartObject();
        switch (stage.Distribution)
        {
            case ContiguousSplit contiguous:
                writer.WriteString("$type", "Contiguous");
                if (contiguous.Granularity is { } granularity)
                {
                    if (!granularity.IsFixed)
                    {
                        throw new JsonException(
                            $"Dynamic contiguous split granularity '{granularity}' cannot be serialized in an SBP policy.");
                    }

                    writer.WriteNumber("Granularity", granularity.FixedValue);
                }

                break;
            case BlockCyclicSplit blockCyclic:
                writer.WriteString("$type", "BlockCyclic");
                writer.WriteNumber("BlockSize", blockCyclic.BlockSize);
                break;
            default:
                throw new JsonException($"Unknown split distribution type '{stage.Distribution.GetType().Name}'.");
        }

        writer.WriteEndObject();
        writer.WriteEndObject();
    }

    private static T[] ReadRequiredArray<T>(JsonElement root, string propertyName, JsonSerializerOptions options)
    {
        if (!root.TryGetProperty(propertyName, out var property) || property.ValueKind != JsonValueKind.Array)
        {
            throw new JsonException($"Expected an array property '{propertyName}'.");
        }

        return JsonSerializer.Deserialize<T[]>(property.GetRawText(), options)
            ?? throw new JsonException($"Could not deserialize '{propertyName}'.");
    }

    private static Dimension ReadFixedDimension(JsonElement property, string propertyName)
    {
        if (!property.TryGetInt64(out var value))
        {
            throw new JsonException($"Split distribution property '{propertyName}' must be a fixed integer.");
        }

        return value;
    }

    private static SBPBroadCast ReadBroadcast(JsonElement root)
    {
        ValidateProperties(root, "$type");
        return SBP.B;
    }

    private static void ValidateProperties(JsonElement element, params string[] allowedProperties)
    {
        var allowed = allowedProperties.ToHashSet(StringComparer.Ordinal);
        var unexpected = element.EnumerateObject()
            .Select(property => property.Name)
            .Where(property => !allowed.Contains(property))
            .ToArray();
        if (unexpected.Length != 0)
        {
            throw new JsonException(
                $"Unexpected SBP properties: {string.Join(", ", unexpected)}.");
        }
    }
}

public sealed record Placement(IRArray<int> Hierarchy, string Name, string HierarchyLevels)
{
    // public enum DeviceKind : uint
    // {
    //     CPU = 0,
    // }
    public int Rank => Hierarchy.Count;

    public string NormalizedHierarchyNames => NormalizeAxisString(Name);

    public string NormalizedHierarchyLevels => NormalizeHierarchyLevels(HierarchyLevels, Name, Rank);

    public bool IsPhysicalBlockAxis(int axis) => NormalizedHierarchyLevels[axis] == 'b';

    public int GetPhysicalLevelSize(char level)
    {
        var normalizedLevel = char.ToLowerInvariant(level);
        var levels = NormalizedHierarchyLevels;
        var size = 1;
        for (var i = 0; i < levels.Length; i++)
        {
            if (levels[i] == normalizedLevel)
            {
                size = checked(size * Hierarchy[i]);
            }
        }

        return size;
    }

    public int GetFirstPhysicalLevelAxis(char level)
    {
        var normalizedLevel = char.ToLowerInvariant(level);
        var levels = NormalizedHierarchyLevels;
        for (var i = 0; i < levels.Length; i++)
        {
            if (levels[i] == normalizedLevel)
            {
                return i;
            }
        }

        return -1;
    }

    public override string ToString() => $"[{string.Join(',', Hierarchy.Zip(Name).Select(t => t.Second.ToString() + ':' + t.First.ToString()))}]";

    public static string NormalizeAxisString(string? value)
        => string.Concat((value ?? string.Empty).Where(ch => char.IsLetterOrDigit(ch)));

    public static string NormalizeHierarchyLevels(string? levels, string names, int rank)
    {
        var normalizedLevels = NormalizeAxisString(levels);
        if (string.IsNullOrWhiteSpace(normalizedLevels))
        {
            if (rank == 0)
            {
                return string.Empty;
            }

            throw new InvalidOperationException("HierarchyLevels must be explicitly provided for non-empty placements.");
        }

        normalizedLevels = string.Concat(normalizedLevels.Select(char.ToLowerInvariant));
        if (normalizedLevels.Length != rank)
        {
            throw new InvalidOperationException($"HierarchyLevels '{levels}' must have {rank} axis entries.");
        }

        foreach (var level in normalizedLevels)
        {
            if (level is not ('c' or 'd' or 'b'))
            {
                throw new InvalidOperationException($"Unsupported hierarchy physical level '{level}'. Only 'c', 'd' and 'b' are supported.");
            }
        }

        return normalizedLevels;
    }
}

public sealed record DistributedType(TensorType TensorType, IRArray<SBP> AxisPolicies, Placement Placement, SBPPartial? Partial = null) : IRType
{
    public override string ToString() => $"{TensorType}, ({string.Join(',', AxisPolicies)}), {Placement}, Partial: {Partial}";
}
