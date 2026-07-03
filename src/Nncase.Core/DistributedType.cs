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

    public static SBPSplit S(IRArray<int> axes, Dimension? granularity = null) => new SBPSplit(axes, granularity);
}

public sealed record SBPSplit(IRArray<int> Axes, Dimension? Granularity = null) : SBP
{
    public override string ToString() => $"S([{string.Join(",", Axes)}], {Granularity})";
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
            throw new JsonException();
        }

        string? typeDiscriminator = null;
        SBPSplit? sbpSplit = null;
        SBPPartial? sbpPartial = null;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                break;
            }

            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                string? propertyName = reader.GetString();
                reader.Read(); // Move to property value

                switch (propertyName)
                {
                    case "$type":
                        typeDiscriminator = reader.GetString();
                        break;
                    case "Axes":
                        int[] axes = JsonSerializer.Deserialize<int[]>(ref reader, options)!;
                        var irAxes = new IRArray<int>(axes);
                        if (typeDiscriminator == "S")
                        {
                            sbpSplit = new SBPSplit(irAxes);
                        }
                        else if (typeDiscriminator == "P")
                        {
                            sbpPartial = new SBPPartial(irAxes, ReduceOp.Sum);
                        }
                        else
                        {
                            throw new InvalidDataException("Axes must be used in SBP split");
                        }

                        break;
                    case "Op":
                        ReduceOp partialOp = JsonSerializer.Deserialize<ReduceOp>(ref reader, options);
                        sbpPartial = new SBPPartial(sbpPartial!.Axes, partialOp);
                        break;
                    default:
                        reader.Skip();
                        break;
                }
            }
        }

        switch (typeDiscriminator)
        {
            case "B":
                return SBP.B;
            case "P":
                return sbpPartial!;
            case "S":
                return sbpSplit!;
            default:
                throw new JsonException($"Unknown '$type' discriminator: {typeDiscriminator}");
        }
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
            writer.WriteString("Op", partialValue.Op.ToString());
        }
        else if (value is SBPSplit splitValue)
        {
            writer.WriteString("$type", "S");
            writer.WritePropertyName("Axes");
            JsonSerializer.Serialize(writer, splitValue.Axes.ToArray(), options);
        }
        else
        {
            throw new JsonException($"Unknown SBP type: {value.GetType()}");
        }

        writer.WriteEndObject();
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
