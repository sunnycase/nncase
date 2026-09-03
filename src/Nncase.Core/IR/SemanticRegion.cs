// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Nncase.IR;

/// <summary>
/// Stable semantic identity of a model region. The kind is architecture
/// independent; the instance distinguishes repeated regions without affecting
/// placement policy.
/// </summary>
public sealed record SemanticRegion
{
    public SemanticRegion(string kind, string instance)
    {
        Kind = string.IsNullOrWhiteSpace(kind)
            ? throw new ArgumentException("Semantic region kind cannot be empty.", nameof(kind))
            : kind;
        Instance = string.IsNullOrWhiteSpace(instance)
            ? throw new ArgumentException("Semantic region instance cannot be empty.", nameof(instance))
            : instance;
    }

    public string Kind { get; }

    public string Instance { get; }
}

/// <summary>
/// Built-in semantic region kinds shared by importers and placement policies.
/// </summary>
public static class SemanticRegionKinds
{
    public const string Attention = "attention";

    public const string Embedding = "embedding";

    public const string PagedAttentionKVCache = "paged_attention_kv_cache";
}

/// <summary>
/// Creates and inspects explicit semantic-region boundaries. The marker target
/// is the region output and its attribute tuple contains the values entering
/// the region. Heterogeneous placement removes these markers after assigning
/// every enclosed operation.
/// </summary>
public static class SemanticRegionUtility
{
    /// <summary>
    /// Formats the stable diagnostic identity used by TIR trace scopes.
    /// </summary>
    public static string GetTraceScopeName(SemanticRegion region)
        => $"{region.Kind}:{region.Instance}";

    /// <summary>
    /// Checks whether expressions may be represented by one fused operation
    /// without crossing a semantic-region boundary.
    /// </summary>
    public static bool HaveUniformRegion(IEnumerable<BaseExpr> expressions)
    {
        var initialized = false;
        SemanticRegion? region = null;
        foreach (var expression in expressions)
        {
            if (!initialized)
            {
                region = expression.Metadata.SemanticRegion;
                initialized = true;
                continue;
            }

            if (expression.Metadata.SemanticRegion != region)
            {
                return false;
            }
        }

        return true;
    }

    public static Marker Mark(
        Expr output,
        IEnumerable<BaseExpr> inputs,
        SemanticRegion region)
    {
        var inputArray = inputs.ToArray();
        if (inputArray.Length == 0)
        {
            throw new ArgumentException("Semantic regions require at least one boundary input.", nameof(inputs));
        }

        var marker = new Marker(
            WellknownMarkerNames.SemanticRegion,
            output,
            new IR.Tuple(inputArray))
        {
            Metadata = new IRMetadata { SemanticRegion = region },
        };
        if (!CompilerServices.InferenceType(marker))
        {
            throw new InvalidOperationException(
                $"Cannot infer semantic-region marker {region.Kind}:{region.Instance} output type.");
        }

        return marker;
    }

    public static bool TryGet(Marker marker, out SemanticRegion region, out IReadOnlyList<BaseExpr> inputs)
    {
        if (marker.Name != WellknownMarkerNames.SemanticRegion)
        {
            region = null!;
            inputs = Array.Empty<BaseExpr>();
            return false;
        }

        region = marker.Metadata.SemanticRegion
            ?? throw new InvalidOperationException("SemanticRegion marker has no semantic region metadata.");
        if (marker.Attribute is not IR.Tuple inputTuple || inputTuple.Fields.Length == 0)
        {
            throw new InvalidOperationException(
                $"SemanticRegion marker {region.Kind}:{region.Instance} requires a non-empty input tuple attribute.");
        }

        inputs = inputTuple.Fields.ToArray();
        return true;
    }
}
