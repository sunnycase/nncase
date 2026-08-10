// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Nncase.IR.NN;

/// <summary>
/// Utilities for mapping logical attention tensors between named axis layouts.
/// </summary>
public static class AttentionLayoutUtility
{
    private static readonly AttentionDimKind[] RequiredAxes =
    [
        AttentionDimKind.Seq,
        AttentionDimKind.Head,
        AttentionDimKind.Dim,
    ];

    public static bool IsValid(IRArray<AttentionDimKind> layout) =>
        layout.Count == RequiredAxes.Length && RequiredAxes.All(layout.Contains);

    public static int[] GetPermutation(
        IRArray<AttentionDimKind> inputLayout,
        IRArray<AttentionDimKind> outputLayout)
    {
        Validate(inputLayout, nameof(inputLayout));
        Validate(outputLayout, nameof(outputLayout));
        return outputLayout.Select(inputLayout.IndexOf).ToArray();
    }

    public static IRArray<AttentionDimKind> GetInputLayout(
        IRArray<AttentionDimKind> outputLayout,
        IReadOnlyList<int> permutation)
    {
        Validate(outputLayout, nameof(outputLayout));
        if (permutation.Count != outputLayout.Count ||
            permutation.Any(axis => axis < 0 || axis >= outputLayout.Count) ||
            permutation.Distinct().Count() != outputLayout.Count)
        {
            throw new ArgumentException("Attention layout permutation must contain every axis exactly once.", nameof(permutation));
        }

        var inputLayout = new AttentionDimKind[outputLayout.Count];
        for (int outputAxis = 0; outputAxis < outputLayout.Count; outputAxis++)
        {
            inputLayout[permutation[outputAxis]] = outputLayout[outputAxis];
        }

        return inputLayout;
    }

    public static (int[] Lanes, int[] Axes) GetVectorizeParams(
        IPagedAttentionConfig config,
        IRArray<AttentionDimKind> layout,
        AttentionCacheKind cacheKind)
    {
        Validate(layout, nameof(layout));
        var vectorizedAxes = config.GetVectorizedAxes(cacheKind);
        var cacheLanes = config.GetLanes(cacheKind);
        if (vectorizedAxes.Count != cacheLanes.Count)
        {
            throw new InvalidOperationException(
                $"Paged-attention {cacheKind} vectorized axes and lanes must have the same length.");
        }

        var lanes = new List<int>();
        var axes = new List<int>();
        for (int index = 0; index < vectorizedAxes.Count; index++)
        {
            var attentionAxis = vectorizedAxes[index] switch
            {
                PagedKVCacheDimKind.BlockSize => AttentionDimKind.Seq,
                PagedKVCacheDimKind.NumKVHeads => AttentionDimKind.Head,
                PagedKVCacheDimKind.HeadDim => AttentionDimKind.Dim,
                _ => throw new NotSupportedException(
                    $"Paged-attention tensor packing does not support cache axis {vectorizedAxes[index]}."),
            };
            lanes.Add(cacheLanes[index]);
            axes.Add(layout.IndexOf(attentionAxis));
        }

        return (lanes.ToArray(), axes.ToArray());
    }

    public static void Validate(IRArray<AttentionDimKind> layout, string paramName)
    {
        if (!IsValid(layout))
        {
            throw new ArgumentException(
                "An attention layout must contain Seq, Head, and Dim exactly once.",
                paramName);
        }
    }
}
