// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Utilities;

/// <summary>
/// Canonical view of one physical hierarchy axis. This is a derived view of
/// tensor-axis SBP policies, not a second serialized sharding representation.
/// </summary>
public abstract record HierarchyAxisPolicy;

public sealed record HierarchyAxisBroadcast : HierarchyAxisPolicy
{
    public static readonly HierarchyAxisBroadcast Instance = new();

    private HierarchyAxisBroadcast()
    {
    }
}

/// <summary>
/// Identifies the tensor axis and exact split-stage position owned by one
/// hierarchy axis.
/// </summary>
public sealed record HierarchyAxisSplit(
    int TensorAxis,
    int StageIndex,
    int StageAxisIndex,
    SplitDistribution Distribution) : HierarchyAxisPolicy;

/// <summary>
/// Identifies tensor axes reduced across one hierarchy axis.
/// </summary>
public sealed record HierarchyAxisPartial(IRArray<int> TensorAxes, ReduceOp Op) : HierarchyAxisPolicy;
