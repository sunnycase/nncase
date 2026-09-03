// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR;

/// <summary>
/// Publishes one local tensor value through a runtime-owned heterogeneous channel.
/// </summary>
public sealed partial class ChannelProduce : Op
{
    public static readonly ParameterInfo Channel = new(
        typeof(ChannelProduce),
        0,
        "channel",
        IsTensor(),
        ParameterKind.Input,
        MemoryEffect.SystemReadWrite);

    public static readonly ParameterInfo Value = new(
        typeof(ChannelProduce),
        1,
        "value",
        IsIRType(),
        ParameterKind.Input,
        MemoryEffect.SystemRead);

    public string ChannelId { get; }

    public int Phase { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"{ChannelId}, Phase: {Phase}";
}

/// <summary>
/// Acquires one heterogeneous channel phase into caller-selected local storage.
/// </summary>
public sealed partial class ChannelConsume : Op
{
    public static readonly ParameterInfo Channel = new(
        typeof(ChannelConsume),
        0,
        "channel",
        IsTensor(),
        ParameterKind.Input,
        MemoryEffect.SystemReadWrite);

    public static readonly ParameterInfo Destination = new(
        typeof(ChannelConsume),
        1,
        "destination",
        IsIRType(),
        ParameterKind.Input,
        MemoryEffect.SystemWrite);

    public string ChannelId { get; }

    public int Phase { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"{ChannelId}, Phase: {Phase}";
}
