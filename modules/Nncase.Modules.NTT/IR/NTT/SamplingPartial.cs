// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Applies token-local sampling processors to one vocabulary shard and emits
/// the shard's processed logits and partial maximum state.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class SamplingPartial : Op
{
    public static readonly ParameterInfo Logits = new(
        typeof(SamplingPartial),
        0,
        "logits",
        ParameterKind.Input,
        MemoryEffect.Read);

    public static readonly ParameterInfo State = new(
        typeof(SamplingPartial),
        1,
        "state",
        ParameterKind.Attribute,
        MemoryEffect.ChipRead);

    public SamplerConfig Config { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"Config: {Config}";
}
