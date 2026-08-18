// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Combines vocabulary-shard sampling state, filters the global distribution,
/// samples tokens, and materializes optional logprobs.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class SamplingCombine : Op
{
    public static readonly ParameterInfo Logits = new(
        typeof(SamplingCombine),
        0,
        "logits",
        ParameterKind.Input,
        MemoryEffect.ChipRead);

    public static readonly ParameterInfo ProcessedLogits = new(
        typeof(SamplingCombine),
        1,
        "processed_logits",
        ParameterKind.Input,
        MemoryEffect.ChipRead);

    public static readonly ParameterInfo ArgMaxState = new(
        typeof(SamplingCombine),
        2,
        "argmax_state",
        ParameterKind.Input,
        MemoryEffect.ChipRead);

    public static readonly ParameterInfo State = new(
        typeof(SamplingCombine),
        3,
        "state",
        ParameterKind.Attribute,
        MemoryEffect.ChipReadWrite);

    public SamplerConfig Config { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"Config: {Config}";
}
