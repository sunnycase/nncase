// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Lowers token-local sampling processors for one vocabulary shard.
/// </summary>
public sealed partial class SamplingPartial : NTTKernelOp
{
    public static readonly ParameterInfo Logits = new(typeof(SamplingPartial), 0, "logits", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo State = new(typeof(SamplingPartial), 1, "state", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo ProcessedLogits = new(typeof(SamplingPartial), 2, "processed_logits", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo ArgMaxState = new(typeof(SamplingPartial), 3, "argmax_state", memoryEffect: MemoryEffect.Write);

    public SamplerConfig Config { get; }

    public override string DisplayProperty() => $"Config: {Config}";
}
