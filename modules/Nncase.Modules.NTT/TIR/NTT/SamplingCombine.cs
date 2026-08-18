// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Lowers cross-shard sampling, filtering, and logprob materialization.
/// </summary>
public sealed partial class SamplingCombine : NTTKernelOp
{
    public static readonly ParameterInfo Logits = new(typeof(SamplingCombine), 0, "logits", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo ProcessedLogits = new(typeof(SamplingCombine), 1, "processed_logits", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo ArgMaxState = new(typeof(SamplingCombine), 2, "argmax_state", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo State = new(typeof(SamplingCombine), 3, "state", memoryEffect: MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo Summary = new(typeof(SamplingCombine), 4, "summary", memoryEffect: MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo SampledIds = new(typeof(SamplingCombine), 5, "sampled_ids", memoryEffect: MemoryEffect.ChipWrite);

    public static readonly ParameterInfo LogprobIds = new(typeof(SamplingCombine), 6, "logprob_ids", memoryEffect: MemoryEffect.ChipWrite);

    public static readonly ParameterInfo Logprobs = new(typeof(SamplingCombine), 7, "logprobs", memoryEffect: MemoryEffect.ChipWrite);

    public static readonly ParameterInfo Ranks = new(typeof(SamplingCombine), 8, "ranks", memoryEffect: MemoryEffect.ChipWrite);

    public static readonly ParameterInfo Counts = new(typeof(SamplingCombine), 9, "counts", memoryEffect: MemoryEffect.ChipWrite);

    public SamplerConfig Config { get; }

    public int BlockCount { get; }

    public int RadixBits { get; }

    public override string DisplayProperty()
        => $"Config: {Config}, BlockCount: {BlockCount}, RadixBits: {RadixBits}";
}
