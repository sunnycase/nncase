// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.PatternMatch;

namespace Nncase.IR.NN;

public enum SamplerLogprobsMode : byte
{
    RawLogprobs = 0,
    RawLogits = 1,
    ProcessedLogprobs = 2,
    ProcessedLogits = 3,
}

[Flags]
public enum SamplerProcessorFlags : uint
{
    None = 0,
    AllowedTokenMask = 1U << 0,
    ForbiddenTokenMask = 1U << 1,
    LogitBias = 1U << 2,
    RepetitionPenalty = 1U << 3,
    FrequencyPenalty = 1U << 4,
    PresencePenalty = 1U << 5,
}

/// <summary>
/// Canonical mutable sampler state. Runtime integrations own its storage and
/// keep every field address stable across prepared launches and graph replay.
/// </summary>
public interface ISamplerState
{
    SamplerConfig Config { get; }

    Tensor Active { get; }

    /// <summary>
    /// Gets the enabled dense logits processors for each batch row. Disabled
    /// processors must have no semantic effect and their backing tensors need
    /// not be read by a target kernel.
    /// </summary>
    Tensor ProcessorFlags { get; }

    Tensor Temperature { get; }

    Tensor TopP { get; }

    Tensor TopK { get; }

    Tensor MinP { get; }

    Tensor FrequencyPenalty { get; }

    Tensor PresencePenalty { get; }

    Tensor RepetitionPenalty { get; }

    /// <summary>
    /// Gets the requested top-logprob count per row. A value of -1 disables
    /// logprob materialization; zero requests only the sampled-token logprob.
    /// </summary>
    Tensor RequestedLogprobs { get; }

    Tensor Seeds { get; }

    Tensor Counters { get; }

    Tensor PromptTokenMask { get; }

    Tensor OutputTokenCounts { get; }

    Tensor AllowedTokenMask { get; }

    Tensor ForbiddenTokenMask { get; }

    Tensor LogitBias { get; }
}

public sealed record SamplerConfig
{
    public SamplerConfig(
        int vocabSize,
        int maxBatchSize,
        int maxLogprobs,
        SamplerLogprobsMode logprobsMode = SamplerLogprobsMode.RawLogprobs)
    {
        if (vocabSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(vocabSize), vocabSize, "Sampler vocabulary size must be positive.");
        }

        if (maxBatchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxBatchSize), maxBatchSize, "Sampler maximum batch size must be positive.");
        }

        if (maxLogprobs < 0 || maxLogprobs > vocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLogprobs), maxLogprobs, $"Sampler maximum logprobs must be in [0, {vocabSize}].");
        }

        if (!Enum.IsDefined(logprobsMode))
        {
            throw new ArgumentOutOfRangeException(nameof(logprobsMode), logprobsMode, "Unsupported sampler logprobs mode.");
        }

        VocabSize = vocabSize;
        MaxBatchSize = maxBatchSize;
        MaxLogprobs = maxLogprobs;
        LogprobsMode = logprobsMode;
    }

    public int VocabSize { get; }

    public int MaxBatchSize { get; }

    public int MaxLogprobs { get; }

    public SamplerLogprobsMode LogprobsMode { get; }

    public int LogprobsCapacity => checked(MaxLogprobs + 1);
}

public sealed record SamplerStateType : ValueType
{
    public SamplerConfig Config { get; init; } = null!;

    public override Type CLRType => typeof(ISamplerState);

    public override int SizeInBytes => IntPtr.Size;

    public override Guid Uuid { get; } = new("e591a74a-894b-4f21-a76b-d5ed9b6a6f37");

    public override string ToString() => "SamplerState";
}

[PatternFunctionalGenerator]
public sealed partial class Sampling : Op
{
    public static readonly ParameterInfo Logits = new(
        typeof(Sampling),
        0,
        "logits",
        ParameterKind.Input,
        MemoryEffect.ChipRead);

    public static readonly ParameterInfo State = new(
        typeof(Sampling),
        1,
        "state",
        ParameterKind.Attribute,
        MemoryEffect.ChipReadWrite);

    public SamplerConfig Config { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty()
        => $"VocabSize: {Config.VocabSize}, MaxBatchSize: {Config.MaxBatchSize}, " +
           $"MaxLogprobs: {Config.MaxLogprobs}, LogprobsMode: {Config.LogprobsMode}";
}
