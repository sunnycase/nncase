// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.Evaluator.NN;

public sealed class SamplingEvaluator :
    IEvaluator<Sampling>,
    ITypeInferencer<Sampling>,
    ICostEvaluator<Sampling>
{
    private const float SamplingEpsilon = 1e-5F;

    public static IRType InferType(Sampling target, IRType logitsType, TensorType stateType)
        => InferType(target.Config, logitsType, stateType);

    public static IRType InferType(SamplerConfig config, IRType logitsType, TensorType stateType)
    {
        if (stateType is not
            {
                DType: ReferenceType { ElemType: SamplerStateType samplerStateType },
                Shape: RankedShape { Rank: 0 },
            })
        {
            return new InvalidType($"Sampler state must be a scalar Reference<SamplerStateType>, got {stateType}.");
        }

        if (samplerStateType.Config is null || samplerStateType.Config != config)
        {
            return new InvalidType("Sampler state config does not match the Sampling operation config.");
        }

        var logitsTensor = logitsType switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null,
        };
        if (logitsTensor?.Shape is not RankedShape { Rank: 2 } logitsShape)
        {
            return new InvalidType($"Sampling logits must be a ranked [batch, vocab] tensor, got {logitsType}.");
        }

        if (logitsTensor.DType is not PrimType { Attributes: var attributes } ||
            (attributes & PrimTypeAttributes.IsFloat) == 0)
        {
            return new InvalidType($"Sampling logits must use a floating-point scalar dtype, got {logitsTensor.DType}.");
        }

        var vocab = logitsShape[1].Simplify();
        if (vocab.IsFixed && vocab.FixedValue != config.VocabSize)
        {
            return new InvalidType($"Sampling logits vocabulary {vocab.FixedValue} does not match config {config.VocabSize}.");
        }

        var batch = logitsShape[0].Simplify();
        var maxBatch = CompilerServices.GetMaxShape(new RankedShape(batch))[0];
        if (maxBatch > config.MaxBatchSize)
        {
            return new InvalidType($"Sampling logits batch maximum {maxBatch} exceeds config {config.MaxBatchSize}.");
        }

        IRType[] resultTypes = CreateTensorResultTypes(batch, config);
        if (logitsType is DistributedType distributedLogits)
        {
            resultTypes = resultTypes
                .Cast<TensorType>()
                .Select(type => (IRType)new DistributedType(
                    type,
                    Enumerable.Repeat<SBP>(SBP.B, type.Shape.Rank).ToArray(),
                    distributedLogits.Placement))
                .ToArray();
        }

        return new TupleType(resultTypes.Append(stateType).ToArray());
    }

    public IRType Visit(ITypeInferenceContext context, Sampling target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, Sampling.Logits),
            context.CheckArgumentType<TensorType>(target, Sampling.State));

    public Cost Visit(ICostEvaluateContext context, Sampling target)
    {
        var logitsType = context.GetArgumentType<IRType>(target, Sampling.Logits);
        var localLogits = logitsType is DistributedType distributed
            ? DistributedUtility.GetDividedTensorType(distributed, DistributedUtility.DivideFlags.MaxShape)
            : (TensorType)logitsType;
        var elements = (UInt128)TensorUtilities.GetProduct(CompilerServices.GetMaxShape(localLogits.Shape));
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = elements * (UInt128)localLogits.DType.SizeInBytes,
            [CostFactorNames.BlockLocalMemoryStoreBytes] = elements * sizeof(float),
            [CostFactorNames.CPUCycles] = elements * 8,
        };
    }

    public IValue Visit(IEvaluateContext context, Sampling target)
    {
        var logits = context.GetArgumentValueAsTensor(target, Sampling.Logits);
        var stateValue = context.GetArgumentValue(target, Sampling.State);
        return Evaluate(logits, null, stateValue, target.Config);
    }

    public static IValue Evaluate(
        Tensor logits,
        Tensor? preparedLogits,
        IValue stateValue,
        SamplerConfig config)
    {
        var state = stateValue.AsTensor().Cast<Reference<ISamplerState>>().Single().Value;
        ValidateState(state, config);
        var shape = logits.Dimensions.ToArray();
        var batch = checked((int)shape[0]);
        var vocab = checked((int)shape[1]);
        if (preparedLogits is not null && !preparedLogits.Dimensions.SequenceEqual(logits.Dimensions))
        {
            throw new ArgumentException("Prepared sampling logits must match the raw logits shape.", nameof(preparedLogits));
        }

        var capacity = config.LogprobsCapacity;
        var sampledIds = Tensor.From<int>(new int[batch], [batch, 1]);
        var logprobIds = Tensor.From<int>(Enumerable.Repeat(-1, batch * capacity).ToArray(), [batch, capacity]);
        var logprobs = Tensor.From<float>(Enumerable.Repeat(float.NegativeInfinity, batch * capacity).ToArray(), [batch, capacity]);
        var ranks = Tensor.From<int>(new int[batch], [batch]);
        var counts = Tensor.From<int>(new int[batch], [batch]);

        for (var row = 0; row < batch; row++)
        {
            if (ReadScalar<byte>(state.Active, row) == 0)
            {
                continue;
            }

            var raw = new float[vocab];
            for (var token = 0; token < vocab; token++)
            {
                raw[token] = Convert.ToSingle(logits[new long[] { row, token }]);
            }

            var processed = preparedLogits is null
                ? ApplyProcessors(raw, state, row, vocab)
                : Enumerable.Range(0, vocab)
                    .Select(token => Convert.ToSingle(preparedLogits[new long[] { row, token }]))
                    .ToArray();
            var temperature = ReadScalar<float>(state.Temperature, row);
            var samplingLogits = processed.ToArray();
            int sampled;
            if (temperature < SamplingEpsilon)
            {
                sampled = ArgMax(samplingLogits);
            }
            else
            {
                for (var token = 0; token < vocab; token++)
                {
                    samplingLogits[token] /= temperature;
                }

                ApplyMinP(samplingLogits, ReadScalar<float>(state.MinP, row));
                ApplyTopK(samplingLogits, ReadScalar<int>(state.TopK, row));
                ApplyTopP(samplingLogits, ReadScalar<float>(state.TopP, row));
                var seed = ReadScalar<ulong>(state.Seeds, row);
                var counter = ReadScalar<ulong>(state.Counters, row);
                sampled = SampleCategorical(samplingLogits, seed, counter);
                state.Counters[new long[] { row }] = counter + 1;
            }

            sampledIds[new long[] { row, 0 }] = sampled;
            var oldCount = Convert.ToInt32(state.OutputTokenCounts[new long[] { row, sampled }]);
            state.OutputTokenCounts[new long[] { row, sampled }] = oldCount + 1;

            var requestedValue = ReadScalar<int>(state.RequestedLogprobs, row);
            if (requestedValue < 0)
            {
                continue;
            }

            var requested = System.Math.Clamp(requestedValue, 0, config.MaxLogprobs);
            counts[new long[] { row }] = requested + 1;
            var logValues = GetLogprobValues(config.LogprobsMode, raw, processed, samplingLogits);
            var normalized = config.LogprobsMode is SamplerLogprobsMode.RawLogits or SamplerLogprobsMode.ProcessedLogits
                ? logValues
                : LogSoftmax(logValues);
            var sorted = Enumerable.Range(0, vocab)
                .OrderByDescending(token => normalized[token])
                .ThenBy(token => token)
                .ToArray();
            logprobIds[new long[] { row, 0 }] = sampled;
            logprobs[new long[] { row, 0 }] = normalized[sampled];
            ranks[new long[] { row }] = normalized.Count(value => value >= normalized[sampled]);
            for (var index = 0; index < requested; index++)
            {
                var token = sorted[index];
                logprobIds[new long[] { row, index + 1 }] = token;
                logprobs[new long[] { row, index + 1 }] = normalized[token];
            }
        }

        return new TupleValue([
            Value.FromTensor(sampledIds),
            Value.FromTensor(logprobIds),
            Value.FromTensor(logprobs),
            Value.FromTensor(ranks),
            Value.FromTensor(counts),
            stateValue,
        ]);
    }

    public static TensorType[] CreateTensorResultTypes(Dimension batch, SamplerConfig config)
        =>
        [
            new(DataTypes.Int32, new RankedShape(batch, 1)),
            new(DataTypes.Int32, new RankedShape(batch, config.LogprobsCapacity)),
            new(DataTypes.Float32, new RankedShape(batch, config.LogprobsCapacity)),
            new(DataTypes.Int32, new RankedShape(batch)),
            new(DataTypes.Int32, new RankedShape(batch)),
        ];

    public static void ValidateState(ISamplerState state, SamplerConfig config)
    {
        if (state.Config != config)
        {
            throw new ArgumentException("Sampler runtime state config does not match the operation config.");
        }
    }

    public static float[] ApplyProcessors(float[] logits, ISamplerState state, int row, int vocab)
    {
        var result = logits.ToArray();
        var flags = (SamplerProcessorFlags)ReadScalar<uint>(state.ProcessorFlags, row);
        var frequency = flags.HasFlag(SamplerProcessorFlags.FrequencyPenalty)
            ? ReadScalar<float>(state.FrequencyPenalty, row)
            : 0F;
        var presence = flags.HasFlag(SamplerProcessorFlags.PresencePenalty)
            ? ReadScalar<float>(state.PresencePenalty, row)
            : 0F;
        var repetition = flags.HasFlag(SamplerProcessorFlags.RepetitionPenalty)
            ? ReadScalar<float>(state.RepetitionPenalty, row)
            : 1F;
        var readsOutputCounts = (flags &
            (SamplerProcessorFlags.RepetitionPenalty |
             SamplerProcessorFlags.FrequencyPenalty |
             SamplerProcessorFlags.PresencePenalty)) != 0;
        for (var token = 0; token < vocab; token++)
        {
            if ((flags.HasFlag(SamplerProcessorFlags.AllowedTokenMask) &&
                 Convert.ToByte(state.AllowedTokenMask[new long[] { row, token }]) == 0) ||
                (flags.HasFlag(SamplerProcessorFlags.ForbiddenTokenMask) &&
                 Convert.ToByte(state.ForbiddenTokenMask[new long[] { row, token }]) != 0))
            {
                result[token] = float.NegativeInfinity;
                continue;
            }

            if (flags.HasFlag(SamplerProcessorFlags.LogitBias))
            {
                result[token] += Convert.ToSingle(state.LogitBias[new long[] { row, token }]);
            }

            var outputCount = readsOutputCounts
                ? Convert.ToInt32(state.OutputTokenCounts[new long[] { row, token }])
                : 0;
            if (flags.HasFlag(SamplerProcessorFlags.RepetitionPenalty))
            {
                var repeated = outputCount != 0 || Convert.ToByte(state.PromptTokenMask[new long[] { row, token }]) != 0;
                if (repeated && repetition != 1F)
                {
                    result[token] = result[token] > 0F ? result[token] / repetition : result[token] * repetition;
                }
            }

            if (flags.HasFlag(SamplerProcessorFlags.FrequencyPenalty))
            {
                result[token] -= outputCount * frequency;
            }

            if (flags.HasFlag(SamplerProcessorFlags.PresencePenalty) && outputCount != 0)
            {
                result[token] -= presence;
            }
        }

        return result;
    }

    private static void ApplyMinP(float[] logits, float minP)
    {
        if (minP <= 0F)
        {
            return;
        }

        var threshold = logits.Max() + MathF.Log(minP);
        for (var index = 0; index < logits.Length; index++)
        {
            if (logits[index] < threshold)
            {
                logits[index] = float.NegativeInfinity;
            }
        }
    }

    private static void ApplyTopK(float[] logits, int topK)
    {
        if (topK <= 0 || topK >= logits.Length)
        {
            return;
        }

        var threshold = logits.OrderByDescending(value => value).ElementAt(topK - 1);
        for (var index = 0; index < logits.Length; index++)
        {
            if (logits[index] < threshold)
            {
                logits[index] = float.NegativeInfinity;
            }
        }
    }

    private static void ApplyTopP(float[] logits, float topP)
    {
        if (topP >= 1F)
        {
            return;
        }

        var sorted = Enumerable.Range(0, logits.Length)
            .Where(index => float.IsFinite(logits[index]))
            .OrderByDescending(index => logits[index])
            .ToArray();
        if (sorted.Length == 0)
        {
            throw new InvalidOperationException("Sampler filtering removed every token.");
        }

        var max = logits[sorted[0]];
        var weights = sorted.Select(index => MathF.Exp(logits[index] - max)).ToArray();
        var total = weights.Sum();
        var cumulative = 0F;
        for (var index = 0; index < sorted.Length; index++)
        {
            cumulative += weights[index] / total;
            if (cumulative >= topP)
            {
                for (var tail = index + 1; tail < sorted.Length; tail++)
                {
                    logits[sorted[tail]] = float.NegativeInfinity;
                }

                break;
            }
        }
    }

    private static int SampleCategorical(float[] logits, ulong seed, ulong counter)
    {
        var random = new System.Random(unchecked((int)(seed ^ (counter * 0x9E3779B97F4A7C15UL))));
        var bestToken = -1;
        var bestScore = float.NegativeInfinity;
        for (var token = 0; token < logits.Length; token++)
        {
            if (!float.IsFinite(logits[token]))
            {
                continue;
            }

            var uniform = System.Math.Clamp(random.NextDouble(), 1e-12, 1.0 - 1e-12);
            var score = logits[token] - MathF.Log(-MathF.Log((float)uniform));
            if (score > bestScore)
            {
                bestScore = score;
                bestToken = token;
            }
        }

        return bestToken >= 0 ? bestToken : throw new InvalidOperationException("Sampler filtering removed every token.");
    }

    private static int ArgMax(float[] values)
    {
        var index = 0;
        for (var candidate = 1; candidate < values.Length; candidate++)
        {
            if (values[candidate] > values[index])
            {
                index = candidate;
            }
        }

        return index;
    }

    private static float[] GetLogprobValues(
        SamplerLogprobsMode mode,
        float[] raw,
        float[] processed,
        float[] sampled)
        => mode switch
        {
            SamplerLogprobsMode.RawLogits or SamplerLogprobsMode.RawLogprobs => raw,
            SamplerLogprobsMode.ProcessedLogits or SamplerLogprobsMode.ProcessedLogprobs => sampled,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    private static float[] LogSoftmax(float[] values)
    {
        var max = values.Max();
        var sum = values.Where(float.IsFinite).Sum(value => MathF.Exp(value - max));
        var normalizer = max + MathF.Log(sum);
        return values.Select(value => value - normalizer).ToArray();
    }

    internal static T ReadScalar<T>(Tensor tensor, int index)
        => (T)Convert.ChangeType(tensor[new long[] { index }], typeof(T));
}
