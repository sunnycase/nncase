// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Evaluator.IR.NTT;

public sealed class SamplingPartialEvaluator :
    IEvaluator<SamplingPartial>,
    ITypeInferencer<SamplingPartial>,
    ICostEvaluator<SamplingPartial>
{
    public IValue Visit(IEvaluateContext context, SamplingPartial target)
    {
        var logits = context.GetArgumentValueAsTensor(target, SamplingPartial.Logits);
        var stateValue = context.GetArgumentValue(target, SamplingPartial.State);
        return Evaluate(logits, stateValue, target.Config);
    }

    internal static TupleValue Evaluate(Tensor logits, IValue stateValue, SamplerConfig config)
    {
        var state = stateValue.AsTensor().Cast<Reference<ISamplerState>>().Single().Value;
        SamplingEvaluator.ValidateState(state, config);
        var dimensions = logits.Dimensions.ToArray();
        var batch = checked((int)dimensions[0]);
        var vocab = checked((int)dimensions[1]);
        var processed = Tensor.From<float>(new float[batch * vocab], dimensions);
        var argMaxState = Tensor.From<ulong>(new ulong[batch], [batch]);

        for (var row = 0; row < batch; row++)
        {
            var raw = Enumerable.Range(0, vocab)
                .Select(token => Convert.ToSingle(logits[new long[] { row, token }]))
                .ToArray();
            var values = SamplingEvaluator.ApplyProcessors(raw, state, row, vocab);
            for (var token = 0; token < vocab; token++)
            {
                processed[new long[] { row, token }] = values[token];
            }

            var bestToken = Enumerable.Range(0, vocab)
                .OrderByDescending(token => values[token])
                .ThenBy(token => token)
                .First();
            argMaxState[new long[] { row }] = SamplingSplitTypeUtility.EncodeArgMax(
                values[bestToken],
                bestToken);
        }

        return new TupleValue([
            Value.FromTensor(processed),
            Value.FromTensor(argMaxState),
        ]);
    }

    public IRType Visit(ITypeInferenceContext context, SamplingPartial target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, SamplingPartial.Logits),
            context.CheckArgumentType<TensorType>(target, SamplingPartial.State));

    public static IRType InferType(
        SamplingPartial target,
        IRType logitsType,
        TensorType stateType)
    {
        var samplingType = SamplingEvaluator.InferType(target.Config, logitsType, stateType);
        if (samplingType is InvalidType)
        {
            return samplingType;
        }

        var logitsTensor = SamplingSplitTypeUtility.GetTensorType(logitsType)!;
        var batch = ((RankedShape)logitsTensor.Shape)[0];
        var processedTensor = new TensorType(DataTypes.Float32, logitsTensor.Shape);
        if (logitsType is TensorType)
        {
            return new TupleType([
                processedTensor,
                new TensorType(DataTypes.UInt64, new RankedShape(batch)),
            ]);
        }

        var distributed = (DistributedType)logitsType;
        if (!SamplingSplitTypeUtility.IsFullyVocabularySharded(distributed))
        {
            return new InvalidType(
                "SamplingPartial requires the vocabulary axis to cover every placement axis exactly once.");
        }

        var argMaxStateType = new TensorType(DataTypes.UInt64, new RankedShape(batch));
        var reductionAxes = Enumerable.Range(0, distributed.Placement.Rank).ToArray();
        return new TupleType([
            new DistributedType(
                processedTensor,
                distributed.AxisPolicies,
                distributed.Placement),
            new DistributedType(
                argMaxStateType,
                [SBP.B],
                distributed.Placement,
                SBP.P(reductionAxes, ReduceOp.Max)),
        ]);
    }

    public Cost Visit(ICostEvaluateContext context, SamplingPartial target)
    {
        var logitsType = context.GetArgumentType<IRType>(target, SamplingPartial.Logits);
        var localLogits = SamplingSplitTypeUtility.GetLocalTensorType(logitsType);
        var elements = (UInt128)TensorUtilities.GetProduct(CompilerServices.GetMaxShape(localLogits.Shape));
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = elements * (UInt128)localLogits.DType.SizeInBytes,
            [CostFactorNames.BlockLocalMemoryStoreBytes] = elements * sizeof(float) + sizeof(ulong),
            [CostFactorNames.CPUCycles] = elements * 8,
        };
    }
}

public sealed class SamplingCombineEvaluator :
    IEvaluator<SamplingCombine>,
    ITypeInferencer<SamplingCombine>,
    ICostEvaluator<SamplingCombine>
{
    public IValue Visit(IEvaluateContext context, SamplingCombine target)
        => SamplingEvaluator.Evaluate(
            context.GetArgumentValueAsTensor(target, SamplingCombine.Logits),
            context.GetArgumentValueAsTensor(target, SamplingCombine.ProcessedLogits),
            context.GetArgumentValue(target, SamplingCombine.State),
            target.Config);

    public IRType Visit(ITypeInferenceContext context, SamplingCombine target)
        => InferType(
            target,
            context.CheckArgumentType<IRType>(target, SamplingCombine.Logits),
            context.CheckArgumentType<IRType>(target, SamplingCombine.ProcessedLogits),
            context.CheckArgumentType<IRType>(target, SamplingCombine.ArgMaxState),
            context.CheckArgumentType<TensorType>(target, SamplingCombine.State));

    public static IRType InferType(
        SamplingCombine target,
        IRType logitsType,
        IRType processedType,
        IRType argMaxStateType,
        TensorType stateType)
    {
        var resultType = SamplingEvaluator.InferType(target.Config, logitsType, stateType);
        if (resultType is InvalidType)
        {
            return resultType;
        }

        var logitsTensor = SamplingSplitTypeUtility.GetTensorType(logitsType)!;
        var processedTensor = SamplingSplitTypeUtility.GetTensorType(processedType);
        if (processedTensor is null ||
            processedTensor.DType != DataTypes.Float32 ||
            processedTensor.Shape != logitsTensor.Shape)
        {
            return new InvalidType(
                "SamplingCombine requires FP32 processed logits with the raw logits shape.");
        }

        var batch = ((RankedShape)logitsTensor.Shape)[0];
        var expectedStateShape = new RankedShape(batch);
        if (!SamplingSplitTypeUtility.IsTensor(argMaxStateType, DataTypes.UInt64, expectedStateShape))
        {
            return new InvalidType(
                "SamplingCombine requires a compatible UInt64 partial argmax state.");
        }

        if (logitsType is DistributedType distributedLogits)
        {
            if (!SamplingSplitTypeUtility.IsFullyVocabularySharded(distributedLogits) ||
                processedType is not DistributedType distributedProcessed ||
                !distributedProcessed.AxisPolicies.SequenceEqual(distributedLogits.AxisPolicies) ||
                distributedProcessed.Placement != distributedLogits.Placement ||
                !SamplingSplitTypeUtility.IsArgMaxPartial(argMaxStateType, distributedLogits.Placement))
            {
                return new InvalidType(
                    "SamplingCombine requires matching vocabulary shards and a full-mesh P(Max) state.");
            }
        }
        else if (processedType is not TensorType || argMaxStateType is not TensorType)
        {
            return new InvalidType("SamplingCombine inputs must use one consistent placement.");
        }

        return resultType;
    }

    public Cost Visit(ICostEvaluateContext context, SamplingCombine target)
    {
        var processedType = context.GetArgumentType<IRType>(target, SamplingCombine.ProcessedLogits);
        var localProcessed = SamplingSplitTypeUtility.GetLocalTensorType(processedType);
        var elements = (UInt128)TensorUtilities.GetProduct(CompilerServices.GetMaxShape(localProcessed.Shape));
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = elements * sizeof(float),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = (UInt128)(target.Config.LogprobsCapacity * 8 + 12),
            [CostFactorNames.CPUCycles] = elements * 7,
        };
    }
}

internal static class SamplingSplitTypeUtility
{
    public static bool IsFullyVocabularySharded(DistributedType type)
        => type.Partial is null &&
           type.TensorType.Shape.Rank == 2 &&
           type.AxisPolicies[0] is SBPBroadCast &&
           type.AxisPolicies[1] is SBPSplit split &&
           split.HierarchyAxes.OrderBy(axis => axis)
               .SequenceEqual(Enumerable.Range(0, type.Placement.Rank));

    public static TensorType? GetTensorType(IRType type)
        => type switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null,
        };

    public static TensorType GetLocalTensorType(IRType type)
        => type switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => DistributedUtility.GetDividedTensorType(
                distributed,
                DistributedUtility.DivideFlags.MaxShape),
            _ => throw new InvalidOperationException($"Sampling cost requires a tensor, got {type}."),
        };

    public static bool IsTensor(IRType type, DataType dataType, RankedShape shape)
        => GetTensorType(type) is { DType: var actualDataType, Shape: var actualShape } &&
           actualDataType == dataType &&
           actualShape == shape;

    public static bool IsArgMaxPartial(IRType type, Placement placement)
        => type is DistributedType distributed &&
           distributed.Placement == placement &&
           distributed.AxisPolicies.Count == 1 &&
           distributed.AxisPolicies[0] is SBPBroadCast &&
           distributed.Partial is { Op: ReduceOp.Max } partial &&
           partial.Axes.OrderBy(axis => axis)
               .SequenceEqual(Enumerable.Range(0, placement.Rank));

    public static ulong EncodeArgMax(float value, int token)
    {
        var bits = BitConverter.SingleToUInt32Bits(value);
        var ordered = (bits & 0x8000_0000U) != 0 ? ~bits : bits ^ 0x8000_0000U;
        return ((ulong)ordered << 32) | (uint.MaxValue - checked((uint)token));
    }
}
