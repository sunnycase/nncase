// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public sealed class GatedDeltaNetProjectionEvaluator :
    IEvaluator<GatedDeltaNetProjection>,
    ITypeInferencer<GatedDeltaNetProjection>,
    ICostEvaluator<GatedDeltaNetProjection>
{
    public IRType Visit(ITypeInferenceContext context, GatedDeltaNetProjection target)
    {
        var arguments = target.Parameters
            .Select(parameter => context.CheckArgumentType<IRType>(target, parameter))
            .ToArray();
        return InferType(target, arguments);
    }

    public Cost Visit(ICostEvaluateContext context, GatedDeltaNetProjection target)
    {
        var input = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetProjection.Input));
        var qkvWeight = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetProjection.QKVWeight));
        var convWeight = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetProjection.ConvWeight));
        var output = context.GetReturnType<TupleType>();
        var qkvOutput = GatedDeltaNetStageUtility.GetLocalTensorType(output[0]);
        var scalarQkvOutput = GatedDeltaNetStageUtility.UnpackTensorAxis(
            qkvOutput,
            1);
        if (!GatedDeltaNetStageUtility.TryGetMaxShape(scalarQkvOutput, out var qkvShape))
        {
            return Cost.Zero;
        }

        var cost = Cost.Zero;
        GatedDeltaNetStageUtility.AddMatMulCost(context, input, qkvWeight, scalarQkvOutput, ref cost);
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.CPUCycles,
            checked((UInt128)qkvShape[0] * (UInt128)qkvShape[1] * (UInt128)((target.ConvKernelSize * 2) + 4)));
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryLoadBytes,
            checked(
                (UInt128)qkvShape[1] * (UInt128)(target.ConvKernelSize - 1) *
                (UInt128)GatedDeltaNetStageUtility.GetScalarDataType(input.DType).SizeInBytes +
                CostUtility.GetMemoryAccess(convWeight)));
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryStoreBytes,
            checked(
                (UInt128)qkvShape[1] * (UInt128)(target.ConvKernelSize - 1) *
                (UInt128)GatedDeltaNetStageUtility.GetScalarDataType(input.DType).SizeInBytes));
        return cost;
    }

    public IValue Visit(IEvaluateContext context, GatedDeltaNetProjection target)
    {
        var inputValue = context.GetArgumentValueAsTensor(target, GatedDeltaNetProjection.Input);
        var input = inputValue.ToOrtTensor();
        var stateValue = context.GetArgumentValue(target, GatedDeltaNetProjection.State);
        var state = stateValue.AsTensor().Cast<Reference<IGatedDeltaNetState>>().Single().Value;
        var layerId = checked((int)context.GetArgumentValue(target, GatedDeltaNetProjection.LayerId).AsTensor().ToScalar<long>());
        var convState = state.GetState(GatedDeltaNetStateKind.Convolution, layerId).ToOrtTensor();
        var qkvWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetProjection.QKVWeight).ToOrtTensor();
        var convWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetProjection.ConvWeight).ToOrtTensor();
        var outputs = new List<OrtKISharp.Tensor>(checked((int)input.Shape[0]));
        for (var token = 0L; token < input.Shape[0]; token++)
        {
            var hidden = OrtKI.Slice(input, new[] { token }, new[] { token + 1 }, new[] { 0L }, new[] { 1L });
            var projected = OrtKI.MatMul(hidden, qkvWeight);
            var current = OrtKI.Transpose(projected, new[] { 1L, 0L });
            var history = OrtKI.Concat(new[] { convState, current }, 1L);
            convState = OrtKI.Slice(
                history,
                new[] { 1L },
                new[] { target.ConvKernelSize },
                new[] { 1L },
                new[] { 1L });
            var convolved = OrtKI.ReduceSum(
                OrtKI.Mul(history, convWeight),
                GatedDeltaNetStageUtility.Shape(1L),
                0L,
                0L);
            convolved = OrtKI.Mul(convolved, OrtKI.Sigmoid(convolved));
            outputs.Add(OrtKI.Unsqueeze(convolved, new[] { 0L }));
        }

        state.UpdateState(
            GatedDeltaNetStateKind.Convolution,
            layerId,
            convState.ToTensor().CastElementTo(inputValue.ElementType));
        var output = OrtKI.Concat(outputs.ToArray(), 0L);
        var outputLanes = GatedDeltaNetStageUtility.GetPackedLanes(
            state.Config,
            GatedDeltaNetStateKind.Convolution,
            GatedDeltaNetStateDimKind.ConvChannels);
        var outputValue = GatedDeltaNetStageUtility.PackValue(
            output,
            inputValue.ElementType,
            outputLanes,
            1);
        return new TupleValue([
            outputValue,
            stateValue,
        ]);
    }

    public static IRType InferType(GatedDeltaNetProjection target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Count != target.Parameters.Count)
        {
            return new InvalidType($"GatedDeltaNetProjection expects {target.Parameters.Count} inputs, got {arguments.Count}.");
        }

        if (arguments.OfType<InvalidType>().FirstOrDefault() is { } invalid)
        {
            return invalid;
        }

        if (arguments.Any(type => type is AnyType))
        {
            return AnyType.Default;
        }

        if (GatedDeltaNetStageUtility.AreTensorArguments(arguments, GatedDeltaNetProjection.LayerId.Index))
        {
            return InferTensorType(target, arguments);
        }

        if (!GatedDeltaNetStageUtility.TryGetDistributedArguments(
                arguments,
                GatedDeltaNetProjection.LayerId.Index,
                GatedDeltaNetProjection.State.Index,
                out var distributed,
                out var placement))
        {
            return new InvalidType(
                "GatedDeltaNetProjection inputs must be either all tensors or compatible distributed tensors.");
        }

        var localArguments = GatedDeltaNetStageUtility.GetTensorArguments(arguments, distributed);
        var tensorResult = InferTensorType(target, localArguments);
        if (tensorResult is not TupleType tuple)
        {
            return tensorResult;
        }

        var channel = distributed[GatedDeltaNetProjection.QKVWeight.Index].AxisPolicies[1];
        if (!GatedDeltaNetStageUtility.TryGetContiguousAxes(channel, placement.Rank, out var channelAxes))
        {
            return new InvalidType(
                "GatedDeltaNetProjection QKV-weight output channels must be broadcast or contiguously split.");
        }

        if (!GatedDeltaNetStageUtility.CoversPlacement(channelAxes, placement.Rank))
        {
            return new InvalidType(
                "GatedDeltaNetProjection channel split must cover the block placement so each state channel has one writer.");
        }

        var expected = new Dictionary<int, DistributedType>
        {
            [GatedDeltaNetProjection.Input.Index] = GatedDeltaNetStageUtility.CreateBroadcast(distributed[GatedDeltaNetProjection.Input.Index].TensorType, placement),
            [GatedDeltaNetProjection.QKVWeight.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetProjection.QKVWeight.Index].TensorType, [SBP.B, channel], placement),
            [GatedDeltaNetProjection.ConvWeight.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetProjection.ConvWeight.Index].TensorType, [channel, SBP.B], placement),
        };
        foreach (var (index, expectedType) in expected)
        {
            if (distributed[index] != expectedType)
            {
                return new InvalidType(
                    $"GatedDeltaNetProjection input {target.Parameters[index].Name} has distributed type " +
                    $"{distributed[index]}; expected {expectedType}.");
            }
        }

        return new TupleType([
            GatedDeltaNetStageUtility.Create((TensorType)tuple[0], [SBP.B, channel], placement),
            tuple[1],
        ]);
    }

    public static IRType InferTensorType(
        GatedDeltaNetProjection target,
        IReadOnlyList<IRType> arguments)
    {
        if (target.ConvKernelSize < 2)
        {
            return new InvalidType("GatedDeltaNetProjection convolution kernel size must be at least two.");
        }

        if (arguments[GatedDeltaNetProjection.Input.Index] is not TensorType input ||
            arguments[GatedDeltaNetProjection.State.Index] is not TensorType state ||
            arguments[GatedDeltaNetProjection.QKVWeight.Index] is not TensorType qkvWeight ||
            arguments[GatedDeltaNetProjection.ConvWeight.Index] is not TensorType convWeight ||
            arguments[GatedDeltaNetProjection.LayerId.Index] is not DimensionType)
        {
            return new InvalidType("GatedDeltaNetProjection expects tensor operands and a dimension-valued layer id.");
        }

        if (!GatedDeltaNetEvaluator.TryGetStateConfig(state, out var stateConfig, out var stateError))
        {
            return new InvalidType(stateError);
        }

        if (input.Shape is not RankedShape { Rank: 2 } inputShape ||
            qkvWeight.Shape is not RankedShape { Rank: 2 } qkvWeightShape ||
            convWeight.Shape is not RankedShape { Rank: 2 } convWeightShape)
        {
            return new InvalidType("GatedDeltaNetProjection expects rank-2 tensor inputs.");
        }

        if (input.DType != stateConfig.ActivationPrimType ||
            qkvWeight.DType != input.DType ||
            convWeight.DType != input.DType)
        {
            return new InvalidType(
                $"GatedDeltaNetProjection inputs must use scalar activation dtype {stateConfig.ActivationPrimType}.");
        }

        if (!GatedDeltaNetStageUtility.AreCompatible(inputShape[1], qkvWeightShape[0]) ||
            !GatedDeltaNetStageUtility.AreCompatible(qkvWeightShape[1], convWeightShape[0]) ||
            !GatedDeltaNetStageUtility.IsFixedValue(convWeightShape[1], target.ConvKernelSize) ||
            !GatedDeltaNetStageUtility.IsFixedValue(
                qkvWeightShape[1],
                stateConfig.GetDimension(GatedDeltaNetStateDimKind.ConvChannels)) ||
            stateConfig.ConvKernelSize != target.ConvKernelSize ||
            !GatedDeltaNetStageUtility.IsFixedValue(inputShape[1], stateConfig.HiddenSize))
        {
            return new InvalidType(
                "GatedDeltaNetProjection input shapes do not satisfy the projection/convolution contract.");
        }

        var scalarOutput = new TensorType(
            input.DType,
            new RankedShape(inputShape[0], qkvWeightShape[1]));
        var outputLanes = GatedDeltaNetStageUtility.GetPackedLanes(
            stateConfig,
            GatedDeltaNetStateKind.Convolution,
            GatedDeltaNetStateDimKind.ConvChannels);
        var packedOutput = GatedDeltaNetStageUtility.PackTensorAxis(
            scalarOutput,
            outputLanes,
            1);
        return packedOutput is InvalidType
            ? packedOutput
            : new TupleType([packedOutput, state]);
    }
}

public sealed class GatedDeltaNetRecurrentCoreEvaluator :
    IEvaluator<GatedDeltaNetRecurrentCore>,
    ITypeInferencer<GatedDeltaNetRecurrentCore>,
    ICostEvaluator<GatedDeltaNetRecurrentCore>
{
    public IRType Visit(ITypeInferenceContext context, GatedDeltaNetRecurrentCore target)
    {
        var arguments = target.Parameters
            .Select(parameter => context.CheckArgumentType<IRType>(target, parameter))
            .ToArray();
        return InferType(target, arguments);
    }

    public Cost Visit(ICostEvaluateContext context, GatedDeltaNetRecurrentCore target)
    {
        var input = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.Input));
        var zWeight = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.ZWeight));
        var bWeight = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.BWeight));
        var aWeight = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.AWeight));
        var result = context.GetReturnType<TupleType>();
        var output = GatedDeltaNetStageUtility.GetLocalTensorType(result[0]);
        if (!GatedDeltaNetStageUtility.TryGetMaxShape(input, out var inputShape) ||
            !GatedDeltaNetStageUtility.TryGetMaxShape(bWeight, out var bWeightShape))
        {
            return Cost.Zero;
        }

        var tokens = inputShape[0];
        var localHeads = bWeightShape[1];
        var localValueDim = checked(localHeads * target.ValueHeadDim);
        var localQkvDim = checked(localHeads * ((target.KeyHeadDim * 2) + target.ValueHeadDim));
        var cost = Cost.Zero;
        GatedDeltaNetStageUtility.AddMatMulCost(
            context,
            input,
            zWeight,
            new TensorType(input.DType, new RankedShape(tokens, localValueDim)),
            ref cost);
        foreach (var weight in new[] { bWeight, aWeight })
        {
            GatedDeltaNetStageUtility.AddMatMulCost(
                context,
                input,
                weight,
                new TensorType(input.DType, new RankedShape(tokens, localHeads)),
                ref cost);
        }

        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.CPUCycles,
            checked(
                (UInt128)tokens * (UInt128)localHeads *
                (UInt128)((target.KeyHeadDim * target.ValueHeadDim * 5) +
                    (target.KeyHeadDim * 8) + (target.ValueHeadDim * 12))));
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryLoadBytes,
            checked(
                (UInt128)localQkvDim * (UInt128)input.DType.SizeInBytes +
                ((UInt128)localHeads * (UInt128)target.KeyHeadDim *
                    (UInt128)target.ValueHeadDim * (UInt128)DataTypes.Float32.SizeInBytes * 2)));
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryStoreBytes,
            checked(
                (UInt128)localHeads * (UInt128)target.KeyHeadDim *
                    (UInt128)target.ValueHeadDim * (UInt128)DataTypes.Float32.SizeInBytes +
                CostUtility.GetMemoryAccess(output)));
        return cost;
    }

    public IValue Visit(IEvaluateContext context, GatedDeltaNetRecurrentCore target)
    {
        var inputValue = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.Input);
        var input = inputValue.ToOrtTensor();
        var stateValue = context.GetArgumentValue(target, GatedDeltaNetRecurrentCore.State);
        var state = stateValue.AsTensor().Cast<Reference<IGatedDeltaNetState>>().Single().Value;
        var layerId = checked((int)context.GetArgumentValue(target, GatedDeltaNetRecurrentCore.LayerId).AsTensor().ToScalar<long>());
        var recurrentState = state.GetState(GatedDeltaNetStateKind.Recurrent, layerId).ToOrtTensor();
        var qkv = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.QKV).ToOrtTensor();
        var qkvLanes = GatedDeltaNetStageUtility.GetPackedLanes(
            state.Config,
            GatedDeltaNetStateKind.Convolution,
            GatedDeltaNetStateDimKind.ConvChannels);
        if (qkvLanes.Count != 0)
        {
            qkv = qkv.Unpack(
                qkvLanes.Count,
                Enumerable.Repeat(1, qkvLanes.Count).ToArray());
        }

        var zWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.ZWeight).ToOrtTensor();
        var bWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.BWeight).ToOrtTensor();
        var aWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.AWeight).ToOrtTensor();
        var aLog = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.ALog).ToOrtTensor().Cast(OrtDataType.Float);
        var dtBias = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.DtBias).ToOrtTensor().Cast(OrtDataType.Float);
        var normWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.NormWeight).ToOrtTensor().Cast(OrtDataType.Float);
        var keyDim = checked(target.NumKeyHeads * target.KeyHeadDim);
        var valueDim = checked(target.NumValueHeads * target.ValueHeadDim);
        var convDim = checked((keyDim * 2) + valueDim);
        var repeats = checked(target.NumValueHeads / target.NumKeyHeads);
        var outputs = new List<OrtKISharp.Tensor>(checked((int)input.Shape[0]));
        for (var token = 0L; token < input.Shape[0]; token++)
        {
            var hidden = OrtKI.Slice(input, new[] { token }, new[] { token + 1 }, new[] { 0L }, new[] { 1L });
            var currentQkv = OrtKI.Reshape(
                OrtKI.Slice(qkv, new[] { token }, new[] { token + 1 }, new[] { 0L }, new[] { 1L }),
                GatedDeltaNetStageUtility.Shape(convDim),
                0L);
            var z = OrtKI.MatMul(hidden, zWeight);
            var beta = OrtKI.Sigmoid(OrtKI.MatMul(hidden, bWeight));
            var a = OrtKI.MatMul(hidden, aWeight);
            var query = GatedDeltaNetStageUtility.Slice(currentQkv, 0, keyDim);
            var key = GatedDeltaNetStageUtility.Slice(currentQkv, keyDim, keyDim * 2);
            var value = GatedDeltaNetStageUtility.Slice(currentQkv, keyDim * 2, convDim);
            query = GatedDeltaNetStageUtility.RepeatHeads(
                query,
                target.NumKeyHeads,
                repeats,
                target.KeyHeadDim,
                target.NumValueHeads);
            key = GatedDeltaNetStageUtility.RepeatHeads(
                key,
                target.NumKeyHeads,
                repeats,
                target.KeyHeadDim,
                target.NumValueHeads);
            query = GatedDeltaNetStageUtility.L2Normalize(query);
            key = GatedDeltaNetStageUtility.L2Normalize(key);
            value = OrtKI.Reshape(value, GatedDeltaNetStageUtility.Shape(target.NumValueHeads, target.ValueHeadDim), 0L);

            var g = OrtKI.Mul(
                OrtKI.Neg(OrtKI.Exp(aLog)),
                OrtKI.Softplus(OrtKI.Add(a.Cast(OrtDataType.Float), dtBias)));
            var decay = OrtKI.Reshape(OrtKI.Exp(g), GatedDeltaNetStageUtility.Shape(target.NumValueHeads, 1L, 1L), 0L);
            var decayedState = OrtKI.Mul(recurrentState, decay);
            var keyFp32 = key.Cast(OrtDataType.Float);
            var valueFp32 = value.Cast(OrtDataType.Float);
            var keyColumn = OrtKI.Unsqueeze(keyFp32, new[] { -1L });
            var recalled = OrtKI.ReduceSum(
                OrtKI.Mul(decayedState, keyColumn),
                GatedDeltaNetStageUtility.Shape(1L),
                0L,
                0L);
            var delta = OrtKI.Mul(
                OrtKI.Sub(valueFp32, recalled),
                OrtKI.Reshape(beta.Cast(OrtDataType.Float), GatedDeltaNetStageUtility.Shape(target.NumValueHeads, 1L), 0L));
            recurrentState = OrtKI.Add(
                decayedState,
                OrtKI.Mul(keyColumn, OrtKI.Unsqueeze(delta, new[] { 1L })));

            var scaledQuery = OrtKI.Mul(
                query.Cast(OrtDataType.Float),
                (float)(1.0 / System.Math.Sqrt(target.KeyHeadDim)));
            var core = OrtKI.ReduceSum(
                OrtKI.Mul(recurrentState, OrtKI.Unsqueeze(scaledQuery, new[] { -1L })),
                GatedDeltaNetStageUtility.Shape(1L),
                0L,
                0L);
            var squareSum = OrtKI.ReduceSum(
                OrtKI.Mul(core, core),
                GatedDeltaNetStageUtility.Shape(-1L),
                1L,
                0L);
            var invRms = OrtKI.Reciprocal(
                OrtKI.Sqrt(OrtKI.Add(OrtKI.Div(squareSum, (float)target.ValueHeadDim), target.Epsilon)));
            var gate = OrtKI.Reshape(
                z.Cast(OrtDataType.Float),
                GatedDeltaNetStageUtility.Shape(target.NumValueHeads, target.ValueHeadDim),
                0L);
            var normalized = OrtKI.Mul(OrtKI.Mul(core, invRms), normWeight);
            normalized = OrtKI.Mul(normalized, OrtKI.Mul(gate, OrtKI.Sigmoid(gate)));
            outputs.Add(OrtKI.Reshape(
                normalized.Cast(input.DataType),
                GatedDeltaNetStageUtility.Shape(1L, valueDim),
                0L));
        }

        state.UpdateState(
            GatedDeltaNetStateKind.Recurrent,
            layerId,
            recurrentState.ToTensor().CastElementTo(DataTypes.Float32));
        var output = OrtKI.Concat(outputs.ToArray(), 0L);
        return new TupleValue([
            output.ToValue(inputValue.ElementType),
            stateValue,
        ]);
    }

    public static IRType InferType(GatedDeltaNetRecurrentCore target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Count != target.Parameters.Count)
        {
            return new InvalidType($"GatedDeltaNetRecurrentCore expects {target.Parameters.Count} inputs, got {arguments.Count}.");
        }

        if (arguments.OfType<InvalidType>().FirstOrDefault() is { } invalid)
        {
            return invalid;
        }

        if (arguments.Any(type => type is AnyType))
        {
            return AnyType.Default;
        }

        if (GatedDeltaNetStageUtility.AreTensorArguments(arguments, GatedDeltaNetRecurrentCore.LayerId.Index))
        {
            return InferTensorType(target, arguments);
        }

        if (!GatedDeltaNetStageUtility.TryGetDistributedArguments(
                arguments,
                GatedDeltaNetRecurrentCore.LayerId.Index,
                GatedDeltaNetRecurrentCore.State.Index,
                out var distributed,
                out var placement))
        {
            return new InvalidType(
                "GatedDeltaNetRecurrentCore inputs must be either all tensors or compatible distributed tensors.");
        }

        var localArguments = GatedDeltaNetStageUtility.GetTensorArguments(arguments, distributed);
        var tensorResult = InferTensorType(target, localArguments);
        if (tensorResult is not TupleType tuple)
        {
            return tensorResult;
        }

        if (!GatedDeltaNetStageUtility.TryGetContiguousAxes(
                distributed[GatedDeltaNetRecurrentCore.BWeight.Index].AxisPolicies[1],
                placement.Rank,
                out var headAxes))
        {
            return new InvalidType(
                "GatedDeltaNetRecurrentCore value-head split must be contiguous.");
        }

        var headShardCount = headAxes.Aggregate(
            1L,
            (product, axis) => checked(product * placement.Hierarchy[axis]));
        if (!GatedDeltaNetStageUtility.CoversPlacement(headAxes, placement.Rank))
        {
            return new InvalidType(
                "GatedDeltaNetRecurrentCore value-head split must cover the block placement so each state head has one writer.");
        }

        if (target.NumValueHeads % headShardCount != 0)
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore head split count {headShardCount} must divide value heads.");
        }

        var head = GatedDeltaNetStageUtility.CreateSplitPolicy(headAxes);
        var expected = new Dictionary<int, DistributedType>
        {
            [GatedDeltaNetRecurrentCore.Input.Index] = GatedDeltaNetStageUtility.CreateBroadcast(distributed[GatedDeltaNetRecurrentCore.Input.Index].TensorType, placement),
            [GatedDeltaNetRecurrentCore.QKV.Index] = GatedDeltaNetStageUtility.CreateBroadcast(distributed[GatedDeltaNetRecurrentCore.QKV.Index].TensorType, placement),
            [GatedDeltaNetRecurrentCore.ZWeight.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetRecurrentCore.ZWeight.Index].TensorType, [SBP.B, head], placement),
            [GatedDeltaNetRecurrentCore.BWeight.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetRecurrentCore.BWeight.Index].TensorType, [SBP.B, head], placement),
            [GatedDeltaNetRecurrentCore.AWeight.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetRecurrentCore.AWeight.Index].TensorType, [SBP.B, head], placement),
            [GatedDeltaNetRecurrentCore.ALog.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetRecurrentCore.ALog.Index].TensorType, [head], placement),
            [GatedDeltaNetRecurrentCore.DtBias.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetRecurrentCore.DtBias.Index].TensorType, [head], placement),
            [GatedDeltaNetRecurrentCore.NormWeight.Index] = GatedDeltaNetStageUtility.CreateBroadcast(distributed[GatedDeltaNetRecurrentCore.NormWeight.Index].TensorType, placement),
        };
        foreach (var (index, expectedType) in expected)
        {
            if (distributed[index] != expectedType)
            {
                return new InvalidType(
                    $"GatedDeltaNetRecurrentCore input {target.Parameters[index].Name} has distributed type " +
                    $"{distributed[index]}; expected {expectedType}.");
            }
        }

        return new TupleType([
            new DistributedType((TensorType)tuple[0], [SBP.B, head], placement),
            tuple[1],
        ]);
    }

    public static IRType InferTensorType(
        GatedDeltaNetRecurrentCore target,
        IReadOnlyList<IRType> arguments)
    {
        if (target.NumKeyHeads <= 0 || target.NumValueHeads <= 0 ||
            target.NumValueHeads % target.NumKeyHeads != 0 ||
            target.KeyHeadDim <= 0 || target.ValueHeadDim <= 0)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore head configuration is invalid.");
        }

        if (arguments[GatedDeltaNetRecurrentCore.Input.Index] is not TensorType input ||
            arguments[GatedDeltaNetRecurrentCore.State.Index] is not TensorType state ||
            arguments[GatedDeltaNetRecurrentCore.LayerId.Index] is not DimensionType)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore expects tensor operands and a dimension-valued layer id.");
        }

        if (!GatedDeltaNetEvaluator.TryGetStateConfig(state, out var stateConfig, out var stateError))
        {
            return new InvalidType(stateError);
        }

        if (input.Shape is not RankedShape { Rank: 2 } inputShape ||
            input.DType != stateConfig.ActivationPrimType)
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore input must be a rank-2 scalar {stateConfig.ActivationPrimType} tensor.");
        }

        if (!inputShape[1].IsFixed)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore hidden size must be fixed.");
        }

        var hidden = inputShape[1].FixedValue;
        var keyDim = checked(target.NumKeyHeads * target.KeyHeadDim);
        var valueDim = checked(target.NumValueHeads * target.ValueHeadDim);
        var convDim = checked((keyDim * 2) + valueDim);
        if (stateConfig.NumKeyHeads != target.NumKeyHeads ||
            stateConfig.NumValueHeads != target.NumValueHeads ||
            stateConfig.KeyHeadDim != target.KeyHeadDim ||
            stateConfig.ValueHeadDim != target.ValueHeadDim ||
            stateConfig.HiddenSize != hidden)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore state config does not match the operator geometry.");
        }

        var checks = new (ParameterInfo Parameter, long[] Shape, DataType? DType)[]
        {
            (GatedDeltaNetRecurrentCore.ZWeight, [hidden, valueDim], input.DType),
            (GatedDeltaNetRecurrentCore.BWeight, [hidden, target.NumValueHeads], input.DType),
            (GatedDeltaNetRecurrentCore.AWeight, [hidden, target.NumValueHeads], input.DType),
            (GatedDeltaNetRecurrentCore.ALog, [target.NumValueHeads], null),
            (GatedDeltaNetRecurrentCore.DtBias, [target.NumValueHeads], null),
            (GatedDeltaNetRecurrentCore.NormWeight, [target.ValueHeadDim], null),
        };
        foreach (var (parameter, expectedShape, expectedDType) in checks)
        {
            if (arguments[parameter.Index] is not TensorType type)
            {
                return new InvalidType($"GatedDeltaNetRecurrentCore {parameter.Name} must be a tensor.");
            }

            if (type.Shape is not RankedShape shape || shape.Rank != expectedShape.Length)
            {
                return new InvalidType(
                    $"GatedDeltaNetRecurrentCore {parameter.Name} has invalid shape {type.Shape}.");
            }

            for (var axis = 0; axis < shape.Rank; axis++)
            {
                if (expectedShape[axis] >= 0 &&
                    !GatedDeltaNetStageUtility.IsFixedValue(shape[axis], expectedShape[axis]))
                {
                    return new InvalidType(
                        $"GatedDeltaNetRecurrentCore {parameter.Name} axis {axis} must be {expectedShape[axis]}, got {shape[axis]}.");
                }
            }

            if (expectedDType is not null && type.DType != expectedDType)
            {
                return new InvalidType(
                    $"GatedDeltaNetRecurrentCore {parameter.Name} must use {expectedDType}, got {type.DType}.");
            }
        }

        if (arguments[GatedDeltaNetRecurrentCore.QKV.Index] is not TensorType qkv)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore QKV must be a tensor.");
        }

        var qkvLanes = GatedDeltaNetStageUtility.GetPackedLanes(
            stateConfig,
            GatedDeltaNetStateKind.Convolution,
            GatedDeltaNetStateDimKind.ConvChannels);
        var expectedQkv = GatedDeltaNetStageUtility.PackTensorAxis(
            new TensorType(input.DType, new RankedShape(inputShape[0], convDim)),
            qkvLanes,
            1);
        if (expectedQkv is not TensorType expectedQkvType || qkv != expectedQkvType)
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore QKV must use the state projection layout {expectedQkv}, got {qkv}.");
        }

        return new TupleType([
            new TensorType(input.DType, new RankedShape(inputShape[0], valueDim)),
            state,
        ]);
    }
}

internal static class GatedDeltaNetStageUtility
{
    public static bool AreTensorArguments(IReadOnlyList<IRType> arguments, int dimensionIndex) =>
        arguments.Select((type, index) => (type, index)).All(item =>
            item.index == dimensionIndex
                ? item.type is DimensionType
                : item.type is TensorType);

    public static bool TryGetDistributedArguments(
        IReadOnlyList<IRType> arguments,
        int dimensionIndex,
        int invariantTensorIndex,
        out DistributedType[] distributed,
        out Placement placement)
    {
        distributed = new DistributedType[arguments.Count];
        var values = arguments
            .Select((type, index) => (type, index))
            .Where(item => item.index != dimensionIndex && item.index != invariantTensorIndex)
            .ToArray();
        var commonPlacement = values
            .Select(item => item.type)
            .OfType<DistributedType>()
            .FirstOrDefault()?.Placement;
        placement = commonPlacement;
        if (arguments[dimensionIndex] is not DimensionType ||
            arguments[invariantTensorIndex] is not TensorType ||
            commonPlacement is null)
        {
            return false;
        }

        foreach (var (type, index) in values)
        {
            if (type is not DistributedType distributedType ||
                distributedType.Placement != commonPlacement ||
                distributedType.Partial is not null ||
                distributedType.AxisPolicies.Count != distributedType.TensorType.Shape.Rank)
            {
                return false;
            }

            distributed[index] = distributedType;
        }

        return true;
    }

    public static IRType[] GetTensorArguments(
        IReadOnlyList<IRType> arguments,
        IReadOnlyList<DistributedType> distributed)
    {
        var result = new IRType[arguments.Count];
        for (var index = 0; index < arguments.Count; index++)
        {
            result[index] = arguments[index] is DimensionType
                ? arguments[index]
                : distributed[index]?.TensorType ?? arguments[index];
        }

        return result;
    }

    public static bool TryGetContiguousAxes(SBP policy, int placementRank, out int[] axes)
    {
        axes = policy switch
        {
            SBPBroadCast => Array.Empty<int>(),
            SBPSplit { IsContiguous: true } split => split.HierarchyAxes.ToArray(),
            _ => null!,
        };
        return axes is not null &&
            axes.Distinct().Count() == axes.Length &&
            axes.All(axis => axis >= 0 && axis < placementRank);
    }

    public static bool CoversPlacement(IReadOnlyList<int> axes, int placementRank) =>
        axes.OrderBy(axis => axis).SequenceEqual(Enumerable.Range(0, placementRank));

    public static SBP CreateSplitPolicy(IReadOnlyList<int> axes)
        => axes.Count == 0 ? SBP.B : SBP.SContiguous(axes.ToArray());

    public static DistributedType Create(
        TensorType tensorType,
        IReadOnlyList<SBP> policies,
        Placement placement) =>
        new(tensorType, policies.ToArray(), placement);

    public static DistributedType CreateBroadcast(TensorType tensorType, Placement placement) =>
        Create(tensorType, Enumerable.Repeat<SBP>(SBP.B, tensorType.Shape.Rank).ToArray(), placement);

    public static bool AreCompatible(Dimension lhs, Dimension rhs) =>
        !lhs.IsFixed || !rhs.IsFixed || lhs.FixedValue == rhs.FixedValue;

    public static bool IsFixedValue(Dimension dimension, long expected) =>
        !dimension.IsFixed || dimension.FixedValue == expected;

    public static TensorType GetLocalTensorType(IRType type) => type switch
    {
        TensorType tensor => tensor,
        DistributedType distributed => DistributedUtility.GetDividedTensorType(
            distributed,
            DistributedUtility.DivideFlags.MaxShape),
        _ => TensorType.Invalid(DataTypes.Float32),
    };

    public static DataType GetScalarDataType(DataType type) => type switch
    {
        VectorType vector => GetScalarDataType(vector.ElemType),
        _ => type,
    };

    public static IRArray<int> GetPackedLanes(
        GatedDeltaNetStateConfig config,
        GatedDeltaNetStateKind kind,
        GatedDeltaNetStateDimKind axis)
    {
        var vectorizedAxes = config.GetVectorizedAxes(kind);
        var lanes = config.GetLanes(kind);
        var result = vectorizedAxes
            .Select((vectorizedAxis, index) => (vectorizedAxis, lane: lanes[index]))
            .Where(item => item.vectorizedAxis == axis)
            .Select(item => item.lane)
            .ToArray();
        return result;
    }

    public static IRType PackTensorAxis(TensorType input, IRArray<int> lanes, int axis)
    {
        if (lanes.Count == 0)
        {
            return input;
        }

        return TypeInference.PackType(
            input,
            lanes,
            Enumerable.Repeat(axis, lanes.Count).ToArray());
    }

    public static TensorType UnpackTensorAxis(TensorType input, int axis)
    {
        var laneCount = input.DType is VectorType vector ? vector.Lanes.Count : 0;
        if (laneCount == 0)
        {
            return input;
        }

        return TypeInference.UnpackType(
            input,
            Enumerable.Repeat(axis, laneCount).ToArray()) switch
        {
            TensorType tensor => tensor,
            IRType invalid => throw new InvalidOperationException(
                $"Cannot unpack GatedDeltaNet tensor type {input}: {invalid}."),
        };
    }

    public static IValue PackValue(
        OrtKISharp.Tensor input,
        DataType scalarType,
        IRArray<int> lanes,
        int axis)
    {
        if (lanes.Count == 0)
        {
            return Value.FromTensor(input.ToTensor().CastElementTo(scalarType));
        }

        var axes = Enumerable.Repeat(axis, lanes.Count).ToArray();
        return input
            .Pack(0, lanes, axes)
            .ToValue(TypeInference.PackType(scalarType, lanes));
    }

    public static bool TryGetMaxShape(TensorType type, out long[] shape)
    {
        if (CompilerServices.TryGetMaxShape(type.Shape, out var maxShape) && maxShape is not null)
        {
            shape = maxShape;
            return true;
        }

        shape = Array.Empty<long>();
        return false;
    }

    public static void AddMatMulCost(
        ICostEvaluateContext context,
        TensorType lhs,
        TensorType rhs,
        TensorType output,
        ref Cost cost)
    {
        if (context.TargetCostModel.TryGetMatMulCost(
                new(
                    new TargetCostTensor(lhs.DType, lhs.Shape),
                    new TargetCostTensor(rhs.DType, rhs.Shape),
                    new TargetCostTensor(output.DType, output.Shape),
                    output.DType,
                    MatMulOpCostKind.Simt),
                out var matMulCost))
        {
            cost += matMulCost;
        }
        else if (TryGetMaxShape(output, out var outputShape) && TryGetMaxShape(lhs, out var lhsShape))
        {
            AddCostFactor(
                cost,
                CostFactorNames.CPUCycles,
                checked(
                    (UInt128)outputShape.Aggregate(1L, static (a, b) => a * b) *
                    (UInt128)lhsShape[^1] * 2));
        }
    }

    public static void AddCostFactor(Cost cost, string name, UInt128 value)
    {
        cost.Factors.TryGetValue(name, out var current);
        cost[name] = checked(current + value);
    }

    public static OrtKISharp.Tensor Shape(params long[] dimensions) =>
        OrtKISharp.Tensor.MakeTensor(dimensions);

    public static OrtKISharp.Tensor Slice(OrtKISharp.Tensor input, long start, long end) =>
        OrtKI.Slice(input, new[] { start }, new[] { end }, new[] { -1L }, new[] { 1L });

    public static OrtKISharp.Tensor RepeatHeads(
        OrtKISharp.Tensor input,
        long numKeyHeads,
        long repeats,
        long headDim,
        long numValueHeads)
    {
        var reshaped = OrtKI.Reshape(input, Shape(numKeyHeads, headDim), 0L);
        if (repeats != 1)
        {
            reshaped = OrtKI.Unsqueeze(reshaped, new[] { 1L });
            reshaped = OrtKI.Expand(reshaped, Shape(numKeyHeads, repeats, headDim));
        }

        return OrtKI.Reshape(reshaped, Shape(numValueHeads, headDim), 0L);
    }

    public static OrtKISharp.Tensor L2Normalize(OrtKISharp.Tensor input)
    {
        var inputFp32 = input.Cast(OrtDataType.Float);
        var squareSum = OrtKI.ReduceSum(OrtKI.Mul(inputFp32, inputFp32), Shape(-1L), 1L, 0L);
        var invNorm = OrtKI.Reciprocal(OrtKI.Sqrt(OrtKI.Add(squareSum, 1e-6f)));
        return OrtKI.Mul(inputFp32, invNorm);
    }
}
