// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="GatedDeltaNet"/>.
/// </summary>
public sealed class GatedDeltaNetEvaluator :
    IEvaluator<GatedDeltaNet>,
    ITypeInferencer<GatedDeltaNet>
{
    public IRType Visit(ITypeInferenceContext context, GatedDeltaNet target)
    {
        var arguments = target.Parameters
            .Select(parameter => context.CheckArgumentType<IRType>(target, parameter))
            .ToArray();
        return InferType(target, arguments);
    }

    public IValue Visit(IEvaluateContext context, GatedDeltaNet target)
    {
        var inputValue = context.GetArgumentValueAsTensor(target, GatedDeltaNet.Input);
        var inputType = inputValue.ElementType;
        var input = inputValue.ToOrtTensor();
        var stateValue = context.GetArgumentValue(target, GatedDeltaNet.State);
        var state = stateValue.AsTensor().Cast<Reference<IGatedDeltaNetState>>().Single().Value;
        var layerId = checked((int)context.GetArgumentValue(target, GatedDeltaNet.LayerId).AsTensor().ToScalar<long>());
        var convState = state.GetState(GatedDeltaNetStateKind.Convolution, layerId).ToOrtTensor();
        var recurrentState = state.GetState(GatedDeltaNetStateKind.Recurrent, layerId).ToOrtTensor();
        var qkvWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNet.QKVWeight).ToOrtTensor();
        var zWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNet.ZWeight).ToOrtTensor();
        var bWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNet.BWeight).ToOrtTensor();
        var aWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNet.AWeight).ToOrtTensor();
        var convWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNet.ConvWeight).ToOrtTensor();
        var aLog = context.GetArgumentValueAsTensor(target, GatedDeltaNet.ALog).ToOrtTensor().Cast(OrtDataType.Float);
        var dtBias = context.GetArgumentValueAsTensor(target, GatedDeltaNet.DtBias).ToOrtTensor().Cast(OrtDataType.Float);
        var normWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNet.NormWeight).ToOrtTensor().Cast(OrtDataType.Float);
        var outputWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNet.OutputWeight).ToOrtTensor();

        var keyDim = checked(target.NumKeyHeads * target.KeyHeadDim);
        var valueDim = checked(target.NumValueHeads * target.ValueHeadDim);
        var convDim = checked((keyDim * 2) + valueDim);
        var repeats = checked(target.NumValueHeads / target.NumKeyHeads);
        var outputs = new List<OrtKISharp.Tensor>(checked((int)input.Shape[0]));
        for (var token = 0L; token < input.Shape[0]; token++)
        {
            var hidden = OrtKI.Slice(input, new[] { token }, new[] { token + 1 }, new[] { 0L }, new[] { 1L });
            var mixedQkv = OrtKI.MatMul(hidden, qkvWeight);
            var z = OrtKI.MatMul(hidden, zWeight);
            var beta = OrtKI.Sigmoid(OrtKI.MatMul(hidden, bWeight));
            var a = OrtKI.MatMul(hidden, aWeight);

            var currentQkv = OrtKI.Transpose(mixedQkv, new[] { 1L, 0L });
            var convHistory = OrtKI.Concat(new[] { convState, currentQkv }, 1L);
            convState = OrtKI.Slice(
                convHistory,
                new[] { 1L },
                new[] { target.ConvKernelSize },
                new[] { 1L },
                new[] { 1L });
            var convolved = OrtKI.ReduceSum(
                OrtKI.Mul(convHistory, convWeight),
                Shape(1L),
                0L,
                0L);
            convolved = OrtKI.Mul(convolved, OrtKI.Sigmoid(convolved));

            var query = Slice(convolved, 0, keyDim);
            var key = Slice(convolved, keyDim, keyDim * 2);
            var value = Slice(convolved, keyDim * 2, convDim);
            query = RepeatHeads(query, target.NumKeyHeads, repeats, target.KeyHeadDim, target.NumValueHeads);
            key = RepeatHeads(key, target.NumKeyHeads, repeats, target.KeyHeadDim, target.NumValueHeads);
            query = L2Normalize(query);
            key = L2Normalize(key);
            value = OrtKI.Reshape(value, Shape(target.NumValueHeads, target.ValueHeadDim), 0L);

            var g = OrtKI.Mul(
                OrtKI.Neg(OrtKI.Exp(aLog)),
                OrtKI.Softplus(OrtKI.Add(a.Cast(OrtDataType.Float), dtBias)));
            var decay = OrtKI.Reshape(OrtKI.Exp(g), Shape(target.NumValueHeads, 1L, 1L), 0L);
            var decayedState = OrtKI.Mul(recurrentState, decay);
            var keyFp32 = key.Cast(OrtDataType.Float);
            var valueFp32 = value.Cast(OrtDataType.Float);
            var keyColumn = OrtKI.Unsqueeze(keyFp32, new[] { -1L });
            var recalled = OrtKI.ReduceSum(
                OrtKI.Mul(decayedState, keyColumn),
                Shape(1L),
                0L,
                0L);
            var delta = OrtKI.Mul(
                OrtKI.Sub(valueFp32, recalled),
                OrtKI.Reshape(beta.Cast(OrtDataType.Float), Shape(target.NumValueHeads, 1L), 0L));
            recurrentState = OrtKI.Add(
                decayedState,
                OrtKI.Mul(keyColumn, OrtKI.Unsqueeze(delta, new[] { 1L })));

            var scaledQuery = OrtKI.Mul(
                query.Cast(OrtDataType.Float),
                (float)(1.0 / System.Math.Sqrt(target.KeyHeadDim)));
            var core = OrtKI.ReduceSum(
                OrtKI.Mul(recurrentState, OrtKI.Unsqueeze(scaledQuery, new[] { -1L })),
                Shape(1L),
                0L,
                0L);
            var squareSum = OrtKI.ReduceSum(OrtKI.Mul(core, core), Shape(-1L), 1L, 0L);
            var invRms = OrtKI.Reciprocal(OrtKI.Sqrt(OrtKI.Add(OrtKI.Div(squareSum, (float)target.ValueHeadDim), target.Epsilon)));
            var gate = OrtKI.Reshape(z.Cast(OrtDataType.Float), Shape(target.NumValueHeads, target.ValueHeadDim), 0L);
            var normalized = OrtKI.Mul(OrtKI.Mul(core, invRms), normWeight);
            normalized = OrtKI.Mul(normalized, OrtKI.Mul(gate, OrtKI.Sigmoid(gate)));
            var projectedInput = OrtKI.Reshape(
                normalized.Cast(input.DataType),
                Shape(1L, valueDim),
                0L);
            outputs.Add(OrtKI.MatMul(projectedInput, outputWeight));
        }

        var output = OrtKI.Concat(outputs.ToArray(), 0L).ToTensor().CastElementTo(inputType);
        state.UpdateState(
            GatedDeltaNetStateKind.Convolution,
            layerId,
            convState.ToTensor().CastElementTo(inputType));
        state.UpdateState(
            GatedDeltaNetStateKind.Recurrent,
            layerId,
            recurrentState.ToTensor().CastElementTo(DataTypes.Float32));
        return new TupleValue([
            Value.FromTensor(output),
            stateValue,
        ]);
    }

    public static IRType InferType(GatedDeltaNet target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Count != target.Parameters.Count)
        {
            return new InvalidType($"GatedDeltaNet expects {target.Parameters.Count} inputs, got {arguments.Count}.");
        }

        if (arguments.OfType<InvalidType>().FirstOrDefault() is { } invalid)
        {
            return invalid;
        }

        if (arguments.Any(type => type is AnyType))
        {
            return AnyType.Default;
        }

        return InferTensorType(target, arguments);
    }

    private static IRType InferTensorType(GatedDeltaNet target, IReadOnlyList<IRType> arguments)
    {
        if (target.NumKeyHeads <= 0 || target.NumValueHeads <= 0 ||
            target.NumValueHeads % target.NumKeyHeads != 0 ||
            target.KeyHeadDim <= 0 || target.ValueHeadDim <= 0 || target.ConvKernelSize < 2)
        {
            return new InvalidType("GatedDeltaNet head counts, head dimensions, and convolution kernel size are invalid.");
        }

        if (arguments[GatedDeltaNet.Input.Index] is not TensorType input ||
            arguments[GatedDeltaNet.State.Index] is not TensorType state ||
            arguments[GatedDeltaNet.LayerId.Index] is not DimensionType)
        {
            return new InvalidType("GatedDeltaNet expects tensor input/state and a dimension-valued layer id.");
        }

        if (!TryGetStateConfig(state, out var stateConfig, out var stateError))
        {
            return new InvalidType(stateError);
        }

        if (input.Shape is not RankedShape { Rank: 2 } inputShape)
        {
            return new InvalidType($"GatedDeltaNet input must have rank 2, got {input.Shape}.");
        }

        if (!input.DType.IsFloat())
        {
            return new InvalidType($"GatedDeltaNet input must be floating point, got {input.DType}.");
        }

        var hidden = inputShape[1];
        if (!hidden.IsFixed)
        {
            return new InvalidType("GatedDeltaNet hidden size must be fixed.");
        }

        var keyDim = checked(target.NumKeyHeads * target.KeyHeadDim);
        var valueDim = checked(target.NumValueHeads * target.ValueHeadDim);
        var convDim = checked((keyDim * 2) + valueDim);
        if (!ValidateConfig(target, stateConfig, hidden.FixedValue, out var configError))
        {
            return new InvalidType(configError);
        }

        var checks = new (ParameterInfo Parameter, long[] Shape, DataType? DType)[]
        {
            (GatedDeltaNet.QKVWeight, [hidden.FixedValue, convDim], input.DType),
            (GatedDeltaNet.ZWeight, [hidden.FixedValue, valueDim], input.DType),
            (GatedDeltaNet.BWeight, [hidden.FixedValue, target.NumValueHeads], input.DType),
            (GatedDeltaNet.AWeight, [hidden.FixedValue, target.NumValueHeads], input.DType),
            (GatedDeltaNet.ConvWeight, [convDim, target.ConvKernelSize], input.DType),
            (GatedDeltaNet.ALog, [target.NumValueHeads], null),
            (GatedDeltaNet.DtBias, [target.NumValueHeads], null),
            (GatedDeltaNet.NormWeight, [target.ValueHeadDim], null),
            (GatedDeltaNet.OutputWeight, [valueDim, hidden.FixedValue], input.DType),
        };
        foreach (var (parameter, expectedShape, expectedDType) in checks)
        {
            if (arguments[parameter.Index] is not TensorType type)
            {
                return new InvalidType($"GatedDeltaNet {parameter.Name} must be a tensor.");
            }

            if (type.Shape is not RankedShape shape || shape.Rank != expectedShape.Length)
            {
                return new InvalidType($"GatedDeltaNet {parameter.Name} must have shape [{string.Join(',', expectedShape)}], got {type.Shape}.");
            }

            for (var axis = 0; axis < expectedShape.Length; axis++)
            {
                if (shape[axis].IsFixed && shape[axis].FixedValue != expectedShape[axis])
                {
                    return new InvalidType($"GatedDeltaNet {parameter.Name} axis {axis} must be {expectedShape[axis]}, got {shape[axis]}.");
                }
            }

            if (expectedDType is not null && type.DType != expectedDType)
            {
                return new InvalidType($"GatedDeltaNet {parameter.Name} must have dtype {expectedDType}, got {type.DType}.");
            }
        }

        return new TupleType([
            input,
            state,
        ]);
    }

    internal static bool TryGetStateConfig(
        IRType type,
        out GatedDeltaNetStateConfig config,
        out string error)
    {
        var tensor = type switch
        {
            TensorType value => value,
            DistributedType value => value.TensorType,
            _ => null,
        };
        if (tensor is not
            {
                Shape: RankedShape { Rank: 0 },
                DType: ReferenceType
                {
                    ElemType: GatedDeltaNetStateType { Config: { } stateConfig },
                },
            })
        {
            config = null!;
            error = $"GatedDeltaNet state must be a scalar Reference<GatedDeltaNetState>, got {type}.";
            return false;
        }

        config = stateConfig;
        error = string.Empty;
        return true;
    }

    internal static bool ValidateConfig(
        GatedDeltaNet target,
        GatedDeltaNetStateConfig config,
        long hiddenSize,
        out string error)
    {
        if (config.NumKeyHeads != target.NumKeyHeads ||
            config.NumValueHeads != target.NumValueHeads ||
            config.KeyHeadDim != target.KeyHeadDim ||
            config.ValueHeadDim != target.ValueHeadDim ||
            config.ConvKernelSize != target.ConvKernelSize ||
            config.HiddenSize != hiddenSize)
        {
            error = $"GatedDeltaNet state config {config} does not match the operator geometry.";
            return false;
        }

        error = string.Empty;
        return true;
    }

    private static OrtKISharp.Tensor Shape(params long[] dimensions) => OrtKISharp.Tensor.MakeTensor(dimensions);

    private static OrtKISharp.Tensor Slice(OrtKISharp.Tensor input, long start, long end) =>
        OrtKI.Slice(input, new[] { start }, new[] { end }, new[] { -1L }, new[] { 1L });

    private static OrtKISharp.Tensor RepeatHeads(
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

    private static OrtKISharp.Tensor L2Normalize(OrtKISharp.Tensor input)
    {
        var inputFp32 = input.Cast(OrtDataType.Float);
        var squareSum = OrtKI.ReduceSum(OrtKI.Mul(inputFp32, inputFp32), Shape(-1L), 1L, 0L);
        var invNorm = OrtKI.Reciprocal(OrtKI.Sqrt(OrtKI.Add(squareSum, 1e-6f)));
        return OrtKI.Mul(inputFp32, invNorm);
    }
}
