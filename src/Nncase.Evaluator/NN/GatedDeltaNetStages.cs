// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public sealed class GatedDeltaNetConvolutionEvaluator :
    IEvaluator<GatedDeltaNetConvolution>,
    ITypeInferencer<GatedDeltaNetConvolution>,
    ICostEvaluator<GatedDeltaNetConvolution>
{
    public IRType Visit(ITypeInferenceContext context, GatedDeltaNetConvolution target)
    {
        var arguments = target.Parameters
            .Select(parameter => context.CheckArgumentType<IRType>(target, parameter))
            .ToArray();
        return InferType(target, arguments);
    }

    public Cost Visit(ICostEvaluateContext context, GatedDeltaNetConvolution target)
    {
        var qkv = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetConvolution.QKV));
        var convWeight = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetConvolution.ConvWeight));
        var output = context.GetReturnType<TupleType>();
        var qkvOutput = GatedDeltaNetStageUtility.GetLocalTensorType(output[0]);
        var scalarQkvOutput = GatedDeltaNetStageUtility.UnpackTensorAxis(
            qkvOutput,
            1);
        if (!GatedDeltaNetStageUtility.TryGetMaxShape(scalarQkvOutput, out var qkvShape))
        {
            return Cost.Zero;
        }

        var cost = new Cost();
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.CPUCycles,
            checked((UInt128)qkvShape[0] * (UInt128)qkvShape[1] * (UInt128)((target.ConvKernelSize * 2) + 4)));
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryLoadBytes,
            checked(
                (UInt128)qkvShape[1] * (UInt128)(target.ConvKernelSize - 1) *
                (UInt128)GatedDeltaNetStageUtility.GetScalarDataType(qkv.DType).SizeInBytes +
                CostUtility.GetMemoryAccess(qkv) +
                CostUtility.GetMemoryAccess(convWeight)));
        GatedDeltaNetStageUtility.AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryStoreBytes,
            checked(
                (UInt128)qkvShape[1] * (UInt128)(target.ConvKernelSize - 1) *
                (UInt128)GatedDeltaNetStageUtility.GetScalarDataType(qkv.DType).SizeInBytes +
                CostUtility.GetMemoryAccess(qkvOutput)));
        return cost;
    }

    public IValue Visit(IEvaluateContext context, GatedDeltaNetConvolution target)
    {
        var qkvValue = context.GetArgumentValueAsTensor(target, GatedDeltaNetConvolution.QKV);
        var qkv = qkvValue.ToOrtTensor();
        var stateValue = context.GetArgumentValue(target, GatedDeltaNetConvolution.State);
        var state = stateValue.AsTensor().Cast<Reference<IGatedDeltaNetState>>().Single().Value;
        var layerId = checked((int)context.GetArgumentValue(target, GatedDeltaNetConvolution.LayerId).AsTensor().ToScalar<long>());
        var convState = state.GetState(GatedDeltaNetStateKind.Convolution, layerId).ToOrtTensor();
        var qkvLanes = GatedDeltaNetStageUtility.GetDataTypeLanes(qkvValue.ElementType);
        if (qkvLanes.Count != 0)
        {
            qkv = qkv.Unpack(
                qkvLanes.Count,
                Enumerable.Repeat(1, qkvLanes.Count).ToArray());
        }

        var scalarType = GatedDeltaNetStageUtility.GetScalarDataType(qkvValue.ElementType);
        var convWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetConvolution.ConvWeight).ToOrtTensor();
        var outputs = new List<OrtKISharp.Tensor>(checked((int)qkv.Shape[0]));
        for (var token = 0L; token < qkv.Shape[0]; token++)
        {
            var current = OrtKI.Transpose(
                OrtKI.Slice(qkv, new[] { token }, new[] { token + 1 }, new[] { 0L }, new[] { 1L }),
                new[] { 1L, 0L });
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
            convState.ToTensor().CastElementTo(scalarType));
        var output = OrtKI.Concat(outputs.ToArray(), 0L);
        var outputValue = GatedDeltaNetStageUtility.PackValue(
            output,
            scalarType,
            qkvLanes,
            1);
        return new TupleValue([
            outputValue,
            stateValue,
        ]);
    }

    public static IRType InferType(GatedDeltaNetConvolution target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Count != target.Parameters.Count)
        {
            return new InvalidType($"GatedDeltaNetConvolution expects {target.Parameters.Count} inputs, got {arguments.Count}.");
        }

        if (arguments
            .Select((type, index) => (type, index))
            .FirstOrDefault(item => item.type is InvalidType) is { type: InvalidType invalid, index: var invalidIndex })
        {
            return new InvalidType(
                $"GatedDeltaNetConvolution argument {target.Parameters[invalidIndex].Name} is invalid: {invalid.Reason}");
        }

        if (arguments.Any(type => type is AnyType))
        {
            return AnyType.Default;
        }

        if (GatedDeltaNetStageUtility.AreTensorArguments(arguments, GatedDeltaNetConvolution.LayerId.Index))
        {
            return InferTensorType(target, arguments);
        }

        if (!GatedDeltaNetStageUtility.TryGetDistributedArguments(
                arguments,
                GatedDeltaNetConvolution.LayerId.Index,
                GatedDeltaNetConvolution.State.Index,
                out var distributed,
                out var placement))
        {
            return new InvalidType(
                "GatedDeltaNetConvolution inputs must be either all tensors or compatible distributed tensors.");
        }

        var localArguments = GatedDeltaNetStageUtility.GetTensorArguments(arguments, distributed);
        var tensorResult = InferTensorType(target, localArguments);
        if (tensorResult is not TupleType tuple)
        {
            return tensorResult;
        }

        var channel = distributed[GatedDeltaNetConvolution.QKV.Index].AxisPolicies[1];
        if (!GatedDeltaNetStageUtility.TryGetContiguousAxes(channel, placement.Rank, out var channelAxes))
        {
            return new InvalidType(
                "GatedDeltaNetConvolution QKV channels must be broadcast or contiguously split.");
        }

        if (!GatedDeltaNetStageUtility.CoversPlacement(channelAxes, placement.Rank))
        {
            return new InvalidType(
                "GatedDeltaNetConvolution channel split must cover the block placement so each state channel has one writer.");
        }

        var expected = new Dictionary<int, DistributedType>
        {
            [GatedDeltaNetConvolution.QKV.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetConvolution.QKV.Index].TensorType, [SBP.B, channel], placement),
            [GatedDeltaNetConvolution.ConvWeight.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetConvolution.ConvWeight.Index].TensorType, [channel, SBP.B], placement),
        };
        foreach (var (index, expectedType) in expected)
        {
            if (distributed[index] != expectedType)
            {
                return new InvalidType(
                    $"GatedDeltaNetConvolution input {target.Parameters[index].Name} has distributed type " +
                    $"{distributed[index]}; expected {expectedType}.");
            }
        }

        return new TupleType([
            GatedDeltaNetStageUtility.Create((TensorType)tuple[0], [SBP.B, channel], placement),
            tuple[1],
        ]);
    }

    public static IRType InferTensorType(
        GatedDeltaNetConvolution target,
        IReadOnlyList<IRType> arguments)
    {
        if (target.ConvKernelSize < 2)
        {
            return new InvalidType("GatedDeltaNetConvolution convolution kernel size must be at least two.");
        }

        if (arguments[GatedDeltaNetConvolution.QKV.Index] is not TensorType qkv ||
            arguments[GatedDeltaNetConvolution.State.Index] is not TensorType state ||
            arguments[GatedDeltaNetConvolution.ConvWeight.Index] is not TensorType convWeight ||
            arguments[GatedDeltaNetConvolution.LayerId.Index] is not DimensionType)
        {
            return new InvalidType("GatedDeltaNetConvolution expects tensor operands and a dimension-valued layer id.");
        }

        if (!GatedDeltaNetEvaluator.TryGetStateConfig(state, out var stateConfig, out var stateError))
        {
            return new InvalidType(stateError);
        }

        if (qkv.Shape is not RankedShape { Rank: 2 } ||
            convWeight.Shape is not RankedShape { Rank: 2 } convWeightShape)
        {
            return new InvalidType("GatedDeltaNetConvolution expects rank-2 tensor inputs.");
        }

        var qkvLanes = stateConfig.GetLanes(
            GatedDeltaNetStateKind.Convolution,
            GatedDeltaNetStateDimKind.ConvChannels);
        DataType expectedQkvType = qkvLanes.Count == 0
            ? stateConfig.ActivationPrimType
            : new VectorType(stateConfig.ActivationPrimType, qkvLanes);
        if (qkv.DType != expectedQkvType || convWeight.DType != stateConfig.ActivationPrimType)
        {
            return new InvalidType(
                $"GatedDeltaNetConvolution QKV must use packed activation dtype " +
                $"{expectedQkvType}, and convolution weight must use " +
                $"{stateConfig.ActivationPrimType}.");
        }

        var scalarQkv = GatedDeltaNetStageUtility.UnpackTensorAxis(qkv, 1);
        if (scalarQkv.Shape is not RankedShape scalarQkvShape ||
            !GatedDeltaNetStageUtility.AreCompatible(scalarQkvShape[1], convWeightShape[0]) ||
            !GatedDeltaNetStageUtility.IsFixedValue(convWeightShape[1], target.ConvKernelSize) ||
            !GatedDeltaNetStageUtility.IsFixedValue(
                scalarQkvShape[1],
                stateConfig.GetDimension(GatedDeltaNetStateDimKind.ConvChannels)) ||
            stateConfig.ConvKernelSize != target.ConvKernelSize)
        {
            return new InvalidType(
                "GatedDeltaNetConvolution input shapes do not satisfy the stateful convolution contract.");
        }

        return new TupleType([qkv, state]);
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
        var qkv = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.QKV));
        var z = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.Z));
        var bProjection = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.BProjection));
        var aProjection = GatedDeltaNetStageUtility.GetLocalTensorType(
            context.GetArgumentType<IRType>(target, GatedDeltaNetRecurrentCore.AProjection));
        var result = context.GetReturnType<TupleType>();
        var output = GatedDeltaNetStageUtility.GetLocalTensorType(result[0]);
        if (!GatedDeltaNetStageUtility.TryGetMaxShape(output, out var outputShape))
        {
            return Cost.Zero;
        }

        var tokens = outputShape[0];
        var localHeads = outputShape[1] / target.ValueHeadDim;
        var localQkvDim = checked(localHeads * ((target.KeyHeadDim * 2) + target.ValueHeadDim));
        var cost = new Cost();
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
                (UInt128)localQkvDim * (UInt128)GatedDeltaNetStageUtility.GetScalarDataType(qkv.DType).SizeInBytes +
                CostUtility.GetMemoryAccess(z) +
                CostUtility.GetMemoryAccess(bProjection) +
                CostUtility.GetMemoryAccess(aProjection) +
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

        var zValue = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.Z);
        var z = zValue.ToOrtTensor();
        var zLanes = GatedDeltaNetStageUtility.GetDataTypeLanes(zValue.ElementType);
        if (zLanes.Count != 0)
        {
            z = z.Unpack(
                zLanes.Count,
                Enumerable.Repeat(1, zLanes.Count).ToArray());
        }

        var bProjectionValue = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.BProjection);
        var bProjection = bProjectionValue.ToOrtTensor();
        var aProjectionValue = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.AProjection);
        var aProjection = aProjectionValue.ToOrtTensor();
        var projectionLanes = GatedDeltaNetStageUtility.GetDataTypeLanes(bProjectionValue.ElementType);
        if (projectionLanes.Count != 0)
        {
            var unpackAxes = Enumerable.Repeat(1, projectionLanes.Count).ToArray();
            bProjection = bProjection.Unpack(projectionLanes.Count, unpackAxes);
            aProjection = aProjection.Unpack(projectionLanes.Count, unpackAxes);
        }
        var aLog = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.ALog).ToOrtTensor().Cast(OrtDataType.Float);
        var dtBias = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.DtBias).ToOrtTensor().Cast(OrtDataType.Float);
        var normWeight = context.GetArgumentValueAsTensor(target, GatedDeltaNetRecurrentCore.NormWeight).ToOrtTensor().Cast(OrtDataType.Float);
        var keyDim = checked(target.NumKeyHeads * target.KeyHeadDim);
        var valueDim = checked(target.NumValueHeads * target.ValueHeadDim);
        var convDim = checked((keyDim * 2) + valueDim);
        var repeats = checked(target.NumValueHeads / target.NumKeyHeads);
        var activationType = state.Config.ActivationPrimType;
        var outputs = new List<OrtKISharp.Tensor>(checked((int)qkv.Shape[0]));
        for (var token = 0L; token < qkv.Shape[0]; token++)
        {
            var currentQkv = OrtKI.Reshape(
                OrtKI.Slice(qkv, new[] { token }, new[] { token + 1 }, new[] { 0L }, new[] { 1L }),
                GatedDeltaNetStageUtility.Shape(convDim),
                0L);
            var currentZ = OrtKI.Slice(
                z,
                new[] { token },
                new[] { token + 1 },
                new[] { 0L },
                new[] { 1L });
            var beta = OrtKI.Sigmoid(OrtKI.Slice(
                bProjection,
                new[] { token },
                new[] { token + 1 },
                new[] { 0L },
                new[] { 1L }));
            var a = OrtKI.Slice(
                aProjection,
                new[] { token },
                new[] { token + 1 },
                new[] { 0L },
                new[] { 1L });
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
                currentZ.Cast(OrtDataType.Float),
                GatedDeltaNetStageUtility.Shape(target.NumValueHeads, target.ValueHeadDim),
                0L);
            var normalized = OrtKI.Mul(OrtKI.Mul(core, invRms), normWeight);
            normalized = OrtKI.Mul(normalized, OrtKI.Mul(gate, OrtKI.Sigmoid(gate)));
            outputs.Add(OrtKI.Reshape(
                normalized.Cast(qkv.DataType),
                GatedDeltaNetStageUtility.Shape(1L, valueDim),
                0L));
        }

        state.UpdateState(
            GatedDeltaNetStateKind.Recurrent,
            layerId,
            recurrentState.ToTensor().CastElementTo(DataTypes.Float32));
        var output = OrtKI.Concat(outputs.ToArray(), 0L);
        return new TupleValue([
            output.ToValue(activationType),
            stateValue,
        ]);
    }

    public static IRType InferType(GatedDeltaNetRecurrentCore target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Count != target.Parameters.Count)
        {
            return new InvalidType($"GatedDeltaNetRecurrentCore expects {target.Parameters.Count} inputs, got {arguments.Count}.");
        }

        if (arguments
            .Select((type, index) => (type, index))
            .FirstOrDefault(item => item.type is InvalidType) is { type: InvalidType invalid, index: var invalidIndex })
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore argument {target.Parameters[invalidIndex].Name} is invalid: {invalid.Reason}");
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
                distributed[GatedDeltaNetRecurrentCore.Z.Index].AxisPolicies[1],
                placement.Rank,
                out var headAxes))
        {
            return new InvalidType(
                "GatedDeltaNetRecurrentCore value-head split must be contiguous.");
        }

        if (!GatedDeltaNetStageUtility.CoversPlacement(headAxes, placement.Rank))
        {
            return new InvalidType(
                "GatedDeltaNetRecurrentCore value-head split must cover the block placement so each state head has one writer.");
        }

        var head = DistributedUtility.CreateUnitAlignedContiguousSplit(
            headAxes,
            placement,
            target.NumValueHeads);
        var value = DistributedUtility.CreateUnitAlignedContiguousSplit(
            headAxes,
            placement,
            target.NumValueHeads,
            target.ValueHeadDim);
        var zLaneCount = GatedDeltaNetStageUtility.GetVectorLaneCount(
            distributed[GatedDeltaNetRecurrentCore.Z.Index].TensorType.DType);
        if (!DistributedUtility.TryScaleSplitUnits(
                value,
                1,
                zLaneCount,
                out var packedValue))
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore value-head split {value} is not representable " +
                $"in Z's packed dtype {distributed[GatedDeltaNetRecurrentCore.Z.Index].TensorType.DType}.");
        }

        var expected = new Dictionary<int, DistributedType>
        {
            [GatedDeltaNetRecurrentCore.QKV.Index] = GatedDeltaNetStageUtility.CreateBroadcast(distributed[GatedDeltaNetRecurrentCore.QKV.Index].TensorType, placement),
            [GatedDeltaNetRecurrentCore.Z.Index] = GatedDeltaNetStageUtility.Create(distributed[GatedDeltaNetRecurrentCore.Z.Index].TensorType, [SBP.B, packedValue], placement),
            [GatedDeltaNetRecurrentCore.BProjection.Index] = GatedDeltaNetStageUtility.CreateBroadcast(distributed[GatedDeltaNetRecurrentCore.BProjection.Index].TensorType, placement),
            [GatedDeltaNetRecurrentCore.AProjection.Index] = GatedDeltaNetStageUtility.CreateBroadcast(distributed[GatedDeltaNetRecurrentCore.AProjection.Index].TensorType, placement),
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
            new DistributedType((TensorType)tuple[0], [SBP.B, value], placement),
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

        if (arguments[GatedDeltaNetRecurrentCore.State.Index] is not TensorType state ||
            arguments[GatedDeltaNetRecurrentCore.LayerId.Index] is not DimensionType)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore expects tensor operands and a dimension-valued layer id.");
        }

        if (!GatedDeltaNetEvaluator.TryGetStateConfig(state, out var stateConfig, out var stateError))
        {
            return new InvalidType(stateError);
        }

        if (arguments[GatedDeltaNetRecurrentCore.QKV.Index] is not TensorType qkv)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore QKV must be a tensor.");
        }

        var qkvLanes = GatedDeltaNetStageUtility.GetPackedLanes(
            stateConfig,
            GatedDeltaNetStateKind.Convolution,
            GatedDeltaNetStateDimKind.ConvChannels);
        var scalarQkv = GatedDeltaNetStageUtility.UnpackTensorAxis(qkv, 1);
        if (scalarQkv.Shape is not RankedShape { Rank: 2 } qkvShape ||
            scalarQkv.DType != stateConfig.ActivationPrimType)
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore QKV must unpack to a rank-2 " +
                $"{stateConfig.ActivationPrimType} tensor.");
        }

        var keyDim = checked(target.NumKeyHeads * target.KeyHeadDim);
        var valueDim = checked(target.NumValueHeads * target.ValueHeadDim);
        var convDim = checked((keyDim * 2) + valueDim);
        if (stateConfig.NumKeyHeads != target.NumKeyHeads ||
            stateConfig.NumValueHeads != target.NumValueHeads ||
            stateConfig.KeyHeadDim != target.KeyHeadDim ||
            stateConfig.ValueHeadDim != target.ValueHeadDim)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore state config does not match the operator geometry.");
        }

        var checks = new (ParameterInfo Parameter, long[] Shape, DataType? DType)[]
        {
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

        var expectedProjection = GatedDeltaNetStageUtility.PackTensorAxis(
            new TensorType(
                stateConfig.ActivationPrimType,
                new RankedShape(qkvShape[0], target.NumValueHeads)),
            stateConfig.ActivationLanes,
            1);
        foreach (var parameter in new[]
                 {
                     GatedDeltaNetRecurrentCore.BProjection,
                     GatedDeltaNetRecurrentCore.AProjection,
                 })
        {
            if (arguments[parameter.Index] != expectedProjection)
            {
                return new InvalidType(
                    $"GatedDeltaNetRecurrentCore {parameter.Name} must use the packed " +
                    $"activation layout {expectedProjection}, got {arguments[parameter.Index]}.");
            }
        }

        var expectedQkv = GatedDeltaNetStageUtility.PackTensorAxis(
            new TensorType(stateConfig.ActivationPrimType, new RankedShape(qkvShape[0], convDim)),
            qkvLanes,
            1);
        if (expectedQkv is not TensorType expectedQkvType || qkv != expectedQkvType)
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore QKV must use the state projection layout {expectedQkv}, got {qkv}.");
        }

        if (arguments[GatedDeltaNetRecurrentCore.Z.Index] is not TensorType z)
        {
            return new InvalidType("GatedDeltaNetRecurrentCore Z must be a tensor.");
        }

        var expectedZ = GatedDeltaNetStageUtility.PackTensorAxis(
            new TensorType(stateConfig.ActivationPrimType, new RankedShape(qkvShape[0], valueDim)),
            stateConfig.ActivationLanes,
            1);
        if (expectedZ is not TensorType expectedZType || z != expectedZType)
        {
            return new InvalidType(
                $"GatedDeltaNetRecurrentCore Z must use the packed activation layout {expectedZ}, got {z}.");
        }

        return new TupleType([
            new TensorType(stateConfig.ActivationPrimType, new RankedShape(qkvShape[0], valueDim)),
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
                : item.type is TensorType or NoneType);

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
            .Where(item => item.index != dimensionIndex &&
                item.index != invariantTensorIndex &&
                item.type is not NoneType)
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

    public static IRArray<int> GetDataTypeLanes(DataType type) => type switch
    {
        VectorType vector => vector.Lanes,
        _ => Array.Empty<int>(),
    };

    public static long GetVectorLaneCount(DataType type) => type switch
    {
        VectorType vector => vector.Lanes.Aggregate(
            GetVectorLaneCount(vector.ElemType),
            static (product, lane) => checked(product * lane)),
        _ => 1,
    };

    public static IRArray<int> GetPackedLanes(
        GatedDeltaNetStateConfig config,
        GatedDeltaNetStateKind kind,
        GatedDeltaNetStateDimKind axis)
    {
        return config.GetLanes(kind, axis);
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
