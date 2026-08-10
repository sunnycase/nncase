// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#pragma warning disable SA1008 // Opening parenthesis should be spaced correctly

using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// Evaluator for <see cref="VectorizedCast"/>.
/// </summary>
public class VectorizedCastEvaluator : IEvaluator<VectorizedCast>, ITypeInferencer<VectorizedCast>, IOpPrinter<VectorizedCast>, ICostEvaluator<VectorizedCast>, IMetricEvaluator<VectorizedCast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, VectorizedCast cast)
    {
        var input = context.GetArgumentValue(cast, VectorizedCast.Input).AsTensor();
        IValue result;
        if (cast.NewType is VectorType vt && !cast.VectorizeAxes.IsDefaultOrEmpty)
        {
            if (cast.VectorizeAxes.Count > 1)
            {
                throw new NotSupportedException("Vectorize axes must be one");
            }

            input = Nncase.IR.F.Tensors.Unpack(input, ((VectorType)input.ElementType).Lanes.ToArray(), cast.VectorizeAxes.ToArray()).Evaluate().AsTensor();
            input = input.CastTo(vt.ElemType);
            input = Nncase.IR.F.Tensors.Pack(input, vt.Lanes.ToArray(), cast.VectorizeAxes.ToArray()).Evaluate().AsTensor();
            result = Value.FromTensor(input);
        }
        else
        {
            result = Value.FromTensor(input.CastTo(cast.NewType, cast.CastMode));
        }

        if (context.CurrentCall[VectorizedCast.PostOps] is Fusion lambda)
        {
            return CompilerServices.Evaluate(lambda.Body, new Dictionary<IVar, IValue>() { { lambda.Parameters[0], result } });
        }

        return result;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, VectorizedCast target)
    {
        var input = context.CheckArgumentType<IRType>(target, VectorizedCast.Input);
        var postOps = context.CheckArgumentType<IRType>(target, VectorizedCast.PostOps);
        if (!(postOps is NoneType || postOps is CallableType))
        {
            return new InvalidType($"PostOps must be None or Callable, but got {postOps}");
        }

        return input switch
        {
            TensorType t => Visit(target, t),
            DistributedType d => Visit(target, d),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, VectorizedCast target)
    {
        return $"{CompilerServices.Print(target.NewType)}({context.GetArgument(target, VectorizedCast.Input)})";
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, VectorizedCast target)
    {
        var input = context.GetArgumentType<IRType>(target, VectorizedCast.Input);
        var output = context.GetReturnType<IRType>();
        if (TargetOpCostModelUtility.TryGetTargetElementwiseCost(context.TargetCostModel, "vectorized_cast", [input], output, workPerElement: 1.0, out var targetCost))
        {
            return targetCost;
        }

        var macPerElement = 4;
        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(input, macPerElement),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, VectorizedCast target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, VectorizedCast.Input);
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
        };
    }

    private IRType Visit(VectorizedCast target, TensorType input)
    {
        if (input.DType is VectorType inputVectorType)
        {
            if (target.NewType is not VectorType outputVectorType)
            {
                return new InvalidType("A VectorizedCast with vectorized input requires a vectorized output dtype");
            }

            if (target.VectorizeAxes.IsDefaultOrEmpty)
            {
                return new TensorType(target.NewType, input.Shape);
            }

            var axes = target.VectorizeAxes.ToArray();
            if (axes.Length != inputVectorType.Lanes.Count || axes.Length != outputVectorType.Lanes.Count)
            {
                return new InvalidType(
                    $"VectorizedCast requires one input and output lane group per vectorized axis, but got " +
                    $"{axes.Length} axes, {inputVectorType.Lanes.Count} input lane groups, and " +
                    $"{outputVectorType.Lanes.Count} output lane groups");
            }

            if (axes.Distinct().Count() != axes.Length || axes.Any(axis => axis < 0 || axis >= input.Shape.Rank))
            {
                return new InvalidType("VectorizedCast axes must be distinct and in range");
            }

            if (TypeInference.UnpackType(input, axes) is not TensorType unpackedInput)
            {
                return new InvalidType($"VectorizedCast cannot unpack input type {input}");
            }

            for (int i = 0; i < axes.Length; i++)
            {
                if (!Dimension.TryDivExactly(unpackedInput.Shape[axes[i]], outputVectorType.Lanes[i], out _))
                {
                    return new InvalidType(
                        $"VectorizedCast unpacked axis {axes[i]} extent {unpackedInput.Shape[axes[i]]} " +
                        $"is not divisible by output lane {outputVectorType.Lanes[i]}");
                }
            }

            var scalarOutput = new TensorType(outputVectorType.ElemType, unpackedInput.Shape);
            return TypeInference.PackType(scalarOutput, outputVectorType.Lanes.ToArray(), axes);
        }

        return new TensorType(target.NewType, input.Shape);
    }

    private IRType Visit(VectorizedCast target, DistributedType inType)
    {
        var outType = Visit(target, inType.TensorType);
        if (outType is not TensorType outTensorType)
        {
            return outType;
        }

        if (inType.AxisPolicies.Any(static policy => policy is SBPPartial))
        {
            return new InvalidType("VectorizedCast does not support partial distributed inputs");
        }

        if (target.VectorizeAxes.IsDefaultOrEmpty || inType.TensorType.DType is not VectorType)
        {
            return inType with { TensorType = outTensorType };
        }

        if (target.NewType is not VectorType outputVectorType ||
            TypeInference.UnpackType(inType, target.VectorizeAxes) is not DistributedType unpackedInput)
        {
            return new InvalidType($"VectorizedCast cannot unpack distributed input type {inType}");
        }

        var scalarOutput = unpackedInput with
        {
            TensorType = new TensorType(outputVectorType.ElemType, unpackedInput.TensorType.Shape),
        };

        return TypeInference.PackType(
            scalarOutput,
            outputVectorType.Lanes.ToArray(),
            target.VectorizeAxes.ToArray());
    }
}
