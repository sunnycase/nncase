// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public sealed class SparseExpertsEvaluator : ITypeInferencer<SparseExperts>, ICostEvaluator<SparseExperts>, IEvaluator<SparseExperts>
{
    public IRType Visit(ITypeInferenceContext context, SparseExperts target)
    {
        var arguments = target.Parameters
            .Select(parameter => context.CheckArgumentType<IRType>(target, parameter))
            .ToArray();
        return InferType(target, arguments);
    }

    public Cost Visit(ICostEvaluateContext context, SparseExperts target)
    {
        var packedQ = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.Q));
        var ids = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.RouterExpertIds));
        var routerWeights = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.RouterExpertWeights));
        var gateInputScale = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertGateInputScale));
        var gateWeight = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertGateProjW));
        var gateScale = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertGateProjScale));
        var downInputScale = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertDownInputScale));
        var downWeight = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertDownProjW));
        var downScale = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertDownProjScale));
        var upInputScale = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertUpInputScale));
        var upWeight = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertUpProjW));
        var upScale = GetLocalTensorType(context.GetArgumentType<IRType>(target, SparseExperts.MoeExpertUpProjScale));
        var packedOutput = GetLocalTensorType(context.GetReturnType<IRType>());
        var q = GetLogicalActivationType(packedQ);
        var output = GetLogicalActivationType(packedOutput);
        if (!TryGetMaxShape(q, out var qShape) ||
            !TryGetMaxShape(gateWeight, out var gateShape) ||
            !TryGetMaxShape(downWeight, out var downShape) ||
            !TryGetMaxShape(output, out var outputShape))
        {
            return Cost.Zero;
        }

        var tokens = qShape[0];
        var hidden = qShape[1];
        var intermediate = gateShape[1];
        var outputFeatures = outputShape[1];
        var topK = System.Math.Max(0, target.NumTopK);
        var inputTensor = new TargetCostTensor(q.DType, new RankedShape(tokens, hidden));
        var intermediateTensor = new TargetCostTensor(output.DType, new RankedShape(tokens, intermediate));
        var outputTensor = new TargetCostTensor(output.DType, new RankedShape(tokens, outputFeatures));
        var gateMatrix = new TargetCostTensor(gateWeight.DType, new RankedShape(hidden, intermediate));
        var upMatrix = new TargetCostTensor(upWeight.DType, new RankedShape(hidden, intermediate));
        var downMatrix = new TargetCostTensor(downWeight.DType, new RankedShape(intermediate, outputFeatures));
        var cost = Cost.Zero;
        if (context.TargetCostModel.TryGetMatMulCost(
                new(inputTensor, gateMatrix, intermediateTensor, output.DType, MatMulOpCostKind.Simt),
                out var gateCost) &&
            context.TargetCostModel.TryGetMatMulCost(
                new(inputTensor, upMatrix, intermediateTensor, output.DType, MatMulOpCostKind.Simt),
                out var upCost) &&
            context.TargetCostModel.TryGetMatMulCost(
                new(intermediateTensor, downMatrix, outputTensor, output.DType, MatMulOpCostKind.Simt),
                out var downCost))
        {
            cost = (gateCost + upCost + downCost) * (UInt128)topK;
        }
        else
        {
            cost[CostFactorNames.CPUCycles] = checked(
                (UInt128)tokens * (UInt128)topK *
                (((UInt128)2 * (UInt128)hidden * (UInt128)intermediate) +
                 ((UInt128)intermediate * (UInt128)outputFeatures)));
        }

        if (context.TargetCostModel.TryGetElementwiseCost(
                new("sparse_experts_swiglu", [intermediateTensor, intermediateTensor], intermediateTensor, 9.0),
                out var activationCost))
        {
            cost += activationCost * (UInt128)topK;
        }

        cost.Factors[CostFactorNames.BlockLocalMemoryLoadBytes] = checked(
            (UInt128)tokens * (UInt128)topK *
            (((UInt128)hidden * (UInt128)GetScalarByteCount(q.DType)) +
             ((UInt128)intermediate * (UInt128)hidden * (UInt128)(GetScalarByteCount(gateWeight.DType) + GetScalarByteCount(upWeight.DType))) +
             ((UInt128)outputFeatures * (UInt128)intermediate * (UInt128)GetScalarByteCount(downWeight.DType)) +
             (UInt128)(GetScalarByteCount(ids.DType) + GetScalarByteCount(routerWeights.DType) +
                 GetScalarByteCount(gateInputScale.DType) + GetScalarByteCount(gateScale.DType) +
                 GetScalarByteCount(downInputScale.DType) + GetScalarByteCount(downScale.DType) +
                 GetScalarByteCount(upInputScale.DType) + GetScalarByteCount(upScale.DType))));
        cost.Factors[CostFactorNames.BlockLocalMemoryStoreBytes] = checked(
            (UInt128)tokens * (UInt128)outputFeatures * (UInt128)GetScalarByteCount(output.DType));
        return cost;
    }

    public static IRType InferType(SparseExperts target, IReadOnlyList<IRType> arguments)
    {
        if (arguments.Count != target.Parameters.Count)
        {
            return new InvalidType($"SparseExperts expects {target.Parameters.Count} inputs, got {arguments.Count}.");
        }

        if (arguments.OfType<InvalidType>().FirstOrDefault() is { } invalid)
        {
            return invalid;
        }

        if (arguments.Any(type => type is AnyType))
        {
            return AnyType.Default;
        }

        if (arguments.All(type => type is TensorType))
        {
            var tensors = arguments.Cast<TensorType>().ToArray();
            return ValidateTensorContract(target, tensors) is { } plainTensorCheck
                ? plainTensorCheck
                : tensors[SparseExperts.Q.Index];
        }

        if (!arguments.All(type => type is DistributedType))
        {
            return new InvalidType("SparseExperts requires either tensor inputs or distributed inputs with one common placement.");
        }

        var distributed = arguments.Cast<DistributedType>().ToArray();
        var tensorCheck = ValidateTensorContract(target, distributed.Select(type => type.TensorType).ToArray());
        if (tensorCheck is not null)
        {
            return tensorCheck;
        }

        var placement = distributed[SparseExperts.Q.Index].Placement;
        if (distributed.Any(type => type.Placement != placement))
        {
            return new InvalidType("SparseExperts distributed inputs must use one placement.");
        }

        if (distributed.Any(HasPartial))
        {
            return new InvalidType("SparseExperts does not accept partial inputs.");
        }

        var q = distributed[SparseExperts.Q.Index];
        var ids = distributed[SparseExperts.RouterExpertIds.Index];
        var routerWeights = distributed[SparseExperts.RouterExpertWeights.Index];
        var gate = distributed[SparseExperts.MoeExpertGateProjW.Index];
        var down = distributed[SparseExperts.MoeExpertDownProjW.Index];
        var up = distributed[SparseExperts.MoeExpertUpProjW.Index];
        var tokenPolicy = q.AxisPolicies[0];
        var intermediatePolicy = gate.AxisPolicies[1];
        var outputPolicy = down.AxisPolicies[1];
        if (q.AxisPolicies[1] is not SBPBroadCast ||
            ids.AxisPolicies[0] != tokenPolicy || ids.AxisPolicies[1] is not SBPBroadCast ||
            routerWeights.AxisPolicies[0] != tokenPolicy || routerWeights.AxisPolicies[1] is not SBPBroadCast ||
            gate.AxisPolicies[0] is not SBPBroadCast || gate.AxisPolicies[2] is not SBPBroadCast ||
            up.AxisPolicies[0] is not SBPBroadCast || up.AxisPolicies[1] != intermediatePolicy || up.AxisPolicies[2] is not SBPBroadCast ||
            down.AxisPolicies[0] is not SBPBroadCast || down.AxisPolicies[2] != intermediatePolicy)
        {
            return new InvalidType("SparseExperts requires token-aligned router inputs, matching gate/up intermediate sharding, and matching down reduction sharding.");
        }

        foreach (var parameter in new[]
                 {
                     SparseExperts.MoeExpertGateInputScale,
                     SparseExperts.MoeExpertGateProjScale,
                     SparseExperts.MoeExpertDownInputScale,
                     SparseExperts.MoeExpertDownProjScale,
                     SparseExperts.MoeExpertUpInputScale,
                     SparseExperts.MoeExpertUpProjScale,
                 })
        {
            if (distributed[parameter.Index].AxisPolicies.Any(policy => policy is not SBPBroadCast))
            {
                return new InvalidType($"SparseExperts {parameter.Name} must be broadcast because experts are selected dynamically.");
            }
        }

        if (!TryGetRoleAxes(tokenPolicy, placement.Rank, out var tokenAxes) ||
            !TryGetRoleAxes(intermediatePolicy, placement.Rank, out var intermediateAxes) ||
            !TryGetRoleAxes(outputPolicy, placement.Rank, out var outputAxes) ||
            !AreDisjoint(tokenAxes, intermediateAxes, outputAxes))
        {
            return new InvalidType("SparseExperts token, intermediate, and output sharding must own disjoint hierarchy axes and use contiguous splits.");
        }

        return new DistributedType(
            q.TensorType,
            [tokenPolicy, outputPolicy],
            placement,
            intermediateAxes.Length == 0 ? null : SBP.P(intermediateAxes));
    }

    public IValue Visit(IEvaluateContext context, SparseExperts target)
    {
        var qValue = context.GetArgumentValueAsTensor(target, SparseExperts.Q);
        var q = qValue.ToOrtTensor();
        var qLanes = qValue.ElementType is VectorType vector
            ? vector.Lanes.ToArray()
            : Array.Empty<int>();
        if (qLanes.Length != 0)
        {
            q = q.Unpack(qLanes.Length, Enumerable.Repeat(1, qLanes.Length).ToArray());
        }

        var qType = q.DataType;
        q = q.Cast(OrtDataType.Float);
        var selectedExperts = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.RouterExpertIds).AsTensor());
        var routerWeights = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.RouterExpertWeights).AsTensor());

        var moeExpertDownInputScale = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertDownInputScale).AsTensor());
        var moeExpertDownProjW = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertDownProjW).AsTensor());
        var moeExpertDownProjScale = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertDownProjScale).AsTensor());

        var moeExpertGateInputScale = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertGateInputScale).AsTensor());
        var moeExpertGateProjW = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertGateProjW).AsTensor());
        var moeExpertGateProjScale = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertGateProjScale).AsTensor());

        var moeExpertUpInputScale = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertUpInputScale).AsTensor());
        var moeExpertUpProjW = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertUpProjW).AsTensor());
        var moeExpertUpProjScale = GetOrtTensor(context.GetArgumentValue(target, SparseExperts.MoeExpertUpProjScale).AsTensor());

        var hiddenSize = target.HiddenSize;
        var moeIntermediateSize = target.MoEIntermediateSize;
        var numExpert = target.NumExpert;
        var numTopK = target.NumTopK;
        var chunkSize = target.ChunkSize;

        Console.WriteLine($"SparseExperts: hiddenSize={hiddenSize}, moeIntermediateSize={moeIntermediateSize}, numExpert={numExpert}, numTopK={numTopK}, chunkSize={chunkSize}");

        Console.WriteLine($"SparseExperts: q.shape={string.Join(" ", q.Shape)}, selectedExperts.shape={string.Join(" ", selectedExperts.Shape)}, " +
            $"moeExpertDownInputScale.shape={string.Join(" ", moeExpertDownInputScale.Shape)}, moeExpertDownProjW.shape={string.Join(" ", moeExpertDownProjW.Shape)}, " +
            $"moeExpertDownProjScale.shape={string.Join(" ", moeExpertDownProjScale.Shape)}, " +
            $"moeExpertGateInputScale.shape={string.Join(" ", moeExpertGateInputScale.Shape)}, moeExpertGateProjW.shape={string.Join(" ", moeExpertGateProjW.Shape)}, " +
            $"moeExpertGateProjScale.shape={string.Join(" ", moeExpertGateProjScale.Shape)}, " +
            $"moeExpertUpInputScale.shape={string.Join(" ", moeExpertUpInputScale.Shape)}, moeExpertUpProjW.shape={string.Join(" ", moeExpertUpProjW.Shape)}, " +
            $"moeExpertUpProjScale.shape={string.Join(" ", moeExpertUpProjScale.Shape)}");

        // var (seqLen, hiddenDim) = (q.Shape[0], q.Shape[1]);
        var seqLen = q.Shape[0];

        routerWeights = OrtKI.Cast(routerWeights, (long)q.DataType);

        var finalHiddenStates = OrtKISharp.Tensor.MakeTensor(
            Enumerable.Range(0, (int)(seqLen * hiddenSize)).Select(i => 0).ToArray());

        finalHiddenStates = OrtKI.Reshape(finalHiddenStates, OrtKISharp.Tensor.MakeTensor(new[] { seqLen, hiddenSize }), 0L);
        finalHiddenStates = OrtKI.Cast(finalHiddenStates, (long)q.DataType);

        var expertMask = OrtKI.OneHot(selectedExperts, numExpert, Tensor.From(new[] { 0L, 1L }).ToOrtTensor(), -1L);
        expertMask = OrtKI.Cast(expertMask, (long)q.DataType);
        expertMask = OrtKI.Transpose(expertMask, [2L, 1L, 0L]); // [num_experts, topk, seq_length]

        for (var expertIndex = 0L; expertIndex < numExpert; expertIndex++)
        {
            var singleExpertMask = OrtKI.Slice(expertMask, new[] { expertIndex }, new[] { expertIndex + 1L }, new[] { 0L }, new[] { 1L }); // [num_experts -> 1, topk, seq_length]
            singleExpertMask = OrtKI.Squeeze(singleExpertMask, new[] { 0L }); // [topk, seq_length]
            var nonZero = OrtKI.NonZero(singleExpertMask).ToArray<long>();
            var idx = nonZero[..(nonZero.Length / 2)];
            var topX = nonZero[(nonZero.Length / 2)..];
            if (nonZero.Length == 0)
            {
                continue; // 没有被选中的专家
            }

            var currentState = OrtKI.Gather(q, topX, 0);

            // prepare expertMaskReduceSum
            var expertMaskReduceSum = OrtKI.ReduceSum(singleExpertMask, Tensor.FromArray(new[] { 0L, 1L }).ToOrtTensor(), keepdims: 0L, 0L);

            // // prepare q
            // var qExpand = OrtKI.Unsqueeze(currentState, new[] { 0L });

            // prepare gate matmul
            var gateInputScale = SliceAndSqueeze(moeExpertGateInputScale, expertIndex);
            var gateProjW = SliceAndSqueeze(moeExpertGateProjW, expertIndex);
            var gateProjScale = SliceAndSqueeze(moeExpertGateProjScale, expertIndex);

            // prepare up matmul
            var upInputScale = SliceAndSqueeze(moeExpertUpInputScale, expertIndex);
            var upProjW = SliceAndSqueeze(moeExpertUpProjW, expertIndex);
            var upProjScale = SliceAndSqueeze(moeExpertUpProjScale, expertIndex);

            // prepare down matmul
            var downInputScale = SliceAndSqueeze(moeExpertDownInputScale, expertIndex);
            var downProjW = SliceAndSqueeze(moeExpertDownProjW, expertIndex);
            var downProjScale = SliceAndSqueeze(moeExpertDownProjScale, expertIndex);

            // MLP
            var expertOutput = MLP(currentState, gateInputScale, gateProjW, gateProjScale, upInputScale, upProjW, upProjScale, downInputScale, downProjW, downProjScale, hiddenSize, moeIntermediateSize);

            var weightsForSeq = OrtKI.Gather(routerWeights, Tensor.FromArray(topX).ToOrtTensor(), 0L); // [N, topk]

            var idx2D = OrtKI.Unsqueeze(Tensor.FromArray(idx).ToOrtTensor(), new[] { -1L }); // [N,1]
            var selectedWeights = OrtKI.GatherElements(weightsForSeq, idx2D, 1L); // [N,1]

            expertOutput = OrtKI.Mul(expertOutput, selectedWeights); // [N, hidden]

            var updates = OrtKI.Cast(expertOutput, (long)q.DataType); // [N, hidden]
            var idxCol = OrtKI.Unsqueeze(Tensor.FromArray(topX).ToOrtTensor(), new[] { -1L });        // [N, 1]

            var indices = OrtKI.Tile(idxCol, Tensor.FromArray(new[] { 1L, hiddenSize }).ToOrtTensor()); // [N, hidden]

            finalHiddenStates = OrtKI.ScatterElements(finalHiddenStates, indices, updates, 0L, "add");
        }

        finalHiddenStates = OrtKI.Cast(finalHiddenStates, (long)qType);

        if (qLanes.Length == 0)
        {
            return finalHiddenStates.ToValue();
        }

        return finalHiddenStates
            .Pack(0, qLanes, Enumerable.Repeat(1, qLanes.Length).ToArray())
            .ToValue(qValue.ElementType);
    }

    private OrtKISharp.Tensor GetOrtTensor(Tensor tensor)
    {
        return IR.F.Tensors.Cast(tensor, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
    }

    private OrtKISharp.Tensor SliceAndSqueeze(OrtKISharp.Tensor tensor, long index)
    {
        // Slices the tensor at the specified index and squeezes the first dimension (expert_batch: 1).
        var slicedTensor = OrtKI.Slice(tensor, new[] { index }, new[] { index + 1L }, new[] { 0L }, new[] { 1L });
        return OrtKI.Squeeze(slicedTensor, new[] { 0L });
    }

    private OrtKISharp.Tensor MLP(OrtKISharp.Tensor q, OrtKISharp.Tensor? gateInputScale, OrtKISharp.Tensor gateProjW, OrtKISharp.Tensor gateProjScale, OrtKISharp.Tensor? upInputScale, OrtKISharp.Tensor upProjW, OrtKISharp.Tensor upProjScale, OrtKISharp.Tensor? downInputScale, OrtKISharp.Tensor downProjW, OrtKISharp.Tensor downProjScale, long hiddenSize, long moeIntermediateSize)
    {
        // gate_proj(q)
        // q: [seq_len, hidden_size]
        // gateInputScale: null or [1]
        // gateW: [hidden_size, moe_intermediate_size]
        // gateWScale: [moe_intermediate_size, 1] or [1]
        var gateStates = Matmul(q, gateInputScale, gateProjW, gateProjScale); // [seq_len, moe_intermediate_size]

        // silu(gate)
        var gateType = gateStates.DataType;
        gateStates = OrtKI.Cast(gateStates, (long)OrtDataType.Float);
        gateStates = OrtKI.Sigmoid(gateStates) * gateStates; // [seq_len, moe_intermediate_size]
        gateStates = OrtKI.Cast(gateStates, (long)gateType);

        // up_proj(q)
        // upW: [hidden_size, moe_intermediate_size]
        var upStates = Matmul(q, upInputScale, upProjW, upProjScale); // [seq_len, moe_intermediate_size]

        // silu(gate(q)) * up(q)
        var downInput = OrtKI.Mul(gateStates, upStates); // [seq_len, moe_intermediate_size]

        // Down(silu(gate(q)) * up(q))
        var downStates = Matmul(downInput, downInputScale, downProjW, downProjScale); // [seq_len, moe_intermediate_size]
        return downStates;
    }

    private OrtKISharp.Tensor Matmul(OrtKISharp.Tensor q, OrtKISharp.Tensor? inputScale, OrtKISharp.Tensor projW, OrtKISharp.Tensor projScale)
    {
        if (inputScale != null)
        {
            q = OrtKI.Div(q, inputScale.Cast(OrtDataType.Float));
        }

        // gateProjScale = OrtKI.Reshape(gateProjScale, OrtKISharp.Tensor.MakeTensor(new[] { 1L, moeIntermediateSize, 1L }), 0L);
        // gateProjW = OrtKI.Mul(gateProjW, gateProjScale);
        var states = OrtKI.Einsum(new[] { q.Cast(OrtDataType.Float), projW.Cast(OrtDataType.Float) }, "hs,ds->hd");
        if (projScale.Rank == 1)
        {
            states = OrtKI.Mul(states, projScale.Cast(OrtDataType.Float)); // [seq_len,     moe_intermediate_size]
        }
        else
        {
            var scale = OrtKI.Transpose(projScale, new[] { 1L, 0L }); // [1, moe_intermediate_size]
            states = OrtKI.Mul(states, scale.Cast(OrtDataType.Float)); // [seq_len, moe_intermediate_size]
        }

        if (inputScale != null)
        {
            states = OrtKI.Mul(states, inputScale.Cast(OrtDataType.Float));
        }

        states = states.Cast(q.DataType); // [seq_len, moe_intermediate_size]
        return states;
    }

    private static InvalidType? ValidateTensorContract(SparseExperts target, IReadOnlyList<TensorType> arguments)
    {
        var q = GetLogicalActivationType(arguments[SparseExperts.Q.Index]);
        var checks = new (ParameterInfo Parameter, long[] Shape)[]
        {
            (SparseExperts.Q, [target.ChunkSize, target.HiddenSize]),
            (SparseExperts.RouterExpertIds, [target.ChunkSize, target.NumTopK]),
            (SparseExperts.RouterExpertWeights, [target.ChunkSize, target.NumTopK]),
            (SparseExperts.MoeExpertGateInputScale, [target.NumExpert, 1]),
            (SparseExperts.MoeExpertGateProjW, [target.NumExpert, target.MoEIntermediateSize, target.HiddenSize]),
            (SparseExperts.MoeExpertGateProjScale, [target.NumExpert, 1]),
            (SparseExperts.MoeExpertDownInputScale, [target.NumExpert, 1]),
            (SparseExperts.MoeExpertDownProjW, [target.NumExpert, target.HiddenSize, target.MoEIntermediateSize]),
            (SparseExperts.MoeExpertDownProjScale, [target.NumExpert, 1]),
            (SparseExperts.MoeExpertUpInputScale, [target.NumExpert, 1]),
            (SparseExperts.MoeExpertUpProjW, [target.NumExpert, target.MoEIntermediateSize, target.HiddenSize]),
            (SparseExperts.MoeExpertUpProjScale, [target.NumExpert, 1]),
        };
        foreach (var (parameter, expectedShape) in checks)
        {
            var type = parameter == SparseExperts.Q ? q : arguments[parameter.Index];
            if (type.Shape is not RankedShape shape || shape.Rank != expectedShape.Length)
            {
                return new InvalidType($"SparseExperts {parameter.Name} must have rank {expectedShape.Length}, got {type.Shape}.");
            }

            for (var axis = 0; axis < expectedShape.Length; axis++)
            {
                if (shape[axis].IsFixed && shape[axis].FixedValue != expectedShape[axis])
                {
                    return new InvalidType(
                        $"SparseExperts {parameter.Name} axis {axis} must be {expectedShape[axis]}, got {shape[axis]}.");
                }
            }
        }

        if (arguments[SparseExperts.RouterExpertIds.Index].DType != DataTypes.Int64)
        {
            return new InvalidType(
                $"SparseExperts RouterExpertIds must be int64, got {arguments[SparseExperts.RouterExpertIds.Index].DType}.");
        }

        if (!q.DType.IsFloat())
        {
            return new InvalidType($"SparseExperts q must be floating point, got {q.DType}.");
        }

        return null;
    }

    private static TensorType GetLocalTensorType(IRType type)
        => type switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => DistributedUtility.GetDividedTensorType(
                distributed,
                DistributedUtility.DivideFlags.MaxShape),
            _ => TensorType.Invalid(DataTypes.Float32),
        };

    private static TensorType GetLogicalActivationType(TensorType type)
    {
        if (type.DType is not VectorType vector)
        {
            return type;
        }

        return TypeInference.UnpackType(
            type,
            Enumerable.Repeat(1, vector.Lanes.Count).ToArray()) switch
        {
            TensorType tensor => tensor,
            IRType invalid => throw new InvalidOperationException(
                $"SparseExperts activation lanes must map to hidden axis 1: {invalid}."),
        };
    }

    private static bool TryGetMaxShape(TensorType type, out long[] shape)
    {
        if (CompilerServices.TryGetMaxShape(type.Shape, out var maxShape) && maxShape is not null)
        {
            shape = maxShape;
            return true;
        }

        shape = Array.Empty<long>();
        return false;
    }

    private static int GetScalarByteCount(DataType dataType)
        => dataType is VectorType vector ? GetScalarByteCount(vector.ElemType) : dataType.SizeInBytes;

    private static bool HasPartial(DistributedType type)
        => type.Partial is not null || type.AxisPolicies.Any(policy => policy is SBPPartial);

    private static bool TryGetRoleAxes(SBP policy, int placementRank, out int[] axes)
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

    private static bool AreDisjoint(params IReadOnlyList<int>[] groups)
        => groups.SelectMany(static group => group).Distinct().Count() == groups.Sum(static group => group.Count);
}
