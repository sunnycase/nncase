// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

public sealed class PackedMatMulEvaluator : IEvaluator<PackedMatMul>, ITypeInferencer<PackedMatMul>, ICostEvaluator<PackedMatMul>
{
    public IValue Visit(IEvaluateContext context, PackedMatMul target)
    {
        var lhs = context.GetOrtArgumentValue(target, PackedMatMul.Lhs); // [x, m, k]
        var rhs = context.GetArgumentValueAsTensor(target, PackedMatMul.Rhs); // [x, n/Nr/lanes, k, Nr, lanes]
        var scale = context.GetArgumentValue(target, PackedMatMul.Scale);
        var rhsOrt = rhs.ToOrtTensor();

        var rhsVectorType = (VectorType)rhs.ElementType;
        var nr = rhsVectorType.Lanes[0];
        var nLanes = rhsVectorType.Lanes[1];
        var outRank = context.CurrentCall.CheckedShape.Rank;

        // 1. Unpack B to scalar reference layout.
        var rN = rhs.Rank - 2;
        rhsOrt = rhsOrt.Unpack(rhsVectorType.Lanes.Count, [rN, rN]);

        // 2. Transpose B
        {
            var perm = Enumerable.Range(0, rhsOrt.Rank).Select(i => (long)i).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            rhsOrt = OrtKI.Transpose(rhsOrt, perm);
        }

        var matmul = Math.MatMulEvaluator.InferValue(lhs.DataType.ToDataType(), lhs.ToTensor(), rhsOrt.ToTensor(), target.OutputDataType, scale).AsTensor().ToOrtTensor();
        var cN = matmul.Rank - 1;
        matmul = matmul.Pack(0, [nr, nLanes], [cN, cN]);
        return matmul.ToValue(new VectorType(target.OutputDataType, [nr, nLanes]));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Rhs);
        var scale = context.CheckArgumentType<IRType>(target, PackedMatMul.Scale);
        IRType rType;
        string? errorMessage = null;
        switch (lhs, rhs)
        {
            case (DistributedType a, DistributedType b):
                {
                    var bVectorType = (VectorType)b.TensorType.DType;
                    var nr = bVectorType.Lanes[0];
                    var unpackedB = b with { TensorType = UnpackedBType(b.TensorType) };
                    var dimInfo = VectorizedMatMul.GetDimInfo(false, true, a.TensorType.Shape.Rank, unpackedB.TensorType.Shape.Rank);
                    rType = Math.MatMulEvaluator.VisitDistributedType(a, unpackedB, scale, dimInfo: dimInfo, transB: true, outputDataType: target.OutputDataType);
                    if (rType is not DistributedType drType)
                    {
                        return rType;
                    }

                    if (target.FusedReduce)
                    {
                        drType = (DistributedType)Math.MatMulEvaluator.ConvertPartialToBroadcast(drType);
                    }

                    rType = drType with { TensorType = (TensorType)TypeInference.PackType(drType.TensorType, [nr], [drType.TensorType.Shape.Rank - 1]) };
                }

                break;
            case (TensorType a, TensorType b):
                {
                    var bVectorType = (VectorType)b.DType;
                    var nr = bVectorType.Lanes[0];
                    var unpackedB = UnpackedBType(b);
                    var dimInfo = VectorizedMatMul.GetDimInfo(false, true, a.Shape.Rank, unpackedB.Shape.Rank);
                    rType = Math.MatMulEvaluator.VisitTensorType(a, unpackedB, scale, dimInfo: dimInfo, outputDataType: target.OutputDataType);
                    rType = TypeInference.PackType((TensorType)rType, [nr], [((TensorType)rType).Shape.Rank - 1]);
                }

                break;
            default:
                rType = new InvalidType($"lhs: {lhs}, rhs: {rhs}, in {target.DisplayProperty()} not support: {errorMessage}");
                break;
        }

        return rType;
    }

    public Cost Visit(ICostEvaluateContext context, PackedMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedMatMul.Rhs);
        var outputType = context.GetReturnType<IRType>();
        bool hasAllReduce = false;
        if (TryGetTargetCost(context, target, lhs, rhs, outputType, out var targetCost, out hasAllReduce))
        {
            return AddAllReduceCost(targetCost, outputType, hasAllReduce);
        }

        uint macPerElement = 1;
        if (lhs is TensorType { Shape: Shape lhsShape })
        {
            var k = lhsShape.Rank - 1;
            macPerElement = lhsShape[k].IsFixed ? (uint)lhsShape[k].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
            var k = distributedType.TensorType.Shape.Rank - 1;
            macPerElement = lhsType.Shape[k].IsFixed ? (uint)lhsType.Shape[k].FixedValue : 1U;
            hasAllReduce = distributedType.AxisPolicies[^1] is SBPSplit;
        }

        var cost = new Cost()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };

        return AddAllReduceCost(cost, outputType, hasAllReduce);
    }

    private TensorType UnpackedBType(TensorType tensorType)
    {
        var vectorType = (VectorType)tensorType.DType;
        var nr = vectorType.Lanes[0];
        var nLanes = vectorType.Lanes[1];
        var newShape = tensorType.Shape.ToArray();
        newShape[^2] *= nr;
        return tensorType with { DType = vectorType with { Lanes = [nLanes] }, Shape = newShape };
    }

    private bool TryGetTargetCost(ICostEvaluateContext context, PackedMatMul target, IRType lhs, IRType rhs, IRType outputType, out Cost cost, out bool hasAllReduce)
    {
        hasAllReduce = lhs is DistributedType distributedType && distributedType.AxisPolicies[^1] is SBPSplit;
        if (!TargetCostTensor.TryFromType(lhs, out var lhsTensor)
            || !TargetCostTensor.TryFromType(rhs, out var rhsTensor)
            || !TargetCostTensor.TryFromType(outputType, out var outputTensor)
            || !context.TargetCostModel.TryGetMatMulCost(new(lhsTensor, rhsTensor, outputTensor, GetScalarType(target.OutputDataType), MatMulOpCostKind.Simt), out cost))
        {
            cost = Cost.Zero;
            return false;
        }

        return true;
    }

    private DataType GetScalarType(DataType dtype) => dtype switch
    {
        VectorType vectorType => GetScalarType(vectorType.ElemType),
        _ => dtype,
    };

    private Cost AddAllReduceCost(Cost cost, IRType outputType, bool hasAllReduce)
    {
        if (!hasAllReduce)
        {
            return cost;
        }

        AddCostFactor(cost, CostFactorNames.ChipGlobalMemoryLoadBytes, CostUtility.GetMemoryAccess(outputType) * 2);
        AddCostFactor(cost, CostFactorNames.ChipGlobalMemoryStoreBytes, CostUtility.GetMemoryAccess(outputType));
        AddCostFactor(cost, CostFactorNames.CPUCycles, CostUtility.GetCPUCycles(outputType, 1));
        AddCostFactor(cost, CostFactorNames.GridSynchronization, (UInt128)3);
        return cost;
    }

    private void AddCostFactor(Cost cost, string name, UInt128 value)
    {
        if (cost.Factors.TryGetValue(name, out var oldValue))
        {
            cost.Factors[name] = oldValue + value;
        }
        else
        {
            cost.Factors.Add(name, value);
        }
    }
}
