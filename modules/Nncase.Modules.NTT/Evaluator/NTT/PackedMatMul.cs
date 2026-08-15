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
        var rhs = context.GetArgumentValueAsTensor(target, PackedMatMul.Rhs);
        var scale = context.GetArgumentValue(target, PackedMatMul.Scale);
        var addend = context.GetArgumentValue(target, PackedMatMul.Addend);
        return Evaluate(target, lhs, rhs, scale, addend);
    }

    public static IValue Evaluate(
        PackedMatMul target,
        OrtKISharp.Tensor lhs,
        Tensor rhs,
        IValue scale,
        IValue addend)
    {
        var rhsOrt = rhs.ToOrtTensor();

        if (rhs.ElementType is not VectorType rhsVectorType)
        {
            throw new InvalidOperationException($"PackedMatMul expects a vector RHS, got {rhs.ElementType}.");
        }

        int[] outputLanes;
        switch (target.RhsLayout)
        {
            case PackedMatMulRhsLayout.NMajor when rhsVectorType.Lanes.Count == 2:
                {
                    var rN = rhs.Rank - 2;
                    rhsOrt = rhsOrt.Unpack(rhsVectorType.Lanes.Count, [rN, rN]);
                    var perm = Enumerable.Range(0, rhsOrt.Rank).Select(i => (long)i).ToArray();
                    (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
                    rhsOrt = OrtKI.Transpose(rhsOrt, perm);
                    outputLanes = rhsVectorType.Lanes.ToArray();
                    break;
                }

            case PackedMatMulRhsLayout.KMajor when rhsVectorType.Lanes.Count == 3:
                {
                    var rK = rhs.Rank - 2;
                    var rN = rhs.Rank - 1;
                    rhsOrt = rhsOrt.Unpack(rhsVectorType.Lanes.Count, [rN, rK, rK]);
                    outputLanes = [rhsVectorType.Lanes[0]];
                    break;
                }

            default:
                throw new InvalidOperationException(
                    $"PackedMatMul {target.RhsLayout} RHS has invalid vector lanes [{string.Join(",", rhsVectorType.Lanes)}].");
        }

        var matmul = Math.MatMulEvaluator.InferValue(lhs.DataType.ToDataType(), lhs.ToTensor(), rhsOrt.ToTensor(), target.OutputDataType, scale).AsTensor().ToOrtTensor();
        var cN = matmul.Rank - 1;
        matmul = matmul.Pack(0, outputLanes, Enumerable.Repeat(cN, outputLanes.Length).ToArray());
        if (!IsNone(addend))
        {
            matmul = OrtKI.Add(matmul, addend.AsTensor().ToOrtTensor());
        }

        return matmul.ToValue(new VectorType(target.OutputDataType, outputLanes));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Rhs);
        var scale = context.CheckArgumentType<IRType>(target, PackedMatMul.Scale);
        var addend = context.CheckArgumentType<IRType>(target, PackedMatMul.Addend);
        return InferType(target, lhs, rhs, scale, addend);
    }

    public static IRType InferType(
        PackedMatMul target,
        IRType lhs,
        IRType rhs,
        IRType scale,
        IRType addend)
    {
        IRType rType;
        string? errorMessage = null;
        switch (lhs, rhs)
        {
            case (DistributedType a, DistributedType b):
                {
                    if (b.TensorType.DType is not VectorType bVectorType ||
                        !TryGetLayoutInfo(target.RhsLayout, bVectorType, b.TensorType.Shape.Rank, out var rhsUnpackAxes, out var outputLanes, out var transposeB, out errorMessage))
                    {
                        return new InvalidType(errorMessage ?? $"PackedMatMul expects a vector RHS, got {b.TensorType.DType}.");
                    }

                    var unpackedBType = UnpackType(b, rhsUnpackAxes);
                    if (unpackedBType is not DistributedType unpackedB)
                    {
                        return unpackedBType;
                    }

                    var dimInfo = VectorizedMatMul.GetDimInfo(false, transposeB, a.TensorType.Shape.Rank, unpackedB.TensorType.Shape.Rank);
                    if (a.AxisPolicies[dimInfo.Lk] != unpackedB.AxisPolicies[dimInfo.Rk])
                    {
                        return new InvalidType(
                            "PackedMatMul requires lhs and rhs reduction axes to use the same " +
                            $"distributed policy, got lhs={a.AxisPolicies[dimInfo.Lk]} and " +
                            $"rhs={unpackedB.AxisPolicies[dimInfo.Rk]}.");
                    }

                    rType = Math.MatMulEvaluator.VisitDistributedType(a, unpackedB, scale, dimInfo: dimInfo, transB: transposeB, outputDataType: target.OutputDataType);
                    if (rType is not DistributedType drType)
                    {
                        return rType;
                    }

                    if (target.FusedReduce)
                    {
                        drType = (DistributedType)Math.MatMulEvaluator.ConvertPartialToBroadcast(drType);
                    }

                    rType = PackType(drType, outputLanes, Enumerable.Repeat(drType.TensorType.Shape.Rank - 1, outputLanes.Length).ToArray());
                }

                break;
            case (TensorType a, TensorType b):
                {
                    if (b.DType is not VectorType bVectorType ||
                        !TryGetLayoutInfo(target.RhsLayout, bVectorType, b.Shape.Rank, out var rhsUnpackAxes, out var outputLanes, out var transposeB, out errorMessage))
                    {
                        return new InvalidType(errorMessage ?? $"PackedMatMul expects a vector RHS, got {b.DType}.");
                    }

                    var unpackedBType = UnpackType(b, rhsUnpackAxes);
                    if (unpackedBType is not TensorType unpackedB)
                    {
                        return unpackedBType;
                    }

                    var dimInfo = VectorizedMatMul.GetDimInfo(false, transposeB, a.Shape.Rank, unpackedB.Shape.Rank);
                    rType = Math.MatMulEvaluator.VisitTensorType(a, unpackedB, scale, dimInfo: dimInfo, outputDataType: target.OutputDataType);
                    if (rType is TensorType outputType)
                    {
                        rType = TypeInference.PackType(outputType, outputLanes, Enumerable.Repeat(outputType.Shape.Rank - 1, outputLanes.Length).ToArray());
                    }
                }

                break;
            default:
                rType = new InvalidType($"lhs: {lhs}, rhs: {rhs}, in {target.DisplayProperty()} not support: {errorMessage}");
                break;
        }

        if (rType is InvalidType || addend is NoneType)
        {
            return rType;
        }

        return Equals(rType, addend)
            ? rType
            : new InvalidType(
                $"PackedMatMul addend must have exactly the packed output type, got output={rType}, addend={addend}.");
    }

    public Cost Visit(ICostEvaluateContext context, PackedMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedMatMul.Rhs);
        var addend = context.GetArgumentType<IRType>(target, PackedMatMul.Addend);
        var outputType = context.GetReturnType<IRType>();
        bool hasAllReduce = false;
        if (TryGetTargetCost(context, target, lhs, rhs, outputType, out var targetCost, out hasAllReduce))
        {
            AddAddendCost(targetCost, outputType, addend);
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
            hasAllReduce = target.FusedReduce && distributedType.AxisPolicies[^1] is SBPSplit;
        }

        var cost = new Cost()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                outputType,
                checked(macPerElement + (addend is NoneType ? 0U : 1U))),
        };

        if (addend is not NoneType)
        {
            AddCostFactor(
                cost,
                CostFactorNames.BlockLocalMemoryLoadBytes,
                CostUtility.GetMemoryAccess(addend));
        }

        return AddAllReduceCost(cost, outputType, hasAllReduce);
    }

    internal static bool TryGetLayoutInfo(
        PackedMatMulRhsLayout layout,
        VectorType vectorType,
        int rhsRank,
        out int[] rhsUnpackAxes,
        out int[] outputLanes,
        out bool transposeB,
        out string? errorMessage)
    {
        switch (layout, vectorType.Lanes.Count)
        {
            case (PackedMatMulRhsLayout.NMajor, 2):
                rhsUnpackAxes = [rhsRank - 2, rhsRank - 2];
                outputLanes = vectorType.Lanes.ToArray();
                transposeB = true;
                errorMessage = null;
                return true;
            case (PackedMatMulRhsLayout.KMajor, 3):
                rhsUnpackAxes = [rhsRank - 1, rhsRank - 2, rhsRank - 2];
                outputLanes = [vectorType.Lanes[0]];
                transposeB = false;
                errorMessage = null;
                return true;
            default:
                rhsUnpackAxes = [];
                outputLanes = [];
                transposeB = false;
                errorMessage = $"PackedMatMul {layout} expects {(layout == PackedMatMulRhsLayout.NMajor ? 2 : 3)} RHS vector lanes, got [{string.Join(",", vectorType.Lanes)}].";
                return false;
        }
    }

    private bool TryGetTargetCost(ICostEvaluateContext context, PackedMatMul target, IRType lhs, IRType rhs, IRType outputType, out Cost cost, out bool hasAllReduce)
    {
        hasAllReduce = target.FusedReduce &&
            lhs is DistributedType distributedType &&
            distributedType.AxisPolicies[^1] is SBPSplit;
        if (target.RhsLayout == PackedMatMulRhsLayout.KMajor)
        {
            if (GetTensorType(rhs)?.DType is not VectorType rhsVectorType ||
                !TryGetLayoutInfo(target.RhsLayout, rhsVectorType, GetTensorType(rhs)!.Shape.Rank, out var rhsUnpackAxes, out _, out _, out _) ||
                UnpackType(rhs, rhsUnpackAxes) is not { } logicalRhs ||
                GetTensorType(outputType)?.DType is not VectorType ||
                UnpackType(outputType, [GetTensorType(outputType)!.Shape.Rank - 1]) is not { } logicalOutput)
            {
                cost = Cost.Zero;
                return false;
            }

            rhs = logicalRhs;
            outputType = logicalOutput;
        }

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

    private static TensorType? GetTensorType(IRType type) => type switch
    {
        TensorType tensorType => tensorType,
        DistributedType distributedType => distributedType.TensorType,
        _ => null,
    };

    private static bool IsNone(IValue value) => value is NoneValue || value.Type is NoneType;

    private static void AddAddendCost(Cost cost, IRType outputType, IRType addend)
    {
        if (addend is NoneType)
        {
            return;
        }

        AddCostFactor(
            cost,
            CostFactorNames.BlockLocalMemoryLoadBytes,
            CostUtility.GetMemoryAccess(addend));
        AddCostFactor(
            cost,
            CostFactorNames.CPUCycles,
            CostUtility.GetCPUCycles(outputType, 1));
    }

    private static IRType UnpackType(IRType input, int[] axes) => input switch
    {
        DistributedType distributedType => TypeInference.UnpackType(distributedType, axes),
        TensorType tensorType => TypeInference.UnpackType(tensorType, axes),
        _ => new InvalidType($"Cannot unpack {input} with axes [{string.Join(",", axes)}]."),
    };

    private static IRType PackType(IRType input, int[] lanes, int[] axes) => input switch
    {
        DistributedType distributedType => TypeInference.PackType(distributedType, lanes, axes),
        TensorType tensorType => TypeInference.PackType(tensorType, lanes, axes),
        _ => new InvalidType($"Cannot pack {input}."),
    };

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

    private static void AddCostFactor(Cost cost, string name, UInt128 value)
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
