// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using MatMul = Nncase.IR.Math.MatMul;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="MatMul"/>.
/// </summary>
public class MatMulEvaluator : IEvaluator<MatMul>, ITypeInferencer<MatMul>, ICostEvaluator<MatMul>, IMetricEvaluator<MatMul>
{
    public static IRType VisitDistributedType(DistributedType a, DistributedType b, IRType scale, bool vectorizeK = false, MatMulDimInfo? dimInfo = null, bool transB = false, DataType outputDataType = null!)
    {
        if (VisitTensorType(a.TensorType, b.TensorType, scale, vectorizeK, dimInfo, outputDataType) is not TensorType outType)
        {
            return new InvalidType($"{a.TensorType} {b.TensorType} not support");
        }

        if (a.Placement != b.Placement)
        {
            return new InvalidType("placement not equal");
        }

        var aRank = a.TensorType.Shape.Rank;
        var bRank = b.TensorType.Shape.Rank;
        var oRank = outType.Shape.Rank;
        var aPad = oRank - aRank;
        var bPad = oRank - bRank;
        var (lm, lk, rk, rn) = dimInfo ?? new(aRank - 2, aRank - 1, bRank - 2, bRank - 1);
        var aMaxShape = CompilerServices.GetMaxShape(a.TensorType.Shape);
        var bMaxShape = CompilerServices.GetMaxShape(b.TensorType.Shape);
        SBPPartial? partial = null;

        // TODO: keep summa only
        if (!a.TensorType.Shape.IsFixed || !b.TensorType.Shape.IsFixed || transB)
        {
            var ndsbp = new SBP[oRank];
            for (int i = 0; i < ndsbp.Length - 2; i++)
            {
                var policyA = i < aPad ? null : a.AxisPolicies[i - aPad];
                var policyB = i < bPad ? null : b.AxisPolicies[i - bPad];
                var invalid = new InvalidType($"({policyA}, {policyB}) not support");
                switch (policyA, policyB)
                {
                    case (null, _):
                        ndsbp[i] = policyB!;
                        break;
                    case (_, null):
                        ndsbp[i] = policyA!;
                        break;
                    case (SBPSplit sa, SBPSplit sb):
                        if (!DistributedUtility.IsSamePolicy(sa, sb, checkGranularity: false))
                        {
                            return invalid;
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPSplit sa, SBPBroadCast):
                        // invalid (S, B) if B is not broacast
                        if (i < lm + aPad && bMaxShape[i - bPad] != 1)
                        {
                            return invalid;
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPBroadCast, SBPSplit sb):
                        // invalid (B, S) if A is not broacast
                        if (i < rk + bPad && aMaxShape[i - aPad] != 1)
                        {
                            return invalid;
                        }

                        ndsbp[i] = sb;
                        break;
                    case (SBPBroadCast, SBPBroadCast):
                        ndsbp[i] = SBP.B;
                        break;
                    default:
                        return invalid;
                }
            }

            if (a.AxisPolicies[lk] != b.AxisPolicies[rk])
            {
                return new InvalidType($"not support different policy on k: {a.AxisPolicies[lk]} vs {b.AxisPolicies[rk]}");
            }

            ndsbp[oRank - 2] = a.AxisPolicies[lm];
            ndsbp[oRank - 1] = b.AxisPolicies[rn];

            if (a.AxisPolicies[lk] is SBPSplit sk && b.AxisPolicies[rk] is SBPSplit)
            {
                ndsbp[oRank - 2] = ndsbp[oRank - 2];
                ndsbp[oRank - 1] = ndsbp[oRank - 1];
                partial = SBP.P(sk.HierarchyAxes);
            }

            if (!DistributedUtility.IsDistributable(outType, ndsbp, a.Placement))
            {
                return new InvalidType("no valid sbp.");
            }

            return new DistributedType(outType, ndsbp, a.Placement, Partial: partial);
        }
        else
        {
            var ndsbp = new SBP[oRank];
            var lhsReductionPolicy = a.AxisPolicies[lk];
            var rhsReductionPolicy = b.AxisPolicies[rk];
            if (lhsReductionPolicy is SBPSplit || rhsReductionPolicy is SBPSplit)
            {
                var isSumma = false;
                if (a.Placement.Rank >= 2)
                {
                    var (lmMeshAxis, lkMeshAxis) = (a.Placement.Rank - 2, a.Placement.Rank - 1);

                    // TODO: support SUMMA split on more than two mesh axes.
                    isSumma = a.AxisPolicies[lm] is SBPSplit slm
                        && lhsReductionPolicy is SBPSplit slk
                        && rhsReductionPolicy is SBPSplit srk
                        && b.AxisPolicies[rn] is SBPSplit srn
                        && slm.HierarchyAxes.Count == 1
                        && slk.HierarchyAxes.Count == 1
                        && srk.HierarchyAxes.Count == 1
                        && srn.HierarchyAxes.Count == 1
                        && slm.HierarchyAxes[0] == srk.HierarchyAxes[0]
                        && slk.HierarchyAxes[0] == srn.HierarchyAxes[0]
                        && slm.HierarchyAxes[0] == lmMeshAxis
                        && slk.HierarchyAxes[0] == lkMeshAxis;
                }

                if (!isSumma)
                {
                    if (lhsReductionPolicy != rhsReductionPolicy || lhsReductionPolicy is not SBPSplit split)
                    {
                        return new InvalidType(
                            $"not support different policy on k: {lhsReductionPolicy} vs {rhsReductionPolicy}");
                    }

                    partial = SBP.P(split.HierarchyAxes);
                }
            }

            ndsbp[oRank - 2] = a.AxisPolicies[lm];
            ndsbp[oRank - 1] = b.AxisPolicies[rn];

            for (int i = 0; i < ndsbp.Length - 2; i++)
            {
                var policyA = i < aPad ? null : a.AxisPolicies[i - aPad];
                var policyB = i < bPad ? null : b.AxisPolicies[i - bPad];
                switch (policyA, policyB)
                {
                    case (null, _):
                        ndsbp[i] = policyB!;
                        break;
                    case (_, null):
                        ndsbp[i] = policyA!;
                        break;
                    case (SBPSplit sa, SBPSplit sb):
                        if (!DistributedUtility.IsSamePolicy(sa, sb, checkGranularity: false))
                        {
                            return new InvalidType($"lhs rhs sbp at {i} not equal");
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPSplit sa, SBPBroadCast):
                        // invalid (S, B) if B is not broacast
                        if (b.TensorType.Shape[i - bPad] != 1)
                        {
                            return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPBroadCast, SBPSplit sb):
                        // invalid (B, S) if A is not broacast
                        if (a.TensorType.Shape[i - aPad] != 1)
                        {
                            return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                        }

                        ndsbp[i] = sb;
                        break;
                    case (SBPBroadCast, SBPBroadCast):
                        ndsbp[i] = SBP.B;
                        break;
                    default:
                        return new InvalidType("not support binary sbp.");
                }
            }

            if (DistributedUtility.IsDistributable(outType, ndsbp, a.Placement))
            {
                return new DistributedType(outType, ndsbp, a.Placement, partial);
            }

            return new InvalidType("no valid sbp.");
        }
    }

    public static IRType ConvertPartialToBroadcast(DistributedType a)
    {
        var ndsbp = a.AxisPolicies.Select(x => x is SBPPartial ? SBP.B : x).ToArray();
        return new DistributedType(a.TensorType, ndsbp, a.Placement);
    }

    public static IRType VisitTensorType(TensorType lhs, TensorType rhs, IRType scale, bool vectorizeK = false, MatMulDimInfo? dimInfo = null, DataType outputDataType = null!)
    {
        if (lhs.Shape is not RankedShape lhsShape
            || rhs.Shape is not RankedShape rhsShape)
        {
            return new TensorType(lhs.DType, Shape.Unranked);
        }

        var (lm, lk, rk, rn) = dimInfo ?? new(lhsShape.Rank - 2, lhsShape.Rank - 1, rhsShape.Rank - 2, rhsShape.Rank - 1);
        DataType dtype = lhs.DType;
        DataType lhsDType = lhs.DType is VectorType l ? l.ElemType : lhs.DType;
        DataType rhsDType = rhs.DType is VectorType r ? r.ElemType : rhs.DType;
        if (lhsDType != rhsDType)
        {
            return new InvalidType("MatMul lhs and rhs have different DType");
        }

        if (lhsShape[lk] != rhsShape[rk] && lhsShape[lk].IsFixed && rhsShape[rk].IsFixed)
        {
            return new InvalidType("MatMul lhs and rhs have not compatiable shape");
        }

        if (lhsDType == DataTypes.Float8E4M3 || lhsDType == DataTypes.Float8E5M2 || lhsDType == DataTypes.Int8)
        {
            dtype = outputDataType;
        }

        if (scale is TensorType scaleType)
        {
            if (!scaleType.IsScalar || (scaleType.DType != DataTypes.Float32) || (lhsDType.SizeInBytes != 1))
            {
                return new InvalidType("Scale should be a float32 scalar when lhs or rhs is float8/int8.");
            }
        }

        if (lhs.DType is VectorType vl1 && rhs.DType is not VectorType)
        {
            if (vl1.Lanes.Count != 1)
            {
                return new InvalidType("Only vectorize m is supported when rhs is not vector type.");
            }

            dtype = vl1;
        }
        else if (lhs.DType is not VectorType && rhs.DType is VectorType vr1)
        {
            if (vr1.Lanes.Count != 1)
            {
                return new InvalidType("Only vectorize n is supported when lhs is not vector type.");
            }

            dtype = vr1;
            if (vr1.ElemType == DataTypes.Float8E4M3)
            {
                var interType = new VectorType(outputDataType, vr1.Lanes);
                dtype = interType;
            }
        }
        else if (lhs.DType is VectorType vl && rhs.DType is VectorType vr)
        {
            // vectorize k or m&n
            var elemType = vl.ElemType;
            if (elemType.IsFloat() && elemType != outputDataType)
            {
                elemType = outputDataType;
            }

            if (vl.Lanes.Count == 1 && vr.Lanes.Count == 1)
            {
                dtype = vectorizeK ? elemType : new VectorType(elemType, vl.Lanes[0], vr.Lanes[0]);
            }
            else if (vl.Lanes.Count == 1 && vr.Lanes.Count == 2)
            {
                dtype = new VectorType(elemType, vl.Lanes[0] == vr.Lanes[0] ? vr.Lanes[1] : vr.Lanes[0]);
            }
            else if (vl.Lanes.Count == 2 && vr.Lanes.Count == 1)
            {
                dtype = new VectorType(elemType, vl.Lanes[0]);
            }
            else if (vl.Lanes.Count == 2 && vr.Lanes.Count == 2)
            {
                // TODO: only support transpose vector B for now
                if (lhsDType == DataTypes.Float16 && vl.Lanes[0] == 64 && vr.Lanes[1] == 64)
                {
                    elemType = lhsDType;
                }

                dtype = new VectorType(elemType, vl.Lanes[0], vl.Lanes[1] == vr.Lanes[0] ? vr.Lanes[1] : vr.Lanes[0]);
            }
            else
            {
                return new InvalidType("Not supported vectorize.");
            }
        }

        lhsShape = lhsShape.Rank >= rhsShape.Rank ? lhsShape.ToArray() : Enumerable.Repeat((Dimension)1, rhsShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        rhsShape = lhsShape.Rank <= rhsShape.Rank ? rhsShape.ToArray() : Enumerable.Repeat((Dimension)1, lhsShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();

        var bigShape = Enumerable.Zip(lhsShape, rhsShape).SkipLast(2).Select(t => Dimension.Max(t.First, t.Second)).ToArray();

        // batch and channel
        var front = bigShape;

        // currently the output keep the m,n.
        var end = new[] { lhs.Shape[lm], rhs.Shape[rn] };
        return new TensorType(dtype, front.Concat(end).ToArray());
    }

    public static IValue InferValue(DataType dataType, Tensor lhs, Tensor rhs, DataType outputDataType = null!, IValue scale = null!)
    {
        IValue result;
        if (dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            var lhsOrt = lhs.CastElementTo(DataTypes.Float32).ToOrtTensor();
            var rhsOrt = rhs.CastElementTo(DataTypes.Float32).ToOrtTensor();
            var matmul = OrtKI.MatMul(lhsOrt, rhsOrt);
            if (scale is TensorValue)
            {
                matmul = OrtKI.Mul(matmul, scale.AsTensor().ToOrtTensor());
            }

            var ret = matmul.ToTensor().CastElementTo(outputDataType);
            result = Value.FromTensor(ret);
        }
        else
        {
            var input = lhs.ToOrtTensor();
            var other = rhs.ToOrtTensor();
            var matmul = OrtKI.MatMul(input, other);
            if (scale is TensorValue)
            {
                matmul = OrtKI.Mul(matmul, scale.AsTensor().ToOrtTensor());
            }

            result = Value.FromTensor(matmul.ToTensor());
        }

        return result;
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, MatMul matMul)
    {
        var dataType = context.CurrentCall.Arguments[MatMul.Lhs.Index].CheckedDataType;
        var lhs = context.GetArgumentValue(matMul, MatMul.Lhs).AsTensor();
        var rhs = context.GetArgumentValue(matMul, MatMul.Rhs).AsTensor();
        var scale = context.GetArgumentValue(matMul, MatMul.Scale);
        var outputDataType = matMul.OutputDataType;
        return InferValue(dataType, lhs, rhs, outputDataType, scale);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, MatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, MatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, MatMul.Rhs);
        var scale = context.CheckArgumentType<IRType>(target, MatMul.Scale);
        var outputDataType = target.OutputDataType;
        return (lhs, rhs) switch
        {
            (DistributedType a, DistributedType b) => VisitDistributedType(a, b, scale, false, null, false, outputDataType),
            (TensorType a, TensorType b) => VisitTensorType(a, b, scale, false, null, outputDataType),
            _ => new InvalidType($"{lhs} {rhs} not support"),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, MatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, MatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, MatMul.Rhs);
        var outputType = context.GetReturnType<IRType>();
        if (TargetCostTensor.TryFromType(lhs, out var lhsTensor)
            && TargetCostTensor.TryFromType(rhs, out var rhsTensor)
            && TargetCostTensor.TryFromType(outputType, out var outputTensor)
            && context.TargetCostModel.TryGetMatMulCost(new(lhsTensor, rhsTensor, outputTensor, target.OutputDataType), out var targetCost))
        {
            return targetCost;
        }

        uint macPerElement = 1;
        if (lhs is TensorType { Shape: RankedShape lhsShape })
        {
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
            lhsShape = (RankedShape)lhsType.Shape;
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
        }

        return new()
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.BlockLocalMemoryStoreBytes] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, MatMul target)
    {
        var lhs = context.GetArgumentType<TensorType>(target, MatMul.Lhs);
        var rhs = context.GetArgumentType<TensorType>(target, MatMul.Rhs);
        var outputType = context.GetReturnType<TensorType>();
        var lhsShape = (RankedShape)lhs.Shape;
        var k = (UInt128)lhsShape[^1].FixedValue;
        var m = MetricUtility.GetFLOPs(lhs) / k;
        var n = MetricUtility.GetFLOPs(rhs) / k;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = m * n * ((2 * k) - 1),
            [MetricFactorNames.Parallel] = 4,
        };
    }
}
