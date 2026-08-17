// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Collections.Generic;
using System.Linq;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Evaluator.IR.Distributed;

public sealed class BoxingEvaluator : ITypeInferencer<Boxing>, ICostEvaluator<Boxing>, IEvaluator<Boxing>
{
    private static readonly UInt128 SynchronizationEventCount = 1;

    public static IRType VisitType(IRType inType, IRType outType, bool isReshape = false)
    {
        IRType VisitD2D(DistributedType inv, DistributedType outv)
        {
            if (inv.TensorType != outv.TensorType)
            {
                return new InvalidType($"D2D boxing requires the same tensor type, but got {inv.TensorType} -> {outv.TensorType}");
            }

            if (inv.Placement != outv.Placement)
            {
                return new InvalidType($"D2D boxing requires the same placement, but got {inv.Placement} -> {outv.Placement}");
            }

            if (inv == outv)
            {
                return new InvalidType("Same DistributedType");
            }

            if (inv.AxisPolicies.Any(sbp => sbp is SBPPartial) || outv.AxisPolicies.Any(sbp => sbp is SBPPartial))
            {
                return new InvalidType("Not Support Partial in Policeis.");
            }

            var partialDims = new List<int>();
            if (inv.Partial is not null)
            {
                for (int i = 0; i < inv.AxisPolicies.Count; i++)
                {
                    if (inv.AxisPolicies[i] is SBPSplit && outv.AxisPolicies[i] is SBPBroadCast)
                    {
                        return new InvalidType("Not supported input is BroadCast output is Split");
                    }

                    if (outv.AxisPolicies[i] is SBPSplit s)
                    {
                        if (inv.AxisPolicies[i] is SBPSplit splitIn)
                        {
                            if (splitIn.HierarchyAxes.Except(s.HierarchyAxes).Any())
                            {
                                return new InvalidType("Not Supported Split-> Split.");
                            }
                        }

                        if (s.HierarchyAxes.Any(inv.Partial.Axes.Contains))
                        {
                            partialDims.Add(i);
                        }
                    }
                }

                var ndspsIn = DistributedUtility.GetHierarchyAxisPolicies(inv.AxisPolicies, inv.Placement.Rank);
                var ndspsOut = DistributedUtility.GetHierarchyAxisPolicies(outv.AxisPolicies, outv.Placement.Rank);
                if (Enumerable.Range(0, ndspsIn.Count).Any(i =>
                    ndspsIn[i] is HierarchyAxisSplit splitIn &&
                    (ndspsOut[i] is HierarchyAxisBroadcast ||
                     (ndspsOut[i] is HierarchyAxisSplit splitOut && splitOut != splitIn))))
                {
                    return new InvalidType("Not Supported Split-> Broadcast.");
                }
            }

            if (partialDims.Count > 0 && !Enumerable.Range(0, inv.AxisPolicies.Count).Except(partialDims.ToArray()).All(i => DistributedUtility.IsSamePolicy(inv.AxisPolicies[i], outv.AxisPolicies[i])))
            {
                return new InvalidType("Not Supported Partial.");
            }

            return outv;
        }

        IRType VisitD2T(DistributedType inv, TensorType outv)
        {
            if (inv.AxisPolicies.Any(s => s is SBPPartial) || inv.Partial is not null)
            {
                return new InvalidType("Not supported input is Partial output is Unshard");
            }

            return outv;
        }

        IRType VisitT2D(TensorType inv, DistributedType outv)
        {
            if (outv.AxisPolicies.Any(s => s is SBPPartial) || outv.Partial is not null)
            {
                return new InvalidType("Not supported input is Unshard output is Partial");
            }

            return outv;
        }

        return (inType, outType) switch
        {
            (InvalidType inv, _) => inv,
            (_, InvalidType inv) => inv,
            (DistributedType d, DistributedType d1) => VisitD2D(d, d1),
            (TensorType t, DistributedType d) => VisitT2D(t, d),
            (DistributedType d, TensorType t) => VisitD2T(d, t),
            _ => new InvalidType($"not support boxing {inType} to {outType}"),
        };
    }

    public IRType Visit(ITypeInferenceContext context, Boxing target)
    {
        return VisitType(context.GetArgumentType(target, Boxing.Input), target.NewType);
    }

    public Cost Visit(ICostEvaluateContext context, Boxing target)
    {
        var inType = context.GetArgumentType<IRType>(target, Boxing.Input);
        var returnType = context.GetReturnType<IRType>();
        if (TryGetTargetBoxingCost(context.TargetCostModel, inType, returnType, out var targetCost))
        {
            return targetCost;
        }

        var cost = new Cost() { [CostFactorNames.CPUCycles] = 1, [CostFactorNames.ChipGlobalMemoryLoadBytes] = 0, [CostFactorNames.ChipGlobalMemoryStoreBytes] = 0, [CostFactorNames.GridSynchronization] = SynchronizationEventCount };
        switch (inType, returnType)
        {
            case (TensorType _, DistributedType distributedType):
                switch (context.CompileOptions.TargetOptions)
                {
                    default:
                        cost = new Cost()
                        {
                            [CostFactorNames.ChipGlobalMemoryLoadBytes] = CostUtility.GetMemoryAccess(distributedType),
                            [CostFactorNames.ChipGlobalMemoryStoreBytes] = CostUtility.GetMemoryAccess(distributedType),
                        };
                        break;
                }

                break;
            case (DistributedType distributedType, TensorType _):
                switch (context.CompileOptions.TargetOptions)
                {
                    default:
                        cost = new Cost()
                        {
                            [CostFactorNames.ChipGlobalMemoryLoadBytes] = CostUtility.GetMemoryAccess(distributedType),
                            [CostFactorNames.ChipGlobalMemoryStoreBytes] = CostUtility.GetMemoryAccess(distributedType),
                        };
                        break;
                }

                break;

            case (DistributedType a, DistributedType b) when a.TensorType == b.TensorType && a.Placement == b.Placement && a.AxisPolicies != b.AxisPolicies:
                {
                    var fullLoadStore = new Cost()
                    {
                        [CostFactorNames.ChipGlobalMemoryStoreBytes] = CostUtility.GetMemoryAccess(a),
                        [CostFactorNames.ChipGlobalMemoryLoadBytes] = CostUtility.GetMemoryAccess(b),
                        [CostFactorNames.GridSynchronization] = SynchronizationEventCount,
                    };

                    float gatherPart = 1;
                    float scatterPart = 1;
                    var hierarchyPenalty = Enumerable.Range(1, a.Placement.Hierarchy.Count).Reverse().ToArray();
                    for (int i = 0; i < a.AxisPolicies.Count; i++)
                    {
                        switch (a.AxisPolicies[i], b.AxisPolicies[i])
                        {
                            case (SBPSplit splitIn, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPSplit splitOut:
                                        {
                                            var setA = new HashSet<int>(splitIn.HierarchyAxes);
                                            var setB = new HashSet<int>(splitOut.HierarchyAxes);
                                            var aContainsB = setA.IsSupersetOf(setB);
                                            var bContainsA = setB.IsSupersetOf(setA);
                                            if (bContainsA && aContainsB)
                                            {
                                                cost += new Cost()
                                                {
                                                    [CostFactorNames.CPUCycles] = 1,
                                                };
                                            }
                                            else if (bContainsA)
                                            {
                                                var diff = setB.Except(setA).ToArray();
                                                if (diff.All(d => d > splitIn.HierarchyAxes[^1]))
                                                {
                                                    diff.ForEach(s => scatterPart *= hierarchyPenalty[s]);
                                                }
                                                else
                                                {
                                                    return fullLoadStore;
                                                }
                                            }
                                            else if (aContainsB)
                                            {
                                                setA.Except(setB).ToArray().ForEach(s => gatherPart *= hierarchyPenalty[s]);
                                            }
                                            else
                                            {
                                                // when split different axis, need global load store.
                                                return fullLoadStore;
                                            }
                                        }

                                        break;
                                    case SBPBroadCast:
                                        // scatterPart *= a.Placement.Hierarchy[i];
                                        splitIn.HierarchyAxes.ToArray().ForEach(s => gatherPart *= hierarchyPenalty[s]);
                                        break;
                                    default:
                                        throw new NotSupportedException("split to partial");
                                }

                                break;
                            case (SBPBroadCast, SBPBroadCast):
                                // no cost.
                                cost += new Cost()
                                {
                                    [CostFactorNames.CPUCycles] = 1,
                                };
                                break;
                            case (SBPBroadCast, SBPSplit splitOut):
                                splitOut.HierarchyAxes.ToArray().ForEach(s => scatterPart *= hierarchyPenalty[s]);
                                break;
                            case (SBPPartial, SBPSplit splitOut):
                                // actually partial to split needs gather.
                                break;
                            case (SBPPartial sBPPartial, SBPBroadCast):
                                sBPPartial.Axes.ToArray().ForEach(s => gatherPart *= hierarchyPenalty[s]);
                                break;
                            default:
                                throw new NotSupportedException($"{a} to {b}");
                        }
                    }

                    if (gatherPart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.ChipGlobalMemoryStoreBytes] = (UInt128)((gatherPart - 1) / scatterPart * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(a))),
                        };
                    }
                }

                break;
            case (DistributedType a, DistributedType b) when a.TensorType != b.TensorType && a.Placement == b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.ChipGlobalMemoryStoreBytes] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.ChipGlobalMemoryLoadBytes] = CostUtility.GetMemoryAccess(b),
                    [CostFactorNames.GridSynchronization] = SynchronizationEventCount,
                };
                break;
            case (DistributedType a, DistributedType b) when a.Placement != b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.ChipGlobalMemoryStoreBytes] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.ChipGlobalMemoryLoadBytes] = CostUtility.GetMemoryAccess(b),
                    [CostFactorNames.GridSynchronization] = SynchronizationEventCount,
                };
                break;
            case (DistributedType a, DistributedType b) when a.Partial != b.Partial:
                cost = new Cost()
                {
                    [CostFactorNames.ChipGlobalMemoryStoreBytes] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.ChipGlobalMemoryLoadBytes] = CostUtility.GetMemoryAccess(b),
                    [CostFactorNames.GridSynchronization] = SynchronizationEventCount,
                };
                break;
            case (DistributedType a, DistributedType b) when a == b:
                throw new InvalidOperationException($"the boxing inType == outType");
            default:
                throw new NotSupportedException($"{inType} {returnType}");
        }

        return cost;
    }

    public IValue Visit(IEvaluateContext context, Boxing target)
    {
        var input = context.GetArgumentValueAsTensor(target, Boxing.Input);
        return target.NewType switch
        {
            TensorType t => Value.FromTensor(Tensor.FromBytes(input.ElementType, input.BytesBuffer.ToArray(), (RankedShape)t.Shape)),
            DistributedType d => Value.FromTensor(Tensor.FromBytes(input.ElementType, input.BytesBuffer.ToArray(), (RankedShape)d.TensorType.Shape), d.AxisPolicies, d.Placement),
            _ => Value.FromTensor(input),
        };
    }

    private static bool TryGetTargetBoxingCost(ITargetOpCostModel targetCostModel, IRType inputType, IRType outputType, out Cost cost)
    {
        cost = Cost.Zero;
        return (inputType, outputType) switch
        {
            (TensorType inputTensor, DistributedType outputDistributed) =>
                TryGetTargetTensorLoadCost(targetCostModel, inputTensor, outputDistributed, out cost),
            (DistributedType inputDistributed, TensorType outputTensor) =>
                TryGetTargetTensorStoreCost(targetCostModel, inputDistributed, outputTensor, out cost),
            (DistributedType inputDistributed, DistributedType outputDistributed) =>
                TryGetTargetDistributedCopyCost(targetCostModel, inputDistributed, outputDistributed, out cost),
            _ => false,
        };
    }

    private static bool TryGetTargetTensorLoadCost(ITargetOpCostModel targetCostModel, TensorType inputTensor, DistributedType outputDistributed, out Cost cost)
    {
        var outputLocal = GetMaxDividedTensorType(outputDistributed);
        var localInputTensor = new TargetCostTensor(inputTensor.DType, outputLocal.Shape);
        var localOutputTensor = new TargetCostTensor(outputLocal.DType, outputLocal.Shape);
        return targetCostModel.TryGetElementwiseCost(new("boxing_tensor_load", [localInputTensor], localOutputTensor, WorkPerElement: 0.0), out cost);
    }

    private static bool TryGetTargetTensorStoreCost(ITargetOpCostModel targetCostModel, DistributedType inputDistributed, TensorType outputTensor, out Cost cost)
    {
        var inputLocal = GetMaxDividedTensorType(inputDistributed);
        var localInputTensor = new TargetCostTensor(inputLocal.DType, inputLocal.Shape);
        var localOutputTensor = new TargetCostTensor(outputTensor.DType, inputLocal.Shape);
        return targetCostModel.TryGetElementwiseCost(new("boxing_tensor_store", [localInputTensor], localOutputTensor, WorkPerElement: 0.0), out cost);
    }

    private static bool TryGetTargetDistributedCopyCost(ITargetOpCostModel targetCostModel, DistributedType inputDistributed, DistributedType outputDistributed, out Cost cost)
    {
        if (inputDistributed.Partial is { } partial && outputDistributed.Partial is null)
        {
            var reducedOutputLocal = GetMaxDividedTensorType(outputDistributed);
            var peerInputTensor = new TargetCostTensor(inputDistributed.TensorType.DType, reducedOutputLocal.Shape);
            var reducedOutputTensor = new TargetCostTensor(reducedOutputLocal.DType, reducedOutputLocal.Shape);
            var reductionGroupSize = partial.Axes.Aggregate(
                1.0,
                (product, axis) => product * inputDistributed.Placement.Hierarchy[axis]);
            return TryGetSynchronizedTargetElementwiseCost(
                targetCostModel,
                new(
                    "boxing_partial_reduce",
                    [peerInputTensor],
                    reducedOutputTensor,
                    WorkPerElement: System.Math.Max(0.0, reductionGroupSize - 1.0),
                    InputReadMultiplicity: reductionGroupSize),
                out cost);
        }

        var inputLocal = GetMaxDividedTensorType(inputDistributed);
        var outputLocal = GetMaxDividedTensorType(outputDistributed);
        var localInputTensor = new TargetCostTensor(inputLocal.DType, inputLocal.Shape);
        var localOutputTensor = new TargetCostTensor(outputLocal.DType, outputLocal.Shape);
        return TryGetSynchronizedTargetElementwiseCost(targetCostModel, new("boxing_reshard_copy", [localInputTensor], localOutputTensor, WorkPerElement: 0.0), out cost);
    }

    private static bool TryGetSynchronizedTargetElementwiseCost(ITargetOpCostModel targetCostModel, ElementwiseOpCostQuery query, out Cost cost)
    {
        if (!targetCostModel.TryGetElementwiseCost(query, out cost))
        {
            return false;
        }

        cost += new Cost { [CostFactorNames.GridSynchronization] = SynchronizationEventCount };
        return true;
    }

    private static TensorType GetMaxDividedTensorType(DistributedType distributedType)
    {
        return DistributedUtility.GetDividedTensorType(distributedType, DistributedUtility.DivideFlags.MaxShape);
    }
}
