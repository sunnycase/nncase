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
    public static IRType VisitType(IRType inType, IRType outType, bool isReshape = false)
    {
        IRType VisitD2D(DistributedType inv, DistributedType outv)
        {
            if (inv == outv)
            {
                return new InvalidType("Same DistributedType");
            }

            if (inv.TensorType != outv.TensorType)
            {
                if (!inv.AxisPolicies.Any(sbp => sbp is SBPPartial) && inv.Partial is null)
                {
                    return outv;
                }
                else
                {
                    return new InvalidType("Not Support Partial when shape changes.");
                }
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
                            if (splitIn.Axes.Except(s.Axes).Any())
                            {
                                return new InvalidType("Not Supported Split-> Split.");
                            }
                        }

                        if (s.Axes.Except(inv.Partial.Axes).ToArray() != s.Axes)
                        {
                            partialDims.Add(i);
                        }
                    }
                }

                var ndspsIn = DistributedUtility.AxisPolicesToNDSBP(inv.AxisPolicies, inv.Placement.Rank);
                var ndspsOut = DistributedUtility.AxisPolicesToNDSBP(outv.AxisPolicies, outv.Placement.Rank);
                if (Enumerable.Range(0, ndspsIn.Count).Any(i => ndspsIn[i] is SBPSplit si && (ndspsOut[i] is SBPBroadCast || (ndspsOut[i] is SBPSplit so && so.Axes[0] != si.Axes[0]))))
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

        UInt128 synchronizeCost = inType is DistributedType dt && dt.Placement.HierarchyKind == HierarchyKind.SMT ? 1500U : 25_000; // 25k cycles on 5GHz CPU is about 5us.
        var cost = new Cost() { [CostFactorNames.CPUCycles] = 1, [CostFactorNames.MemoryLoad] = 0, [CostFactorNames.MemoryStore] = 0, [CostFactorNames.Synchronization] = synchronizeCost };
        switch (inType, returnType)
        {
            case (TensorType _, DistributedType distributedType):
                switch (context.CompileOptions.TargetOptions)
                {
                    default:
                        cost = new Cost()
                        {
                            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(distributedType),
                            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(distributedType),
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
                            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(distributedType),
                            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(distributedType),
                            [CostFactorNames.Synchronization] = synchronizeCost,
                        };
                        break;
                }

                break;

            case (DistributedType a, DistributedType b) when a.TensorType == b.TensorType && a.Placement == b.Placement && a.AxisPolicies != b.AxisPolicies:
#if false
                {
                    var fullLoadStore = new Cost()
                    {
                        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a),
                        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b),
                        [CostFactorNames.Synchronization] = synchronizeCost,
                    };

                    float scatterPart = 1;
                    float gatherPart = 1;
                    float reducePart = 1;
                    float latency = 0;

                    // TODO: calculate cost using NTTD.
                    var ndsbpsA = DistributedUtility.AxisPolicesToNDSBP(a.AxisPolicies, a.Placement.Rank);
                    var ndsbpsB = DistributedUtility.AxisPolicesToNDSBP(b.AxisPolicies, b.Placement.Rank);
                    for (int i = 0; i < a.Placement.Rank; i++)
                    {
                        switch (ndsbpsA[i], ndsbpsB[i])
                        {
                            case (SBPSplit splitIn, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPSplit splitOut:
                                        if (splitIn.Axes[0] != splitOut.Axes[0])
                                        {
                                            // when split different axis, need global load store.
                                            return fullLoadStore;
                                        }

                                        break;
                                    case SBPBroadCast:
                                        scatterPart *= a.Placement.Hierarchy[i];
                                        gatherPart *= a.Placement.Hierarchy[i];
                                        break;
                                    default:
                                        throw new NotSupportedException("split to partial");
                                }

                                break;
                            case (SBPBroadCast, SBPBroadCast or SBPSplit):
                                // no cost.
                                cost += new Cost()
                                {
                                    [CostFactorNames.CPUCycles] = 1,
                                };
                                break;
                            case (SBPPartial, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPPartial:
                                        break;
                                    case SBPBroadCast:
                                        latency = MathF.Max(latency, ((INTTTargetOptions)context.CompileOptions.TargetOptions).HierarchyLatencies[i]);
                                        reducePart *= a.Placement.Hierarchy[i];
                                        gatherPart *= a.Placement.Hierarchy[i];
                                        if (i == 0)
                                        {
                                            scatterPart *= a.Placement.Hierarchy[i];
                                        }

                                        break;
                                    case SBPSplit:
                                        throw new NotSupportedException("split to partial");
                                }

                                break;
                            case (SBPBroadCast, SBPPartial):
                                // note this case only for tests.
                                cost += new Cost()
                                {
                                    [CostFactorNames.CPUCycles] = 1,
                                };
                                break;
                            default:
                                throw new NotSupportedException($"{a} to {b}");
                        }
                    }

                    if (gatherPart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.MemoryStore] = (UInt128)((gatherPart - 1) * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(a)) / gatherPart),
                        };

                        if (a.Placement.HierarchyKind == HierarchyKind.SMT && (ndsbpsA[1] is SBPPartial))
                        {
                            cost[CostFactorNames.MemoryStore] *= 8;
                        }
                    }

                    if (reducePart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.Comm] = (UInt128)((reducePart - 1) * latency),
                        };
                    }

                    if (scatterPart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.MemoryLoad] = (UInt128)((scatterPart - 1) * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(b)) / scatterPart),
                        };
                    }
                }
#endif
                {
                    var fullLoadStore = new Cost()
                    {
                        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a) * (UInt128)a.TensorType.DType.SizeInBytes,
                        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b) * (UInt128)b.TensorType.DType.SizeInBytes,
                        [CostFactorNames.Synchronization] = synchronizeCost,
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
                                            var setA = new HashSet<int>(splitIn.Axes);
                                            var setB = new HashSet<int>(splitOut.Axes);
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
                                                if (diff.All(d => d > splitIn.Axes[^1]))
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
                                        splitIn.Axes.ToArray().ForEach(s => gatherPart *= hierarchyPenalty[s]);
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
                                splitOut.Axes.ToArray().ForEach(s => scatterPart *= hierarchyPenalty[s]);
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
                            [CostFactorNames.MemoryStore] = (UInt128)((gatherPart - 1) / scatterPart * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(a))) * (UInt128)a.TensorType.DType.SizeInBytes,
                        };
                    }
                }

                break;
            case (DistributedType a, DistributedType b) when a.TensorType != b.TensorType && a.Placement == b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b),
                    [CostFactorNames.Synchronization] = synchronizeCost,
                };
                break;
            case (DistributedType a, DistributedType b) when a.Placement != b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b),
                    [CostFactorNames.Synchronization] = synchronizeCost,
                };
                break;
            case (DistributedType a, DistributedType b) when a.Partial != b.Partial:
                cost = new Cost()
                {
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(a),
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(b),
                    [CostFactorNames.Synchronization] = synchronizeCost,
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
        if (ReferenceEquals(targetCostModel, DefaultTargetOpCostModel.Instance))
        {
            return false;
        }

        return (inputType, outputType) switch
        {
            (TensorType inputTensor, DistributedType outputDistributed) =>
                TryGetTargetTensorLoadCost(targetCostModel, inputTensor, outputDistributed, out cost),
            (DistributedType inputDistributed, TensorType outputTensor) =>
                TryGetTargetTensorStoreCost(targetCostModel, inputDistributed, outputTensor, out cost),
            (DistributedType inputDistributed, DistributedType outputDistributed) when IsThreadLocalMetadataReshard(inputDistributed, outputDistributed) =>
                TryGetThreadLocalMetadataReshardCost(out cost),
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
        var inputLocal = GetMaxDividedTensorType(inputDistributed);
        var outputLocal = GetMaxDividedTensorType(outputDistributed);
        var localInputTensor = new TargetCostTensor(inputLocal.DType, inputLocal.Shape);
        var localOutputTensor = new TargetCostTensor(outputLocal.DType, outputLocal.Shape);
        return targetCostModel.TryGetElementwiseCost(new("boxing_reshard_copy", [localInputTensor], localOutputTensor, WorkPerElement: 0.0), out cost);
    }

    private static bool TryGetThreadLocalMetadataReshardCost(out Cost cost)
    {
        cost = new Cost { [CostFactorNames.CPUCycles] = 1 };
        return true;
    }

    private static TensorType GetMaxDividedTensorType(DistributedType distributedType)
    {
        return DistributedUtility.GetDividedTensorType(distributedType, DistributedUtility.DivideFlags.MaxShape);
    }

    private static bool IsThreadLocalMetadataReshard(DistributedType inputType, DistributedType outputType)
    {
        if (inputType.TensorType != outputType.TensorType
            || inputType.Placement != outputType.Placement
            || inputType.Partial is not null
            || outputType.Partial is not null
            || DistributedUtility.AreSamePolicies(inputType.AxisPolicies, outputType.AxisPolicies, false))
        {
            return false;
        }

        var threadAxis = inputType.Placement.Rank - 1;
        var reducedInputPolicies = inputType.AxisPolicies.Select(sbp => ReduceThreadAxis(sbp, threadAxis)).ToArray();
        if (DistributedUtility.AreSamePolicies(reducedInputPolicies, outputType.AxisPolicies.ToArray(), false))
        {
            return true;
        }

        var reducedOutputPolicies = outputType.AxisPolicies.Select(sbp => ReduceThreadAxis(sbp, threadAxis)).ToArray();
        return DistributedUtility.AreSamePolicies(reducedOutputPolicies, inputType.AxisPolicies.ToArray(), false);
    }

    private static SBP ReduceThreadAxis(SBP sbp, int threadAxis)
    {
        if (sbp is not SBPSplit split || !split.Axes.Contains(threadAxis))
        {
            return sbp;
        }

        return split.Axes.Count == 1 ? SBP.B : SBP.S(split.Axes.Except([threadAxis]).ToArray());
    }
}
