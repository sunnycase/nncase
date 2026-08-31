// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator;
using Nncase.Evaluator.IR.NTT;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Couples the packed gate/up projection layouts and permits split-K only as
/// a two-field projection partial consumed by PackedMatMulGluCombine.
/// </summary>
internal sealed class PackedMatMulGluCandidateProvider :
    DistributedCandidateProvider<PackedMatMulGlu>
{
    public override IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        PackedMatMulGlu target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes
            .Concat(Enumerate(context, target).Select(candidate => candidate.Output))
            .Concat(EnumerateOutputCandidates(context, target).Select(candidate => candidate.Output))
            .Distinct()
            .ToArray();

    public override bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        PackedMatMulGlu target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        tuples = Enumerate(context, target)
            .Concat(EnumerateFromOutput(context, target, returnType))
            .Where(candidate => candidate.Output == returnType)
            .Distinct()
            .Select(candidate => new DistributedCandidateTuple(
                [
                    candidate.Input,
                    candidate.GateWeight,
                    candidate.UpWeight,
                    candidate.GateBias,
                    candidate.UpBias,
                    candidate.GateInputScale,
                    candidate.UpInputScale,
                    candidate.GateWeightScale,
                    candidate.UpWeightScale,
                ],
                "packed-matmul-glu-sbp"))
            .ToArray();
        return true;
    }

    private static IEnumerable<PackedMatMulGluDistributedCandidate> Enumerate(
        DistributedCandidateContext context,
        PackedMatMulGlu target)
    {
        if (context.AvailableInputTypes.Count != 9)
        {
            yield break;
        }

        foreach (var input in context.AvailableInputTypes[PackedMatMulGlu.Input.Index]
                     .OfType<DistributedType>()
                     .Where(type =>
                         type.Partial is null &&
                         IsScaleGroupAlignedReductionShard(target, type)))
        {
            foreach (var gateWeight in GetAlignedWeights(
                         context,
                         target,
                         PackedMatMulGlu.GateWeight.Index,
                         input))
            {
                foreach (var upWeight in GetAlignedWeights(
                             context,
                             target,
                             PackedMatMulGlu.UpWeight.Index,
                             input)
                         .Where(type => HasSameProjectionLayout(gateWeight, type)))
                {
                    foreach (var gateBias in GetOptionalOperandCandidates(
                                 context,
                                 PackedMatMulGlu.GateBias.Index,
                                 input.Placement))
                    {
                        foreach (var upBias in GetOptionalOperandCandidates(
                                     context,
                                     PackedMatMulGlu.UpBias.Index,
                                     input.Placement))
                        {
                            foreach (var gateInputScale in GetOptionalOperandCandidates(
                                         context,
                                         PackedMatMulGlu.GateInputScale.Index,
                                         input.Placement))
                            {
                                foreach (var upInputScale in GetOptionalOperandCandidates(
                                             context,
                                             PackedMatMulGlu.UpInputScale.Index,
                                             input.Placement))
                                {
                                    foreach (var gateWeightScale in GetOptionalOperandCandidates(
                                                 context,
                                                 PackedMatMulGlu.GateWeightScale.Index,
                                                 input.Placement))
                                    {
                                        foreach (var upWeightScale in GetOptionalOperandCandidates(
                                                     context,
                                                     PackedMatMulGlu.UpWeightScale.Index,
                                                     input.Placement))
                                        {
                                            var output = PackedMatMulGluEvaluator.InferType(
                                                target,
                                                input,
                                                gateWeight,
                                                upWeight,
                                                gateBias,
                                                upBias,
                                                gateInputScale,
                                                upInputScale,
                                                gateWeightScale,
                                                upWeightScale);
                                            if (IsSupportedOutput(output))
                                            {
                                                yield return new(
                                                    input,
                                                    gateWeight,
                                                    upWeight,
                                                    gateBias,
                                                    upBias,
                                                    gateInputScale,
                                                    upInputScale,
                                                    gateWeightScale,
                                                    upWeightScale,
                                                    output);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static IEnumerable<PackedMatMulGluDistributedCandidate> EnumerateOutputCandidates(
        DistributedCandidateContext context,
        PackedMatMulGlu target)
    {
        if (context.SourceCall.CheckedType is not TensorType outputTensor)
        {
            yield break;
        }

        var placements = context.AvailableInputTypes
            .SelectMany(types => types)
            .OfType<DistributedType>()
            .Select(type => type.Placement)
            .Distinct()
            .ToArray();
        foreach (var output in context.GetLeafCandidateTypes(outputTensor, placements))
        {
            foreach (var candidate in EnumerateFromOutput(context, target, output))
            {
                yield return candidate;
            }
        }

        foreach (var placement in placements)
        {
            var blockAxes = Enumerable.Range(0, placement.Rank)
                .Where(placement.IsPhysicalBlockAxis)
                .ToArray();
            if (blockAxes.Length == 0)
            {
                continue;
            }

            var outputPolicies = Enumerable.Repeat<SBP>(SBP.B, outputTensor.Shape.Rank).ToArray();
            outputPolicies[^1] = SBP.SBlockCyclic(blockAxes, 1);
            var output = new DistributedType(outputTensor, outputPolicies, placement);
            if (!DistributedUtility.IsDistributable(
                    output.TensorType,
                    output.AxisPolicies,
                    output.Placement))
            {
                continue;
            }

            foreach (var candidate in EnumerateFromOutput(context, target, output))
            {
                yield return candidate;
            }
        }
    }

    private static IEnumerable<PackedMatMulGluDistributedCandidate> EnumerateFromOutput(
        DistributedCandidateContext context,
        PackedMatMulGlu target,
        IRType returnType)
    {
        if (context.AvailableInputTypes.Count != 9 ||
            returnType is not DistributedType { Partial: null } output ||
            output.TensorType.DType is not VectorType outputVector ||
            output.TensorType.Shape is not RankedShape { Rank: 2 } ||
            output.AxisPolicies.Any(policy => policy is SBPPartial) ||
            GetSourceTensorType(context, PackedMatMulGlu.GateWeight.Index) is not
                { DType: VectorType gateWeightVector, Shape: RankedShape { Rank: 2 } } gateWeightTensor ||
            GetSourceTensorType(context, PackedMatMulGlu.UpWeight.Index) is not
                { DType: VectorType upWeightVector, Shape: RankedShape { Rank: 2 } } upWeightTensor ||
            gateWeightTensor != upWeightTensor ||
            !gateWeightVector.Lanes.SequenceEqual(upWeightVector.Lanes) ||
            !PackedMatMulGluEvaluator.TryGetLayoutInfo(
                target.RhsLayout,
                gateWeightVector,
                gateWeightTensor.Shape.Rank,
                target.OutputDataType,
                out var weightUnpackAxes,
                out var outputLanes,
                out var transposeB,
                out _) ||
            !outputVector.Lanes.SequenceEqual(outputLanes) ||
            TypeInference.UnpackType(
                output,
                Enumerable.Repeat(output.TensorType.Shape.Rank - 1, outputLanes.Length).ToArray()) is not
                DistributedType { TensorType.Shape: RankedShape { Rank: 2 } } logicalOutput ||
            TypeInference.UnpackType(gateWeightTensor, weightUnpackAxes) is not
                TensorType { Shape: RankedShape { Rank: 2 } } logicalWeightTensor)
        {
            yield break;
        }

        foreach (var input in context.AvailableInputTypes[PackedMatMulGlu.Input.Index]
                     .OfType<DistributedType>()
                     .Where(type =>
                         type.Partial is null &&
                         type.Placement == output.Placement &&
                         type.TensorType.Shape is RankedShape { Rank: 2 }))
        {
            var dimInfo = VectorizedMatMul.GetDimInfo(
                false,
                transposeB,
                input.TensorType.Shape.Rank,
                logicalWeightTensor.Shape.Rank);
            var weightPolicies = Enumerable.Repeat<SBP>(SBP.B, logicalWeightTensor.Shape.Rank).ToArray();
            weightPolicies[dimInfo.Rk] = input.AxisPolicies[dimInfo.Lk];
            weightPolicies[dimInfo.Rn] = logicalOutput.AxisPolicies[^1];
            var logicalWeight = new DistributedType(
                logicalWeightTensor,
                weightPolicies,
                output.Placement);
            if (!DistributedUtility.IsDistributable(
                    logicalWeight.TensorType,
                    logicalWeight.AxisPolicies,
                    logicalWeight.Placement) ||
                TypeInference.PackType(
                    logicalWeight,
                    gateWeightVector.Lanes.ToArray(),
                    weightUnpackAxes) is not DistributedType packedWeight ||
                packedWeight.TensorType != gateWeightTensor)
            {
                continue;
            }

            foreach (var gateBias in GetBiasCandidates(
                         context,
                         PackedMatMulGlu.GateBias.Index,
                         output))
            {
                foreach (var upBias in GetBiasCandidates(
                             context,
                             PackedMatMulGlu.UpBias.Index,
                             output))
                {
                    foreach (var gateInputScale in GetOptionalOperandCandidates(
                                 context,
                                 PackedMatMulGlu.GateInputScale.Index,
                                 output.Placement))
                    {
                        foreach (var upInputScale in GetOptionalOperandCandidates(
                                     context,
                                     PackedMatMulGlu.UpInputScale.Index,
                                     output.Placement))
                        {
                            foreach (var gateWeightScale in GetOptionalOperandCandidates(
                                         context,
                                         PackedMatMulGlu.GateWeightScale.Index,
                                         output.Placement))
                            {
                                foreach (var upWeightScale in GetOptionalOperandCandidates(
                                             context,
                                             PackedMatMulGlu.UpWeightScale.Index,
                                             output.Placement))
                                {
                                    var inferredOutput = PackedMatMulGluEvaluator.InferType(
                                        target,
                                        input,
                                        packedWeight,
                                        packedWeight,
                                        gateBias,
                                        upBias,
                                        gateInputScale,
                                        upInputScale,
                                        gateWeightScale,
                                        upWeightScale);
                                    if (inferredOutput == output)
                                    {
                                        yield return new(
                                            input,
                                            packedWeight,
                                            packedWeight,
                                            gateBias,
                                            upBias,
                                            gateInputScale,
                                            upInputScale,
                                            gateWeightScale,
                                            upWeightScale,
                                            output);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static IEnumerable<DistributedType> GetAlignedWeights(
        DistributedCandidateContext context,
        PackedMatMulGlu target,
        int index,
        DistributedType input)
    {
        foreach (var weight in context.AvailableInputTypes[index]
                     .OfType<DistributedType>()
                     .Where(type => type.Partial is null))
        {
            if (PackedMatMulDistributedCandidates.TryAlignRhsReductionPolicy(
                    target.RhsLayout,
                    input,
                    weight,
                    out var alignedWeight))
            {
                yield return alignedWeight;
            }
        }
    }

    private static IEnumerable<IRType> GetOptionalOperandCandidates(
        DistributedCandidateContext context,
        int index,
        Placement placement)
        => context.AvailableInputTypes[index].Where(type => type switch
        {
            NoneType => true,
            TensorType => true,
            DistributedType distributed =>
                distributed.Placement == placement &&
                distributed.Partial is null &&
                distributed.AxisPolicies.All(policy => policy is SBPBroadCast),
            _ => false,
        });

    private static IEnumerable<IRType> GetBiasCandidates(
        DistributedCandidateContext context,
        int index,
        DistributedType output)
    {
        if (context.SourceCall.Arguments[index].CheckedType is NoneType)
        {
            yield return NoneType.Default;
        }
        else
        {
            yield return output;
        }
    }

    private static TensorType? GetSourceTensorType(
        DistributedCandidateContext context,
        int index)
        => context.SourceCall.Arguments[index].CheckedType switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null,
        };

    private static bool HasSameProjectionLayout(DistributedType lhs, DistributedType rhs)
        => lhs.Placement == rhs.Placement &&
            lhs.TensorType == rhs.TensorType &&
            lhs.AxisPolicies.SequenceEqual(rhs.AxisPolicies);

    private static bool IsScaleGroupAlignedReductionShard(
        PackedMatMulGlu target,
        DistributedType input)
    {
        if (target.QuantizationMode != IR.Math.MatMulQuantizationMode.DynamicBlock)
        {
            return true;
        }

        if (input.TensorType.Shape is not RankedShape { Rank: > 0 } ||
            input.AxisPolicies[^1] is not SBPSplit reductionSplit)
        {
            return true;
        }

        if (!reductionSplit.IsContiguous || target.WeightBlockK <= 0)
        {
            return false;
        }

        var localInput = DistributedUtility.GetDividedTensorType(
            input,
            DistributedUtility.DivideFlags.MaxShape);
        if (localInput.Shape[^1] is not { IsFixed: true } localK)
        {
            return false;
        }

        var vectorLanes = localInput.DType is VectorType vectorType
            ? vectorType.Lanes.Aggregate(1L, (product, lane) => checked(product * lane))
            : 1L;
        var scalarK = checked(localK.FixedValue * vectorLanes);
        return scalarK % target.WeightBlockK == 0;
    }

    private static bool IsSupportedOutput(IRType output) => output switch
    {
        DistributedType { Partial: null } distributed =>
            distributed.AxisPolicies.All(policy => policy is not SBPPartial),
        TupleType { Count: 2 } tuple => tuple.Fields.All(field =>
            field is DistributedType { Partial: { Op: ReduceOp.Sum } } partial &&
            partial.AxisPolicies.All(policy => policy is not SBPPartial)),
        _ => false,
    };
}

internal sealed record PackedMatMulGluDistributedCandidate(
    IRType Input,
    IRType GateWeight,
    IRType UpWeight,
    IRType GateBias,
    IRType UpBias,
    IRType GateInputScale,
    IRType UpInputScale,
    IRType GateWeightScale,
    IRType UpWeightScale,
    IRType Output);
