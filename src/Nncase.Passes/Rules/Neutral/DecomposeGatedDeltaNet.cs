// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Exposes independently distributable projection, recurrent-core, and output-projection stages.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeGatedDeltaNet : IRewriteRule
{
    public IPattern Pattern { get; } = IsGatedDeltaNet(
        "gatedDeltaNet",
        "call",
        _ => true,
        IsWildcard("input"),
        IsWildcard("state"),
        IsWildcard("qkvWeight"),
        IsWildcard("qkvWeightScale"),
        IsWildcard("zWeight"),
        IsWildcard("zWeightScale"),
        IsWildcard("bWeight"),
        IsWildcard("aWeight"),
        IsWildcard("convWeight"),
        IsWildcard("aLog"),
        IsWildcard("dtBias"),
        IsWildcard("normWeight"),
        IsWildcard("outputWeight"),
        IsWildcard("outputWeightScale"),
        IsWildcard("layerId"));

    private BaseExpr GetReplace(
        GatedDeltaNet gatedDeltaNet,
        Call call,
        Expr input,
        Expr state,
        Expr qkvWeight,
        Expr qkvWeightScale,
        Expr zWeight,
        Expr zWeightScale,
        Expr bWeight,
        Expr aWeight,
        Expr convWeight,
        Expr aLog,
        Expr dtBias,
        Expr normWeight,
        Expr outputWeight,
        Expr outputWeightScale,
        Dimension layerId)
    {
        if (state.CheckedDataType is not ReferenceType
            {
                ElemType: GatedDeltaNetStateType { Config: { } stateConfig },
            })
        {
            throw new InvalidOperationException(
                $"GatedDeltaNet decomposition requires a configured GatedDeltaNetState, got {state.CheckedDataType}.");
        }

        Expr Project(Expr weight, Expr weightScale) => gatedDeltaNet.QuantizationMode switch
        {
            IR.Math.MatMulQuantizationMode.None => IR.F.Tensors.MatMul(
                input,
                weight,
                stateConfig.ActivationPrimType),
            IR.Math.MatMulQuantizationMode.DynamicBlock => IR.F.Math.BlockScaledMatMul(
                input,
                weight,
                weightScale,
                stateConfig.ActivationPrimType,
                gatedDeltaNet.WeightBlockN,
                gatedDeltaNet.WeightBlockK),
            _ => throw new NotSupportedException(
                $"GatedDeltaNet decomposition does not support quantization mode " +
                $"{gatedDeltaNet.QuantizationMode}."),
        };

        Expr Pack(Expr value, IReadOnlyList<int> lanes) => IR.F.Tensors.Pack(
            value,
            lanes.ToArray(),
            Enumerable.Repeat(1, lanes.Count).ToArray());

        var qkv = Pack(
            Project(qkvWeight, qkvWeightScale),
            stateConfig.GetLanes(
                GatedDeltaNetStateKind.Convolution,
                GatedDeltaNetStateDimKind.ConvChannels));
        var convolution = IR.F.NN.GatedDeltaNetConvolution(
            qkv,
            state,
            convWeight,
            layerId,
            gatedDeltaNet.ConvKernelSize);
        var z = Pack(Project(zWeight, zWeightScale), stateConfig.ActivationLanes);
        var recurrent = IR.F.NN.GatedDeltaNetRecurrentCore(
            convolution[1],
            convolution[0],
            z,
            input,
            IR.F.Tensors.Transpose(bWeight, [1, 0]),
            IR.F.Tensors.Transpose(aWeight, [1, 0]),
            aLog,
            dtBias,
            normWeight,
            layerId,
            gatedDeltaNet.NumKeyHeads,
            gatedDeltaNet.NumValueHeads,
            gatedDeltaNet.KeyHeadDim,
            gatedDeltaNet.ValueHeadDim,
            gatedDeltaNet.Epsilon);
        var output = gatedDeltaNet.QuantizationMode switch
        {
            IR.Math.MatMulQuantizationMode.None => IR.F.Tensors.MatMul(
                recurrent[0],
                outputWeight,
                stateConfig.ActivationPrimType),
            IR.Math.MatMulQuantizationMode.DynamicBlock => IR.F.Math.BlockScaledMatMul(
                recurrent[0],
                outputWeight,
                outputWeightScale,
                stateConfig.ActivationPrimType,
                gatedDeltaNet.WeightBlockN,
                gatedDeltaNet.WeightBlockK),
            _ => throw new NotSupportedException(
                $"GatedDeltaNet decomposition does not support quantization mode " +
                $"{gatedDeltaNet.QuantizationMode}."),
        };

        return new IR.Tuple(output, recurrent[1]).InheritMetaData(call);
    }
}
