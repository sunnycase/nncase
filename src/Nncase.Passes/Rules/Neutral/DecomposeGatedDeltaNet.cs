// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
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
        IsWildcard("zWeight"),
        IsWildcard("bWeight"),
        IsWildcard("aWeight"),
        IsWildcard("convWeight"),
        IsWildcard("aLog"),
        IsWildcard("dtBias"),
        IsWildcard("normWeight"),
        IsWildcard("outputWeight"),
        IsWildcard("layerId"));

    private BaseExpr GetReplace(
        GatedDeltaNet gatedDeltaNet,
        Call call,
        Expr input,
        Expr state,
        Expr qkvWeight,
        Expr zWeight,
        Expr bWeight,
        Expr aWeight,
        Expr convWeight,
        Expr aLog,
        Expr dtBias,
        Expr normWeight,
        Expr outputWeight,
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

        var projection = IR.F.NN.GatedDeltaNetProjection(
            input,
            state,
            qkvWeight,
            convWeight,
            layerId,
            gatedDeltaNet.ConvKernelSize);
        var recurrent = IR.F.NN.GatedDeltaNetRecurrentCore(
            input,
            projection[1],
            projection[0],
            zWeight,
            bWeight,
            aWeight,
            aLog,
            dtBias,
            normWeight,
            layerId,
            gatedDeltaNet.NumKeyHeads,
            gatedDeltaNet.NumValueHeads,
            gatedDeltaNet.KeyHeadDim,
            gatedDeltaNet.ValueHeadDim,
            gatedDeltaNet.Epsilon);
        var output = IR.F.Tensors.MatMul(
            recurrent[0],
            outputWeight,
            stateConfig.ActivationPrimType);

        return new IR.Tuple(output, recurrent[1]).InheritMetaData(call);
    }
}
