// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Exposes independently distributable gate/up and down expert stages.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeSparseExperts : IRewriteRule
{
    public IPattern Pattern { get; } = IsSparseExperts(
        "sparseExperts",
        "call",
        _ => true,
        IsWildcard("q"),
        IsWildcard("routerExpertIds"),
        IsWildcard("routerExpertWeights"),
        IsWildcard("moeExpertGateInputScale"),
        IsWildcard("moeExpertGateProjW"),
        IsWildcard("moeExpertGateProjScale"),
        IsWildcard("moeExpertDownInputScale"),
        IsWildcard("moeExpertDownProjW"),
        IsWildcard("moeExpertDownProjScale"),
        IsWildcard("moeExpertUpInputScale"),
        IsWildcard("moeExpertUpProjW"),
        IsWildcard("moeExpertUpProjScale"));

    private Expr GetReplace(
        SparseExperts sparseExperts,
        Call call,
        Expr q,
        Expr routerExpertIds,
        Expr routerExpertWeights,
        Expr moeExpertGateInputScale,
        Expr moeExpertGateProjW,
        Expr moeExpertGateProjScale,
        Expr moeExpertDownInputScale,
        Expr moeExpertDownProjW,
        Expr moeExpertDownProjScale,
        Expr moeExpertUpInputScale,
        Expr moeExpertUpProjW,
        Expr moeExpertUpProjScale)
    {
        var activations = IR.F.NN.SparseExpertsGateUp(
            q,
            routerExpertIds,
            moeExpertGateInputScale,
            moeExpertGateProjW,
            moeExpertGateProjScale,
            moeExpertUpInputScale,
            moeExpertUpProjW,
            moeExpertUpProjScale,
            q.CheckedDataType,
            sparseExperts.HiddenSize,
            sparseExperts.MoEIntermediateSize,
            sparseExperts.NumExpert,
            sparseExperts.NumTopK,
            sparseExperts.ChunkSize);
        return IR.F.NN.SparseExpertsDown(
            activations,
            routerExpertIds,
            routerExpertWeights,
            moeExpertDownInputScale,
            moeExpertDownProjW,
            moeExpertDownProjScale,
            q.CheckedDataType,
            sparseExperts.HiddenSize,
            sparseExperts.MoEIntermediateSize,
            sparseExperts.NumExpert,
            sparseExperts.NumTopK,
            sparseExperts.ChunkSize).InheritMetaData(call);
    }
}
