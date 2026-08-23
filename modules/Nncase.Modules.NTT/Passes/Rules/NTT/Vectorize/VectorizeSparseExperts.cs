// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

/// <summary>
/// Absorbs hidden-axis output packing into SparseExpertsDown.
/// </summary>
[RuleGenerator]
public sealed partial class VectorizeSparseExpertsPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack(
        "vectorize",
        "caller",
        _ => true,
        PatternMatch.F.NN.IsSparseExpertsDown(
            "sparseExpertsDown",
            "callee",
            _ => true,
            IsWildcard("activations"),
            IsWildcard("routerIdx"),
            IsWildcard("routerWeights"),
            IsWildcard("moeExpertDownInputScale"),
            IsWildcard("moeExpertDownProjW"),
            IsWildcard("moeExpertDownProjScale")));

    private Expr? GetReplace(
        Pack vectorize,
        Call caller,
        Call callee,
        SparseExpertsDown sparseExpertsDown)
    {
        if (sparseExpertsDown.OutputDataType is VectorType ||
            caller.CheckedDataType is not VectorType outputDataType ||
            vectorize.Axes.Count == 0 ||
            vectorize.Axes.Any(axis => axis != 1))
        {
            return null;
        }

        var hidden = callee.CheckedShape[1];
        foreach (var lane in vectorize.Lanes)
        {
            if (!Dimension.TryDivExactly(hidden, lane, out hidden))
            {
                return null;
            }
        }

        return new Call(
            new SparseExpertsDown(
                outputDataType,
                sparseExpertsDown.HiddenSize,
                sparseExpertsDown.MoEIntermediateSize,
                sparseExpertsDown.NumExpert,
                sparseExpertsDown.NumTopK,
                sparseExpertsDown.ChunkSize),
            callee.Arguments.ToArray()).InheritMetaData(caller);
    }
}
