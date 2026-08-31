// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.NTT;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Rules.NTT;

/// <summary>
/// Removes a GLU combine selected as an exact materialized identity.
/// </summary>
[RuleGenerator]
public sealed partial class FoldMaterializedPackedMatMulGluCombine : IRewriteRule
{
    public IPattern Pattern { get; } = IsPackedMatMulGluCombine(
        "combine",
        "combineCall",
        _ => true,
        IsWildcard("projections"));

    private Expr? GetReplace(
        PackedMatMulGluCombine combine,
        Call combineCall,
        Expr projections)
        => projections.CheckedType == combineCall.CheckedType && IsMaterialized(projections.CheckedType)
            ? projections.InheritMetaData(combineCall)
            : null;

    private static bool IsMaterialized(IRType type)
        => type is not DistributedType distributed ||
            (distributed.Partial is null && distributed.AxisPolicies.All(policy => policy is not SBPPartial));
}

/// <summary>
/// Lowers a split-K GLU combine to one tuple boxing followed by SwiGLU.
/// </summary>
[RuleGenerator]
public sealed partial class LowerPackedMatMulGluCombine : IRewriteRule
{
    public IPattern Pattern { get; } = IsPackedMatMulGluCombine(
        "combine",
        "combineCall",
        _ => true,
        IsWildcard("projections"));

    private Expr? GetReplace(
        PackedMatMulGluCombine combine,
        Call combineCall,
        Expr projections)
    {
        if (projections.CheckedType is not TupleType { Count: 2 } input ||
            input.Fields.Any(field => field is not DistributedType { Partial: { Op: ReduceOp.Sum } }))
        {
            return null;
        }

        var materializedType = new TupleType(new[] { combine.OutputType, combine.OutputType });
        var materialized = IR.F.Distributed.Boxing(projections, materializedType);
        var gate = IR.F.Tensors.GetItem(materialized, 0);
        var up = IR.F.Tensors.GetItem(materialized, 1);
        return (IR.F.NN.Swish(gate) * up).InheritMetaData(combineCall);
    }
}
