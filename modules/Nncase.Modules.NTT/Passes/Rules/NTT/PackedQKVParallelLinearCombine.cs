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
/// Removes a combine selected as an exact materialized identity.
/// </summary>
[RuleGenerator]
public sealed partial class FoldMaterializedPackedQKVParallelLinearCombine : IRewriteRule
{
    public IPattern Pattern { get; } = IsPackedQKVParallelLinearCombine(
        "combine",
        "combineCall",
        _ => true,
        IsWildcard("qkv"));

    private Expr? GetReplace(
        PackedQKVParallelLinearCombine combine,
        Call combineCall,
        Expr qkv)
        => qkv.CheckedType == combineCall.CheckedType && IsMaterialized(qkv.CheckedType)
            ? qkv.InheritMetaData(combineCall)
            : null;

    private static bool IsMaterialized(IRType type)
        => type is TupleType tuple && tuple.Fields.All(field =>
            field is not DistributedType distributed ||
            (distributed.Partial is null && distributed.AxisPolicies.All(policy => policy is not SBPPartial)));
}

/// <summary>
/// Lowers an unfused partial QKV combine to the generic tuple boxing implementation.
/// </summary>
[RuleGenerator]
public sealed partial class LowerPackedQKVParallelLinearCombine : IRewriteRule
{
    public IPattern Pattern { get; } = IsPackedQKVParallelLinearCombine(
        "combine",
        "combineCall",
        _ => true,
        IsWildcard("qkv"));

    private Expr? GetReplace(
        PackedQKVParallelLinearCombine combine,
        Call combineCall,
        Expr qkv)
    {
        if (qkv.CheckedType is not TupleType { Count: 3 } input ||
            input.Fields.Any(field => field is not DistributedType { Partial: { Op: ReduceOp.Sum } }))
        {
            return null;
        }

        return IR.F.Distributed.Boxing(qkv, combine.OutputType).InheritMetaData(combineCall);
    }
}
