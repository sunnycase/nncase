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
/// Folds a materialized combine back into the direct packed-matmul microkernel.
/// Partial combines remain explicit for reduce-scatter lowering.
/// </summary>
[RuleGenerator]
public sealed partial class LowerMaterializedPackedMatMulNormStatsCombine : IRewriteRule
{
    public IPattern Pattern { get; } = IsPackedMatMulNormStatsCombine(
        "combine",
        "combineCall",
        _ => true,
        IsWildcard("input"),
        IsWildcard("addend"));

    private Expr? GetReplace(
        PackedMatMulNormStatsCombine combine,
        Call combineCall,
        Expr input,
        Expr addend)
    {
        if (input is not Call { Target: PackedMatMul packed } packedCall ||
            input.CheckedType is DistributedType { Partial: not null } ||
            packedCall[PackedMatMul.Addend].CheckedType is not NoneType ||
            combineCall.CheckedType is not TupleType { Count: 2 } output ||
            !Equals(input.CheckedType, output[0]) ||
            !Equals(addend.CheckedType, output[0]))
        {
            return null;
        }

        return IR.F.NTT.PackedMatMulNormStats(
                (Expr)packedCall[PackedMatMul.Lhs],
                (Expr)packedCall[PackedMatMul.Rhs],
                packed.OutputDataType,
                packed.RhsLayout,
                combine.Axis,
                combine.UseMean,
                (Expr)packedCall[PackedMatMul.Scale],
                addend)
            .InheritMetaData(combineCall);
    }
}
