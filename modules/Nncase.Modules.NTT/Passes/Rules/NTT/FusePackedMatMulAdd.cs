// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NTT;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Rules.NTT;

/// <summary>
/// Fuses an exact-layout add into a packed matmul after distributed layouts
/// have been selected.
/// </summary>
[RuleGenerator]
public sealed partial class FusePackedMatMulAdd : IRewriteRule
{
    public FusePackedMatMulAdd()
    {
        var packedMatMul = IsPackedMatMul(
            "packedMatMul",
            "packedCall",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"),
            IsWildcard("scale"),
            IsNone());

        Pattern = IsAlt(
            IsBinary(
                "binary",
                "binaryCall",
                op => op.BinaryOp == BinaryOp.Add,
                packedMatMul,
                IsWildcard("addend")),
            IsBinary(
                "binary",
                "binaryCall",
                op => op.BinaryOp == BinaryOp.Add,
                IsWildcard("addend"),
                packedMatMul));
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(
        PackedMatMul packedMatMul,
        Call packedCall,
        Expr lhs,
        Expr rhs,
        Expr scale,
        Call binaryCall,
        Expr addend)
    {
        if (packedMatMul.FusedReduce ||
            ReferenceEquals(addend, packedCall) ||
            packedCall.Users.Count() != 1 ||
            !ReferenceEquals(packedCall.Users.Single(), binaryCall) ||
            !Equals(packedCall.CheckedType, addend.CheckedType) ||
            !Equals(binaryCall.CheckedType, packedCall.CheckedType))
        {
            return null;
        }

        return IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            packedMatMul.FusedReduce,
            packedMatMul.OutputDataType,
            scale,
            packedMatMul.RhsLayout,
            addend).InheritMetaData(binaryCall);
    }
}
