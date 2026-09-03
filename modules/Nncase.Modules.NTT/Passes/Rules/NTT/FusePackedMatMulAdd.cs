// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Distributed;
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
            addend).InheritMetaData(packedCall);
    }
}

/// <summary>
/// Fuses an exact-layout add into a packed block-scaled matmul before
/// distributed layouts are selected.
/// </summary>
[RuleGenerator]
public sealed partial class FusePackedBlockScaledMatMulAdd : IRewriteRule
{
    public FusePackedBlockScaledMatMulAdd()
    {
        var packedMatMul = IsPackedBlockScaledMatMul(
            "packedMatMul",
            "packedCall",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"),
            IsWildcard("rhsScale"),
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
        PackedBlockScaledMatMul packedMatMul,
        Call packedCall,
        Expr lhs,
        Expr rhs,
        Expr rhsScale,
        Call binaryCall,
        Expr addend)
    {
        if (ReferenceEquals(addend, packedCall) ||
            packedCall.Users.Count() != 1 ||
            !ReferenceEquals(packedCall.Users.Single(), binaryCall) ||
            !Equals(packedCall.CheckedType, addend.CheckedType) ||
            !Equals(binaryCall.CheckedType, packedCall.CheckedType))
        {
            return null;
        }

        return IR.F.NTT.PackedBlockScaledMatMul(
            lhs,
            rhs,
            rhsScale,
            packedMatMul.OutputDataType,
            packedMatMul.WeightBlockN,
            packedMatMul.WeightBlockK,
            packedMatMul.RhsLayout,
            packedMatMul.OutputNVectorLaneCount,
            addend).InheritMetaData(packedCall);
    }
}

/// <summary>
/// Fuses an exact-layout add into a packed NVFP4 matmul before distributed
/// layouts are selected.
/// </summary>
[RuleGenerator]
public sealed partial class FusePackedNVFP4MatMulAdd : IRewriteRule
{
    public FusePackedNVFP4MatMulAdd()
    {
        var packedMatMul = IsPackedNVFP4MatMul(
            "packedMatMul",
            "packedCall",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhsPacked"),
            IsWildcard("rhsScale"),
            IsWildcard("lhsGlobalScale"),
            IsWildcard("rhsGlobalScale"),
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
        PackedNVFP4MatMul packedMatMul,
        Call packedCall,
        Expr lhs,
        Expr rhsPacked,
        Expr rhsScale,
        Expr lhsGlobalScale,
        Expr rhsGlobalScale,
        Call binaryCall,
        Expr addend)
    {
        if (ReferenceEquals(addend, packedCall) ||
            packedCall.Users.Count() != 1 ||
            !ReferenceEquals(packedCall.Users.Single(), binaryCall) ||
            !Equals(packedCall.CheckedType, addend.CheckedType) ||
            !Equals(binaryCall.CheckedType, packedCall.CheckedType))
        {
            return null;
        }

        return IR.F.NTT.PackedNVFP4MatMul(
            lhs,
            rhsPacked,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            packedMatMul.OutputDataType,
            packedMatMul.GroupSize,
            packedMatMul.InputKVectorLaneCount,
            packedMatMul.RhsKPackLaneCount,
            packedMatMul.RhsKVectorLaneCount,
            packedMatMul.OutputNVectorLaneCount,
            addend).InheritMetaData(packedCall);
    }
}

/// <summary>
/// Fuses an add through a layout-only sharded view of a packed matmul result.
/// </summary>
[RuleGenerator]
public sealed partial class FusePackedMatMulAddThroughShardedView : IRewriteRule
{
    public FusePackedMatMulAddThroughShardedView()
    {
        var packedMatMul = IsPackedMatMul(
            "packedMatMul",
            "packedCall",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"),
            IsWildcard("scale"),
            IsNone());
        var viewedPackedMatMul = IsShardedView(
            "view",
            _ => true,
            packedMatMul);

        Pattern = IsAlt(
            IsBinary(
                "binary",
                "binaryCall",
                op => op.BinaryOp == BinaryOp.Add,
                viewedPackedMatMul,
                IsWildcard("addend")),
            IsBinary(
                "binary",
                "binaryCall",
                op => op.BinaryOp == BinaryOp.Add,
                IsWildcard("addend"),
                viewedPackedMatMul));
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(
        PackedMatMul packedMatMul,
        Call packedCall,
        Expr lhs,
        Expr rhs,
        Expr scale,
        ShardedView view,
        Call binaryCall,
        Expr addend)
    {
        if (packedMatMul.FusedReduce ||
            packedCall.CheckedType is not DistributedType { Partial: null } packedType ||
            view.NewType is not DistributedType viewType ||
            binaryCall.CheckedType is not DistributedType binaryType ||
            !Equals(viewType, binaryType) ||
            !Equals(addend.CheckedType, binaryType) ||
            ReferenceEquals(addend, packedCall))
        {
            return null;
        }

        var packedUsers = packedCall.Users.ToArray();
        if (packedUsers.Length != 1 ||
            packedUsers[0] is not Call { Target: ShardedView } viewCall ||
            !ReferenceEquals(viewCall.Target, view) ||
            viewCall.Users.Count() != 1 ||
            !ReferenceEquals(viewCall.Users.Single(), binaryCall))
        {
            return null;
        }

        var localAddend = IR.F.Distributed.ShardedView(addend, packedType);
        if (localAddend.CheckedType is InvalidType || !Equals(localAddend.CheckedType, packedType))
        {
            return null;
        }

        var fused = IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            packedMatMul.FusedReduce,
            packedMatMul.OutputDataType,
            scale,
            packedMatMul.RhsLayout,
            localAddend).InheritMetaData(packedCall);
        if (fused.CheckedType is InvalidType || !Equals(fused.CheckedType, packedType))
        {
            return null;
        }

        var result = IR.F.Distributed.ShardedView(fused, binaryType).InheritMetaData(binaryCall);
        return result.CheckedType is not InvalidType && Equals(result.CheckedType, binaryType)
            ? result
            : null;
    }
}
