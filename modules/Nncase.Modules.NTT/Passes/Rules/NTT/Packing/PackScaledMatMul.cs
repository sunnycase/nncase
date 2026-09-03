// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

/// <summary>
/// Packs an E4M3 scaled matmul RHS into the GPU K-major ABI.
/// </summary>
[RuleGenerator]
public sealed partial class PackScaledMatMulRhsKMajor : RewriteRule<Pattern>
{
    private readonly int _vectorBytes;
    private readonly int _kPack;

    public PackScaledMatMulRhsKMajor(int vectorBytes, int kPack)
    {
        _vectorBytes = vectorBytes > 0
            ? vectorBytes
            : throw new ArgumentOutOfRangeException(nameof(vectorBytes));
        _kPack = kPack > 0 ? kPack : throw new ArgumentOutOfRangeException(nameof(kPack));
    }

    public override Pattern Pattern { get; } = IsScaledMatMul(
        "scaledMatMul",
        "caller",
        _ => true,
        IsWildcard("lhs"),
        IsWildcard("rhs"),
        IsWildcard("lhsScale"),
        IsWildcard("rhsScale"));

    private Expr? GetReplace(
        ScaledMatMul scaledMatMul,
        Call caller,
        Expr lhs,
        Expr rhs,
        Expr lhsScale,
        Expr rhsScale)
    {
        if (lhs.CheckedDataType is not PrimType lhsType ||
            rhs.CheckedDataType != DataTypes.Float8E4M3 ||
            scaledMatMul.OutputDataType is not PrimType outputType ||
            rhs.CheckedShape.IsUnranked ||
            rhs.CheckedShape.Rank < 2 ||
            _vectorBytes % rhs.CheckedDataType.SizeInBytes != 0 ||
            _vectorBytes % outputType.SizeInBytes != 0)
        {
            return null;
        }

        var kVectorLanes = _vectorBytes / rhs.CheckedDataType.SizeInBytes;
        var nVectorLanes = _vectorBytes / outputType.SizeInBytes;
        if (kVectorLanes <= 0 || nVectorLanes <= 0 ||
            !Dimension.TryDivExactly(rhs.CheckedShape[^2], checked(_kPack * kVectorLanes), out _) ||
            !Dimension.TryDivExactly(rhs.CheckedShape[^1], nVectorLanes, out _))
        {
            return null;
        }

        var rank = rhs.CheckedShape.Rank;
        var permutation = Enumerable.Range(0, rank).ToArray();
        (permutation[^2], permutation[^1]) = (permutation[^1], permutation[^2]);
        Expr packedRhs = IR.F.Tensors.Transpose(rhs, permutation);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [kVectorLanes], [rank - 1]);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [_kPack], [rank - 1]);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [nVectorLanes], [rank - 2]);
        packedRhs = IR.F.Tensors.Transpose(packedRhs, permutation);
        if (packedRhs.CheckedType is InvalidType)
        {
            return null;
        }

        var packed = IR.F.NTT.PackedScaledMatMul(
            lhs,
            packedRhs,
            lhsScale,
            rhsScale,
            scaledMatMul.OutputDataType).InheritMetaData(caller);
        return IR.F.Tensors.Unpack(
            packed,
            [nVectorLanes],
            [caller.CheckedShape.Rank - 1]);
    }
}

/// <summary>
/// Packs an E4M3 block-scaled matmul RHS into a row-major scalar [N, K]
/// physical layout that can be staged directly into an MMA operand.
/// </summary>
[RuleGenerator]
public sealed partial class PackBlockScaledMatMulRhsNMajorKPacked : RewriteRule<Pattern>
{
    private readonly int _vectorBytes;
    private readonly int _kPack;

    public PackBlockScaledMatMulRhsNMajorKPacked(int vectorBytes, int kPack)
    {
        _vectorBytes = vectorBytes > 0
            ? vectorBytes
            : throw new ArgumentOutOfRangeException(nameof(vectorBytes));
        _kPack = kPack > 0 ? kPack : throw new ArgumentOutOfRangeException(nameof(kPack));
    }

    public override Pattern Pattern { get; } = IsBlockScaledMatMul(
        "blockScaledMatMul",
        "caller",
        _ => true,
        IsWildcard("lhs"),
        IsWildcard("rhs"),
        IsWildcard("rhsScale"));

    private Expr? GetReplace(
        BlockScaledMatMul blockScaledMatMul,
        Call caller,
        Expr lhs,
        Expr rhs,
        Expr rhsScale)
    {
        if (lhs.CheckedDataType is not PrimType ||
            rhs.CheckedDataType != DataTypes.Float8E4M3 ||
            blockScaledMatMul.OutputDataType is not PrimType outputType ||
            rhs.CheckedShape.IsUnranked ||
            rhs.CheckedShape.Rank < 2 ||
            _vectorBytes % rhs.CheckedDataType.SizeInBytes != 0 ||
            _vectorBytes % outputType.SizeInBytes != 0)
        {
            return null;
        }

        var kVectorLanes = _vectorBytes / rhs.CheckedDataType.SizeInBytes;
        var nVectorLanes = _vectorBytes / outputType.SizeInBytes;
        if (kVectorLanes <= 0 || nVectorLanes <= 0 ||
            !Dimension.TryDivExactly(rhs.CheckedShape[^2], checked(_kPack * kVectorLanes), out _) ||
            !Dimension.TryDivExactly(rhs.CheckedShape[^1], nVectorLanes, out _))
        {
            return null;
        }

        var rank = rhs.CheckedShape.Rank;
        var permutation = Enumerable.Range(0, rank).ToArray();
        (permutation[^2], permutation[^1]) = (permutation[^1], permutation[^2]);
        Expr packedRhs = IR.F.Tensors.Transpose(rhs, permutation);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [kVectorLanes], [rank - 1]);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [_kPack], [rank - 1]);
        if (packedRhs.CheckedType is InvalidType)
        {
            return null;
        }

        var packed = IR.F.NTT.PackedBlockScaledMatMul(
            lhs,
            packedRhs,
            rhsScale,
            blockScaledMatMul.OutputDataType,
            blockScaledMatMul.WeightBlockN,
            blockScaledMatMul.WeightBlockK,
            IR.NTT.PackedMatMulRhsLayout.NMajorKPacked,
            nVectorLanes).InheritMetaData(caller);
        return IR.F.Tensors.Unpack(
            packed,
            [nVectorLanes],
            [caller.CheckedShape.Rank - 1]);
    }
}
