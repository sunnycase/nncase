// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NTT;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class PackMatMulByN : RewriteRule<Pattern>
{
    private readonly int _nr;

    public PackMatMulByN(int nr)
    {
        _nr = nr;
    }

    public override Pattern Pattern { get; } =
        IsVectorizedMatMul(
            "matMul",
            "caller",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"),
            IsNone());

    private Expr? GetReplace(VectorizedMatMul matMul, Call caller, Expr lhs, Expr rhs)
    {
        if (lhs.CheckedDataType == DataTypes.Float8E4M3 || lhs.CheckedDataType == DataTypes.Float8E5M2)
        {
            return null;
        }

        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var dimInfo = matMul.GetDimInfo(lhsShape.Rank, rhsShape.Rank);
        (var lhsVectorizeKind, var rhsVectorizeKind) = matMul.GetVectorizeKind(lhsShape.Rank, rhsShape.Rank);
        if (lhsVectorizeKind == VectorizedMatMul.VectorizeKind.None && rhsVectorizeKind == VectorizedMatMul.VectorizeKind.N
            && !matMul.TransposeA && !matMul.TransposeB
            && rhs.CheckedDataType is VectorType rhsVectorType
            && rhsVectorType.Lanes.Count == 1)
        {
            var cN = Math.Max(lhsShape.Rank, rhsShape.Rank) - 1;
            if (!Dimension.TryDivExactly(rhsShape[dimInfo.Rn], _nr, out _))
            {
                return null;
            }

            // 1. Transpose B outer dimensions to [N/lanes, K].
            var newRhsPerm = Enumerable.Range(0, rhsShape.Rank).ToArray();
            (newRhsPerm[^2], newRhsPerm[^1]) = (newRhsPerm[^1], newRhsPerm[^2]);
            Expr newRhs = IR.F.Tensors.Transpose(rhs, newRhsPerm);

            // 2. Pack B's N axis to vector<Nr, lanes>.
            var rN = rhsShape.Rank - 2;
            newRhs = IR.F.Tensors.Pack(newRhs, [_nr], [rN]);

            var output = IR.F.NTT.PackedMatMul(
                lhs,
                newRhs,
                false,
                matMul.OutputDataType);

            // 3. Unpack only the packed-N lane, preserving the original N vector lane.
            return IR.F.Tensors.Unpack(output, [_nr], [cN]);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class PackMatMulRhsKMajor : RewriteRule<Pattern>
{
    private readonly int _vectorBytes;
    private readonly int _kPack;

    public PackMatMulRhsKMajor(int vectorBytes, int kPack)
    {
        if (vectorBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(vectorBytes));
        }

        if (kPack <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(kPack));
        }

        _vectorBytes = vectorBytes;
        _kPack = kPack;
    }

    public override Pattern Pattern { get; } =
        IsVectorizedMatMul(
            "matMul",
            "caller",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"),
            IsNone());

    private Expr? GetReplace(VectorizedMatMul matMul, Call caller, Expr lhs, Expr rhs)
    {
        if (lhs.CheckedDataType == DataTypes.Float8E4M3 || lhs.CheckedDataType == DataTypes.Float8E5M2)
        {
            return null;
        }

        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        if (lhs.CheckedDataType is VectorType ||
            rhs.CheckedDataType is not VectorType { Lanes.Count: 1 } rhsVectorType ||
            rhsVectorType.ElemType.SizeInBytes <= 0 ||
            _vectorBytes % rhsVectorType.ElemType.SizeInBytes != 0)
        {
            return null;
        }

        var dimInfo = matMul.GetDimInfo(lhsShape.Rank, rhsShape.Rank);
        (var lhsVectorizeKind, var rhsVectorizeKind) = matMul.GetVectorizeKind(lhsShape.Rank, rhsShape.Rank);
        var kVectorLane = _vectorBytes / rhsVectorType.ElemType.SizeInBytes;
        var nVectorLane = rhsVectorType.Lanes[0];
        if (lhsVectorizeKind != VectorizedMatMul.VectorizeKind.None ||
            rhsVectorizeKind != VectorizedMatMul.VectorizeKind.N ||
            matMul.TransposeA ||
            matMul.TransposeB ||
            nVectorLane != kVectorLane ||
            !Dimension.TryDivExactly(rhsShape[dimInfo.Rk], checked(_kPack * kVectorLane), out _))
        {
            return null;
        }

        // Recover scalar [..., K, N], form [..., N, K], then pack it as
        // [..., N/K lanes, K/K lanes]<NVector, KPack, KVector>. Transposing
        // the two tensor axes back yields [..., K, N]<NVector,KPack,KVector>.
        var rhsRank = rhsShape.Rank;
        Expr packedRhs = IR.F.Tensors.Unpack(rhs, [nVectorLane], [dimInfo.Rn]);
        var permutation = Enumerable.Range(0, rhsRank).ToArray();
        (permutation[^2], permutation[^1]) = (permutation[^1], permutation[^2]);
        packedRhs = IR.F.Tensors.Transpose(packedRhs, permutation);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [kVectorLane], [rhsRank - 1]);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [_kPack], [rhsRank - 1]);
        packedRhs = IR.F.Tensors.Pack(packedRhs, [nVectorLane], [rhsRank - 2]);
        packedRhs = IR.F.Tensors.Transpose(packedRhs, permutation);
        if (packedRhs.CheckedType is InvalidType)
        {
            return null;
        }

        return IR.F.NTT.PackedMatMul(
            lhs,
            packedRhs,
            matMul.FusedReduce,
            matMul.OutputDataType,
            rhsLayout: PackedMatMulRhsLayout.KMajor);
    }
}
