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
