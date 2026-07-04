// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class PackMatMulGluByN : RewriteRule<Pattern>
{
    private readonly int _nr;
    private readonly int _laneBytes;

    public PackMatMulGluByN(int nr, int laneBytes)
    {
        _nr = nr;
        _laneBytes = laneBytes;
    }

    public override Pattern Pattern { get; } =
        IsMatMulGlu(
            "matmulGlu",
            "caller",
            _ => true,
            IsWildcard("input"),
            IsWildcard("gateWeight"),
            IsWildcard("upWeight"),
            IsWildcard("gateBias"),
            IsWildcard("upBias"),
            IsWildcard("gateInputScale"),
            IsWildcard("upInputScale"),
            IsWildcard("gateWeightScale"),
            IsWildcard("upWeightScale"));

    private BaseExpr? GetReplace(
        MatMulGlu matmulGlu,
        Call caller,
        Expr input,
        Expr gateWeight,
        Expr upWeight,
        Expr gateBias,
        Expr upBias,
        Expr gateInputScale,
        Expr upInputScale,
        Expr gateWeightScale,
        Expr upWeightScale)
    {
        if (input.CheckedDataType is VectorType ||
            input.CheckedDataType == DataTypes.Float8E4M3 ||
            input.CheckedDataType == DataTypes.Float8E5M2 ||
            !IsNone(gateInputScale) ||
            !IsNone(upInputScale) ||
            !IsNone(gateWeightScale) ||
            !IsNone(upWeightScale))
        {
            return null;
        }

        var laneCount = GetLaneCount(gateWeight);
        if (laneCount <= 0 ||
            laneCount != GetLaneCount(upWeight) ||
            !TryPackWeight(gateWeight, laneCount, out var packedGateWeight) ||
            !TryPackWeight(upWeight, laneCount, out var packedUpWeight) ||
            !TryPackBias(gateBias, laneCount, out var packedGateBias) ||
            !TryPackBias(upBias, laneCount, out var packedUpBias))
        {
            return null;
        }

        var rank = GetRank(caller.CheckedType);
        var packed = IR.F.NTT.PackedMatMulGlu(
            input,
            packedGateWeight,
            packedUpWeight,
            packedGateBias,
            packedUpBias,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            matmulGlu.GluType,
            matmulGlu.OutputDataType);

        return UnpackOutput(packed, rank, laneCount);
    }

    private bool IsNone(Expr expr) => expr is None;

    private int GetLaneCount(Expr expr)
    {
        return expr.CheckedDataType is PrimType { SizeInBytes: > 0 } dtype
            ? _laneBytes / dtype.SizeInBytes
            : -1;
    }

    private bool TryPackWeight(Expr weight, int laneCount, out Expr packedWeight)
    {
        packedWeight = weight;
        if (weight.CheckedDataType is not PrimType ||
            weight.CheckedShape.IsUnranked ||
            weight.CheckedShape.Rank < 2 ||
            !Dimension.TryDivExactly(weight.CheckedShape[^1], laneCount, out _) ||
            !Dimension.TryDivExactly(weight.CheckedShape[^1], checked(laneCount * _nr), out _))
        {
            return false;
        }

        var rank = weight.CheckedShape.Rank;
        packedWeight = IR.F.Tensors.Pack(weight, [laneCount], [rank - 1]);
        var perm = Enumerable.Range(0, rank).ToArray();
        (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
        packedWeight = IR.F.Tensors.Transpose(packedWeight, perm);
        packedWeight = IR.F.Tensors.Pack(packedWeight, [_nr], [rank - 2]);
        return packedWeight.CheckedType is not InvalidType;
    }

    private bool TryPackBias(Expr bias, int laneCount, out Expr packedBias)
    {
        packedBias = bias;
        if (IsNone(bias))
        {
            return true;
        }

        if (bias.CheckedDataType is not PrimType ||
            bias.CheckedShape.IsUnranked ||
            bias.CheckedShape.Rank != 1 ||
            !Dimension.TryDivExactly(bias.CheckedShape[0], laneCount, out _) ||
            !Dimension.TryDivExactly(bias.CheckedShape[0], checked(laneCount * _nr), out _))
        {
            return false;
        }

        packedBias = IR.F.Tensors.Pack(bias, [laneCount], [0]);
        packedBias = IR.F.Tensors.Pack(packedBias, [_nr], [0]);
        return packedBias.CheckedType is not InvalidType;
    }

    private Expr UnpackOutput(Expr packed, int rank, int laneCount)
    {
        Expr output = IR.F.Tensors.Unpack(packed, [_nr], [rank - 1]);
        output = IR.F.Tensors.Unpack(output, [laneCount], [rank - 1]);
        return output;
    }

    private int GetRank(IRType type) => type switch
    {
        TensorType tensor => tensor.Shape.Rank,
        DistributedType distributed => distributed.TensorType.Shape.Rank,
        _ => throw new NotSupportedException($"PackedMatMulGlu output should be tensor-like, got {type}."),
    };
}
