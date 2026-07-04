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
public sealed partial class PackQKVParallelLinearByN : RewriteRule<Pattern>
{
    private readonly int _nr;
    private readonly int _laneBytes;

    public PackQKVParallelLinearByN(int nr, int laneBytes)
    {
        _nr = nr;
        _laneBytes = laneBytes;
    }

    public override Pattern Pattern { get; } =
        IsQKVParallelLinear(
            "qkv",
            "caller",
            _ => true,
            IsWildcard("input"),
            IsWildcard("qWeight"),
            IsWildcard("kWeight"),
            IsWildcard("vWeight"),
            IsWildcard("qBias"),
            IsWildcard("kBias"),
            IsWildcard("vBias"),
            IsWildcard("qInputScale"),
            IsWildcard("kInputScale"),
            IsWildcard("vInputScale"),
            IsWildcard("qWeightScale"),
            IsWildcard("kWeightScale"),
            IsWildcard("vWeightScale"));

    private BaseExpr? GetReplace(
        QKVParallelLinear qkv,
        Call caller,
        Expr input,
        Expr qWeight,
        Expr kWeight,
        Expr vWeight,
        Expr qBias,
        Expr kBias,
        Expr vBias,
        Expr qInputScale,
        Expr kInputScale,
        Expr vInputScale,
        Expr qWeightScale,
        Expr kWeightScale,
        Expr vWeightScale)
    {
        if (input.CheckedDataType is VectorType ||
            input.CheckedDataType == DataTypes.Float8E4M3 ||
            input.CheckedDataType == DataTypes.Float8E5M2 ||
            !IsNone(qInputScale) ||
            !IsNone(kInputScale) ||
            !IsNone(vInputScale) ||
            !IsNone(qWeightScale) ||
            !IsNone(kWeightScale) ||
            !IsNone(vWeightScale))
        {
            return null;
        }

        var laneCount = GetLaneCount(qWeight);
        if (laneCount <= 0 ||
            laneCount != GetLaneCount(kWeight) ||
            laneCount != GetLaneCount(vWeight) ||
            !TryPackWeight(qWeight, laneCount, out var packedQWeight) ||
            !TryPackWeight(kWeight, laneCount, out var packedKWeight) ||
            !TryPackWeight(vWeight, laneCount, out var packedVWeight) ||
            !TryPackBias(qBias, laneCount, out var packedQBias) ||
            !TryPackBias(kBias, laneCount, out var packedKBias) ||
            !TryPackBias(vBias, laneCount, out var packedVBias))
        {
            return null;
        }

        if (caller.CheckedType is not TupleType { Fields.Count: 3 } tupleType)
        {
            return null;
        }

        var packed = IR.F.NTT.PackedQKVParallelLinear(
            input,
            packedQWeight,
            packedKWeight,
            packedVWeight,
            packedQBias,
            packedKBias,
            packedVBias,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            qkv.NumHeads,
            qkv.NumKvHeads,
            qkv.OutputDataType);

        return new IR.Tuple(
            UnpackOutput(packed, 0, GetRank(tupleType.Fields[0]), laneCount),
            UnpackOutput(packed, 1, GetRank(tupleType.Fields[1]), laneCount),
            UnpackOutput(packed, 2, GetRank(tupleType.Fields[2]), laneCount));
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

    private Expr UnpackOutput(Expr packed, int index, int rank, int laneCount)
    {
        Expr output = IR.F.Tensors.GetItem(packed, index);
        output = IR.F.Tensors.Unpack(output, [_nr], [rank - 1]);
        output = IR.F.Tensors.Unpack(output, [laneCount], [rank - 1]);
        return output;
    }

    private int GetRank(IRType type) => type switch
    {
        TensorType tensor => tensor.Shape.Rank,
        DistributedType distributed => distributed.TensorType.Shape.Rank,
        _ => throw new NotSupportedException($"PackedQKVParallelLinear output should be tensor-like, got {type}."),
    };
}
