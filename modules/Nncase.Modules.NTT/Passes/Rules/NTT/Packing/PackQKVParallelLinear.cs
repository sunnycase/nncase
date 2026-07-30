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

[RuleGenerator]
public sealed partial class PackQKVParallelLinearRhsKMajor : RewriteRule<Pattern>
{
    private readonly int _vectorBytes;
    private readonly int _kPack;

    public PackQKVParallelLinearRhsKMajor(int vectorBytes, int kPack)
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
        if (input.CheckedDataType is not PrimType inputType ||
            inputType == DataTypes.Float8E4M3 ||
            inputType == DataTypes.Float8E5M2 ||
            !IsNone(qInputScale) ||
            !IsNone(kInputScale) ||
            !IsNone(vInputScale) ||
            !IsNone(qWeightScale) ||
            !IsNone(kWeightScale) ||
            !IsNone(vWeightScale) ||
            _vectorBytes % inputType.SizeInBytes != 0)
        {
            return null;
        }

        var vectorLanes = _vectorBytes / inputType.SizeInBytes;
        if (vectorLanes <= 0 ||
            !TryPackWeight(qWeight, inputType, vectorLanes, out var packedQWeight) ||
            !TryPackWeight(kWeight, inputType, vectorLanes, out var packedKWeight) ||
            !TryPackWeight(vWeight, inputType, vectorLanes, out var packedVWeight) ||
            !TryPackBias(qBias, inputType, vectorLanes, out var packedQBias) ||
            !TryPackBias(kBias, inputType, vectorLanes, out var packedKBias) ||
            !TryPackBias(vBias, inputType, vectorLanes, out var packedVBias) ||
            caller.CheckedType is not TupleType { Fields.Count: 3 } tupleType)
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
            qkv.OutputDataType,
            rhsLayout: IR.NTT.PackedMatMulRhsLayout.KMajor);

        return new IR.Tuple(
            UnpackOutput(packed, 0, GetRank(tupleType.Fields[0]), vectorLanes),
            UnpackOutput(packed, 1, GetRank(tupleType.Fields[1]), vectorLanes),
            UnpackOutput(packed, 2, GetRank(tupleType.Fields[2]), vectorLanes));
    }

    private static bool IsNone(Expr expr) => expr is None;

    private bool TryPackWeight(
        Expr weight,
        PrimType inputType,
        int vectorLanes,
        out Expr packedWeight)
    {
        packedWeight = weight;
        if (weight.CheckedDataType != inputType ||
            weight.CheckedShape.IsUnranked ||
            weight.CheckedShape.Rank < 2 ||
            !Dimension.TryDivExactly(weight.CheckedShape[^2], checked(_kPack * vectorLanes), out _) ||
            !Dimension.TryDivExactly(weight.CheckedShape[^1], vectorLanes, out _))
        {
            return false;
        }

        var rank = weight.CheckedShape.Rank;
        var permutation = Enumerable.Range(0, rank).ToArray();
        (permutation[^2], permutation[^1]) = (permutation[^1], permutation[^2]);

        // [K,N] -> [N,K]
        // -> [N/NVector,K/(KPack*KVector)]<NVector,KPack,KVector>
        // -> [K/(KPack*KVector),N/NVector]<NVector,KPack,KVector>.
        packedWeight = IR.F.Tensors.Transpose(weight, permutation);
        packedWeight = IR.F.Tensors.Pack(packedWeight, [vectorLanes], [rank - 1]);
        packedWeight = IR.F.Tensors.Pack(packedWeight, [_kPack], [rank - 1]);
        packedWeight = IR.F.Tensors.Pack(packedWeight, [vectorLanes], [rank - 2]);
        packedWeight = IR.F.Tensors.Transpose(packedWeight, permutation);
        return packedWeight.CheckedType is not InvalidType;
    }

    private static bool TryPackBias(
        Expr bias,
        PrimType inputType,
        int vectorLanes,
        out Expr packedBias)
    {
        packedBias = bias;
        if (IsNone(bias))
        {
            return true;
        }

        if (bias.CheckedDataType != inputType ||
            bias.CheckedShape.IsUnranked ||
            bias.CheckedShape.Rank != 1 ||
            !Dimension.TryDivExactly(bias.CheckedShape[0], vectorLanes, out _))
        {
            return false;
        }

        packedBias = IR.F.Tensors.Pack(bias, [vectorLanes], [0]);
        return packedBias.CheckedType is not InvalidType;
    }

    private static Expr UnpackOutput(Expr packed, int index, int rank, int vectorLanes)
    {
        Expr output = IR.F.Tensors.GetItem(packed, index);
        return IR.F.Tensors.Unpack(output, [vectorLanes], [rank - 1]);
    }

    private static int GetRank(IRType type) => type switch
    {
        TensorType tensor => tensor.Shape.Rank,
        DistributedType distributed => distributed.TensorType.Shape.Rank,
        _ => throw new NotSupportedException($"PackedQKVParallelLinear output should be tensor-like, got {type}."),
    };
}
