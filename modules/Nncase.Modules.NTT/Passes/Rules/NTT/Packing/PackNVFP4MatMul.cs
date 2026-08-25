// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;

using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

internal readonly record struct NVFP4PackingContract(
    int InputLanes,
    int WeightVectorLanes,
    int OutputLanes,
    int KPack);

/// <summary>
/// Rewrites semantic NVFP4 matmul into the PyNTT target-packed ABI.
/// </summary>
[RuleGenerator]
public sealed partial class PackNVFP4MatMulRhsKMajor : RewriteRule<Pattern>
{
    private readonly NVFP4PackingLayout _layout;

    public PackNVFP4MatMulRhsKMajor(int vectorBytes, int kPack)
    {
        _layout = new NVFP4PackingLayout(vectorBytes, kPack);
    }

    public override Pattern Pattern { get; } = IsNVFP4MatMul(
        "matmul",
        "caller",
        _ => true,
        IsWildcard("lhs"),
        IsWildcard("rhsPacked"),
        IsWildcard("rhsScale"),
        IsWildcard("lhsGlobalScale"),
        IsWildcard("rhsGlobalScale"));

    private Expr? GetReplace(
        NVFP4MatMul matmul,
        Call caller,
        Expr lhs,
        Expr rhsPacked,
        Expr rhsScale,
        Expr lhsGlobalScale,
        Expr rhsGlobalScale)
    {
        if (!_layout.TryGetContract(
                lhs,
                rhsPacked,
                matmul.OutputDataType,
                matmul.GroupSize,
                out var contract) ||
            !_layout.TryPackInput(lhs, contract, out var packedLhs) ||
            !_layout.TryPackWeight(rhsPacked, contract, out var targetPackedRhs))
        {
            return null;
        }

        var packed = IR.F.NTT.PackedNVFP4MatMul(
            packedLhs,
            targetPackedRhs,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            matmul.OutputDataType,
            matmul.GroupSize,
            contract.InputLanes,
            contract.KPack,
            contract.WeightVectorLanes,
            contract.OutputLanes);
        return _layout.UnpackOutput(packed, caller.CheckedShape.Rank, contract.OutputLanes);
    }
}

/// <summary>
/// Rewrites semantic NVFP4 GLU into the PyNTT target-packed ABI.
/// </summary>
[RuleGenerator]
public sealed partial class PackNVFP4MatMulGluRhsKMajor : RewriteRule<Pattern>
{
    private readonly NVFP4PackingLayout _layout;

    public PackNVFP4MatMulGluRhsKMajor(int vectorBytes, int kPack)
    {
        _layout = new NVFP4PackingLayout(vectorBytes, kPack);
    }

    public override Pattern Pattern { get; } = IsNVFP4MatMulGlu(
        "matmulGlu",
        "caller",
        _ => true,
        IsWildcard("input"),
        IsWildcard("gateWeightPacked"),
        IsWildcard("upWeightPacked"),
        IsWildcard("gateWeightScale"),
        IsWildcard("upWeightScale"),
        IsWildcard("gateInputGlobalScale"),
        IsWildcard("upInputGlobalScale"),
        IsWildcard("gateWeightGlobalScale"),
        IsWildcard("upWeightGlobalScale"));

    private Expr? GetReplace(
        NVFP4MatMulGlu matmulGlu,
        Call caller,
        Expr input,
        Expr gateWeightPacked,
        Expr upWeightPacked,
        Expr gateWeightScale,
        Expr upWeightScale,
        Expr gateInputGlobalScale,
        Expr upInputGlobalScale,
        Expr gateWeightGlobalScale,
        Expr upWeightGlobalScale)
    {
        if (!_layout.TryGetContract(
                input,
                gateWeightPacked,
                matmulGlu.OutputDataType,
                matmulGlu.GroupSize,
                out var contract) ||
            !_layout.HasMatchingWeightShape(gateWeightPacked, upWeightPacked) ||
            !_layout.TryPackInput(input, contract, out var packedInput) ||
            !_layout.TryPackWeight(gateWeightPacked, contract, out var packedGateWeight) ||
            !_layout.TryPackWeight(upWeightPacked, contract, out var packedUpWeight))
        {
            return null;
        }

        var packed = IR.F.NTT.PackedNVFP4MatMulGlu(
            packedInput,
            packedGateWeight,
            packedUpWeight,
            gateWeightScale,
            upWeightScale,
            gateInputGlobalScale,
            upInputGlobalScale,
            gateWeightGlobalScale,
            upWeightGlobalScale,
            matmulGlu.GluType,
            matmulGlu.OutputDataType,
            matmulGlu.GroupSize,
            contract.InputLanes,
            contract.KPack,
            contract.WeightVectorLanes,
            contract.OutputLanes);
        return _layout.UnpackOutput(packed, caller.CheckedShape.Rank, contract.OutputLanes);
    }
}

internal sealed class NVFP4PackingLayout
{
    private const int NVFP4GroupSize = 16;

    private readonly int _vectorBytes;
    private readonly int _kPack;

    public NVFP4PackingLayout(int vectorBytes, int kPack)
    {
        _vectorBytes = vectorBytes > 0
            ? vectorBytes
            : throw new ArgumentOutOfRangeException(nameof(vectorBytes));
        _kPack = kPack > 0
            ? kPack
            : throw new ArgumentOutOfRangeException(nameof(kPack));
    }

    public bool TryGetContract(
        Expr input,
        Expr weightPacked,
        DataType outputDataType,
        long groupSize,
        out NVFP4PackingContract contract)
    {
        contract = default;
        if (input.CheckedDataType != DataTypes.BFloat16 ||
            weightPacked.CheckedDataType != DataTypes.UInt8 ||
            outputDataType != DataTypes.BFloat16 ||
            groupSize != NVFP4GroupSize ||
            input.CheckedShape.IsUnranked || input.CheckedShape.Rank < 2 ||
            weightPacked.CheckedShape.IsUnranked || weightPacked.CheckedShape.Rank != 2 ||
            _vectorBytes % DataTypes.BFloat16.SizeInBytes != 0 ||
            _vectorBytes % DataTypes.UInt8.SizeInBytes != 0)
        {
            return false;
        }

        var inputLanes = _vectorBytes / DataTypes.BFloat16.SizeInBytes;
        var weightVectorLanes = _vectorBytes / DataTypes.UInt8.SizeInBytes;
        var outputLanes = _vectorBytes / DataTypes.BFloat16.SizeInBytes;
        if (inputLanes <= 0 || weightVectorLanes <= 0 || outputLanes <= 0 ||
            !Dimension.TryDivExactly(input.CheckedShape[^1], inputLanes, out var packedInputK) ||
            !Dimension.TryDivExactly(
                weightPacked.CheckedShape[^1],
                checked(_kPack * weightVectorLanes),
                out _) ||
            !Dimension.TryDivExactly(weightPacked.CheckedShape[0], outputLanes, out _) ||
            packedInputK * inputLanes / 2 != weightPacked.CheckedShape[^1])
        {
            return false;
        }

        contract = new NVFP4PackingContract(
            inputLanes,
            weightVectorLanes,
            outputLanes,
            _kPack);
        return true;
    }

    public bool TryPackInput(
        Expr input,
        NVFP4PackingContract contract,
        out Expr packedInput)
    {
        var rank = input.CheckedShape.Rank;
        if (input is Call { Target: Unpack unpack } unpackCall &&
            unpack.Axes.SequenceEqual([rank - 1]) &&
            unpack.Lanes.SequenceEqual([contract.InputLanes]) &&
            unpackCall.Arguments[Unpack.Input.Index] is Expr packed &&
            packed.CheckedDataType == new VectorType(DataTypes.BFloat16, [contract.InputLanes]))
        {
            packedInput = packed;
            return true;
        }

        packedInput = IR.F.Tensors.Pack(input, [contract.InputLanes], [rank - 1]);
        return packedInput.CheckedType is not InvalidType;
    }

    public bool TryPackWeight(
        Expr weightPacked,
        NVFP4PackingContract contract,
        out Expr targetPackedWeight)
    {
        var kAxis = weightPacked.CheckedShape.Rank - 1;
        targetPackedWeight = IR.F.Tensors.Pack(
            weightPacked,
            [contract.WeightVectorLanes],
            [kAxis]);
        targetPackedWeight = IR.F.Tensors.Pack(
            targetPackedWeight,
            [contract.KPack],
            [kAxis]);
        return targetPackedWeight.CheckedType is not InvalidType;
    }

    public bool HasMatchingWeightShape(Expr lhs, Expr rhs) =>
        lhs.CheckedDataType == rhs.CheckedDataType && lhs.CheckedShape == rhs.CheckedShape;

    public Expr UnpackOutput(Expr packed, int rank, int lanes) =>
        IR.F.Tensors.Unpack(packed, [lanes], [rank - 1]);
}
