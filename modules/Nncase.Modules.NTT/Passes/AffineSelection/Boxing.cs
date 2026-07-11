// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectBoxing(Boxing op, Call call, Expr output)
    {
        var input = (Expr)call[Boxing.Input];
        var inputTensorType = GetTensorType(input.CheckedType);
        var outputTensorType = GetTensorType(op.NewType);
        if (inputTensorType.DType != outputTensorType.DType || inputTensorType.Shape != outputTensorType.Shape)
        {
            throw new InvalidOperationException($"Boxing requires identical logical tensor types, got {input.CheckedType} -> {op.NewType}.");
        }

        var rank = inputTensorType.Shape.Rank;
        if (rank <= 0)
        {
            throw new NotSupportedException("Scalar Boxing cannot be tiled as an affine transfer.");
        }

        var inputConstrainsDomain = input.CheckedType is DistributedType;
        var inputDomainMode = inputConstrainsDomain ? GridDomainMode.Constraint : GridDomainMode.Footprint;
        var outputDomainMode = inputConstrainsDomain ? GridDomainMode.Footprint : GridDomainMode.Constraint;
        var builder = IR.F.Affine.Grid().Domain(rank, out var _);
        Var inputTile;
        builder = input.CheckedType is TensorType
            ? builder.ReadRoot(input, AffineMap.Identity(rank), inputDomainMode, out inputTile)
            : builder.Read(input, AffineMap.Identity(rank), inputDomainMode, out inputTile);
        builder = builder.WriteRoot(output, AffineMap.Identity(rank), outputDomainMode, out var outputRoot);

        var transfer = (input.CheckedType, op.NewType) switch
        {
            (TensorType, DistributedType outputDistributed) =>
                TIR.F.NTT.TensorLoad(outputRoot, inputTile, outputDistributed.AxisPolicies, outputDistributed.Placement),
            (DistributedType inputDistributed, TensorType) =>
                TIR.F.NTT.TensorStore(inputTile, outputRoot, inputDistributed.AxisPolicies, inputDistributed.Placement),
            (DistributedType inputDistributed, DistributedType outputDistributed) =>
                TIR.F.NTT.GatherReduceScatter(inputTile, outputRoot, inputDistributed, outputDistributed),
            _ => throw new NotSupportedException($"Unsupported Boxing types: {input.CheckedType} -> {op.NewType}."),
        };

        return builder.Body(transfer).Build();
    }

    private static TensorType GetTensorType(IRType type) => type switch
    {
        TensorType tensorType => tensorType,
        DistributedType distributedType => distributedType.TensorType,
        _ => throw new NotSupportedException($"Boxing requires tensor or distributed tensor types, got {type}."),
    };
}
