// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class ReduceEvaluator : ITypeInferencer<Reduce>, ITileWorkloadEvaluator<Reduce>
{
    public IRType Visit(ITypeInferenceContext context, Reduce target)
    {
        context.CheckArgumentType<TensorType>(target, Reduce.Input);
        context.CheckArgumentType<TensorType>(target, Reduce.Output);
        return TupleType.Void;
    }

    public TileWorkload Visit(Reduce op, TileWorkloadContext context) => new ReductionTileWorkload(GetComputeWork, GetReductionStateBytes);

    private static IntExpr GetComputeWork(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
    }

    private static IntExpr GetReductionStateBytes(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        var outputElements = bufferShapes[1]
            .Aggregate((IntExpr)solver.MakeIntConst(1), (product, extent) => solver.MakeProd(product, extent));
        var scalarInputType = GetScalarDataType(context.BufferDataTypes[0]);
        var accumulatorElementBytes = DataTypes.IsFloat(scalarInputType)
            ? DataTypes.Float32.SizeInBytes
            : scalarInputType.SizeInBytes;
        var elementStateBytes = outputElements * accumulatorElementBytes;
        return ((Reduce)context.Op).ReduceOp == ReduceOp.Mean
            ? elementStateBytes + DataTypes.Int64.SizeInBytes
            : elementStateBytes;
    }

    private static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vectorType
            ? GetScalarDataType(vectorType.ElemType)
            : dataType;
}
