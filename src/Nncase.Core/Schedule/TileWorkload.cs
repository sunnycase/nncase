// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;

namespace Nncase.Schedule;

public interface ITileWorkloadProvider
{
    TileWorkload GetWorkload(Op op, TileWorkloadContext context);
}

/// <summary>
/// Semantic reduction state that must remain live across reduction tiles.
/// AutoTiling constrains its size, while the backend owns its physical
/// lowering and does not expose it as a TIR memory space.
/// </summary>
public interface IReductionStateTileWorkload
{
    IntExpr GetReductionStateBytes(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context, TargetMachineModel machine);
}

/// <summary>
/// Target-independent compute work performed by one tiled operator.
/// </summary>
public abstract record TileWorkload;

/// <summary>
/// Descriptor-only operation with no compute or memory traffic.
/// </summary>
public sealed record BufferAliasTileWorkload : TileWorkload
{
    public static BufferAliasTileWorkload Default { get; } = new();
}

public sealed record ElementwiseTileWorkload(
    Func<IntExpr[][], Solver, TileWorkloadContext, IntExpr> GetWork) : TileWorkload;

public sealed record ReductionTileWorkload(
    Func<IntExpr[][], Solver, TileWorkloadContext, IntExpr> GetWork,
    Func<IntExpr[][], Solver, TileWorkloadContext, IntExpr> GetStateBytes) : TileWorkload, IReductionStateTileWorkload
{
    public IntExpr GetReductionStateBytes(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context, TargetMachineModel machine)
        => GetStateBytes(bufferShapes, solver, context);
}

public sealed record MatrixTileWorkload(
    Func<IntExpr[][], Solver, TileWorkloadContext, MatrixTileWorkloadShape> GetShape,
    int AccumulatorElementSizeBytes) : TileWorkload, IReductionStateTileWorkload
{
    public IntExpr GetReductionStateBytes(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context, TargetMachineModel machine)
    {
        if (AccumulatorElementSizeBytes <= 0)
        {
            throw new InvalidOperationException($"Accumulator element size must be positive, got {AccumulatorElementSizeBytes}.");
        }

        var shape = GetShape(bufferShapes, solver, context);
        var fullBufferShapes = context.BufferShapes
            .Select(bufferShape => bufferShape
                .Select(extent => (IntExpr)solver.MakeIntConst(extent))
                .ToArray())
            .ToArray();
        var fullShape = GetShape(fullBufferShapes, solver, context);
        var fullM = fullShape.M.Var();
        if (fullM.Min() != fullM.Max())
        {
            throw new InvalidOperationException($"Full matrix M extent for {context.Op.GetType().Name} must be constant.");
        }

        var useGemv = fullM.Max() == 1;
        var accumulatorM = useGemv
            ? shape.M
            : AlignUp(shape.M, machine.Execution.BackendPrivateMatrixAccumulatorMinM, solver);
        var minimumAccumulatorN = useGemv
            ? machine.Execution.BackendPrivateGemvAccumulatorMinN
            : machine.Execution.BackendPrivateMatrixAccumulatorMinN;
        var accumulatorN = AlignUp(shape.N, minimumAccumulatorN, solver);
        return accumulatorM * accumulatorN * shape.Multiplicity * AccumulatorElementSizeBytes;
    }

    private static IntExpr AlignUp(IntExpr value, int alignment, Solver solver)
        => solver.MakeDiv(value + (alignment - 1), alignment) * alignment;
}

public sealed record MatrixTileWorkloadShape(
    IntExpr M,
    IntExpr N,
    IntExpr K,
    IntExpr Multiplicity)
{
    public IntExpr GetWork() => M * N * K * Multiplicity;
}

public sealed record TileWorkloadContext(
    Op Op,
    ImmutableArray<ImmutableArray<long>> BufferShapes,
    ImmutableArray<DataType> BufferDataTypes);

public static class TileWorkloadUtility
{
    public static int GetVectorLaneCount(DataType dataType)
        => dataType switch
        {
            VectorType vectorType => vectorType.Lanes.Aggregate(1, static (product, lane) => checked(product * lane)) * GetVectorLaneCount(vectorType.ElemType),
            MaskVectorType maskVectorType => maskVectorType.Lanes,
            _ => 1,
        };
}
