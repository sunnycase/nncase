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
/// The workload only describes logical state. Target microkernel models own
/// alignment, physical placement, resource use, and lowering.
/// </summary>
public interface IReductionStateTileWorkload
{
    IReadOnlyList<TileReductionState> GetReductionStates(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context);
}

public sealed record TileReductionState(string Name, IntExpr ElementCount, int ElementSizeBytes)
{
    public IntExpr GetLogicalBytes() => ElementCount * ElementSizeBytes;
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
    public IReadOnlyList<TileReductionState> GetReductionStates(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
        => [new("reduction", GetStateBytes(bufferShapes, solver, context), 1)];
}

public sealed record MatrixTileWorkload(
    Func<IntExpr[][], Solver, TileWorkloadContext, MatrixTileWorkloadShape> GetShape,
    int AccumulatorElementSizeBytes) : TileWorkload, IReductionStateTileWorkload
{
    public IReadOnlyList<TileReductionState> GetReductionStates(IntExpr[][] bufferShapes, Solver solver, TileWorkloadContext context)
    {
        if (AccumulatorElementSizeBytes <= 0)
        {
            throw new InvalidOperationException($"Accumulator element size must be positive, got {AccumulatorElementSizeBytes}.");
        }

        var shape = GetShape(bufferShapes, solver, context);
        return [new("matrix_accumulator", shape.M * shape.N * shape.Multiplicity, AccumulatorElementSizeBytes)];
    }
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
