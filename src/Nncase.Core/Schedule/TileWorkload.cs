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
/// Target-independent compute work performed by one tiled operator.
/// </summary>
public abstract record TileWorkload;

public sealed record ElementwiseTileWorkload(
    Func<IntExpr[][], Solver, TileWorkloadContext, IntExpr> GetWork) : TileWorkload;

public sealed record MatrixTileWorkload(
    Func<IntExpr[][], Solver, TileWorkloadContext, MatrixTileWorkloadShape> GetShape) : TileWorkload;

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
