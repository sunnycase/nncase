// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Schedule;

namespace Nncase.Evaluator;

public interface ITileWorkloadEvaluator
{
    TileWorkload Visit(Op op, TileWorkloadContext context);
}

public interface ITileWorkloadEvaluator<T> : ITileWorkloadEvaluator
    where T : Op
{
    TileWorkload Visit(T op, TileWorkloadContext context);

    TileWorkload ITileWorkloadEvaluator.Visit(Op op, TileWorkloadContext context)
    {
        return Visit((T)op, context);
    }
}
