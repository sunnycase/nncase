// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.Schedule;

namespace Nncase.Evaluator;

internal sealed class TileWorkloadProvider : ITileWorkloadProvider
{
    private readonly IServiceProvider _serviceProvider;

    public TileWorkloadProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public TileWorkload GetWorkload(Op op, TileWorkloadContext context)
    {
        var evaluatorType = typeof(ITileWorkloadEvaluator<>).MakeGenericType(op.GetType());
        var evaluator = (ITileWorkloadEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
        return evaluator.Visit(op, context);
    }
}
