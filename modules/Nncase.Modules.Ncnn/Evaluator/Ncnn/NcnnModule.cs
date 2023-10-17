﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Ncnn module.
/// </summary>
internal class NcnnModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<NcnnSoftmaxEvaluator>(reuse: Reuse.Singleton);
    }
}
