// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.Heterogeneous;

/// <summary>
/// Heterogeneous pipeline evaluator module.
/// </summary>
internal sealed class HeterogeneousModule : IApplicationPart
{
    /// <inheritdoc/>
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<CreatePipelineChannelEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ProduceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ConsumeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PipelineTokenEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PipelineYieldEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PipelineLaunchEvaluator>(reuse: Reuse.Singleton);
    }
}
