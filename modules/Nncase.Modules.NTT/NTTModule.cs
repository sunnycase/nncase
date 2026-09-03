// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;
using Nncase.Passes.Distributed;
using Nncase.Targets;

namespace Nncase;

/// <summary>
/// CPU module.
/// </summary>
internal class NTTModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<ITarget, CPUTarget>(reuse: Reuse.Singleton);
        registrator.Register<ITarget, CUDATarget>(reuse: Reuse.Singleton);
        registrator.Register<ITarget, PyNTTTarget>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PagedAttentionPartialCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PagedAttentionCombineCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedMatMulCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedScaledMatMulCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedBlockScaledMatMulCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedBlockScaledMatMulNormStatsCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedMatMulGluCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedMatMulGluCombineCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedNVFP4MatMulCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedNVFP4MatMulNormStatsCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedNVFP4MatMulGluCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedMatMulNormStatsCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedMatMulNormStatsCombineCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SamplingPartialCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SamplingCombineCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedQKVParallelLinearCandidateProvider>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedQKVParallelLinearCombineCandidateProvider>(reuse: Reuse.Singleton);
    }
}
