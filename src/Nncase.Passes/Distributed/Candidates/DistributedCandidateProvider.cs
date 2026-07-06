// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.Targets;

namespace Nncase.Passes.Distributed;

public interface IDistributedCandidateProvider
{
    bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        Op target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples);
}

public interface IDistributedCandidateProvider<T> : IDistributedCandidateProvider
    where T : Op
{
    bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        T target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples);
}

public interface IDistributedCandidateProviderResolver
{
    bool TryGetProvider(Op op, out IDistributedCandidateProvider provider);
}

public sealed class DistributedCandidateContext
{
    public DistributedCandidateContext(
        CompileOptions compileOptions,
        INTTTargetOptions targetOptions,
        string moduleKind,
        Call sourceCall,
        IReadOnlyList<IReadOnlyList<IRType>> availableInputTypes)
    {
        CompileOptions = compileOptions;
        TargetOptions = targetOptions;
        ModuleKind = moduleKind;
        SourceCall = sourceCall;
        AvailableInputTypes = availableInputTypes;
    }

    public CompileOptions CompileOptions { get; }

    public INTTTargetOptions TargetOptions { get; }

    public string ModuleKind { get; }

    public Call SourceCall { get; }

    public IReadOnlyList<IReadOnlyList<IRType>> AvailableInputTypes { get; }
}

public abstract class DistributedCandidateProvider<T> : IDistributedCandidateProvider<T>
    where T : Op
{
    public abstract bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        T target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples);

    bool IDistributedCandidateProvider.TryGetInputTypeTuples(
        DistributedCandidateContext context,
        Op target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples)
    {
        if (target is T typedTarget)
        {
            return TryGetInputTypeTuples(context, typedTarget, returnType, out tuples);
        }

        tuples = Array.Empty<DistributedCandidateTuple>();
        return false;
    }
}

public sealed record DistributedCandidateTuple(IReadOnlyList<IRType> InputTypes, string? Reason = null);

internal sealed class DistributedCandidateProviderResolver : IDistributedCandidateProviderResolver
{
    private readonly IServiceProvider _serviceProvider;
    private readonly Dictionary<Type, IDistributedCandidateProvider?> _memo = new();
    private readonly object _memoLock = new();

    public DistributedCandidateProviderResolver(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public bool TryGetProvider(Op op, out IDistributedCandidateProvider provider)
    {
        var opType = op.GetType();
        lock (_memoLock)
        {
            if (!_memo.TryGetValue(opType, out var cached))
            {
                var providerType = typeof(IDistributedCandidateProvider<>).MakeGenericType(opType);
                cached = _serviceProvider.GetService(providerType) as IDistributedCandidateProvider;
                _memo.Add(opType, cached);
            }

            provider = cached!;
            return cached is not null;
        }
    }
}
