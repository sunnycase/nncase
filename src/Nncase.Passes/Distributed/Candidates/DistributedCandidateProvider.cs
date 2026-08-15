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
    bool AllowsPartialInputs { get; }

    bool IsExhaustive { get; }

    IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        Op target,
        IReadOnlyList<IRType> defaultReturnTypes);

    bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        Op target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples);

    Op CreateCandidateTarget(
        DistributedCandidateContext context,
        Op target,
        IRType returnType);
}

public interface IDistributedCandidateProvider<T> : IDistributedCandidateProvider
    where T : Op
{
    IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        T target,
        IReadOnlyList<IRType> defaultReturnTypes);

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
    public virtual bool AllowsPartialInputs => false;

    public virtual bool IsExhaustive => false;

    public virtual IReadOnlyList<IRType> GetReturnCandidateTypes(
        DistributedCandidateContext context,
        T target,
        IReadOnlyList<IRType> defaultReturnTypes)
        => defaultReturnTypes;

    public abstract bool TryGetInputTypeTuples(
        DistributedCandidateContext context,
        T target,
        IRType returnType,
        out IReadOnlyList<DistributedCandidateTuple> tuples);

    public virtual T CreateCandidateTarget(
        DistributedCandidateContext context,
        T target,
        IRType returnType)
        => target;

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

    IReadOnlyList<IRType> IDistributedCandidateProvider.GetReturnCandidateTypes(
        DistributedCandidateContext context,
        Op target,
        IReadOnlyList<IRType> defaultReturnTypes)
    {
        if (target is T typedTarget)
        {
            return GetReturnCandidateTypes(context, typedTarget, defaultReturnTypes);
        }

        return defaultReturnTypes;
    }

    Op IDistributedCandidateProvider.CreateCandidateTarget(
        DistributedCandidateContext context,
        Op target,
        IRType returnType)
    {
        if (target is not T typedTarget)
        {
            throw new ArgumentException(
                $"Candidate provider {GetType().Name} cannot handle target {target.GetType().Name}.",
                nameof(target));
        }

        return CreateCandidateTarget(context, typedTarget, returnType);
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
