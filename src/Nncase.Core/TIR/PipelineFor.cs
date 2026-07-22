// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// Exact TIR binding of one loop-selected staging channel.
/// </summary>
public sealed record PipelineBufferBindingDescriptor
{
    public PipelineBufferBindingDescriptor(
        string channelId,
        TargetMemorySpaceId sourceMemorySpace,
        TargetMemorySpaceId destinationMemorySpace)
    {
        _ = new PipelineStageChannelPlan(channelId, sourceMemorySpace, destinationMemorySpace);
        ChannelId = channelId;
        SourceMemorySpace = sourceMemorySpace;
        DestinationMemorySpace = destinationMemorySpace;
    }

    public string ChannelId { get; }

    public TargetMemorySpaceId SourceMemorySpace { get; }

    public TargetMemorySpaceId DestinationMemorySpace { get; }
}

/// <summary>
/// A producer/consumer loop constructed directly from the selected lexical
/// schedule. Producer and consumer bodies are separate operands from birth;
/// no later pass classifies statements or searches for allocation owners.
/// </summary>
public sealed class PipelineFor : Expr
{
    private readonly int _bindingCount;

    public PipelineFor(
        DimVar loopVar,
        Range domain,
        LoopMode mode,
        LoopPartition partition,
        Sequential produceBody,
        Sequential consumeBody,
        PipelineRegionPlan plan,
        PipelineRegionId regionId,
        IEnumerable<PipelineBufferBindingDescriptor> bindingDescriptors,
        IEnumerable<IVar> stagedAccesses,
        IEnumerable<BaseExpr> stagedAllocations,
        IEnumerable<Buffer> stagedBuffers)
        : this(
            loopVar,
            domain,
            mode,
            partition,
            produceBody,
            consumeBody,
            plan,
            regionId,
            bindingDescriptors?.ToImmutableArray() ?? throw new ArgumentNullException(nameof(bindingDescriptors)),
            stagedAccesses?.ToArray() ?? throw new ArgumentNullException(nameof(stagedAccesses)),
            stagedAllocations?.ToArray() ?? throw new ArgumentNullException(nameof(stagedAllocations)),
            stagedBuffers?.ToArray() ?? throw new ArgumentNullException(nameof(stagedBuffers)))
    {
    }

    private PipelineFor(
        DimVar loopVar,
        Range domain,
        LoopMode mode,
        LoopPartition partition,
        Sequential produceBody,
        Sequential consumeBody,
        PipelineRegionPlan plan,
        PipelineRegionId regionId,
        ImmutableArray<PipelineBufferBindingDescriptor> bindingDescriptors,
        IVar[] stagedAccesses,
        BaseExpr[] stagedAllocations,
        Buffer[] stagedBuffers)
        : base(BuildOperands(
            loopVar,
            domain,
            produceBody,
            consumeBody,
            stagedAccesses,
            stagedAllocations,
            stagedBuffers))
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(regionId);
        if (bindingDescriptors.IsDefaultOrEmpty ||
            bindingDescriptors.Length != stagedAccesses.Length ||
            bindingDescriptors.Length != stagedAllocations.Length ||
            bindingDescriptors.Length != stagedBuffers.Length)
        {
            throw new ArgumentException(
                "A pipeline requires equally sized non-empty binding, access, allocation, and buffer arrays.",
                nameof(bindingDescriptors));
        }

        var duplicate = bindingDescriptors.GroupBy(binding => binding.ChannelId, StringComparer.Ordinal)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
        {
            throw new ArgumentException(
                $"Pipeline execution contains duplicate channel {duplicate.Key}.", nameof(bindingDescriptors));
        }

        var channels = plan.Channels.ToDictionary(channel => channel.ChannelId, StringComparer.Ordinal);
        if (channels.Count != bindingDescriptors.Length)
        {
            throw new ArgumentException(
                $"Pipeline plan {plan.ScheduleId} declares {channels.Count} channels but binds {bindingDescriptors.Length}.",
                nameof(bindingDescriptors));
        }

        for (var index = 0; index < bindingDescriptors.Length; index++)
        {
            var descriptor = bindingDescriptors[index];
            if (!channels.TryGetValue(descriptor.ChannelId, out var channel) ||
                channel.SourceMemorySpace != descriptor.SourceMemorySpace ||
                channel.DestinationMemorySpace != descriptor.DestinationMemorySpace)
            {
                throw new ArgumentException(
                    $"Pipeline binding {descriptor.ChannelId} does not match plan {plan.ScheduleId}.",
                    nameof(bindingDescriptors));
            }

            if (stagedBuffers[index].StagedLayout is not { } layout ||
                layout.StageCount != plan.StageCount)
            {
                throw new ArgumentException(
                    $"Pipeline channel {descriptor.ChannelId} requires a {plan.StageCount}-stage allocation.",
                    nameof(stagedBuffers));
            }
        }

        Mode = mode;
        Partition = partition;
        Plan = plan;
        RegionId = regionId;
        BindingDescriptors = bindingDescriptors;
        _bindingCount = bindingDescriptors.Length;
    }

    public DimVar LoopVar => (DimVar)Operands[0];

    public Range Domain => (Range)Operands[1];

    public Sequential ProduceBody => (Sequential)Operands[2];

    public Sequential ConsumeBody => (Sequential)Operands[3];

    public ReadOnlySpan<IVar> StagedAccesses
        => SpanUtility.UnsafeCast<BaseExpr, IVar>(Operands.Slice(4, _bindingCount));

    public ReadOnlySpan<BaseExpr> StagedAllocations
        => Operands.Slice(4 + _bindingCount, _bindingCount);

    public ReadOnlySpan<Buffer> StagedBuffers
        => SpanUtility.UnsafeCast<BaseExpr, Buffer>(Operands.Slice(4 + (2 * _bindingCount), _bindingCount));

    public LoopMode Mode { get; }

    public LoopPartition Partition { get; }

    public PipelineRegionPlan Plan { get; }

    public PipelineRegionId RegionId { get; }

    public ImmutableArray<PipelineBufferBindingDescriptor> BindingDescriptors { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(
        ExprFunctor<TExprResult, TTypeResult, TContext> functor,
        TContext context)
        => functor.VisitPipelineFor(this, context);

    public PipelineFor With(
        DimVar? loopVar = null,
        Range? domain = null,
        LoopMode? loopMode = null,
        LoopPartition? partition = null,
        Sequential? produceBody = null,
        Sequential? consumeBody = null,
        IVar[]? stagedAccesses = null,
        BaseExpr[]? stagedAllocations = null,
        Buffer[]? stagedBuffers = null,
        PipelineRegionId? regionId = null)
        => new(
            loopVar ?? LoopVar,
            domain ?? Domain,
            loopMode ?? Mode,
            partition ?? Partition,
            produceBody ?? ProduceBody,
            consumeBody ?? ConsumeBody,
            Plan,
            regionId ?? RegionId,
            BindingDescriptors,
            stagedAccesses ?? StagedAccesses.ToArray(),
            stagedAllocations ?? StagedAllocations.ToArray(),
            stagedBuffers ?? StagedBuffers.ToArray());

    public override bool Equals(object? obj)
        => ReferenceEquals(this, obj) ||
            (obj is PipelineFor other &&
             Mode == other.Mode &&
             Partition == other.Partition &&
             Plan == other.Plan &&
             RegionId == other.RegionId &&
             BindingDescriptors.SequenceEqual(other.BindingDescriptors) &&
             base.Equals(other));

    protected override int GetHashCodeCore()
    {
        HashCode hash = default;
        hash.Add(Mode);
        hash.Add(Partition);
        hash.Add(Plan);
        hash.Add(RegionId);
        foreach (var descriptor in BindingDescriptors)
        {
            hash.Add(descriptor);
        }

        hash.Add(base.GetHashCodeCore());
        return hash.ToHashCode();
    }

    private static BaseExpr[] BuildOperands(
        DimVar loopVar,
        Range domain,
        Sequential produceBody,
        Sequential consumeBody,
        IVar[] stagedAccesses,
        BaseExpr[] stagedAllocations,
        Buffer[] stagedBuffers)
        => [
            loopVar,
            domain,
            produceBody,
            consumeBody,
            .. stagedAccesses.Cast<BaseExpr>(),
            .. stagedAllocations,
            .. stagedBuffers.Cast<BaseExpr>(),
        ];
}
