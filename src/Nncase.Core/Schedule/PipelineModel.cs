// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;

namespace Nncase.Schedule;

/// <summary>
/// How a boundary iteration that is smaller than the nominal tile is executed.
/// </summary>
public enum PipelineTailPolicy
{
    /// <summary>
    /// Execute peeled boundary iterations serially with the same staged
    /// allocation and logical producer/consumer bodies.
    /// </summary>
    Serial,
}

/// <summary>
/// Backend contract for compiler-visible loop pipelines. Hardware capability
/// remains on <see cref="TargetMemoryTransferSpec"/>; this interface answers
/// whether the backend can represent a channel and names the exact template
/// used after selection.
/// </summary>
public interface ILoopPipelineBackend
{
    bool SupportsStageCount(int stageCount);

    IntExpr GetChannelLegality(LoopPipelineChannelModelContext context, int stageCount);

    PipelineTemplateDescriptor GetTemplate(int stageCount, TargetMachineModel machine);
}

/// <summary>
/// Opaque, versioned identity of a backend-owned loop-pipeline template.
/// </summary>
public sealed record PipelineTemplateId
{
    public PipelineTemplateId(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException("Pipeline template identity must not be empty.", nameof(value));
        }

        Value = value;
    }

    public string Value { get; }

    public override string ToString() => Value;
}

/// <summary>
/// Target-independent synchronization semantics of a loop pipeline.
/// </summary>
public sealed record PipelineSynchronizationProtocol
{
    public PipelineSynchronizationProtocol(
        bool asynchronousProduce,
        bool requiresProducerCommit,
        bool requiresConsumerWait,
        bool waitProvidesConsumerAcquire,
        bool requiresConsumerRelease)
    {
        if (asynchronousProduce && (!requiresProducerCommit || !requiresConsumerWait))
        {
            throw new ArgumentException(
                "An asynchronous pipeline must commit produced work and wait before consumption.");
        }

        if (waitProvidesConsumerAcquire && !requiresConsumerWait)
        {
            throw new ArgumentException(
                "A wait cannot provide acquire semantics when the protocol has no wait event.");
        }

        AsynchronousProduce = asynchronousProduce;
        RequiresProducerCommit = requiresProducerCommit;
        RequiresConsumerWait = requiresConsumerWait;
        WaitProvidesConsumerAcquire = waitProvidesConsumerAcquire;
        RequiresConsumerRelease = requiresConsumerRelease;
    }

    public bool AsynchronousProduce { get; }

    public bool RequiresProducerCommit { get; }

    public bool RequiresConsumerWait { get; }

    public bool WaitProvidesConsumerAcquire { get; }

    public bool RequiresConsumerRelease { get; }

    public TargetPipelineControlCostSpec GetControlCost(
        TargetMachineModel machine,
        TargetMemoryTransferSpec transfer)
    {
        ArgumentNullException.ThrowIfNull(machine);
        ArgumentNullException.ThrowIfNull(transfer);
        if (!AsynchronousProduce || transfer.Asynchronous is not { } asynchronous)
        {
            throw new InvalidOperationException(
                $"Pipeline protocol requires asynchronous transfer support on {transfer.Source}->{transfer.Destination}.");
        }

        var commit = RequiresProducerCommit ? asynchronous.CommitCycles : 0;
        var waitAcquire = RequiresConsumerWait ? asynchronous.WaitCycles : 0;
        if (RequiresConsumerWait && !WaitProvidesConsumerAcquire)
        {
            waitAcquire = checked(waitAcquire + machine.Synchronization.BlockCycles);
        }

        var release = RequiresConsumerRelease ? machine.Synchronization.BlockCycles : 0;
        return new(commit, waitAcquire, release);
    }
}

/// <summary>
/// Backend lowering selected for one stage count.
/// </summary>
public sealed record PipelineTemplateDescriptor(
    PipelineTemplateId Id,
    PipelineSynchronizationProtocol Synchronization);

/// <summary>
/// Symbolic facts for one potential parent-to-local transfer channel. This is
/// deliberately independent of TileGraph identities so targets can validate
/// representability without depending on AutoTiling internals.
/// </summary>
public sealed record LoopPipelineChannelModelContext(
    TargetMemorySpaceSpec SourceMemorySpace,
    TargetMemorySpaceSpec DestinationMemorySpace,
    DataType DataType,
    ImmutableArray<IntExpr> LocalShape,
    ImmutableArray<IntExpr> FullShape,
    TargetMachineModel Machine,
    Solver Solver);

/// <summary>
/// Backend used by targets with no asynchronous loop-pipeline lowering.
/// </summary>
public sealed class EmptyLoopPipelineBackend : ILoopPipelineBackend
{
    public bool SupportsStageCount(int stageCount) => false;

    public IntExpr GetChannelLegality(LoopPipelineChannelModelContext context, int stageCount)
        => context.Solver.MakeIntConst(0);

    public PipelineTemplateDescriptor GetTemplate(int stageCount, TargetMachineModel machine)
        => throw new NotSupportedException(
            $"Target {machine.Id} does not implement a {stageCount}-stage loop pipeline.");
}

/// <summary>
/// One concrete producer-to-consumer buffer channel selected at a loop entry.
/// The identity is assigned by AutoTiling from lexical placement provenance;
/// it is not recovered from operation metadata during lowering.
/// </summary>
public sealed record PipelineStageChannelPlan
{
    public PipelineStageChannelPlan(
        string channelId,
        TargetMemorySpaceId sourceMemorySpace,
        TargetMemorySpaceId destinationMemorySpace)
    {
        if (string.IsNullOrWhiteSpace(channelId))
        {
            throw new ArgumentException("Pipeline channel identity must not be empty.", nameof(channelId));
        }

        if (string.IsNullOrWhiteSpace(sourceMemorySpace.Value) ||
            string.IsNullOrWhiteSpace(destinationMemorySpace.Value) ||
            sourceMemorySpace == destinationMemorySpace)
        {
            throw new ArgumentException(
                $"Pipeline channel {channelId} must cross two named memory spaces.");
        }

        ChannelId = channelId;
        SourceMemorySpace = sourceMemorySpace;
        DestinationMemorySpace = destinationMemorySpace;
    }

    public string ChannelId { get; }

    public TargetMemorySpaceId SourceMemorySpace { get; }

    public TargetMemorySpaceId DestinationMemorySpace { get; }
}

/// <summary>
/// Concrete schedule selected for one lexical TIR loop. Stage count is a loop
/// property; no operation or microkernel owns this plan.
/// </summary>
public sealed class PipelineRegionPlan : IEquatable<PipelineRegionPlan>
{
    public PipelineRegionPlan(
        string scheduleId,
        PipelineTemplateId templateId,
        PipelineSynchronizationProtocol synchronization,
        int stageCount,
        int prefetchDistance,
        PipelineTailPolicy tailPolicy,
        IEnumerable<PipelineStageChannelPlan> channels)
    {
        if (string.IsNullOrWhiteSpace(scheduleId))
        {
            throw new ArgumentException("Pipeline schedule identity must not be empty.", nameof(scheduleId));
        }

        ArgumentNullException.ThrowIfNull(templateId);
        ArgumentNullException.ThrowIfNull(synchronization);
        if (stageCount <= 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stageCount), stageCount, "A materialized pipeline requires more than one stage.");
        }

        if (prefetchDistance <= 0 || prefetchDistance >= stageCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(prefetchDistance), prefetchDistance, "Prefetch distance must be in [1, stageCount).");
        }

        var channelArray = channels?.ToImmutableArray() ?? throw new ArgumentNullException(nameof(channels));
        if (channelArray.IsDefaultOrEmpty)
        {
            throw new ArgumentException("A materialized pipeline requires at least one transfer channel.", nameof(channels));
        }

        var duplicate = channelArray.GroupBy(channel => channel.ChannelId, StringComparer.Ordinal)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
        {
            throw new ArgumentException(
                $"Pipeline schedule {scheduleId} contains duplicate channel {duplicate.Key}.", nameof(channels));
        }

        if (!synchronization.AsynchronousProduce)
        {
            throw new ArgumentException(
                $"Multi-stage pipeline schedule {scheduleId} must use asynchronous production.", nameof(synchronization));
        }

        ScheduleId = scheduleId;
        TemplateId = templateId;
        Synchronization = synchronization;
        StageCount = stageCount;
        PrefetchDistance = prefetchDistance;
        TailPolicy = tailPolicy;
        Channels = channelArray;
    }

    public string ScheduleId { get; }

    public PipelineTemplateId TemplateId { get; }

    public PipelineSynchronizationProtocol Synchronization { get; }

    public int StageCount { get; }

    public int PrefetchDistance { get; }

    public PipelineTailPolicy TailPolicy { get; }

    public ImmutableArray<PipelineStageChannelPlan> Channels { get; }

    public bool Equals(PipelineRegionPlan? other)
        => other is not null
        && ScheduleId == other.ScheduleId
        && TemplateId == other.TemplateId
        && Synchronization == other.Synchronization
        && StageCount == other.StageCount
        && PrefetchDistance == other.PrefetchDistance
        && TailPolicy == other.TailPolicy
        && Channels.SequenceEqual(other.Channels);

    public override bool Equals(object? obj) => obj is PipelineRegionPlan other && Equals(other);

    public override int GetHashCode()
    {
        HashCode hash = default;
        hash.Add(ScheduleId, StringComparer.Ordinal);
        hash.Add(TemplateId);
        hash.Add(Synchronization);
        hash.Add(StageCount);
        hash.Add(PrefetchDistance);
        hash.Add(TailPolicy);
        foreach (var channel in Channels)
        {
            hash.Add(channel);
        }

        return hash.ToHashCode();
    }
}

/// <summary>
/// Physical layout of one staged allocation. The logical Buffer shape and
/// strides still describe one stage; this descriptor adds only the physical
/// leading stage dimension.
/// </summary>
public sealed class StagedBufferLayout : IEquatable<StagedBufferLayout>
{
    public StagedBufferLayout(int stageCount, long stagePhysicalBytes, long stageStrideBytes)
    {
        if (stageCount <= 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stageCount), stageCount, "A staged allocation requires more than one stage.");
        }

        if (stagePhysicalBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(stagePhysicalBytes));
        }

        if (stageStrideBytes < stagePhysicalBytes)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stageStrideBytes), stageStrideBytes, "Stage stride must cover one physical stage.");
        }

        StageCount = stageCount;
        StagePhysicalBytes = stagePhysicalBytes;
        StageStrideBytes = stageStrideBytes;
        PhysicalBytes = checked(stageCount * stageStrideBytes);
    }

    public int StageCount { get; }

    public long StagePhysicalBytes { get; }

    public long StageStrideBytes { get; }

    public long PhysicalBytes { get; }

    public bool Equals(StagedBufferLayout? other)
        => other is not null
        && StageCount == other.StageCount
        && StagePhysicalBytes == other.StagePhysicalBytes
        && StageStrideBytes == other.StageStrideBytes;

    public override bool Equals(object? obj) => obj is StagedBufferLayout other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(StageCount, StagePhysicalBytes, StageStrideBytes);

    public override string ToString()
        => $"stages={StageCount},stage_bytes={StagePhysicalBytes},stage_stride={StageStrideBytes},total={PhysicalBytes}";
}
