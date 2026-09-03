// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Heterogeneous;

/// <summary>
/// Runtime-owned channel shared by heterogeneous worker programs.
/// </summary>
public interface IPipelineChannel
{
}

/// <summary>
/// Compiler/runtime ABI constants for heterogeneous pipeline channels.
/// </summary>
public static class PipelineChannelContract
{
    /// <summary>
    /// Gets the cache-line-isolated synchronization header size in bytes.
    /// </summary>
    public const int HeaderBytes = 64;
}

/// <summary>
/// Opaque reference type for a pipeline channel. The channel contract lives on
/// its create/endpoint operations so one runtime type can serve every payload.
/// </summary>
public sealed record PipelineChannelType : ValueType
{
    public override Type CLRType => typeof(IPipelineChannel);

    public override int SizeInBytes => IntPtr.Size;

    public override Guid Uuid { get; } = new("6f8f4f59-1d42-42e7-98f0-a732dd8fb894");

    public override string ToString() => "PipelineChannel";
}

/// <summary>
/// Creates one pipeline-owned channel resource.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class CreatePipelineChannel : Op
{
    public string ChannelId { get; }

    public string ProducerModuleKind { get; }

    public string ConsumerModuleKind { get; }

    public IRType PayloadType { get; }

    public int Capacity { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty()
        => $"{ChannelId}, {ProducerModuleKind}->{ConsumerModuleKind}, {PayloadType}, Capacity: {Capacity}";
}

/// <summary>
/// Publishes a payload to one heterogeneous pipeline channel phase.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Produce : Op
{
    public static readonly ParameterInfo Channel = new(
        typeof(Produce),
        0,
        "channel",
        IsTensor(),
        ParameterKind.Input,
        MemoryEffect.SystemReadWrite);

    public static readonly ParameterInfo Value = new(
        typeof(Produce),
        1,
        "value",
        IsIRType(),
        ParameterKind.Input,
        MemoryEffect.SystemRead);

    public static readonly ParameterInfo Dependency = new(
        typeof(Produce),
        2,
        "dependency",
        IsNoneType(),
        ParameterKind.Input);

    public string ChannelId { get; }

    public int Phase { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"{ChannelId}, Phase: {Phase}";
}

/// <summary>
/// Acquires a payload from one heterogeneous pipeline channel phase.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Consume : Op
{
    public static readonly ParameterInfo Channel = new(
        typeof(Consume),
        0,
        "channel",
        IsTensor(),
        ParameterKind.Input,
        MemoryEffect.SystemReadWrite);

    public static readonly ParameterInfo Dependency = new(
        typeof(Consume),
        1,
        "dependency",
        IsNoneType(),
        ParameterKind.Input);

    public string ChannelId { get; }

    public int Phase { get; }

    public IRType PayloadType { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"{ChannelId}, Phase: {Phase}, {PayloadType}";
}

/// <summary>
/// Converts completion of a value-producing operation into an ordering token.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PipelineToken : Op
{
    public static readonly ParameterInfo Value = new(
        typeof(PipelineToken),
        0,
        "value",
        IsIRType(),
        ParameterKind.Input);

    public override bool CanFoldConstCall => false;
}

/// <summary>
/// Binds a worker result to its final channel ordering token.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PipelineYield : Op
{
    public static readonly ParameterInfo Value = new(
        typeof(PipelineYield),
        0,
        "value",
        IsIRType(),
        ParameterKind.Input);

    public static readonly ParameterInfo Dependency = new(
        typeof(PipelineYield),
        1,
        "dependency",
        IsNoneType(),
        ParameterKind.Input);

    public override bool CanFoldConstCall => false;
}

/// <summary>
/// Concurrently launches all worker calls in one heterogeneous pipeline.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PipelineLaunch : Op
{
    public static readonly ParameterInfo Workers = new(
        typeof(PipelineLaunch),
        0,
        "workers",
        IsTuple() & !IsUnit(),
        ParameterKind.Input);

    public int ResultWorkerIndex { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"ResultWorkerIndex: {ResultWorkerIndex}";
}
