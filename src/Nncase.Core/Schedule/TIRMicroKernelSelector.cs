// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;

namespace Nncase.Schedule;

/// <summary>
/// Selects one concrete target microkernel while semantic operators are lowered
/// to TIR. Unlike <see cref="IBlockMicroKernelModelProvider"/>, this interface
/// does not participate in AutoTiling and only consumes concrete TIR operands.
/// </summary>
public interface ITIRMicroKernelSelector
{
    /// <summary>
    /// Selects a microkernel for <paramref name="context"/>, or returns
    /// <see langword="null"/> when the operation needs no target microkernel.
    /// </summary>
    TIRMicroKernelSelection? Select(TIRMicroKernelSelectionContext context);
}

/// <summary>
/// Concrete TIR operation and operands available to a target selector.
/// </summary>
public sealed record TIRMicroKernelSelectionContext(
    Op Op,
    IReadOnlyList<BaseExpr> Arguments,
    TargetMachineModel Machine);

/// <summary>
/// One typed, target-private shared-memory allocation required by a selected
/// microkernel. TIR Selection materializes this descriptor as a
/// <see cref="TIR.Buffer"/>; Bufferize owns lifetime, reuse, and byte offsets.
/// </summary>
public sealed record TIRSharedWorkspaceDescriptor(
    string Name,
    TensorType Type,
    int AlignmentBytes);

/// <summary>
/// One independently synchronized transfer channel owned by a selected
/// microkernel. Source indexes refer to semantic kernel operands; workspace
/// indexes refer to <see cref="TIRMicroKernelSelection.SharedWorkspaces"/>.
/// </summary>
public sealed record TIRTransferPipelineChannel
{
    public TIRTransferPipelineChannel(
        string name,
        IEnumerable<int> sourceArgumentIndices,
        IEnumerable<int> sharedWorkspaceIndices,
        int sourceAlignmentBytes = 1)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException(
                "Transfer-pipeline channel name must not be empty.",
                nameof(name));
        }

        Name = name;
        SourceArgumentIndices = ValidateIndices(
            sourceArgumentIndices,
            nameof(sourceArgumentIndices));
        SharedWorkspaceIndices = ValidateIndices(
            sharedWorkspaceIndices,
            nameof(sharedWorkspaceIndices));
        if (sourceAlignmentBytes <= 0 ||
            (sourceAlignmentBytes & (sourceAlignmentBytes - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sourceAlignmentBytes),
                sourceAlignmentBytes,
                "Transfer source alignment must be a positive power of two.");
        }

        SourceAlignmentBytes = sourceAlignmentBytes;
    }

    public string Name { get; }

    public ImmutableArray<int> SourceArgumentIndices { get; }

    public ImmutableArray<int> SharedWorkspaceIndices { get; }

    /// <summary>
    /// Gets the minimum base-address alignment required for every source
    /// operand in this channel.
    /// </summary>
    public int SourceAlignmentBytes { get; }

    private static ImmutableArray<int> ValidateIndices(
        IEnumerable<int> indices,
        string parameterName)
    {
        ArgumentNullException.ThrowIfNull(indices);
        var values = indices.ToImmutableArray();
        if (values.IsDefaultOrEmpty ||
            values.Any(index => index < 0) ||
            values.Distinct().Count() != values.Length)
        {
            throw new ArgumentException(
                "Pipeline operand indexes must be non-empty, non-negative, and unique.",
                parameterName);
        }

        return values;
    }
}

/// <summary>
/// Declares independently executable global-to-Shared transfer channels of a
/// selected microkernel. A Shared workspace is owned either by exactly one
/// transfer channel or by the consumer role, while multiple channels may read
/// the same semantic source operand.
/// </summary>
public sealed record TIRTransferPipelineContract
{
    public TIRTransferPipelineContract(
        IEnumerable<TIRTransferPipelineChannel> channels,
        IEnumerable<int>? consumerSharedWorkspaceIndices = null)
    {
        ArgumentNullException.ThrowIfNull(channels);
        var values = channels.ToImmutableArray();
        if (values.IsDefaultOrEmpty)
        {
            throw new ArgumentException(
                "A transfer pipeline must contain at least one channel.",
                nameof(channels));
        }

        var duplicateName = values
            .GroupBy(channel => channel.Name, StringComparer.Ordinal)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateName is not null)
        {
            throw new ArgumentException(
                $"Transfer pipeline contains duplicate channel {duplicateName.Key}.",
                nameof(channels));
        }

        var duplicateWorkspace = values
            .SelectMany(channel => channel.SharedWorkspaceIndices)
            .GroupBy(index => index)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateWorkspace is not null)
        {
            throw new ArgumentException(
                $"Shared workspace {duplicateWorkspace.Key} is owned by multiple transfer channels.",
                nameof(channels));
        }

        var consumerWorkspaces = consumerSharedWorkspaceIndices?.ToImmutableArray() ??
            ImmutableArray<int>.Empty;
        if (consumerWorkspaces.Any(index => index < 0) ||
            consumerWorkspaces.Distinct().Count() != consumerWorkspaces.Length)
        {
            throw new ArgumentException(
                "Consumer Shared workspace indexes must be non-negative and unique.",
                nameof(consumerSharedWorkspaceIndices));
        }

        var channelWorkspaces = values
            .SelectMany(channel => channel.SharedWorkspaceIndices)
            .ToImmutableArray();
        var conflictingWorkspace = channelWorkspaces
            .Intersect(consumerWorkspaces)
            .Cast<int?>()
            .FirstOrDefault();
        if (conflictingWorkspace is not null)
        {
            throw new ArgumentException(
                $"Shared workspace {conflictingWorkspace.Value} is owned by both a transfer channel and the consumer.",
                nameof(consumerSharedWorkspaceIndices));
        }

        Channels = values;
        SourceArgumentIndices = values
            .SelectMany(channel => channel.SourceArgumentIndices)
            .Distinct()
            .ToImmutableArray();
        SharedWorkspaceIndices = channelWorkspaces;
        ConsumerSharedWorkspaceIndices = consumerWorkspaces;
    }

    public ImmutableArray<TIRTransferPipelineChannel> Channels { get; }

    /// <summary>
    /// Gets the stable union of transfer source operands for memory-effect
    /// analysis.
    /// </summary>
    public ImmutableArray<int> SourceArgumentIndices { get; }

    /// <summary>
    /// Gets all channel-owned Shared workspaces for arena lifetime analysis.
    /// </summary>
    public ImmutableArray<int> SharedWorkspaceIndices { get; }

    /// <summary>
    /// Gets Shared workspaces owned directly by the consumer role rather than
    /// by a transfer channel.
    /// </summary>
    public ImmutableArray<int> ConsumerSharedWorkspaceIndices { get; }
}

/// <summary>
/// Concrete target microkernel selected during TIR Selection.
/// </summary>
public sealed record TIRMicroKernelSelection(
    string Family,
    string Variant,
    ImmutableDictionary<string, long> Parameters,
    ImmutableArray<TIRSharedWorkspaceDescriptor> SharedWorkspaces,
    TIRTransferPipelineContract? TransferPipeline);

/// <summary>
/// Canonical expression representation for a microkernel's shared workspaces.
/// </summary>
public static class TIRSharedWorkspace
{
    /// <summary>
    /// Packs zero, one, or multiple workspace values as None, the value itself,
    /// or a tuple, respectively.
    /// </summary>
    public static BaseExpr Pack(IEnumerable<BaseExpr> workspaces)
    {
        ArgumentNullException.ThrowIfNull(workspaces);
        var items = workspaces.ToImmutableArray();
        ValidateItems(items);
        return items.Length switch
        {
            0 => None.Default,
            1 => items[0],
            _ => new IR.Tuple(items.ToArray()),
        };
    }

    /// <summary>
    /// Unpacks the canonical None/value/tuple workspace representation.
    /// </summary>
    public static ImmutableArray<BaseExpr> Unpack(BaseExpr workspace)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        var items = workspace switch
        {
            None => ImmutableArray<BaseExpr>.Empty,
            IR.Tuple { Count: < 2 } tuple => throw new InvalidOperationException(
                $"Shared workspace tuples must contain at least two values, got {tuple.Count}."),
            IR.Tuple tuple => tuple.Fields.ToArray().ToImmutableArray(),
            _ => ImmutableArray.Create(workspace),
        };
        ValidateItems(items);
        return items;
    }

    private static void ValidateItems(ImmutableArray<BaseExpr> items)
    {
        if (items.Any(item => item is None or IR.Tuple))
        {
            throw new InvalidOperationException(
                "Shared workspace values must be flat, non-None expressions.");
        }
    }
}

/// <summary>
/// Default selector for targets whose TIR kernels require no compiler-managed
/// shared-memory workspace.
/// </summary>
public sealed class EmptyTIRMicroKernelSelector : ITIRMicroKernelSelector
{
    public TIRMicroKernelSelection? Select(TIRMicroKernelSelectionContext context) => null;
}
