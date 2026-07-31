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
/// Declares the independently executable weight-transfer phase of a selected
/// microkernel. Argument indexes refer to semantic kernel operands; workspace
/// indexes refer to <see cref="TIRMicroKernelSelection.SharedWorkspaces"/>.
/// </summary>
public sealed record TIRWeightPipelineContract
{
    public TIRWeightPipelineContract(
        IEnumerable<int> weightArgumentIndices,
        IEnumerable<int> sharedWorkspaceIndices)
    {
        WeightArgumentIndices = ValidateIndices(
            weightArgumentIndices,
            nameof(weightArgumentIndices));
        SharedWorkspaceIndices = ValidateIndices(
            sharedWorkspaceIndices,
            nameof(sharedWorkspaceIndices));
    }

    public ImmutableArray<int> WeightArgumentIndices { get; }

    public ImmutableArray<int> SharedWorkspaceIndices { get; }

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
/// Concrete target microkernel selected during TIR Selection.
/// </summary>
public sealed record TIRMicroKernelSelection(
    string Family,
    string Variant,
    ImmutableDictionary<string, long> Parameters,
    ImmutableArray<TIRSharedWorkspaceDescriptor> SharedWorkspaces,
    TIRWeightPipelineContract? WeightPipeline);

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
