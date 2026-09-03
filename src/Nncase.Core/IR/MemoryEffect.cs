// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

[Flags]
public enum MemoryAccessMode
{
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = Read | Write,
}

public enum MemoryAccessScope
{
    Inferred,
    Block,
    Chip,
    System,
}

/// <summary>
/// Describes when an operand's memory effect becomes externally observable.
/// </summary>
public enum MemoryEffectKind
{
    /// <summary>
    /// The operation directly accesses the operand when it executes.
    /// </summary>
    Direct,

    /// <summary>
    /// The operand is the logical state/output of a reduction region. Inside a
    /// <see cref="TIR.For"/> with <see cref="TIR.LoopMode.Reduction"/>, the
    /// backend carries this state privately. The read component describes
    /// accumulator feedback rather than physical-buffer traffic; the backend
    /// commits the write component once when the region completes.
    /// </summary>
    ReductionAccumulator,
}

public enum MemoryAccessDomainKind
{
    AllBlocks,
    FixedBlock,
}

/// <summary>
/// Describes which distributed-owner components each participating block
/// accesses through an operand.
/// </summary>
public enum MemoryOwnerAccess
{
    /// <summary>
    /// Each block accesses only the component addressed by its own placement
    /// coordinate.
    /// </summary>
    Local,

    /// <summary>
    /// Each block accesses the owner components in the input distributed
    /// type's partial-reduction group.
    /// </summary>
    PartialGroup,
}

public enum MemoryAccessPartitionKind
{
    WholeResource,
    Argument,
}

/// <summary>
/// Refines the default operand effects declared by <see cref="ParameterInfo"/>
/// when an operation's attributes or arguments change its memory access.
/// </summary>
public interface IOpMemoryEffectProvider
{
    MemoryEffect GetMemoryEffect(ParameterInfo parameter, IReadOnlyList<BaseExpr> arguments);
}

/// <summary>
/// Describes a logical partition of one memory operand. Operations that access
/// a disjoint subresource selected by another argument can expose that relation
/// to interprocedural synchronization planning.
/// </summary>
public readonly record struct MemoryAccessPartition
{
    private MemoryAccessPartition(MemoryAccessPartitionKind kind, int argumentIndex)
    {
        if (kind == MemoryAccessPartitionKind.Argument && argumentIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(argumentIndex));
        }

        Kind = kind;
        ArgumentIndex = argumentIndex;
    }

    public static MemoryAccessPartition WholeResource => default;

    public MemoryAccessPartitionKind Kind { get; }

    public int ArgumentIndex { get; }

    public static MemoryAccessPartition ByArgument(int argumentIndex)
        => new(MemoryAccessPartitionKind.Argument, argumentIndex);
}

/// <summary>
/// Describes which physical blocks participate in one operand access. This is
/// independent from <see cref="MemoryAccessScope"/>, which describes the
/// visibility and synchronization scope of the accessed storage.
/// </summary>
public readonly record struct MemoryAccessDomain
{
    private MemoryAccessDomain(MemoryAccessDomainKind kind, int blockIndex)
    {
        if (kind == MemoryAccessDomainKind.FixedBlock && blockIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockIndex));
        }

        Kind = kind;
        BlockIndex = blockIndex;
    }

    public static MemoryAccessDomain AllBlocks => default;

    public MemoryAccessDomainKind Kind { get; }

    public int BlockIndex { get; }

    public static MemoryAccessDomain FixedBlock(int blockIndex)
        => new(MemoryAccessDomainKind.FixedBlock, blockIndex);

    public bool IsSameFixedBlock(MemoryAccessDomain other)
        => Kind == MemoryAccessDomainKind.FixedBlock && this == other;
}

/// <summary>
/// Describes the possible memory accesses performed through one call operand.
/// </summary>
public readonly record struct MemoryEffect(
    MemoryAccessMode Mode,
    MemoryAccessScope Scope = MemoryAccessScope.Inferred,
    MemoryEffectKind Kind = MemoryEffectKind.Direct,
    MemoryAccessDomain AccessDomain = default,
    MemoryAccessPartition AccessPartition = default,
    MemoryOwnerAccess OwnerAccess = default)
{
    public static MemoryEffect None { get; } = new(MemoryAccessMode.None);

    public static MemoryEffect Read { get; } = new(MemoryAccessMode.Read);

    public static MemoryEffect Write { get; } = new(MemoryAccessMode.Write);

    public static MemoryEffect ReadWrite { get; } = new(MemoryAccessMode.ReadWrite);

    public static MemoryEffect ChipRead { get; } = new(MemoryAccessMode.Read, MemoryAccessScope.Chip);

    public static MemoryEffect ChipWrite { get; } = new(MemoryAccessMode.Write, MemoryAccessScope.Chip);

    public static MemoryEffect ChipReadWrite { get; } = new(MemoryAccessMode.ReadWrite, MemoryAccessScope.Chip);

    public static MemoryEffect SystemRead { get; } = new(MemoryAccessMode.Read, MemoryAccessScope.System);

    public static MemoryEffect SystemWrite { get; } = new(MemoryAccessMode.Write, MemoryAccessScope.System);

    public static MemoryEffect SystemReadWrite { get; } = new(MemoryAccessMode.ReadWrite, MemoryAccessScope.System);

    public static MemoryEffect ReductionWrite { get; } = new(
        MemoryAccessMode.Write,
        MemoryAccessScope.Inferred,
        MemoryEffectKind.ReductionAccumulator);

    public static MemoryEffect ReductionReadWrite { get; } = new(
        MemoryAccessMode.ReadWrite,
        MemoryAccessScope.Inferred,
        MemoryEffectKind.ReductionAccumulator);

    public MemoryEffect InFixedBlock(int blockIndex)
        => this with { AccessDomain = MemoryAccessDomain.FixedBlock(blockIndex) };

    public MemoryEffect PartitionedByArgument(int argumentIndex)
        => this with { AccessPartition = MemoryAccessPartition.ByArgument(argumentIndex) };

    public MemoryEffect AcrossPartialOwners()
        => this with { OwnerAccess = MemoryOwnerAccess.PartialGroup };
}

/// <summary>
/// Shared utilities for interpreting operand memory-effect contracts.
/// </summary>
public static class MemoryEffectUtility
{
    /// <summary>
    /// Gets the accesses that reach the operand's physical buffer. Reduction
    /// feedback remains backend-private, so only its final write is visible.
    /// </summary>
    public static MemoryAccessMode GetPhysicalBufferAccessMode(MemoryEffect effect)
        => effect.Kind == MemoryEffectKind.ReductionAccumulator
            ? effect.Mode & MemoryAccessMode.Write
            : effect.Mode;

    /// <summary>
    /// Visits every expression operand with a non-empty memory effect. Tuple and
    /// variadic operands are expanded according to the call's parameter contract.
    /// </summary>
    public static void VisitCallEffects(Call call, Action<Expr, ParameterInfo, MemoryEffect> visitor)
        => VisitCallEffects(
            call,
            (argument, parameter, _, effect) => visitor(argument, parameter, effect));

    /// <summary>
    /// Visits every expression operand with a non-empty memory effect and
    /// reports the concrete call-argument index. Tuple fields retain the index
    /// of their containing argument.
    /// </summary>
    public static void VisitCallEffects(
        Call call,
        Action<Expr, ParameterInfo, int, MemoryEffect> visitor)
    {
        if (call.Target is not Op)
        {
            throw new ArgumentException("Operand memory effects can only be read from an Op call.", nameof(call));
        }

        var arguments = call.Arguments.ToArray();
        var argumentIndex = 0;
        call.ParametersForeach((argument, parameter) =>
        {
            var currentArgumentIndex = argumentIndex++;
            var effect = call.Target is IOpMemoryEffectProvider provider
                ? provider.GetMemoryEffect(parameter, arguments)
                : parameter.MemoryEffect ?? MemoryEffect.None;
            if (effect.Mode == MemoryAccessMode.None)
            {
                return;
            }

            VisitArgument(argument, parameter, currentArgumentIndex, effect);
        });

        void VisitArgument(
            BaseExpr argument,
            ParameterInfo parameter,
            int currentArgumentIndex,
            MemoryEffect effect)
        {
            switch (argument)
            {
                case None:
                    return;
                case IR.Tuple tuple:
                    foreach (var field in tuple.Fields)
                    {
                        VisitArgument(field, parameter, currentArgumentIndex, effect);
                    }

                    return;
                case Expr expression:
                    visitor(expression, parameter, currentArgumentIndex, effect);
                    return;
                default:
                    throw new InvalidOperationException(
                        $"Memory-effect operand {call.Target.GetType().Name}.{parameter.Name} must be an expression, got {argument.GetType().Name}.");
            }
        }
    }

    public static MemoryEffect Merge(MemoryEffect lhs, MemoryEffect rhs)
    {
        if (lhs.Mode == MemoryAccessMode.None)
        {
            return rhs;
        }

        if (rhs.Mode == MemoryAccessMode.None)
        {
            return lhs;
        }

        return new(
            lhs.Mode | rhs.Mode,
            MergeScope(lhs.Scope, rhs.Scope),
            lhs.Kind == rhs.Kind ? lhs.Kind : MemoryEffectKind.Direct,
            MergeAccessDomain(lhs.AccessDomain, rhs.AccessDomain),
            MergeAccessPartition(lhs.AccessPartition, rhs.AccessPartition),
            MergeOwnerAccess(lhs.OwnerAccess, rhs.OwnerAccess));
    }

    public static MemoryAccessScope MergeScope(MemoryAccessScope lhs, MemoryAccessScope rhs)
        => lhs == MemoryAccessScope.System || rhs == MemoryAccessScope.System
            ? MemoryAccessScope.System
            : lhs == MemoryAccessScope.Chip || rhs == MemoryAccessScope.Chip
                ? MemoryAccessScope.Chip
            : lhs == MemoryAccessScope.Block || rhs == MemoryAccessScope.Block
                ? MemoryAccessScope.Block
                : MemoryAccessScope.Inferred;

    public static MemoryAccessDomain MergeAccessDomain(
        MemoryAccessDomain lhs,
        MemoryAccessDomain rhs)
        => lhs.IsSameFixedBlock(rhs)
            ? lhs
            : MemoryAccessDomain.AllBlocks;

    public static MemoryAccessPartition MergeAccessPartition(
        MemoryAccessPartition lhs,
        MemoryAccessPartition rhs)
        => lhs == rhs
            ? lhs
            : MemoryAccessPartition.WholeResource;

    public static MemoryOwnerAccess MergeOwnerAccess(
        MemoryOwnerAccess lhs,
        MemoryOwnerAccess rhs)
        => lhs == MemoryOwnerAccess.PartialGroup || rhs == MemoryOwnerAccess.PartialGroup
            ? MemoryOwnerAccess.PartialGroup
            : MemoryOwnerAccess.Local;
}
