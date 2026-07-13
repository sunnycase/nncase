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
    /// backend carries this state privately and commits the declared memory
    /// effect once when the region completes.
    /// </summary>
    ReductionAccumulator,
}

/// <summary>
/// Refines the default operand effects declared by <see cref="ParameterInfo"/>
/// when an operation's static attributes change its memory access scope.
/// </summary>
public interface IOpMemoryEffectProvider
{
    MemoryEffect GetMemoryEffect(ParameterInfo parameter);
}

/// <summary>
/// Describes the possible memory accesses performed through one call operand.
/// </summary>
public readonly record struct MemoryEffect(
    MemoryAccessMode Mode,
    MemoryAccessScope Scope = MemoryAccessScope.Inferred,
    MemoryEffectKind Kind = MemoryEffectKind.Direct)
{
    public static MemoryEffect None { get; } = new(MemoryAccessMode.None);

    public static MemoryEffect Read { get; } = new(MemoryAccessMode.Read);

    public static MemoryEffect Write { get; } = new(MemoryAccessMode.Write);

    public static MemoryEffect ReadWrite { get; } = new(MemoryAccessMode.ReadWrite);

    public static MemoryEffect ChipRead { get; } = new(MemoryAccessMode.Read, MemoryAccessScope.Chip);

    public static MemoryEffect ChipWrite { get; } = new(MemoryAccessMode.Write, MemoryAccessScope.Chip);

    public static MemoryEffect ChipReadWrite { get; } = new(MemoryAccessMode.ReadWrite, MemoryAccessScope.Chip);

    public static MemoryEffect ReductionWrite { get; } = new(
        MemoryAccessMode.Write,
        MemoryAccessScope.Inferred,
        MemoryEffectKind.ReductionAccumulator);

    public static MemoryEffect ReductionReadWrite { get; } = new(
        MemoryAccessMode.ReadWrite,
        MemoryAccessScope.Inferred,
        MemoryEffectKind.ReductionAccumulator);
}

/// <summary>
/// Shared utilities for interpreting operand memory-effect contracts.
/// </summary>
public static class MemoryEffectUtility
{
    /// <summary>
    /// Visits every expression operand with a non-empty memory effect. Tuple and
    /// variadic operands are expanded according to the call's parameter contract.
    /// </summary>
    public static void VisitCallEffects(Call call, Action<Expr, ParameterInfo, MemoryEffect> visitor)
    {
        if (call.Target is not Op)
        {
            throw new ArgumentException("Operand memory effects can only be read from an Op call.", nameof(call));
        }

        call.ParametersForeach((argument, parameter) =>
        {
            var effect = call.Target is IOpMemoryEffectProvider provider
                ? provider.GetMemoryEffect(parameter)
                : parameter.MemoryEffect ?? MemoryEffect.None;
            if (effect.Mode == MemoryAccessMode.None)
            {
                return;
            }

            VisitArgument(argument, parameter, effect);
        });

        void VisitArgument(BaseExpr argument, ParameterInfo parameter, MemoryEffect effect)
        {
            switch (argument)
            {
                case None:
                    return;
                case IR.Tuple tuple:
                    foreach (var field in tuple.Fields)
                    {
                        VisitArgument(field, parameter, effect);
                    }

                    return;
                case Expr expression:
                    visitor(expression, parameter, effect);
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
            lhs.Kind == rhs.Kind ? lhs.Kind : MemoryEffectKind.Direct);
    }

    public static MemoryAccessScope MergeScope(MemoryAccessScope lhs, MemoryAccessScope rhs)
        => lhs == MemoryAccessScope.Chip || rhs == MemoryAccessScope.Chip
            ? MemoryAccessScope.Chip
            : lhs == MemoryAccessScope.Block || rhs == MemoryAccessScope.Block
                ? MemoryAccessScope.Block
                : MemoryAccessScope.Inferred;
}
