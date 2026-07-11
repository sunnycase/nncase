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
/// Describes the possible memory accesses performed through one call operand.
/// </summary>
public readonly record struct MemoryEffect(MemoryAccessMode Mode, MemoryAccessScope Scope = MemoryAccessScope.Inferred)
{
    public static MemoryEffect None { get; } = new(MemoryAccessMode.None);

    public static MemoryEffect Read { get; } = new(MemoryAccessMode.Read);

    public static MemoryEffect Write { get; } = new(MemoryAccessMode.Write);

    public static MemoryEffect ReadWrite { get; } = new(MemoryAccessMode.ReadWrite);

    public static MemoryEffect ChipRead { get; } = new(MemoryAccessMode.Read, MemoryAccessScope.Chip);

    public static MemoryEffect ChipWrite { get; } = new(MemoryAccessMode.Write, MemoryAccessScope.Chip);

    public static MemoryEffect ChipReadWrite { get; } = new(MemoryAccessMode.ReadWrite, MemoryAccessScope.Chip);
}
