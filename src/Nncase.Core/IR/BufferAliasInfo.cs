// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

/// <summary>
/// Implemented by descriptor-only operations whose result operands alias input storage.
/// </summary>
public interface IBufferAliasOp
{
    IReadOnlyList<BufferAliasInfo> BufferAliases { get; }
}

/// <summary>
/// Describes one zero-copy buffer alias produced by an operation.
/// </summary>
/// <param name="Source">The operand that owns the physical storage.</param>
/// <param name="Result">The operand whose logical descriptor aliases <paramref name="Source"/>.</param>
public sealed record BufferAliasInfo(ParameterInfo Source, ParameterInfo Result);
