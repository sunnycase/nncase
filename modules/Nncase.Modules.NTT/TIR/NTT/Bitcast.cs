// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Reinterprets the element lane layout without moving storage.
/// </summary>
public sealed partial class Bitcast : NTTKernelOp, IBufferAliasOp
{
    public static readonly ParameterInfo Input = new(typeof(Bitcast), 0, "input", memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Output = new(typeof(Bitcast), 1, "output", memoryEffect: MemoryEffect.None);

    public IReadOnlyList<BufferAliasInfo> BufferAliases => [new(Input, Output)];
}
