// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Creates a logical reshape view over an existing buffer.
/// </summary>
public sealed partial class Reshape : NTTKernelOp, IBufferAliasOp
{
    public static readonly ParameterInfo Input = new(typeof(Reshape), 0, "input", memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Output = new(typeof(Reshape), 1, "output", memoryEffect: MemoryEffect.None);

    public IReadOnlyList<BufferAliasInfo> BufferAliases => [new(Input, Output)];
}
