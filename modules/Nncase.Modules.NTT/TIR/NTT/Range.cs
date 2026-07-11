// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class Range : NTTKernelOp
{
    /// <summary>
    /// Gets begin.
    /// </summary>
    public static readonly ParameterInfo Begin = new(typeof(Range), 0, "begin", memoryEffect: MemoryEffect.Read);

    /// <summary>
    /// Gets end.
    /// </summary>
    public static readonly ParameterInfo End = new(typeof(Range), 1, "end", memoryEffect: MemoryEffect.Read);

    /// <summary>
    /// Gets step.
    /// </summary>
    public static readonly ParameterInfo Step = new(typeof(Range), 2, "step", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(Range), 3, "output", memoryEffect: MemoryEffect.Write);
}
