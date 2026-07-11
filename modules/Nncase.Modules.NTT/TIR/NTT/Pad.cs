// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.NTT;

/// <summary>
/// Concat expression.
/// </summary>
public sealed partial class Pad : NTTKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Pad), 0, "input", memoryEffect: MemoryEffect.Read);

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Pads = new(typeof(Pad), 1, "pads", IsPaddingsType(), memoryEffect: MemoryEffect.None);

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(Pad), 2, "output", memoryEffect: MemoryEffect.Write);

    public float PadValue { get; }

    public IRArray<int> ActualPadAxes { get; }
}
