// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Reinterprets the element lane layout while preserving the scalar value stream.
/// </summary>
public sealed partial class Bitcast : NTTKernelOp
{
    /// <summary>
    /// Gets the input buffer.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Bitcast), 0, "input");

    /// <summary>
    /// Gets the output buffer.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(Bitcast), 1, "output");
}
