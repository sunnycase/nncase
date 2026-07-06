// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// Computes additive normalization statistics for a LayerNorm/RMSNorm suffix.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NormStats : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NormStats), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets first normalized axis.
    /// </summary>
    public int Axis { get; }

    /// <summary>
    /// Gets a value indicating whether mean should be computed.
    /// </summary>
    public bool UseMean { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, UseMean: {UseMean}";
}
