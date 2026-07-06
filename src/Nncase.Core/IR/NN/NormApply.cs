// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// Applies normalization from additive statistics.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NormApply : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NormApply), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets additive normalization statistics.
    /// </summary>
    public static readonly ParameterInfo Stats = new(typeof(NormApply), 1, "stats", ParameterKind.Input);

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(NormApply), 2, "scale", ParameterKind.Input);

    /// <summary>
    /// Gets bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(NormApply), 3, "bias", ParameterKind.Input);

    /// <summary>
    /// Gets first normalized axis.
    /// </summary>
    public int Axis { get; }

    /// <summary>
    /// Gets epsilon.
    /// </summary>
    public float Epsilon { get; }

    /// <summary>
    /// Gets a value indicating whether mean should be applied.
    /// </summary>
    public bool UseMean { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}";
}
