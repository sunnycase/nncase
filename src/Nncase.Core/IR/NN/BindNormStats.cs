// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// Associates materialized normalization statistics with their source tensor.
/// </summary>
/// <remarks>
/// This operation is an optimization constraint: its value is <see cref="Stats"/>,
/// while <see cref="Input"/> preserves the distribution relation of the original
/// <see cref="NormStats"/> computation. It is removed after AutoDistributed.
/// </remarks>
[PatternFunctionalGenerator]
public sealed partial class BindNormStats : Op
{
    /// <summary>
    /// Gets the tensor described by the statistics.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(BindNormStats), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets the materialized normalization statistics.
    /// </summary>
    public static readonly ParameterInfo Stats = new(typeof(BindNormStats), 1, "stats", ParameterKind.Input);

    /// <summary>
    /// Gets first normalized axis.
    /// </summary>
    public int Axis { get; }

    /// <summary>
    /// Gets a value indicating whether mean statistics are present.
    /// </summary>
    public bool UseMean { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, UseMean: {UseMean}";
}
