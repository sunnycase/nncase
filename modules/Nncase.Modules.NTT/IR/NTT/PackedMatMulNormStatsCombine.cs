// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Materializes an optional split-K packed-matmul result, adds the residual,
/// and produces local additive normalization statistics.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedMatMulNormStatsCombine : Op
{
    public static readonly ParameterInfo Input = new(typeof(PackedMatMulNormStatsCombine), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Addend = new(typeof(PackedMatMulNormStatsCombine), 1, "addend", ParameterKind.Input);

    public IRType OutputType { get; }

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() =>
        $"OutputType: {OutputType}, Axis: {Axis}, UseMean: {UseMean}";
}
