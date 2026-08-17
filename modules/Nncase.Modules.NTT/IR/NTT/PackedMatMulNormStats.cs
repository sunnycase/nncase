// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Packed matrix multiplication that also produces block-local normalization
/// statistics for its final output values.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedMatMulNormStats : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMulNormStats), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMulNormStats), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo Scale = new(typeof(PackedMatMulNormStats), 2, "scale", ParameterKind.Attribute);

    public static readonly ParameterInfo Addend = new(typeof(PackedMatMulNormStats), 3, "addend", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public PackedMatMulRhsLayout RhsLayout { get; }

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, RhsLayout: {RhsLayout}, Axis: {Axis}, UseMean: {UseMean}";
}
