// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Target-packed form of a scaled low-precision matrix multiplication.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedScaledMatMul : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedScaledMatMul), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo Rhs = new(typeof(PackedScaledMatMul), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo LhsScale = new(typeof(PackedScaledMatMul), 2, "lhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedScaledMatMul), 3, "rhs_scale", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public PackedMatMulRhsLayout RhsLayout { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, RhsLayout: {RhsLayout}";
}
