// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Math;

/// <summary>
/// Matrix multiplication with explicitly scaled low-precision operands.
/// </summary>
/// <remarks>
/// The logical computation is
/// <c>cast(cast(lhs / lhs_scale, rhs.dtype), f32) * lhs_scale @
/// cast(rhs, f32) * rhs_scale</c>, accumulated in f32 and converted to
/// <see cref="OutputDataType"/>. The scale tensors describe the quantized
/// values and are semantic inputs rather than backend tuning attributes.
/// </remarks>
[PatternFunctionalGenerator]
public sealed partial class ScaledMatMul : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(ScaledMatMul), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo Rhs = new(typeof(ScaledMatMul), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo LhsScale = new(typeof(ScaledMatMul), 2, "lhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(ScaledMatMul), 3, "rhs_scale", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public override string DisplayProperty() => $"OutputDataType: {OutputDataType}";
}
