// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Math;

/// <summary>
/// Matrix multiplication with dynamically quantized NVFP4 activations and
/// pre-quantized NVFP4 weights.
/// </summary>
/// <remarks>
/// <para>
/// The logical lhs has shape [..., M, K]. Two E2M1 weight values are packed,
/// low nibble first, in each byte of <see cref="RhsPacked"/>, whose physical
/// shape is [N, K / 2]. <see cref="RhsScale"/> has physical shape
/// [N, K / <see cref="GroupSize"/>] and E4M3 element type.
/// </para>
/// <para>
/// The lhs is dynamically quantized per row and K group. Both block-scale
/// tensors include their respective global scale; the accumulated result is
/// multiplied by the reciprocal product of <see cref="LhsGlobalScale"/> and
/// <see cref="RhsGlobalScale"/>.
/// </para>
/// </remarks>
[PatternFunctionalGenerator]
public sealed partial class NVFP4MatMul : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(NVFP4MatMul), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo RhsPacked = new(typeof(NVFP4MatMul), 1, "rhs_packed", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(NVFP4MatMul), 2, "rhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo LhsGlobalScale = new(typeof(NVFP4MatMul), 3, "lhs_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo RhsGlobalScale = new(typeof(NVFP4MatMul), 4, "rhs_global_scale", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public long GroupSize { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, GroupSize: {GroupSize}";
}
