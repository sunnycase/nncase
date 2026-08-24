// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Math;

/// <summary>
/// Matrix multiplication with dynamically block-quantized activations and
/// block-scaled low-precision weights.
/// </summary>
/// <remarks>
/// The lhs is quantized independently for every row and K block. The rhs scale
/// tensor is indexed as [ceil(N / <see cref="WeightBlockN"/>),
/// ceil(K / <see cref="WeightBlockK"/>)]. Both operands are dequantized per K
/// block, accumulated in f32, and converted to <see cref="OutputDataType"/>.
/// </remarks>
[PatternFunctionalGenerator]
public sealed partial class BlockScaledMatMul : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(BlockScaledMatMul), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo Rhs = new(typeof(BlockScaledMatMul), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(BlockScaledMatMul), 2, "rhs_scale", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, WeightBlock: [{WeightBlockN}, {WeightBlockK}]";
}
