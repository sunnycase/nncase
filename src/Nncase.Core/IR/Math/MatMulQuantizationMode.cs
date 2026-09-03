// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Math;

/// <summary>
/// Quantization contract for matrix projections embedded in fused operators.
/// </summary>
public enum MatMulQuantizationMode
{
    /// <summary>
    /// Both operands use their declared floating-point element types.
    /// </summary>
    None,

    /// <summary>
    /// The operands use independently supplied tensor-wide scales.
    /// </summary>
    StaticTensor,

    /// <summary>
    /// Activations are quantized dynamically per logical row and weights carry
    /// one inverse scale per output channel.
    /// </summary>
    DynamicTensor,

    /// <summary>
    /// Activations are quantized dynamically per K block and weights carry a
    /// two-dimensional [N-block, K-block] inverse-scale tensor.
    /// </summary>
    DynamicBlock,
}
