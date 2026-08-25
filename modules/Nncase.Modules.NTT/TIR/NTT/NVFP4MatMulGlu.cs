// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Direct TIR fused NVFP4 gate/up projections and GLU activation.
/// </summary>
public sealed partial class NVFP4MatMulGlu : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(NVFP4MatMulGlu), 0, "input", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeightPacked = new(typeof(NVFP4MatMulGlu), 1, "gate_weight_packed", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeightPacked = new(typeof(NVFP4MatMulGlu), 2, "up_weight_packed", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeightScale = new(typeof(NVFP4MatMulGlu), 3, "gate_weight_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeightScale = new(typeof(NVFP4MatMulGlu), 4, "up_weight_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateInputGlobalScale = new(typeof(NVFP4MatMulGlu), 5, "gate_input_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpInputGlobalScale = new(typeof(NVFP4MatMulGlu), 6, "up_input_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeightGlobalScale = new(typeof(NVFP4MatMulGlu), 7, "gate_weight_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeightGlobalScale = new(typeof(NVFP4MatMulGlu), 8, "up_weight_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(NVFP4MatMulGlu), 9, "output", memoryEffect: MemoryEffect.Write);

    public GluType GluType { get; }

    public long GroupSize { get; }

    public override string DisplayProperty() => $"GluType: {GluType}, GroupSize: {GroupSize}";
}
