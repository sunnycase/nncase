// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

public sealed partial class PackedMatMulGlu : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(PackedMatMulGlu), 0, "input", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeight = new(typeof(PackedMatMulGlu), 1, "gate_weight", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeight = new(typeof(PackedMatMulGlu), 2, "up_weight", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateBias = new(typeof(PackedMatMulGlu), 3, "gate_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpBias = new(typeof(PackedMatMulGlu), 4, "up_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateInputScale = new(typeof(PackedMatMulGlu), 5, "gate_input_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpInputScale = new(typeof(PackedMatMulGlu), 6, "up_input_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo GateWeightScale = new(typeof(PackedMatMulGlu), 7, "gate_weight_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo UpWeightScale = new(typeof(PackedMatMulGlu), 8, "up_weight_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PackedMatMulGlu), 9, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public GluType GluType { get; }

    public override string DisplayProperty() => $"GluType: {GluType}";
}
