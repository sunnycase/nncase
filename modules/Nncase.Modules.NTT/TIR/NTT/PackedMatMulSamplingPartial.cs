// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Computes one local packed LM-head shard together with token-local sampling
/// processors. All outputs are materialized for a separate sampling combine.
/// </summary>
public sealed partial class PackedMatMulSamplingPartial : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMulSamplingPartial), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMulSamplingPartial), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo State = new(typeof(PackedMatMulSamplingPartial), 2, "state", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo Logits = new(typeof(PackedMatMulSamplingPartial), 3, "logits", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo ProcessedLogits = new(typeof(PackedMatMulSamplingPartial), 4, "processed_logits", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo ArgMaxState = new(typeof(PackedMatMulSamplingPartial), 5, "argmax_state", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo Scale = new(typeof(PackedMatMulSamplingPartial), 6, "scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Addend = new(typeof(PackedMatMulSamplingPartial), 7, "addend", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo LhsScale = new(typeof(PackedMatMulSamplingPartial), 8, "lhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedMatMulSamplingPartial), 9, "rhs_scale", memoryEffect: MemoryEffect.Read);

    public DataType AccumulatorDataType { get; }

    public DataType OutputDataType { get; }

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    public DistributedType PackedOutputType { get; }

    public DistributedType LogitsType { get; }

    public SamplerConfig Config { get; }

    public override string DisplayProperty()
        => $"AccumulatorDataType: {AccumulatorDataType}, OutputDataType: {OutputDataType}, " +
           $"RhsLayout: {RhsLayout}, Config: {Config}";
}
