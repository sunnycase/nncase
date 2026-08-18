// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Packed matrix multiplication fused with the token-local sampling phase.
/// Raw and processed logits are materialized for a separate cross-shard combine.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedMatMulSamplingPartial : Op
{
    public static readonly ParameterInfo Lhs = new(
        typeof(PackedMatMulSamplingPartial),
        0,
        "lhs",
        ParameterKind.Input,
        MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(
        typeof(PackedMatMulSamplingPartial),
        1,
        "rhs",
        ParameterKind.Input,
        MemoryEffect.Read);

    public static readonly ParameterInfo State = new(
        typeof(PackedMatMulSamplingPartial),
        2,
        "state",
        ParameterKind.Attribute,
        MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo Scale = new(
        typeof(PackedMatMulSamplingPartial),
        3,
        "scale",
        ParameterKind.Attribute);

    public static readonly ParameterInfo Addend = new(
        typeof(PackedMatMulSamplingPartial),
        4,
        "addend",
        ParameterKind.Input,
        MemoryEffect.Read);

    public DataType OutputDataType { get; }

    public PackedMatMulRhsLayout RhsLayout { get; }

    public SamplerConfig Config { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty()
        => $"OutputDataType: {OutputDataType}, RhsLayout: {RhsLayout}, Config: {Config}";
}
