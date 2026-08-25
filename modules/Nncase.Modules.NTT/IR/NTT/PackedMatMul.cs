// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Physical RHS layout used by <see cref="PackedMatMul"/>.
/// </summary>
public enum PackedMatMulRhsLayout
{
    /// <summary>
    /// CPU-oriented [..., N, K]&lt;NPack, NVector&gt; layout.
    /// </summary>
    NMajor,

    /// <summary>
    /// GPU-oriented [..., K, N]&lt;NVector, KPack, KVector&gt; layout.
    /// </summary>
    KMajor,

    /// <summary>
    /// GPU MMA-oriented [..., N, K / (KPack * KVector)]&lt;KPack, KVector&gt;
    /// layout. Its scalar physical order is row-major [..., N, K].
    /// </summary>
    NMajorKPacked,
}

[PatternFunctionalGenerator]
public sealed partial class PackedMatMul : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMul), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMul), 1, "rhs", ParameterKind.Input);

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(PackedMatMul), 2, "scale", ParameterKind.Attribute);

    /// <summary>
    /// Gets the optional tensor added to the packed matmul result.
    /// </summary>
    public static readonly ParameterInfo Addend = new(typeof(PackedMatMul), 3, "addend", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public bool FusedReduce { get; }

    public PackedMatMulRhsLayout RhsLayout { get; }

    public override string DisplayProperty() => $"FusedReduce: {FusedReduce}, RhsLayout: {RhsLayout}";
}
