// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class NormApply : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(NormApply), 0, "input", ParameterKind.Input, MemoryEffect.Read);

    public static readonly ParameterInfo Stats = new(typeof(NormApply), 1, "stats", ParameterKind.Input, MemoryEffect.Read);

    public static readonly ParameterInfo Scale = new(typeof(NormApply), 2, "scale", ParameterKind.Input, MemoryEffect.Read);

    public static readonly ParameterInfo Bias = new(typeof(NormApply), 3, "bias", ParameterKind.Input, MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(NormApply), 4, "output", ParameterKind.Input, MemoryEffect.Write);

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}";
}
