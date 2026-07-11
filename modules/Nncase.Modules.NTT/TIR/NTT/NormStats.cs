// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class NormStats : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(NormStats), 0, "input", ParameterKind.Input, MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(NormStats), 1, "output", ParameterKind.Input, MemoryEffect.Write);

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, UseMean: {UseMean}";
}
