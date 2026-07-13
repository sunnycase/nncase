// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class GatherReduceScatter : NTTKernelOp, IOpMemoryEffectProvider
{
    public static readonly ParameterInfo Input = new(typeof(GatherReduceScatter), 0, "input", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(GatherReduceScatter), 1, "output", memoryEffect: MemoryEffect.ChipWrite);

    public DistributedType InType { get; }

    public DistributedType OutType { get; }

    public MemoryEffect GetMemoryEffect(ParameterInfo parameter)
    {
        if (ReferenceEquals(parameter, Input))
        {
            return InType.Partial is null ? MemoryEffect.Read : MemoryEffect.ChipRead;
        }

        if (ReferenceEquals(parameter, Output))
        {
            return MemoryEffect.ChipWrite;
        }

        throw new ArgumentOutOfRangeException(nameof(parameter), parameter, "Unknown GatherReduceScatter operand.");
    }
}
