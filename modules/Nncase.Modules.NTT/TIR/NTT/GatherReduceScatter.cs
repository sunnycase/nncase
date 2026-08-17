// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class GatherReduceScatter : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(GatherReduceScatter), 0, "input", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(GatherReduceScatter), 1, "output", memoryEffect: MemoryEffect.Write);

    public DistributedType InType { get; }

    public DistributedType OutType { get; }

    public override MemoryEffect GetMemoryEffect(ParameterInfo parameter)
    {
        if (ReferenceEquals(parameter, Input))
        {
            return InType.Partial is null ? MemoryEffect.Read : MemoryEffect.ChipRead;
        }

        if (ReferenceEquals(parameter, Output))
        {
            // Output visibility follows its physical backing. CompactLocal
            // outputs are private replicas, while CanonicalGlobal outputs use
            // chip-visible storage and are promoted by resource inference.
            return MemoryEffect.Write;
        }

        return base.GetMemoryEffect(parameter);
    }
}
