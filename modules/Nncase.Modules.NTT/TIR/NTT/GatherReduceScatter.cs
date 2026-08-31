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

    public override MemoryEffect GetMemoryEffect(
        ParameterInfo parameter,
        IReadOnlyList<BaseExpr> arguments)
    {
        if (ReferenceEquals(parameter, Input))
        {
            return InType.Partial is null ? MemoryEffect.Read : MemoryEffect.ChipRead;
        }

        if (ReferenceEquals(parameter, Output))
        {
            // A CanonicalGlobal reshard may be materialized by one source
            // owner and consumed by every owner of a broadcast output shard.
            // Expose that remote publication explicitly so synchronization
            // planning cannot treat equal producer/consumer layouts as an
            // owner-local RAW dependence.
            var isCanonicalGlobal = arguments.Count > Output.Index &&
                arguments[Output.Index] is TIR.Buffer
                {
                    DistributedStorageKind: DistributedBufferStorageKind.CanonicalGlobal,
                };
            return isCanonicalGlobal ? MemoryEffect.ChipWrite : MemoryEffect.Write;
        }

        return base.GetMemoryEffect(parameter, arguments);
    }
}
