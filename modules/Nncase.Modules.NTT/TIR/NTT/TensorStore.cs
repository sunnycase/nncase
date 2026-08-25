// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class TensorStore : NTTKernelOp
{
    public static readonly ParameterInfo Src = new(typeof(TensorStore), 0, "src", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Dest = new(typeof(TensorStore), 1, "dest", memoryEffect: MemoryEffect.ChipWrite);

    public IRArray<SBP> NdSbp { get; }

    public Placement Placement { get; }

    public bool RequiresCoordinatorMaterialization(
        TIR.Buffer source,
        TIR.Buffer? destination)
        => source.IsCanonicalReplicated &&
           (destination is null || destination.MemSpan.Buffer.Location == MemoryLocation.Output);

    public override MemoryEffect GetMemoryEffect(
        ParameterInfo parameter,
        IReadOnlyList<BaseExpr> arguments)
    {
        var effect = base.GetMemoryEffect(parameter, arguments);
        if (ReferenceEquals(parameter, Dest) &&
            arguments.Count > Dest.Index &&
            IsCompactPerOwner(arguments[Dest.Index]))
        {
            // Every block writes only its own ABI component. Cross-block reads
            // performed while forming that component remain effects of Src;
            // publishing Dest itself requires only owner-local ordering.
            effect = effect with { Scope = MemoryAccessScope.Inferred };
        }

        if ((ReferenceEquals(parameter, Src) || ReferenceEquals(parameter, Dest)) &&
            arguments.Count > Dest.Index &&
            arguments[Src.Index] is TIR.Buffer source &&
            arguments[Dest.Index] is TIR.Buffer destination &&
            RequiresCoordinatorMaterialization(source, destination))
        {
            return effect.InFixedBlock(0);
        }

        return effect;
    }

    private static bool IsCompactPerOwner(BaseExpr expression)
        => expression switch
        {
            TIR.Buffer buffer =>
                buffer.DistributedStorageKind == DistributedBufferStorageKind.CompactPerOwner,
            BufferVar bufferVar =>
                bufferVar.LayoutAnnotation.DistributedStorageKind == DistributedBufferStorageKind.CompactPerOwner,
            _ => false,
        };
}
