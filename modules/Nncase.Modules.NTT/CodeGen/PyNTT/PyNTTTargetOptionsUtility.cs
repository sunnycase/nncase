// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Targets;

namespace Nncase.CodeGen.PyNTT;

internal static class PyNTTTargetOptionsUtility
{
    public static PyNTTTargetOptions Normalize(CompileOptions compileOptions)
    {
        return compileOptions.TargetOptions switch
        {
            PyNTTTargetOptions pynttOptions => pynttOptions,
            NTTTargetOptions nttOptions => FromNTTTargetOptions(nttOptions),
            _ => new PyNTTTargetOptions(),
        };
    }

    private static PyNTTTargetOptions FromNTTTargetOptions(NTTTargetOptions nttOptions)
    {
        return new PyNTTTargetOptions
        {
            ModelName = nttOptions.ModelName,
            Vectorize = nttOptions.Vectorize,
            UnifiedMemoryArch = nttOptions.UnifiedMemoryArch,
            MemoryAccessArch = nttOptions.MemoryAccessArch,
            NocArch = nttOptions.NocArch,
            HierarchyKind = nttOptions.HierarchyKind,
            Hierarchies = nttOptions.Hierarchies,
            HierarchyNames = nttOptions.HierarchyNames,
            HierarchySizes = nttOptions.HierarchySizes,
            HierarchyLatencies = nttOptions.HierarchyLatencies,
            HierarchyBandWidths = nttOptions.HierarchyBandWidths,
            MemoryCapacities = nttOptions.MemoryCapacities,
            MemoryBandWidths = nttOptions.MemoryBandWidths,
            DistributedScheme = nttOptions.DistributedScheme,
            CustomOpScheme = nttOptions.CustomOpScheme,
        };
    }
}
