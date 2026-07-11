// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.NTT;

public sealed partial class UpdatePagedAttentionKVCache : NTTKernelOp
{
    public static readonly ParameterInfo Slots = new(typeof(UpdatePagedAttentionKVCache), 0, "slots", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KVCaches = new(typeof(UpdatePagedAttentionKVCache), 1, "kvCaches", memoryEffect: MemoryEffect.ChipWrite);

    public static readonly ParameterInfo LayerId = new(typeof(UpdatePagedAttentionKVCache), 2, "layerId", IsDimensionType(), memoryEffect: MemoryEffect.None);

    public AttentionCacheKind CacheKind { get; }

    public IRArray<AttentionDimKind> Layout { get; }
}
