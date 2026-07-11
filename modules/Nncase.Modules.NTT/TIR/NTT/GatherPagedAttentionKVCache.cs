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

namespace Nncase.TIR.NTT;

public sealed partial class GatherPagedAttentionKVCache : NTTKernelOp
{
    public static readonly ParameterInfo ShardId = new(typeof(GatherPagedAttentionKVCache), 0, "ShardId", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KVCaches = new(typeof(GatherPagedAttentionKVCache), 1, "kvCaches", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo Output = new(typeof(GatherPagedAttentionKVCache), 2, "Output", memoryEffect: MemoryEffect.Write);
}
