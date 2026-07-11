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

public sealed partial class PagedAttention : NTTKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(PagedAttention), 0, "q", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KVCaches = new(typeof(PagedAttention), 1, "kvCaches", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo Extra = new(typeof(PagedAttention), 2, "extra", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Scale = new(typeof(PagedAttention), 3, "scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo LayerId = new(typeof(PagedAttention), 4, "layerId", IsDimensionType(), memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Output = new(typeof(PagedAttention), 5, "Output", memoryEffect: MemoryEffect.Write);

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }
}
