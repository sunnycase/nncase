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
    public static readonly ParameterInfo Q = new(typeof(PagedAttention), 0, "q");

    public static readonly ParameterInfo KVCaches = new(typeof(PagedAttention), 1, "kvCaches");

    public static readonly ParameterInfo Extra = new(typeof(PagedAttention), 2, "extra");

    public static readonly ParameterInfo Scale = new(typeof(PagedAttention), 3, "scale");

    public static readonly ParameterInfo LayerId = new(typeof(PagedAttention), 4, "layerId", IsDimensionType());

    public static readonly ParameterInfo Output = new(typeof(PagedAttention), 5, "Output");

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }
}
