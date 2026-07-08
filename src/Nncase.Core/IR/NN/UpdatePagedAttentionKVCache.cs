// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

[PatternFunctionalGenerator]
public sealed partial class UpdatePagedAttentionKVCache : Op
{
    public static readonly ParameterInfo Slots = new(typeof(UpdatePagedAttentionKVCache), 0, "slots", ParameterKind.Input);

    public static readonly ParameterInfo KVCaches = new(typeof(UpdatePagedAttentionKVCache), 1, "kvCaches", ParameterKind.Attribute);

    public static readonly ParameterInfo LayerId = new(typeof(UpdatePagedAttentionKVCache), 2, "layerId", IsDimensionType(), ParameterKind.Attribute);

    public AttentionCacheKind CacheKind { get; }

    public IRArray<AttentionDimKind> Layout { get; }

    public override string DisplayProperty() => $"CacheKind: {CacheKind}, Layout [{string.Join(',', Layout)}]";
}
