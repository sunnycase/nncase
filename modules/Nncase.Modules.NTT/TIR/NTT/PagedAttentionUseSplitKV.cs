// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.NTT;

/// <summary>
/// Tests whether a paged-attention invocation must use its planned split-KV path.
/// </summary>
public sealed partial class PagedAttentionUseSplitKV : Op
{
    public static readonly ParameterInfo KVCaches = new(
        typeof(PagedAttentionUseSplitKV),
        0,
        "kv_caches",
        IsTensor());

    public long DirectContextThreshold { get; }

    public override bool CanFoldConstCall => false;
}
