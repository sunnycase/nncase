// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.NN;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectUpdatePagedAttentionKVCache(UpdatePagedAttentionKVCache op, Call call)
    {
        var slots = (Expr)call[UpdatePagedAttentionKVCache.Slots];
        var kvCache = (Expr)call[UpdatePagedAttentionKVCache.KVCaches];
        var layerId = (Dimension)call[UpdatePagedAttentionKVCache.LayerId];
        var rank = slots.CheckedShape.Rank;
        if (rank <= 0 || kvCache.CheckedShape.Rank != 0)
        {
            return call;
        }

        var objectMap = AffineMap.FromCallable((domains, symbols) => Array.Empty<AffineRange>(), rank, 0);
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(slots, AffineMap.Identity(rank), out var slotsTile)
            .Read(kvCache, objectMap, out var kvCacheTile)
            .Write(kvCache, objectMap, out var _)
            .Body(TIR.F.NTT.UpdatePagedAttentionKVCache(slotsTile, kvCacheTile, layerId, op.CacheKind, op.Layout))
            .Build();
    }
}
