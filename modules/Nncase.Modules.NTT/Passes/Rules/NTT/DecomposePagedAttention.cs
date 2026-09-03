// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Rules.NTT;

/// <summary>
/// Makes split-KV paged-attention states explicit before distributed search.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposePagedAttention : IRewriteRule
{
    private readonly int _splitHierarchyAxis;
    private readonly int _splitCount;

    public DecomposePagedAttention(int splitHierarchyAxis, int splitCount)
    {
        if (splitHierarchyAxis < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(splitHierarchyAxis));
        }

        if (splitCount <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(splitCount));
        }

        _splitHierarchyAxis = splitHierarchyAxis;
        _splitCount = splitCount;
    }

    public IPattern Pattern { get; } = IsPagedAttention(
        "pagedAttention",
        "pagedAttentionCall",
        _ => true,
        IsWildcard("q"),
        IsWildcard("kvCaches"),
        IsWildcard("extra"),
        IsWildcard("scale"),
        IsWildcard("layerId"),
        IsWildcard("outputGate"));

    private Expr? GetReplace(
        PagedAttention pagedAttention,
        Call pagedAttentionCall,
        Expr q,
        Expr kvCaches,
        Expr extra,
        Expr scale,
        Dimension layerId,
        Expr outputGate)
    {
        var seqAxis = pagedAttention.Layout.IndexOf(AttentionDimKind.Seq);
        if (seqAxis < 0 ||
            pagedAttentionCall.CheckedType is not TensorType and not DistributedType ||
            pagedAttentionCall.CheckedTensorType.Shape is not RankedShape shape ||
            !shape[seqAxis].IsFixed ||
            shape[seqAxis].FixedValue != 1)
        {
            return null;
        }

        var partial = IR.F.NTT.PagedAttentionPartial(
            q,
            kvCaches,
            extra,
            scale,
            layerId,
            pagedAttention.Layout,
            pagedAttention.HiddenSize,
            _splitHierarchyAxis,
            _splitCount);
        return IR.F.NTT.PagedAttentionCombine(
                IR.F.Tensors.GetItem(partial, 0),
                IR.F.Tensors.GetItem(partial, 1),
                IR.F.Tensors.GetItem(partial, 2),
                outputGate,
                pagedAttention.Layout,
                pagedAttention.HiddenSize,
                pagedAttentionCall.CheckedTensorType.DType,
                pagedAttentionCall.CheckedType,
                _splitHierarchyAxis,
                _splitCount)
            .InheritMetaData(pagedAttentionCall);
    }
}
