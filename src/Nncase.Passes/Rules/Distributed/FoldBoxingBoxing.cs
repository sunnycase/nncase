// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Distributed;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules;

[RuleGenerator]
public partial class FoldBoxingBoxing : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBoxing(
        target_name: "outer",
        _ => true,
        IsBoxing(
            target_name: "inner",
            _ => true,
            IsWildcard("input")));

    private Expr? GetReplace(Boxing outer, Boxing inner, Expr input)
    {
        if (Equals(outer.NewType, input.CheckedType))
        {
            return input;
        }

        if (outer.NewType is DistributedType outerDistributedType
            && inner.NewType is TensorType innerTensorType
            && input.CheckedType is DistributedType inputDistributedType
            && Equals(innerTensorType, inputDistributedType.TensorType)
            && Equals(outerDistributedType.TensorType, inputDistributedType.TensorType))
        {
            return IR.F.Distributed.Boxing(input, outerDistributedType);
        }

        return null;
    }
}
