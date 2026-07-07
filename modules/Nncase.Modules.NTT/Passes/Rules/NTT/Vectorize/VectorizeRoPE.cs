// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class VectorizeRoPEPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "vectorize",
            "caller",
            _ => true,
            IsRoPE(
                "rope",
                "callee",
                _ => true,
                IsWildcard("input"),
                IsWildcard("cos"),
                IsWildcard("sin")));

    private Expr? GetReplace(Call caller, Pack vectorize, Expr cos, Expr sin, Call callee, Expr input)
    {
        var outputShape = (RankedShape)caller.CheckedShape;
        var outputRank = outputShape.Rank;
        if (vectorize.Axes.Any(axis => axis != outputRank - 1))
        {
            return null;
        }

        if (vectorize.Axes.Contains(outputRank - 1))
        {
            var lastDim = outputShape[^1];
            if (lastDim is not DimConst dc
                || dc.Value % 2 != 0)
            {
                // The last dimension must be a constant and even
                return null;
            }
        }

        if (vectorize.Axes.Count != 1 || vectorize.Lanes.Count != 1)
        {
            return null;
        }

        var lane = vectorize.Lanes[0];
        var rotaryAxis = outputRank - 1;
        var cosAxis = GetArgumentAxis(outputRank, cos.CheckedShape, rotaryAxis);
        var sinAxis = GetArgumentAxis(outputRank, sin.CheckedShape, rotaryAxis);
        if (!CanPackRoPESinCos(cos.CheckedShape, cosAxis, checked(2 * lane))
            || !CanPackRoPESinCos(sin.CheckedShape, sinAxis, checked(2 * lane)))
        {
            return null;
        }

        var inputT = caller.WithArguments([(Pack.Input, input)]);
        var cosT = IR.F.Tensors.Pack(CastToFloat32IfNeeded(cos), [2, lane], [cosAxis, cosAxis]);
        var sinT = IR.F.Tensors.Pack(CastToFloat32IfNeeded(sin), [2, lane], [sinAxis, sinAxis]);
        return IR.F.NTT.VectorizedRoPE(inputT, cosT, sinT);
    }

    private static Expr CastToFloat32IfNeeded(Expr expr)
    {
        return expr.CheckedDataType == DataTypes.Float32
            ? expr
            : IR.F.Tensors.Cast(expr, DataTypes.Float32);
    }

    private static int GetArgumentAxis(int outputRank, Shape argumentShape, int outputAxis)
    {
        return argumentShape is RankedShape rankedShape ? outputAxis - (outputRank - rankedShape.Rank) : -1;
    }

    private static bool CanPackRoPESinCos(Shape shape, int axis, int laneProduct)
    {
        if (shape is not RankedShape rankedShape || axis < 0 || axis >= rankedShape.Rank)
        {
            return false;
        }

        return rankedShape[axis] is DimConst dim && dim.Value % laneProduct == 0;
    }
}
