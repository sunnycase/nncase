// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldPackTranspose : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsPack(
            target_name: "pack",
            "call",
            _ => true,
            IsTranspose(
                target_name: "transpose",
                "transposeCall",
                IsWildcard("input") with { TypePattern = HasRankedShape() },
                IsFixedShape("perm")));

    public Expr? GetReplace(Pack pack, Transpose transpose, Expr input, int[] perm)
    {
        if (pack.Axes.Any(axis => axis < 0 || axis >= perm.Length))
        {
            return null;
        }

        var inputAxes = pack.Axes.Select(axis => perm[axis]).ToArray();
        return IR.F.Tensors.Transpose(
            IR.F.Tensors.Pack(input, pack.Lanes.ToArray(), inputAxes),
            perm);
    }
}

[RuleGenerator]
public sealed partial class FoldPackReshape : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsPack(
            target_name: "pack",
            "call",
            _ => true,
            IsReshape(
                target_name: "reshape",
                "reshapeCall",
                IsWildcard("input") with { TypePattern = HasRankedShape() },
                IsRankedShape("newShape")));

    public Expr? GetReplace(Pack pack, Expr input, RankedShape newShape)
    {
        var inputShape = input.CheckedShape;
        var maxInputShape = CompilerServices.GetMaxShape(inputShape);
        var maxNewShape = CompilerServices.GetMaxShape(newShape);
        if (!IRUtility.TryGetShapeMapMatrix(maxInputShape, maxNewShape, out var mat))
        {
            return null;
        }

        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsCompleteDict(mat);
        var inputAxes = new List<int>();
        var inputLanes = new List<int>();
        var rewrittenNewShape = newShape.ToArray();
        for (int i = 0; i < pack.Axes.Count; i++)
        {
            var axis = pack.Axes[i];
            var lanes = pack.Lanes[i];

            foreach ((var inAxis, var newAxes) in forwardDict)
            {
                var vectorizeAxisIndex = newAxes.IndexOf(axis);
                if (vectorizeAxisIndex < 0)
                {
                    continue;
                }

                if (vectorizeAxisIndex == newAxes.Count - 1 ||
                    newAxes.Skip(vectorizeAxisIndex + 1).All(x => x == 1))
                {
                    inputAxes.Add(inAxis);
                    inputLanes.Add(lanes);
                    rewrittenNewShape[axis] /= lanes;
                }
                else
                {
                    return null;
                }
            }

            if (backwardDict.TryGetValue(axis, out var inAxes))
            {
                var found = false;
                var reversedInAxes = Enumerable.Reverse(inAxes).ToArray();
                foreach (var inAxis in reversedInAxes)
                {
                    if (inputAxes.Contains(inAxis))
                    {
                        found = true;
                        continue;
                    }

                    if (inputShape[inAxis] != 1)
                    {
                        if (!Dimension.TryDivExactly(inputShape[inAxis], lanes, out var newDim))
                        {
                            return null;
                        }

                        inputAxes.Add(inAxis);
                        inputLanes.Add(lanes);
                        rewrittenNewShape[axis] = newDim * reversedInAxes
                            .Skip(reversedInAxes.IndexOf(inAxis) + 1)
                            .Select(a => inputShape[a])
                            .Aggregate((a, b) => a * b);
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    return null;
                }
            }
        }

        return IR.F.Tensors.Reshape(
            IR.F.Tensors.Pack(input, inputLanes.ToArray(), inputAxes.ToArray()),
            new RankedShape(rewrittenNewShape));
    }
}
