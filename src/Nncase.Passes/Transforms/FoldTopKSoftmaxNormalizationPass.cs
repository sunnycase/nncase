// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Moves a full-axis Softmax after TopK when callers renormalize the selected values.
/// </summary>
public sealed class FoldTopKSoftmaxNormalizationPass : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function)
        {
            return Task.FromResult(input);
        }

        var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        foreach (var topKCall in ExprCollector.Collect(function.Body)
                     .OfType<Call>()
                     .Where(call => call.Target is TopK))
        {
            TryAddReplacements(topKCall, replacements);
        }

        if (replacements.Count == 0)
        {
            return Task.FromResult(input);
        }

        var rewritten = (BaseFunction)new ReplacementRewriter(replacements).Rewrite(function);
        if (!CompilerServices.InferenceType(rewritten))
        {
            throw new InvalidOperationException(
                $"TopK-Softmax normalization folding failed to infer function {function.Name}.");
        }

        if (rewritten.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException(
                $"TopK-Softmax normalization folding produced an invalid function {function.Name}: {invalid}.");
        }

        return Task.FromResult(rewritten);
    }

    private static void TryAddReplacements(
        Call topKCall,
        IDictionary<BaseExpr, BaseExpr> replacements)
    {
        if (topKCall[TopK.X] is not Call { Target: Softmax } softmaxCall ||
            !TryGetNormalizedAxis(softmaxCall[Softmax.Axis], softmaxCall.CheckedShape.Rank, out var softmaxAxis) ||
            !TryGetNormalizedAxis(topKCall[TopK.Axis], softmaxCall.CheckedShape.Rank, out var topKAxis) ||
            softmaxAxis != topKAxis ||
            !TryGetScalarBoolean(topKCall[TopK.Largest], out var largest) ||
            !largest)
        {
            return;
        }

        var projections = topKCall.Users.OfType<Call>().ToArray();
        if (projections.Length == 0 ||
            projections.Length != topKCall.Users.Count() ||
            projections.Any(call => call.Target is not GetItem))
        {
            return;
        }

        var valueProjections = new List<Call>();
        var indexProjections = new List<Call>();
        foreach (var projection in projections)
        {
            if (!TryGetScalarInt64(projection[GetItem.Index], out var index))
            {
                return;
            }

            switch (index)
            {
                case 0:
                    valueProjections.Add(projection);
                    break;
                case 1:
                    indexProjections.Add(projection);
                    break;
                default:
                    return;
            }
        }

        if (valueProjections.Count == 0)
        {
            return;
        }

        var normalizations = new Dictionary<Call, Call[]>(ReferenceEqualityComparer.Instance);
        foreach (var valueProjection in valueProjections)
        {
            if (!TryGetNormalizations(valueProjection, topKAxis, out var divisions))
            {
                return;
            }

            normalizations.Add(valueProjection, divisions);
        }

        var rawTopK = IR.F.Tensors.TopK(
                (Expr)softmaxCall[Softmax.Input],
                (Expr)topKCall[TopK.K],
                (Expr)topKCall[TopK.Axis],
                (Expr)topKCall[TopK.Largest],
                (Expr)topKCall[TopK.Sorted])
            .InheritMetaData(topKCall);
        InferOrThrow(rawTopK, "TopK over pre-Softmax logits");

        foreach (var (valueProjection, divisions) in normalizations)
        {
            var rawValues = IR.F.Tensors.GetItem(rawTopK, 0).InheritMetaData(valueProjection);
            InferOrThrow(rawValues, "TopK value projection over pre-Softmax logits");
            var selectedSoftmax = IR.F.NN.Softmax(rawValues, topKAxis)
                .InheritMetaData(divisions[0]);
            InferOrThrow(selectedSoftmax, "Softmax over selected TopK logits");
            foreach (var division in divisions)
            {
                if (selectedSoftmax.CheckedType != division.CheckedType)
                {
                    throw new InvalidOperationException(
                        "TopK-Softmax normalization folding changed the normalized value type: " +
                        $"expected {division.CheckedType}, got {selectedSoftmax.CheckedType}.");
                }

                replacements.Add(division, selectedSoftmax);
            }
        }

        foreach (var indexProjection in indexProjections)
        {
            var rawIndices = IR.F.Tensors.GetItem(rawTopK, 1).InheritMetaData(indexProjection);
            InferOrThrow(rawIndices, "TopK index projection over pre-Softmax logits");
            if (rawIndices.CheckedType != indexProjection.CheckedType)
            {
                throw new InvalidOperationException(
                    "TopK-Softmax normalization folding changed the index type: " +
                    $"expected {indexProjection.CheckedType}, got {rawIndices.CheckedType}.");
            }

            replacements.Add(indexProjection, rawIndices);
        }
    }

    private static bool TryGetNormalizations(Call values, int axis, out Call[] divisions)
    {
        divisions = Array.Empty<Call>();
        var users = values.Users.OfType<Call>().ToArray();
        if (users.Length == 0 || users.Length != values.Users.Count())
        {
            return false;
        }

        var reductions = new HashSet<Call>(
            users.Where(user => IsNormalizationReduction(user, values, axis)),
            ReferenceEqualityComparer.Instance);
        if (reductions.Count == 0)
        {
            return false;
        }

        var matchedDivisions = users
            .Where(user => user.Target is Binary { BinaryOp: BinaryOp.Div } &&
                           ReferenceEquals(user[Binary.Lhs], values) &&
                           user[Binary.Rhs] is Call reduction &&
                           reductions.Contains(reduction))
            .ToArray();
        if (matchedDivisions.Length == 0 ||
            users.Any(user => !reductions.Contains(user) && !matchedDivisions.Contains(user)))
        {
            return false;
        }

        var divisionSet = new HashSet<Call>(
            matchedDivisions,
            ReferenceEqualityComparer.Instance);
        if (reductions.Any(reduction =>
                !reduction.Users.Any() ||
                reduction.Users.Any(user => user is not Call call || !divisionSet.Contains(call))))
        {
            return false;
        }

        divisions = matchedDivisions;
        return true;
    }

    private static bool IsNormalizationReduction(Call call, Call values, int axis)
    {
        if (call.Target is not Reduce { ReduceOp: ReduceOp.Sum } ||
            !ReferenceEquals(call[Reduce.Input], values) ||
            !TryGetScalarBoolean(call[Reduce.KeepDims], out var keepDims) ||
            !keepDims ||
            !TryGetScalarZero(call[Reduce.InitValue]) ||
            call[Reduce.Axes] is not RankedShape axes ||
            axes.Count != 1 ||
            axes[0].Kind != DimensionKind.Fixed)
        {
            return false;
        }

        var reduceAxis = checked((int)axes[0].FixedValue);
        if (reduceAxis < 0)
        {
            reduceAxis += values.CheckedShape.Rank;
        }

        return reduceAxis == axis;
    }

    private static bool TryGetNormalizedAxis(BaseExpr expression, int rank, out int axis)
    {
        axis = default;
        if (!TryGetScalarInt64(expression, out var value) ||
            value < -rank || value >= rank)
        {
            return false;
        }

        axis = checked((int)(value < 0 ? value + rank : value));
        return true;
    }

    private static bool TryGetScalarInt64(BaseExpr expression, out long value)
    {
        switch (expression)
        {
            case DimConst dimension:
                value = dimension.Value;
                return true;
            case TensorConst tensor when tensor.Value.Length == 1:
                value = tensor.Value.ToScalar<long>();
                return true;
            default:
                value = default;
                return false;
        }
    }

    private static bool TryGetScalarBoolean(BaseExpr expression, out bool value)
    {
        if (expression is TensorConst tensor && tensor.Value.Length == 1)
        {
            value = tensor.Value.ToScalar<bool>();
            return true;
        }

        value = default;
        return false;
    }

    private static bool TryGetScalarZero(BaseExpr expression) =>
        expression is TensorConst tensor &&
        tensor.Value.Length == 1 &&
        tensor.Value.ToScalar<double>() == 0.0;

    private static void InferOrThrow(BaseExpr expression, string context)
    {
        if (!CompilerServices.InferenceType(expression))
        {
            throw new InvalidOperationException($"Failed to infer {context}.");
        }

        if (expression.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException($"Failed to infer {context}: {invalid}.");
        }
    }

    private sealed class ReplacementRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;

        public ReplacementRewriter(IReadOnlyDictionary<BaseExpr, BaseExpr> replacements)
        {
            _replacements = replacements;
        }

        protected override BaseExpr DefaultRewriteLeaf(BaseExpr expr) =>
            _replacements.TryGetValue(expr, out var replacement) ? replacement : expr;
    }
}
