// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Canonicalizes TIR index and region expressions with lexical loop constraints.
/// </summary>
public sealed class CanonicalizeTIRIndexExpressionsPass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var function in input.Functions.OfType<PrimFunction>())
        {
            var canonicalizer = new TIRIndexExpressionCanonicalizer();
            canonicalizer.Rewrite(function);
            if (canonicalizer.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after canonicalizing TIR index expressions in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private sealed class TIRIndexExpressionCanonicalizer
    {
        private readonly Dictionary<BaseExpr, ConstraintContext> _visited = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<ConstraintContext, DimensionCanonicalizer> _dimensionCanonicalizers = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<BaseExpr, Dictionary<int, Dimension>> _originalDimensionOperands = new(ReferenceEqualityComparer.Instance);

        public bool IsMutated { get; private set; }

        public void Rewrite(PrimFunction function)
        {
            RewriteExpression(function, ConstraintContext.Empty, function);
        }

        private BaseExpr RewriteExpression(BaseExpr expression, ConstraintContext context, PrimFunction root)
        {
            if (expression is Dimension dimension)
            {
                return GetDimensionCanonicalizer(context).Canonicalize(dimension);
            }

            if (expression is BaseFunction function && !ReferenceEquals(function, root))
            {
                return function;
            }

            if (_visited.TryGetValue(expression, out var previousContext))
            {
                ValidateSharedExpression(expression, previousContext, context, root);
                return expression;
            }

            _visited.Add(expression, context);
            switch (expression)
            {
                case For loop:
                    RewriteFor(loop, context, root);
                    break;
                case Let let:
                    RewriteLet(let, context, root);
                    break;
                case Call call:
                    RewriteCall(call, context, root);
                    break;
                default:
                    RewriteOperands(expression, context, root);
                    break;
            }

            return expression;
        }

        private void RewriteFor(For loop, ConstraintContext context, PrimFunction root)
        {
            ReplaceOperand(loop, 1, RewriteExpression(loop.Domain, context, root));

            var domain = loop.Domain;
            if (domain.Step.IsFixed && domain.Step.FixedValue <= 0)
            {
                throw new InvalidOperationException(
                    $"TIR loop {loop.LoopVar.Name} has non-positive step {domain.Step.FixedValue}; " +
                    "index canonicalization requires positive-step half-open loop semantics.");
            }

            var bodyContext = context.WithLoop(loop.LoopVar, domain);
            ReplaceOperand(loop, 2, RewriteExpression(loop.Body, bodyContext, root));
        }

        private void RewriteLet(Let let, ConstraintContext context, PrimFunction root)
        {
            ReplaceOperand(let, 1, RewriteExpression(let.Expression, context, root));
            var bodyContext = let.Var is DimVar variable && let.Expression is Dimension value
                ? context.WithSubstitution(variable, value)
                : context;
            ReplaceOperand(let, 2, RewriteExpression(let.Body, bodyContext, root));
        }

        private void RewriteCall(Call call, ConstraintContext context, PrimFunction root)
        {
            for (var i = 1; i < call.Operands.Length; i++)
            {
                ReplaceOperand(call, i, RewriteExpression(call.Operands[i], context, root));
            }
        }

        private void RewriteOperands(BaseExpr expression, ConstraintContext context, PrimFunction root)
        {
            for (var i = 0; i < expression.Operands.Length; i++)
            {
                ReplaceOperand(expression, i, RewriteExpression(expression.Operands[i], context, root));
            }
        }

        private void ValidateSharedExpression(
            BaseExpr expression,
            ConstraintContext previousContext,
            ConstraintContext currentContext,
            PrimFunction root)
        {
            if (ReferenceEquals(previousContext, currentContext))
            {
                return;
            }

            var visited = new Dictionary<BaseExpr, HashSet<(ConstraintContext Previous, ConstraintContext Current)>>(ReferenceEqualityComparer.Instance);
            ValidateSharedExpressionCore(expression, previousContext, currentContext, root, visited);
        }

        private void ValidateSharedExpressionCore(
            BaseExpr expression,
            ConstraintContext previousContext,
            ConstraintContext currentContext,
            PrimFunction root,
            Dictionary<BaseExpr, HashSet<(ConstraintContext Previous, ConstraintContext Current)>> visited)
        {
            if (expression is BaseFunction function && !ReferenceEquals(function, root))
            {
                return;
            }

            if (!visited.TryGetValue(expression, out var contexts))
            {
                contexts = new();
                visited.Add(expression, contexts);
            }

            if (!contexts.Add((previousContext, currentContext)))
            {
                return;
            }

            switch (expression)
            {
                case For loop:
                    {
                        ValidateSharedExpressionCore(loop.Domain, previousContext, currentContext, root, visited);
                        var previousBodyContext = previousContext.WithLoop(
                            loop.LoopVar,
                            CanonicalizeRange(loop.Domain, previousContext));
                        var currentBodyContext = currentContext.WithLoop(
                            loop.LoopVar,
                            CanonicalizeRange(loop.Domain, currentContext));
                        ValidateSharedExpressionCore(
                            loop.Body,
                            previousBodyContext,
                            currentBodyContext,
                            root,
                            visited);
                        break;
                    }

                case Let let:
                    {
                        ValidateOperand(let, 1, previousContext, currentContext, root, visited);
                        var previousBodyContext = previousContext;
                        var currentBodyContext = currentContext;
                        if (let.Var is DimVar variable && let.Expression is Dimension)
                        {
                            previousBodyContext = previousContext.WithSubstitution(
                                variable,
                                CanonicalizeOriginalDimension(let, 1, previousContext));
                            currentBodyContext = currentContext.WithSubstitution(
                                variable,
                                CanonicalizeOriginalDimension(let, 1, currentContext));
                        }

                        ValidateSharedExpressionCore(
                            let.Body,
                            previousBodyContext,
                            currentBodyContext,
                            root,
                            visited);
                        break;
                    }

                case Call call:
                    for (var i = 1; i < call.Operands.Length; i++)
                    {
                        ValidateOperand(call, i, previousContext, currentContext, root, visited);
                    }

                    break;
                default:
                    for (var i = 0; i < expression.Operands.Length; i++)
                    {
                        ValidateOperand(expression, i, previousContext, currentContext, root, visited);
                    }

                    break;
            }
        }

        private void ValidateOperand(
            BaseExpr owner,
            int index,
            ConstraintContext previousContext,
            ConstraintContext currentContext,
            PrimFunction root,
            Dictionary<BaseExpr, HashSet<(ConstraintContext Previous, ConstraintContext Current)>> visited)
        {
            if (owner.Operands[index] is Dimension)
            {
                var original = GetOriginalDimension(owner, index);
                var previous = GetDimensionCanonicalizer(previousContext).Canonicalize(original);
                var current = GetDimensionCanonicalizer(currentContext).Canonicalize(original);
                if (!previous.Equals(current))
                {
                    throw new InvalidOperationException(
                        $"Shared TIR expression {owner.GetType().Name} has context-dependent dimension {original}. " +
                        "A TIR node that depends on a lexical loop variable must be owned by that loop scope.");
                }

                return;
            }

            ValidateSharedExpressionCore(
                owner.Operands[index],
                previousContext,
                currentContext,
                root,
                visited);
        }

        private TIR.Range CanonicalizeRange(TIR.Range range, ConstraintContext context)
            => new(
                CanonicalizeOriginalDimension(range, 0, context),
                CanonicalizeOriginalDimension(range, 1, context),
                CanonicalizeOriginalDimension(range, 2, context));

        private Dimension CanonicalizeOriginalDimension(
            BaseExpr owner,
            int index,
            ConstraintContext context)
            => GetDimensionCanonicalizer(context).Canonicalize(GetOriginalDimension(owner, index));

        private Dimension GetOriginalDimension(BaseExpr owner, int index)
            => _originalDimensionOperands.TryGetValue(owner, out var operands)
                && operands.TryGetValue(index, out var original)
                ? original
                : (Dimension)owner.Operands[index];

        private DimensionCanonicalizer GetDimensionCanonicalizer(ConstraintContext context)
        {
            if (!_dimensionCanonicalizers.TryGetValue(context, out var canonicalizer))
            {
                canonicalizer = new DimensionCanonicalizer(context.Resolve, context.CanProveLessOrEqual);
                _dimensionCanonicalizers.Add(context, canonicalizer);
            }

            return canonicalizer;
        }

        private void ReplaceOperand(BaseExpr owner, int index, BaseExpr replacement)
        {
            if (owner.Operands[index] is Dimension dimension)
            {
                if (!_originalDimensionOperands.TryGetValue(owner, out var operands))
                {
                    operands = new();
                    _originalDimensionOperands.Add(owner, operands);
                }

                operands.TryAdd(index, dimension);
            }

            if (!ReferenceEquals(owner.Operands[index], replacement))
            {
                owner.ReplaceOperand(index, replacement);
                IsMutated = true;
            }
        }
    }

    private sealed class ConstraintContext
    {
        private readonly ConstraintContext? _parent;
        private readonly DimVar? _substitutionVariable;
        private readonly Dimension? _substitutionValue;
        private readonly LinearConstraint[] _constraints;

        private ConstraintContext(
            ConstraintContext? parent,
            DimVar? substitutionVariable,
            Dimension? substitutionValue,
            LinearConstraint[] constraints)
        {
            _parent = parent;
            _substitutionVariable = substitutionVariable;
            _substitutionValue = substitutionValue;
            _constraints = constraints;
        }

        public static ConstraintContext Empty { get; } = new(null, null, null, Array.Empty<LinearConstraint>());

        public ConstraintContext WithSubstitution(DimVar variable, Dimension value)
            => new(this, variable, value, Array.Empty<LinearConstraint>());

        public ConstraintContext WithLoop(DimVar variable, TIR.Range domain)
        {
            Dimension? substitution = null;
            if (domain.Start.IsFixed && domain.Stop.IsFixed && domain.Step.IsFixed)
            {
                var start = domain.Start.FixedValue;
                var stop = domain.Stop.FixedValue;
                var step = domain.Step.FixedValue;
                if (start < stop && step > 0 && new BigInteger(step) >= new BigInteger(stop) - start)
                {
                    substitution = domain.Start;
                }
            }

            var hasPositiveStep = domain.Step.IsFixed
                ? domain.Step.FixedValue > 0
                : domain.Step.Metadata.Range is { Min: > 0 };
            var constraints = new List<LinearConstraint>(2);
            if (hasPositiveStep)
            {
                if (LinearConstraint.TryCreate(domain.Start, variable, out var lowerBound))
                {
                    constraints.Add(lowerBound);
                }

                if (LinearConstraint.TryCreate(variable + 1, domain.Stop, out var upperBound))
                {
                    constraints.Add(upperBound);
                }
            }

            return new(this, variable, substitution, constraints.ToArray());
        }

        public Dimension? Resolve(DimVar variable)
        {
            for (var current = this; current is not null; current = current._parent)
            {
                if (ReferenceEquals(current._substitutionVariable, variable))
                {
                    return current._substitutionValue;
                }
            }

            return null;
        }

        public bool CanProveLessOrEqual(Dimension lhs, Dimension rhs)
        {
            if (!LinearExpression.TryCreateDifference(lhs, rhs, out var target))
            {
                return false;
            }

            if (target.IsConstant)
            {
                return target.Constant <= 0;
            }

            for (var current = this; current is not null; current = current._parent)
            {
                foreach (var constraint in current._constraints)
                {
                    if (constraint.Proves(target))
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }

    private sealed class LinearConstraint
    {
        private LinearConstraint(LinearExpression expression)
        {
            Expression = expression;
        }

        private LinearExpression Expression { get; }

        public static bool TryCreate(Dimension lhs, Dimension rhs, out LinearConstraint constraint)
        {
            if (!LinearExpression.TryCreateDifference(lhs, rhs, out var expression))
            {
                constraint = null!;
                return false;
            }

            constraint = new LinearConstraint(expression);
            return true;
        }

        public bool Proves(LinearExpression target)
        {
            if (!Expression.TryGetPositiveScaleTo(target, out var scale))
            {
                return false;
            }

            return new BigInteger(target.Constant) <= new BigInteger(Expression.Constant) * scale;
        }
    }

    private sealed class LinearExpression
    {
        private readonly Dictionary<Dimension, long> _coefficients = new();

        private LinearExpression()
        {
        }

        public long Constant { get; private set; }

        public bool IsConstant => _coefficients.Count == 0;

        public static bool TryCreateDifference(
            Dimension lhs,
            Dimension rhs,
            out LinearExpression result)
        {
            result = new LinearExpression();
            try
            {
                result.Add(lhs, 1);
                result.Add(rhs, -1);
                return true;
            }
            catch (OverflowException)
            {
                result = null!;
                return false;
            }
        }

        public bool TryGetPositiveScaleTo(LinearExpression target, out long scale)
        {
            scale = 0;
            if (_coefficients.Count == 0 || _coefficients.Count != target._coefficients.Count)
            {
                return false;
            }

            foreach (var (atom, coefficient) in _coefficients)
            {
                if (!target._coefficients.TryGetValue(atom, out var targetCoefficient)
                    || coefficient == 0)
                {
                    return false;
                }

                if (coefficient == -1 && targetCoefficient == long.MinValue)
                {
                    return false;
                }

                if (targetCoefficient % coefficient != 0)
                {
                    return false;
                }

                var candidateScale = targetCoefficient / coefficient;
                if (candidateScale <= 0 || (scale != 0 && scale != candidateScale))
                {
                    return false;
                }

                scale = candidateScale;
            }

            return scale > 0;
        }

        private static bool TryGetLinearFactor(
            DimProduct product,
            out long factor,
            out Dimension? operand)
        {
            factor = product.Scale;
            operand = null;
            foreach (var candidate in product.Operands)
            {
                if (candidate.IsFixed)
                {
                    factor = checked(factor * candidate.FixedValue);
                }
                else if (operand is null)
                {
                    operand = candidate;
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

        private void Add(Dimension expression, long scale)
        {
            switch (expression)
            {
                case DimConst constant:
                    Constant = checked(Constant + (constant.Value * scale));
                    return;
                case DimSum sum:
                    Constant = checked(Constant + (sum.Bias * scale));
                    foreach (var operand in sum.Operands)
                    {
                        Add(operand, scale);
                    }

                    return;
                case DimProduct product when TryGetLinearFactor(product, out var factor, out var operand):
                    if (operand is null)
                    {
                        Constant = checked(Constant + (factor * scale));
                    }
                    else
                    {
                        Add(operand, checked(scale * factor));
                    }

                    return;
                default:
                    AddAtom(expression, scale);
                    return;
            }
        }

        private void AddAtom(Dimension atom, long coefficient)
        {
            _coefficients.TryGetValue(atom, out var current);
            var updated = checked(current + coefficient);
            if (updated == 0)
            {
                _coefficients.Remove(atom);
            }
            else
            {
                _coefficients[atom] = updated;
            }
        }
    }
}
