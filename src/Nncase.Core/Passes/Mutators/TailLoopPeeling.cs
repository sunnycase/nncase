// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// Splits tiled serial and reduction loops into a statically full path and a
/// clipped tail path. Nested tails are emitted as mutually exclusive slabs:
/// only the full path is recursively peeled, so code size grows linearly with
/// the tiled rank instead of producing every full/tail Cartesian combination.
/// </summary>
public sealed class TailLoopPeeling : ExprRewriter
{
    protected internal override BaseExpr VisitFor(For expr, Unit context)
    {
        if (expr.Partition != LoopPartition.Unpartitioned)
        {
            return expr;
        }

        var domain = (TIR.Range)Visit(expr.Domain, context);
        if (!CanPeel(expr, domain))
        {
            Visit(expr.LoopVar, context);
            Visit(expr.Body, context);
            return VisitLeafFor(expr, context);
        }

        var fullEnd = domain.Start + (((domain.Stop - domain.Start) / domain.Step) * domain.Step);
        var fields = new List<Expr>(2);
        var tailLoopVar = expr.LoopVar.With(name: $"{expr.LoopVar.Name}_tail");
        var tailBody = new LoopSlabCloner(expr.Body, expr.LoopVar, tailLoopVar).Clone(expr.Body, default);
        if (ExprCollector.Collect(tailBody).Any(node => ReferenceEquals(node, expr.LoopVar)))
        {
            throw new InvalidOperationException(
                $"Cannot peel loop {expr.LoopVar.Name}: its tail slab retains a reference to the full-loop binder.");
        }

        // Recursing only through the full path yields one full interior and
        // one boundary slab per tiled axis. The tail path keeps all inner
        // loops clipped, which is both complete and non-overlapping.
        var fullBody = (Sequential)Visit(expr.Body, context);
        if (!IsStaticallyEmpty(domain.Start, fullEnd))
        {
            fields.Add(new For(
                expr.LoopVar,
                new TIR.Range(domain.Start, fullEnd, domain.Step),
                expr.Mode,
                fullBody,
                LoopPartition.Full));
        }

        fields.Add(new For(
            tailLoopVar,
            new TIR.Range(fullEnd, domain.Stop, domain.Step),
            expr.Mode,
            tailBody,
            LoopPartition.Tail));

        return new Sequential(fields.ToArray());
    }

    private static bool CanPeel(For loop, TIR.Range domain)
    {
        if (loop.Mode is not (LoopMode.Serial or LoopMode.Reduction) ||
            !domain.Step.IsFixed ||
            domain.Step.FixedValue <= 1)
        {
            return false;
        }

        if (domain.Start.IsFixed && domain.Stop.IsFixed)
        {
            var extent = domain.Stop.FixedValue - domain.Start.FixedValue;
            return extent > 0 && (extent % domain.Step.FixedValue) != 0;
        }

        return true;
    }

    private static bool IsStaticallyEmpty(Dimension start, Dimension stop)
        => start.IsFixed && stop.IsFixed && start.FixedValue == stop.FixedValue;

    private sealed class LoopSlabCloner : ExprCloner<Unit>
    {
        private readonly HashSet<BaseExpr> _localBinders = new(ReferenceEqualityComparer.Instance);
        private readonly DimVar _source;
        private readonly DimVar _replacement;

        public LoopSlabCloner(Sequential body, DimVar source, DimVar replacement)
        {
            _source = source;
            _replacement = replacement;

            foreach (var node in ExprCollector.Collect(body))
            {
                switch (node)
                {
                    case Let let:
                        _localBinders.Add((BaseExpr)let.Var);
                        break;
                    case For loop:
                        _localBinders.Add(loop.LoopVar);
                        break;
                }
            }
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (ReferenceEquals(expr, _source))
            {
                return _replacement;
            }

            // Free variables belong to the surrounding function or loop and
            // retain their identity. Binders owned by the duplicated slab are
            // cloned, with ExprCloner's memo remapping all of their uses.
            if (expr is IVar && !_localBinders.Contains(expr))
            {
                return expr;
            }

            return base.DispatchVisit(expr, context);
        }

        protected override BaseExpr VisitLeafPhysicalBuffer(PhysicalBuffer expr, Unit context)
        {
            if (ExprCollector.Collect(expr).Any(node => ReferenceEquals(node, _source)))
            {
                throw new InvalidOperationException(
                    $"Cannot peel loop {_source.Name}: a PhysicalBuffer descriptor depends on its loop variable.");
            }

            // Peeling duplicates logical execution paths, not storage. Both
            // slabs must retain the same physical allocation while their
            // MemSpan and logical Buffer views are cloned independently.
            return expr;
        }
    }
}
