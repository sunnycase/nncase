// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Schedule.Bufferize;

public sealed record LifetimeCollectionResult(TIR.Buffer[] Buffers, IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> Lifetimes);

public sealed class LifetimeCollector
{
    public LifetimeCollectionResult Collect(Expr expr)
    {
        var buffers = new HashSet<TIR.Buffer>(ReferenceEqualityComparer.Instance);
        var lifetimes = new Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)>(ReferenceEqualityComparer.Instance);
        var bindings = LetBindingCollector.Collect(expr);
        var resolver = new BufferResourceResolver(bindings);
        new BufferCollector(buffers, lifetimes, resolver).Visit(expr);
        new LifetimeRecoder(lifetimes, resolver).Visit(expr);
        ValidateZeroRefCounts(lifetimes);
        return new(buffers.ToArray(), lifetimes.ToDictionary(x => x.Key, x => x.Value.Lifetime, (IEqualityComparer<TIR.PhysicalBuffer>)ReferenceEqualityComparer.Instance));
    }

    private static void ValidateZeroRefCounts(Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> lifetimes)
    {
        foreach (var (_, refCount) in lifetimes.Values)
        {
            if (refCount != 0)
            {
                throw new InvalidOperationException($"Non-zero ref count found");
            }
        }
    }

    private sealed class LetBindingCollector : ExprWalker
    {
        private readonly Dictionary<BaseExpr, BaseExpr> _bindings = new(ReferenceEqualityComparer.Instance);

        public static IReadOnlyDictionary<BaseExpr, BaseExpr> Collect(BaseExpr expression)
        {
            var collector = new LetBindingCollector();
            collector.Visit(expression);
            return collector._bindings;
        }

        protected override Unit VisitLeafLet(Let expr)
        {
            if (!_bindings.TryAdd((BaseExpr)expr.Var, expr.Expression))
            {
                throw new InvalidOperationException($"TIR variable {expr.Var.Name} has more than one Let binding.");
            }

            return default;
        }
    }

    private sealed class BufferResourceResolver
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _bindings;

        public BufferResourceResolver(IReadOnlyDictionary<BaseExpr, BaseExpr> bindings)
        {
            _bindings = bindings;
        }

        public IEnumerable<(TIR.Buffer Buffer, TIR.PhysicalBuffer PhysicalBuffer)> Resolve(BaseExpr expression)
        {
            var resources = new List<(TIR.Buffer Buffer, TIR.PhysicalBuffer PhysicalBuffer)>();
            var active = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            ResolveCore(expression, resources, active);
            return resources;
        }

        private void ResolveCore(
            BaseExpr expression,
            List<(TIR.Buffer Buffer, TIR.PhysicalBuffer PhysicalBuffer)> resources,
            HashSet<BaseExpr> active)
        {
            if (!active.Add(expression))
            {
                throw new InvalidOperationException("Cyclic TIR buffer-view alias detected while collecting buffer lifetimes.");
            }

            switch (expression)
            {
                case TIR.Buffer buffer:
                    resources.Add((buffer, buffer.MemSpan.Buffer));
                    break;
                case IR.Tuple tuple:
                    foreach (var field in tuple.Fields)
                    {
                        ResolveCore(field, resources, active);
                    }

                    break;
                case IVar when _bindings.TryGetValue(expression, out var boundExpression):
                    ResolveCore(boundExpression, resources, active);
                    break;
                case Call { Target: IR.Buffers.BufferSubview or IR.Buffers.AllocateBufferView } view when view.Arguments.Length > 0:
                    ResolveCore(view.Arguments[0], resources, active);
                    break;
            }

            active.Remove(expression);
        }
    }

    private sealed class BufferCollector : ExprWalker
    {
        private readonly HashSet<TIR.Buffer> _buffers;
        private readonly Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> _lifetimes;
        private readonly BufferResourceResolver _resolver;

        public BufferCollector(
            HashSet<TIR.Buffer> buffers,
            Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> lifetimes,
            BufferResourceResolver resolver)
        {
            _buffers = buffers;
            _lifetimes = lifetimes;
            _resolver = resolver;
        }

        protected override Unit VisitLeafPhysicalBuffer(TIR.PhysicalBuffer expr)
        {
            if (expr.Start is None or Call { Target: IR.Buffers.AddressOf })
            {
                var bufferSize = CompilerServices.GetMaxShape([expr.Size])[0];
                var lifetime = new BufferLifetime(expr) { Memory = new(0, bufferSize) };
                _lifetimes.Add(expr, (lifetime, 0));
            }

            return default;
        }

        protected override Unit VisitLeafCall(Call expr)
        {
            foreach (var arg in expr.Arguments)
            {
                AcquireBuffer(arg);
            }

            return default;
        }

        private void AcquireBuffer(BaseExpr expr)
        {
            foreach ((var buffer, var physicalBuffer) in _resolver.Resolve(expr))
            {
                _buffers.Add(buffer);
                if (physicalBuffer.Start is None or Call { Target: IR.Buffers.AddressOf })
                {
                    ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, physicalBuffer);
                    record.RefCount++;
                }
            }
        }
    }

    private sealed class LifetimeRecoder : ExprWalker
    {
        private readonly Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> _lifetimes;
        private readonly BufferResourceResolver _resolver;
        private int _currentAge;

        public LifetimeRecoder(
            Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> lifetimes,
            BufferResourceResolver resolver)
        {
            _lifetimes = lifetimes;
            _resolver = resolver;
        }

        protected override Unit VisitLeafPhysicalBuffer(TIR.PhysicalBuffer expr)
        {
            if (expr.Start is None or Call { Target: IR.Buffers.AddressOf })
            {
                ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, expr);
                record.Lifetime.Time.Start = _currentAge;
            }

            return default;
        }

        protected override Unit VisitLeafCall(Call expr)
        {
            _currentAge++;
            foreach (var arg in expr.Arguments)
            {
                ReleaseBuffer(arg);
            }

            return default;
        }

        private void ReleaseBuffer(BaseExpr expr)
        {
            foreach ((_, var physicalBuffer) in _resolver.Resolve(expr))
            {
                if (physicalBuffer.Start is None or Call { Target: IR.Buffers.AddressOf })
                {
                    ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, physicalBuffer);
                    if (--record.RefCount == 0)
                    {
                        record.Lifetime.Time.Stop = _currentAge;
                    }
                }
            }
        }
    }
}
