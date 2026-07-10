// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// PrimFunction expression.
/// </summary>
public sealed class PrimFunction : BaseFunction
{
    private static int _globalFuncIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="parameters">Arguments.</param>
    /// <param name="body">Body.</param>
    /// <param name="results">Ordered logical results.</param>
    public PrimFunction(string name, string moduleKind, Sequential body, Return results, ReadOnlySpan<IVar> parameters)
        : base(name, moduleKind, new BaseExpr[] { body, results }.Concat(SpanUtility.UnsafeCast<IVar, BaseExpr>(parameters).ToArray()).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class with no logical results.
    /// </summary>
    public PrimFunction(string name, string moduleKind, Sequential body, ReadOnlySpan<IVar> parameters)
        : this(name, moduleKind, body, new Return(Array.Empty<Expr>()), parameters)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="parameters">Arguments.</param>
    /// <param name="body">Body.</param>
    /// <param name="results">Ordered logical results.</param>
    public PrimFunction(string moduleKind, Sequential body, Return results, ReadOnlySpan<IVar> parameters)
        : this($"primfunc_{_globalFuncIndex++}", moduleKind, body, results, parameters)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class with no logical results.
    /// </summary>
    public PrimFunction(string moduleKind, Sequential body, ReadOnlySpan<IVar> parameters)
        : this(moduleKind, body, new Return(Array.Empty<Expr>()), parameters)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    public PrimFunction(string moduleKind, Sequential body, Return results, params IVar[] parameters)
        : this($"primfunc_{_globalFuncIndex++}", moduleKind, body, results, new(parameters))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class with no logical results.
    /// </summary>
    public PrimFunction(string moduleKind, Sequential body, params IVar[] parameters)
        : this(moduleKind, body, new Return(Array.Empty<Expr>()), parameters)
    {
    }

    /// <summary>
    /// Gets body.
    /// </summary>
    public Sequential Body => (Sequential)Operands[0];

    /// <summary>
    /// Gets the ordered logical results.
    /// </summary>
    public Return Results => (Return)Operands[1];

    public ReadOnlySpan<IVar> Parameters => SpanUtility.UnsafeCast<BaseExpr, IVar>(Operands.Slice(2));

    public override IEnumerable<IRType> ParameterTypes => Parameters.AsValueEnumerable().Select(x => ((BaseExpr)x).CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPrimFunction(this, context);

    public override BaseFunction With(string? name = null, string? moduleKind = null)
    {
        return new PrimFunction(name ?? Name, moduleKind ?? ModuleKind, Body, Results, Parameters);
    }

    public PrimFunction With(string? name = null, string? moduleKind = null, Sequential? body = null, Return? results = null, IVar[]? parameters = null, Schedule.SchedFunctionResult? sched = null)
        => new PrimFunction(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, results ?? Results, parameters ?? Parameters)
        {
            // note maybe add SchedResult into ctor.
            SchedResult = sched ?? SchedResult,
        };
}
