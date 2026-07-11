// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Affine;

[Flags]
public enum GridAccessMode
{
    Read = 1,
    Write = 2,
    ReadWrite = Read | Write,
}

public enum GridBindingMode
{
    Subview,
    Root,
}

public enum GridDomainMode
{
    Constraint,
    Footprint,
}

/// <summary>
/// Describes one logical operand of a <see cref="Grid"/> and how it is bound
/// when the grid is tiled.
/// </summary>
public sealed class GridAccess : Expr
{
    public GridAccess(
        Expr value,
        Expr buffer,
        Var parameter,
        BaseExpr region,
        GridAccessMode accessMode,
        GridBindingMode bindingMode,
        GridDomainMode domainMode)
        : base([value, buffer, parameter, region])
    {
        if (accessMode is not (GridAccessMode.Read or GridAccessMode.Write or GridAccessMode.ReadWrite))
        {
            throw new ArgumentOutOfRangeException(nameof(accessMode));
        }

        if (region is not Nncase.IR.Affine.AffineMap and not None)
        {
            throw new ArgumentException("Grid access region must be an AffineMap or None for an opaque resource.", nameof(region));
        }

        if (bindingMode == GridBindingMode.Subview && region is not Nncase.IR.Affine.AffineMap)
        {
            throw new ArgumentException("A subview-bound grid access requires an affine region.", nameof(region));
        }

        AccessMode = accessMode;
        BindingMode = bindingMode;
        DomainMode = domainMode;
    }

    public Expr Value => (Expr)Operands[0];

    public Expr Buffer => (Expr)Operands[1];

    public Var Parameter => (Var)Operands[2];

    /// <summary>
    /// Gets the affine footprint, or <see cref="None.Default"/> for an opaque resource.
    /// </summary>
    public BaseExpr Region => Operands[3];

    public GridAccessMode AccessMode { get; }

    public GridBindingMode BindingMode { get; }

    public GridDomainMode DomainMode { get; }

    public bool IsRead => AccessMode.HasFlag(GridAccessMode.Read);

    public bool IsWrite => AccessMode.HasFlag(GridAccessMode.Write);

    public bool IsAffine => Region is Nncase.IR.Affine.AffineMap;

    public AffineMap AffineMap => Region as Nncase.IR.Affine.AffineMap
        ?? throw new InvalidOperationException("Opaque grid access does not have an affine map.");

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitGridAccess(this, context);

    public GridAccess With(
        Expr? value = null,
        Expr? buffer = null,
        Var? parameter = null,
        BaseExpr? region = null,
        GridAccessMode? accessMode = null,
        GridBindingMode? bindingMode = null,
        GridDomainMode? domainMode = null)
        => new(
            value ?? Value,
            buffer ?? Buffer,
            parameter ?? Parameter,
            region ?? Region,
            accessMode ?? AccessMode,
            bindingMode ?? BindingMode,
            domainMode ?? DomainMode);
}
