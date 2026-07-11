// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;

using Nncase.TIR;

namespace Nncase.IR.Affine.Builders;

/// <summary>
/// builfer the grid.
/// </summary>
public interface IGridBuilder : IExprBuilder<Grid>
{
    /// <summary>
    /// else grid.
    /// </summary>
    /// <param name="exprOrBuilders"> statements. </param>
    /// <returns> GridBuilder. </returns>
    IGridBuilder Body(params object[] exprOrBuilders);

    IGridBuilder Read(Expr argument, AffineMap accessMap, out Var parameter);

    IGridBuilder Read(Expr argument, AffineMap accessMap, GridDomainMode domainMode, out Var parameter);

    IGridBuilder Write(Expr buffer, AffineMap accessMap, out Var parameter);

    IGridBuilder ReadRoot(Expr argument, AffineMap accessMap, out Var parameter);

    IGridBuilder ReadRoot(Expr argument, AffineMap accessMap, GridDomainMode domainMode, out Var parameter);

    IGridBuilder WriteRoot(Expr buffer, AffineMap accessMap, out Var parameter);

    IGridBuilder WriteRoot(Expr buffer, AffineMap accessMap, GridDomainMode domainMode, out Var parameter);

    IGridBuilder ReadWriteRoot(Expr argument, BaseExpr region, out Var parameter);

    IGridBuilder Domain(int dims, out Var parameter);

    IGridBuilder Domain(ReadOnlySpan<GridTileAxisPolicy> tileAxisPolicies, out Var parameter);
}

internal class GridBuilder : IGridBuilder
{
    private readonly List<GridAccess> _accesses = new();
    private readonly List<object> _body = new();
    private Var? _domainParameter;
    private GridTileAxisPolicy[]? _tileAxisPolicies;

    public GridBuilder()
    {
    }

    public IGridBuilder Body(params object[] exprOrBuilders)
    {
        _body.AddRange(exprOrBuilders);
        return this;
    }

    public Grid Build()
    {
        return new Grid(
            _domainParameter ?? throw new InvalidOperationException("domain dims is not set."),
            CollectionsMarshal.AsSpan(_accesses),
            Sequential.Flatten(CollectionsMarshal.AsSpan(_body)),
            _tileAxisPolicies ?? throw new InvalidOperationException("domain tile-axis policies are not set."));
    }

    public IGridBuilder Domain(int dims, out Var parameter)
        => Domain(Enumerable.Repeat(GridTileAxisPolicy.Search(), dims).ToArray(), out parameter);

    public IGridBuilder Domain(ReadOnlySpan<GridTileAxisPolicy> tileAxisPolicies, out Var parameter)
    {
        if (tileAxisPolicies.IsEmpty)
        {
            throw new ArgumentException("Grid domain must have at least one axis.", nameof(tileAxisPolicies));
        }

        parameter = new Var(new IR.TupleType(Enumerable.Repeat(new IR.TupleType(new IRType[] { TensorType.Scalar(DataTypes.Int64), TensorType.Scalar(DataTypes.Int64) }), tileAxisPolicies.Length)));
        _domainParameter = parameter;
        _tileAxisPolicies = tileAxisPolicies.ToArray();
        return this;
    }

    public IGridBuilder Read(Expr argument, AffineMap accessMap, out Var parameter)
        => Read(argument, accessMap, GridDomainMode.Constraint, out parameter);

    public IGridBuilder Read(Expr argument, AffineMap accessMap, GridDomainMode domainMode, out Var parameter)
        => AddAccess(argument, F.Buffer.BufferOf(argument), accessMap, GridAccessMode.Read, GridBindingMode.Subview, domainMode, out parameter);

    public IGridBuilder Write(Expr buffer, AffineMap accessMap, out Var parameter)
        => AddAccess(buffer, buffer, accessMap, GridAccessMode.Write, GridBindingMode.Subview, GridDomainMode.Constraint, out parameter);

    public IGridBuilder ReadRoot(Expr argument, AffineMap accessMap, out Var parameter)
        => ReadRoot(argument, accessMap, GridDomainMode.Constraint, out parameter);

    public IGridBuilder ReadRoot(Expr argument, AffineMap accessMap, GridDomainMode domainMode, out Var parameter)
        => AddAccess(argument, F.Buffer.BufferOf(argument), accessMap, GridAccessMode.Read, GridBindingMode.Root, domainMode, out parameter);

    public IGridBuilder WriteRoot(Expr buffer, AffineMap accessMap, out Var parameter)
        => WriteRoot(buffer, accessMap, GridDomainMode.Footprint, out parameter);

    public IGridBuilder WriteRoot(Expr buffer, AffineMap accessMap, GridDomainMode domainMode, out Var parameter)
        => AddAccess(buffer, buffer, accessMap, GridAccessMode.Write, GridBindingMode.Root, domainMode, out parameter);

    public IGridBuilder ReadWriteRoot(Expr argument, BaseExpr region, out Var parameter)
        => AddAccess(argument, argument, region, GridAccessMode.ReadWrite, GridBindingMode.Root, GridDomainMode.Footprint, out parameter);

    private static IRType GetParameterType(IRType type, GridBindingMode bindingMode)
    {
        if (bindingMode == GridBindingMode.Root)
        {
            return type switch
            {
                DistributedType distributedType => distributedType.TensorType,
                _ => type,
            };
        }

        return GetTileParameterType(type);
    }

    private static IRType GetTileParameterType(IRType type)
    {
        return type switch
        {
            TensorType { DType: ReferenceType } tensorType => tensorType,
            TensorType tensorType => new TensorType(tensorType.DType, Shape.Unknown(tensorType.Shape.Rank)),
            DistributedType { TensorType.DType: ReferenceType } distributedType => distributedType.TensorType,
            DistributedType distributedType => new TensorType(distributedType.TensorType.DType, Shape.Unknown(distributedType.TensorType.Shape.Rank)),
            _ => type,
        };
    }

    private IGridBuilder AddAccess(
        Expr value,
        Expr buffer,
        BaseExpr region,
        GridAccessMode accessMode,
        GridBindingMode bindingMode,
        GridDomainMode domainMode,
        out Var parameter)
    {
        parameter = new Var(GetParameterType(value.CheckedType, bindingMode));
        _accesses.Add(new GridAccess(value, buffer, parameter, region, accessMode, bindingMode, domainMode));
        return this;
    }
}
