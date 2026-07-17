// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// Physical layout contract carried by a prim function buffer parameter.
/// </summary>
public enum BufferLayoutKind
{
    /// <summary>
    /// The logical tensor has the exact element strides recorded by the annotation.
    /// </summary>
    ExactStrided,

    /// <summary>
    /// Element strides are supplied by the runtime ABI.
    /// </summary>
    RuntimeStrided,

    /// <summary>
    /// The parameter is an opaque resource or workspace pointer with no tensor layout.
    /// </summary>
    Opaque,
}

/// <summary>
/// TIR buffer parameter role in a prim function ABI.
/// </summary>
public enum BufferVarRole
{
    /// <summary>
    /// Regular input buffer parameter.
    /// </summary>
    Input,

    /// <summary>
    /// Caller-allocated output buffer parameter.
    /// </summary>
    Output,

    /// <summary>
    /// Input/output buffer parameter.
    /// </summary>
    InOut,

    /// <summary>
    /// Caller-provided runtime workspace buffer parameter.
    /// </summary>
    Workspace,
}

/// <summary>
/// Immutable physical layout contract for a <see cref="BufferVar"/>.
/// </summary>
public sealed record BufferLayoutAnnotation
{
    /// <summary>
    /// Initializes a new instance of the <see cref="BufferLayoutAnnotation"/> class.
    /// </summary>
    public BufferLayoutAnnotation(BufferLayoutKind kind, IRArray<Dimension> strides)
    {
        if (kind != BufferLayoutKind.ExactStrided && strides.Count != 0)
        {
            throw new ArgumentException($"{kind} buffer layouts cannot carry exact strides.", nameof(strides));
        }

        Kind = kind;
        Strides = strides;
    }

    /// <summary>
    /// Gets the layout kind.
    /// </summary>
    public BufferLayoutKind Kind { get; }

    /// <summary>
    /// Gets exact logical element strides. The array is non-empty only for
    /// <see cref="BufferLayoutKind.ExactStrided"/> ranked tensors.
    /// </summary>
    public IRArray<Dimension> Strides { get; }

    /// <summary>
    /// Gets the runtime-strided tensor layout.
    /// </summary>
    public static BufferLayoutAnnotation RuntimeStrided { get; } = new(BufferLayoutKind.RuntimeStrided, new IRArray<Dimension>());

    /// <summary>
    /// Gets the opaque non-tensor layout.
    /// </summary>
    public static BufferLayoutAnnotation Opaque { get; } = new(BufferLayoutKind.Opaque, new IRArray<Dimension>());

    /// <summary>
    /// Creates an exact strided tensor layout.
    /// </summary>
    public static BufferLayoutAnnotation ExactStrided(ReadOnlySpan<Dimension> strides)
        => new(BufferLayoutKind.ExactStrided, new IRArray<Dimension>(strides));

    /// <inheritdoc/>
    public override string ToString()
        => Kind == BufferLayoutKind.ExactStrided
            ? $"exact[{string.Join(",", Strides)}]"
            : Kind.ToString();
}

/// <summary>
/// TIR-only variable used for prim function buffer ABI parameters.
/// </summary>
public sealed class BufferVar : Expr, IVar, IEquatable<BufferVar?>
{
    private static int _globalVarIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="BufferVar"/> class.
    /// </summary>
    public BufferVar(
        string name,
        IRType typeAnnotation,
        BufferVarRole role,
        MemoryLocation location,
        BufferLayoutAnnotation? layoutAnnotation = null)
        : base(Array.Empty<BaseExpr>())
    {
        Name = name;
        TypeAnnotation = typeAnnotation;
        Role = role;
        Location = location;
        LayoutAnnotation = layoutAnnotation ?? CreateDefaultLayout(typeAnnotation, role);
        ValidateLayout(typeAnnotation, role, LayoutAnnotation);
        RawCheckedType = TypeAnnotation;
        GlobalVarIndex = GetNextId();
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public int GlobalVarIndex { get; }

    /// <summary>
    /// Gets the type annotation.
    /// </summary>
    public IRType TypeAnnotation { get; }

    /// <summary>
    /// Gets the prim function ABI role.
    /// </summary>
    public BufferVarRole Role { get; }

    /// <summary>
    /// Gets the buffer memory location.
    /// </summary>
    public MemoryLocation Location { get; }

    /// <summary>
    /// Gets the physical layout contract for this ABI parameter.
    /// </summary>
    public BufferLayoutAnnotation LayoutAnnotation { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitBufferVar(this, context);

    /// <summary>
    /// Creates a copy with updated fields.
    /// </summary>
    public BufferVar With(
        string? name = null,
        IRType? typeAnnotation = null,
        BufferVarRole? role = null,
        MemoryLocation? location = null,
        BufferLayoutAnnotation? layoutAnnotation = null)
        => new(
            name ?? Name,
            typeAnnotation ?? TypeAnnotation,
            role ?? Role,
            location ?? Location,
            layoutAnnotation ?? LayoutAnnotation)
        { Metadata = Metadata };

    IVar IVar.With(string? name) => With(name);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as BufferVar);

    /// <inheritdoc/>
    public bool Equals(BufferVar? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && GlobalVarIndex == other.GlobalVarIndex;
    }

    bool IEquatable<IVar?>.Equals(IVar? other) => other is BufferVar bufferVar && Equals(bufferVar);

    /// <inheritdoc/>
    public override string ToString() => Name;

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(GlobalVarIndex);

    private static int GetNextId()
    {
        return Interlocked.Increment(ref _globalVarIndex);
    }

    private static BufferLayoutAnnotation CreateDefaultLayout(IRType type, BufferVarRole role)
    {
        if (role == BufferVarRole.Workspace)
        {
            return BufferLayoutAnnotation.Opaque;
        }

        var tensorType = type switch
        {
            DistributedType distributedType => DistributedUtility.GetDividedTensorType(distributedType),
            TensorType value => value,
            _ => null,
        };
        if (tensorType?.Shape is not RankedShape shape || tensorType.DType is PointerType or ReferenceType)
        {
            return BufferLayoutAnnotation.Opaque;
        }

        return BufferLayoutAnnotation.ExactStrided(TensorUtilities.GetDefaultStrides(shape.Dimensions));
    }

    private static void ValidateLayout(IRType type, BufferVarRole role, BufferLayoutAnnotation layout)
    {
        if (role == BufferVarRole.Workspace && layout.Kind != BufferLayoutKind.Opaque)
        {
            throw new ArgumentException($"Workspace BufferVar {type} must use an opaque layout.", nameof(layout));
        }

        var tensorType = type switch
        {
            DistributedType distributedType => distributedType.TensorType,
            TensorType value => value,
            _ => null,
        };
        var isTensorLayout = tensorType?.Shape is RankedShape && tensorType.DType is not (PointerType or ReferenceType);
        if (!isTensorLayout)
        {
            if (layout.Kind != BufferLayoutKind.Opaque)
            {
                throw new ArgumentException($"Non-tensor BufferVar {type} must use an opaque layout.", nameof(layout));
            }

            return;
        }

        if (layout.Kind == BufferLayoutKind.Opaque)
        {
            throw new ArgumentException($"Tensor BufferVar {type} cannot use an opaque layout.", nameof(layout));
        }

        if (layout.Kind == BufferLayoutKind.ExactStrided && layout.Strides.Count != tensorType!.Shape.Rank)
        {
            throw new ArgumentException(
                $"Exact BufferVar layout rank mismatch: type rank={tensorType.Shape.Rank}, strides={layout.Strides.Count}.",
                nameof(layout));
        }
    }
}
