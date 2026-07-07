// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

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
/// TIR-only variable used for prim function buffer ABI parameters.
/// </summary>
public sealed class BufferVar : Expr, IVar, IEquatable<BufferVar?>
{
    private static int _globalVarIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="BufferVar"/> class.
    /// </summary>
    public BufferVar(string name, IRType typeAnnotation, BufferVarRole role, MemoryLocation location)
        : base(Array.Empty<BaseExpr>())
    {
        Name = name;
        TypeAnnotation = typeAnnotation;
        Role = role;
        Location = location;
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

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitBufferVar(this, context);

    /// <summary>
    /// Creates a copy with updated fields.
    /// </summary>
    public BufferVar With(string? name = null, IRType? typeAnnotation = null, BufferVarRole? role = null, MemoryLocation? location = null)
        => new(name ?? Name, typeAnnotation ?? TypeAnnotation, role ?? Role, location ?? Location) { Metadata = Metadata };

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
}
