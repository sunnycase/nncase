// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Affine;

/// <summary>
/// A zero-copy tensor view described by an affine relation over shared storage.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class AffineView : Op
{
    /// <summary>
    /// Gets the source tensor.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(AffineView), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets the logical result type of the view.
    /// </summary>
    public IRType NewType { get; }

    /// <summary>
    /// Gets the storage index transform.
    /// </summary>
    public AffineViewTransform Transform { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;

    /// <inheritdoc/>
    public override string DisplayProperty() => $"NewType: {NewType}, Transform: {Transform}";
}
