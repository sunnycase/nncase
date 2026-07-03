// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Distributed;

/// <summary>
/// Read-only distributed alias view over an unsharded tensor.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ShardedView : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ShardedView), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets the distributed view type.
    /// </summary>
    public DistributedType NewType { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;

    /// <inheritdoc/>
    public override string DisplayProperty() => $"NewType: {NewType}";
}
