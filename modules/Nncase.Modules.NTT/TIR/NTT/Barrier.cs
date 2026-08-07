// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

public enum BarrierScope
{
    Block,
    Chip,
}

public sealed partial class Barrier : NTTKernelOp
{
    public BarrierScope Scope { get; }

    /// <summary>
    /// Gets the physical placement axes that vary inside the current chip
    /// axis group. An empty set denotes the full chip mesh. Block barriers do
    /// not carry axis-group axes.
    /// </summary>
    public IRArray<int> AxisGroupAxes { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty()
        => $"Scope: {Scope}, AxisGroupAxes: [{string.Join(",", AxisGroupAxes)}]";
}
