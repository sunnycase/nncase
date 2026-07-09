// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR.NTT;

public enum BarrierScope
{
    Block,
    Chip,
}

public sealed partial class Barrier : NTTKernelOp
{
    public BarrierScope Scope { get; }

    public override bool CanFoldConstCall => false;
}
