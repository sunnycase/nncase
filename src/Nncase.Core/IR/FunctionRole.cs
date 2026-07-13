// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

/// <summary>
/// Describes a function's execution role independently of its name.
/// </summary>
public enum FunctionRole
{
    /// <summary>
    /// General compute function whose call boundary carries backend scheduling semantics.
    /// </summary>
    Compute,

    /// <summary>
    /// Control-flow-only dispatcher that may be composed into its caller.
    /// </summary>
    Dispatch,

    /// <summary>
    /// A schedule-selected compute region owned by the backend below the block hierarchy.
    /// </summary>
    ScheduledRegion,
}
