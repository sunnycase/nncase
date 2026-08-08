// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Distributed;

/// <summary>
/// Physical realization selected for one semantically legal distributed reshard.
/// </summary>
public enum DistributedReshardRealization
{
    /// <summary>
    /// Materialize the target shard through a data-moving boxing operation.
    /// </summary>
    Boxing,

    /// <summary>
    /// Materialize the target shard as an alias over canonical shared backing storage.
    /// </summary>
    ShardedView,

    /// <summary>
    /// The target cannot realize this reshard at the specified program boundary.
    /// </summary>
    Unsupported,
}

/// <summary>
/// Describes where the source value of a reshard originates.
/// </summary>
public enum DistributedReshardSourceKind
{
    /// <summary>
    /// Immutable tensor constant.
    /// </summary>
    Constant,

    /// <summary>
    /// Temporary produced and consumed within one high-level function.
    /// </summary>
    Internal,

    /// <summary>
    /// Function parameter whose physical storage is owned by the caller.
    /// </summary>
    FunctionParameter,
}

/// <summary>
/// Describes where the result of a reshard is consumed.
/// </summary>
public enum DistributedReshardUsageKind
{
    /// <summary>
    /// Value remains within one high-level function.
    /// </summary>
    Internal,

    /// <summary>
    /// Value crosses a high-level function ABI boundary.
    /// </summary>
    FunctionBoundary,

    /// <summary>
    /// Value is returned from the program entry through caller-allocated output storage.
    /// </summary>
    ProgramOutput,
}
