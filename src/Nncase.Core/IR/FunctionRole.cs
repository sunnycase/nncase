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
    /// Control-flow-only dispatcher resolved outside backend compute kernels.
    /// </summary>
    Dispatch,

    /// <summary>
    /// Host-side dispatcher whose callees belong to different backend modules.
    /// It is preserved through target lowering and interpreted by the owning
    /// host runtime.
    /// </summary>
    ModuleDispatch,

    /// <summary>
    /// One persistent execution worker for a module in a heterogeneous pipeline.
    /// Calls within the worker are backend-local device or block functions.
    /// </summary>
    PipelineWorker,

    /// <summary>
    /// Backend-local projection of one semantic function in a heterogeneous
    /// pipeline. The call boundary is part of the persistent pipeline structure.
    /// </summary>
    PipelineProjection,

    /// <summary>
    /// A schedule-selected compute region owned by the backend below the block hierarchy.
    /// </summary>
    ScheduledRegion,
}
