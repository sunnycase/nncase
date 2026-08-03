// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Context supplied by AutoDistributed when selecting a reshard realization.
/// </summary>
public sealed record DistributedReshardRealizationContext(
    INTTTargetOptions TargetOptions,
    string ModuleKind,
    IRType SourceType,
    IRType TargetType,
    DistributedReshardSourceKind SourceKind,
    DistributedReshardUsageKind UsageKind);
