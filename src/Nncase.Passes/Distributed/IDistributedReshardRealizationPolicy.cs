// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Distributed;

/// <summary>
/// Target policy that maps a semantic reshard to exactly one physical realization.
/// </summary>
public interface IDistributedReshardRealizationPolicy
{
    /// <summary>
    /// Returns whether constants are represented by sharded alias views for the target configuration.
    /// </summary>
    bool UsesShardedViewsForConstants(INTTTargetOptions targetOptions);

    /// <summary>
    /// Selects the unique physical realization for <paramref name="context"/>.
    /// </summary>
    DistributedReshardRealization Classify(DistributedReshardRealizationContext context);
}

/// <summary>
/// Implemented by target options that provide target-specific reshard realization rules.
/// </summary>
public interface IDistributedReshardRealizationPolicyProvider
{
    /// <summary>
    /// Gets the target's reshard realization policy.
    /// </summary>
    IDistributedReshardRealizationPolicy ReshardRealizationPolicy { get; }
}
