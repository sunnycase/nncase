// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Resolves the reshard policy attached to target options.
/// </summary>
public static class DistributedReshardRealizationPolicy
{
    /// <summary>
    /// Gets the target policy.
    /// </summary>
    public static IDistributedReshardRealizationPolicy Get(INTTTargetOptions targetOptions)
        => targetOptions is IDistributedReshardRealizationPolicyProvider provider
            ? provider.ReshardRealizationPolicy
            : throw new InvalidOperationException(
                $"Target options {targetOptions.GetType().Name} must provide an " +
                $"{nameof(IDistributedReshardRealizationPolicy)}.");
}
