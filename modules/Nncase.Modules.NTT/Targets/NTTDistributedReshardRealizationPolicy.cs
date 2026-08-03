// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Utilities;

namespace Nncase.Targets;

/// <summary>
/// Selects reshard realizations shared by NTT-family targets.
/// </summary>
public sealed class NTTDistributedReshardRealizationPolicy : IDistributedReshardRealizationPolicy
{
    private NTTDistributedReshardRealizationPolicy()
    {
    }

    /// <summary>
    /// Gets the shared policy instance.
    /// </summary>
    public static NTTDistributedReshardRealizationPolicy Instance { get; } = new();

    /// <inheritdoc/>
    public bool UsesShardedViewsForConstants(INTTTargetOptions targetOptions)
        => HasUnifiedSharedStorage(targetOptions);

    /// <inheritdoc/>
    public DistributedReshardRealization Classify(DistributedReshardRealizationContext context)
        => CanAliasConstant(context)
            ? DistributedReshardRealization.ShardedView
            : DistributedReshardRealization.Boxing;

    internal static bool CanAliasConstant(DistributedReshardRealizationContext context)
        => HasUnifiedSharedStorage(context.TargetOptions) &&
            context.SourceKind == DistributedReshardSourceKind.Constant &&
            context.SourceType is TensorType &&
            context.TargetType is DistributedType targetType &&
            DistributedUtility.TryValidateShardedView(context.SourceType, targetType, out _);

    private static bool HasUnifiedSharedStorage(INTTTargetOptions targetOptions)
        => targetOptions.UnifiedMemoryArch &&
            targetOptions.MemoryAccessArch == MemoryAccessArchitecture.UMA;
}
