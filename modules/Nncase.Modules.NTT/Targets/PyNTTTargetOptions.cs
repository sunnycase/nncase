// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.Passes.Distributed;
using Nncase.Schedule;
using Nncase.Utilities;

namespace Nncase.Targets;

/// <summary>
/// Target options for PyNTT.
/// </summary>
public sealed class PyNTTTargetOptions : NTTTargetOptions, IPagedAttentionExecutionPlanProvider
{
    private string _backend = "triton";
    private TritonPagedAttentionExecutionPlanner _pagedAttentionExecutionPlanner = null!;
    private long _blockCyclicBlockBytes = 128;
    private int _cpuCoreCount = Math.Max(1, Environment.ProcessorCount);
    private IReadOnlySet<string> _cpuOffloadRegions = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>
    /// Initializes a new instance of the <see cref="PyNTTTargetOptions"/> class.
    /// </summary>
    public PyNTTTargetOptions()
    {
        HierarchyNames = "yx";
        HierarchyLevels = "bb";
        Hierarchies = new[] { new[] { 4, 8 } };
        TargetMachine = NTTTargetMachineCatalog.Rtx5060Ti16Gb;
    }

    /// <summary>
    /// Gets or sets the PyNTT backend.
    /// </summary>
    [DisplayName("--pyntt-backend")]
    [Description("PyNTT backend name.")]
    [DefaultValue("triton")]
    public string Backend
    {
        get => _backend;
        set
        {
            if (!string.Equals(value, "triton", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"PyNTT supports only the triton backend, got '{value}'.", nameof(value));
            }

            _backend = "triton";
            RefreshTargetCostModel();
        }
    }

    /// <summary>
    /// Gets or sets the generated Python model directory.
    /// </summary>
    [DisplayName("--pyntt-output-dir")]
    [Description("PyNTT generated Python model directory.")]
    [DefaultValue("")]
    public string OutputDirectory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the comma-separated semantic region kinds assigned to CPU
    /// NTT. An empty value disables CPU placement.
    /// </summary>
    [DisplayName("--pyntt-cpu-offload-regions")]
    [Description("Comma-separated semantic region kinds assigned to CPU NTT.")]
    [DefaultValue("")]
    public string CpuOffloadRegions
    {
        get => string.Join(",", _cpuOffloadRegions.OrderBy(kind => kind, StringComparer.Ordinal));
        set
        {
            var regions = (value ?? string.Empty)
                .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            _cpuOffloadRegions = new HashSet<string>(regions, StringComparer.Ordinal);
        }
    }

    /// <summary>
    /// Gets or sets the number of CPU NTT block workers.
    /// </summary>
    [DisplayName("--pyntt-cpu-core-count")]
    [Description("CPU NTT block workers used by heterogeneous PyNTT models.")]
    public int CpuCoreCount
    {
        get => _cpuCoreCount;
        set => _cpuCoreCount = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "CPU core count must be positive.");
    }

    /// <summary>
    /// Gets or sets the CPU target machine model.
    /// </summary>
    [DisplayName("--pyntt-cpu-target-machine")]
    [Description("CPU target machine model used by heterogeneous PyNTT models.")]
    [DefaultValue(NTTTargetMachineCatalog.CpuGeneric)]
    public string CpuTargetMachine { get; set; } = NTTTargetMachineCatalog.CpuGeneric;

    /// <summary>
    /// Gets or sets the independent contiguous byte granule used by physical
    /// block-cyclic split candidates.
    /// </summary>
    [DisplayName("--pyntt-block-cyclic-block-bytes")]
    [Description("Independent contiguous byte granule for block-cyclic split stages.")]
    [DefaultValue(128L)]
    public long BlockCyclicBlockBytes
    {
        get => _blockCyclicBlockBytes;
        set
        {
            if (value <= 0 || !System.Numerics.BitOperations.IsPow2((ulong)value))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(value),
                    value,
                    "PyNTT block-cyclic byte granularity must be a positive power of two.");
            }

            _blockCyclicBlockBytes = value;
        }
    }

    /// <inheritdoc/>
    public override IDistributedReshardRealizationPolicy ReshardRealizationPolicy
        => PyNTTDistributedReshardRealizationPolicy.Instance;

    public override IDistributedSplitCandidateProvider DistributedSplitCandidateProvider
        => new PyNTTDistributedSplitCandidateProvider(BlockCyclicBlockBytes);

    public PagedAttentionExecutionPlan GetPagedAttentionExecutionPlan(
        PagedAttentionExecutionPlanQuery query)
        => _pagedAttentionExecutionPlanner.Plan(query);

    public bool IsCpuOffloadRegion(string regionKind)
        => _cpuOffloadRegions.Contains(regionKind);

    public static PyNTTTargetOptions FromNTTTargetOptions(NTTTargetOptions nttOptions)
    {
        return new PyNTTTargetOptions
        {
            ModelName = nttOptions.ModelName,
            Vectorize = nttOptions.Vectorize,
            UnifiedMemoryArch = nttOptions.UnifiedMemoryArch,
            MemoryAccessArch = nttOptions.MemoryAccessArch,
            NocArch = nttOptions.NocArch,
            Hierarchies = nttOptions.Hierarchies,
            HierarchyNames = nttOptions.HierarchyNames,
            HierarchyLevels = nttOptions.HierarchyLevels,
            HierarchySizes = nttOptions.HierarchySizes,
            HierarchyLatencies = nttOptions.HierarchyLatencies,
            HierarchyBandWidths = nttOptions.HierarchyBandWidths,
            TargetMachine = nttOptions.TargetMachine,
            DistributedScheme = nttOptions.DistributedScheme,
            CustomOpScheme = nttOptions.CustomOpScheme,
        };
    }

    protected override void OnTargetMachineChanged()
    {
        TargetCostModel = new TritonTargetOpCostModel(TargetMachineModel);
        _pagedAttentionExecutionPlanner = new TritonPagedAttentionExecutionPlanner(TargetMachineModel);
        BlockMicroKernelModel = new DefaultBlockMicroKernelModel();
        TIRMicroKernelSelector = new TritonTIRMicroKernelSelector();
        StorageEncodingModel = new DefaultTargetStorageEncodingModel();
        LoopPipelineBackend = new EmptyLoopPipelineBackend();
    }

    private void RefreshTargetCostModel() => OnTargetMachineChanged();
}
