// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.Schedule;

namespace Nncase.Targets;

public class NTTTargetOptions : INTTTargetOptions, ITargetOpCostModelProvider, ITargetMachineModelProvider
{
    private string _targetMachine = NTTTargetMachineCatalog.CpuGeneric;
    private TargetMachineModel _targetMachineModel = NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.CpuGeneric);

    public NTTTargetOptions()
    {
        TargetCostModel = new DefaultTargetOpCostModel(_targetMachineModel);
        BlockMicroKernelModel = new DefaultBlockMicroKernelModel();
        StorageEncodingModel = new DefaultTargetStorageEncodingModel();
        LoopPipelineBackend = new EmptyLoopPipelineBackend();
    }

    [DisplayName("--model-name")]
    [Description("the input model name.")]
    [DefaultValue("")]
    public string ModelName { get; set; } = string.Empty;

    [DisplayName("--vectorize")]
    [Description("enable simd layout optimization.")]
    [DefaultValue(false)]
    public bool Vectorize { get; set; }

    [DisplayName("--unified-memory-arch")]
    [Description("whether Unified Memory Architecture. see https://en.wikipedia.org/wiki/Unified_Memory_Access.")]
    [DefaultValue(true)]
    public bool UnifiedMemoryArch { get; set; } = true;

    [DisplayName("--memory-access-arch")]
    [Description("Memory Access Architecture.")]
    [DefaultValue(MemoryAccessArchitecture.UMA)]
    [CommandLine.FromAmong(MemoryAccessArchitecture.UMA, MemoryAccessArchitecture.NUMA)]
    public MemoryAccessArchitecture MemoryAccessArch { get; set; } = MemoryAccessArchitecture.UMA;

    [DisplayName("--noc-arch")]
    [Description("Noc Architecture.")]
    [DefaultValue(NocArchitecture.Mesh)]
    [CommandLine.FromAmong(NocArchitecture.Mesh, NocArchitecture.CrossBar)]
    public NocArchitecture NocArch { get; set; } = NocArchitecture.Mesh;

    [DisplayName("--hierarchies")]
    [Description("the distributed hierarchies of hardware. eg. `8,4 4,8` for dynamic cluster search or `4` for fixed hardware.")]
    [DefaultValue("() => new int[][] { new int[] { 1 } }")]
    [AmbientValue("ParseNestedIntArray")]
    [CommandLine.AllowMultiplePerToken]
    public int[][] Hierarchies { get; set; } = new int[][] { new int[] { 1 } };

    [DisplayName("--hierarchy-names")]
    [Description("the name identify of hierarchies.")]
    [DefaultValue("b")]
    public string HierarchyNames { get; set; } = "b";

    [DisplayName("--hierarchy-levels")]
    [Description("the physical level mapping of logical hierarchy axes. Supported levels are c(chip), d(die), b(block).")]
    [DefaultValue("b")]
    public string HierarchyLevels { get; set; } = "b";

    [DisplayName("--hierarchy-sizes")]
    [Description("the memory capacity of hierarchies.")]
    [DefaultValue("() => new long[] { 1099511627776 }")]
    [CommandLine.AllowMultiplePerToken]
    public long[] HierarchySizes { get; set; } = new[] { 1 * (long)MathF.Pow(2, 40) };

    [DisplayName("--hierarchy-latencies")]
    [Description("the latency of hierarchies.")]
    [DefaultValue("() => new int[] { 10000 }")]
    [CommandLine.AllowMultiplePerToken]
    public int[] HierarchyLatencies { get; set; } = new[] { 10000 };

    [DisplayName("--hierarchy-bandwiths")]
    [Description("the bandwidth of hierarchies.")]
    [DefaultValue("() => new int[] { 1 }")]
    [CommandLine.AllowMultiplePerToken]
    public int[] HierarchyBandWidths { get; set; } = new[] { 1 };

    [DisplayName("--target-machine")]
    [Description("the canonical target machine model used by cost evaluation and AutoTiling.")]
    [DefaultValue(NTTTargetMachineCatalog.CpuGeneric)]
    public string TargetMachine
    {
        get => _targetMachine;
        set
        {
            var model = NTTTargetMachineCatalog.Resolve(value);
            TargetMachineModel = model;
        }
    }

    public TargetMachineModel TargetMachineModel
    {
        get => _targetMachineModel;
        set
        {
            _targetMachineModel = value ?? throw new ArgumentNullException(nameof(value));
            _targetMachine = value.Id;
            OnTargetMachineChanged();
        }
    }

    [DisplayName("--distributed--scheme")]
    [Description("the distributed scheme path.")]
    [DefaultValue("")]
    public string DistributedScheme { get; set; } = string.Empty;

    [DisplayName("--custom-op-scheme")]
    [Description("the custom-op scheme path.")]
    [DefaultValue("")]
    public string CustomOpScheme { get; set; } = string.Empty;

    public ITargetOpCostModel TargetCostModel { get; set; }

    public IBlockMicroKernelModelProvider BlockMicroKernelModel { get; protected set; }

    public ITargetStorageEncodingModelProvider StorageEncodingModel { get; protected set; }

    public ILoopPipelineBackend LoopPipelineBackend { get; protected set; }

    protected virtual void OnTargetMachineChanged()
    {
        TargetCostModel = new DefaultTargetOpCostModel(TargetMachineModel);
        BlockMicroKernelModel = new DefaultBlockMicroKernelModel();
        StorageEncodingModel = new DefaultTargetStorageEncodingModel();
        LoopPipelineBackend = new EmptyLoopPipelineBackend();
    }
}
