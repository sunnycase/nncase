// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.ComponentModel;
using Nncase.CostModel;

namespace Nncase.Targets;

/// <summary>
/// Target options for PyNTT.
/// </summary>
public sealed class PyNTTTargetOptions : NTTTargetOptions
{
    private string _backend = "triton";
    private TritonTargetCapability _tritonCapability = TritonTargetCapability.Default;

    /// <summary>
    /// Initializes a new instance of the <see cref="PyNTTTargetOptions"/> class.
    /// </summary>
    public PyNTTTargetOptions()
    {
        HierarchyNames = "yx";
        HierarchyLevels = "bb";
        Hierarchies = new[] { new[] { 4, 8 } };
        ConstShardedView = true;
        RefreshTargetCostModel();
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
            _backend = value;
            RefreshTargetCostModel();
        }
    }

    /// <summary>
    /// Gets or sets the Triton backend hardware capability.
    /// </summary>
    public TritonTargetCapability TritonCapability
    {
        get => _tritonCapability;
        set
        {
            _tritonCapability = value;
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
    /// Gets or sets a value indicating whether PyNTT should reject unsupported constructs strictly.
    /// </summary>
    [DisplayName("--pyntt-strict")]
    [Description("Reject unsupported PyNTT constructs instead of falling back.")]
    [DefaultValue(true)]
    public bool Strict { get; set; } = true;

    public static PyNTTTargetOptions FromNTTTargetOptions(NTTTargetOptions nttOptions)
    {
        return new PyNTTTargetOptions
        {
            ModelName = nttOptions.ModelName,
            Vectorize = nttOptions.Vectorize,
            UnifiedMemoryArch = nttOptions.UnifiedMemoryArch,
            ConstShardedView = nttOptions.ConstShardedView,
            MemoryAccessArch = nttOptions.MemoryAccessArch,
            NocArch = nttOptions.NocArch,
            Hierarchies = nttOptions.Hierarchies,
            HierarchyNames = nttOptions.HierarchyNames,
            HierarchyLevels = nttOptions.HierarchyLevels,
            HierarchySizes = nttOptions.HierarchySizes,
            HierarchyLatencies = nttOptions.HierarchyLatencies,
            HierarchyBandWidths = nttOptions.HierarchyBandWidths,
            MemoryCapacities = nttOptions.MemoryCapacities,
            MemoryBandWidths = nttOptions.MemoryBandWidths,
            DistributedScheme = nttOptions.DistributedScheme,
            CustomOpScheme = nttOptions.CustomOpScheme,
        };
    }

    private void RefreshTargetCostModel()
    {
        TargetCostModel = string.Equals(_backend, "triton", StringComparison.OrdinalIgnoreCase)
            ? new TritonTargetOpCostModel(_tritonCapability)
            : DefaultTargetOpCostModel.Instance;
    }
}
