// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.ComponentModel;

namespace Nncase.Targets;

/// <summary>
/// Target options for PyNTT.
/// </summary>
public sealed class PyNTTTargetOptions : NTTTargetOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PyNTTTargetOptions"/> class.
    /// </summary>
    public PyNTTTargetOptions()
    {
        HierarchyNames = "yx";
        Hierarchies = new[] { new[] { 4, 8 } };
    }

    /// <summary>
    /// Gets or sets the PyNTT backend.
    /// </summary>
    [DisplayName("--pyntt-backend")]
    [Description("PyNTT backend name.")]
    [DefaultValue("triton")]
    public string Backend { get; set; } = "triton";

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
}
