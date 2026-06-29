// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using Nncase.Passes;

namespace Nncase.Targets;

/// <summary>
/// Target for PyNTT.
/// </summary>
public sealed class PyNTTTarget : NTTTarget
{
    /// <summary>
    /// PyNTT module kind.
    /// </summary>
    public const string Kind = "pyntt";

    private readonly PyNTTModuleCompiler _moduleCompiler = new();

    /// <inheritdoc/>
    protected override INTTModuleCompiler NTTModuleCompiler => _moduleCompiler;

    /// <inheritdoc/>
    public override void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options)
    {
        // PyNTT consumes tensor-level TIR directly; NTT auto-pack rewrites are not part of this backend path yet.
    }

    /// <inheritdoc/>
    public override void RegisterAutoVectorizeRules(IRulesAddable pass, CompileOptions options)
    {
        // PyNTT emits Triton code from tensor-level TIR and does not use NTT lane-vectorized IR yet.
    }

    /// <inheritdoc/>
    public override (Command Command, Func<InvocationContext, Command, ITargetOptions> Parser) RegisterCommandAndParser()
    {
        var cmd = new NTTTargetOptionsCommand(Name);
        var backendOption = new Option<string>(
            name: "--pyntt-backend",
            description: "PyNTT backend name.",
            getDefaultValue: () => "triton");
        var outputDirOption = new Option<string>(
            name: "--pyntt-output-dir",
            description: "PyNTT generated Python model directory.",
            getDefaultValue: () => string.Empty);
        var strictOption = new Option<bool>(
            name: "--pyntt-strict",
            description: "Reject unsupported PyNTT constructs instead of falling back.",
            getDefaultValue: () => true);

        cmd.Add(backendOption);
        cmd.Add(outputDirOption);
        cmd.Add(strictOption);

        ITargetOptions ParseTargetCompileOptions(InvocationContext context, Command command)
        {
            var nttOptions = new NTTTargetOptionsBinder(cmd).GetBoundValue(context);
            return new PyNTTTargetOptions
            {
                ModelName = nttOptions.ModelName,
                Vectorize = nttOptions.Vectorize,
                UnifiedMemoryArch = nttOptions.UnifiedMemoryArch,
                MemoryAccessArch = nttOptions.MemoryAccessArch,
                NocArch = nttOptions.NocArch,
                HierarchyKind = nttOptions.HierarchyKind,
                Hierarchies = nttOptions.Hierarchies,
                HierarchyNames = nttOptions.HierarchyNames,
                HierarchySizes = nttOptions.HierarchySizes,
                HierarchyLatencies = nttOptions.HierarchyLatencies,
                HierarchyBandWidths = nttOptions.HierarchyBandWidths,
                MemoryCapacities = nttOptions.MemoryCapacities,
                MemoryBandWidths = nttOptions.MemoryBandWidths,
                DistributedScheme = nttOptions.DistributedScheme,
                CustomOpScheme = nttOptions.CustomOpScheme,
                Backend = context.ParseResult.GetValueForOption(backendOption)!,
                OutputDirectory = context.ParseResult.GetValueForOption(outputDirOption)!,
                Strict = context.ParseResult.GetValueForOption(strictOption),
            };
        }

        return (cmd, ParseTargetCompileOptions);
    }
}
