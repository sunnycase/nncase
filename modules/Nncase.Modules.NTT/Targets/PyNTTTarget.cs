// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using Nncase.Passes;
using Nncase.Passes.Transforms;

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
        base.RegisterAutoPackingRules(pass, options);
    }

    /// <inheritdoc/>
    public override void RegisterAutoVectorizeRules(IRulesAddable pass, CompileOptions options)
    {
        base.RegisterAutoVectorizeRules(pass, options);
    }

    /// <inheritdoc/>
    public override void RegisterTIRPreBufferizePass(IPassManager passManager, CompileOptions options)
    {
        passManager.AddWithName<LowerReshardToChipLocalDataPass>("LowerReshardToChipLocalData", Kind);
    }

    /// <inheritdoc/>
    public override void RegisterTIRPostBufferizePass(IPassManager passManager, CompileOptions options)
    {
        passManager.AddWithName<PlanMemorySynchronizationPass>("PlanMemorySynchronization", Kind);
    }

    /// <inheritdoc/>
    public override (Command Command, Func<InvocationContext, Command, ITargetOptions> Parser) RegisterCommandAndParser()
    {
        var cmd = new NTTTargetOptionsCommand(Name, NTTTargetMachineCatalog.Rtx5060Ti16Gb);
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
            var pynttOptions = PyNTTTargetOptions.FromNTTTargetOptions(nttOptions);
            pynttOptions.Backend = context.ParseResult.GetValueForOption(backendOption)!;
            pynttOptions.OutputDirectory = context.ParseResult.GetValueForOption(outputDirOption)!;
            pynttOptions.Strict = context.ParseResult.GetValueForOption(strictOption);
            return pynttOptions;
        }

        return (cmd, ParseTargetCompileOptions);
    }
}
