// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Linq;
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
    public override bool IsAutoTilingEnabled => false;

    /// <inheritdoc/>
    protected override INTTModuleCompiler NTTModuleCompiler => _moduleCompiler;

    /// <inheritdoc/>
    public override void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options)
    {
        const int kPack = 2;
        var vectorBytes = NTTModuleCompiler.GetLane(options);

        pass.Add<Passes.Rules.NTT.PackMatMulRhsKMajor>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackQKVParallelLinearRhsKMajor>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackMatMulGluRhsKMajor>(vectorBytes, kPack);
    }

    /// <inheritdoc/>
    public override void RegisterAutoTilingPass(IPassManager passManager, CompileOptions options)
    {
        // PyNTT templates own all block-local tiling, staging, and pipelining.
    }

    /// <inheritdoc/>
    public override void RegisterPostAutoPackingPass(IPassManager passManager, CompileOptions options)
    {
        var (splitHierarchyAxis, splitCount) = GetPagedAttentionSplitPlan(options);
        passManager.AddWithName<DataflowPass>("DecomposePagedAttention").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.DecomposePagedAttention>(splitHierarchyAxis, splitCount);
        });
        passManager.AddWithName<DataflowPass>("FusePackedMatMulAddBeforeAutoDistributed").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.FusePackedMatMulAdd>();
        });
        passManager.Add<FusePackedMatMulNormStatsPass>();
    }

    /// <inheritdoc/>
    public override void RegisterTIRPostBufferizePass(IPassManager passManager, CompileOptions options)
    {
        passManager.AddWithName<InlineSingleCallPrimFunctionsPass>("InlineSingleCallPrimFunctions", Kind);
        passManager.AddWithName<PlanMemorySynchronizationPass>(
            "PlanMemorySynchronization",
            Kind,
            MemorySynchronizationScopes.All);
        passManager.AddWithName<LowerTransferPipelineRegionsPass>(
            "LowerTransferPipelineRegions",
            Kind);
    }

    /// <inheritdoc/>
    public override void RegisterTIRPreBufferizePass(IPassManager passManager, CompileOptions options)
    {
        passManager.Add<CanonicalizePackedQKVWeightsPass>(Kind);
        base.RegisterTIRPreBufferizePass(passManager, options);
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
        cmd.Add(backendOption);
        cmd.Add(outputDirOption);

        ITargetOptions ParseTargetCompileOptions(InvocationContext context, Command command)
        {
            var nttOptions = new NTTTargetOptionsBinder(cmd).GetBoundValue(context);
            var pynttOptions = PyNTTTargetOptions.FromNTTTargetOptions(nttOptions);
            pynttOptions.Backend = context.ParseResult.GetValueForOption(backendOption)!;
            pynttOptions.OutputDirectory = context.ParseResult.GetValueForOption(outputDirOption)!;
            return pynttOptions;
        }

        return (cmd, ParseTargetCompileOptions);
    }

    private static (int Axis, int Count) GetPagedAttentionSplitPlan(CompileOptions options)
    {
        if (options.TargetOptions is not INTTTargetOptions targetOptions ||
            targetOptions.Hierarchies.Length == 0)
        {
            throw new InvalidOperationException(
                "PyNTT paged-attention decomposition requires at least one target hierarchy.");
        }

        var hierarchyRank = targetOptions.Hierarchies[0].Length;
        var levels = Nncase.IR.Placement.NormalizeHierarchyLevels(
            targetOptions.HierarchyLevels,
            targetOptions.HierarchyNames,
            hierarchyRank);
        if (targetOptions.Hierarchies.Any(hierarchy => hierarchy.Length != hierarchyRank))
        {
            throw new InvalidOperationException(
                "PyNTT paged-attention decomposition requires all target hierarchies to have the same rank.");
        }

        var candidates = Enumerable.Range(0, hierarchyRank)
            .Where(axis => levels[axis] == 'b')
            .Select(axis => new
            {
                Axis = axis,
                Extents = targetOptions.Hierarchies.Select(hierarchy => hierarchy[axis]).Distinct().ToArray(),
            })
            .Where(candidate => candidate.Extents.Length == 1 && candidate.Extents[0] > 1)
            .OrderBy(candidate => candidate.Extents[0])
            .ThenBy(candidate => candidate.Axis)
            .FirstOrDefault();
        return candidates is not null
            ? (candidates.Axis, candidates.Extents[0])
            : throw new InvalidOperationException(
                "PyNTT split-KV paged attention requires a physical block hierarchy axis " +
                "with one fixed extent greater than one across all placement candidates.");
    }
}
