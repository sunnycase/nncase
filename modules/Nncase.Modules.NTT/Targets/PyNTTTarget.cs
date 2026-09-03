// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Linq;
using Nncase.IR;
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
    private readonly CPUModuleCompiler _cpuModuleCompiler = new();
    private readonly CPUTarget _cpuTarget = new();

    /// <inheritdoc/>
    public override IReadOnlyList<IModuleCompiler> ModuleCompilers
        => [_moduleCompiler, _cpuModuleCompiler];

    /// <inheritdoc/>
    public override bool IsAutoTilingEnabled => false;

    /// <inheritdoc/>
    protected override INTTModuleCompiler NTTModuleCompiler => _moduleCompiler;

    /// <inheritdoc/>
    public override ITarget GetModuleTarget(string moduleKind)
        => moduleKind switch
        {
            Kind => this,
            CPUTarget.Kind => _cpuTarget,
            _ => throw new NotSupportedException($"PyNTT has no module target for {moduleKind}."),
        };

    /// <inheritdoc/>
    public override CompileOptions GetModuleCompileOptions(string moduleKind, CompileOptions options)
    {
        if (moduleKind == Kind)
        {
            return options;
        }

        if (moduleKind != CPUTarget.Kind || options.TargetOptions is not PyNTTTargetOptions pynttOptions)
        {
            throw new NotSupportedException($"PyNTT cannot create compile options for module {moduleKind}.");
        }

        var cpuOptions = new NTTTargetOptions
        {
            ModelName = pynttOptions.ModelName,
            Vectorize = true,
            UnifiedMemoryArch = true,
            MemoryAccessArch = MemoryAccessArchitecture.UMA,
            NocArch = NocArchitecture.CrossBar,
            Hierarchies = [new[] { pynttOptions.CpuCoreCount }],
            HierarchyNames = "b",
            HierarchyLevels = "b",
            TargetMachine = pynttOptions.CpuTargetMachine,
            DistributedScheme = string.Empty,
            CustomOpScheme = pynttOptions.CustomOpScheme,
        };
        return options with { TargetOptions = cpuOptions };
    }

    /// <inheritdoc/>
    public override string GetPreferredModuleKind(BaseFunction owner, Call call, CompileOptions options)
    {
        var targetOptions = options.TargetOptions as PyNTTTargetOptions
            ?? throw new InvalidOperationException(
                $"PyNTT placement requires {nameof(PyNTTTargetOptions)}, got {options.TargetOptions?.GetType().Name ?? "null"}.");
        return (call.Metadata.SemanticRegion ?? owner.Metadata.SemanticRegion) is { } region &&
            targetOptions.IsCpuOffloadRegion(region.Kind)
            ? CPUTarget.Kind
            : Kind;
    }

    /// <inheritdoc/>
    public override void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options)
    {
        const int kPack = 2;
        var vectorBytes = NTTModuleCompiler.GetLane(options);

        pass.Add<Passes.Rules.NTT.PackMatMulRhsKMajor>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackScaledMatMulRhsKMajor>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackBlockScaledMatMulRhsNMajorKPacked>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackNVFP4MatMulRhsKMajor>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackNVFP4MatMulGluRhsKMajor>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackQKVParallelLinearRhsForGpu>(vectorBytes, kPack);
        pass.Add<Passes.Rules.NTT.PackBlockScaledMatMulGluRhsNMajorKPacked>(vectorBytes, kPack);
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
        if (TryGetPagedAttentionSplitPlan(options) is { } splitPlan)
        {
            passManager.AddWithName<DataflowPass>("DecomposePagedAttention").Configure(p =>
            {
                p.Add<Passes.Rules.NTT.DecomposePagedAttention>(splitPlan.Axis, splitPlan.Count);
            });
        }

        passManager.AddWithName<DataflowPass>("DecomposeSampling").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.DecomposeSampling>();
        });
        passManager.AddWithName<DataflowPass>("FusePackedMatMulAddBeforeAutoDistributed").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.FusePackedMatMulAdd>();
            p.Add<Passes.Rules.NTT.FusePackedBlockScaledMatMulAdd>();
            p.Add<Passes.Rules.NTT.FusePackedNVFP4MatMulAdd>();
        });
        passManager.Add<FormPackedMatMulNormStatsCombinePass>();
        passManager.Add<FusePackedMatMulNormStatsPass>(false, false, true);
    }

    /// <inheritdoc/>
    public override void RegisterPostAutoDistributedPass(IPassManager passManager, CompileOptions options)
    {
        base.RegisterPostAutoDistributedPass(passManager, options);
        passManager.AddWithName<DataflowPass>("LowerMaterializedPackedMatMulNormStatsCombine").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.LowerMaterializedPackedMatMulNormStatsCombine>();
        });
        passManager.AddWithName<DataflowPass>("FoldMaterializedPackedQKVParallelLinearCombine").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.FoldMaterializedPackedQKVParallelLinearCombine>();
        });
        passManager.AddWithName<DataflowPass>("FoldMaterializedPackedMatMulGluCombine").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.FoldMaterializedPackedMatMulGluCombine>();
        });
        passManager.AddWithName<DataflowPass>("LowerPackedMatMulGluCombine").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.LowerPackedMatMulGluCombine>();
        });
        passManager.AddWithName<DataflowPass>("LowerPackedQKVParallelLinearCombine").Configure(p =>
        {
            p.Add<Passes.Rules.NTT.LowerPackedQKVParallelLinearCombine>();
        });
        passManager.Add<FusePackedMatMulSamplingPartialPass>();
        passManager.Add<SinkNormStatsBoxingAcrossFunctionBoundariesPass>();
        passManager.AddWithName<FunctionBoundaryLayoutPropagationPass>(
            "PropagatePostAutoDistributedFunctionBoundaryLayouts",
            true,
            false);
        passManager.AddWithName<DataflowPass>(
            "FoldPostAutoDistributedFunctionBoundaryBoxing").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldGetItemTuple>();
            p.Add<Passes.Rules.FoldBoxingBoxing>();
            p.Add<Passes.Rules.FoldBoxingShardedView>();
        });
    }

    /// <inheritdoc/>
    public override void RegisterTIRPostBufferizePass(IPassManager passManager, CompileOptions options)
    {
        passManager.AddWithName<InlineSingleCallPrimFunctionsPass>("InlineSingleCallPrimFunctions", Kind);
        passManager.AddWithName<FuseGatherReduceNormApplyPass>(
            "FuseGatherReduceNormApplyAfterInlining",
            Kind);
        passManager.Add<FusePagedAttentionMergePackedMatMulPass>(Kind);
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
        passManager.Add<FuseGatherReduceQKVRoPEWithCachePass>(Kind);
        passManager.Add<FuseGatherReduceAddNormApplyPass>(Kind);
        passManager.Add<ForwardGatherReduceAddNormValuesPass>(Kind);
        passManager.Add<FuseGatherReduceNormApplyPass>(Kind);
        passManager.Add<FuseGatherReduceNormApplyNVFP4MatMulGluPass>(Kind);
        passManager.Add<CanonicalizePackedQKVWeightsPass>(Kind);
        passManager.Add<ForwardTerminalStoreDestinationsPass>(Kind);
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
        var cpuOffloadRegionsOption = new Option<string>(
            name: "--pyntt-cpu-offload-regions",
            description: "Comma-separated semantic region kinds assigned to CPU NTT.",
            getDefaultValue: () => string.Empty);
        var cpuCoreCountOption = new Option<int>(
            name: "--pyntt-cpu-core-count",
            description: "CPU NTT block workers.",
            getDefaultValue: () => Math.Max(1, Environment.ProcessorCount));
        var cpuTargetMachineOption = new Option<string>(
            name: "--pyntt-cpu-target-machine",
            description: "CPU target machine model.",
            getDefaultValue: () => NTTTargetMachineCatalog.CpuGeneric);
        cmd.Add(backendOption);
        cmd.Add(outputDirOption);
        cmd.Add(cpuOffloadRegionsOption);
        cmd.Add(cpuCoreCountOption);
        cmd.Add(cpuTargetMachineOption);

        ITargetOptions ParseTargetCompileOptions(InvocationContext context, Command command)
        {
            var nttOptions = new NTTTargetOptionsBinder(cmd).GetBoundValue(context);
            var pynttOptions = PyNTTTargetOptions.FromNTTTargetOptions(nttOptions);
            pynttOptions.Backend = context.ParseResult.GetValueForOption(backendOption)!;
            pynttOptions.OutputDirectory = context.ParseResult.GetValueForOption(outputDirOption)!;
            pynttOptions.CpuOffloadRegions = context.ParseResult.GetValueForOption(cpuOffloadRegionsOption)!;
            pynttOptions.CpuCoreCount = context.ParseResult.GetValueForOption(cpuCoreCountOption);
            pynttOptions.CpuTargetMachine = context.ParseResult.GetValueForOption(cpuTargetMachineOption)!;
            return pynttOptions;
        }

        return (cmd, ParseTargetCompileOptions);
    }

    private static (int Axis, int Count)? TryGetPagedAttentionSplitPlan(CompileOptions options)
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
            : null;
    }
}
