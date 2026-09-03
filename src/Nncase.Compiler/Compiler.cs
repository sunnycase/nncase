// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Security.AccessControl;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Distributed;
using Nncase.Passes.Mutators;
using Nncase.Passes.Rules;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.Passes.Rules.WithMarker;
using Nncase.Passes.Transforms;
using Nncase.Quantization;
using CombinePadTranspose = Nncase.Passes.Rules.WithMarker.CombinePadTranspose;
using CombineReshapePad = Nncase.Passes.Rules.Neutral.CombineReshapePad;
using FoldConstCall = Nncase.Passes.Rules.Neutral.FoldConstCall;

namespace Nncase.Compiler;

public class Compiler : ICompiler
{
    private readonly CompileSession _compileSession;
    private readonly IModelBuilder _modelBuilder;
    private IRModule? _module;
    private int _runPassCount;

    private CompileSession CurrentSession => CompileSessionScope.Current ?? _compileSession;

    public Compiler(CompileSession compileSession, IModelBuilder modelBuilder)
    {
        _compileSession = compileSession;
        _modelBuilder = modelBuilder;
        _runPassCount = 0;
    }

    public IRModule Module => _module ?? throw new InvalidOperationException("Module has not been imported");

    /// <inheritdoc/>
    public void ImportIRModule(IRModule module) => _module = module;

    public async Task<IRModule> ImportTFLiteModuleAsync(Stream content)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var module = Importers.ImportTFLite(content, _compileSession);
        return await InitializeModuleAsync(module);
    }

    public async Task<IRModule> ImportOnnxModuleAsync(Stream content)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var module = Importers.ImportOnnx(content, _compileSession);
        return await InitializeModuleAsync(module);
    }

    public async Task<IRModule> ImportNcnnModuleAsync(Stream ncnnParam, Stream ncnnBin)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var module = Importers.ImportNcnn(ncnnParam, ncnnBin, _compileSession);
        return await InitializeModuleAsync(module);
    }

    public void RegisterPreAndPostProcess(IPassManager passManager)
    {
        passManager.Add<AddPreProcess>();
        passManager.Add<AddPostProcess>();
        passManager.AddWithName<DataflowPass>("FoldNopBinary").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldNopBinary>();
        });
    }

    public Task<IRModule> ImportHuggingFaceModuleAsync(string modelDir, ImportOptions importOptions)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var module = Importers.ImportHuggingFace(modelDir, importOptions, _compileSession);
        return InitializeModuleAsync(module);
    }

    public void ProcessAfterImportPass(IPassManager passManager)
    {
        passManager.AddWithName<DataflowPass>("FoldQuantDeQuant").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldQuantDeQuant>();
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
        });
        passManager.AddWithName<DataflowPass>("BroadcastOutputNamesAfterImportPass").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.BroadcastTransposeOutputNames>();
            p.Add<Passes.Rules.Neutral.BroadcastReshapeOutputNames>();
            p.Add<Passes.Rules.Neutral.BroadcastNopPadOutputNames>();
        });
        passManager.Add<OptimizeByRangePass>();
        passManager.Add<ShapeInferPass>();
        RegisterPreAndPostProcess(passManager);
        passManager.Add<AddFunctionToModule>();
    }

    public void TargetIndependentPass(IPassManager passManager)
    {
        passManager.AddWithName<DataflowPass>("NormAxisAndShape").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.ReshapeMatMul>();
            p.Add<Passes.Rules.Neutral.NormAxisGather>();
            p.Add<Passes.Rules.Neutral.NormAxisConcat>();
            p.Add<Passes.Rules.Neutral.NormAxisReduce>();
            p.Add<Passes.Rules.Neutral.NormAxisReshape>();
            p.Add<Passes.Rules.Neutral.NormAxisReduceArg>();
            p.Add<Passes.Rules.Neutral.NormAxisSlice>();
            p.Add<Passes.Rules.Neutral.NormAxisLayernorm>();
            p.Add<Passes.Rules.Neutral.NormAxisSoftmax>();
            p.Add<Passes.Rules.Neutral.SqueezeTransposeShape>();
            p.Add<Passes.Rules.Neutral.Squeeze5DTranspose>();
            p.Add<Passes.Rules.Neutral.SqueezeBinaryShape>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern1>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern2>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern3>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern4>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern5>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern6>();
            p.Add<Passes.Rules.Neutral.ConvertLayerNormChannelFirstToLast>();
            p.Add<Passes.Rules.Neutral.FoldGeluWithScale>();
            p.Add<Passes.Rules.Neutral.FoldGeneralGelu>();
            p.Add<Passes.Rules.Neutral.FoldSwishPattern1>();
            p.Add<Passes.Rules.Neutral.FoldSwishPattern2>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish1>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish2>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish3>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish4>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish5>();
            p.Add<Passes.Rules.Neutral.FoldTwoSlices>();
            p.Add<Passes.Rules.Neutral.FocusFull>();
            p.Add<Passes.Rules.Neutral.ReshapeMatMul>();
            p.Add<Passes.Rules.Neutral.ReshapeExpand>();
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldShapeOf>();
            p.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
            p.Add<Passes.Rules.Neutral.FoldTwoSlices>();
            p.Add<Passes.Rules.Neutral.FoldNopBinary>();
            p.Add<Passes.Rules.Neutral.FoldNopCast>();
            p.Add<Passes.Rules.Neutral.FoldNopReshape>();
            p.Add<Passes.Rules.Neutral.FoldNopSlice>();
            p.Add<Passes.Rules.Neutral.FoldSqueezeUnsqueeze>();
            p.Add<Passes.Rules.Neutral.FoldUnsqueezeSqueeze>();
            p.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
            p.Add<Passes.Rules.Neutral.FoldNopClamp>();
            p.Add<Passes.Rules.ShapeBucket.FoldRepeatMarker>();
            p.Add<Passes.Rules.Neutral.SqueezeToReshape>();
            p.Add<Passes.Rules.Neutral.UnSqueezeToReshape>();
            p.Add<Passes.Rules.ShapeExpr.GatherToGetItem>();
            p.Add<Passes.Rules.ShapeExpr.FoldGetItemShapeOf>();
            p.Add<Passes.Rules.Neutral.FoldGetItemConcat>();
            p.Add<Passes.Rules.Neutral.FoldGetItemReshape>();
            p.Add<Passes.Rules.Neutral.FoldIf>();
            p.Add<Passes.Rules.Neutral.FoldNopReduce>();
            p.Add<Passes.Rules.Neutral.SliceToGetItem>();
            p.Add<Passes.Rules.Neutral.FoldTwoPads>();
            p.Add<Passes.Rules.Neutral.SwapBinaryArgs>();
            p.Add<Passes.Rules.Neutral.FoldDilatedConv2D>();
            p.Add<Passes.Rules.Neutral.PowOf2ToSquare>();
            p.Add<Passes.Rules.Neutral.ScalarConstToTensor>();
            p.Add<Passes.Rules.Neutral.TileToExpand>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormBinary>();
            p.Add<Passes.Rules.Neutral.FoldTwoNopCasts>();
        });

        // passManager.Add<HorizontalMergePass>();

        // Decompose complex ops
        passManager.AddWithName<DataflowPass>("DecomposeComplexOps").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.SwapBinaryArgs>();
            p.Add<Passes.Rules.Neutral.DecomposeInstanceNorm>();
            p.Add<Passes.Rules.Neutral.DecomposeLayerNorm>();
            p.Add<Passes.Rules.Neutral.DecomposeGatedDeltaNet>();
            p.Add<Passes.Rules.Neutral.DecomposeSparseExperts>();
            p.Add<Passes.Rules.Neutral.DecomposeGelu>();
            p.Add<Passes.Rules.Neutral.ScalarConstToTensor>();
        });

        passManager.AddWithName<FoldTopKSoftmaxNormalizationPass>("FoldTopKSoftmaxNormalization");

        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
        passManager.AddWithName<FormQKVRoPEWithCachePass>("FormQKVRoPEWithCache");

        passManager.AddWithName<EGraphRulesPass>("NeutralOptimizeTranspose").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldNopTranspose>();
            p.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
            p.Add<FoldRepeatMarker>();
            p.Add<Passes.Rules.WithMarker.FoldTransposeActTranspose>();
            p.Add<Passes.Rules.WithMarker.FoldTransposeBinaryActTranspose>();
            p.Add<Passes.Rules.WithMarker.CombineReshapePad>();
            p.Add<Passes.Rules.WithMarker.CombinePadTranspose>();
            p.Add<Passes.Rules.Neutral.CombineUnaryTranspose>();
            if (CurrentSession.CompileOptions.ShapeBucketOptions.Enable)
            {
                p.Add<Passes.Rules.WithMarker.CombineTransposePad>();
            }
            else
            {
                p.Add<Passes.Rules.Neutral.CombineTransposePad>();
            }

            p.Add<Passes.Rules.Neutral.CombinePadTranspose>();
            p.Add<Passes.Rules.Neutral.CombineBinaryTranspose>();
            p.Add<Passes.Rules.Neutral.CombineConstBinaryTranspose>();
            p.Add<Passes.Rules.Neutral.CombineTransposeConstBinary>();
            p.Add<Passes.Rules.Neutral.CombineTransposeReduce>();
            p.Add<Passes.Rules.Neutral.CombineTransposeActivations>();
            p.Add<Passes.Rules.Neutral.CombineActivationsTranspose>();
            p.Add<Passes.Rules.Neutral.CombineTransposeConcat>();
            p.Add<Passes.Rules.Neutral.CombineBinaryReshape>();
            p.Add<Passes.Rules.Neutral.CombineConstBinaryReshape>();
            p.Add<Passes.Rules.Neutral.CombineUnaryReshape>();
            p.Add<Passes.Rules.Neutral.CombineActivationsReshape>();
            p.Add<Passes.Rules.Neutral.CombineReshapePad>();
            p.Add<Passes.Rules.Neutral.CombineReshapeTranspose>();
            p.Add<Passes.Rules.Neutral.CombineTransposeReshape>();
            p.Add<Passes.Rules.Neutral.FoldNopPad>();
            p.Add<Passes.Rules.Neutral.FoldConv2DPads>();
            p.Add<Passes.Rules.Neutral.FuseClampConv2D>();
            p.Add<Passes.Rules.Neutral.FoldReduceWindow2DPads>();
            p.Add<Passes.Rules.Neutral.SqueezeToReshape>();
            p.Add<Passes.Rules.Neutral.UnSqueezeToReshape>();
            p.Add<Passes.Rules.Neutral.TransposeToReshape>();
            p.Add<Passes.Rules.Neutral.FlattenToReshape>();
            p.Add<Passes.Rules.Neutral.ReshapeToTranspose>();
            p.Add<Passes.Rules.Neutral.FoldNopReshape>();
            p.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
            p.Add<Passes.Rules.Neutral.FoldReshapeBinaryConstReshape>();
            p.Add<Passes.Rules.Neutral.ReluToClamp>();
            p.Add<Passes.Rules.Neutral.Relu6ToClamp>();
            p.Add<Passes.Rules.Neutral.FoldNopSlice>();
            p.Add<Passes.Rules.Neutral.FoldTwoSlices>();
            p.Add<Passes.Rules.Neutral.SpaceToBatchToPad>();
            p.Add<Passes.Rules.Neutral.FoldConv2DAddMul>();
            p.Add<Passes.Rules.Neutral.CombineSliceTranspose>();
            p.Add<Passes.Rules.Neutral.CombineTwoBinaryMul>();
            p.Add<Passes.Rules.Neutral.FoldTwoBinaryMul>();
        });

        CurrentSession.Target.RegisterTargetInDependentPass(passManager, CurrentSession.CompileOptions);

        passManager.AddWithName<DataflowPass>("BroadcastMarker").Configure(p =>
        {
            p.Add<FoldTransposeActTranspose>();
            p.Add<BroadcastInputMarker>();
            p.Add<BroadcastOutputMarker>();
        });

        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }

    public void QuantizePass(IPassManager passManager)
    {
        var options = CurrentSession.CompileOptions;
        CurrentSession.Target.RegisterQuantizePass(passManager, options);
        if (options.QuantizeOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            passManager.AddWithName<DataflowPass>("RemoveMarker").Configure(p =>
            {
                p.Add<Passes.Rules.Lower.RemoveMarker>();
            });
        }

        CurrentSession.Target.RegisterPostQuantizePass(passManager, options);
    }

    public void HeterogeneousPipelineFormationPass(IPassManager passManager)
    {
        passManager.AddWithName<DataflowPass>("ExposeHeterogeneousPostOps").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.ExposePagedAttentionOutputGate>();
        });
        passManager.Add<AssignModulePlacementPass>();
        passManager.Add<Passes.Transforms.HeterogeneousPipelineFormationPass>();
        passManager.Add<RemoveUnusedFunctions>();
        passManager.Add<InferRangePass>();
    }

    public void AutoPackingPass(IPassManager passManager)
    {
        var target = CurrentSession.Target;
        passManager.AddWithName<DataflowPass>("AutoPacking").Configure(p =>
        {
            target.RegisterAutoPackingRules(p, CurrentSession.CompileOptions);

            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldGetItemTuple>();
            p.Add<Passes.Rules.Neutral.FoldPackTranspose>();
            p.Add<Passes.Rules.Neutral.FoldPackReshape>();
            p.Add<Passes.Rules.Neutral.FoldPackBitcast>();
            p.Add<Passes.Rules.Neutral.FoldBitcastBitcast>();
        });

        passManager.AddWithName<FunctionBoundaryLayoutPropagationPass>("FunctionBoundaryLayoutPropagation");
        passManager.AddWithName<EGraphRulesPass>("PostFunctionBoundaryPackPropagation").Configure(p =>
        {
            target.RegisterPackPropagationRules(p, CurrentSession.CompileOptions);
            p.Add<Passes.Rules.Neutral.UnpackToBitcast>();
            p.Add<Passes.Rules.Neutral.FoldGetItemTuple>();
            p.Add<Passes.Rules.Neutral.FoldPackTranspose>();
            p.Add<Passes.Rules.Neutral.FoldPackReshape>();
            p.Add<Passes.Rules.Neutral.FoldPackBitcast>();
            p.Add<Passes.Rules.Neutral.FoldBitcastBitcast>();
        });

        passManager.AddWithName<MaterializeConstantsPass>("MaterializeFunctionBoundaryConstants");

        passManager.AddWithName<ThreadNormStatsAcrossFunctionBoundariesPass>("ThreadNormStatsAcrossFunctionBoundaries");
        target.RegisterPostAutoPackingPass(passManager, CurrentSession.CompileOptions);
        passManager.Add<RemoveUnusedFunctions>();
        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }

    public void AutoVectorizePass(IPassManager passManager)
    {
        var target = CurrentSession.Target;
        passManager.AddWithName<EGraphRulesPass>("AutoVectorize").Configure(p =>
        {
            target.RegisterAutoVectorizeRules(p, CurrentSession.CompileOptions);
        });

        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();

        target.RegisterPostAutoVectorizePass(passManager, CurrentSession.CompileOptions);
    }

    public void AutoDistributedPass(IPassManager passManager)
    {
        var moduleKinds = CurrentSession.ActiveModuleKind is { } activeModuleKind
            ? new[] { activeModuleKind }
            : CurrentSession.Target.ModuleCompilers.Select(moduleCompiler => moduleCompiler.ModuleKind);
        foreach (var moduleKind in moduleKinds)
        {
            passManager.AddWithName<AutoDistributedWithShapeBucketPass>($"AutoDistributed_{moduleKind}", false, moduleKind);
        }

        passManager.Add<AddFunctionToModule>();
        passManager.Add<RemoveUnusedFunctions>();
        passManager.AddWithName<DataflowPass>("OptimizeAfterAutoDistributed").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldGetItemTuple>();
            p.Add<FoldBoxingBoxing>();
            p.Add<FoldBoxingUninitialized>();
        });

        CurrentSession.Target.RegisterPostAutoDistributedPass(
            passManager,
            CurrentSession.CompileOptions);
        passManager.AddWithName<DataflowPass>("FinalizeAutoDistributedBindings").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldBindNormStats>();
        });
        passManager.Add<RemoveUnusedFunctions>();
        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }

    public void AutoTilingPass(IPassManager passManager)
    {
        var target = CurrentSession.Target;
        target.RegisterAutoTilingPass(passManager, CurrentSession.CompileOptions);
    }

    public void TIRSelectionPass(IPassManager passManager)
    {
        var target = CurrentSession.Target;
        target.RegisterTIRSelectionPass(passManager, CurrentSession.CompileOptions);
        passManager.Add<AddFunctionToModule>();
    }

    public void FinalizeTIRCallGraphPass(IPassManager passManager)
    {
        passManager.AddWithName<RemoveFunctionWrapperPass>("RemoveFunctionWrapper");
        passManager.Add<PropagatePrimFunctionBufferLayoutsPass>();
        passManager.Add<SpecializePrimFunctionBufferLayoutsPass>();
        passManager.Add<RemoveUnusedFunctions>();
        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }

    public void TIRLoweringPass(IPassManager passManager)
    {
        var target = CurrentSession.Target;
        target.RegisterTIRPreBufferizePass(passManager, CurrentSession.CompileOptions);
        RegisterTIRBufferization(passManager);
        target.RegisterTIRPostBufferizePass(passManager, CurrentSession.CompileOptions);
        RegisterTIRCleanup(passManager);
    }

    public void ModuleDispatchTIRLoweringPass(IPassManager passManager)
    {
        RegisterTIRBufferization(passManager);
        RegisterTIRCleanup(passManager);
    }

    public void ModuleDispatchTIRSelectionPass(IPassManager passManager)
    {
        var target = CurrentSession.Target;
        target.RegisterTIRSelectionPass(passManager, CurrentSession.CompileOptions);
        passManager.Add<AddFunctionToModule>();
    }

    public async Task CompileAsync(IProgress<int>? progress = null, CancellationToken token = default)
    {
        Task RunPassAsync(CompileSession session, Action<IPassManager> register, string name)
        {
            return this.RunPassAsync(session, register, name, progress, token);
        }

        var target = _compileSession.Target;
        using var scope = new CompileSessionScope(_compileSession);
        await RunPassAsync(_compileSession, TargetIndependentPass, "TargetIndependentPass");
        await RunPassAsync(_compileSession, TargetIndependQuantPass, "TargetIndependentQuantPass");

        if (target.ModuleCompilers.Count > 1)
        {
            await RunPassAsync(
                _compileSession,
                HeterogeneousPipelineFormationPass,
                "HeterogeneousPipelineFormationPass");
        }

        foreach (var moduleCompiler in target.ModuleCompilers)
        {
            var moduleKind = moduleCompiler.ModuleKind;
            if (!_module!.Functions.Any(function =>
                    string.Equals(function.ModuleKind, moduleKind, StringComparison.Ordinal)))
            {
                continue;
            }

            var moduleTarget = target.GetModuleTarget(moduleKind);
            var moduleOptions = target.GetModuleCompileOptions(moduleKind, _compileSession.CompileOptions);
            using var moduleSession = CompileSession.Create(moduleTarget, moduleOptions, moduleKind);

            await RunPassAsync(
                moduleSession,
                p => moduleTarget.RegisterTargetDependentPass(p, moduleOptions),
                $"{moduleKind}_TargetDependentPass");
            await RunPassAsync(moduleSession, QuantizePass, $"{moduleKind}_QuantizePass");
            await RunPassAsync(moduleSession, AutoVectorizePass, $"{moduleKind}_AutoVectorizePass");
            await RunPassAsync(moduleSession, AutoPackingPass, $"{moduleKind}_AutoPackingPass");
            await RunPassAsync(moduleSession, AutoDistributedPass, $"{moduleKind}_AutoDistributedPass");

            if (moduleTarget.IsAutoTilingEnabled)
            {
                await RunPassAsync(moduleSession, AutoTilingPass, $"{moduleKind}_AutoTilingPass");
            }
        }

        foreach (var moduleCompiler in target.ModuleCompilers)
        {
            var moduleKind = moduleCompiler.ModuleKind;
            if (!_module!.Functions.Any(function =>
                    string.Equals(function.ModuleKind, moduleKind, StringComparison.Ordinal)))
            {
                continue;
            }

            var moduleTarget = target.GetModuleTarget(moduleKind);
            var moduleOptions = target.GetModuleCompileOptions(moduleKind, _compileSession.CompileOptions);
            using var moduleSession = CompileSession.Create(moduleTarget, moduleOptions, moduleKind);
            await RunPassAsync(moduleSession, TIRSelectionPass, $"{moduleKind}_TIRSelectionPass");
        }

        await RunPassAsync(_compileSession, FinalizeTIRCallGraphPass, "FinalizeTIRCallGraphPass");

        foreach (var moduleCompiler in target.ModuleCompilers)
        {
            var moduleKind = moduleCompiler.ModuleKind;
            if (!_module!.Functions.Any(function =>
                    string.Equals(function.ModuleKind, moduleKind, StringComparison.Ordinal)))
            {
                continue;
            }

            var moduleTarget = target.GetModuleTarget(moduleKind);
            var moduleOptions = target.GetModuleCompileOptions(moduleKind, _compileSession.CompileOptions);
            using var moduleSession = CompileSession.Create(moduleTarget, moduleOptions, moduleKind);
            await RunPassAsync(moduleSession, TIRLoweringPass, $"{moduleKind}_TIRLoweringPass");
            await RunPassAsync(
                moduleSession,
                p => moduleTarget.RegisterTargetDependentBeforeCodeGen(p, moduleOptions),
                $"{moduleKind}_TargetDependentBeforeCodeGen");
        }

        if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
        {
            DumpScope.Current.DumpModule(_module!, "ModuleAfterCompile");
        }
    }

    public void Gencode(Stream output)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var linkedModel = _modelBuilder.Build(Module);
        linkedModel.Serialize(output);
    }

    private static void RegisterTIRBufferization(IPassManager passManager)
    {
        passManager.Add<CanonicalizeTIRIndexExpressionsPass>();
        passManager.Add<BufferizePass>();
        passManager.AddWithName<PrimFuncPass>("PeelTiledLoopTails").Configure(p =>
        {
            p.Add<Passes.Mutators.TailLoopPeeling>();
        });
    }

    private static void RegisterTIRCleanup(IPassManager passManager)
    {
        passManager.Add<CanonicalizeTIRIndexExpressionsPass>();

        passManager.AddWithName<PrimFuncPass>("Optimize").Configure(p =>
        {
            p.Add<Passes.Mutators.UnFoldBlock>();
            p.Add<Passes.Mutators.FlattenSequential>();
            p.Add<Passes.Mutators.FoldConstCall>();
            p.Add<Passes.Mutators.EliminateEmptyLoops>();
            p.Add<Passes.Mutators.FlattenBuffer>();
            p.Add<Passes.Mutators.RemoveNop>();
        });
    }

    private async Task<IRModule> InitializeModuleAsync(IRModule module)
    {
        _module = module;

        if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
        {
            DumpScope.Current.DumpModule(module, "IRImport");
        }

        var preprocess_option = _compileSession.CompileOptions;

        await RunPassAsync(pmg => ProcessAfterImportPass(pmg), "ProcessAfterImportPass");

        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        if (!inferSucc)
        {
            throw new InvalidOperationException("Type inference failed");
        }

        return module;
    }

    private void TargetIndependQuantPass(IPassManager passManager)
    {
        var quantMode = _compileSession.CompileOptions.QuantizeOptions.ModelQuantMode;
        if (quantMode == ModelQuantMode.UsePTQ)
        {
            passManager.AddWithName<DataflowPass>("AddRangeOfMarker").Configure(p =>
            {
                p.Add<Passes.Rules.Neutral.AddRangeOfAndMarker>();
            });
            passManager.AddWithName<EGraphPassWithQuantize>("AssignRanges");
        }
    }

    private Task RunPassAsync(
        Action<IPassManager> register,
        string name,
        IProgress<int>? progress = null,
        CancellationToken token = default)
        => RunPassAsync(_compileSession, register, name, progress, token);

    private async Task RunPassAsync(
        CompileSession compileSession,
        Action<IPassManager> register,
        string name,
        IProgress<int>? progress = null,
        CancellationToken token = default)
    {
        var newName = $"{_runPassCount++:D2}_{name}";
        using var scope = new CompileSessionScope(compileSession);
        var pmgr = compileSession.CreatePassManager(newName);
        register(pmgr);
        _ = compileSession.GetService<ILogger<Compiler>>();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        using var animationCts = new CancellationTokenSource();
        var animationTask = ShowPassProgressAnimation(newName, animationCts.Token);

        try
        {
            _module = await pmgr.RunAsync(Module).ConfigureAwait(false);

            animationCts.Cancel();
            await animationTask.ConfigureAwait(false);

            stopwatch.Stop();
            var duration = stopwatch.Elapsed.TotalSeconds;

            ClearCurrentLine();
            var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            var passNamePadded = newName.PadRight(35);
            var timeFormatted = FormatDuration(duration);
            Console.WriteLine($"[{timestamp}] {ColorText("✓", ConsoleColor.Green)} Completed pass: {passNamePadded} {ColorText(timeFormatted, ConsoleColor.Cyan)}");

            if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
            {
                DumpScope.Current.DumpModule(_module, newName);
                DumpScope.Current.DumpDotIR(_module.Entry!, newName);
            }

            progress?.Report(_runPassCount);
            token.ThrowIfCancellationRequested();
        }
        catch (Exception ex)
        {
            animationCts.Cancel();
            await animationTask.ConfigureAwait(false);

            stopwatch.Stop();
            var duration = stopwatch.Elapsed.TotalSeconds;

            ClearCurrentLine();
            var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            var passNamePadded = newName.PadRight(35);
            var timeFormatted = FormatDuration(duration);
            Console.WriteLine($"[{timestamp}] {ColorText("✗", ConsoleColor.Red)} Failed pass: {passNamePadded} {ColorText(timeFormatted, ConsoleColor.Cyan)} - {ColorText(ex.Message, ConsoleColor.Yellow)}");
            throw;
        }
    }

    private async Task ShowPassProgressAnimation(string passName, CancellationToken cancellationToken)
    {
        var spinnerChars = new[] { '⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏' };
        var index = 0;
        var startTime = DateTime.Now;

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var elapsed = DateTime.Now - startTime;
                var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
                var spinner = spinnerChars[index % spinnerChars.Length];
                var passNamePadded = passName.PadRight(35);
                var timeFormatted = FormatDuration(elapsed.TotalSeconds);

                Console.Write($"\r[{timestamp}] {ColorText(spinner.ToString(), ConsoleColor.Blue)} Running pass:   {passNamePadded} {ColorText(timeFormatted, ConsoleColor.Cyan)}");

                index++;
                await Task.Delay(100, default).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
        }
    }

    private string ColorText(string text, ConsoleColor color)
    {
        var colorCode = color switch
        {
            ConsoleColor.Red => "\x1b[31m",
            ConsoleColor.Green => "\x1b[32m",
            ConsoleColor.Yellow => "\x1b[33m",
            ConsoleColor.Blue => "\x1b[34m",
            ConsoleColor.Magenta => "\x1b[35m",
            ConsoleColor.Cyan => "\x1b[36m",
            ConsoleColor.White => "\x1b[37m",
            ConsoleColor.Gray => "\x1b[90m",
            ConsoleColor.DarkRed => "\x1b[91m",
            ConsoleColor.DarkGreen => "\x1b[92m",
            ConsoleColor.DarkYellow => "\x1b[93m",
            ConsoleColor.DarkBlue => "\x1b[94m",
            ConsoleColor.DarkMagenta => "\x1b[95m",
            ConsoleColor.DarkCyan => "\x1b[96m",
            _ => "\x1b[37m",
        };

        return $"{colorCode}{text}\x1b[0m";
    }

    private void ClearCurrentLine()
    {
        try
        {
            Console.Write("\r" + new string(' ', Math.Max(0, Console.WindowWidth - 1)) + "\r");
        }
        catch
        {
            Console.Write("\r" + new string(' ', 120) + "\r");
        }
    }

    private string FormatDuration(double seconds)
    {
        string timeStr;

        var integerPart = ((int)seconds).ToString().PadLeft(8);
        var decimalPart = $"{seconds % 1:F3}".Substring(1);
        timeStr = $"{integerPart}{decimalPart}s";

        return $"({timeStr})";
    }
}
