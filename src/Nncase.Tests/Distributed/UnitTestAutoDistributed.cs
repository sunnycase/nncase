// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.Evaluator.IR.NTT;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.IR.Shapes;
using Nncase.Passes.Distributed;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using QuikGraph;
using Xunit;

namespace Nncase.Tests.DistributedTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestDistribAutoDistributed : TestClassBase
{
    public UnitTestDistribAutoDistributed()
    {
        DefaultTargetName = CPUTarget.Kind;
        CompileOptions.TargetOptions = new NTTTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.EGraphCost | DumpFlags.CodeGen | DumpFlags.Compile;
#endif
    }

    [Fact]
    public void TestDistributeBinary()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [32, 1]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [16]));
        var main = new Function("main", lhs + rhs, [lhs, rhs]);
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);
        pass.RunAsync(main, new()).Wait();
    }

    [Fact]
    public void TestDistributeDynamicBinaryWithRhsVector()
    {
        var dimX = new DimVar("dimX") { Metadata = { Range = (1, 256) } };
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [dimX, 1]));
        var rhs = new Var("rhs", new TensorType(new VectorType(DataTypes.Float32, [8]), [16]));
        var main = new Function("main", lhs + rhs, [lhs, rhs]);
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);
        pass.RunAsync(main, new()).Wait();
    }

    [Fact]
    public void TestBinaryCandidateProviderPropagatesExactProducerSplit()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(1, 256));
        var placement = new Placement([4, 8], "yx", "bb");
        var broadcastType = new DistributedType(tensorType, [SBP.B, SBP.B], placement);
        var canonicalSplitType = new DistributedType(
            tensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 8)],
            placement);
        var producerSplitType = new DistributedType(
            tensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var lhs = new Var("lhs", tensorType);
        var rhs = new Var("rhs", tensorType);
        var sourceCall = Assert.IsType<Call>(IR.F.Math.Add(lhs, rhs));
        Assert.True(sourceCall.InferenceType());
        var context = new DistributedCandidateContext(
            CompileOptions,
            options,
            PyNTTTarget.Kind,
            sourceCall,
            [
                new IRType[] { broadcastType, canonicalSplitType },
                new IRType[] { producerSplitType },
            ]);
        var provider = new BinaryCandidateProvider();
        var target = Assert.IsType<Binary>(sourceCall.Target);

        var returnTypes = provider.GetReturnCandidateTypes(
            context,
            target,
            [broadcastType, canonicalSplitType]);
        Assert.Contains(producerSplitType, returnTypes);
        Assert.True(provider.TryGetInputTypeTuples(
            context,
            target,
            producerSplitType,
            out var tuples));
        var tuple = Assert.Single(tuples);
        Assert.Equal(producerSplitType, tuple.InputTypes[0]);
        Assert.Equal(producerSplitType, tuple.InputTypes[1]);
    }

    [Fact]
    public void TestNormApplyCandidateProviderPreservesExactInputSplit()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var inputTensorType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(1, 256));
        var parameterTensorType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(256));
        var placement = new Placement([4, 8], "yx", "bb");
        var broadcastInputType = new DistributedType(inputTensorType, [SBP.B, SBP.B], placement);
        var canonicalInputType = new DistributedType(
            inputTensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 8)],
            placement);
        var exactInputType = new DistributedType(
            inputTensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var partialStatsType = Assert.IsType<DistributedType>(
            NormStatsEvaluator.InferType(new NormStats(1, false), exactInputType));
        var materializedStatsType = partialStatsType with { Partial = null };
        var broadcastParameterType = new DistributedType(parameterTensorType, [SBP.B], placement);

        var input = new Var("input", inputTensorType);
        var stats = new Var("stats", partialStatsType.TensorType);
        var scale = new Var("scale", parameterTensorType);
        var bias = new Var("bias", parameterTensorType);
        var sourceCall = IR.F.NN.NormApply(1, 1e-6f, input, stats, scale, bias, useMean: false);
        var context = new DistributedCandidateContext(
            CompileOptions,
            options,
            PyNTTTarget.Kind,
            sourceCall,
            [
                new IRType[] { broadcastInputType, canonicalInputType, exactInputType },
                new IRType[] { partialStatsType },
                new IRType[] { broadcastParameterType },
                new IRType[] { broadcastParameterType },
            ]);
        var provider = new NormApplyCandidateProvider();
        var target = Assert.IsType<NormApply>(sourceCall.Target);

        var returnTypes = provider.GetReturnCandidateTypes(
            context,
            target,
            [broadcastInputType, canonicalInputType]);
        Assert.Contains(exactInputType, returnTypes);
        Assert.True(provider.TryGetInputTypeTuples(
            context,
            target,
            exactInputType,
            out var tuples));
        var tuple = Assert.Single(tuples);
        Assert.Equal(exactInputType, tuple.InputTypes[NormApply.Input.Index]);
        Assert.Equal(materializedStatsType, tuple.InputTypes[NormApply.Stats.Index]);

        var scaleType = Assert.IsType<DistributedType>(tuple.InputTypes[NormApply.Scale.Index]);
        var biasType = Assert.IsType<DistributedType>(tuple.InputTypes[NormApply.Bias.Index]);
        var expectedParameterPolicy = SBP.SBlockCyclic([0, 1], 1);
        Assert.Equal(expectedParameterPolicy, Assert.Single(scaleType.AxisPolicies.ToArray()));
        Assert.Equal(expectedParameterPolicy, Assert.Single(biasType.AxisPolicies.ToArray()));

        var distributedCall = new Call(
            target,
            new Var("distributed_input", exactInputType),
            new Var("materialized_stats", materializedStatsType),
            new Var("distributed_scale", scaleType),
            new Var("distributed_bias", biasType));
        Assert.True(distributedCall.InferenceType());
        Assert.Equal(exactInputType, distributedCall.CheckedType);
    }

    [Fact]
    public void TestBindNormStatsCandidateAcceptsMaterializedPartialStats()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var placement = new Placement([4, 8], "yx", "bb");
        var inputTensorType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(1, 256));
        var inputType = new DistributedType(
            inputTensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var normStats = new NormStats(1, false);
        var partialStatsType = Assert.IsType<DistributedType>(
            NormStatsEvaluator.InferType(normStats, inputType));
        Assert.NotNull(partialStatsType.Partial);
        var materializedStatsType = partialStatsType with { Partial = null };

        var input = new Var("input", inputTensorType);
        var stats = new Var("stats", partialStatsType.TensorType);
        var sourceCall = Assert.IsType<Call>(IR.F.NN.BindNormStats(1, input, stats, false));
        var context = new DistributedCandidateContext(
            CompileOptions,
            options,
            PyNTTTarget.Kind,
            sourceCall,
            [
                new IRType[] { inputType },
                new IRType[] { materializedStatsType },
            ]);
        var provider = new BindNormStatsCandidateProvider();
        var target = Assert.IsType<BindNormStats>(sourceCall.Target);

        Assert.True(provider.TryGetInputTypeTuples(
            context,
            target,
            materializedStatsType,
            out var tuples));
        var tuple = Assert.Single(tuples);
        Assert.Equal(inputType, tuple.InputTypes[BindNormStats.Input.Index]);
        Assert.Equal(materializedStatsType, tuple.InputTypes[BindNormStats.Stats.Index]);

        var distributedCall = new Call(
            target,
            new Var("distributed_input", inputType),
            new Var("materialized_stats", materializedStatsType));
        Assert.True(distributedCall.InferenceType());
        Assert.Equal(materializedStatsType, distributedCall.CheckedType);
    }

    [Fact]
    public void TestPackedMatMulCandidateProviderRejectsPartialOutputWithFullAddend()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var placement = new Placement([4, 8], "yx", "bb");
        var lhsTensorType = new TensorType(DataTypes.BFloat16, new long[] { 1, 2048 });
        var rhsTensorType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8, 2, 8]),
            new long[] { 128, 256 });
        var lhsType = new DistributedType(
            lhsTensorType,
            new SBP[] { SBP.B, SBP.SBlockCyclic([1], 256) },
            placement);
        var rhsType = new DistributedType(
            rhsTensorType,
            new SBP[] { SBP.SBlockCyclic([1], 16), SBP.SBlockCyclic([0], 8) },
            placement);
        var partialOutput = Assert.IsType<DistributedType>(PackedMatMulEvaluator.InferType(
            new PackedMatMul(DataTypes.BFloat16, false, PackedMatMulRhsLayout.KMajor),
            lhsType,
            rhsType,
            NoneType.Default,
            NoneType.Default));
        Assert.NotNull(partialOutput.Partial);
        var fullAddendType = new DistributedType(
            partialOutput.TensorType,
            Enumerable.Repeat<SBP>(SBP.B, partialOutput.TensorType.Shape.Rank).ToArray(),
            placement);

        var lhs = new Var("lhs", lhsTensorType);
        var rhs = new Var("rhs", rhsTensorType);
        var output = IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor);
        var addend = new Var("addend", output.CheckedType);
        var sourceCall = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor,
            addend: addend));
        var context = new DistributedCandidateContext(
            CompileOptions,
            options,
            PyNTTTarget.Kind,
            sourceCall,
            [
                new IRType[] { lhsType },
                new IRType[] { rhsType },
                new IRType[] { NoneType.Default },
                new IRType[] { fullAddendType },
            ]);
        var provider = new PackedMatMulCandidateProvider();
        var target = Assert.IsType<PackedMatMul>(sourceCall.Target);

        var returnTypes = provider.GetReturnCandidateTypes(context, target, []);

        Assert.DoesNotContain(
            returnTypes,
            type => type is DistributedType { Partial: not null });
    }

    [Fact]
    public void TestPagedAttentionPartialCombineDistributionContract()
    {
        var options = new PyNTTTargetOptions();
        CompileOptions.TargetOptions = options;
        var placement = new Placement([4, 8], "yx", "bb");
        var layout = new IRArray<AttentionDimKind>(
            new[] { AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq });
        var queryTensorType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(16, 16, 1));
        var queryType = new DistributedType(
            queryTensorType,
            [SBP.SBlockCyclic([1], 2), SBP.B, SBP.B],
            placement);
        var extraTensorType = new TensorType(DataTypes.UInt8, new RankedShape(8_404_992));
        var extraType = new DistributedType(
            extraTensorType,
            [SBP.SBlockCyclic([0, 1], 128)],
            placement);
        var cacheConfig = new PagedAttentionConfig(
            1,
            8,
            128,
            DataTypes.BFloat16,
            256,
            [
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.HeadDim,
                PagedKVCacheDimKind.BlockSize,
            ],
            [
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.BlockSize,
                PagedKVCacheDimKind.HeadDim,
            ],
            [PagedKVCacheDimKind.HeadDim],
            [PagedKVCacheDimKind.BlockSize],
            [8],
            [8],
            [PagedKVCacheDimKind.NumBlocks],
            [SBP.SContiguous([0, 1])]);
        var cacheTensorType = TensorType.Scalar(new ReferenceType(
            new PagedAttentionKVCacheType { Config = cacheConfig }));
        var scaleTensorType = TensorType.Scalar(DataTypes.BFloat16);
        var partialTarget = new PagedAttentionPartial(layout, 2048, 0, 4);
        var partialType = Assert.IsType<TupleType>(PagedAttentionPartialEvaluator.InferType(
            partialTarget,
            queryType,
            cacheTensorType,
            extraType,
            scaleTensorType,
            new DimensionType(DimensionKind.Fixed)));
        var partialMaxType = Assert.IsType<DistributedType>(partialType[0]);
        var partialSumType = Assert.IsType<DistributedType>(partialType[1]);
        var partialAccType = Assert.IsType<DistributedType>(partialType[2]);
        Assert.Equal(queryTensorType.Shape.Rank, partialMaxType.TensorType.Shape.Rank);
        Assert.Equal(queryTensorType.Shape.Rank, partialAccType.TensorType.Shape.Rank);
        Assert.Equal(SBP.P([0], ReduceOp.Max), partialMaxType.Partial);
        Assert.Equal(SBP.P([0], ReduceOp.Sum), partialSumType.Partial);
        Assert.Equal(SBP.P([0], ReduceOp.Sum), partialAccType.Partial);
        Assert.Equal(queryType.AxisPolicies, partialMaxType.AxisPolicies);
        Assert.Equal(queryType.AxisPolicies, partialSumType.AxisPolicies);
        Assert.Equal(queryType.AxisPolicies, partialAccType.AxisPolicies);

        var combineTarget = new PagedAttentionCombine(
            layout,
            2048,
            queryTensorType.DType,
            queryType,
            0,
            4);
        var combinedType = Assert.IsType<DistributedType>(
            PagedAttentionCombineEvaluator.InferType(
                combineTarget,
                partialMaxType,
                partialSumType,
                partialAccType));
        Assert.Equal(queryType, combinedType);

        var broadcastQueryType = new DistributedType(
            queryTensorType,
            [SBP.B, SBP.B, SBP.B],
            placement);
        var broadcastPartialType = Assert.IsType<TupleType>(
            PagedAttentionPartialEvaluator.InferType(
                partialTarget,
                broadcastQueryType,
                cacheTensorType,
                extraType,
                scaleTensorType,
                new DimensionType(DimensionKind.Fixed)));
        var splitOutputType = new DistributedType(
            queryTensorType,
            [SBP.SBlockCyclic([0], 2), SBP.B, SBP.B],
            placement);
        var splitCombineTarget = new PagedAttentionCombine(
            layout,
            2048,
            queryTensorType.DType,
            splitOutputType,
            0,
            4);
        Assert.Equal(
            splitOutputType,
            PagedAttentionCombineEvaluator.InferType(
                splitCombineTarget,
                broadcastPartialType[0],
                broadcastPartialType[1],
                broadcastPartialType[2]));

        var splitDimOutputType = new DistributedType(
            queryTensorType,
            [SBP.SBlockCyclic([1], 2), SBP.SBlockCyclic([0], 4), SBP.B],
            placement);
        var splitDimCombineTarget = new PagedAttentionCombine(
            layout,
            2048,
            queryTensorType.DType,
            splitDimOutputType,
            0,
            4);
        Assert.Equal(
            splitDimOutputType,
            PagedAttentionCombineEvaluator.InferType(
                splitDimCombineTarget,
                partialMaxType,
                partialSumType,
                partialAccType));

        var invalidOutputType = new DistributedType(
            queryTensorType,
            [SBP.B, SBP.SBlockCyclic([1], 2), SBP.B],
            placement);
        Assert.IsType<InvalidType>(PagedAttentionCombineEvaluator.InferType(
            new PagedAttentionCombine(
                layout,
                2048,
                queryTensorType.DType,
                invalidOutputType,
                0,
                4),
            partialMaxType,
            partialSumType,
            partialAccType));
    }

    [Fact]
    public void TestSamplingPartialCombineDistributionContract()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var config = new SamplerConfig(
            vocabSize: 1024,
            maxBatchSize: 1,
            maxLogprobs: 4,
            SamplerLogprobsMode.RawLogprobs);
        var stateType = TensorType.Scalar(
            new ReferenceType(new SamplerStateType { Config = config }));
        var logitsTensorType = new TensorType(DataTypes.BFloat16, new RankedShape(1, 1024));
        var logitsType = new DistributedType(
            logitsTensorType,
            [SBP.B, SBP.SContiguous([0, 1])],
            placement);

        var partialTarget = new SamplingPartial(config);
        var partialType = Assert.IsType<TupleType>(
            SamplingPartialEvaluator.InferType(partialTarget, logitsType, stateType));
        Assert.Equal(
            new DistributedType(
                new TensorType(DataTypes.Float32, logitsTensorType.Shape),
                logitsType.AxisPolicies,
                placement),
            partialType[0]);

        var argMaxStateType = Assert.IsType<DistributedType>(partialType[1]);
        Assert.Equal(new RankedShape(1), argMaxStateType.TensorType.Shape);
        Assert.Equal(DataTypes.UInt64, argMaxStateType.TensorType.DType);
        Assert.Equal(new SBP[] { SBP.B }, argMaxStateType.AxisPolicies);
        Assert.Equal(SBP.P([0, 1], ReduceOp.Max), argMaxStateType.Partial);

        var combineTarget = new SamplingCombine(config);
        var resultType = Assert.IsType<TupleType>(SamplingCombineEvaluator.InferType(
            combineTarget,
            logitsType,
            partialType[0],
            argMaxStateType,
            stateType));
        Assert.Equal(6, resultType.Count);
        Assert.All(
            resultType.Fields.Take(5),
            field => Assert.All(
                Assert.IsType<DistributedType>(field).AxisPolicies,
                policy => Assert.Equal(SBP.B, policy)));
        Assert.Equal(stateType, resultType[5]);

        var nonPartialStateType = new DistributedType(
            argMaxStateType.TensorType,
            [SBP.B],
            placement);
        Assert.IsType<InvalidType>(SamplingCombineEvaluator.InferType(
            combineTarget,
            logitsType,
            partialType[0],
            nonPartialStateType,
            stateType));
    }

    [Fact]
    public void TestAutoDistributedPropagatesPackedMatMulSplitThroughAdd()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var lhs = new Var(
            "lhs",
            new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var rhs = new Var(
            "rhs",
            new TensorType(
                new VectorType(DataTypes.BFloat16, [8, 2, 8]),
                new RankedShape(4, 16)));
        var packedMatMul = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: IR.NTT.PackedMatMulRhsLayout.KMajor));
        var residual = new Var("residual", packedMatMul.CheckedType);
        var main = new Function(
            "main",
            IR.F.Math.Add(residual, packedMatMul),
            [lhs, rhs, residual]);
        Assert.True(main.InferenceType());

        var post = Assert.IsType<Function>(
            new AutoDistributedPass(false, PyNTTTarget.Kind, CompileOptions)
                .RunAsync(main, new()).Result);

        var add = Assert.Single(
            ExprCollector.Collect(post.Body)
                .OfType<Call>()
                .Where(call => call.Target is Binary { BinaryOp: BinaryOp.Add }));
        var outputType = Assert.IsType<DistributedType>(add.CheckedType);
        var split = Assert.IsType<SBPSplit>(outputType.AxisPolicies[1]);
        Assert.NotEmpty(split.HierarchyAxes);
        Assert.Equal(1, Assert.IsType<BlockCyclicSplit>(Assert.Single(split.Stages).Distribution).BlockSize);
        Assert.All(
            add.Arguments.ToArray(),
            argument => Assert.Equal(outputType, argument.CheckedType));
        Assert.Contains(
            add.Arguments.ToArray(),
            argument => argument is Call { Target: IR.Distributed.ShardedView });
    }

    [Fact]
    public void TestFunctionTupleCallGetItemCanBeDistributed()
    {
        var inputType = new TensorType(DataTypes.Float32, [16, 32]);
        var input = new Var("input", inputType);
        var layerInput = new Var("layer_input", inputType);
        var layer = new Function("layer", new IR.Tuple(layerInput, layerInput), [layerInput]);
        var layerCall = new Call(layer, input);
        var output = IR.F.Tensors.GetItem(layerCall, 0) + input;
        var main = new Function("main", output, [input]);
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);

        var post = pass.RunAsync(main, new()).Result;

        Assert.NotNull(post);
    }

    [Fact]
    public void TestStaticInvocationCountsAccumulateAcrossRepeatedFunctionCalls()
    {
        var inputType = new TensorType(DataTypes.Float32, [16, 32]);
        var leafInput = new Var("leaf_input", inputType);
        var leaf = new Function("leaf", IR.F.Math.Unary(UnaryOp.Abs, leafInput), [leafInput]);

        var layerInput = new Var("layer_input", inputType);
        var layerCall0 = new Call(leaf, layerInput);
        var layerCall1 = new Call(leaf, layerCall0);
        var layer = new Function("layer", layerCall1, [layerInput]);

        var input = new Var("input", inputType);
        var mainCall0 = new Call(layer, input);
        var mainCall1 = new Call(layer, mainCall0);
        var main = new Function("main", mainCall1, [input]);
        Assert.True(main.InferenceType());

        var reachable = DistributedFunctionGraphUtility.GetReachableFunctionsInCalleeFirstOrder(main);
        var invocationCounts = DistributedFunctionGraphUtility.GetStaticInvocationCounts(main, reachable);

        Assert.Equal(1L, invocationCounts[main]);
        Assert.Equal(2L, invocationCounts[layer]);
        Assert.Equal(4L, invocationCounts[leaf]);
    }

    [Fact]
    public void TestFunctionCallUsesDistributedParameterSignature()
    {
        var inputType = new TensorType(DataTypes.Float32, [16, 32]);
        var input = new Var("input", inputType);
        var layerInput = new Var("layer_input", inputType);
        var layer = new Function("layer", IR.F.Math.Unary(UnaryOp.Abs, layerInput), [layerInput]);
        Assert.True(layer.InferenceType());

        var main = new Function("main", new Call(layer, input), [input]);
        Assert.True(main.InferenceType());
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);

        var post = Assert.IsType<Function>(pass.RunAsync(main, new()).Result);

        var layerCall = Assert.Single(ExprCollector.Collect(post.Body).OfType<Call>().Where(call => call.Target is Function { Name: "layer" }));
        var rewrittenLayer = Assert.IsType<Function>(layerCall.Target);
        var parameter = Assert.IsType<Var>(Assert.Single(rewrittenLayer.Parameters.ToArray()));
        Assert.IsType<DistributedType>(parameter.CheckedType);
        var argument = Assert.Single(layerCall.Arguments.ToArray());
        Assert.True(
            EqualityComparer<IRType>.Default.Equals(argument.CheckedType, parameter.CheckedType),
            $"Function call ABI mismatch: argument is {argument.CheckedType}, parameter is {parameter.CheckedType}.");
        Assert.DoesNotContain("Boxing(", CompilerServices.Print(rewrittenLayer.Body), System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestFunctionBoundarySelectsConsistentShardedSignature()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(DataTypes.Float32, [256]);
        var placement = new Placement([4, 8], "yx", "bb");
        var producerType = new DistributedType(
            tensorType,
            [SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var producer = new TensorConst(
            Tensor.FromScalar(1.0f, [256]),
            producerType.AxisPolicies,
            producerType.Placement);
        var layerInput = new Var("layer_input", tensorType);
        var layer = new Function("layer", IR.F.Math.Unary(UnaryOp.Abs, layerInput), [layerInput]);
        Assert.True(layer.InferenceType());

        var main = new Function("main", new Call(layer, producer), Array.Empty<Var>());
        Assert.True(main.InferenceType());
        var pass = new AutoDistributedPass(false, PyNTTTarget.Kind, CompileOptions);

        var post = Assert.IsType<Function>(pass.RunAsync(main, new()).Result);

        var layerCall = Assert.Single(
            ExprCollector.Collect(post.Body).OfType<Call>().Where(call => call.Target is Function { Name: "layer" }));
        var rewrittenLayer = Assert.IsType<Function>(layerCall.Target);
        var rewrittenParameter = Assert.IsType<Var>(Assert.Single(rewrittenLayer.Parameters.ToArray()));
        var parameterType = Assert.IsType<DistributedType>(rewrittenParameter.CheckedType);
        Assert.Contains(parameterType.AxisPolicies, policy => policy is SBPSplit);
        var argument = Assert.Single(layerCall.Arguments.ToArray());
        Assert.Equal(parameterType, argument.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(argument).OfType<Call>(),
            call => call.Target is IR.Distributed.ShardedView { NewType: var targetType }
                && EqualityComparer<IRType>.Default.Equals(call.Arguments[0].CheckedType, targetType));
    }

    [Fact]
    public void TestFunctionBoundaryDemandPropagatesThroughWhereAndGather()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var placement = new Placement([4, 8], "yx", "bb");
        var outputTensorType = new TensorType(DataTypes.Float32, [1, 256]);
        var exactType = new DistributedType(
            outputTensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);

        var index = new Var("index", new TensorType(DataTypes.Int64, [1]));
        var table = Tensor.FromScalar(1.0f, [64, 256]);
        var gathered = IR.F.Tensors.Gather(table, 0, index);
        var where = IR.F.Tensors.Where(
            Tensor.FromScalar(true, [1, 1]),
            Tensor.FromScalar(0.0f, [1, 1]),
            gathered);
        Assert.Equal(outputTensorType, where.CheckedType);

        var layerInput = new Var("layer_input", outputTensorType);
        var layer = new Function("layer", IR.F.Math.Unary(UnaryOp.Abs, layerInput), [layerInput]);
        var exactProducer0 = new TensorConst(
            Tensor.FromScalar(2.0f, [1, 256]),
            exactType.AxisPolicies,
            exactType.Placement);
        var exactProducer1 = new TensorConst(
            Tensor.FromScalar(3.0f, [1, 256]),
            exactType.AxisPolicies,
            exactType.Placement);
        var whereLayer = new Call(layer, where);
        var exactLayer0 = new Call(layer, exactProducer0);
        var exactLayer1 = new Call(layer, exactProducer1);
        var main = new Function(
            "main",
            IR.F.Math.Add(IR.F.Math.Add(whereLayer, exactLayer0), exactLayer1),
            [index]);
        Assert.True(main.InferenceType());

        var post = Assert.IsType<Function>(
            new AutoDistributedPass(false, PyNTTTarget.Kind, CompileOptions)
                .RunAsync(main, new()).Result);

        var layerCalls = ExprCollector.Collect(post.Body)
            .OfType<Call>()
            .Where(call => call.Target is Function { Name: "layer" })
            .ToArray();
        Assert.Equal(3, layerCalls.Length);
        var rewrittenLayer = Assert.IsType<Function>(layerCalls[0].Target);
        var parameter = Assert.IsType<Var>(Assert.Single(rewrittenLayer.Parameters.ToArray()));
        Assert.Equal(exactType, parameter.CheckedType);
        Assert.All(layerCalls, call => Assert.Equal(exactType, Assert.Single(call.Arguments.ToArray()).CheckedType));

        var selectedWhere = Assert.Single(
            ExprCollector.Collect(post.Body).OfType<Call>().Where(call => call.Target is IR.Tensors.Where));
        Assert.Equal(exactType, selectedWhere.CheckedType);
        var selectedGather = Assert.Single(
            ExprCollector.Collect(selectedWhere).OfType<Call>().Where(call => call.Target is IR.Tensors.Gather));
        Assert.Equal(exactType, selectedGather.CheckedType);
        var whereLayerCall = Assert.Single(layerCalls.Where(call =>
            ExprCollector.Collect(Assert.Single(call.Arguments.ToArray()))
                .OfType<Call>()
                .Any(argumentCall => argumentCall.Target is IR.Tensors.Where)));
        Assert.Same(selectedWhere, Assert.Single(whereLayerCall.Arguments.ToArray()));
    }

    [Fact]
    public void TestLateFunctionSignatureCandidateConnectsDerivedParameterUses()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var lhsType = new TensorType(DataTypes.BFloat16, new RankedShape(1, 64));
        var rhsType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8, 2, 8]),
            new RankedShape(4, 256));
        var layerLhs = new Var("layer_lhs", lhsType);
        var layerRhs = new Var("layer_rhs", rhsType);
        var packed = IR.F.NTT.PackedMatMul(
            layerLhs,
            layerRhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor);
        var tensorType = Assert.IsType<TensorType>(packed.CheckedType);
        var placement = new Placement([4, 8], "yx", "bb");
        var exactType = new DistributedType(
            tensorType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);
        var layerInput = new Var("layer_input", tensorType);
        var layer = new Function(
            "layer",
            IR.F.NTT.PackedMatMulNormStats(
                layerLhs,
                layerRhs,
                DataTypes.BFloat16,
                PackedMatMulRhsLayout.KMajor,
                axis: 1,
                useMean: false,
                addend: layerInput),
            [layerLhs, layerRhs, layerInput]);
        var mainLhs = new Var("main_lhs", lhsType);
        var mainRhs = new Var("main_rhs", rhsType);
        var broadcast = new TensorConst(
            Tensor.FromScalar((BFloat16)1.0f, tensorType.Shape.ToValueArray()));
        var exact = IR.F.NTT.PackedMatMul(
            mainLhs,
            mainRhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor);
        var main = new Function(
            "main",
            new IR.Tuple(
                new Call(layer, mainLhs, mainRhs, broadcast),
                new Call(layer, mainLhs, mainRhs, exact),
                new Call(layer, mainLhs, mainRhs, broadcast)),
            [mainLhs, mainRhs]);
        Assert.True(main.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind);
        var buildFunctionGraph = typeof(AutoDistributedRewriter).GetMethod(
            "BuildFunctionSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(buildFunctionGraph);
        _ = buildFunctionGraph!.Invoke(rewriter, [layer, false]);
        _ = buildFunctionGraph.Invoke(rewriter, [main, true]);
        var graphField = typeof(AutoDistributedRewriter).GetField(
            "_rootSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var graph = Assert.IsType<DistributedSearchGraph>(graphField?.GetValue(rewriter));
        var formalNodes = graph.Vertices.Where(node =>
            node.Kind == SearchableNodeKind.FunctionParameter
            && ReferenceEquals(node.OriginParameter, layerInput)).ToArray();
        var exactFormal = formalNodes.SingleOrDefault(
            node => EqualityComparer<IRType>.Default.Equals(node.IRType, exactType));
        Assert.True(
            exactFormal is not null,
            $"Expected {exactType}, but formal candidates were:{Environment.NewLine}{string.Join(Environment.NewLine, formalNodes.Select(node => node.IRType))}");
        var callNodes = graph.Vertices.Where(node =>
            node.Kind == SearchableNodeKind.FunctionCall
            && ReferenceEquals(node.Expr, layer)).ToArray();
        Assert.NotEmpty(callNodes);

        var exactCallCandidates = callNodes.Where(callNode =>
            graph.TryGetOutEdges(callNode, out var callEdges)
            && callEdges.Any(edge => edge.InputIndex == 2
                && edge.Target.Kind == SearchableNodeKind.FunctionBoundaryAdapter
                && EqualityComparer<IRType>.Default.Equals(edge.Target.IRType, exactType)
                && graph.TryGetOutEdges(edge.Target, out var adapterEdges)
                && adapterEdges.Any(adapterEdge => ReferenceEquals(adapterEdge.Target, exactFormal)))).ToArray();
        Assert.NotEmpty(exactCallCandidates);

        Assert.Contains(
            graph.Vertices,
            node => node.Kind == SearchableNodeKind.TypeAdapter
                && ReferenceEquals(node.OriginParameter, exactFormal.OriginParameter)
                && EqualityComparer<IRType>.Default.Equals(node.IRType, exactType)
                && graph.TryGetOutEdges(node, out var edges)
                && edges.Any(edge => ReferenceEquals(edge.Target, exactFormal)));
    }

    [Fact]
    public void TestLateFunctionSignaturePropagatesThroughNormApplyProvider()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var lhsType = new TensorType(DataTypes.BFloat16, new RankedShape(1, 64));
        var rhsType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8, 2, 8]),
            new RankedShape(4, 256));
        var hiddenType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(1, 256));
        var statsType = new TensorType(DataTypes.Float32, new RankedShape(1, 1, 1));
        var parameterType = new TensorType(
            new VectorType(DataTypes.BFloat16, [8]),
            new RankedShape(256));
        var placement = new Placement([4, 8], "yx", "bb");
        var exactType = new DistributedType(
            hiddenType,
            [SBP.B, SBP.SBlockCyclic([0, 1], 1)],
            placement);

        var layerHidden = new Var("layer_hidden", hiddenType);
        var layerStats = new Var("layer_stats", statsType);
        var layerScale = new Var("layer_scale", parameterType);
        var layerBias = new Var("layer_bias", parameterType);
        var layer = new Function(
            "layer",
            IR.F.NN.NormApply(
                axis: 1,
                epsilon: 1e-6f,
                layerHidden,
                layerStats,
                layerScale,
                layerBias,
                useMean: false),
            [layerHidden, layerStats, layerScale, layerBias]);
        Assert.True(layer.InferenceType());

        var mainLhs = new Var("main_lhs", lhsType);
        var mainRhs = new Var("main_rhs", rhsType);
        var mainStats = new Var("main_stats", statsType);
        var mainScale = new Var("main_scale", parameterType);
        var mainBias = new Var("main_bias", parameterType);
        var packed = IR.F.NTT.PackedMatMul(
            mainLhs,
            mainRhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor);
        Assert.Equal(hiddenType, packed.CheckedType);
        var main = new Function(
            "main",
            new Call(layer, packed, mainStats, mainScale, mainBias),
            [mainLhs, mainRhs, mainStats, mainScale, mainBias]);
        Assert.True(main.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind);
        var buildFunctionGraph = typeof(AutoDistributedRewriter).GetMethod(
            "BuildFunctionSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var propagateCandidateClosure = typeof(AutoDistributedRewriter).GetMethod(
            "PropagateCandidateClosure",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(buildFunctionGraph);
        Assert.NotNull(propagateCandidateClosure);
        _ = buildFunctionGraph!.Invoke(rewriter, [layer, false]);
        _ = buildFunctionGraph.Invoke(rewriter, [main, true]);
        _ = propagateCandidateClosure!.Invoke(rewriter, null);

        var graphField = typeof(AutoDistributedRewriter).GetField(
            "_rootSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var graph = Assert.IsType<DistributedSearchGraph>(graphField?.GetValue(rewriter));
        var exactNormApply = graph.Vertices.SingleOrDefault(node =>
            node.Expr is NormApply
            && EqualityComparer<IRType>.Default.Equals(node.IRType, exactType));
        Assert.True(exactNormApply is not null, "The exact function signature must reach the NormApply candidate domain.");
        Assert.True(graph.TryGetOutEdges(exactNormApply!, out var normApplyEdges));
        var hiddenEdge = Assert.Single(normApplyEdges.Where(edge => edge.InputIndex == NormApply.Input.Index));
        Assert.Equal(exactType, hiddenEdge.InputGraph.Vertices.First().IRType);
        Assert.Contains(
            hiddenEdge.InputGraph.Vertices,
            node => node.Kind == SearchableNodeKind.TypeAdapter
                && ReferenceEquals(node.OriginParameter, layerHidden));
    }

    [Fact]
    public void TestFunctionBoundaryDoesNotForceBroadcastFromUpstreamGetItem()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(DataTypes.Float32, [65536]);
        var placement = new Placement([4, 8], "yx", "bb");
        var broadcastType = new DistributedType(tensorType, [SBP.B], placement);
        var sourceValue = new TensorConst(
            Tensor.FromScalar(1.0f, [65536]),
            broadcastType.AxisPolicies,
            broadcastType.Placement);
        var source = new Function("source", new IR.Tuple(sourceValue), Array.Empty<Var>());
        Assert.True(source.InferenceType());

        var layerInput = new Var("layer_input", tensorType);
        var layer = new Function("layer", IR.F.Math.Unary(UnaryOp.Abs, layerInput), [layerInput]);
        Assert.True(layer.InferenceType());

        BaseExpr value = IR.F.Tensors.GetItem(new Call(source), 0);
        for (var index = 0; index < 8; index++)
        {
            value = new Call(layer, value);
        }

        var main = new Function("main", value, Array.Empty<Var>());
        Assert.True(main.InferenceType());
        var pass = new AutoDistributedPass(false, PyNTTTarget.Kind, CompileOptions);

        var post = Assert.IsType<Function>(pass.RunAsync(main, new()).Result);

        var layerCalls = ExprCollector.Collect(post.Body)
            .OfType<Call>()
            .Where(call => call.Target is Function { Name: "layer" })
            .ToArray();
        Assert.Equal(8, layerCalls.Length);
        var rewrittenLayer = Assert.IsType<Function>(layerCalls[0].Target);
        var rewrittenParameter = Assert.IsType<Var>(Assert.Single(rewrittenLayer.Parameters.ToArray()));
        var parameterType = Assert.IsType<DistributedType>(rewrittenParameter.CheckedType);
        Assert.Contains(parameterType.AxisPolicies, policy => policy is SBPSplit);
        Assert.All(
            layerCalls,
            call => Assert.Equal(parameterType, Assert.Single(call.Arguments.ToArray()).CheckedType));
    }

    [Fact]
    public void TestFunctionBoundaryKeepsReferenceTensorStandalone()
    {
        var inputType = new TensorType(DataTypes.Float32, [16, 32]);
        var objectType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var input = new Var("input", inputType);
        var cache = new Var("cache", objectType);
        var layerInput = new Var("layer_input", inputType);
        var layerCache = new Var("layer_cache", objectType);
        var layer = new Function(
            "hf_decoder_layer",
            new IR.Tuple(IR.F.Math.Unary(UnaryOp.Abs, layerInput), layerCache),
            [layerInput, layerCache]);
        Assert.True(layer.InferenceType());

        var layerCall0 = new Call(layer, input, cache);
        var hidden0 = IR.F.Tensors.GetItem(layerCall0, 0);
        var cache0 = IR.F.Tensors.GetItem(layerCall0, 1);
        var layerCall1 = new Call(layer, hidden0, cache0);
        var main = new Function("main", IR.F.Tensors.GetItem(layerCall1, 0), [input, cache]);
        Assert.True(main.InferenceType());
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);

        var post = Assert.IsType<Function>(pass.RunAsync(main, new()).Result);

        var layerCalls = ExprCollector.Collect(post.Body).OfType<Call>().Where(call => call.Target is Function { Name: "hf_decoder_layer" }).ToArray();
        Assert.Equal(2, layerCalls.Length);
        var rewrittenLayer = Assert.IsType<Function>(layerCalls[0].Target);
        var parameters = rewrittenLayer.Parameters.ToArray();
        Assert.IsType<DistributedType>(parameters[0].CheckedType);
        Assert.Equal(objectType, parameters[1].CheckedType);
        foreach (var layerCall in layerCalls)
        {
            Assert.Equal(objectType, layerCall.Arguments[1].CheckedType);
            Assert.DoesNotContain(
                ExprCollector.Collect(layerCall.Arguments[1]).OfType<Call>(),
                call => call.Target is IR.Distributed.Boxing boxing && EqualityComparer<IRType>.Default.Equals(boxing.NewType, objectType));
        }
    }

    [Fact]
    public void TestAutoDistributedMaterializerReusesSameSourceAndTargetBoxing()
    {
        var inputType = new TensorType(DataTypes.Float32, [16]);
        var input = new Var("input", inputType);
        var distributedType0 = new DistributedType(inputType, new SBP[] { SBP.B }, new Placement([4], "b", "b"));
        var distributedType1 = new DistributedType(inputType, new SBP[] { SBP.B }, new Placement([4], "b", "b"));

        var graph = new DistributedSearchGraph(new AdjacencyGraph<SearchableNode, CrossEdge>(true), SearchGraphKind.Root);
        var rootBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var inputBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var boxedBucket0 = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var boxedBucket1 = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);

        var tupleNode = new SearchableNode(new IR.Tuple(), new TupleType([distributedType0, distributedType1]));
        var inputNode = new SearchableNode(input, inputType);
        var boxedNode0 = new SearchableNode(new IR.Distributed.Boxing(distributedType0), distributedType0);
        var boxedNode1 = new SearchableNode(new IR.Distributed.Boxing(distributedType1), distributedType1);
        rootBucket.AddVertex(tupleNode);
        inputBucket.AddVertex(inputNode);
        boxedBucket0.AddVertex(boxedNode0);
        boxedBucket1.AddVertex(boxedNode1);
        graph.AddEdge(new(tupleNode, boxedNode0, 0, boxedBucket0));
        graph.AddEdge(new(tupleNode, boxedNode1, 1, boxedBucket1));
        graph.AddEdge(new(boxedNode0, inputNode, 0, inputBucket));
        graph.AddEdge(new(boxedNode1, inputNode, 0, inputBucket));

        var picks = new Dictionary<SearchableNode, bool>
        {
            [tupleNode] = true,
            [inputNode] = true,
            [boxedNode0] = true,
            [boxedNode1] = true,
        };

        var tuple = Assert.IsType<IR.Tuple>(new ExprBuildVisitor(graph, picks).Visit([rootBucket]));
        Assert.Same(tuple.Fields[0], tuple.Fields[1]);
        Assert.Single(ExprCollector.Collect(tuple).OfType<Call>().Where(call => call.Target is IR.Distributed.Boxing));
    }

    [Fact]
    public void TestAutoDistributedMaterializerReusesSameSourceAndTargetShardedView()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32]);
        var placement = new Placement([4, 8], "yx", "bb");
        var inputType = new DistributedType(tensorType, [SBP.SContiguous([0, 1])], placement);
        var outputType = new DistributedType(tensorType, [SBP.B], placement);
        var input = new Var("input", inputType);

        var graph = new DistributedSearchGraph(new AdjacencyGraph<SearchableNode, CrossEdge>(true), SearchGraphKind.Root);
        var rootBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var inputBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var viewBucket0 = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var viewBucket1 = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);

        var tupleNode = new SearchableNode(new IR.Tuple(), new TupleType([outputType, outputType]));
        var inputNode = new SearchableNode(input, inputType);
        var viewNode0 = new SearchableNode(new IR.Distributed.ShardedView(outputType), outputType);
        var viewNode1 = new SearchableNode(new IR.Distributed.ShardedView(outputType), outputType);
        rootBucket.AddVertex(tupleNode);
        inputBucket.AddVertex(inputNode);
        viewBucket0.AddVertex(viewNode0);
        viewBucket1.AddVertex(viewNode1);
        graph.AddEdge(new(tupleNode, viewNode0, 0, viewBucket0));
        graph.AddEdge(new(tupleNode, viewNode1, 1, viewBucket1));
        graph.AddEdge(new(viewNode0, inputNode, 0, inputBucket));
        graph.AddEdge(new(viewNode1, inputNode, 0, inputBucket));

        var picks = new Dictionary<SearchableNode, bool>
        {
            [tupleNode] = true,
            [inputNode] = true,
            [viewNode0] = true,
            [viewNode1] = true,
        };

        var tuple = Assert.IsType<IR.Tuple>(new ExprBuildVisitor(graph, picks).Visit([rootBucket]));
        Assert.Same(tuple.Fields[0], tuple.Fields[1]);
        Assert.Single(ExprCollector.Collect(tuple).OfType<Call>().Where(call => call.Target is IR.Distributed.ShardedView));
        Assert.DoesNotContain(ExprCollector.Collect(tuple).OfType<Call>(), call => call.Target is IR.Distributed.Boxing);
    }

    [Fact]
    public void TestAutoDistributedMaterializerKeepsBoundaryAndInternalRealizationsDistinct()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32]);
        var placement = new Placement([4, 8], "yx", "bb");
        var inputType = new DistributedType(tensorType, [SBP.SContiguous([0, 1])], placement);
        var outputType = new DistributedType(tensorType, [SBP.B], placement);
        var input = new Var("input", inputType);

        var graph = new DistributedSearchGraph(new AdjacencyGraph<SearchableNode, CrossEdge>(true), SearchGraphKind.Root);
        var rootBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var inputBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var viewBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
        var boxingBucket = graph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);

        var tupleNode = new SearchableNode(new IR.Tuple(), new TupleType([outputType, outputType]));
        var inputNode = new SearchableNode(input, inputType);
        var viewNode = new SearchableNode(new IR.Distributed.ShardedView(outputType), outputType);
        var boxingNode = new SearchableNode(new IR.Distributed.Boxing(outputType), outputType);
        rootBucket.AddVertex(tupleNode);
        inputBucket.AddVertex(inputNode);
        viewBucket.AddVertex(viewNode);
        boxingBucket.AddVertex(boxingNode);
        graph.AddEdge(new(tupleNode, viewNode, 0, viewBucket));
        graph.AddEdge(new(tupleNode, boxingNode, 1, boxingBucket));
        graph.AddEdge(new(viewNode, inputNode, 0, inputBucket));
        graph.AddEdge(new(boxingNode, inputNode, 0, inputBucket));

        var picks = new Dictionary<SearchableNode, bool>
        {
            [tupleNode] = true,
            [inputNode] = true,
            [viewNode] = true,
            [boxingNode] = true,
        };

        var tuple = Assert.IsType<IR.Tuple>(new ExprBuildVisitor(graph, picks).Visit([rootBucket]));
        Assert.NotSame(tuple.Fields[0], tuple.Fields[1]);
        Assert.IsType<IR.Distributed.ShardedView>(Assert.IsType<Call>(tuple.Fields[0]).Target);
        Assert.IsType<IR.Distributed.Boxing>(Assert.IsType<Call>(tuple.Fields[1]).Target);
    }

    [Fact]
    public void TestPyNTTReshardPolicySelectsOneLegalRealization()
    {
        var options = new PyNTTTargetOptions();
        var policy = options.ReshardRealizationPolicy;
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var exclusiveSplit = new DistributedType(
            tensorType,
            [SBP.SContiguous([0]), SBP.SContiguous([1])],
            placement);
        var ySplitXReplicated = new DistributedType(
            tensorType,
            [SBP.SContiguous([0]), SBP.B],
            placement);
        var yReplicatedXSplit = new DistributedType(
            tensorType,
            [SBP.B, SBP.SContiguous([1])],
            placement);
        var broadcast = new DistributedType(tensorType, [SBP.B, SBP.B], placement);
        var partial = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            placement,
            SBP.P([0, 1]));
        var reduceScatter = new DistributedType(
            tensorType,
            [SBP.SContiguous([0, 1], 1), SBP.B],
            placement);

        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(tensorType, exclusiveSplit, DistributedReshardSourceKind.Constant));
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(exclusiveSplit, broadcast, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(ySplitXReplicated, broadcast, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(yReplicatedXSplit, broadcast, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(broadcast, exclusiveSplit, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(broadcast, exclusiveSplit, DistributedReshardSourceKind.FunctionParameter));
        Assert.Equal(
            DistributedReshardRealization.Boxing,
            Classify(partial, broadcast, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.Boxing,
            Classify(partial, reduceScatter, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(reduceScatter, broadcast, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.Boxing,
            Classify(
                exclusiveSplit,
                broadcast,
                DistributedReshardSourceKind.Internal,
                DistributedReshardUsageKind.FunctionBoundary));
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(
                exclusiveSplit,
                broadcast,
                DistributedReshardSourceKind.Internal,
                DistributedReshardUsageKind.ProgramOutput));

        var multiLevelPlacement = new Placement([2, 4], "cb", "cb");
        var multiLevelSource = new DistributedType(
            tensorType,
            [SBP.SContiguous([0]), SBP.SContiguous([1])],
            multiLevelPlacement);
        var sameChipShardTarget = new DistributedType(
            tensorType,
            [SBP.SContiguous([0]), SBP.B],
            multiLevelPlacement);
        var crossChipTarget = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            multiLevelPlacement);
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(multiLevelSource, sameChipShardTarget, DistributedReshardSourceKind.Internal));
        Assert.Equal(
            DistributedReshardRealization.Boxing,
            Classify(multiLevelSource, crossChipTarget, DistributedReshardSourceKind.Internal));

        var degenerateBlockPlacement = new Placement([2, 1], "yx", "bb");
        var degenerateSource = new DistributedType(
            tensorType,
            [SBP.SContiguous([0]), SBP.B],
            degenerateBlockPlacement);
        var degenerateTarget = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            degenerateBlockPlacement);
        Assert.Equal(
            DistributedReshardRealization.ShardedView,
            Classify(degenerateSource, degenerateTarget, DistributedReshardSourceKind.Internal));

        options.MemoryAccessArch = MemoryAccessArchitecture.NUMA;
        Assert.Equal(
            DistributedReshardRealization.Boxing,
            Classify(exclusiveSplit, broadcast, DistributedReshardSourceKind.Internal));

        DistributedReshardRealization Classify(
            IRType source,
            IRType target,
            DistributedReshardSourceKind sourceKind,
            DistributedReshardUsageKind usageKind = DistributedReshardUsageKind.Internal)
            => policy.Classify(
                new DistributedReshardRealizationContext(
                    options,
                    PyNTTTarget.Kind,
                    source,
                    target,
                    sourceKind,
                    usageKind));
    }

    [Fact]
    public void TestPyNTTEntryTerminatorOffersProgramOutputShardedView()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var input = new Var("input", tensorType);
        var output = IR.F.Math.Unary(UnaryOp.Abs, input);
        var function = new Function(
            "main",
            new IR.Tuple(output),
            [input]);
        Assert.True(function.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind);
        var buildMethod = typeof(AutoDistributedRewriter).GetMethod(
            "BuildFunctionSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(buildMethod);
        var resultCluster = Assert.IsType<DistributedSearchGraph>(
            buildMethod!.Invoke(rewriter, [function, true]));

        var memoField = typeof(AutoDistributedRewriter).GetField(
            "_reshardCandidateMemo",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var memo = Assert.IsType<
            Dictionary<
                ReshardCandidateKey,
                (DistributedSearchGraph Bucket, SearchableNode Node)>>(memoField?.GetValue(rewriter));
        var outputViews = memo
            .Where(entry =>
                entry.Key.UsageKind == DistributedReshardUsageKind.ProgramOutput)
            .Select(entry => entry.Value.Node)
            .ToArray();

        Assert.NotEmpty(outputViews);
        Assert.All(outputViews, candidate =>
        {
            Assert.IsType<IR.Distributed.ShardedView>(candidate.Expr);
            var outputType = Assert.IsType<DistributedType>(candidate.IRType);
            Assert.All(outputType.AxisPolicies, policy => Assert.IsType<SBPBroadCast>(policy));
        });

        var post = rewriter.SolveAndExtract(function, resultCluster);
        var resultTuple = Assert.IsType<IR.Tuple>(post.Body);
        var resultView = Assert.IsType<Call>(Assert.Single(resultTuple.Fields.ToArray()));
        Assert.IsType<IR.Distributed.ShardedView>(resultView.Target);
    }

    [Fact]
    public void TestFunctionResultSignatureUsesOnlyDirectProducerCandidates()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var input = new Var("input", tensorType);
        var function = new Function(
            "layer",
            IR.F.Math.Unary(UnaryOp.Abs, input),
            [input]);
        Assert.True(function.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind);
        var buildMethod = typeof(AutoDistributedRewriter).GetMethod(
            "BuildFunctionSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(buildMethod);
        var resultCluster = Assert.IsType<DistributedSearchGraph>(
            buildMethod!.Invoke(rewriter, [function, false]));
        var resultCandidates = resultCluster.Clusters
            .OfType<DistributedSearchGraph>()
            .SelectMany(bucket => bucket.Vertices)
            .ToArray();
        Assert.NotEmpty(resultCandidates);
        Assert.All(resultCandidates, candidate => Assert.Equal(SearchableNodeKind.FunctionResult, candidate.Kind));
        Assert.DoesNotContain(
            resultCandidates,
            node => node.Expr is IR.Distributed.Boxing or IR.Distributed.ShardedView);

        var graphField = typeof(AutoDistributedRewriter).GetField(
            "_rootSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var graph = Assert.IsType<DistributedSearchGraph>(graphField?.GetValue(rewriter));
        foreach (var candidate in resultCandidates)
        {
            Assert.True(graph.TryGetOutEdges(candidate, out var edges));
            var producer = Assert.Single(edges.Where(edge => edge.InputIndex == 0));
            Assert.IsType<IR.Math.Unary>(producer.Input.Expr);
            Assert.Equal(candidate.IRType, producer.Input.IRType);
        }

        var memoField = typeof(AutoDistributedRewriter).GetField(
            "_reshardCandidateMemo",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var memo = Assert.IsType<
            Dictionary<
                ReshardCandidateKey,
                (DistributedSearchGraph Bucket, SearchableNode Node)>>(memoField?.GetValue(rewriter));
        Assert.DoesNotContain(memo, entry => ReferenceEquals(entry.Key.OwnerCluster, resultCluster));
        Assert.Contains(
            memo,
            entry => entry.Key.UsageKind == DistributedReshardUsageKind.Internal &&
                entry.Value.Node.Expr is IR.Distributed.ShardedView);

        var mainInput = new Var("main_input", tensorType);
        var main = new Function("main", new Call(function, mainInput), [mainInput]);
        Assert.True(main.InferenceType());
        var post = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind).RewriteProgram(main, [function, main]);
        var layerCall = Assert.Single(
            ExprCollector.Collect(post.Body).OfType<Call>().Where(call => call.Target is Function { Name: "layer" }));
        var rewrittenLayer = Assert.IsType<Function>(layerCall.Target);
        Assert.DoesNotContain(
            ExprCollector.Collect(rewrittenLayer.Body).OfType<Call>(),
            call => call.Target is IR.Distributed.Boxing or IR.Distributed.ShardedView);
    }

    [Fact]
    public void TestFunctionCallResultOffersConsumerSideShardedViews()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var layerInput = new Var("layer_input", tensorType);
        var layer = new Function(
            "layer",
            IR.F.Math.Unary(UnaryOp.Abs, layerInput),
            [layerInput]);
        Assert.True(layer.InferenceType());

        var input = new Var("input", tensorType);
        var layerCall = new Call(layer, input);
        var main = new Function("main", layerCall, [input]);
        Assert.True(main.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind);
        var buildMethod = typeof(AutoDistributedRewriter).GetMethod(
            "BuildFunctionSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(buildMethod);
        var layerResultCluster = Assert.IsType<DistributedSearchGraph>(
            buildMethod!.Invoke(rewriter, [layer, false]));
        _ = Assert.IsType<DistributedSearchGraph>(buildMethod.Invoke(rewriter, [main, true]));

        var memoField = typeof(AutoDistributedRewriter).GetField(
            "_reshardCandidateMemo",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var memo = Assert.IsType<
            Dictionary<
                ReshardCandidateKey,
                (DistributedSearchGraph Bucket, SearchableNode Node)>>(memoField?.GetValue(rewriter));
        var callerViews = memo.Where(entry =>
            entry.Key.InputNode.Kind == SearchableNodeKind.FunctionCall &&
            entry.Key.UsageKind == DistributedReshardUsageKind.Internal &&
            entry.Value.Node.Expr is IR.Distributed.ShardedView).ToArray();

        Assert.NotEmpty(callerViews);
        Assert.All(
            callerViews,
            entry => Assert.False(ReferenceEquals(entry.Key.OwnerCluster, layerResultCluster)));
    }

    [Fact]
    public void TestPyNTTFunctionParameterUsesCanNarrowBroadcastSignatureLocally()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var broadcastType = new DistributedType(
            tensorType,
            [SBP.B, SBP.B],
            placement);
        var input = new Var("input", tensorType);
        var function = new Function(
            "layer",
            IR.F.Math.Unary(UnaryOp.Abs, input),
            [input]);
        Assert.True(function.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind);
        var buildMethod = typeof(AutoDistributedRewriter).GetMethod(
            "BuildFunctionSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(buildMethod);
        _ = buildMethod!.Invoke(rewriter, [function, false]);

        var memoField = typeof(AutoDistributedRewriter).GetField(
            "_reshardCandidateMemo",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var memo = Assert.IsType<
            Dictionary<
                ReshardCandidateKey,
                (DistributedSearchGraph Bucket, SearchableNode Node)>>(memoField?.GetValue(rewriter));
        var candidates = memo.Where(entry =>
            entry.Key.InputNode.Kind == SearchableNodeKind.TypeAdapter &&
            entry.Key.InputNode.IRType is DistributedType sourceType &&
            sourceType.AxisPolicies.All(policy => policy is SBPBroadCast) &&
            entry.Key.TargetType is DistributedType targetType &&
            targetType.AxisPolicies[0] is SBPBroadCast &&
            targetType.AxisPolicies[1] is SBPSplit split &&
            split.HierarchyAxes.Order().SequenceEqual(new[] { 0, 1 }) &&
            entry.Key.UsageKind == DistributedReshardUsageKind.Internal).ToArray();
        Assert.NotEmpty(candidates);
        Assert.All(
            candidates,
            candidate => Assert.IsType<IR.Distributed.ShardedView>(candidate.Value.Node.Expr));

        var graphField = typeof(AutoDistributedRewriter).GetField(
            "_rootSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var graph = Assert.IsType<DistributedSearchGraph>(graphField?.GetValue(rewriter));
        foreach (var candidate in candidates)
        {
            Assert.True(graph.TryGetOutEdges(candidate.Key.InputNode, out var edges));
            Assert.Contains(
                edges,
                edge =>
                    edge.Input.Kind == SearchableNodeKind.FunctionParameter &&
                    edge.Input.IRType == broadcastType);
        }
    }

    [Fact]
    public void TestPyNTTInternalReshardUsesOnlyOneTargetRealizationKind()
    {
        var options = new PyNTTTargetOptions
        {
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
            Hierarchies = [new[] { 4, 8 }],
        };
        CompileOptions.TargetOptions = options;
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var input = new Var("input", tensorType);
        var producer = IR.F.Math.Unary(UnaryOp.Abs, input);
        var main = new Function(
            "main",
            IR.F.Math.Unary(UnaryOp.Neg, producer),
            [input]);
        Assert.True(main.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            options,
            AutoDistributedPhase.Final,
            PyNTTTarget.Kind);
        _ = rewriter.BuildSearchGraph(main);

        var graphField = typeof(AutoDistributedRewriter).GetField(
            "_rootSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var graph = Assert.IsType<DistributedSearchGraph>(graphField?.GetValue(rewriter));
        var realizations = graph.Vertices.Where(node =>
        {
            if (node.Expr is not (IR.Distributed.Boxing or IR.Distributed.ShardedView) ||
                node.IRType is not DistributedType targetType ||
                targetType.AxisPolicies.Any(policy => policy is not SBPBroadCast) ||
                !graph.TryGetOutEdges(node, out var edges))
            {
                return false;
            }

            return edges.Any(edge =>
                edge.InputIndex == 0 &&
                edge.Target.Expr is IR.Math.Unary { UnaryOp: UnaryOp.Abs } &&
                edge.Target.IRType is DistributedType sourceType &&
                sourceType.AxisPolicies
                    .OfType<SBPSplit>()
                    .SelectMany(split => split.HierarchyAxes)
                    .Order()
                    .SequenceEqual(new[] { 0, 1 }));
        }).ToArray();

        Assert.NotEmpty(realizations);
        Assert.All(
            realizations,
            realization => Assert.IsType<IR.Distributed.ShardedView>(realization.Expr));
    }

    [Fact]
    public async Task TestShapeBucketSegmentsFromEntryAndClonesInternalFunctions()
    {
        var n = new DimVar("n") { Metadata = { Range = (1, 32) } };
        var tensorType = new TensorType(DataTypes.Float32, [n, 16]);
        var layerInput = new Var("layer_input", tensorType);
        var layer = new Function("layer", IR.F.Math.Unary(UnaryOp.Abs, layerInput), [layerInput]);
        Assert.True(layer.InferenceType());

        var input = new Var("input", tensorType);
        var main = new Function("main", new Call(layer, input), [input]);
        Assert.True(main.InferenceType());

        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 2;
        CompileOptions.ShapeBucketOptions.VarMap.Add(input, input.CheckedShape.ToArray());

        var module = new IRModule(main);
        module.Add(layer);
        module = await new AutoDistributedWithShapeBucketPass(false, CPUTarget.Kind, CompileOptions).RunAsync(module, new());
        module = await new AddFunctionToModule(CompileOptions).RunAsync(module, new());
        module = await new RemoveUnusedFunctions(CompileOptions).RunAsync(module, new());

        var names = module.Functions.Select(function => function.Name).ToArray();
        Assert.Contains("main", names);
        Assert.DoesNotContain("main_prim", names);
        Assert.Equal(2, names.Count(name => name.StartsWith("main_segment_", System.StringComparison.Ordinal)));
        Assert.Equal(2, names.Count(name => name.StartsWith("layer_segment_", System.StringComparison.Ordinal)));
        Assert.DoesNotContain("layer", names);
        Assert.DoesNotContain("layer_prim", names);
    }

    [Fact]
    public void TestNonUniformSplitCandidateIsGenerated()
    {
        var tensorType = new TensorType(DataTypes.Float32, [1024]);
        var placement = new Placement([36], "b", "b");
        var policies = DistributedUtility.GetLeafCandidatePolicies(
            tensorType,
            placement,
            ContiguousDistributedSplitCandidateProvider.Instance);

        Assert.Contains(policies, policy => policy.Count == 1 && policy[0] is SBPSplit split && split.HierarchyAxes.SequenceEqual(new[] { 0 }));

        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.SContiguous([0]) }, placement);
        Assert.Equal(new[] { 1015L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 35 }).Offset).ToValueArray());
        Assert.Equal(new[] { 9L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 35 }).Shape).ToValueArray());

        var skinnyType = new DistributedType(new TensorType(DataTypes.Float32, new[] { 37L }), new SBP[] { SBP.SContiguous([0]) }, placement);
        Assert.Equal(new[] { 0L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(skinnyType, new[] { 35 }).Shape).ToValueArray());
    }

    [Fact]
    public void TestBlockCyclicLocalShardDescriptor()
    {
        var tensorType = new TensorType(DataTypes.Float32, [29]);
        var placement = new Placement([4], "b", "b");
        var distributedType = new DistributedType(
            tensorType,
            new SBP[] { SBP.SBlockCyclic([0], 3) },
            placement);

        var expectedActiveExtents = new long[] { 9, 8, 6, 6 };
        for (var shard = 0; shard < placement.Hierarchy[0]; shard++)
        {
            var descriptor = DistributedUtility.GetLocalShardDescriptor(
                distributedType,
                new[] { shard });
            Assert.Equal(new[] { 9L }, descriptor.LocalCapacityShape.ToValueArray());
            Assert.Equal(new[] { expectedActiveExtents[shard] }, descriptor.ActiveShape.ToValueArray());
        }

        var shard0 = DistributedUtility.GetLocalShardDescriptor(distributedType, new[] { 0 });
        Assert.Equal(
            new long[] { 0, 1, 2, 12, 13, 14, 24, 25, 26 },
            Enumerable.Range(0, 9)
                .Select(index => shard0.Axes[0].MapLocalToGlobal(index).FixedValue)
                .ToArray());
        var shard1 = DistributedUtility.GetLocalShardDescriptor(distributedType, new[] { 1 });
        Assert.Equal(
            new long[] { 3, 4, 5, 15, 16, 17, 27, 28 },
            Enumerable.Range(0, 8)
                .Select(index => shard1.Axes[0].MapLocalToGlobal(index).FixedValue)
                .ToArray());
        Assert.Throws<InvalidOperationException>(
            () => DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 0 }));
    }

    [Fact]
    public void TestSingleBlockCyclicShardIsAContiguousRegion()
    {
        var tensorType = new TensorType(DataTypes.Float32, [8]);
        var placement = new Placement([4], "b", "b");
        var distributedType = new DistributedType(
            tensorType,
            new SBP[] { SBP.SBlockCyclic([0], 2) },
            placement);

        var descriptor = DistributedUtility.GetLocalShardDescriptor(distributedType, new[] { 3 });
        Assert.True(descriptor.TryGetContiguousRegion(out var offset, out var shape));
        Assert.Equal(new[] { 6L }, new RankedShape(offset).ToValueArray());
        Assert.Equal(new[] { 2L }, new RankedShape(shape).ToValueArray());
    }

    [Fact]
    public void TestOrderedSplitStagesComposeAcrossPhysicalLevels()
    {
        var tensorType = new TensorType(DataTypes.Float32, [1000]);
        var placement = new Placement([2, 4, 8], "cyx", "cbb");
        var split = SBP.S(
            SplitStage.Contiguous([0]),
            SplitStage.BlockCyclic([1, 2], 4));
        var distributedType = new DistributedType(tensorType, new SBP[] { split }, placement);
        var descriptor = DistributedUtility.GetLocalShardDescriptor(
            distributedType,
            new[] { 1, 2, 7 });

        Assert.Equal(new[] { 16L }, descriptor.LocalCapacityShape.ToValueArray());
        Assert.Equal(new[] { 16L }, descriptor.ActiveShape.ToValueArray());
        Assert.Equal(592, descriptor.Axes[0].MapLocalToGlobal(0).FixedValue);
        Assert.Equal(720, descriptor.Axes[0].MapLocalToGlobal(4).FixedValue);
    }

    [Fact]
    public void TestSplitStageJsonIsStrictAndRoundTrips()
    {
        var options = new JsonSerializerOptions();
        options.Converters.Add(new SBPConverter());
        SBP policy = SBP.S(
            SplitStage.Contiguous([0], 64),
            SplitStage.BlockCyclic([1, 2], 8));

        var json = JsonSerializer.Serialize(policy, options);
        var roundTrip = JsonSerializer.Deserialize<SBP>(json, options);

        Assert.Equal(policy, roundTrip);
        Assert.Contains("\"Stages\"", json, StringComparison.Ordinal);
        Assert.Contains("\"BlockCyclic\"", json, StringComparison.Ordinal);
        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize<SBP>(
            "{\"$type\":\"S\",\"Axes\":[0]}",
            options));
        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize<SBP>(
            "{\"$type\":\"S\",\"Stages\":[{\"HierarchyAxes\":[0],\"Distribution\":{\"$type\":\"Contiguous\"}}],\"Axes\":[0]}",
            options));
        Assert.Throws<ArgumentException>(() => SBP.S(
            SplitStage.Contiguous([0]),
            SplitStage.BlockCyclic([0], 8)));
    }

    [Fact]
    public void TestPyNTTSplitCandidateStagesFollowPhysicalLevels()
    {
        var tensorType = new TensorType(DataTypes.BFloat16, [4096]);
        var placement = new Placement([2, 4, 8], "cyx", "cbb");
        var context = new DistributedSplitCandidateContext(
            tensorType,
            0,
            placement,
            new[] { 0, 1, 2 },
            64,
            4096);
        var candidates = new PyNTTDistributedSplitCandidateProvider(128)
            .GetCandidates(context);

        var staged = Assert.Single(candidates);
        Assert.Collection(
            staged.Stages,
            stage =>
            {
                Assert.Equal(new[] { 0 }, stage.HierarchyAxes.ToArray());
                Assert.IsType<ContiguousSplit>(stage.Distribution);
            },
            stage =>
            {
                Assert.Equal(new[] { 1, 2 }, stage.HierarchyAxes.ToArray());
                Assert.Equal(64, Assert.IsType<BlockCyclicSplit>(stage.Distribution).BlockSize);
            });
    }

    [Fact]
    public void TestD2DBoxingRejectsDifferentTensorType()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var source = new DistributedType(new TensorType(DataTypes.Float32, [16, 32]), new SBP[] { SBP.SContiguous([0]), SBP.SContiguous([1]) }, placement);
        var target = new DistributedType(new TensorType(DataTypes.Float32, [16, 8, 4]), new SBP[] { SBP.SContiguous([0]), SBP.SContiguous([1]), SBP.B }, placement);
        var rewriter = new AutoDistributedRewriter(CompileOptions, (INTTTargetOptions)CompileOptions.TargetOptions, AutoDistributedPhase.Final, CPUTarget.Kind);

        var method = typeof(AutoDistributedRewriter).GetMethod("CheckBoxingType", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        var result = method!.Invoke(rewriter, new object[] { source, target, false });

        Assert.IsType<InvalidType>(Assert.IsAssignableFrom<IRType>(result));
    }

    [Fact]
    public void TestTupleGetItemTypeInferenceCacheKeepsAttributeIndex()
    {
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 128) } };
        var q = new Var("q", new TensorType(new VectorType(DataTypes.BFloat16, [4, 8]), new Dimension[] { sequenceLength, 64 }));
        var k = new Var("k", new TensorType(new VectorType(DataTypes.BFloat16, [4, 8]), new Dimension[] { sequenceLength, 32 }));
        var v = new Var("v", new TensorType(new VectorType(DataTypes.BFloat16, [4, 8]), new Dimension[] { sequenceLength, 32 }));
        var tuple = new IR.Tuple(q, k, v);
        var qItem = IR.F.Tensors.GetItem(tuple, 0);
        var qVector = IR.F.Tensors.Bitcast(qItem, new VectorType(DataTypes.BFloat16, [8]));
        var qReshape = IR.F.Tensors.Reshape(qVector, new Dimension[] { sequenceLength, 16, 16 });
        var vItem = IR.F.Tensors.GetItem(tuple, 2);
        var vReshape = IR.F.Tensors.Reshape(vItem, new Dimension[] { sequenceLength, 8, 4 });
        var main = new Function("main", new IR.Tuple(qReshape, vReshape), [q, k, v]);
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);

        var post = pass.RunAsync(main, new()).Result;

        Assert.NotNull(post);
    }

    [Fact]
    public void TestGetItemOutputReshardClosureLinksEveryInferredSource()
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = [new[] { 4, 8 }];

        var tensorType = new TensorType(DataTypes.Float32, [32]);
        var input = new Var("input", tensorType);
        var tuple = new IR.Tuple(input, new DimConst(0));
        var item = IR.F.Tensors.GetItem(tuple, 0);
        var main = new Function("main", item, [input]);
        Assert.True(main.InferenceType());

        var rewriter = new AutoDistributedRewriter(
            CompileOptions,
            targetOptions,
            AutoDistributedPhase.Final,
            CPUTarget.Kind);
        _ = rewriter.BuildSearchGraph(main);

        var graphField = typeof(AutoDistributedRewriter).GetField(
            "_rootSearchGraph",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var graph = Assert.IsType<DistributedSearchGraph>(graphField?.GetValue(rewriter));
        var placement = new Placement([4, 8], "yx", "bb");
        var splitYXGetItems = graph.Vertices
            .Where(node => node.Expr is IR.Tensors.GetItem && HasSplit(node.IRType, [0, 1]))
            .ToHashSet(ReferenceEqualityComparer.Instance);
        Assert.NotEmpty(splitYXGetItems);

        var matchingTargetBuckets = graph.Clusters
            .OfType<DistributedSearchGraph>()
            .SelectMany(cluster => cluster.Clusters.OfType<DistributedSearchGraph>())
            .Where(bucket => bucket.Vertices.Any(node =>
                node.Expr is IR.Distributed.Boxing
                && HasSplit(node.IRType, [1])
                && graph.TryGetOutEdges(node, out var edges)
                && edges.Any(edge => splitYXGetItems.Contains(edge.Target))))
            .ToArray();

        Assert.Single(matchingTargetBuckets);

        bool HasSplit(IRType type, int[] axes)
            => type is DistributedType distributed
                && distributed.TensorType == tensorType
                && distributed.Placement == placement
                && distributed.AxisPolicies is [SBPSplit split]
                && split.HierarchyAxes.SequenceEqual(axes);
    }

    [Fact]
    public void TestDynamicSplitGranularityUsesRuntimeShape()
    {
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 1024) } };
        var tensorType = new TensorType(DataTypes.Float32, [sequenceLength]);
        var placement = new Placement([4], "y", "b");
        var policies = DistributedUtility.GetLeafCandidatePolicies(
            tensorType,
            placement,
            ContiguousDistributedSplitCandidateProvider.Instance);

        var split = policies
            .Select(policy => policy.SingleOrDefault() as SBPSplit)
            .Single(policy => policy is not null && policy.HierarchyAxes.SequenceEqual(new[] { 0 }))!;
        var contiguous = Assert.IsType<ContiguousSplit>(Assert.Single(split.Stages).Distribution);
        Assert.NotNull(contiguous.Granularity);
        Assert.False(contiguous.Granularity.IsFixed);
        Assert.True(contiguous.Granularity.Metadata.Range.HasValue);
        Assert.Equal(1d, contiguous.Granularity.Metadata.Range.Value.Min);
        Assert.Equal(256d, contiguous.Granularity.Metadata.Range.Value.Max);

        var distributedType = new DistributedType(tensorType, new SBP[] { split }, placement);
        var dividedShape = DistributedUtility.GetDividedTensorType(distributedType).Shape;
        Assert.False(dividedShape[0].IsFixed);
        Assert.Equal(256d, dividedShape[0].Metadata.Range!.Value.Max);
        Assert.Equal(new[] { 256L }, DistributedUtility.GetDividedTensorType(distributedType, DistributedUtility.DivideFlags.MaxShape).Shape.ToValueArray());
        Assert.Equal(new[] { 768L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 3 }, DistributedUtility.DivideFlags.MaxShape).Offset).ToValueArray());
        Assert.Equal(new[] { 256L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 3 }, DistributedUtility.DivideFlags.MaxShape).Shape).ToValueArray());
    }

    [Fact]
    public void TestReshardPlannerDecomposesPartialToBroadcastThenSplit()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var source = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.SContiguous([1]) }, placement, SBP.P([1]));
        var noPartial = new DistributedType(tensorType, source.AxisPolicies, placement);
        var broadcast = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.B }, placement);
        var target = new DistributedType(tensorType, new SBP[] { SBP.SContiguous([0]), SBP.B }, placement);

        var plans = DistributedReshardPlanner.Plan(source, target, CanBox);

        Assert.Single(plans);
        Assert.True(plans[0].StepTypes.SequenceEqual(new IRType[] { noPartial, broadcast, target }));

        bool CanBox(IRType input, IRType output)
            => (input, output) switch
            {
                (DistributedType i, DistributedType o) when i == source && o == noPartial => true,
                (DistributedType i, DistributedType o) when i == noPartial && o == broadcast => true,
                (DistributedType i, DistributedType o) when i == broadcast && o == target => true,
                _ => false,
            };
    }

    [Fact]
    public void TestReshardPlannerDecomposesPartialAllReduceThroughReduceScatter()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var source = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement,
            SBP.P([0, 1]));
        var target = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);

        var plans = DistributedReshardPlanner.Plan(source, target, CanRealize);

        Assert.NotEmpty(plans);
        Assert.All(
            plans,
            plan =>
            {
                Assert.Equal(2, plan.StepTypes.Count);
                var reduceScatter = Assert.IsType<DistributedType>(plan.StepTypes[0]);
                Assert.Null(reduceScatter.Partial);
                var split = Assert.Single(reduceScatter.AxisPolicies.OfType<SBPSplit>());
                Assert.Equal(new[] { 0, 1 }, split.HierarchyAxes.ToArray());
                Assert.Equal(target, plan.StepTypes[1]);
            });

        bool CanRealize(IRType input, IRType output)
        {
            if (input == source && output == target)
            {
                return false;
            }

            return input == source
                ? output is DistributedType intermediate &&
                    intermediate.Partial is null &&
                    intermediate.AxisPolicies
                        .OfType<SBPSplit>()
                        .Any(split => split.HierarchyAxes.SequenceEqual(new[] { 0, 1 }))
                : output == target;
        }
    }

    [Fact]
    public void TestReshardPlannerKeepsDirectAndDecomposedPartialAllReducePlans()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var source = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement,
            SBP.P([0, 1]));
        var target = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);

        var plans = DistributedReshardPlanner.Plan(source, target, (_, _) => true);

        Assert.Contains(plans, plan => plan.StepTypes.SequenceEqual(new IRType[] { target }));
        Assert.Contains(
            plans,
            plan => plan.StepTypes.Count == 2 &&
                plan.StepTypes[0] is DistributedType { Partial: null } intermediate &&
                intermediate.AxisPolicies.OfType<SBPSplit>().Any() &&
                plan.StepTypes[1] == target);
    }

    [Fact]
    public void TestReshardPlannerKeepsDirectPathCompact()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var source = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.SContiguous([1]) }, placement);
        var target = new DistributedType(tensorType, new SBP[] { SBP.SContiguous([0]), SBP.B }, placement);

        var plans = DistributedReshardPlanner.Plan(source, target, (_, _) => true);

        Assert.Single(plans);
        Assert.True(plans[0].StepTypes.SequenceEqual(new IRType[] { target }));
    }
}
