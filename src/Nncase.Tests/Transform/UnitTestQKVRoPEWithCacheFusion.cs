// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.Evaluator.NN;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.NTT;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestQKVRoPEWithCacheFusion : TestClassBase
{
    [Fact]
    public async Task TestFusionFormsHighLevelTupleOpBeforeDistribution()
    {
        var q = new Var("q", new TensorType(DataTypes.BFloat16, new[] { 8, 2, 16 }));
        var k = new Var("k", new TensorType(DataTypes.BFloat16, new[] { 8, 1, 16 }));
        var v = new Var("v", new TensorType(DataTypes.BFloat16, new[] { 8, 1, 16 }));
        var qScale = new Var("q_scale", new TensorType(DataTypes.BFloat16, new[] { 16 }));
        var kScale = new Var("k_scale", new TensorType(DataTypes.BFloat16, new[] { 16 }));
        var cos = new Var("cos", new TensorType(DataTypes.Float32, new[] { 8, 1, 16 }));
        var sin = new Var("sin", new TensorType(DataTypes.Float32, new[] { 8, 1, 16 }));
        var extra = new Var("extra", new TensorType(DataTypes.UInt8, new[] { 1 }));
        var scale = new Var("scale", TensorType.Scalar(DataTypes.BFloat16));
        var layerId = new DimVar("layer_id");
        var config = CreateCacheConfig();
        var cache = new Var(
            "cache",
            TensorType.Scalar(new ReferenceType(new PagedAttentionKVCacheType { Config = config })));
        IRArray<AttentionDimKind> qkvLayout =
            [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim];
        IRArray<AttentionDimKind> attentionLayout =
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq];

        var qStats = IR.F.NN.NormStats(2, q, useMean: false);
        var kStats = IR.F.NN.NormStats(2, k, useMean: false);
        var qNorm = IR.F.NN.NormApply(
            2,
            1e-6f,
            q,
            qStats,
            qScale,
            Tensor.Zeros(DataTypes.BFloat16, [16]),
            useMean: false);
        var kNorm = IR.F.NN.NormApply(
            2,
            2e-6f,
            k,
            kStats,
            kScale,
            Tensor.Zeros(DataTypes.BFloat16, [16]),
            useMean: false);
        var qRoPE = IR.F.NN.RoPE(qNorm, cos, sin);
        var kRoPE = IR.F.NN.RoPE(kNorm, cos, sin);
        var permutation = AttentionLayoutUtility.GetPermutation(qkvLayout, attentionLayout);
        var qView = IR.F.Tensors.Pack(IR.F.Tensors.Transpose(qRoPE, permutation), [8], [1]);
        var kView = IR.F.Tensors.Pack(IR.F.Tensors.Transpose(kRoPE, permutation), [8], [1]);
        var vView = IR.F.Tensors.Pack(IR.F.Tensors.Transpose(v, permutation), [8], [2]);
        var cacheAfterKey = IR.F.NN.UpdatePagedAttentionKVCache(
            kView,
            cache,
            layerId,
            AttentionCacheKind.Key,
            attentionLayout.ToArray());
        var cacheAfterValue = IR.F.NN.UpdatePagedAttentionKVCache(
            vView,
            cacheAfterKey,
            layerId,
            AttentionCacheKind.Value,
            attentionLayout.ToArray());
        var attention = IR.F.NN.PagedAttention(
            qView,
            cacheAfterValue,
            extra,
            scale,
            layerId,
            attentionLayout.ToArray(),
            16);
        var function = new Function(
            "decoder",
            string.Empty,
            new IR.Tuple(attention, cacheAfterValue),
            new IVar[] { q, k, v, qScale, kScale, cos, sin, extra, scale, cache, layerId });
        Assert.True(function.InferenceType());
        var originalType = function.CheckedType;

        var rewritten = Assert.IsType<Function>(
            await new FormQKVRoPEWithCachePass().RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var fusedCall = Assert.Single(calls.Where(call => call.Target is QKVRoPEWithCache));
        var fused = Assert.IsType<QKVRoPEWithCache>(fusedCall.Target);
        var fusedInputs = Assert.IsType<IR.Tuple>(fusedCall[QKVRoPEWithCache.QKV]);

        Assert.Equal(3, fusedInputs.Count);
        Assert.Same(q, fusedInputs[0]);
        Assert.Same(k, fusedInputs[1]);
        Assert.Same(v, fusedInputs[2]);
        Assert.Equal(1e-6f, fused.QEpsilon);
        Assert.Equal(2e-6f, fused.KEpsilon);
        Assert.Equal(qkvLayout, fused.QKVLayout);
        Assert.Equal(attentionLayout, fused.AttentionLayout);
        Assert.DoesNotContain(
            calls,
            call => call.Target is NormApply or RoPE or Transpose or Pack or UpdatePagedAttentionKVCache);
        Assert.True(rewritten.InferenceType());
        Assert.Equal(originalType, rewritten.CheckedType);

        var vectorized = Assert.IsType<Call>(
            CompilerServices.Rewrite(fusedCall, [new VectorizeQKVRoPEWithCache()], new()));
        var vectorizedQKV = Assert.IsType<IR.Tuple>(vectorized[QKVRoPEWithCache.QKV]);
        Assert.Equal(new[] { 8 }, Assert.IsType<VectorType>(vectorizedQKV[0].CheckedDataType).Lanes.ToArray());
        Assert.Equal(new[] { 8 }, Assert.IsType<VectorType>(vectorizedQKV[1].CheckedDataType).Lanes.ToArray());
        Assert.Equal(new[] { 8 }, Assert.IsType<VectorType>(vectorizedQKV[2].CheckedDataType).Lanes.ToArray());
        Assert.Equal(
            new[] { 2, 8 },
            Assert.IsType<VectorType>(vectorized[QKVRoPEWithCache.Cos].CheckedDataType).Lanes.ToArray());
        Assert.Equal(fusedCall.CheckedType, vectorized.CheckedType);
    }

    [Fact]
    public void TestVectorizedFusionEvaluatorUsesLogicalLaneSemantics()
    {
        var config = CreateEvaluatorCacheConfig();
        IRArray<AttentionDimKind> qkvLayout =
            [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim];
        IRArray<AttentionDimKind> attentionLayout =
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq];
        var q = CreateBFloat16Tensor([8, 2, 16], 0.01f);
        var k = CreateBFloat16Tensor([8, 1, 16], 0.02f);
        var v = CreateBFloat16Tensor([8, 1, 16], 0.03f);
        var qScale = Tensor.From(Enumerable.Repeat(1.25f, 16).ToArray(), [16]).CastTo(DataTypes.BFloat16);
        var kScale = Tensor.From(Enumerable.Repeat(0.75f, 16).ToArray(), [16]).CastTo(DataTypes.BFloat16);
        var bias = Tensor.Zeros(DataTypes.BFloat16, [16]);
        var cos = Tensor.From(Enumerable.Repeat(1f, 8 * 16).ToArray(), [8, 1, 16]);
        var sin = Tensor.Zeros(DataTypes.Float32, [8, 1, 16]);
        var qStats = IR.F.NN.NormStats(2, q, useMean: false).Evaluate().AsTensor();
        var kStats = IR.F.NN.NormStats(2, k, useMean: false).Evaluate().AsTensor();

        var packedQ = IR.F.Tensors.Pack(q, [8], [2]).Evaluate().AsTensor();
        var packedK = IR.F.Tensors.Pack(k, [8], [2]).Evaluate().AsTensor();
        var packedV = IR.F.Tensors.Pack(v, [8], [2]).Evaluate().AsTensor();
        var packedQScale = IR.F.Tensors.Pack(qScale, [8], [0]).Evaluate().AsTensor();
        var packedKScale = IR.F.Tensors.Pack(kScale, [8], [0]).Evaluate().AsTensor();
        var packedBias = IR.F.Tensors.Pack(bias, [8], [0]).Evaluate().AsTensor();
        var packedCos = IR.F.Tensors.Pack(cos, [2, 8], [2, 2]).Evaluate().AsTensor();
        var packedSin = IR.F.Tensors.Pack(sin, [2, 8], [2, 2]).Evaluate().AsTensor();
        var cache = CreateReferenceCache(config, 8);
        var cacheValue = Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(cache));
        var cacheVar = new Var(
            "cache",
            TensorType.Scalar(new ReferenceType(new PagedAttentionKVCacheType { Config = config })));

        var fused = IR.F.NN.QKVRoPEWithCache(
            new IR.Tuple(
                Const.FromTensor(packedQ),
                Const.FromTensor(packedK),
                Const.FromTensor(packedV)),
            qStats,
            kStats,
            packedQScale,
            packedKScale,
            packedBias,
            packedBias,
            packedCos,
            packedSin,
            cacheVar,
            0,
            2,
            1e-6f,
            false,
            2,
            2e-6f,
            false,
            qkvLayout,
            attentionLayout);
        Assert.True(fused.InferenceType(), fused.CheckedType.ToString());
        var actual = fused.Evaluate(
            new Dictionary<IVar, IValue>
            {
                [cacheVar] = Value.FromTensor(cacheValue),
            }).AsTensors()[0];

        var expectedNorm = IR.F.NN.NormApply(
            2,
            1e-6f,
            q,
            qStats,
            qScale,
            bias,
            useMean: false);
        var permutation = AttentionLayoutUtility.GetPermutation(qkvLayout, attentionLayout);
        var expected = IR.F.Tensors.Pack(
                IR.F.Tensors.Transpose(IR.F.NN.RoPE(expectedNorm, cos, sin), permutation),
                [8],
                [1])
            .Evaluate()
            .AsTensor();

        Assert.Equal(expected.ToOrtTensor(), actual.ToOrtTensor());
    }

    private static PagedAttentionConfig CreateCacheConfig() => new(
        1,
        1,
        16,
        DataTypes.BFloat16,
        256,
        [
            PagedKVCacheDimKind.NumBlocks,
            PagedKVCacheDimKind.NumLayers,
            PagedKVCacheDimKind.KV,
            PagedKVCacheDimKind.NumKVHeads,
            PagedKVCacheDimKind.BlockSize,
            PagedKVCacheDimKind.HeadDim,
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
        [SBP.SContiguous([0])]);

    private static PagedAttentionConfig CreateEvaluatorCacheConfig() => new(
        1,
        1,
        16,
        DataTypes.BFloat16,
        256,
        [
            PagedKVCacheDimKind.NumBlocks,
            PagedKVCacheDimKind.NumLayers,
            PagedKVCacheDimKind.KV,
            PagedKVCacheDimKind.NumKVHeads,
            PagedKVCacheDimKind.BlockSize,
            PagedKVCacheDimKind.HeadDim,
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
        [PagedKVCacheDimKind.HeadDim],
        [8],
        [8],
        [PagedKVCacheDimKind.NumBlocks],
        [SBP.SContiguous([0])]);

    private static Tensor CreateBFloat16Tensor(long[] shape, float step)
    {
        var values = Enumerable.Range(1, checked((int)shape.Aggregate(1L, (product, extent) => product * extent)))
            .Select(index => index * step)
            .ToArray();
        return Tensor.From(values, shape).CastTo(DataTypes.BFloat16);
    }

    private static RefPagedAttentionKVCache CreateReferenceCache(
        IPagedAttentionConfig config,
        int numTokens)
    {
        var placement = new Placement([1], "b", "b");
        var slotMappingType = config.GetSlotMappingTensorType(numTokens);
        var slotMapping = Tensor.Zeros(slotMappingType.DType, slotMappingType.Shape.ToValueArray()).Cast<long>();
        for (long tokenId = 0; tokenId < numTokens; tokenId++)
        {
            RefPagedAttentionKVCache.MaterializeSlotMappingId(
                slotMapping,
                [tokenId, 0],
                tokenId,
                1,
                placement,
                config);
        }

        var blockTablesType = config.GetBlockTablesTensorType(1, numTokens);
        var blockTables = Tensor.Zeros(blockTablesType.DType, blockTablesType.Shape.ToValueArray()).Cast<long>();
        var cacheType = config.GetLogicalShardTensorType(1, placement, AttentionCacheKind.Key);
        return new RefPagedAttentionKVCache(
            config,
            1,
            numTokens,
            Tensor.From([0L]),
            Tensor.From([(long)numTokens]),
            blockTables,
            slotMapping,
            1,
            Tensor.Zeros(cacheType.DType, cacheType.Shape.ToValueArray()));
    }
}
