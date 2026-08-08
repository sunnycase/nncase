// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestQKVRoPEWithCacheFusion : TestClassBase
{
    [Fact]
    public async Task TestFusionFormsHighLevelTupleOpBeforeDistribution()
    {
        var q = new Var("q", new TensorType(DataTypes.BFloat16, new[] { 1, 2, 8 }));
        var k = new Var("k", new TensorType(DataTypes.BFloat16, new[] { 1, 1, 8 }));
        var v = new Var("v", new TensorType(DataTypes.BFloat16, new[] { 1, 1, 8 }));
        var qScale = new Var("q_scale", new TensorType(DataTypes.BFloat16, new[] { 8 }));
        var kScale = new Var("k_scale", new TensorType(DataTypes.BFloat16, new[] { 8 }));
        var cos = new Var("cos", new TensorType(DataTypes.Float32, new[] { 1, 1, 8 }));
        var sin = new Var("sin", new TensorType(DataTypes.Float32, new[] { 1, 1, 8 }));
        var extra = new Var("extra", new TensorType(DataTypes.UInt8, new[] { 1 }));
        var scale = new Var("scale", TensorType.Scalar(DataTypes.BFloat16));
        var layerId = new DimVar("layer_id");
        var config = CreateCacheConfig();
        var cache = new Var(
            "cache",
            TensorType.Scalar(new ReferenceType(new PagedAttentionKVCacheType { Config = config })));
        IRArray<AttentionDimKind> layout =
            [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim];

        var qStats = IR.F.NN.NormStats(2, q, useMean: false);
        var kStats = IR.F.NN.NormStats(2, k, useMean: false);
        var qNorm = IR.F.NN.NormApply(
            2,
            1e-6f,
            q,
            qStats,
            qScale,
            Tensor.Zeros(DataTypes.BFloat16, [8]),
            useMean: false);
        var kNorm = IR.F.NN.NormApply(
            2,
            2e-6f,
            k,
            kStats,
            kScale,
            Tensor.Zeros(DataTypes.BFloat16, [8]),
            useMean: false);
        var qRoPE = IR.F.NN.RoPE(qNorm, cos, sin);
        var kRoPE = IR.F.NN.RoPE(kNorm, cos, sin);
        var cacheAfterKey = IR.F.NN.UpdatePagedAttentionKVCache(
            kRoPE,
            cache,
            layerId,
            AttentionCacheKind.Key,
            layout.ToArray());
        var cacheAfterValue = IR.F.NN.UpdatePagedAttentionKVCache(
            v,
            cacheAfterKey,
            layerId,
            AttentionCacheKind.Value,
            layout.ToArray());
        var attention = IR.F.NN.PagedAttention(qRoPE, cacheAfterValue, extra, scale, layerId, layout.ToArray(), 16);
        var function = new Function(
            "decoder",
            PyNTTTarget.Kind,
            new IR.Tuple(attention, cacheAfterValue),
            new IVar[] { q, k, v, qScale, kScale, cos, sin, extra, scale, cache, layerId });
        Assert.True(function.InferenceType());

        var rewritten = Assert.IsType<Function>(
            await new FuseQKVRoPEWithCachePass(PyNTTTarget.Kind).RunAsync(function, new()));
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
        Assert.DoesNotContain(calls, call => call.Target is NormApply or RoPE or UpdatePagedAttentionKVCache);
        Assert.True(rewritten.InferenceType());
    }

    private static PagedAttentionConfig CreateCacheConfig() => new(
        1,
        1,
        8,
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
        [SBP.S([0])]);
}
