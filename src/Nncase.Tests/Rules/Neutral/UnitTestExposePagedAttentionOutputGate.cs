// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestExposePagedAttentionOutputGate : TestClassBase
{
    [Fact]
    public void TestExposeOutputGatePreservesAttentionSemanticRegion()
    {
        var q = new Var("q", new TensorType(DataTypes.BFloat16, new[] { 1, 16, 1 }));
        var cache = new Var("cache", AnyType.Default);
        var extra = new Var("extra", new TensorType(DataTypes.Int64, new[] { 1 }));
        var gate = new Var("gate", new TensorType(DataTypes.BFloat16, new[] { 1, 16, 1 }));
        var layerId = new DimVar("layer_id");
        var region = new SemanticRegion(
            SemanticRegionKinds.PagedAttentionKVCache,
            "layer.0.paged_attention_kv_cache");
        var attention = IR.F.NN.PagedAttention(
            q,
            cache,
            extra,
            1.0f,
            layerId,
            gate,
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            16);
        attention.Metadata.SemanticRegion = region;

        var rewritten = Assert.IsType<Call>(
            CompilerServices.Rewrite(attention, [new ExposePagedAttentionOutputGate()], new()));
        var binary = Assert.IsType<Binary>(rewritten.Target);
        Assert.Equal(BinaryOp.Mul, binary.BinaryOp);

        var ungatedAttention = Assert.IsType<Call>(rewritten[Binary.Lhs]);
        Assert.IsType<PagedAttention>(ungatedAttention.Target);
        Assert.IsType<None>(ungatedAttention[PagedAttention.OutputGate]);
        Assert.Same(region, ungatedAttention.Metadata.SemanticRegion);
        Assert.Null(rewritten.Metadata.SemanticRegion);

        var sigmoid = Assert.IsType<Call>(rewritten[Binary.Rhs]);
        Assert.IsType<Sigmoid>(sigmoid.Target);
        Assert.Same(gate, sigmoid[Sigmoid.Input]);
    }

    [Fact]
    public void TestUngatedAttentionIsNotRewritten()
    {
        var q = new Var("q", new TensorType(DataTypes.BFloat16, new[] { 1, 16, 1 }));
        var cache = new Var("cache", AnyType.Default);
        var extra = new Var("extra", new TensorType(DataTypes.Int64, new[] { 1 }));
        var layerId = new DimVar("layer_id");
        var attention = IR.F.NN.PagedAttention(
            q,
            cache,
            extra,
            1.0f,
            layerId,
            None.Default,
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            16);

        var rewritten = CompilerServices.Rewrite(
            attention,
            [new ExposePagedAttentionOutputGate()],
            new());

        Assert.Same(attention, rewritten);
    }
}
