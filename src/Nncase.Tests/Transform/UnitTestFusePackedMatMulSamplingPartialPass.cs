// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFusePackedMatMulSamplingPartialPass : TestClassBase
{
    [Fact]
    public async Task TestFuseVocabularyShardedPackedMatMulAndSampling()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var config = new SamplerConfig(
            vocabSize: 128,
            maxBatchSize: 1,
            maxLogprobs: 4,
            SamplerLogprobsMode.RawLogprobs);
        var lhs = new Var(
            "lhs",
            new DistributedType(
                new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }),
                [SBP.B, SBP.B],
                placement));
        var rhs = new Var(
            "rhs",
            new DistributedType(
                new TensorType(
                    new VectorType(DataTypes.BFloat16, [8, 2, 8]),
                    new long[] { 4, 16 }),
                [SBP.B, SBP.SContiguous([0, 1], 1)],
                placement));
        var state = new Var(
            "state",
            TensorType.Scalar(new ReferenceType(new SamplerStateType { Config = config })));
        var packed = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor));
        var logits = Bitcast(packed, DataTypes.BFloat16);
        var partial = IR.F.NTT.SamplingPartial(logits, state, config);
        var combine = IR.F.NTT.SamplingCombine(
            logits,
            partial[0],
            partial[1],
            state,
            config);
        var function = new Function(
            "main",
            string.Empty,
            combine,
            new IVar[] { lhs, rhs, state });
        Assert.True(
            function.InferenceType(),
            $"Function type inference failed: {function.CheckedType}; body: {CompilerServices.Print(function.Body)}");
        var originalType = function.CheckedType;

        var rewritten = Assert.IsType<Function>(
            await new FusePackedMatMulSamplingPartialPass().RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var fusedCall = Assert.Single(
            calls.Where(call => call.Target is PackedMatMulSamplingPartial));
        var fused = Assert.IsType<PackedMatMulSamplingPartial>(fusedCall.Target);

        Assert.Equal(PackedMatMulRhsLayout.KMajor, fused.RhsLayout);
        Assert.Equal(config, fused.Config);
        Assert.Same(lhs, fusedCall[PackedMatMulSamplingPartial.Lhs]);
        Assert.Same(rhs, fusedCall[PackedMatMulSamplingPartial.Rhs]);
        Assert.Same(state, fusedCall[PackedMatMulSamplingPartial.State]);
        Assert.Single(calls.Where(call => call.Target is SamplingCombine));
        Assert.DoesNotContain(
            calls,
            call => call.Target is PackedMatMul or SamplingPartial);
        Assert.True(rewritten.InferenceType());
        Assert.Equal(originalType, rewritten.CheckedType);
    }
}
