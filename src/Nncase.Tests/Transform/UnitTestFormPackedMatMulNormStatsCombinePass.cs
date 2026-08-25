// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFormPackedMatMulNormStatsCombinePass : TestClassBase
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TestMakesResidualAndNormStatsExplicitAroundPartialCapableMatMul(bool useMean)
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new long[] { 1, 64 }));
        var rhs = new Var(
            "rhs",
            new TensorType(
                new VectorType(DataTypes.BFloat16, [8, 2, 8]),
                new long[] { 4, 16 }));
        var outputWithoutAddend = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor));
        var addend = new Var("addend", outputWithoutAddend.CheckedType);
        var packed = Assert.IsType<Call>(IR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            outDataType: DataTypes.BFloat16,
            rhsLayout: PackedMatMulRhsLayout.KMajor,
            addend: addend));
        var stats = IR.F.NN.NormStats(-1, packed, useMean);
        var function = new Function(
            "main",
            string.Empty,
            new IR.Tuple(packed, stats),
            new IVar[] { lhs, rhs, addend });
        Assert.True(function.InferenceType());
        var originalType = function.CheckedType;

        var rewritten = Assert.IsType<Function>(
            await new FormPackedMatMulNormStatsCombinePass().RunAsync(function, new()));
        var calls = ExprCollector.Collect(rewritten.Body).OfType<Call>().ToArray();
        var combineCall = Assert.Single(
            calls.Where(call => call.Target is PackedMatMulNormStatsCombine));
        var combine = Assert.IsType<PackedMatMulNormStatsCombine>(combineCall.Target);
        var inner = Assert.IsType<Call>(combineCall[PackedMatMulNormStatsCombine.Input]);
        var innerPacked = Assert.IsType<PackedMatMul>(inner.Target);

        Assert.IsType<NoneType>(inner[PackedMatMul.Addend].CheckedType);
        Assert.False(innerPacked.FusedReduce);
        Assert.Equal(1, combine.Axis);
        Assert.Equal(useMean, combine.UseMean);
        Assert.Equal(addend, combineCall[PackedMatMulNormStatsCombine.Addend]);
        Assert.DoesNotContain(calls, call => call.Target is NormStats);
        Assert.DoesNotContain(
            calls,
            call => call.Target is PackedMatMul &&
                    call[PackedMatMul.Addend].CheckedType is not NoneType);
        Assert.True(rewritten.InferenceType());
        Assert.Equal(originalType, rewritten.CheckedType);
    }
}
