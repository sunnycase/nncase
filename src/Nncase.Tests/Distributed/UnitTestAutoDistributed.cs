// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.Passes.Distributed;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
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
    public void TestNonUniformSplitCandidateIsGenerated()
    {
        var tensorType = new TensorType(DataTypes.Float32, [1024]);
        var placement = new Placement([36], "b");
        var policies = DistributedUtility.GetLeafCandidatePolicies(tensorType, placement);

        Assert.Contains(policies, policy => policy.Count == 1 && policy[0] is SBPSplit split && split.Axes.SequenceEqual(new[] { 0 }));

        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.S([0]) }, placement);
        Assert.Equal(new[] { 1015L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 35 }).Offset).ToValueArray());
        Assert.Equal(new[] { 9L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 35 }).Shape).ToValueArray());

        var skinnyType = new DistributedType(new TensorType(DataTypes.Float32, new[] { 37L }), new SBP[] { SBP.S([0]) }, placement);
        Assert.Equal(new[] { 0L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(skinnyType, new[] { 35 }).Shape).ToValueArray());
    }

    [Fact]
    public void TestDynamicSplitGranularityUsesRuntimeShape()
    {
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 1024) } };
        var tensorType = new TensorType(DataTypes.Float32, [sequenceLength]);
        var placement = new Placement([4], "y");
        var policies = DistributedUtility.GetLeafCandidatePolicies(tensorType, placement);

        var split = policies
            .Select(policy => policy.SingleOrDefault() as SBPSplit)
            .Single(policy => policy is not null && policy.Axes.SequenceEqual(new[] { 0 }))!;
        Assert.NotNull(split.Granularity);
        Assert.False(split.Granularity.IsFixed);
        Assert.True(split.Granularity.Metadata.Range.HasValue);
        Assert.Equal(1d, split.Granularity.Metadata.Range.Value.Min);
        Assert.Equal(256d, split.Granularity.Metadata.Range.Value.Max);

        var distributedType = new DistributedType(tensorType, new SBP[] { split }, placement);
        var dividedShape = DistributedUtility.GetDividedTensorType(distributedType).Shape;
        Assert.False(dividedShape[0].IsFixed);
        Assert.Equal(256d, dividedShape[0].Metadata.Range!.Value.Max);
        Assert.Equal(new[] { 256L }, DistributedUtility.GetDividedTensorType(distributedType, DistributedUtility.DivideFlags.MaxShape).Shape.ToValueArray());
        Assert.Equal(new[] { 768L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 3 }, DistributedUtility.DivideFlags.MaxShape).Offset).ToValueArray());
        Assert.Equal(new[] { 256L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 3 }, DistributedUtility.DivideFlags.MaxShape).Shape).ToValueArray());
    }
}
