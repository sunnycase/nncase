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
        var placement = new Placement([36], "b", "b");
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
        var placement = new Placement([4], "y", "b");
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

    [Fact]
    public void TestReshardPlannerDecomposesPartialToBroadcastThenSplit()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var source = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([1]) }, placement, SBP.P([1]));
        var noPartial = new DistributedType(tensorType, source.AxisPolicies, placement);
        var broadcast = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.B }, placement);
        var target = new DistributedType(tensorType, new SBP[] { SBP.S([0]), SBP.B }, placement);

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
    public void TestReshardPlannerKeepsDirectPathCompact()
    {
        var tensorType = new TensorType(DataTypes.Float32, [32, 64]);
        var placement = new Placement([4, 8], "yx", "bb");
        var source = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([1]) }, placement);
        var target = new DistributedType(tensorType, new SBP[] { SBP.S([0]), SBP.B }, placement);

        var plans = DistributedReshardPlanner.Plan(source, target, (_, _) => true);

        Assert.Single(plans);
        Assert.True(plans[0].StepTypes.SequenceEqual(new IRType[] { target }));
    }
}
