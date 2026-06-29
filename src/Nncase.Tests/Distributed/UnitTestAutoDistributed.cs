// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
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

        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.S(0) }, placement);
        Assert.Equal(new[] { 1015L }, DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 35 }).Offset);
        Assert.Equal(new[] { 9L }, DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 35 }).Shape);

        var skinnyType = new DistributedType(new TensorType(DataTypes.Float32, new[] { 37L }), new SBP[] { SBP.S(0) }, placement);
        Assert.Equal(new[] { 0L }, DistributedUtility.GetLocalOffsetAndShape(skinnyType, new[] { 35 }).Shape);
    }
}
