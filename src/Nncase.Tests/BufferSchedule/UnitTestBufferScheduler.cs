// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Google.OrTools.Sat;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.BufferSchedule;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Transforms;
using Nncase.Schedule.Bufferize;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.BufferScheduleTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestBufferScheduler : TestClassBase
{
    public UnitTestBufferScheduler()
    {
        DefaultTargetName = Targets.CPUTarget.Kind;
        CompileOptions.TargetOptions = new Targets.NTTTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.Schedule | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    public static TheoryData<Func<Function>, int, int> ScheduleGetItemDatas
    { get; } = new()
    {
        { SampleSwish, 1648, 0 },
    };

    public static Function SampleSwish()
    {
        var ttype = new TensorType(DataTypes.Float32, new[] { 100 });
        var dtype = new DistributedType(ttype, new[] { SBP.B }, new(new[] { 1 }, "b", "b"));
        var a = new Var("a", ttype);
        var b = new Var("b", ttype);
        var boxa = IR.F.Distributed.Boxing(a, dtype);
        var boxb = IR.F.Distributed.Boxing(b, dtype);
        var tp = new IR.Tuple([boxa, boxb]);
        var tc = new TensorConst(Tensor.FromScalar(1.0f, [100]), new[] { SBP.B }, new(new[] { 1 }, "b", "b"));
        var c = IR.F.Math.Sin(tc);
        var d = IR.F.Math.Cos(c);
        var e = IR.F.Math.Neg(d);
        var f = IR.F.Math.Abs(e);
        var g = IR.F.Math.Cos(f);
        var h = IR.F.Math.Neg(g);
        var i = IR.F.Tensors.GetItem(tp, 1) + h;

        var body = new IR.Tuple(IR.F.Distributed.Boxing(IR.F.Tensors.GetItem(tp, 0), ttype), IR.F.Distributed.Boxing(i, ttype));
        return new Function("kernel", Targets.CPUTarget.Kind, body, [a, b]);
    }

    [Fact]
    public void TestNoOverLapWithZeroSize()
    {
        var memCapcity = 10;
        var model = new CpModel();
        var cons = model.AddNoOverlap2D();
        var ax = model.NewIntervalVar(0, 2, 2, "ax");
        var ay_start = model.NewIntVar(0, 0, "ay_start");
        var ay = model.NewFixedSizeIntervalVar(ay_start, memCapcity, "ay");

        var y_size = 0;
        var bx = model.NewIntervalVar(1, 2, 3, "bx");
        var by_start = model.NewIntVar(0, memCapcity - y_size, "by_start");
        var by = model.NewFixedSizeIntervalVar(by_start, y_size, "by");
        cons.AddRectangle(ax, ay);
        cons.AddRectangle(bx, by);

        var solver = new CpSolver();
        CpSolverStatus solve_status = solver.Solve(model);
        Assert.Equal(CpSolverStatus.Optimal, solve_status);
        System.Console.WriteLine(solver.Value(by_start));
    }

    [Fact]
    public void TestLetBoundBufferViewsExtendRootBufferLifetimes()
    {
        var sourcePhysical = new TIR.PhysicalBuffer(DataTypes.Float32.SizeInBytes, 256, TIR.MemoryLocation.Data);
        var destinationPhysical = new TIR.PhysicalBuffer(DataTypes.Float32.SizeInBytes, 256, TIR.MemoryLocation.Data);
        var source = new TIR.Buffer(
            "source",
            DataTypes.Float32,
            new TIR.MemSpan(sourcePhysical),
            new Dimension[] { 64 },
            new Dimension[] { 1 },
            null);
        var destination = new TIR.Buffer(
            "destination",
            DataTypes.Float32,
            new TIR.MemSpan(destinationPhysical),
            new Dimension[] { 64 },
            new Dimension[] { 1 },
            null);
        var sourceView = new Var("source_view");
        var destinationView = new Var("destination_view");
        var body = new TIR.Let(
            sourceView,
            IR.F.Buffer.BufferSubview(source, new Dimension[] { 0 }, new Dimension[] { 64 }),
            new TIR.Sequential(
                new TIR.Let(
                    destinationView,
                    IR.F.Buffer.BufferSubview(destination, new Dimension[] { 0 }, new Dimension[] { 64 }),
                    new TIR.Sequential(TIR.T.Memcopy(destinationView, sourceView)))));

        var result = new LifetimeCollector().Collect(body);

        Assert.True(result.Lifetimes[sourcePhysical].Time.Overlaps(result.Lifetimes[destinationPhysical].Time));
    }

    [Theory]
    [MemberData(nameof(ScheduleGetItemDatas))]
    public async Task TestScheduleGetItem(Func<Function> fusionGetter, int capacity, int number)
    {
        ((Targets.NTTTargetOptions)CompileOptions.TargetOptions).HierarchySizes[^1] = capacity;
        var fusion = fusionGetter();
        var vars = fusion.Parameters.AsValueEnumerable().Select(x => (Var)x).ToArray();
        var module = new IRModule(fusion);

        var inputs = vars.AsValueEnumerable().Select(v =>
        {
            var ttype = v.CheckedTensorType;
            return IR.F.Random.Normal(ttype.DType, ttype.Shape.ToValueArray()).Evaluate().AsTensor();
        }).ToArray();

        var kernelCase = new ModuleCase($"case{number}", module, vars, inputs);
        await Testing.CompileAndRun(kernelCase, CompileOptions, CompileSession, Compile);
    }

    private async Task Compile(IRModule module)
    {
        var passManager = CompileSession.CreatePassManager("pmgr");
        CompileSession.Target.RegisterAffineSelectionPass(passManager, CompileOptions);
        passManager.AddWithName<AutoTilePass>("AutoTiling_cpu", Targets.CPUTarget.Kind);
        CompileSession.Target.RegisterTIRSelectionPass(passManager, CompileOptions);
        passManager.Add<AddFunctionToModule>();
        passManager.Add<RemoveFunctionWrapperPass>();

        // todo add auto fusion merge pass here.
        passManager.Add<PrimFuncPass>().Configure(p =>
        {
            p.Add<Passes.Mutators.UnFoldBlock>();
            p.Add<Passes.Mutators.FlattenSequential>();
            p.Add<Passes.Mutators.TailLoopPeeling>();
            p.Add<Passes.Mutators.FoldConstCall>();
        });

        passManager.Add<RemoveUnusedFunctions>();
        passManager.AddWithName<BufferizePass>("BufferizePass");

        passManager.AddWithName<PrimFuncPass>("InstStage").Configure(p =>
        {
            p.Add<Passes.Mutators.FlattenBuffer>();
            p.Add<Passes.Mutators.FoldConstCall>();
            p.Add<Passes.Mutators.RemoveNop>();
        });
        CompileSession.Target.RegisterTargetDependentBeforeCodeGen(passManager, CompileSession.CompileOptions);
        await passManager.RunAsync(module);
    }
}
