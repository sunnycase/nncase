// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.CodeGen;
using Nncase.CodeGen.NTT;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Runtime.Interop;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;
using static Nncase.IR.F.Tensors;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Tests.TargetTest;

[Collection(nameof(NotThreadSafeResourceCollection))]
[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCPUTarget : TestClassBase
{
    public UnitTestCPUTarget()
    {
        DefaultTargetName = CPUTarget.Kind;
        CompileOptions.TargetOptions = new NTTTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.EGraphCost | DumpFlags.CodeGen | DumpFlags.Compile;
#else
        CompileOptions.DumpFlags = DumpFlags.CodeGen;
#endif
    }

    public static IEnumerable<object[]> TestGetItemData =>
        new[]
        {
            new object[] { new[] { 0, 1 } },
            new object[] { new[] { 0, -1 } },
        };

    public static IEnumerable<object[]> TestIfData =>
        new[]
        {
            new object[] { true },
            new object[] { false },
        };

    [Fact]
    [AutoSetupTestMethod(InitSession = false)]
    public void TestCPUTargetKind()
    {
        Assert.Equal("cpu", CPUTarget.Kind);
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = false)]
    public void TestCreateCPUTarget()
    {
        var target = CompilerServices.GetTarget(CPUTarget.Kind);
        Assert.NotNull(target);
    }

    [Theory]
    [CombinatorialData]
    public void TestCreateCPUModuleBuilder([CombinatorialValues("cpu")] string moduleKind)
    {
        var moduleBuilder = CompileSession.Target.GetModuleCompiler(moduleKind).CreateModuleBuilder(CompileOptions);
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public void TestSimpleCodeGen()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
        TestCodeGen(y, new[] { x });
    }

    [Fact]
    public void TestCodeGenUseVarMultiTimes()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f + x;
        TestCodeGen(y, new[] { x });
    }

    [Fact]
    public void TestCodeGenTuple()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f + x;
        var z = y * 2.0f;
        TestCodeGen(new IR.Tuple(y, z), new[] { x });
    }

    [Fact]
    public void TestQKVParallelLinearNTTCodeGen()
    {
        var input = CreateBuffer("input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [2, 3], [3, 1]);
        var qWeight = CreateBuffer("q_weight", DataTypes.Float32, TIR.MemoryLocation.Data, 24, [3, 4], [4, 1]);
        var kWeight = CreateBuffer("k_weight", DataTypes.Float32, TIR.MemoryLocation.Data, 72, [3, 2], [2, 1]);
        var vWeight = CreateBuffer("v_weight", DataTypes.Float32, TIR.MemoryLocation.Data, 96, [3, 2], [2, 1]);
        var qBias = CreateBuffer("q_bias", DataTypes.Float32, TIR.MemoryLocation.Data, 120, [4], [1]);
        var qOutput = CreateBuffer("q_output", DataTypes.Float32, TIR.MemoryLocation.Output, 0, [2, 4], [4, 1]);
        var kOutput = CreateBuffer("k_output", DataTypes.Float32, TIR.MemoryLocation.Output, 32, [2, 2], [2, 1]);
        var vOutput = CreateBuffer("v_output", DataTypes.Float32, TIR.MemoryLocation.Output, 48, [2, 2], [2, 1]);
        var body = new TIR.Sequential(
            TIR.F.NTT.QKVParallelLinear(
                input,
                qWeight,
                kWeight,
                vWeight,
                qBias,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                qOutput,
                kOutput,
                vOutput,
                2,
                1),
            TIR.T.Return(qOutput, kOutput, vOutput));
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, Array.Empty<IVar>())
        {
            SchedResult =
            {
                IsScheduled = true,
                DataUsage = 136,
                OutputUsage = 64,
                DataAlign = 8,
                OutputAlign = 8,
            },
        };

        Assert.True(main.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor((NTTTargetOptions)CompileOptions.TargetOptions);
        visitor.Visit(main);
        var kernelSource = visitor.GetCSource().Kernel;
        Assert.Contains("qkv_parallel_linear", kernelSource, StringComparison.Ordinal);
        Assert.Equal(3, CountOccurrences(kernelSource, "matmul<false, false, false>"));
        Assert.Contains("binary<ops::add>", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackedQKVParallelLinearNTTCodeGen()
    {
        var packedType = new VectorType(DataTypes.Float32, [2, 2]);
        var input = CreateBuffer("input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [2, 3], [3, 1]);
        var qWeight = CreateBuffer("q_weight", packedType, TIR.MemoryLocation.Data, 24, [1, 3], [3, 1]);
        var kWeight = CreateBuffer("k_weight", packedType, TIR.MemoryLocation.Data, 72, [1, 3], [3, 1]);
        var vWeight = CreateBuffer("v_weight", packedType, TIR.MemoryLocation.Data, 120, [1, 3], [3, 1]);
        var qBias = CreateBuffer("q_bias", packedType, TIR.MemoryLocation.Data, 168, [1], [1]);
        var qOutput = CreateBuffer("q_output", packedType, TIR.MemoryLocation.Output, 0, [2, 1], [1, 1]);
        var kOutput = CreateBuffer("k_output", packedType, TIR.MemoryLocation.Output, 32, [2, 1], [1, 1]);
        var vOutput = CreateBuffer("v_output", packedType, TIR.MemoryLocation.Output, 64, [2, 1], [1, 1]);
        var body = new TIR.Sequential(
            TIR.F.NTT.PackedQKVParallelLinear(
                input,
                qWeight,
                kWeight,
                vWeight,
                qBias,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                qOutput,
                kOutput,
                vOutput,
                2,
                1),
            TIR.T.Return(qOutput, kOutput, vOutput));
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, Array.Empty<IVar>())
        {
            SchedResult =
            {
                IsScheduled = true,
                DataUsage = 184,
                OutputUsage = 96,
                DataAlign = 8,
                OutputAlign = 8,
            },
        };

        Assert.True(main.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor((NTTTargetOptions)CompileOptions.TargetOptions);
        visitor.Visit(main);
        var kernelSource = visitor.GetCSource().Kernel;
        Assert.Contains("packed_qkv_parallel_linear", kernelSource, StringComparison.Ordinal);
        Assert.Equal(3, CountOccurrences(kernelSource, "packed_matmul<false>"));
        Assert.Contains("binary<ops::add>", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestMatMulGluNTTCodeGen()
    {
        var input = CreateBuffer("input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [2, 3], [3, 1]);
        var gateWeight = CreateBuffer("gate_weight", DataTypes.Float32, TIR.MemoryLocation.Data, 24, [3, 4], [4, 1]);
        var upWeight = CreateBuffer("up_weight", DataTypes.Float32, TIR.MemoryLocation.Data, 72, [3, 4], [4, 1]);
        var gateBias = CreateBuffer("gate_bias", DataTypes.Float32, TIR.MemoryLocation.Data, 120, [4], [1]);
        var output = CreateBuffer("output", DataTypes.Float32, TIR.MemoryLocation.Output, 0, [2, 4], [4, 1]);
        var body = new TIR.Sequential(
            TIR.F.NTT.MatMulGlu(
                input,
                gateWeight,
                upWeight,
                gateBias,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                output,
                IR.NN.GluType.SwiGLU),
            TIR.T.Return(output));
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, Array.Empty<IVar>())
        {
            SchedResult =
            {
                IsScheduled = true,
                DataUsage = 136,
                OutputUsage = 32,
                DataAlign = 8,
                OutputAlign = 8,
            },
        };

        Assert.True(main.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor((NTTTargetOptions)CompileOptions.TargetOptions);
        visitor.Visit(main);
        var kernelSource = visitor.GetCSource().Kernel;
        Assert.Contains("matmul_glu", kernelSource, StringComparison.Ordinal);
        Assert.Equal(2, CountOccurrences(kernelSource, "matmul<false, false, false>"));
        Assert.Contains("unary<ops::swish>", kernelSource, StringComparison.Ordinal);
        Assert.Contains("binary<ops::mul>", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackedMatMulGluNTTCodeGen()
    {
        var packedType = new VectorType(DataTypes.Float32, [2, 2]);
        var input = CreateBuffer("input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [2, 3], [3, 1]);
        var gateWeight = CreateBuffer("gate_weight", packedType, TIR.MemoryLocation.Data, 24, [1, 3], [3, 1]);
        var upWeight = CreateBuffer("up_weight", packedType, TIR.MemoryLocation.Data, 72, [1, 3], [3, 1]);
        var gateBias = CreateBuffer("gate_bias", packedType, TIR.MemoryLocation.Data, 120, [1], [1]);
        var output = CreateBuffer("output", packedType, TIR.MemoryLocation.Output, 0, [2, 1], [1, 1]);
        var body = new TIR.Sequential(
            TIR.F.NTT.PackedMatMulGlu(
                input,
                gateWeight,
                upWeight,
                gateBias,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                None.Default,
                output,
                IR.NN.GluType.SwiGLU),
            TIR.T.Return(output));
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, Array.Empty<IVar>())
        {
            SchedResult =
            {
                IsScheduled = true,
                DataUsage = 136,
                OutputUsage = 32,
                DataAlign = 8,
                OutputAlign = 8,
            },
        };

        Assert.True(main.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor((NTTTargetOptions)CompileOptions.TargetOptions);
        visitor.Visit(main);
        var kernelSource = visitor.GetCSource().Kernel;
        Assert.Contains("packed_matmul_glu", kernelSource, StringComparison.Ordinal);
        Assert.Equal(2, CountOccurrences(kernelSource, "packed_matmul<false>"));
        Assert.Contains("unary<ops::swish>", kernelSource, StringComparison.Ordinal);
        Assert.Contains("binary<ops::mul>", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCodeGenVisitLeafVar()
    {
        Assert.Throws<InvalidOperationException>(() => TestCodeGen(Var.Scalar("x", DataTypes.Float32), Array.Empty<Var>()));
    }

    [Fact]
    public void TestSimpleBinary()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { 2.0f });
    }

    [Fact]
    public void TestSimpleUnary()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = IR.F.Math.Abs(x);
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { -1.0f }, new[] { 1.0f });
    }

    [Fact]
    public void TestCodegenCallParamOrder()
    {
        // order is true: x - 3 = 2 - 3 = -1
        // order is false: 3 - x = 3 - 2 = 1
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x - 3f;
        var main = new Function("main", y, new[] { x });
        GenerateKModelAndRunFromFn(main, new[] { 2f }, (Tensor)new[] { -1f });
    }

    [Fact]
    public void TestSimpleTupleOutput()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var main = new Function("main", new IR.Tuple(x + 1.0f, x * 3.0f), new[] { x });
        var module = new IRModule(main);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { (Tensor)2.0f, 3.0f });
    }

    [Fact]
    public void TestTupleOrder()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var main = new Function("main", new IR.Tuple(x + 1.0f, x + 2f, x + 3f), new[] { x });
        GenerateKModelAndRunFromFn(main, new[] { 1f }, new[] { (Tensor)2f, 3f, 4f });
    }

    [Theory]
    [MemberData(nameof(TestGetItemData))]
    public void TestGetItem(int[] index)
    {
        var input = Tensor.From(new[] { 1, 2, 3, 4, 5, 6 }, [1, 2, 3]);
        var x = new Var("x", new TensorType(DataTypes.Int32, new[] { 1, 2, 3 }));
        var second = GetItem(x, index);
        var main = new Function("main", second, new[] { x });
        var dict = new Dictionary<IVar, IValue>() { { x, Value.FromTensor(input) } };
        GenerateKModelAndRunFromFn(main, input, second.Evaluate(dict).AsTensor());
    }

    [Fact]
    public void TestCallFunction()
    {
        var a = new Var("a", TensorType.Scalar(DataTypes.Float32));
        var b = a + 1.0f;
        var funcA = new Function("funcA", b, new[] { a });

        var x = new Var("x", TensorType.Scalar(DataTypes.Float32));
        var y = new Call(funcA, x + 1.0f);
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        module.Add(funcA);
        GenerateKModelAndRun(module, new[] { 1.0f }, new[] { 3.0f });
    }

    [Theory(Skip = "Ntt doesn't support call other functions yet")]
    [MemberData(nameof(TestIfData))]
    public void TestIf(bool input)
    {
        using var dumpScope = new Diagnostics.DumpScope($"{input}", CompileOptions.DumpFlags);
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        var then = new Function((Expr)(-2f));
        var @else = new Function(IR.F.NN.Relu(Cast(3, DataTypes.Float32)));
        var @if = IR.F.Math.Abs(new If(condVar, then, @else));

        Assert.True(@if.InferenceType());
        var main = new Function("main", @if, new[] { condVar });

        var output = @if.Evaluate(new Dictionary<IVar, IValue> { { condVar, Value.FromTensor(input) } }).AsTensor();
        GenerateKModelAndRunFromFn(main, input, output);
    }

    [Fact(Skip = "Ntt doesn't support call other functions yet")]
    public void TestStackVMNestIf()
    {
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        _ = (Expr)3 - 1;
        var @else = new Function((Expr)3 + 1);
        var elseThen = new Function((Expr)8 * 8);
        var elsif = new If(condVar, elseThen, @else);

        var main = new Function("main", 2 * elsif, new[] { condVar });

        var input = (Tensor)true;
        var output = (Tensor)128;
        GenerateKModelAndRunFromFn(main, input, output);
    }

    [Fact(Skip = "Ntt doesn't support call other functions yet")]
    public void TestNestIfWithThenBegin()
    {
        CompileOptions.DumpFlags = DumpFlags.CodeGen;
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        var cast = Cast(condVar, DataTypes.Int32);
        var i = ShapeUtility.If(condVar, (condVar, cast) => cast * ShapeUtility.If(condVar, cast => 3 + cast, cast => 2, cast), (condVar, cast) => 6, condVar, cast);
        var main = new Function("main", i, new[] { condVar });
        Dumpper.DumpIR(main, "main");
        var input = (Tensor)true;
        var output = (Tensor)4;
        GenerateKModelAndRunFromFn(main, input, output);
    }

    [Fact(Skip = "Ntt doesn't support call other functions yet")]
    public void TestNestIfWithElseBegin()
    {
        var condVar = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        var i = ShapeUtility.If(condVar, condVar => 3, condVar => ShapeUtility.If(condVar, () => 1, () => 2), condVar);
        var main = new Function("main", i, new[] { condVar });
        var input = (Tensor)false;
        var output = (Tensor)2;
        GenerateKModelAndRunFromFn(main, input, output);
    }

    private void TestCodeGen(BaseExpr body, Var[] vars, [CallerMemberName] string? name = null)
    {
        var main = new Function("main", CPUTarget.Kind, body, vars);
        var module = new IRModule(main);
        var pmgr = CompileSession.CreatePassManager("pmgr");
        var compiler = (Nncase.Compiler.Compiler)CompileSession.Compiler;
        compiler.TIRPass(pmgr);
        pmgr.RunAsync(module).Wait();

        var modelBuilder = CompileSession.GetRequiredService<IModelBuilder>();
        var linkedModel = modelBuilder.Build(module);
        using var output = File.Open($"{name}.kmodel", FileMode.Create);
        linkedModel.Serialize(output);
        Assert.NotEqual(0, output.Length);
    }

    private void GenerateKModelAndRun(IRModule module, Tensor input, Tensor[] expectedOutput, [CallerMemberName] string? name = null)
    {
        CompileSession.Compiler.ImportIRModule(module);
        CompileSession.Compiler.CompileAsync().Wait();

        var kmodelPath = Path.Combine(CompileSession.CompileOptions.DumpDir, $"{name}.kmodel");
        using (var kmodelFile = Dumpper.OpenFile($"{name}.kmodel"))
        {
            CompileSession.Compiler.Gencode(kmodelFile);
        }

        if (Dumpper.IsEnabled(DumpFlags.CodeGen))
        {
            using (var inputFile = Dumpper.OpenFile($"input.bin", FileMode.Create))
            {
                inputFile.Write(input.BytesBuffer);
            }
        }

        var interp = RTInterpreter.Create();
        interp.LoadModel(kmodelPath);
        var entry = interp.Entry;
        Assert.NotNull(entry);

        var rtInput = RTTensor.FromTensor(input);
        var rtOutput = entry!.Invoke(rtInput);
        var rtOutputs = rtOutput is RTTensor t ? new[] { t } : ((RTTuple)rtOutput).Fields.Cast<RTTensor>().ToArray();
        Assert.Equal(expectedOutput.Length, rtOutputs.Length);

        for (int i = 0; i < rtOutputs.Length; i++)
        {
            var outBuffer = rtOutputs[i].Buffer;
            var outHost = outBuffer.Buffer.AsHost()!;
            using (var mmOwner = outHost.Map(RTMapAccess.Read))
            {
                var outHostSlice = mmOwner.Memory.Slice((int)outBuffer.Start, (int)outBuffer.SizeBytes);
                Assert.Equal(expectedOutput[i].BytesBuffer.ToArray(), outHostSlice.ToArray());
            }
        }
    }

    private void GenerateKModelAndRun(IRModule module, Tensor input, Tensor expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(module, input, new[] { expectedOutput }, name);
    }

    private void GenerateKModelAndRunFromFn(Function fn, Tensor input, Tensor expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(new IRModule(fn), input, new[] { expectedOutput }, name);
    }

    private void GenerateKModelAndRunFromFn(Function fn, Tensor input, Tensor[] expectedOutput, [CallerMemberName] string? name = null)
    {
        GenerateKModelAndRun(new IRModule(fn), input, expectedOutput, name);
    }

    private TIR.Buffer CreateBuffer(string name, DataType elemType, TIR.MemoryLocation location, long startBytes, long[] dimensions, long[] strides)
    {
        var physicalElementCount = dimensions.Aggregate(1L, (acc, dim) => checked(acc * dim));
        var sizeBytes = checked(physicalElementCount * elemType.SizeInBytes);
        return new TIR.Buffer(
            name,
            elemType,
            new TIR.MemSpan(new TIR.PhysicalBuffer(elemType.SizeInBytes, startBytes, sizeBytes, location)),
            dimensions.Select(dim => (Dimension)dim).ToArray(),
            strides.Select(stride => (Dimension)stride).ToArray(),
            null);
    }

    private int CountOccurrences(string text, string value)
    {
        var count = 0;
        var index = 0;
        while ((index = text.IndexOf(value, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += value.Length;
        }

        return count;
    }
}
