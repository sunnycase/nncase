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
using Nncase.IR.Affine;
using Nncase.IR.F;
using Nncase.IR.NN;
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
        Assert.False(target.IsAutoTilingEnabled);
    }

    [Theory]
    [CombinatorialData]
    public void TestCreateCPUModuleBuilder([CombinatorialValues("cpu")] string moduleKind)
    {
        var moduleBuilder = CompileSession.Target.GetModuleCompiler(moduleKind).CreateModuleBuilder(CompileOptions);
        Assert.NotNull(moduleBuilder);
    }

    [Fact]
    public void TestCPUHierarchyUsesOnlyPhysicalBlocks()
    {
        CompileOptions.TargetOptions = new NTTTargetOptions
        {
            Hierarchies = [[2, 4]],
            HierarchyNames = "yx",
            HierarchyLevels = "bb",
        };
        var moduleBuilder = CompileSession.Target.GetModuleCompiler(CPUTarget.Kind).CreateModuleBuilder(CompileOptions);
        Assert.NotNull(moduleBuilder);

        CompileOptions.TargetOptions = new NTTTargetOptions
        {
            Hierarchies = [[2, 4]],
            HierarchyNames = "cb",
            HierarchyLevels = "cb",
        };
        var exception = Assert.Throws<InvalidOperationException>(
            () => CompileSession.Target.GetModuleCompiler(CPUTarget.Kind).CreateModuleBuilder(CompileOptions));
        Assert.Contains("only physical block hierarchy levels", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestCPUCompilationUsesDirectSemanticTIR()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 32 }));
        var main = new Function("main", CPUTarget.Kind, IR.F.Math.Unary(UnaryOp.Abs, x), new[] { x });
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        compiler.ImportIRModule(new IRModule(main));

        await compiler.CompileAsync();

        var expressions = compiler.Module.Functions
            .SelectMany(ExprCollector.Collect)
            .ToArray();
        Assert.Contains(
            expressions.OfType<Call>(),
            call => call.Target is TIR.NTT.NTTKernelOp);
        Assert.Empty(expressions.OfType<Grid>());
        Assert.Empty(expressions.OfType<TIR.For>());
        Assert.Empty(expressions.OfType<TIR.PipelineFor>());
        Assert.DoesNotContain(
            expressions.OfType<Call>(),
            call => call.Target is TIR.TileLoad or TIR.TileStore);
        Assert.Empty(
            Directory.GetDirectories(Dumpper.Directory, "*AutoTilingPass*", SearchOption.AllDirectories));
    }

    [Fact]
    public void TestCPUCodegenRejectsScheduledTIR()
    {
        var loop = new TIR.For(
            new DimVar("tile"),
            new TIR.Range(0, 8, 4),
            TIR.LoopMode.Serial,
            new TIR.Sequential());
        var function = new TIR.PrimFunction(
            "scheduled",
            CPUTarget.Kind,
            new TIR.Sequential(loop),
            Array.Empty<IVar>())
        {
            Role = FunctionRole.ScheduledRegion,
        };
        var moduleBuilder = CompileSession.Target.GetModuleCompiler(CPUTarget.Kind).CreateModuleBuilder(CompileOptions);

        var exception = Assert.Throws<NotSupportedException>(() => moduleBuilder.Build([function]));
        Assert.Contains("does not accept AutoTiling ScheduledRegion", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCPUCodegenRejectsCompilerGeneratedLoops()
    {
        var loop = new TIR.For(
            new DimVar("tile"),
            new TIR.Range(0, 8, 4),
            TIR.LoopMode.Serial,
            new TIR.Sequential());
        var function = new TIR.PrimFunction(
            "loop",
            CPUTarget.Kind,
            new TIR.Sequential(loop),
            Array.Empty<IVar>());
        var moduleBuilder = CompileSession.Target.GetModuleCompiler(CPUTarget.Kind).CreateModuleBuilder(CompileOptions);

        var exception = Assert.Throws<NotSupportedException>(() => moduleBuilder.Build([function]));
        Assert.Contains("contains For", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCPUEntryAbiSkipsExplicitDimensionParametersWhenScanningTensorShapes()
    {
        var sequenceLength = new DimVar("sequence_length")
        {
            Metadata = new() { Range = new(1, 128) },
        };
        var input = new Var(
            "input",
            new TensorType(DataTypes.Float32, new RankedShape(sequenceLength, 64)));
        var function = new TIR.PrimFunction(
            "dynamic_entry",
            CPUTarget.Kind,
            new TIR.Sequential(),
            new IVar[] { sequenceLength, input });

        var layout = KernelEntryAbiLayout.Create(function);

        Assert.Empty(layout.DynamicDimensions);
    }

    [Fact]
    public void TestCPUPagedAttentionObjectRetainsRuntimeDescriptorLifetime()
    {
        var config = new PagedAttentionConfig(
            1,
            1,
            8,
            DataTypes.BFloat16,
            16,
            [
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.BlockSize,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.HeadDim,
            ],
            [
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.BlockSize,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.HeadDim,
            ],
            [PagedKVCacheDimKind.HeadDim],
            [PagedKVCacheDimKind.HeadDim],
            [8],
            [8],
            [],
            []);
        var cache = new Var(
            "cache",
            TensorType.Scalar(new ReferenceType(new PagedAttentionKVCacheType { Config = config })));
        var function = new TIR.PrimFunction(
            "cache_user_prim",
            CPUTarget.Kind,
            new TIR.Sequential(),
            new IVar[] { cache });
        var options = new NTTTargetOptions
        {
            Hierarchies = [[1]],
            HierarchyNames = "b",
            HierarchyLevels = "b",
        };

        var source = CSourceBuiltn.MakeBlockMain(
            function,
            0,
            "block_main_0",
            8,
            0,
            0,
            0,
            0,
            options);

        Assert.Contains("const auto &desc = pid_cache_descs[i];", source, StringComparison.Ordinal);
        Assert.DoesNotContain("auto desc = pid_cache_descs[i];", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCPUReturnWritesOnlyCallerAllocatedOutputDescriptors()
    {
        var objectType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var state = new TIR.BufferVar(
            "state",
            objectType,
            TIR.BufferVarRole.InOut,
            TIR.MemoryLocation.Input);
        var (outputParameter, output) = CreateOutputBuffer(
            "output",
            DataTypes.Float32,
            [4],
            [1]);
        var function = new TIR.PrimFunction(
            "stateful_prim",
            CPUTarget.Kind,
            new TIR.Sequential(),
            new TIR.Return(new Expr[] { output, state }),
            new IVar[] { state, outputParameter });

        Assert.True(function.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor((NTTTargetOptions)CompileOptions.TargetOptions);
        visitor.Visit(function);
        var kernelSource = visitor.GetCSource().Kernel;

        Assert.Contains("output_descs[0].data", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("output_descs[1]", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCPUCodegenMakesRepeatedBufferNamesUnique()
    {
        var firstInput = CreateBuffer("first_input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [4], [1]);
        var secondInput = CreateBuffer("second_input", DataTypes.Float32, TIR.MemoryLocation.Data, 16, [4], [1]);
        var firstOutput = CreateBuffer("repeated_result", DataTypes.Float32, TIR.MemoryLocation.Data, 32, [4], [1]);
        var secondOutput = CreateBuffer("repeated_result", DataTypes.Float32, TIR.MemoryLocation.Data, 48, [4], [1]);
        var function = new TIR.PrimFunction(
            "repeated_buffer_names_prim",
            CPUTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.Unary(UnaryOp.Abs, firstInput, firstOutput),
                TIR.F.NTT.Unary(UnaryOp.Abs, secondInput, secondOutput)),
            Array.Empty<IVar>());

        Assert.True(function.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor((NTTTargetOptions)CompileOptions.TargetOptions);
        visitor.Visit(function);
        var kernelSource = visitor.GetCSource().Kernel;

        Assert.Contains("auto repeated_result ", kernelSource, StringComparison.Ordinal);
        Assert.Contains("auto repeated_result_1 ", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCPUCodegenTreatsDistributedParametersAsLocalShardAddresses()
    {
        var targetOptions = Assert.IsType<NTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Hierarchies = [[24]];
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";

        var tensorType = new TensorType(DataTypes.Float32, new[] { 16, 24 });
        var placement = new Placement(new[] { 24 }, "b", "b");
        var distributedType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0], 1) },
            placement);
        var inputParameter = new TIR.BufferVar(
            "input_storage",
            distributedType,
            TIR.BufferVarRole.Input,
            TIR.MemoryLocation.Input,
            TIR.BufferLayoutAnnotation.ExactStrided([24, 1]));
        var outputParameter = new TIR.BufferVar(
            "output_storage",
            distributedType,
            TIR.BufferVarRole.Output,
            TIR.MemoryLocation.Output,
            TIR.BufferLayoutAnnotation.ExactStrided([24, 1]));
        var input = TIR.T.AttachBuffer(
            inputParameter,
            DistributedUtility.GetDividedTensorType(distributedType),
            TIR.MemoryLocation.Input,
            0,
            out _,
            "input",
            distributedType);
        var output = TIR.T.AttachBuffer(
            outputParameter,
            DistributedUtility.GetDividedTensorType(distributedType),
            TIR.MemoryLocation.Output,
            0,
            out _,
            "output",
            distributedType);
        var function = new TIR.PrimFunction(
            "distributed_parameter_prim",
            CPUTarget.Kind,
            new TIR.Sequential(TIR.F.NTT.Unary(UnaryOp.Abs, input, output)),
            new IVar[] { inputParameter, outputParameter });

        Assert.True(function.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor(targetOptions);
        visitor.Visit(function);
        var kernelSource = visitor.GetCSource().Kernel;

        Assert.Equal(2, kernelSource.Split("make_sharded_tensor_view_from_address", StringSplitOptions.None).Length - 1);
        Assert.DoesNotContain("make_sharded_tensor_view_from_global_buffer", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCPUCodegenSupportsUnevenContiguousReshard()
    {
        var targetOptions = Assert.IsType<NTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Hierarchies = [[24]];
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";

        var tensorType = new TensorType(DataTypes.Float32, new[] { 1, 65 });
        var placement = new Placement(new[] { 24 }, "b", "b");
        var splitType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0]) },
            placement);
        var broadcastType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var splitBuffer = new TIR.Buffer(
            "split_input",
            DataTypes.Float32,
            new TIR.MemSpan(new TIR.PhysicalBuffer(4, 0, 12, TIR.MemoryLocation.Data)),
            new Dimension[] { 1, 65 },
            new Dimension[] { 0, 1 },
            splitType);
        var broadcastBuffer = new TIR.Buffer(
            "broadcast_output",
            DataTypes.Float32,
            new TIR.MemSpan(new TIR.PhysicalBuffer(4, 16, 260, TIR.MemoryLocation.Data)),
            new Dimension[] { 1, 65 },
            new Dimension[] { 0, 1 },
            broadcastType);
        var function = new TIR.PrimFunction(
            "uneven_reshard_prim",
            CPUTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.GatherReduceScatter(
                    splitBuffer,
                    broadcastBuffer,
                    splitType,
                    broadcastType)),
            Array.Empty<IVar>());

        Assert.True(function.InferenceType());
        using var visitor = new KernelCSourceConvertVisitor(targetOptions);
        visitor.Visit(function);
        var kernelSource = visitor.GetCSource().Kernel;

        Assert.Contains("reshard(split_input, broadcast_output);", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("mesh_type::local_index", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestCPUSupportCheckerAcceptsCanonicalReduceAxes()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 2, 4 }));
        var reduce = Assert.IsType<Call>(
            IR.F.Tensors.Reduce(ReduceOp.Sum, input, new[] { 1L }, 0.0f, false));
        Assert.True(CompilerServices.InferenceType(reduce));

        Assert.True(new CPUModuleCompiler().IsSupportedCall(reduce, CompileOptions));
    }

    [Fact]
    public void TestCPUSupportCheckerAcceptsDecomposedBFloat16Norm()
    {
        var input = new Var(
            "input",
            new TensorType(DataTypes.BFloat16, new[] { 1, 1, 128 }));
        var scale = new Var(
            "scale",
            new TensorType(DataTypes.BFloat16, new[] { 128 }));
        var bias = new Var(
            "bias",
            new TensorType(DataTypes.BFloat16, new[] { 128 }));
        var stats = Assert.IsType<Call>(IR.F.NN.NormStats(2, input, useMean: false));
        Assert.True(CompilerServices.InferenceType(stats));
        var apply = Assert.IsType<Call>(
            IR.F.NN.NormApply(2, 1e-6f, input, stats, scale, bias, useMean: false));
        Assert.True(CompilerServices.InferenceType(apply));

        var compiler = new CPUModuleCompiler();
        Assert.True(compiler.IsSupportedCall(stats, CompileOptions));
        Assert.True(compiler.IsSupportedCall(apply, CompileOptions));
        Assert.Equal(DataTypes.Float32, stats.CheckedDataType);
    }

    [Fact]
    public void TestCPUSupportCheckerAcceptsCallerAllocatedStorage()
    {
        var storage = Assert.IsType<Call>(
            IR.F.Buffer.Uninitialized(
                DataTypes.UInt8,
                TIR.MemoryLocation.Data,
                new[] { 1024 }));
        Assert.True(CompilerServices.InferenceType(storage));

        Assert.True(new CPUModuleCompiler().IsSupportedCall(storage, CompileOptions));
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
        var (qOutputParameter, qOutput) = CreateOutputBuffer("q_output", DataTypes.Float32, [2, 4], [4, 1]);
        var (kOutputParameter, kOutput) = CreateOutputBuffer("k_output", DataTypes.Float32, [2, 2], [2, 1]);
        var (vOutputParameter, vOutput) = CreateOutputBuffer("v_output", DataTypes.Float32, [2, 2], [2, 1]);
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
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, [qOutputParameter, kOutputParameter, vOutputParameter])
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
        Assert.DoesNotContain("matmul<false, false, false>", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("binary<ops::add>", kernelSource, StringComparison.Ordinal);
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
        var (qOutputParameter, qOutput) = CreateOutputBuffer("q_output", packedType, [2, 1], [1, 1]);
        var (kOutputParameter, kOutput) = CreateOutputBuffer("k_output", packedType, [2, 1], [1, 1]);
        var (vOutputParameter, vOutput) = CreateOutputBuffer("v_output", packedType, [2, 1], [1, 1]);
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
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, [qOutputParameter, kOutputParameter, vOutputParameter])
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
        Assert.DoesNotContain("packed_matmul<false>", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("binary<ops::add>", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestMatMulGluNTTCodeGen()
    {
        var input = CreateBuffer("input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [2, 3], [3, 1]);
        var gateWeight = CreateBuffer("gate_weight", DataTypes.Float32, TIR.MemoryLocation.Data, 24, [3, 4], [4, 1]);
        var upWeight = CreateBuffer("up_weight", DataTypes.Float32, TIR.MemoryLocation.Data, 72, [3, 4], [4, 1]);
        var gateBias = CreateBuffer("gate_bias", DataTypes.Float32, TIR.MemoryLocation.Data, 120, [4], [1]);
        var (outputParameter, output) = CreateOutputBuffer("output", DataTypes.Float32, [2, 4], [4, 1]);
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
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, [outputParameter])
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
        Assert.Contains("matmul_swiglu", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("matmul<false, false, false>", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("unary<ops::swish>", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("binary<ops::mul>", kernelSource, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackedMatMulGluNTTCodeGen()
    {
        var packedType = new VectorType(DataTypes.Float32, [2, 2]);
        var input = CreateBuffer("input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [2, 3], [3, 1]);
        var gateWeight = CreateBuffer("gate_weight", packedType, TIR.MemoryLocation.Data, 24, [1, 3], [3, 1]);
        var upWeight = CreateBuffer("up_weight", packedType, TIR.MemoryLocation.Data, 72, [1, 3], [3, 1]);
        var gateBias = CreateBuffer("gate_bias", packedType, TIR.MemoryLocation.Data, 120, [1], [1]);
        var (outputParameter, output) = CreateOutputBuffer("output", packedType, [2, 1], [1, 1]);
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
                None.Default,
                None.Default,
                output,
                IR.NN.GluType.SwiGLU),
            TIR.T.Return(output));
        var main = new TIR.PrimFunction("main_prim", CPUTarget.Kind, body, [outputParameter])
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
        Assert.Contains("packed_matmul_swiglu", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("packed_matmul<false>", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("unary<ops::swish>", kernelSource, StringComparison.Ordinal);
        Assert.DoesNotContain("binary<ops::mul>", kernelSource, StringComparison.Ordinal);
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
        compiler.TIRSelectionPass(pmgr);
        compiler.FinalizeTIRCallGraphPass(pmgr);
        compiler.TIRLoweringPass(pmgr);
        module = pmgr.RunAsync(module).GetAwaiter().GetResult();

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

    private (TIR.BufferVar Parameter, TIR.Buffer Buffer) CreateOutputBuffer(
        string name,
        DataType elemType,
        long[] dimensions,
        long[] strides)
    {
        var parameter = new TIR.BufferVar(
            $"{name}_storage",
            new TensorType(elemType, dimensions),
            TIR.BufferVarRole.Output,
            TIR.MemoryLocation.Output,
            TIR.BufferLayoutAnnotation.ExactStrided(strides.Select(stride => (Dimension)stride).ToArray()));
        var physicalElementCount = dimensions.Aggregate(1L, (acc, dim) => checked(acc * dim));
        var sizeBytes = checked(physicalElementCount * elemType.SizeInBytes);
        var buffer = new TIR.Buffer(
            name,
            elemType,
            new TIR.MemSpan(new TIR.PhysicalBuffer(elemType.SizeInBytes, parameter, sizeBytes, TIR.MemoryLocation.Output)),
            dimensions.Select(dim => (Dimension)dim).ToArray(),
            strides.Select(stride => (Dimension)stride).ToArray(),
            null);
        return (parameter, buffer);
    }
}
