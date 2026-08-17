// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.CodeGen.PyNTT;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.Passes;
using Nncase.Passes.Distributed;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;
using TIR = Nncase.TIR;

namespace Nncase.Tests.TargetTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestPyNTTTarget : TestClassBase
{
    private static readonly JsonSerializerOptions PythonStringLiteralOptions = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public UnitTestPyNTTTarget()
    {
        DefaultTargetName = PyNTTTarget.Kind;
        CompileOptions.TargetOptions = new PyNTTTargetOptions();
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = false)]
    public void TestCreatePyNTTTarget()
    {
        var target = CompilerServices.GetTarget(PyNTTTarget.Kind);
        Assert.NotNull(target);
        Assert.Equal(PyNTTTarget.Kind, target.Name);
        Assert.False(target.IsAutoTilingEnabled);
        Assert.True(CompilerServices.GetTarget(CPUTarget.Kind).IsAutoTilingEnabled);
    }

    [Fact]
    public void TestCreatePyNTTModuleBuilder()
    {
        var moduleBuilder = CompileSession.Target.GetModuleCompiler(PyNTTTarget.Kind).CreateModuleBuilder(CompileOptions);
        Assert.NotNull(moduleBuilder);
        Assert.Equal(PyNTTTarget.Kind, moduleBuilder.ModuleKind);
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        Assert.Equal("yx", targetOptions.HierarchyNames);
        Assert.Equal(new[] { 4, 8 }, targetOptions.Hierarchies.Single());
    }

    [Fact]
    public void TestGeneratePyNTTModelDirectory()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var output = CreateOutputVar("output", new TensorType(DataTypes.Float32, new[] { 1 }));
        var main = new TIR.PrimFunction("main", PyNTTTarget.Kind, new TIR.Sequential(TIR.T.Memcopy(output, x)), new IVar[] { x, output });
        var outputDirectory = GeneratePyNTTModelDirectory("generated_model", main);

        Assert.True(File.Exists(Path.Join(outputDirectory, "__init__.py")));
        Assert.True(File.Exists(Path.Join(outputDirectory, "model.py")));
        Assert.True(File.Exists(Path.Join(outputDirectory, "metadata.json")));
        Assert.True(File.Exists(Path.Join(outputDirectory, "kernel_params.json")));
        Assert.True(File.Exists(Path.Join(outputDirectory, "specs.py")));
        Assert.True(File.Exists(Path.Join(outputDirectory, "runtime_config.py")));
        Assert.True(File.Exists(Path.Join(outputDirectory, "requirements.txt")));

        var metadata = File.ReadAllText(Path.Join(outputDirectory, "metadata.json"));
        using var document = JsonDocument.Parse(metadata);
        var root = document.RootElement;
        Assert.Equal(PyNTTTarget.Kind, root.GetProperty("target_kind").GetString());
        Assert.Equal("triton", root.GetProperty("backend").GetString());
        Assert.Equal(NTTTargetMachineCatalog.Rtx5060Ti16Gb, root.GetProperty("target_machine").GetString());
        Assert.False(root.TryGetProperty("pipeline_policy", out _));
        var function = root.GetProperty("functions").EnumerateArray().Single();
        Assert.Equal("main", function.GetProperty("name").GetString());
        Assert.Equal("x", function.GetProperty("inputs").EnumerateArray().Single().GetProperty("name").GetString());
        Assert.Equal("float32", function.GetProperty("inputs").EnumerateArray().Single().GetProperty("dtype").GetString());
        Assert.Equal(1, function.GetProperty("outputs").EnumerateArray().Single().GetProperty("shape").EnumerateArray().Single().GetInt64());

        var kernelParams = File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json"));
        using var kernelParamsDocument = JsonDocument.Parse(kernelParams);
        var kernelParamsRoot = kernelParamsDocument.RootElement;
        Assert.Equal(9, kernelParamsRoot.GetProperty("pyntt_codegen_manifest_version").GetInt32());
        var renderKernel = kernelParamsRoot
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(generatedFunction => generatedFunction.GetProperty("render_kernels").EnumerateArray())
            .Single();
        var kernelMetadata = renderKernel.GetProperty("metadata");
        var launch = kernelMetadata.GetProperty("launch");
        var targetMachine = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions)
            .TargetMachineModel;
        var targetExecution = targetMachine.Execution;
        var registerFile = targetMachine.PrivateResources[NTTTargetMachineCatalog.GpuRegisterFile];
        Assert.False(launch.TryGetProperty("num_warps", out _));
        Assert.False(launch.TryGetProperty("num_stages", out _));
        Assert.False(launch.TryGetProperty("tuning", out _));
        Assert.Equal(
            targetExecution.WorkerWidth,
            kernelMetadata.GetProperty("attrs").GetProperty("target_worker_width").GetInt32());
        Assert.Equal(
            targetExecution.ThreadsPerBlock,
            kernelMetadata.GetProperty("attrs").GetProperty("target_threads_per_block").GetInt32());
        Assert.Equal(
            targetExecution.ResidentBlocksPerComputeUnit,
            kernelMetadata.GetProperty("attrs").GetProperty("target_resident_blocks_per_compute_unit").GetInt32());
        Assert.Equal(
            registerFile.CapacityUnits,
            kernelMetadata.GetProperty("attrs").GetProperty("register_file_capacity_units").GetInt64());
        Assert.Equal(
            registerFile.AllocationGranularityUnits,
            kernelMetadata.GetProperty("attrs").GetProperty("register_file_allocation_granularity_units").GetInt32());
        Assert.False(renderKernel.TryGetProperty("pipeline_executions", out _));
        Assert.False(renderKernel.TryGetProperty("shared_arena", out _));

        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        Assert.Contains("PyNTTGeneratedModel", modelPy, StringComparison.Ordinal);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("PYNTT_KERNEL_CONFIGS", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'source': 'autotune'", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'candidates': (32, 128, 256, 512, 1024)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'num_warps': 8", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'num_stages': 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'resident_blocks_per_compute_unit': 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'producer_warps': 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'producer_registers': 24", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("'consumer_registers': 240", generatedKernelsPy, StringComparison.Ordinal);
        var specsPy = File.ReadAllText(Path.Join(outputDirectory, "specs.py"));
        Assert.Contains("TensorSpec", specsPy, StringComparison.Ordinal);
        Assert.Contains("outputs=", specsPy, StringComparison.Ordinal);

        AssertGeneratedModelImports(outputDirectory);
    }

    [Fact]
    public void TestPyNTTCodegenScopePreservesSemanticTraceForDirectTir()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 1 });
        var input = new Var("x", tensorType);
        var output = CreateOutputVar("output", tensorType);
        var inputBuffer = TIR.T.AttachBuffer(
            input,
            tensorType,
            TIR.MemoryLocation.Input,
            0,
            out _,
            "input_buffer");
        var inputDataBuffer = CreateBuffer(
            "direct_input_data",
            DataTypes.Float32,
            TIR.MemoryLocation.Data,
            0,
            [1],
            [1]);
        var placement = new Placement(new[] { 1 }, "b", "b");
        var scopeName = "fusion[op0:memcopy]";
        var body = new TIR.Sequential(
            TIR.T.CodegenScope(
                scopeName,
                new TIR.Sequential(
                    TIR.F.NTT.TensorLoad(inputDataBuffer, inputBuffer, new[] { SBP.B }, placement),
                    TIR.F.NTT.TensorStore(inputDataBuffer, output, new[] { SBP.B }, placement))));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 4,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_codegen_scope_model", main);
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var helpers = manifest.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
            .SelectMany(kernel => kernel.GetProperty("helpers").EnumerateArray())
            .ToArray();
        Assert.NotEmpty(helpers);
        Assert.All(
            helpers,
            helper => Assert.Equal(
                new[] { "data" },
                helper.GetProperty("workspace_arguments").EnumerateArray().Select(value => value.GetString()).ToArray()));
        RenderGeneratedKernels(outputDirectory);
        var generatedKernels = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("# pyntt_trace_event: begin_function:fusion[op0:memcopy]#0", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("# pyntt_trace_event: end_function:fusion[op0:memcopy]#0", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("main_prim__fusion_op0_memcopy___tensor_load", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("main_prim__fusion_op0_memcopy___output_tensor_store", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("__producer", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("__consumer", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("tle.gpu.warp_specialize", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.gpu.alloc", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.gpu.copy", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain("tile_load", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain("tile_store", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain("call_frame", generatedKernels, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.tensor([3.25], dtype=torch.float32, device='cuda')",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTEquivalentDimUsesAffineSemantics()
    {
        var baseOffset = new DimVar("base_offset");
        var tailOffset = new DimVar("tail_offset");
        var innerOffset = new DimVar("inner_offset");
        var emitter = new Nncase.CodeGen.PyNTT.PyNTTDimExpressionEmitter();
        var lhs = emitter.Emit(baseOffset + tailOffset + innerOffset) with
        {
            PythonExpression = "((base_offset + tail_offset) + inner_offset)",
            TritonExpression = "((base_offset + tail_offset) + inner_offset)",
        };
        var rhs = emitter.Emit(baseOffset + (tailOffset + innerOffset)) with
        {
            PythonExpression = "(base_offset + (tail_offset + inner_offset))",
            TritonExpression = "(base_offset + (tail_offset + inner_offset))",
        };
        var different = emitter.Emit(baseOffset + tailOffset) with
        {
            PythonExpression = "(base_offset + tail_offset)",
            TritonExpression = "(base_offset + tail_offset)",
        };

        Assert.True(lhs.IsEquivalentTo(rhs));
        Assert.False(lhs.IsEquivalentTo(different));
        Assert.DoesNotContain("Equivalence", JsonSerializer.Serialize(lhs), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestPyNTTAutoDistributedPassDumps()
    {
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.EGraphCost | DumpFlags.CodeGen | DumpFlags.Compile;
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };

        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Math.Binary(BinaryOp.Add, lhs, rhs), new[] { lhs, rhs });

        DumpScope.Current.DumpIR(main, "BeforeAutoDistributed", "AutoDistributedCheck");
        var pass = new AutoDistributedPass(false, PyNTTTarget.Kind, CompileOptions);
        var post = await pass.RunAsync(main, new());
        DumpScope.Current.DumpIR(post, "AfterAutoDistributed", "AutoDistributedCheck");

        var distributedType = CollectDistributedTypes(post)
            .FirstOrDefault(type => type.Placement.Name == "yx" && type.AxisPolicies.Any(policy => policy is SBPSplit));
        Assert.NotNull(distributedType);
        var localRegion = DistributedUtility.GetLocalShardDescriptor(distributedType, new[] { 3, 7 });
        var globalShape = ((RankedShape)distributedType.TensorType.Shape).ToValueArray();
        var localShape = localRegion.ActiveShape.ToValueArray();
        Assert.Equal(globalShape.Length, localShape.Length);
        for (var axis = 0; axis < globalShape.Length; axis++)
        {
            Assert.InRange(localShape[axis], 0, globalShape[axis]);
            for (long localIndex = 0; localIndex < localShape[axis]; localIndex++)
            {
                Assert.InRange(
                    localRegion.Axes[axis].MapLocalToGlobal(localIndex).FixedValue,
                    0,
                    globalShape[axis] - 1);
            }
        }

        Assert.True(TensorUtilities.GetProduct(localShape) < TensorUtilities.GetProduct(globalShape));

        var dumpFiles = Directory.GetFiles(Dumpper.Directory, "*", SearchOption.AllDirectories);
        Assert.Contains(dumpFiles, path => path.Contains("AutoDistributedPass", StringComparison.Ordinal) && Path.GetFileName(path).Contains("Start", StringComparison.Ordinal));
        Assert.Contains(dumpFiles, path => path.Contains("AutoDistributedPass", StringComparison.Ordinal) && Path.GetFileName(path).Contains("End", StringComparison.Ordinal));
        Assert.Contains(dumpFiles, path => path.Contains("AutoDistributedCheck", StringComparison.Ordinal) && Path.GetFileName(path).Contains("AfterAutoDistributed", StringComparison.Ordinal));
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedCodegenRun()
    {
        ConfigureAutoDistributedPyNTT();

        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Math.Binary(BinaryOp.Add, lhs, rhs), new[] { lhs, rhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_auto_dist_run_model", main);
        AssertTIRPipelineDump();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var kernel = document.RootElement.GetProperty("functions").EnumerateArray().Single()
            .GetProperty("generated_kernels").EnumerateArray().Single();
        Assert.Equal("binary", kernel.GetProperty("op_kind").GetString());
        var attrs = kernel.GetProperty("attrs");
        Assert.True(attrs.GetProperty("tir").GetBoolean());
        Assert.Equal("add", attrs.GetProperty("op").GetString());
        var launchMeta = kernel.GetProperty("launch").GetProperty("meta");
        Assert.True(launchMeta.GetProperty("data_pool_bytes").GetInt64() > 0);
        Assert.True(launchMeta.GetProperty("data_pool_elements").GetInt64() > 0);
        Assert.Equal("uint8", launchMeta.GetProperty("data_dtype").GetString());
        Assert.Equal(0, launchMeta.GetProperty("rdata_pool_bytes").GetInt64());
        Assert.Equal(0, launchMeta.GetProperty("chip_local_rdata_pool_bytes").GetInt64());
        Assert.Equal(0, launchMeta.GetProperty("block_local_rdata_pool_bytes").GetInt64());
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("thread", "rdata", "pool_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("warp", "rdata", "pool_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("thread", "rdata", "stride_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("warp", "rdata", "stride_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("warp", "data", "pool_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("warp", "data", "scope_count"), out _));
        var sharding = kernel.GetProperty("launch").GetProperty("sharding");
        Assert.Equal("local_shard", sharding.GetProperty("strategy").GetString());
        Assert.Equal("yx", sharding.GetProperty("placement_axis").GetString());
        Assert.Equal(0, sharding.GetProperty("tensor_axis").GetInt32());
        Assert.Equal(new[] { 4, 8 }, sharding.GetProperty("hierarchy").EnumerateArray().Select(value => value.GetInt32()).ToArray());

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        var tensorLoadHelpers = Regex.Matches(generatedKernelsPy, @"^# TensorLoad:", RegexOptions.Multiline);
        Assert.Equal(2, tensorLoadHelpers.Count);
        Assert.Contains("generated from PyNTT Jinja TensorRegionCopy.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja ElementwiseBinary.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("generated from PyNTT Jinja TensorStore.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim__tensor_load_0__0(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim__tensor_load_1__0(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim__elementwise_binary__0(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("def main_prim__output_tensor_store__0(", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedBlockBarrierChain(
            generatedKernelsPy,
            "main_prim_binary_0",
            "main_prim__tensor_load_1__0",
            "main_prim__elementwise_binary__0");
        Assert.Contains("shard_coord0 = tle.shard_id(PYNTT_GRID_MESH, 'block_y')", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shard_coord1 = tle.shard_id(PYNTT_GRID_MESH, 'block_x')", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shard_index = (shard_coord0 * 8 + shard_coord1)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("((8 * shard_coord0) + shard_coord1)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tl.program_id(0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("shard_index //", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("shard_index %", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("result = value0 + value1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tl.store(destination +", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("data, rdata, chip_local_rdata, chip_local_data, block_local_rdata, block_local_data", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.gpu.alloc", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.gpu.copy", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("pipeline_executions", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tile_load", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tile_store", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("thread", "rdata"), generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("warp", "rdata"), generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("warp", "data"), generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("from pyntt.backends.triton.kernels", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("elementwise_binary(input0, input1, output0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("make_data_tensor_view", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("elementwise_binary_tensor", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.distributed_barrier()", generatedKernelsPy, StringComparison.Ordinal);

        var rdataPy = File.ReadAllText(Path.Join(outputDirectory, "rdata.py"));
        Assert.Contains("RDATA_BUNDLES", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"main_prim\"", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"chip_local_rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"block_local_rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("thread", "rdata"), rdataPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("warp", "rdata"), rdataPy, StringComparison.Ordinal);

        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        Assert.Contains("grid = (32,)", modelPy, StringComparison.Ordinal);
        Assert.Contains("from .generated_kernels import main_prim_binary_0", modelPy, StringComparison.Ordinal);
        Assert.Contains("pyntt_prepared_kernel.launch(", modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain("from . import generated_kernels as _generated_kernels", modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain("_generated_kernels.", modelPy, StringComparison.Ordinal);
        Assert.Contains("data = self.allocate_workspace(inputs, ", modelPy, StringComparison.Ordinal);
        Assert.Contains("block_local_data = self.allocate_workspace(inputs, ", modelPy, StringComparison.Ordinal);
        Assert.Contains("rdata, chip_local_rdata, block_local_rdata = self.materialize_rdata_bundle(inputs, \"main_prim\")", modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("thread", "rdata"), modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("warp", "rdata"), modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("warp", "data"), modelPy, StringComparison.Ordinal);
        AssertGeneratedModelRunsBinaryAdd(outputDirectory);
    }

    [Fact]
    public async Task TestPyNTTShapeBucketMainUsesOutputBufferAbi()
    {
        ConfigureAutoDistributedPyNTT();
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.CodeGen;
        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 3;
        CompileOptions.ShapeBucketOptions.SegmentRanges["n"] = [1, 4, 8];

        var dimN = new DimVar("n") { Metadata = { Range = new(1, 8) } };
        var shape = new RankedShape(new Dimension[] { dimN, 1 });
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, shape));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, shape));
        CompileOptions.ShapeBucketOptions.VarMap.Add(lhs, shape.ToArray());
        CompileOptions.ShapeBucketOptions.VarMap.Add(rhs, shape.ToArray());
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Math.Binary(BinaryOp.Add, lhs, rhs), new[] { lhs, rhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_shape_bucket_abi_model", main);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var function = document.RootElement.GetProperty("functions").EnumerateArray()
            .Single(item => item.GetProperty("is_entry").GetBoolean());
        Assert.Equal("main_prim", function.GetProperty("name").GetString());
        Assert.Equal("output0", function.GetProperty("outputs").EnumerateArray().Single().GetProperty("name").GetString());
        Assert.All(
            function.GetProperty("inputs").EnumerateArray().Where(input => input.GetProperty("dtype").GetString() != "object"),
            input => Assert.Equal("cuda", input.GetProperty("device").GetString()));

        var finalMainTir = Directory.GetFiles(Dumpper.Directory, "main_prim.script", SearchOption.AllDirectories)
            .Select(File.ReadAllText)
            .Last(text => text.Contains("T.PrimFunc(\"main_prim\"", StringComparison.Ordinal) && text.Contains("-> ()", StringComparison.Ordinal));
        Assert.Contains("%out_", finalMainTir, StringComparison.Ordinal);
        Assert.Contains("main_segment_", finalMainTir, StringComparison.Ordinal);
        Assert.DoesNotContain("Return(", finalMainTir, StringComparison.Ordinal);

        using var kernelParams = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var manifestFunctions = kernelParams.RootElement.GetProperty("functions").EnumerateArray().ToArray();
        var entryManifest = manifestFunctions.Single(item => item.GetProperty("name").GetString() == "main_prim");
        Assert.Empty(entryManifest.GetProperty("render_kernels").EnumerateArray());

        var segmentManifests = manifestFunctions
            .Where(item => item.GetProperty("name").GetString() is string name &&
                name.StartsWith("main_segment_", StringComparison.Ordinal) &&
                name.EndsWith("_prim", StringComparison.Ordinal))
            .ToArray();
        Assert.Equal(3, segmentManifests.Length);
        var topKernelNames = segmentManifests
            .Select(segment => segment.GetProperty("render_kernels").EnumerateArray().Single()
                .GetProperty("metadata").GetProperty("name").GetString()!)
            .ToArray();
        Assert.Equal(3, topKernelNames.Distinct(StringComparer.Ordinal).Count());
        Assert.All(
            manifestFunctions.Where(item => item.GetProperty("name").GetString()?.Contains("shape_bucket_selector", StringComparison.Ordinal) == true),
            selector => Assert.Empty(selector.GetProperty("render_kernels").EnumerateArray()));

        RenderGeneratedKernels(outputDirectory);
        var generatedKernels = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        foreach (var topKernelName in topKernelNames)
        {
            Assert.Contains($"def {topKernelName}(", generatedKernels, StringComparison.Ordinal);
        }

        var generatedFunctionNames = Regex.Matches(generatedKernels, @"^def (?<name>[A-Za-z_]\w*)\(", RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
        Assert.Equal(generatedFunctionNames.Length, generatedFunctionNames.Distinct(StringComparer.Ordinal).Count());
        Assert.All(
            segmentManifests,
            segment =>
            {
                var segmentName = segment.GetProperty("name").GetString()!;
                Assert.Contains($"def {segmentName}_device_0(", generatedKernels, StringComparison.Ordinal);
            });

        Assert.DoesNotContain("def main_prim_", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain("shape_bucket_selector", generatedKernels, StringComparison.Ordinal);

        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        Assert.Contains("if (shape_env[\"n\"] <=", modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain("main_prim.data", modelPy, StringComparison.Ordinal);
        foreach (var topKernelName in topKernelNames)
        {
            Assert.Contains($"from .generated_kernels import {topKernelName}", modelPy, StringComparison.Ordinal);
            Assert.Contains("pyntt_prepared_kernel.launch(", modelPy, StringComparison.Ordinal);
        }

        var runtimeStatements = topKernelNames
            .Select(name => $"os.environ['PYNTT_TUNE_{name.ToUpperInvariant()}_BLOCK_SIZE'] = '128'")
            .Concat(new[]
            {
                "for n in (1, 4, 8):",
                "    lhs = torch.arange(n, dtype=torch.float32, device='cuda').reshape(n, 1)",
                "    rhs = lhs * 0.25",
                "    output = module(lhs, rhs)",
                "    torch.testing.assert_close(output, lhs + rhs, rtol=0, atol=1e-6)",
            })
            .ToArray();
        AssertGeneratedModelRuns(outputDirectory, runtimeStatements);
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedUnaryRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Math.Unary(UnaryOp.Neg, x), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_unary_run_model", main);
        AssertGeneratedKernel(outputDirectory, "unary", "ElementwiseUnary.py.jinja");
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(32, dtype=torch.float32, device='cuda').reshape(32, 1) - 8",
            "output = module(x)",
            "torch.testing.assert_close(output, -x, rtol=0, atol=1e-6)");
    }

    [Fact]
    public async Task TestPyNTTDirectSelectedTirChainRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1024 }));
        var value = IR.F.Math.Unary(UnaryOp.Abs, IR.F.Math.Unary(UnaryOp.Neg, x));
        var main = new Function("main", PyNTTTarget.Kind, value, new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_direct_tir_chain_model", main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var expressions = compiler.Module.Functions
            .SelectMany(ExprCollector.Collect)
            .ToArray();
        Assert.DoesNotContain(
            expressions.OfType<TIR.PhysicalBuffer>(),
            buffer => buffer.Location is TIR.MemoryLocation.Shared or TIR.MemoryLocation.Register);
        Assert.Empty(expressions.OfType<Nncase.IR.Affine.Grid>());
        Assert.Empty(expressions.OfType<TIR.For>());
        Assert.Empty(expressions.OfType<TIR.PipelineFor>());
        Assert.DoesNotContain(
            expressions.OfType<Call>(),
            call => call.Target is TIR.TileLoad or TIR.TileStore || call.Metadata.BlockMicroKernel is not null);
        Assert.All(
            expressions.OfType<Call>().Where(call => call.Target is TIR.NTT.NTTKernelOp),
            call => Assert.IsType<None>(call.Arguments[^1]));
        Assert.Empty(
            Directory.GetDirectories(Dumpper.Directory, "*AutoTilingPass*", SearchOption.AllDirectories));

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.DoesNotContain("tle.gpu.alloc", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.gpu.copy", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("pipeline_executions", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja ElementwiseUnary.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tl.load", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tl.store", generatedKernelsPy, StringComparison.Ordinal);

        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(1024, dtype=torch.float32, device='cuda') - 511",
            "output = module(x)",
            "torch.testing.assert_close(output, torch.abs(-x), rtol=0, atol=1e-6)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedRDataRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var bias = Tensor.From<float>(Enumerable.Range(0, 32).Select(i => i * 0.5f).ToArray(), [32, 1]);
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Math.Binary(BinaryOp.Add, x, bias), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_rdata_run_model", main);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var launchMeta = document.RootElement.GetProperty("functions").EnumerateArray().Single()
            .GetProperty("generated_kernels").EnumerateArray().Single()
            .GetProperty("launch").GetProperty("meta");
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("thread", "rdata", "pool_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("warp", "rdata", "pool_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("thread", "rdata", "stride_bytes"), out _));
        Assert.False(launchMeta.TryGetProperty(RemovedLocalMeta("warp", "rdata", "stride_bytes"), out _));
        Assert.Equal(0, launchMeta.GetProperty("rdata_pool_bytes").GetInt64());
        Assert.True(launchMeta.GetProperty("chip_local_rdata_pool_bytes").GetInt64() > 0);
        Assert.Equal(0, launchMeta.GetProperty("block_local_rdata_pool_bytes").GetInt64());

        var graphDumps = string.Join(
            Environment.NewLine,
            Directory.GetFiles(Dumpper.Directory, "*.il", SearchOption.AllDirectories).Select(File.ReadAllText));
        var tirDumps = string.Join(
            Environment.NewLine,
            Directory.GetFiles(Dumpper.Directory, "*.script", SearchOption.AllDirectories).Select(File.ReadAllText));
        Assert.Contains("ShardedView", graphDumps, StringComparison.Ordinal);
        Assert.Contains("ChipLocalRdata", tirDumps, StringComparison.Ordinal);
        Assert.DoesNotContain("ShardedView", tirDumps, StringComparison.Ordinal);

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.DoesNotContain(RemovedLocalName("thread", "rdata"), generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain(RemovedLocalName("warp", "rdata"), generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("chip_local_rdata", generatedKernelsPy, StringComparison.Ordinal);

        var rdataPy = File.ReadAllText(Path.Join(outputDirectory, "rdata.py"));
        Assert.Contains("\"rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"chip_local_rdata_bytes\":", rdataPy, StringComparison.Ordinal);
        Assert.DoesNotContain("\"chip_local_rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"block_local_rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);

        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(32, dtype=torch.float32, device='cuda').reshape(32, 1)",
            "bias = torch.arange(32, dtype=torch.float32, device='cuda').reshape(32, 1) * 0.5",
            "output = module(x)",
            "torch.testing.assert_close(output, x + bias, rtol=0, atol=1e-6)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedCastRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Tensors.Cast(x, DataTypes.Float16), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_cast_run_model", main);
        AssertGeneratedKernel(outputDirectory, "cast", "ElementwiseCast.py.jinja");
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = (torch.arange(32, dtype=torch.float32, device='cuda').reshape(32, 1) - 8) * 0.25",
            "output = module(x)",
            "assert output.dtype == torch.float16",
            "torch.testing.assert_close(output, x.to(torch.float16), rtol=0, atol=0)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedWhereRun()
    {
        ConfigureAutoDistributedPyNTT();
        var cond = new Var("cond", new TensorType(DataTypes.Boolean, new[] { 32, 1 }));
        var trueValue = new Var("x", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var falseValue = new Var("y", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Tensors.Where(cond, trueValue, falseValue), new[] { cond, trueValue, falseValue });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_where_run_model", main);
        AssertGeneratedKernel(outputDirectory, "where", "ElementwiseWhere.py.jinja");
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(32, dtype=torch.float32, device='cuda').reshape(32, 1)",
            "y = -x",
            "cond = (x % 3) == 0",
            "output = module(cond, x, y)",
            "torch.testing.assert_close(output, torch.where(cond, x, y), rtol=0, atol=1e-6)");
    }

    [Fact]
    public void TestPyNTTDistributedBoxingSplitToBroadcastRun()
    {
        ConfigureAutoDistributedPyNTT();

        var inputType = new TensorType(DataTypes.BFloat16, new[] { 3, 128 });
        var input = new Var("x", inputType);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var splitByFeatureType = new DistributedType(inputType, new SBP[] { SBP.B, SBP.SContiguous([0, 1], 4) }, placement);
        var splitByTokenType = new DistributedType(inputType, new SBP[] { SBP.SContiguous([0], 1), SBP.B }, placement);
        var featureShard = CreateBuffer("feature_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [3, 128], [4, 1], splitByFeatureType);
        var tokenShard = CreateBuffer("token_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 1024, [3, 128], [128, 1], splitByTokenType);
        var output = CreateOutputVar("output", inputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(featureShard, input, splitByFeatureType.AxisPolicies, placement),
            TIR.F.NTT.GatherReduceScatter(featureShard, tokenShard, splitByFeatureType, splitByTokenType),
            TIR.F.NTT.TensorStore(tokenShard, output, splitByTokenType.AxisPolicies, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 2048,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_boxing_split_to_broadcast_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Reshard.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("stage=tile_scatter", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("stage=to_collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("stage=from_collective", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(3 * 128, dtype=torch.float32, device='cuda').reshape(3, 128) - 17) * 0.01).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTMultiLaneVectorTransposeRun()
    {
        ConfigureAutoDistributedPyNTT();

        var inputType = new TensorType(DataTypes.BFloat16, new[] { 2, 3, 32 });
        var outputType = new TensorType(DataTypes.BFloat16, new[] { 3, 2, 32 });
        var vector32Type = new VectorType(DataTypes.BFloat16, 4, 8);
        var vector32InputType = new TensorType(vector32Type, new[] { 2, 3, 1 });
        var vector32OutputType = new TensorType(vector32Type, new[] { 3, 2, 1 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputDistributedType = new DistributedType(inputType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var vector32InputDistributedType = new DistributedType(vector32InputType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var vector32OutputDistributedType = new DistributedType(vector32OutputType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var outputDistributedType = new DistributedType(outputType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var scalarInput = CreateBuffer("scalar_input", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [2, 3, 32], [96, 32, 1], inputDistributedType);
        var vector32Input = CreateBuffer("vector32_input", vector32Type, TIR.MemoryLocation.Data, 384, [2, 3, 1], [3, 1, 1], vector32InputDistributedType);
        var vector32Output = CreateBuffer("vector32_output", vector32Type, TIR.MemoryLocation.Data, 768, [3, 2, 1], [2, 1, 1], vector32OutputDistributedType);
        var scalarOutput = CreateBuffer("scalar_output", DataTypes.BFloat16, TIR.MemoryLocation.Data, 1152, [3, 2, 32], [64, 32, 1], outputDistributedType);
        var input = new Var("x", inputType);
        var output = CreateOutputVar("output", outputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(scalarInput, input, inputDistributedType.AxisPolicies, placement),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Pack(scalarInput, vector32Input, new[] { 4, 8 }, new[] { 2, 2 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Transpose(vector32Input, vector32Output, new[] { 1, 0, 2 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Unpack(vector32Output, scalarOutput, new[] { 4, 8 }, new[] { 2, 2 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.TensorStore(scalarOutput, output, outputDistributedType.AxisPolicies, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 1536,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_multi_lane_vector_transpose_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Transpose.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("lane_flat = linear % 32", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("lane_raw0 = tl.arange(0, 4)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("lane_raw1 = tl.arange(0, 8)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(coord2) * 32 + (lane_coord0) * 8 + lane_coord1", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(2 * 3 * 32, dtype=torch.float32, device='cuda').reshape(2, 3, 32) - 37) * 0.015625).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x.permute(1, 0, 2).contiguous(), rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTPackUnpackWithDifferentShardCapacityRun()
    {
        ConfigureAutoDistributedPyNTT();

        var scalarType = new TensorType(DataTypes.Float32, new[] { 1, 128 });
        var vectorType = new VectorType(DataTypes.Float32, 2, 8);
        var packedType = new TensorType(vectorType, new[] { 1, 8 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var scalarDistributedType = new DistributedType(scalarType, new SBP[] { SBP.B, SBP.SContiguous([0, 1], 4) }, placement);
        var packedDistributedType = new DistributedType(packedType, new SBP[] { SBP.B, SBP.SContiguous([0, 1], 2) }, placement);
        var scalarShard = CreateBuffer("scalar_shard", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [1, 128], [4, 1], scalarDistributedType);
        var packedShard = CreateBuffer("packed_shard", vectorType, TIR.MemoryLocation.Data, 1024, [1, 8], [2, 1], packedDistributedType);
        var scalarOutputShard = CreateBuffer("scalar_output_shard", DataTypes.Float32, TIR.MemoryLocation.Data, 2048, [1, 128], [4, 1], scalarDistributedType);
        var input = new Var("x", scalarType);
        var output = CreateOutputVar("output", scalarType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(scalarShard, input, scalarDistributedType.AxisPolicies, placement),
            TIR.F.NTT.Pack(scalarShard, packedShard, new[] { 2, 8 }, new[] { 1, 1 }),
            TIR.F.NTT.Unpack(packedShard, scalarOutputShard, new[] { 2, 8 }, new[] { 1, 1 }),
            TIR.F.NTT.TensorStore(scalarOutputShard, output, scalarDistributedType.AxisPolicies, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 4096,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_pack_unpack_different_shard_capacity_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("op=pack", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("op=unpack", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(128, dtype=torch.float32, device='cuda').reshape(1, 128)",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTDistributedBoxingBroadcastAxisPartitionRun()
    {
        ConfigureAutoDistributedPyNTT();

        var inputType = new TensorType(DataTypes.BFloat16, new[] { 3, 128 });
        var input = new Var("x", inputType);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var splitByTokenType = new DistributedType(inputType, new SBP[] { SBP.SContiguous([0], 1), SBP.B }, placement);
        var broadcastType = new DistributedType(inputType, new SBP[] { SBP.B, SBP.B }, placement);
        var tokenShard = CreateBuffer("token_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [1, 128], [128, 1], splitByTokenType);
        var broadcastShard = CreateBuffer("broadcast_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 1024, [3, 128], [128, 1], broadcastType);
        var output = CreateOutputVar("output", inputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(tokenShard, input, splitByTokenType.AxisPolicies, placement),
            TIR.F.NTT.GatherReduceScatter(tokenShard, broadcastShard, splitByTokenType, broadcastType),
            TIR.F.NTT.TensorStore(broadcastShard, output, broadcastType.AxisPolicies, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 2048,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_boxing_broadcast_axis_partition_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Reshard.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("stage=tile_scatter", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("for destination_shard_coord1 in tl.range(0, 8)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("mask = mask & (shard_coord1 == destination_shard_coord1)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(3 * 128, dtype=torch.float32, device='cuda').reshape(3, 128) - 33) * 0.015625).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTDistributedBoxingVectorLaneBroadcastAxisPartitionRun()
    {
        ConfigureAutoDistributedPyNTT();

        var scalarInputType = new TensorType(DataTypes.BFloat16, new[] { 3, 16, 128 });
        var vectorElemType = new VectorType(DataTypes.BFloat16, 8);
        var vectorType = new TensorType(vectorElemType, new[] { 3, 16, 16 });
        var input = new Var("x", scalarInputType);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var scalarSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.B, SBP.SContiguous([1], 2), SBP.B }, placement);
        var vectorSplitType = new DistributedType(vectorType, new SBP[] { SBP.B, SBP.SContiguous([1], 2), SBP.B }, placement);
        var vectorBroadcastType = new DistributedType(vectorType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var scalarBroadcastType = new DistributedType(scalarInputType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var scalarShard = CreateBuffer("scalar_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [3, 2, 128], [256, 128, 1], scalarSplitType);
        var vectorShard = CreateBuffer("vector_shard", vectorElemType, TIR.MemoryLocation.Data, 2048, [3, 2, 16], [32, 16, 1], vectorSplitType);
        var broadcastVectorShard = CreateBuffer("broadcast_vector_shard", vectorElemType, TIR.MemoryLocation.Data, 4096, [3, 16, 16], [256, 16, 1], vectorBroadcastType);
        var broadcastScalarShard = CreateBuffer("broadcast_scalar_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 18432, [3, 16, 128], [2048, 128, 1], scalarBroadcastType);
        var output = CreateOutputVar("output", scalarInputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(scalarShard, input, scalarSplitType.AxisPolicies, placement),
            TIR.F.NTT.Pack(scalarShard, vectorShard, new[] { 8 }, new[] { 2 }),
            TIR.F.NTT.GatherReduceScatter(vectorShard, broadcastVectorShard, vectorSplitType, vectorBroadcastType),
            TIR.F.NTT.Unpack(broadcastVectorShard, broadcastScalarShard, new[] { 8 }, new[] { 2 }),
            TIR.F.NTT.TensorStore(broadcastScalarShard, output, scalarBroadcastType.AxisPolicies, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 32768,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_boxing_vector_lane_broadcast_axis_partition_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Reshard.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("lane=8, stage=tile_scatter", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("for destination_shard_coord0 in tl.range(0, 4)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("mask = mask & (shard_coord0 == destination_shard_coord0)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(3 * 16 * 128, dtype=torch.float32, device='cuda').reshape(3, 16, 128) - 59) * 0.0078125).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTDistributedBoxingVectorLaneMeshSplitToTokenSplitRun()
    {
        ConfigureAutoDistributedPyNTT();

        var scalarInputType = new TensorType(DataTypes.BFloat16, new[] { 3, 1024 });
        var vectorElemType = new VectorType(DataTypes.BFloat16, 8);
        var vectorType = new TensorType(vectorElemType, new[] { 3, 128 });
        var input = new Var("x", scalarInputType);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var scalarFeatureSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.B, SBP.SContiguous([0, 1], 32) }, placement);
        var vectorFeatureSplitType = new DistributedType(vectorType, new SBP[] { SBP.B, SBP.SContiguous([0, 1], 4) }, placement);
        var vectorTokenSplitType = new DistributedType(vectorType, new SBP[] { SBP.SContiguous([0], 1), SBP.B }, placement);
        var scalarTokenSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.SContiguous([0], 1), SBP.B }, placement);
        var scalarFeatureShard = CreateBuffer("scalar_feature_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [3, 32], [32, 1], scalarFeatureSplitType);
        var vectorFeatureShard = CreateBuffer("vector_feature_shard", vectorElemType, TIR.MemoryLocation.Data, 256, [3, 4], [4, 1], vectorFeatureSplitType);
        var vectorTokenShard = CreateBuffer("vector_token_shard", vectorElemType, TIR.MemoryLocation.Data, 512, [1, 128], [128, 1], vectorTokenSplitType);
        var scalarTokenShard = CreateBuffer("scalar_token_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 2560, [1, 1024], [1024, 1], scalarTokenSplitType);
        var output = CreateOutputVar("output", scalarInputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(scalarFeatureShard, input, scalarFeatureSplitType.AxisPolicies, placement),
            TIR.F.NTT.Pack(scalarFeatureShard, vectorFeatureShard, new[] { 8 }, new[] { 1 }),
            TIR.F.NTT.GatherReduceScatter(vectorFeatureShard, vectorTokenShard, vectorFeatureSplitType, vectorTokenSplitType),
            TIR.F.NTT.Unpack(vectorTokenShard, scalarTokenShard, new[] { 8 }, new[] { 1 }),
            TIR.F.NTT.TensorStore(scalarTokenShard, output, scalarTokenSplitType.AxisPolicies, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 8192,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_boxing_vector_lane_mesh_split_to_token_split_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Reshard.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("lane=8, stage=tile_scatter", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("destination_shard_coord0 = tmp_output_split0 % 4", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(3 * 1024, dtype=torch.float32, device='cuda').reshape(3, 1024) - 71) * 0.00390625).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTDistributedBoxingQwenVectorLaneSplitToBroadcastRun()
    {
        ConfigureAutoDistributedPyNTT();

        var scalarInputType = new TensorType(DataTypes.BFloat16, new[] { 16, 16, 160 });
        var vectorElemType = new VectorType(DataTypes.BFloat16, 8);
        var vectorType = new TensorType(vectorElemType, new[] { 16, 16, 20 });
        var input = new Var("x", scalarInputType);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var scalarSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.SContiguous([1], 2), SBP.B, SBP.SContiguous([0], 40) }, placement);
        var vectorSplitType = new DistributedType(vectorType, new SBP[] { SBP.SContiguous([1], 2), SBP.B, SBP.SContiguous([0], 5) }, placement);
        var vectorBroadcastType = new DistributedType(vectorType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var scalarBroadcastType = new DistributedType(scalarInputType, new SBP[] { SBP.B, SBP.B, SBP.B }, placement);
        var scalarShard = CreateBuffer("scalar_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [2, 16, 40], [640, 40, 1], scalarSplitType);
        var vectorShard = CreateBuffer("vector_shard", vectorElemType, TIR.MemoryLocation.Data, 4096, [2, 16, 5], [80, 5, 1], vectorSplitType);
        var broadcastVectorShard = CreateBuffer("broadcast_vector_shard", vectorElemType, TIR.MemoryLocation.Data, 8192, [16, 16, 20], [320, 20, 1], vectorBroadcastType);
        var broadcastScalarShard = CreateBuffer("broadcast_scalar_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 98304, [16, 16, 160], [2560, 160, 1], scalarBroadcastType);
        var output = CreateOutputVar("output", scalarInputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(scalarShard, input, scalarSplitType.AxisPolicies, placement),
            TIR.F.NTT.Pack(scalarShard, vectorShard, new[] { 8 }, new[] { 2 }),
            TIR.F.NTT.GatherReduceScatter(vectorShard, broadcastVectorShard, vectorSplitType, vectorBroadcastType),
            TIR.F.NTT.Unpack(broadcastVectorShard, broadcastScalarShard, new[] { 8 }, new[] { 2 }),
            TIR.F.NTT.TensorStore(broadcastScalarShard, output, scalarBroadcastType.AxisPolicies, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 196608,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_boxing_qwen_vector_lane_split_to_broadcast_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Reshard.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("global_shape=(16, 16, 20)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("input_tile_shape=(2, 16, 5)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("output_local_shape=(16, 16, 20)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(16 * 16 * 160, dtype=torch.float32, device='cuda').reshape(16, 16, 160) - 127) * 0.001953125).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTDistributedBoxingQwenSplitFeatureToBroadcastFeatureRun()
    {
        ConfigureAutoDistributedPyNTT();

        var inputType = new TensorType(DataTypes.BFloat16, new[] { 20, 3072 });
        var input = new Var("x", inputType);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var splitFeatureType = new DistributedType(inputType, new SBP[] { SBP.SContiguous([0], 5), SBP.SContiguous([1], 384) }, placement);
        var broadcastFeatureType = new DistributedType(inputType, new SBP[] { SBP.SContiguous([0], 5), SBP.B }, placement);
        var featureShard = CreateBuffer("feature_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [5, 384], [384, 1], splitFeatureType);
        var broadcastShard = CreateBuffer("broadcast_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 4096, [5, 3072], [3072, 1], broadcastFeatureType);
        var output = CreateOutputVar("output", inputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(featureShard, input, splitFeatureType.AxisPolicies, placement),
            TIR.F.NTT.GatherReduceScatter(featureShard, broadcastShard, splitFeatureType, broadcastFeatureType),
            TIR.F.NTT.TensorStore(broadcastShard, output, broadcastFeatureType.AxisPolicies, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 65536,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_boxing_qwen_split_feature_to_broadcast_feature_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Reshard.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("global_shape=(20, 3072)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("input_tile_shape=(5, 384)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("output_local_shape=(5, 3072)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(20 * 3072, dtype=torch.float32, device='cuda').reshape(20, 3072) - 257) * 0.0009765625).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x, rtol=0, atol=0)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedMatmulRun()
    {
        ConfigureAutoDistributedPyNTT();
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 16, 16 }));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, new[] { 16, 16 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Tensors.MatMul(lhs, rhs), new[] { lhs, rhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_matmul_run_model", main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var tirExpressions = compiler.Module.Functions
            .SelectMany(ExprCollector.Collect)
            .ToArray();
        Assert.Contains(
            tirExpressions.OfType<TIR.Buffer>(),
            buffer => buffer.MemSpan.Buffer.Location == TIR.MemoryLocation.Shared);
        Assert.DoesNotContain(
            tirExpressions.OfType<TIR.Buffer>(),
            buffer => buffer.MemSpan.Buffer.Location == TIR.MemoryLocation.Register ||
                buffer.StorageEncoding is not null);
        var selectedMatmul = Assert.Single(
            tirExpressions.OfType<Call>().Where(
                call => call.Metadata.TIRMicroKernel is
                {
                    Family: "triton.matmul",
                    SharedWorkspaces.Length: 2,
                }));
        Assert.Equal(2, Assert.IsType<IR.Tuple>(selectedMatmul.Arguments[^1]).Count);
        Assert.Contains(
            compiler.Module.Functions.OfType<TIR.PrimFunction>(),
            function => function.SchedResult.SharedDataPoolSize == 32768);
        Assert.Empty(tirExpressions.OfType<TIR.For>());
        Assert.Empty(tirExpressions.OfType<TIR.PipelineFor>());
        AssertGeneratedKernel(outputDirectory, "composite", "matmul/mma.py.jinja");
        var generatedKernels = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("tl.dot", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain("pipeline_executions", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("pyntt_shared_arena = tle.gpu.alloc([32768]", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("alias=pyntt_shared_arena", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("alias_offset_bytes=", generatedKernels, StringComparison.Ordinal);
        Assert.DoesNotContain(".to(tl.pointer_type(tl.uint8, 3))", generatedKernels, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "lhs = torch.arange(256, dtype=torch.float32, device='cuda').reshape(16, 16) * 0.01",
            "rhs = torch.arange(256, dtype=torch.float32, device='cuda').reshape(16, 16) * 0.02",
            "output = module(lhs, rhs)",
            "torch.testing.assert_close(output, lhs @ rhs, rtol=1e-5, atol=1e-5)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedMatmulReductionTailRun()
    {
        ConfigureAutoDistributedPyNTT();
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 3, 17 }));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, new[] { 17, 5 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Tensors.MatMul(lhs, rhs), new[] { lhs, rhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_matmul_tail_run_model", main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var scheduledLoops = compiler.Module.Functions
            .SelectMany(function => ExprCollector.Collect(function).OfType<TIR.For>())
            .ToArray();
        Assert.Empty(scheduledLoops);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernels = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("tl.range", generatedKernels, StringComparison.Ordinal);
        Assert.Contains("mask=", generatedKernels, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "lhs = (torch.arange(3 * 17, dtype=torch.float32, device='cuda').reshape(3, 17) - 11) * 0.01",
            "rhs = (torch.arange(17 * 5, dtype=torch.float32, device='cuda').reshape(17, 5) - 23) * 0.02",
            "output = module(lhs, rhs)",
            "torch.testing.assert_close(output, lhs @ rhs, rtol=1e-5, atol=1e-5)");
    }

    [Theory]
    [InlineData("rtx5060", 8, 1024, 8, 512, 2)]
    [InlineData("rtx5060", 16, 1024, 16, 512, 2)]
    [InlineData("rtx5060", 24, 1024, 32, 256, 3)]
    [InlineData("rtx5060", 32, 1024, 32, 256, 3)]
    [InlineData("rtx5060", 40, 1024, 64, 128, 4)]
    [InlineData("rtx5060", 64, 1024, 64, 128, 4)]
    [InlineData("rtx5060", 64, 2048, 64, 128, 4)]
    [InlineData("rtx5060", 64, 6144, 64, 128, 4)]
    [InlineData("rtx5060", 4752, 2048, 64, 128, 4)]
    [InlineData("h800", 8, 2048, 8, 1024, 2)]
    [InlineData("h800", 16, 2048, 16, 1024, 2)]
    [InlineData("h800", 8, 3072, 8, 1024, 2)]
    [InlineData("h800", 32, 1024, 32, 512, 2)]
    [InlineData("h800", 64, 1024, 64, 512, 2)]
    [InlineData("h800", 1192, 2048, 64, 512, 2)]
    public void TestTritonPackedBFloat16GemvPipelineSelectsLocalNTile(
        string targetMachine,
        int localScalarN,
        int k,
        int expectedBlockN,
        int expectedBlockK,
        int expectedNumStages)
    {
        const int nLane = 8;
        const int kAtom = 16;
        var packedN = localScalarN / nLane;
        var lhs = CreateBuffer(
            "lhs",
            DataTypes.BFloat16,
            TIR.MemoryLocation.Data,
            0,
            [1, k],
            [k, 1]);
        var rhs = CreateBuffer(
            "rhs",
            new VectorType(DataTypes.BFloat16, [nLane, 2, 8]),
            TIR.MemoryLocation.ChipLocalRdata,
            0,
            [k / kAtom, packedN],
            [packedN, 1]);
        var output = CreateBuffer(
            "output",
            new VectorType(DataTypes.BFloat16, [nLane]),
            TIR.MemoryLocation.ChipLocalData,
            0,
            [1, packedN],
            [packedN, 1]);
        var op = new TIR.NTT.PackedMatMul(
            false,
            IR.NTT.PackedMatMulRhsLayout.KMajor);
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);

        var selection = Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(
            targetOptions.TIRMicroKernelSelector.Select(
                new(op, [lhs, rhs, output], NTTTargetMachineCatalog.Resolve(targetMachine))));

        Assert.Equal("triton.matmul", selection.Family);
        Assert.Equal("simt_fma_smem_pipeline", selection.Variant);
        var actualConfiguration = (
            (int)selection.Parameters["block_n"],
            (int)selection.Parameters["block_k"],
            (int)selection.Parameters["num_stages"]);
        Assert.Equal((expectedBlockN, expectedBlockK, expectedNumStages), actualConfiguration);
        var workspace = Assert.Single(selection.SharedWorkspaces);
        Assert.Equal("rhs_stage", workspace.Name);
        Assert.Equal(
            new long[]
            {
                expectedNumStages,
                expectedBlockK / kAtom * (expectedBlockN / nLane),
                nLane * kAtom,
            },
            workspace.Type.Shape.ToValueArray());
        Assert.Equal(new[] { 1 }, selection.TransferPipeline!.SourceArgumentIndices);
        Assert.Equal(new[] { 0 }, selection.TransferPipeline.SharedWorkspaceIndices);
    }

    [Fact]
    public void TestTritonPackedGemvSelectorAcceptsDistributedOutputBufferVar()
    {
        const int k = 2048;
        const int globalScalarN = 1024;
        const int nLane = 8;
        const int kPack = 2;
        const int kLane = 8;
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var outputType = new DistributedType(
            new TensorType(
                new VectorType(DataTypes.BFloat16, [nLane]),
                new long[] { 1, globalScalarN / nLane }),
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 4) },
            placement);
        var rhsType = new DistributedType(
            new TensorType(
                new VectorType(DataTypes.BFloat16, [nLane, kPack, kLane]),
                new long[] { k / (kPack * kLane), globalScalarN / nLane }),
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 4) },
            placement);
        var lhs = CreateBuffer(
            "lhs",
            DataTypes.BFloat16,
            TIR.MemoryLocation.Data,
            0,
            [1, k],
            [k, 1]);
        var rhs = CreateBuffer(
            "rhs",
            rhsType.TensorType.DType,
            TIR.MemoryLocation.ChipLocalRdata,
            0,
            rhsType.TensorType.Shape.ToValueArray(),
            [globalScalarN / nLane, 1],
            rhsType);
        var output = CreateOutputVar("output", outputType);
        var op = new TIR.NTT.PackedMatMul(
            false,
            IR.NTT.PackedMatMulRhsLayout.KMajor);
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);

        var selection = Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(
            targetOptions.TIRMicroKernelSelector.Select(
                new(op, [lhs, rhs, output], targetOptions.TargetMachineModel)));

        Assert.Equal("simt_fma_smem_pipeline", selection.Variant);
        var actualConfiguration = (
            selection.Parameters["block_n"],
            selection.Parameters["block_k"],
            selection.Parameters["num_stages"]);
        Assert.Equal((32L, 512L, 2L), actualConfiguration);
    }

    [Fact]
    public void TestTritonPagedAttentionSelectsCrossPageTmaForVllmLayout()
    {
        var selection = SelectPagedAttentionPartialMicroKernel(DataTypes.BFloat16);

        Assert.Equal("triton.paged_attention_partial", selection.Family);
        Assert.Equal("simt_tma_smem_pipeline", selection.Variant);
        Assert.Equal(128, selection.Parameters["block_n"]);
        Assert.Equal(32, selection.Parameters["page_size"]);
        Assert.Equal(1, selection.Parameters["num_stages"]);
        Assert.Equal(2, selection.SharedWorkspaces.Length);
        Assert.All(
            selection.SharedWorkspaces,
            workspace => Assert.Equal(
                new long[] { 1, 1, 1, 128, 1, 128 },
                workspace.Type.Shape.ToValueArray()));
        var channels = selection.TransferPipeline!.Channels;
        Assert.Collection(
            channels,
            channel =>
            {
                Assert.Equal("key", channel.Name);
                Assert.Equal(new[] { 1 }, channel.SourceArgumentIndices);
                Assert.Equal(new[] { 0 }, channel.SharedWorkspaceIndices);
            },
            channel =>
            {
                Assert.Equal("value", channel.Name);
                Assert.Equal(new[] { 1 }, channel.SourceArgumentIndices);
                Assert.Equal(new[] { 1 }, channel.SharedWorkspaceIndices);
            });
    }

    [Fact]
    public void TestTritonPagedAttentionSelectsMmaFromH800ComputeCost()
    {
        var selection = SelectPagedAttentionPartialMicroKernel(
            DataTypes.BFloat16,
            NTTTargetMachineCatalog.Resolve("h800"),
            pageSize: 256);

        Assert.Equal("triton.paged_attention_partial", selection.Family);
        Assert.Equal("mma_tma_smem_pipeline", selection.Variant);
        Assert.Equal(64, selection.Parameters["block_n"]);
        Assert.Equal(256, selection.Parameters["page_size"]);
        Assert.Equal(2, selection.Parameters["num_stages"]);
        Assert.Equal(2, selection.SharedWorkspaces.Length);
        Assert.All(
            selection.SharedWorkspaces,
            workspace => Assert.Equal(
                new long[] { 2, 1, 1, 64, 1, 128 },
                workspace.Type.Shape.ToValueArray()));
        Assert.Equal(
            new[] { "key", "value" },
            selection.TransferPipeline!.Channels.Select(channel => channel.Name).ToArray());
    }

    [Fact]
    public void TestTritonPagedAttentionSelectsMmaForLargerGqaGroup()
    {
        var selection = SelectPagedAttentionPartialMicroKernel(
            DataTypes.BFloat16,
            queryHeads: 32);

        Assert.Equal("mma_tma_smem_pipeline", selection.Variant);
        Assert.Equal(64, selection.Parameters["block_n"]);
        Assert.Equal(2, selection.Parameters["num_stages"]);
    }

    [Fact]
    public void TestTritonPagedAttentionSelectsSimtIndependentlyOfTmaTransfer()
    {
        var selection = SelectPagedAttentionPartialMicroKernel(DataTypes.Float16);

        Assert.Equal("triton.paged_attention_partial", selection.Family);
        Assert.Equal("simt_direct", selection.Variant);
        Assert.Equal(32, selection.Parameters["block_n"]);
        Assert.Empty(selection.SharedWorkspaces);
        Assert.Null(selection.TransferPipeline);
    }

    [Fact]
    public void TestTritonH800PackedBFloat16QkvPipelineStagesOneFusedRhsTile()
    {
        const int k = 1024;
        const int nLane = 8;
        const int kAtom = 16;
        var input = CreateBuffer(
            "input",
            DataTypes.BFloat16,
            TIR.MemoryLocation.Data,
            0,
            [1, k],
            [k, 1]);
        var weight = CreateBuffer(
            "qkv_weight",
            new VectorType(DataTypes.BFloat16, [nLane, 2, 8]),
            TIR.MemoryLocation.BlockLocalRdata,
            0,
            [k / kAtom, 4],
            [4, 1]);
        var qOutput = CreateBuffer(
            "q_output",
            new VectorType(DataTypes.BFloat16, [nLane]),
            TIR.MemoryLocation.ChipLocalData,
            0,
            [1, 2],
            [2, 1]);
        var kOutput = CreateBuffer(
            "k_output",
            new VectorType(DataTypes.BFloat16, [nLane]),
            TIR.MemoryLocation.ChipLocalData,
            0,
            [1, 1],
            [1, 1]);
        var vOutput = CreateBuffer(
            "v_output",
            new VectorType(DataTypes.BFloat16, [nLane]),
            TIR.MemoryLocation.ChipLocalData,
            0,
            [1, 1],
            [1, 1]);
        var op = new TIR.NTT.PackedQKVParallelLinearFusedRhs(
            16,
            8,
            IR.NTT.PackedMatMulRhsLayout.KMajor,
            [16, 8, 8]);
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);

        var selection = Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(
            targetOptions.TIRMicroKernelSelector.Select(
                new(
                    op,
                    [
                        input,
                        weight,
                        qOutput,
                        qOutput,
                        qOutput,
                        qOutput,
                        qOutput,
                        qOutput,
                        qOutput,
                        qOutput,
                        qOutput,
                        qOutput,
                        kOutput,
                        vOutput,
                    ],
                    NTTTargetMachineCatalog.Resolve("h800"))));

        Assert.Equal("triton.qkv_parallel_linear", selection.Family);
        Assert.Equal("simt_fma_smem_pipeline", selection.Variant);
        Assert.Equal(32, selection.Parameters["block_n"]);
        Assert.Equal(512, selection.Parameters["block_k"]);
        Assert.Equal(2, selection.Parameters["num_stages"]);
        var workspace = Assert.Single(selection.SharedWorkspaces);
        Assert.Equal(new long[] { 2, 128, 128 }, workspace.Type.Shape.ToValueArray());
        Assert.Equal(new[] { 1 }, selection.TransferPipeline!.SourceArgumentIndices);
    }

    [Theory]
    [InlineData(24, 1024, 32, 512)]
    [InlineData(48, 2048, 64, 256)]
    public void TestTritonH800PackedBFloat16GluPipelineSelectsPairedDoubleBuffer(
        int localScalarN,
        int k,
        int expectedBlockN,
        int expectedBlockK)
    {
        const int nLane = 8;
        const int kAtom = 16;
        var packedN = localScalarN / nLane;
        var input = CreateBuffer(
            "input",
            DataTypes.BFloat16,
            TIR.MemoryLocation.Data,
            0,
            [1, k],
            [k, 1]);
        var gateWeight = CreateBuffer(
            "gate_weight",
            new VectorType(DataTypes.BFloat16, [nLane, 2, 8]),
            TIR.MemoryLocation.ChipLocalRdata,
            0,
            [k / kAtom, packedN],
            [packedN, 1]);
        var upWeight = CreateBuffer(
            "up_weight",
            new VectorType(DataTypes.BFloat16, [nLane, 2, 8]),
            TIR.MemoryLocation.ChipLocalRdata,
            0,
            [k / kAtom, packedN],
            [packedN, 1]);
        var output = CreateBuffer(
            "output",
            new VectorType(DataTypes.BFloat16, [nLane]),
            TIR.MemoryLocation.ChipLocalData,
            0,
            [1, packedN],
            [packedN, 1]);
        var op = new TIR.NTT.PackedMatMulGlu(
            GluType.SwiGLU,
            IR.NTT.PackedMatMulRhsLayout.KMajor);
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);

        var selection = Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(
            targetOptions.TIRMicroKernelSelector.Select(
                new(
                    op,
                    [
                        input,
                        gateWeight,
                        upWeight,
                        output,
                        output,
                        output,
                        output,
                        output,
                        output,
                        output,
                    ],
                    NTTTargetMachineCatalog.Resolve("h800"))));

        Assert.Equal("triton.matmul_glu", selection.Family);
        Assert.Equal("simt_fma_smem_pipeline", selection.Variant);
        var actualConfiguration = (
            (int)selection.Parameters["block_n"],
            (int)selection.Parameters["block_k"],
            (int)selection.Parameters["num_stages"]);
        Assert.Equal((expectedBlockN, expectedBlockK, 4), actualConfiguration);
        var workspace = Assert.Single(selection.SharedWorkspaces);
        Assert.Equal(
            new long[]
            {
                4,
                expectedBlockK / kAtom * (expectedBlockN / nLane),
                nLane * kAtom,
            },
            workspace.Type.Shape.ToValueArray());
        Assert.Equal(new[] { 1, 2 }, selection.TransferPipeline!.SourceArgumentIndices);
    }

    [Theory]
    [InlineData(1, 1024, 32)]
    [InlineData(1, 2048, 64)]
    [InlineData(1, 8192, 256)]
    [InlineData(1, 151936, 4752)]
    [InlineData(16, 1024, 32)]
    public async Task TestPyNTTIRAutoDistributedPackedBFloat16MatmulRun(
        int inputRows,
        int outputFeatures,
        int expectedLocalScalarN)
    {
        ConfigureAutoDistributedPyNTT();
        const int inputFeatures = 1024;
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new[] { inputRows, inputFeatures }));
        var rhsValues = Enumerable.Range(0, inputFeatures * outputFeatures)
            .Select(i => (BFloat16)(((float)i - 128f) * 0.0001f))
            .ToArray();
        var rhs = Tensor.From<BFloat16>(rhsValues, [inputFeatures, outputFeatures]);
        var matmul = IR.F.Tensors.MatMul(lhs, rhs, DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, matmul, new[] { lhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline(
            $"generated_bf16_packed_matmul_m{inputRows}_n{expectedLocalScalarN}_run_model",
            main);
        var expectedVariant = inputRows == 1
            ? "simt_fma_smem_pipeline"
            : "mma";
        var expectedPipelineBlockN = 8;
        while (expectedPipelineBlockN < Math.Min(expectedLocalScalarN, 64))
        {
            expectedPipelineBlockN *= 2;
        }

        var expectedPipelineBlockK = expectedPipelineBlockN switch
        {
            <= 16 => 1024,
            32 => 512,
            _ => 256,
        };

        var expectedTemplate = $"triton/kernels/matmul/{expectedVariant}.py.jinja";
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var accumulateModels = document.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
            .SelectMany(kernel => kernel.GetProperty("helpers").EnumerateArray())
            .Where(helper => helper.GetProperty("template").GetString() == expectedTemplate)
            .Select(helper => helper.GetProperty("model"))
            .ToArray();
        var hostTensorDescriptors = document.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
            .SelectMany(kernel => kernel.GetProperty("metadata").GetProperty("launch").GetProperty("host_tensor_descriptors").EnumerateArray())
            .ToArray();
        Assert.NotEmpty(accumulateModels);
        Assert.All(accumulateModels, model =>
        {
            Assert.False(model.TryGetProperty("ReductionPhase", out _));
            Assert.False(model.TryGetProperty("MicroKernelFamily", out _));
            Assert.False(model.TryGetProperty("MicroKernelParameters", out _));
            Assert.Equal("k_major", model.GetProperty("RhsLayout").GetString());
            Assert.Equal(1, model.GetProperty("RhsNPackedLaneCount").GetInt32());
            Assert.Equal(8, model.GetProperty("RhsNVectorLaneCount").GetInt32());
            Assert.Equal(2, model.GetProperty("RhsKPackLaneCount").GetInt32());
            Assert.Equal(8, model.GetProperty("RhsKVectorLaneCount").GetInt32());
            Assert.Equal(1, model.GetProperty("OutputNPackedLaneCount").GetInt32());
            Assert.Equal(8, model.GetProperty("OutputNVectorLaneCount").GetInt32());
            var microKernel = model.GetProperty("MicroKernel");
            Assert.Equal("triton.matmul", microKernel.GetProperty("Family").GetString());
            Assert.Equal(expectedVariant, microKernel.GetProperty("Variant").GetString());
            if (expectedVariant == "simt_fma_smem_pipeline")
            {
                Assert.Equal(
                    expectedPipelineBlockN,
                    microKernel.GetProperty("Parameters").GetProperty("block_n").GetInt32());
                Assert.Equal(
                    expectedPipelineBlockK,
                    microKernel.GetProperty("Parameters").GetProperty("block_k").GetInt32());
                Assert.Equal(
                    2,
                    microKernel.GetProperty("Parameters").GetProperty("num_stages").GetInt32());
            }

            var sharedWorkspaceOffsets = microKernel
                .GetProperty("SharedWorkspaceOffsets")
                .EnumerateObject()
                .Select(property => property.Name)
                .ToArray();
            var expectedSharedWorkspaces = expectedVariant switch
            {
                "simt_fma_smem_pipeline" => new[] { "rhs_stage" },
                "mma" => new[] { "lhs_stage", "rhs_stage" },
                _ => Array.Empty<string>(),
            };
            Assert.Equal(
                expectedSharedWorkspaces,
                sharedWorkspaceOffsets);
            var outputShape = model.GetProperty("OutputShape").EnumerateArray().ToArray();
            var localScalarN = outputShape[^1].GetProperty("MaxValue").GetInt32()
                * model.GetProperty("OutputNPackedLaneCount").GetInt32()
                * model.GetProperty("OutputNVectorLaneCount").GetInt32();
            Assert.Equal(expectedLocalScalarN, localScalarN);
        });

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains(
            $"generated from PyNTT algorithm triton.matmul/{expectedVariant}",
            generatedKernelsPy,
            StringComparison.Ordinal);
        Assert.Contains("rhs_layout=k_major", generatedKernelsPy, StringComparison.Ordinal);
        if (expectedVariant == "simt_fma_smem_pipeline")
        {
            var descriptor = Assert.Single(hostTensorDescriptors);
            Assert.Equal("chip_local_rdata", descriptor.GetProperty("source").GetString());
            Assert.Equal("bfloat16", descriptor.GetProperty("scalar_dtype").GetString());
            Assert.Equal(new[] { 8, 2, 8 }, descriptor.GetProperty("vector_lane_shape").EnumerateArray().Select(value => value.GetInt32()).ToArray());
            Assert.Contains(").to(tl.int32)", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains("PYNTT_HOST_TENSOR_DESCRIPTOR_SPECS", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains(
                $"'block_shape': ({expectedPipelineBlockK / 16}, {expectedPipelineBlockN / 8}, 2, 64)",
                generatedKernelsPy,
                StringComparison.Ordinal);
            Assert.Contains("__rhs_descriptor,\n                slot.weight,", generatedKernelsPy, StringComparison.Ordinal);
            Assert.DoesNotContain("tl.make_tensor_descriptor", generatedKernelsPy, StringComparison.Ordinal);
            Assert.DoesNotContain("nv_mma_shared_layout=True", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains("tle.gpu.nv_tma_shared_layout(", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains("nv_mma_shared_layout=False", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains("alignment=1024", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains("tle.gpu.BlockEncoding(", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains("tle.gpu.SlicedEncoding(", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains("tle.encoding(", generatedKernelsPy, StringComparison.Ordinal);
            Assert.Contains(
                $"tl.cdiv(active_n, {expectedPipelineBlockN})",
                generatedKernelsPy,
                StringComparison.Ordinal);
            Assert.Contains(
                $"[{expectedPipelineBlockN / 8}, {32 / (expectedPipelineBlockN / 8)}]",
                generatedKernelsPy,
                StringComparison.Ordinal);
            var configuredProducerRegisters = Assert.Single(
                    Regex.Matches(
                            generatedKernelsPy,
                            @"'producer_registers': (?<registers>\d+)",
                            RegexOptions.CultureInvariant)
                        .Cast<Match>())
                .Groups["registers"]
                .Value;
            var warpSpecializeRegisterAllocations = Regex.Matches(
                    generatedKernelsPy,
                    @"tle\.gpu\.warp_specialize\([\s\S]*?\n\s*\[\d+\],\n\s*\[(?<registers>\d+)\],\n\s*\)",
                    RegexOptions.CultureInvariant)
                .Cast<Match>()
                .Select(match => match.Groups["registers"].Value)
                .ToArray();
            Assert.NotEmpty(warpSpecializeRegisterAllocations);
            Assert.All(
                warpSpecializeRegisterAllocations,
                registers => Assert.Equal(configuredProducerRegisters, registers));
            Assert.DoesNotContain("num_full_n_tiles", generatedKernelsPy, StringComparison.Ordinal);
            Assert.DoesNotContain("tail_pipeline_", generatedKernelsPy, StringComparison.Ordinal);
            Assert.DoesNotContain("mask=tl.max_constancy(", generatedKernelsPy, StringComparison.Ordinal);
            Assert.DoesNotContain("weight_shared_layout", generatedKernelsPy, StringComparison.Ordinal);
        }
        else
        {
            Assert.Empty(hostTensorDescriptors);
        }

        AssertGeneratedModelRuns(
            outputDirectory,
            $"lhs = ((torch.arange({inputRows} * {inputFeatures}, dtype=torch.float32, device='cuda').reshape({inputRows}, {inputFeatures}) - 16) * 0.001).to(torch.bfloat16)",
            $"rhs = ((torch.arange({inputFeatures} * {outputFeatures}, dtype=torch.float32, device='cuda').reshape({inputFeatures}, {outputFeatures}) - 128) * 0.0001).to(torch.bfloat16)",
            "output = module(lhs)",
            "torch.testing.assert_close(output, lhs @ rhs, rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTSharedOwnerHandoffToTransferPipelineRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 1 } };

        const int k = 128;
        const int n = 64;
        var gemmInput = new Var(
            "gemm_input",
            new TensorType(DataTypes.BFloat16, new[] { 16, k }));
        var gemvInput = new Var(
            "gemv_input",
            new TensorType(DataTypes.BFloat16, new[] { 1, k }));
        var gemmWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * n)
                .Select(index => (BFloat16)((index - 127) * 0.0005f))
                .ToArray(),
            [k, n]);
        var gemvWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * n)
                .Select(index => (BFloat16)((index - 63) * 0.00025f))
                .ToArray(),
            [k, n]);
        var body = new IR.Tuple(
            IR.F.Tensors.MatMul(gemmInput, gemmWeight, DataTypes.BFloat16),
            IR.F.Tensors.MatMul(gemvInput, gemvWeight, DataTypes.BFloat16));
        var main = new Function(
            "main",
            PyNTTTarget.Kind,
            body,
            new[] { gemmInput, gemvInput });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline(
            "generated_shared_owner_handoff_run_model",
            main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernels = File.ReadAllText(
            Path.Join(outputDirectory, "generated_kernels.py"));
        var handoffName = Assert.Single(
                Regex.Matches(
                        generatedKernels,
                        @"(?<name>[A-Za-z_]\w*__handoff_\d+)__pipe\s*=\s*tle\.pipe\(",
                        RegexOptions.CultureInvariant)
                    .Cast<Match>())
            .Groups["name"]
            .Value;
        Assert.Contains("one_shot=True", generatedKernels, StringComparison.Ordinal);
        Assert.Contains(
            $"{handoffName}__writer.commit(0)",
            generatedKernels,
            StringComparison.Ordinal);
        Assert.Contains(
            $"{handoffName}__reader.wait(0)",
            generatedKernels,
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            $"{handoffName}__writer.acquire",
            generatedKernels,
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            $"{handoffName}__reader.release",
            generatedKernels,
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            $"{handoffName}__storage",
            generatedKernels,
            StringComparison.Ordinal);

        AssertGeneratedModelRuns(
            outputDirectory,
            $"gemm_input = ((torch.arange(16 * {k}, dtype=torch.float32, device='cuda').reshape(16, {k}) - 31) * 0.001).to(torch.bfloat16); gemv_input = ((torch.arange({k}, dtype=torch.float32, device='cuda').reshape(1, {k}) - 17) * 0.002).to(torch.bfloat16)",
            "gemm_output, gemv_output = module(gemm_input, gemv_input)",
            $"gemm_weight = ((torch.arange({k} * {n}, dtype=torch.float32, device='cuda').reshape({k}, {n}) - 127) * 0.0005).to(torch.bfloat16); gemv_weight = ((torch.arange({k} * {n}, dtype=torch.float32, device='cuda').reshape({k}, {n}) - 63) * 0.00025).to(torch.bfloat16); torch.testing.assert_close(gemm_output, gemm_input @ gemm_weight, rtol=2e-2, atol=2e-2); torch.testing.assert_close(gemv_output, gemv_input @ gemv_weight, rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTPackedBFloat16LargeKGemvRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 1 } };
        const int k = 3072;
        const int n = 32;
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new[] { 1, k }));
        var rhsValues = Enumerable.Range(0, k * n)
            .Select(i => (BFloat16)(((float)i - (k * n / 2f)) * 0.00001f))
            .ToArray();
        var rhs = Tensor.From<BFloat16>(rhsValues, [k, n]);
        var main = new Function(
            "main",
            PyNTTTarget.Kind,
            IR.F.Tensors.MatMul(lhs, rhs, DataTypes.BFloat16),
            new[] { lhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline(
            "generated_bf16_large_k_gemv_run_model",
            main);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var accumulateModels = document.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
            .SelectMany(kernel => kernel.GetProperty("helpers").EnumerateArray())
            .Where(helper => helper.GetProperty("template").GetString() == "triton/kernels/matmul/simt_fma.py.jinja")
            .Select(helper => helper.GetProperty("model"))
            .ToArray();
        Assert.NotEmpty(accumulateModels);
        Assert.All(accumulateModels, model =>
            Assert.False(model.TryGetProperty("ReductionPhase", out _)));

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("tl.range", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("state_block_k", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            $"lhs = ((torch.arange({k}, dtype=torch.float32, device='cuda').reshape(1, {k}) - 256) * 0.001).to(torch.bfloat16)",
            $"rhs = ((torch.arange({k} * {n}, dtype=torch.float32, device='cuda').reshape({k}, {n}) - {k * n / 2}) * 0.00001).to(torch.bfloat16)",
            "output = module(lhs)",
            "torch.testing.assert_close(output.to(torch.float32), (lhs @ rhs).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTPagedAttentionQwenLikeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };
        targetOptions.Vectorize = true;

        var config = new PagedAttentionConfig(
            1,
            8,
            128,
            DataTypes.BFloat16,
            256,
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
        var (root, queryVar, kvVars, kvCacheObjVar) = Nncase.Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(
            [96],
            [96],
            16,
            32,
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            config,
            new());
        var main = new Function("main", PyNTTTarget.Kind, root, [queryVar, kvVars[0][0], kvVars[0][1], kvCacheObjVar]);

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_paged_attention_qwen_like_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja UpdatePagedAttentionKVCache.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT algorithm triton.paged_attention_partial/", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja PagedAttentionMerge.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("generated from PyNTT Jinja PagedAttention.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("axis_group(('block_y',))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("pyntt_shared_arena +", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain(".to(tl.pointer_type(tl.uint8, 3))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("key_topology_id", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("value_topology_id", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Matches(@"key_block_id_0 = tl\.load\([^\r\n]+\.to\(tl\.int64\)", generatedKernelsPy);
        Assert.Contains("key_descriptor = main_prim__paged_attention_partial__0__key_descriptor__resource0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("value_descriptor = main_prim__paged_attention_partial__0__value_descriptor__resource0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.gpu.copy(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("offsets=[flat_block_id_0, layer_id_value, page_offset_0, kv_head, 0]", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("eviction_policy=\"evict_first\"", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("value_block_id", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("volatile=True", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("key_cache_page", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("single_value_page", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("direct_context = (num_seqs == 1) & (max_seq_len <= 12)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("single_token_context = (num_seqs == 1) & (max_seq_len == 1)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("__regular_producer(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("__regular_consumer(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("run_regular = (~single_token_context) & ((direct_context & direct_owner) | ((~direct_context) & (shard_coord0 < 4)))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("single_token_consumer = single_token_context & direct_owner", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("if run_regular | single_token_consumer:", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("key_writer.close(0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("value_writer.close(0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("split_active = direct_context | (shard_coord0 < 4)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("if ((input4 != 1) | (tl.load(input8) > 12)):", generatedKernelsPy, StringComparison.Ordinal);
        using (var kernelParams = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json"))))
        {
            var helpers = kernelParams.RootElement.GetProperty("functions")
                .EnumerateArray()
                .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
                .SelectMany(kernel => kernel.GetProperty("helpers").EnumerateArray())
                .ToArray();
            var partialModel = Assert.Single(
                helpers,
                helper => helper.GetProperty("template").GetString() == "triton/kernels/paged_attention/mma_tma_smem_pipeline.py.jinja")
                .GetProperty("model");
            var mergeModel = Assert.Single(
                helpers,
                helper => helper.GetProperty("template").GetString() == "triton/kernels/PagedAttentionMerge.py.jinja")
                .GetProperty("model");
            Assert.Equal(0, partialModel.GetProperty("SplitHierarchyAxis").GetInt32());
            Assert.Equal(4, partialModel.GetProperty("SplitCount").GetInt32());
            Assert.Equal(12, partialModel.GetProperty("DirectContextThreshold").GetInt64());
            Assert.NotEqual(JsonValueKind.Null, partialModel.GetProperty("KVCacheArgument").ValueKind);
            Assert.Equal(4, partialModel.GetProperty("Hierarchy")[0].GetInt32());
            Assert.Equal(
                64,
                partialModel
                    .GetProperty("MicroKernel")
                    .GetProperty("Parameters")
                    .GetProperty("block_n")
                    .GetInt32());
            Assert.Equal(
                config.BlockSize,
                partialModel
                    .GetProperty("MicroKernel")
                    .GetProperty("Parameters")
                    .GetProperty("page_size")
                    .GetInt32());
            Assert.Equal(3, partialModel.GetProperty("MaxStateShape").GetArrayLength());
            Assert.Equal(3, mergeModel.GetProperty("MaxStateShape").GetArrayLength());
            Assert.NotEqual(
                "0",
                mergeModel
                    .GetProperty("MaxStateAddress")
                    .GetProperty("PoolStrideBytes")
                    .GetString());

            var kernelInputs = kernelParams.RootElement.GetProperty("functions")
                .EnumerateArray()
                .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
                .SelectMany(kernel => kernel.GetProperty("metadata").GetProperty("inputs").EnumerateArray())
                .Select(input => input.GetString())
                .ToArray();
            Assert.DoesNotContain(kvCacheObjVar.Name, kernelInputs);
            Assert.Contains($"{kvCacheObjVar.Name}.__query_start_loc", kernelInputs);
            Assert.Contains($"{kvCacheObjVar.Name}.__seq_lens", kernelInputs);
            Assert.Contains($"{kvCacheObjVar.Name}.__num_seqs", kernelInputs);
        }

        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            "seq_len = 96",
            "num_q_heads = 16",
            "num_kv_heads = 8",
            "head_dim = 128",
            "query = (torch.randn(seq_len, num_q_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "key = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "value = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "class MockKVCache:",
            "    pass",
            "cache = MockKVCache()",
            "cache.query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')",
            "cache.seq_lens = torch.tensor([seq_len], dtype=torch.int32, device='cuda')",
            "cache.block_table = torch.tensor([[0]], dtype=torch.int32, device='cuda')",
            "cache.slot_mapping = torch.arange(seq_len, dtype=torch.int64, device='cuda')",
            "cache.num_blocks = 32",
            "cache.kv_caches = torch.zeros((32, 2 * 256 * num_kv_heads * head_dim), dtype=torch.bfloat16, device='cuda')",
            "output = module(query, key, value, cache)",
            "ref = torch.empty((seq_len, num_q_heads, head_dim), dtype=torch.float32, device='cuda')",
            "for token in range(seq_len):",
            "    for q_head in range(num_q_heads):",
            "        kv_head = q_head // (num_q_heads // num_kv_heads)",
            "        scores = torch.matmul(key[:token + 1, kv_head, :].to(torch.float32), query[token, q_head, :].to(torch.float32))",
            "        probs = torch.softmax(scores, dim=0)",
            "        ref[token, q_head, :] = torch.matmul(probs, value[:token + 1, kv_head, :].to(torch.float32))",
            "torch.testing.assert_close(output.to(torch.float32), ref.to(torch.bfloat16).to(torch.float32), rtol=3e-2, atol=3e-2)",
            "short_query = torch.zeros_like(query)",
            "short_key = torch.zeros_like(key)",
            "short_value = torch.zeros_like(value)",
            "short_query[0] = query[0]",
            "short_key[0] = key[0]",
            "short_value[0] = value[0]",
            "short_cache = MockKVCache()",
            "short_cache.query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device='cuda')",
            "short_cache.seq_lens = torch.tensor([1], dtype=torch.int32, device='cuda')",
            "short_cache.block_table = torch.tensor([[0]], dtype=torch.int32, device='cuda')",
            "short_cache.slot_mapping = torch.arange(seq_len, dtype=torch.int64, device='cuda')",
            "short_cache.num_blocks = 32",
            "short_cache.kv_caches = torch.zeros_like(cache.kv_caches)",
            "short_output = module(short_query, short_key, short_value, short_cache)",
            "short_ref = short_value[0].repeat_interleave(num_q_heads // num_kv_heads, dim=0)",
            "torch.testing.assert_close(short_output[0].to(torch.float32), short_ref.to(torch.float32), rtol=3e-2, atol=3e-2)");
    }

    [Fact]
    public async Task TestPyNTTPagedAttentionQwenLikeTwoLayersRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };
        targetOptions.Vectorize = true;

        var config = new PagedAttentionConfig(
            2,
            8,
            128,
            DataTypes.BFloat16,
            256,
            [
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.HeadDim,
                PagedKVCacheDimKind.BlockSize,
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
            [SBP.SContiguous([0, 1])]);
        var (root, queryVar, kvVars, kvCacheObjVar) = Nncase.Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(
            [20],
            [20],
            16,
            32,
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            config,
            new());
        var parameters = new List<IVar> { queryVar };
        parameters.AddRange(kvVars.SelectMany(x => x));
        parameters.Add(kvCacheObjVar);
        var main = new Function("main", PyNTTTarget.Kind, root, parameters.ToArray());

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_paged_attention_qwen_like_two_layers_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        using (var kernelParams = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json"))))
        {
            var helpers = kernelParams.RootElement.GetProperty("functions")
                .EnumerateArray()
                .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
                .SelectMany(kernel => kernel.GetProperty("helpers").EnumerateArray())
                .ToArray();
            var updateHelpers = helpers
                .Where(helper => helper.GetProperty("template").GetString() == "triton/kernels/UpdatePagedAttentionKVCache.py.jinja")
                .ToArray();
            var semanticUpdates = updateHelpers
                .Select(helper => helper.GetProperty("model"))
                .Select(model => (
                    LayerId: model.GetProperty("LayerIdExpression").GetString(),
                    CacheKind: model.GetProperty("CacheKind").GetInt32()))
                .Distinct()
                .ToArray();
            Assert.Equal(4, updateHelpers.Length);
            Assert.Equal(4, semanticUpdates.Length);
            Assert.Contains(("0", 0), semanticUpdates);
            Assert.Contains(("0", 1), semanticUpdates);
            Assert.Contains(("1", 0), semanticUpdates);
            Assert.Contains(("1", 1), semanticUpdates);

            var pagedAttentionPartialHelpers = helpers
                .Where(helper => helper.GetProperty("template").GetString() == "triton/kernels/PagedAttentionPartial.py.jinja")
                .ToArray();
            var pagedAttentionMergeHelpers = helpers
                .Where(helper => helper.GetProperty("template").GetString() == "triton/kernels/PagedAttentionMerge.py.jinja")
                .ToArray();
            Assert.Equal(2, pagedAttentionPartialHelpers.Length);
            Assert.Equal(2, pagedAttentionMergeHelpers.Length);
            Assert.Equal(
                new[] { "0", "1" },
                pagedAttentionPartialHelpers
                    .Select(helper => helper.GetProperty("model").GetProperty("LayerIdExpression").GetString())
                    .OrderBy(layerId => layerId, StringComparer.Ordinal)
                    .ToArray());
        }

        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            "seq_len = 20",
            "num_q_heads = 16",
            "num_kv_heads = 8",
            "head_dim = 128",
            "query = (torch.randn(seq_len, num_q_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "key0 = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "value0 = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "key1 = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "value1 = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "class MockKVCache:",
            "    pass",
            "cache = MockKVCache()",
            "cache.query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')",
            "cache.seq_lens = torch.tensor([seq_len], dtype=torch.int32, device='cuda')",
            "cache.block_table = torch.tensor([[[0, 0]]], dtype=torch.int32, device='cuda')",
            "cache.slot_mapping = torch.stack([torch.zeros(seq_len, dtype=torch.int64, device='cuda'), torch.arange(seq_len, dtype=torch.int64, device='cuda')], dim=1)",
            "cache.num_blocks = 32",
            "cache.kv_caches = torch.zeros((4, 8, 1, 2 * num_kv_heads * (head_dim // 8) * 256 * 8), dtype=torch.bfloat16, device='cuda')",
            "output = module(query, key0, value0, key1, value1, cache)",
            "ref = query",
            "for key, value in [(key0, value0), (key1, value1)]:",
            "    next_ref = torch.empty((seq_len, num_q_heads, head_dim), dtype=torch.float32, device='cuda')",
            "    for token in range(seq_len):",
            "        for q_head in range(num_q_heads):",
            "            kv_head = q_head // (num_q_heads // num_kv_heads)",
            "            scores = torch.matmul(key[:token + 1, kv_head, :].to(torch.float32), ref[token, q_head, :].to(torch.float32))",
            "            probs = torch.softmax(scores, dim=0)",
            "            next_ref[token, q_head, :] = torch.matmul(probs, value[:token + 1, kv_head, :].to(torch.float32))",
            "    ref = next_ref.to(torch.bfloat16)",
            "torch.testing.assert_close(output.to(torch.float32), ref.to(torch.float32), rtol=3e-2, atol=3e-2)");
    }

    [Fact]
    public async Task TestPyNTTPagedAttentionQwenLikeDecodeRun()
    {
        ConfigureAutoDistributedPyNTT();
        CompileOptions.ShapeBucketOptions.Enable = true;
        CompileOptions.ShapeBucketOptions.SegmentsCount = 2;
        CompileOptions.ShapeBucketOptions.SegmentRanges["num_tokens"] = [1, 32];
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };
        targetOptions.Vectorize = true;

        var config = new PagedAttentionConfig(
            1,
            8,
            128,
            DataTypes.BFloat16,
            256,
            [
                PagedKVCacheDimKind.NumBlocks,
                PagedKVCacheDimKind.NumLayers,
                PagedKVCacheDimKind.KV,
                PagedKVCacheDimKind.NumKVHeads,
                PagedKVCacheDimKind.HeadDim,
                PagedKVCacheDimKind.BlockSize,
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
            [SBP.SContiguous([0, 1])]);
        var (root, queryVar, kvVars, kvCacheObjVar) = Nncase.Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(
            [20],
            [20],
            16,
            32,
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
            config,
            new(DynamicShape: true, DynamicMaxTokens: 32));
        CompileOptions.ShapeBucketOptions.VarMap.Add(queryVar, queryVar.CheckedShape.ToArray());
        foreach (var kvVar in kvVars.SelectMany(vars => vars))
        {
            CompileOptions.ShapeBucketOptions.VarMap.Add(kvVar, kvVar.CheckedShape.ToArray());
        }

        var main = new Function("main", PyNTTTarget.Kind, root, [queryVar, kvVars[0][0], kvVars[0][1], kvCacheObjVar]);
        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_paged_attention_qwen_like_decode_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        var chipLocalAllocations = modelPy.Split('\n')
            .Where(line => line.Contains("self.allocate_workspace", StringComparison.Ordinal) &&
                line.Contains("chip_local_data", StringComparison.Ordinal))
            .ToArray();
        var chipLocalAllocation = Assert.Single(chipLocalAllocations);
        Assert.Contains("\"main_prim.chip_local_data\", 401408, \"uint8\"", chipLocalAllocation, StringComparison.Ordinal);
        Assert.DoesNotContain("self.allocate_workspace(inputs, \"main_segment_", modelPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            "prefill_len = 20",
            "num_q_heads = 16",
            "num_kv_heads = 8",
            "head_dim = 128",
            "prefill_query = (torch.randn(prefill_len, num_q_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "prefill_key = (torch.randn(prefill_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "prefill_value = (torch.randn(prefill_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "decode_query = (torch.randn(1, num_q_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "decode_key = (torch.randn(1, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "decode_value = (torch.randn(1, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "class MockKVCache:",
            "    pass",
            "storage = torch.zeros((4, 8, 1, 2 * num_kv_heads * (head_dim // 8) * 256 * 8), dtype=torch.bfloat16, device='cuda')",
            "prefill_cache = MockKVCache()",
            "prefill_cache.query_start_loc = torch.tensor([0, prefill_len], dtype=torch.int32, device='cuda')",
            "prefill_cache.seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device='cuda')",
            "prefill_cache.block_table = torch.tensor([[[0, 0]]], dtype=torch.int32, device='cuda')",
            "prefill_cache.slot_mapping = torch.stack([torch.zeros(prefill_len, dtype=torch.int64, device='cuda'), torch.arange(prefill_len, dtype=torch.int64, device='cuda')], dim=1)",
            "prefill_cache.num_blocks = 32",
            "prefill_cache.kv_caches = storage",
            "_ = module(prefill_query, prefill_key, prefill_value, prefill_cache)",
            "decode_cache = MockKVCache()",
            "decode_cache.query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device='cuda')",
            "decode_cache.seq_lens = torch.tensor([prefill_len + 1], dtype=torch.int32, device='cuda')",
            "decode_cache.block_table = torch.tensor([[[0, 0]]], dtype=torch.int32, device='cuda')",
            "decode_cache.slot_mapping = torch.tensor([[0, prefill_len]], dtype=torch.int64, device='cuda')",
            "decode_cache.num_blocks = 32",
            "decode_cache.kv_caches = storage",
            "output = module(decode_query, decode_key, decode_value, decode_cache)",
            "all_key = torch.cat([prefill_key, decode_key], dim=0)",
            "all_value = torch.cat([prefill_value, decode_value], dim=0)",
            "ref = torch.empty((1, num_q_heads, head_dim), dtype=torch.float32, device='cuda')",
            "for q_head in range(num_q_heads):",
            "    kv_head = q_head // (num_q_heads // num_kv_heads)",
            "    scores = torch.matmul(all_key[:, kv_head, :].to(torch.float32), decode_query[0, q_head, :].to(torch.float32))",
            "    probs = torch.softmax(scores, dim=0)",
            "    ref[0, q_head, :] = torch.matmul(probs, all_value[:, kv_head, :].to(torch.float32))",
            "torch.testing.assert_close(output.to(torch.float32), ref.to(torch.bfloat16).to(torch.float32), rtol=3e-2, atol=3e-2)");
    }

    [Fact]
    public async Task TestPyNTTPagedAttentionVllmLayoutCrossPageDecodeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Vectorize = true;

        var cacheLayout = new[]
        {
            PagedKVCacheDimKind.NumBlocks,
            PagedKVCacheDimKind.KV,
            PagedKVCacheDimKind.NumLayers,
            PagedKVCacheDimKind.BlockSize,
            PagedKVCacheDimKind.NumKVHeads,
            PagedKVCacheDimKind.HeadDim,
        };
        var config = new PagedAttentionConfig(
            1,
            1,
            128,
            DataTypes.BFloat16,
            32,
            cacheLayout,
            cacheLayout,
            [PagedKVCacheDimKind.HeadDim],
            [PagedKVCacheDimKind.HeadDim],
            [8],
            [8],
            [],
            []);
        var (root, queryVar, kvVars, kvCacheObjVar) =
            Nncase.Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(
                [1],
                [96],
                1,
                32,
                [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim],
                [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim],
                config,
                new());
        var main = new Function(
            "main",
            PyNTTTarget.Kind,
            root,
            [queryVar, kvVars[0][0], kvVars[0][1], kvCacheObjVar]);

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline(
            "generated_paged_attention_vllm_cross_page_decode_run_model",
            main);
        using (var kernelParams = JsonDocument.Parse(
            File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json"))))
        {
            var partialModel = Assert.Single(
                kernelParams.RootElement
                    .GetProperty("functions")
                    .EnumerateArray()
                    .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
                    .SelectMany(kernel => kernel.GetProperty("helpers").EnumerateArray())
                    .Where(helper =>
                        helper.GetProperty("template").GetString() ==
                        "triton/kernels/paged_attention/mma_tma_smem_pipeline.py.jinja"))
                .GetProperty("model");
            var microKernel = partialModel.GetProperty("MicroKernel");
            Assert.Equal("mma_tma_smem_pipeline", microKernel.GetProperty("Variant").GetString());
            Assert.Equal(64, microKernel.GetProperty("Parameters").GetProperty("block_n").GetInt32());
        }

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("copy_start_0 = context_start", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("copy_start_1 = context_start + 32", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("key_destination_1 = key_slot.key.subslice(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("[0, 0, 32, 0, 0],", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("for context_start in tl.range(split_begin, full_end, 64):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("for context_start in tl.range(full_end, active_end, 32):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shape=(64, 128)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shape=(32, 128)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.gpu.MmaEncoding(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.gpu.DotOperandEncoding(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("q_values = tle.encoding(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("key_values = tle.encoding(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("prob_values = tle.encoding(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("value_values = tle.encoding(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("score = tl.dot(q_values, key_values, acc=score_init)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("scaled_acc = tle.encoding(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("acc = tl.dot(prob_values, value_values, scaled_acc", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("block_table_active_", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("for _tail in tl.static_range(0, 1):", generatedKernelsPy, StringComparison.Ordinal);

        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            "context_len = 96",
            "page_size = 32",
            "num_blocks = 32",
            "num_q_heads = 1",
            "num_kv_heads = 1",
            "head_dim = 128",
            "query = (torch.randn(1, num_q_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "current_key = (torch.randn(1, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "current_value = (torch.randn(1, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "all_key = (torch.randn(context_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "all_value = (torch.randn(context_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "all_key[-1:] = current_key",
            "all_value[-1:] = current_value",
            "class MockKVCache:",
            "    pass",
            "cache = MockKVCache()",
            "cache.query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device='cuda')",
            "cache.seq_lens = torch.tensor([context_len], dtype=torch.int32, device='cuda')",
            "cache.block_table = torch.arange((context_len + page_size - 1) // page_size, dtype=torch.int32, device='cuda').reshape(1, -1)",
            "cache.slot_mapping = torch.tensor([context_len - 1], dtype=torch.int64, device='cuda')",
            "cache.num_blocks = num_blocks",
            "cache.kv_caches = torch.zeros((num_blocks, 2, 1, page_size, num_kv_heads, head_dim), dtype=torch.bfloat16, device='cuda')",
            "for token in range(context_len):",
            "    block = token // page_size",
            "    offset = token % page_size",
            "    cache.kv_caches[block, 0, 0, offset] = all_key[token]",
            "    cache.kv_caches[block, 1, 0, offset] = all_value[token]",
            "output = module(query, current_key, current_value, cache)",
            "ref = torch.empty((1, num_q_heads, head_dim), dtype=torch.float32, device='cuda')",
            "for q_head in range(num_q_heads):",
            "    kv_head = q_head // (num_q_heads // num_kv_heads)",
            "    scores = torch.matmul(all_key[:, kv_head, :].to(torch.float32), query[0, q_head, :].to(torch.float32))",
            "    probs = torch.softmax(scores, dim=0)",
            "    ref[0, q_head, :] = torch.matmul(probs, all_value[:, kv_head, :].to(torch.float32))",
            "torch.testing.assert_close(output.to(torch.float32), ref.to(torch.bfloat16).to(torch.float32), rtol=3e-2, atol=3e-2)");
    }

    [Fact]
    public async Task TestPyNTTPackedQKVParallelLinearQwenLikeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Vectorize = true;

        var seq = 20;
        var k = 256;
        var qn = 512;
        var kvn = 256;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, k }));
        var qWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * qn).Select(i => (BFloat16)(((i % 251) - 125f) * 0.0001f)).ToArray(),
            [k, qn]);
        var kWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * kvn).Select(i => (BFloat16)(((i % 241) - 120f) * 0.0001f)).ToArray(),
            [k, kvn]);
        var vWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * kvn).Select(i => (BFloat16)(((i % 239) - 119f) * 0.0001f)).ToArray(),
            [k, kvn]);
        var qkvInput = IR.F.Math.Unary(UnaryOp.Abs, input);
        var qkv = IR.F.NN.QKVParallelLinear(
            qkvInput,
            qWeight,
            kWeight,
            vWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            numHeads: 16,
            numKvHeads: 8,
            outputDataType: DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, qkv, new[] { input });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_qwen_like_qkv_run_model", main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var qkvFunction = Assert.Single(compiler.Module.Functions.OfType<TIR.PrimFunction>().Where(function =>
            ExprCollector.Collect(function.Body).OfType<Call>().Any(call => call.Target is TIR.NTT.PackedQKVParallelLinearFusedRhs)));
        var qkvCall = Assert.Single(ExprCollector.Collect(qkvFunction.Body).OfType<Call>().Where(call =>
            call.Target is TIR.NTT.PackedQKVParallelLinearFusedRhs));
        var qkvLoops = ExprCollector.Collect(qkvFunction.Body)
            .OfType<TIR.For>()
            .Where(loop => ExprCollector.Collect(loop.Body).Any(expr => ReferenceEquals(expr, qkvCall)))
            .ToArray();
        Assert.Empty(qkvLoops);
        Assert.Contains(
            compiler.Module.Functions.SelectMany(function => ExprCollector.Collect(function).OfType<TIR.PhysicalBuffer>()),
            buffer => buffer.Location == TIR.MemoryLocation.Shared);
        Assert.DoesNotContain(
            compiler.Module.Functions.SelectMany(function => ExprCollector.Collect(function).OfType<TIR.PhysicalBuffer>()),
            buffer => buffer.Location == TIR.MemoryLocation.Register);
        Assert.Equal(2, Assert.IsType<IR.Tuple>(qkvCall.Arguments[^1]).Count);
        Assert.Equal("triton.qkv_parallel_linear", qkvCall.Metadata.TIRMicroKernel?.Family);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT algorithm triton.qkv_parallel_linear/mma", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("alias=pyntt_shared_arena", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tl.gather(input0", generatedKernelsPy, StringComparison.Ordinal);
        var qkvDotCount = Regex.Matches(
            generatedKernelsPy,
            @"acc \+= tl\.dot\(input_values, weight_values\)",
            RegexOptions.CultureInvariant).Count;
        Assert.Equal(3, qkvDotCount);
        Assert.DoesNotContain("_acc += tl.dot", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            $"input = (torch.randn({seq}, {k}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"q_weight = (((torch.arange({k} * {qn}, device='cuda') % 251) - 125).reshape({k}, {qn}) * 0.0001).to(torch.bfloat16)",
            $"k_weight = (((torch.arange({k} * {kvn}, device='cuda') % 241) - 120).reshape({k}, {kvn}) * 0.0001).to(torch.bfloat16)",
            $"v_weight = (((torch.arange({k} * {kvn}, device='cuda') % 239) - 119).reshape({k}, {kvn}) * 0.0001).to(torch.bfloat16)",
            "q, k_out, v_out = module(input)",
            "qkv_input = torch.abs(input)",
            "torch.testing.assert_close(q.to(torch.float32), (qkv_input @ q_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)",
            "torch.testing.assert_close(k_out.to(torch.float32), (qkv_input @ k_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)",
            "torch.testing.assert_close(v_out.to(torch.float32), (qkv_input @ v_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTPackedQKVParallelLinearQwenLikeGemvPipelineRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Vectorize = true;

        const int seq = 1;
        const int k = 1024;
        const int qn = 2048;
        const int kvn = 1024;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, k }));
        var qWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * qn)
                .Select(i => (BFloat16)(((i % 251) - 125f) * 0.0001f))
                .ToArray(),
            [k, qn]);
        var kWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * kvn)
                .Select(i => (BFloat16)(((i % 241) - 120f) * 0.0001f))
                .ToArray(),
            [k, kvn]);
        var vWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * kvn)
                .Select(i => (BFloat16)(((i % 239) - 119f) * 0.0001f))
                .ToArray(),
            [k, kvn]);
        var qkvInput = IR.F.Math.Unary(UnaryOp.Abs, input);
        var qkv = IR.F.NN.QKVParallelLinear(
            qkvInput,
            qWeight,
            kWeight,
            vWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            numHeads: 16,
            numKvHeads: 8,
            outputDataType: DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, qkv, new[] { input });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline(
            "generated_qwen_like_qkv_gemv_pipeline_run_model",
            main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var qkvCall = Assert.Single(compiler.Module.Functions
            .SelectMany(function => ExprCollector.Collect(function).OfType<Call>())
            .Where(call => call.Target is TIR.NTT.PackedQKVParallelLinearFusedRhs));
        var microKernel = Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(qkvCall.Metadata.TIRMicroKernel);
        Assert.Equal("triton.qkv_parallel_linear", microKernel.Family);
        Assert.Equal("simt_fma_smem_pipeline", microKernel.Variant);
        Assert.Equal(64, microKernel.Parameters["block_n"]);
        Assert.Equal(128, microKernel.Parameters["block_k"]);
        Assert.Equal(4, microKernel.Parameters["num_stages"]);
        Assert.Equal(new[] { "rhs_stage" }, microKernel.SharedWorkspaces.Select(workspace => workspace.Name).ToArray());
        Assert.Equal(
            TIR.MemoryLocation.Shared,
            Assert.IsType<TIR.Buffer>(qkvCall.Arguments[^1]).MemSpan.Buffer.Location);

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var descriptorSpecs = document.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
            .SelectMany(kernel => kernel.GetProperty("metadata").GetProperty("launch").GetProperty("host_tensor_descriptors").EnumerateArray())
            .ToArray();
        var descriptor = Assert.Single(descriptorSpecs);
        Assert.Equal("block_local_rdata", descriptor.GetProperty("source").GetString());
        Assert.Equal(256 * 1024, descriptor.GetProperty("owner_stride_bytes").GetInt32());
        Assert.Equal("bfloat16", descriptor.GetProperty("scalar_dtype").GetString());
        Assert.Equal(
            new[] { k / 16, (qn + (2 * kvn)) / (32 * 8) },
            descriptor.GetProperty("logical_shape").EnumerateArray().Select(value => value.GetInt32()).ToArray());
        Assert.Equal(
            new[] { 8, 2, 8 },
            descriptor.GetProperty("vector_lane_shape").EnumerateArray().Select(value => value.GetInt32()).ToArray());

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains(
            "generated from PyNTT algorithm triton.qkv_parallel_linear/simt_fma_smem_pipeline",
            generatedKernelsPy,
            StringComparison.Ordinal);
        Assert.Contains("rhs_layout=k_major", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("nv_mma_shared_layout=True", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.gpu.nv_tma_shared_layout(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Single(Regex.Matches(generatedKernelsPy, @"tle\.gpu\.copy\(", RegexOptions.CultureInvariant).Cast<Match>());
        Assert.DoesNotContain("eviction_policy=", generatedKernelsPy, StringComparison.Ordinal);
        var producerNTileLoops = Regex.Matches(
            generatedKernelsPy,
            @"for n_tile in tl\.static_range\(\s*0,\s*2,\s*\):",
            RegexOptions.CultureInvariant);
        var nTileLoops = Regex.Matches(
            generatedKernelsPy,
            @"for n_tile in tl\.range\(\s*0,\s*2,\s*loop_unroll_factor=1,\s*\):",
            RegexOptions.CultureInvariant);
        var kTileLoops = Regex.Matches(
            generatedKernelsPy,
            $@"for k_tile in tl\.range\(0, {k / 128}\):",
            RegexOptions.CultureInvariant);
        Assert.Single(producerNTileLoops.Cast<Match>());
        Assert.Single(nTileLoops.Cast<Match>());
        Assert.Equal(2, kTileLoops.Count);
        Assert.DoesNotContain("slot.weight.subslice(", generatedKernelsPy, StringComparison.Ordinal);
        var descriptorMatches = Regex.Matches(
            generatedKernelsPy,
            @"'block_shape': \(8, 8, 2, 64\)",
            RegexOptions.CultureInvariant);
        Assert.Single(descriptorMatches.Cast<Match>());
        Assert.Single(Regex.Matches(generatedKernelsPy, @"'kind': 'table'", RegexOptions.CultureInvariant).Cast<Match>());
        Assert.Single(Regex.Matches(generatedKernelsPy, @"tle\.gpu\.tensor_map_table_entry\(", RegexOptions.CultureInvariant).Cast<Match>());
        Assert.Single(Regex.Matches(generatedKernelsPy, @"tle\.gpu\.reinterpret_tensor_map\(", RegexOptions.CultureInvariant).Cast<Match>());
        Assert.Contains("capacity=4", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("[1]", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tl.make_tensor_descriptor", generatedKernelsPy, StringComparison.Ordinal);

        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            $"input = (torch.randn({seq}, {k}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"q_weight = (((torch.arange({k} * {qn}, device='cuda') % 251) - 125).reshape({k}, {qn}) * 0.0001).to(torch.bfloat16)",
            $"k_weight = (((torch.arange({k} * {kvn}, device='cuda') % 241) - 120).reshape({k}, {kvn}) * 0.0001).to(torch.bfloat16)",
            $"v_weight = (((torch.arange({k} * {kvn}, device='cuda') % 239) - 119).reshape({k}, {kvn}) * 0.0001).to(torch.bfloat16)",
            "q, k_out, v_out = module(input)",
            "qkv_input = torch.abs(input)",
            "torch.testing.assert_close(q.to(torch.float32), (qkv_input @ q_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)",
            "torch.testing.assert_close(k_out.to(torch.float32), (qkv_input @ k_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)",
            "torch.testing.assert_close(v_out.to(torch.float32), (qkv_input @ v_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTRoPEQwenLikeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Vectorize = true;

        var seq = 20;
        var heads = 16;
        var headDim = 128;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, heads, headDim }));
        var cos = new Var("cos", new TensorType(DataTypes.Float32, new[] { seq, 1, headDim }));
        var sin = new Var("sin", new TensorType(DataTypes.Float32, new[] { seq, 1, headDim }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.NN.RoPE(input, cos, sin), new[] { input, cos, sin });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_qwen_like_rope_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja RoPE.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(3)",
            $"input = (torch.randn({seq}, {heads}, {headDim}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"angles = torch.randn({seq}, 1, {headDim}, dtype=torch.float32, device='cuda')",
            "cos = torch.cos(angles)",
            "sin = torch.sin(angles)",
            $"half = {headDim} // 2",
            "rotated = torch.cat((-input[..., half:], input[..., :half]), dim=-1)",
            "expect = (input.to(torch.float32) * cos + rotated.to(torch.float32) * sin).to(torch.bfloat16)",
            "output = module(input, cos, sin)",
            "torch.testing.assert_close(output.to(torch.float32), expect.to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTPackedSinCosRoPEQwenLikeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Hierarchies = [new[] { 1, 1 }];

        const int seq = 2;
        const int heads = 1;
        const int headDim = 128;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, heads, headDim }));
        var cos = Tensor.From<float>(
            Enumerable.Range(0, seq * heads * headDim).Select(i => 0.5f + (i % 7 * 0.05f)).ToArray(),
            [seq, heads, headDim]);
        var sin = Tensor.From<float>(
            Enumerable.Range(0, seq * heads * headDim).Select(i => -0.3f + (i % 5 * 0.1f)).ToArray(),
            [seq, heads, headDim]);
        var packedInput = IR.F.Tensors.Pack(input, [8], [2]);
        var packedCos = IR.F.Tensors.Pack(cos, [2, 8], [2, 2]);
        var packedSin = IR.F.Tensors.Pack(sin, [2, 8], [2, 2]);
        var output = IR.F.Tensors.Unpack(
            IR.F.NTT.VectorizedRoPE(packedInput, packedCos, packedSin),
            [8],
            [2]);
        var main = new Function("main", PyNTTTarget.Kind, output, new[] { input });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline(
            "generated_qwen_like_packed_sincos_rope_run_model",
            main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja RoPE.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("logical_rotary = ((major_raw) * 2 + (lane_raw0)) * 8 + (lane_raw1)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(13)",
            $"input = (torch.randn({seq}, {heads}, {headDim}, dtype=torch.float32, device='cuda') * 0.1).to(torch.bfloat16)",
            $"indices = torch.arange({seq * heads * headDim}, dtype=torch.int64, device='cuda').reshape({seq}, {heads}, {headDim})",
            "cos = 0.5 + (indices % 7).to(torch.float32) * 0.05",
            "sin = -0.3 + (indices % 5).to(torch.float32) * 0.1",
            $"half = {headDim} // 2",
            "rotated = torch.cat((-input[..., half:], input[..., :half]), dim=-1)",
            "expect = (input.to(torch.float32) * cos + rotated.to(torch.float32) * sin).to(torch.bfloat16)",
            "output = module(input)",
            "torch.testing.assert_close(output.to(torch.float32), expect.to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTRmsNormRoPEQwenLikeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Vectorize = true;

        var seq = 20;
        var heads = 16;
        var headDim = 128;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, heads, headDim }));
        var scale = new Var("scale", new TensorType(DataTypes.BFloat16, new[] { headDim }));
        var cos = new Var("cos", new TensorType(DataTypes.Float32, new[] { seq, 1, headDim }));
        var sin = new Var("sin", new TensorType(DataTypes.Float32, new[] { seq, 1, headDim }));
        var bias = Tensor.Zeros<BFloat16>([headDim]);
        var normalized = IR.F.NN.LayerNorm(2, 1e-6f, input, scale, bias, hasMean: false);
        var main = new Function("main", PyNTTTarget.Kind, IR.F.NN.RoPE(normalized, cos, sin), new[] { input, scale, cos, sin });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_qwen_like_rms_norm_rope_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja NormStats.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja NormApply.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja RoPE.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(4)",
            $"input = (torch.randn({seq}, {heads}, {headDim}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"scale = (1.0 + torch.randn({headDim}, dtype=torch.float32, device='cuda') * 0.01).to(torch.bfloat16)",
            $"angles = torch.randn({seq}, 1, {headDim}, dtype=torch.float32, device='cuda')",
            "cos = torch.cos(angles)",
            "sin = torch.sin(angles)",
            "normalized = input.to(torch.float32) * torch.rsqrt(torch.mean(input.to(torch.float32) ** 2, dim=2, keepdim=True) + 1e-6) * scale.to(torch.float32)",
            $"half = {headDim} // 2",
            "rotated = torch.cat((-normalized[..., half:], normalized[..., :half]), dim=-1)",
            "expect = (normalized * cos + rotated * sin).to(torch.bfloat16)",
            "output = module(input, scale, cos, sin)",
            "torch.testing.assert_close(output.to(torch.float32), expect.to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTPackedMatMulGluQwenLikeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Vectorize = true;

        var seq = 20;
        var k = 512;
        var n = 768;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, k }));
        var gateWeight = new Var("gate_weight", new TensorType(DataTypes.BFloat16, new[] { k, n }));
        var upWeight = new Var("up_weight", new TensorType(DataTypes.BFloat16, new[] { k, n }));
        var glu = IR.F.NN.MatMulGlu(
            input,
            gateWeight,
            upWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, glu, new[] { input, gateWeight, upWeight });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_qwen_like_glu_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT algorithm triton.matmul_glu/mma", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(1)",
            $"input = (torch.randn({seq}, {k}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"gate_weight = (torch.randn({k}, {n}, dtype=torch.float32, device='cuda') * 0.03).to(torch.bfloat16)",
            $"up_weight = (torch.randn({k}, {n}, dtype=torch.float32, device='cuda') * 0.03).to(torch.bfloat16)",
            "output = module(input, gate_weight, up_weight)",
            "gate = input @ gate_weight",
            "up = input @ up_weight",
            "expect = (gate.to(torch.float32) * torch.sigmoid(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16).to(torch.float32)",
            "torch.testing.assert_close(output.to(torch.float32), expect, rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTPackedMatMulGluQwenLikeGemvPipelineRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Vectorize = true;

        const int seq = 1;
        const int k = 1024;
        const int n = 3072;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, k }));
        var gateWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * n)
                .Select(i => (BFloat16)(((i % 251) - 125f) * 0.001f))
                .ToArray(),
            [k, n]);
        var upWeight = Tensor.From<BFloat16>(
            Enumerable.Range(0, k * n)
                .Select(i => (BFloat16)(((i % 241) - 120f) * 0.001f))
                .ToArray(),
            [k, n]);
        var glu = IR.F.NN.MatMulGlu(
            input,
            gateWeight,
            upWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, glu, new[] { input });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline(
            "generated_qwen_like_glu_gemv_pipeline_run_model",
            main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var gluCall = Assert.Single(compiler.Module.Functions
            .SelectMany(function => ExprCollector.Collect(function).OfType<Call>())
            .Where(call => call.Target is TIR.NTT.PackedMatMulGlu));
        var gluOp = Assert.IsType<TIR.NTT.PackedMatMulGlu>(gluCall.Target);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, gluOp.RhsLayout);
        var microKernel = Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(gluCall.Metadata.TIRMicroKernel);
        Assert.Equal("triton.matmul_glu", microKernel.Family);
        Assert.Equal("simt_fma_smem_pipeline", microKernel.Variant);
        Assert.Equal(64L, microKernel.Parameters["block_n"]);
        Assert.Equal(128L, microKernel.Parameters["block_k"]);
        Assert.Equal(4L, microKernel.Parameters["num_stages"]);
        Assert.Equal(new[] { "rhs_stage" }, microKernel.SharedWorkspaces.Select(workspace => workspace.Name).ToArray());
        Assert.Equal(
            TIR.MemoryLocation.Shared,
            Assert.IsType<TIR.Buffer>(gluCall.Arguments[^1]).MemSpan.Buffer.Location);

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var descriptorSpecs = document.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
            .SelectMany(kernel => kernel.GetProperty("metadata").GetProperty("launch").GetProperty("host_tensor_descriptors").EnumerateArray())
            .ToArray();
        Assert.Equal(2, descriptorSpecs.Length);
        Assert.All(descriptorSpecs, descriptor =>
        {
            Assert.Equal("chip_local_rdata", descriptor.GetProperty("source").GetString());
            Assert.Equal("bfloat16", descriptor.GetProperty("scalar_dtype").GetString());
            Assert.Equal(
                new[] { 8, 2, 8 },
                descriptor.GetProperty("vector_lane_shape").EnumerateArray().Select(value => value.GetInt32()).ToArray());
        });

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains(
            "generated from PyNTT algorithm triton.matmul_glu/simt_fma_smem_pipeline",
            generatedKernelsPy,
            StringComparison.Ordinal);
        var consumerStageDefinitions = Regex.Matches(
            generatedKernelsPy,
            @"def \w+__consumer_stage\(",
            RegexOptions.CultureInvariant);
        Assert.Empty(consumerStageDefinitions.Cast<Match>());
        Assert.Contains("rhs_layout=k_major", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("nv_mma_shared_layout=True", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.gpu.nv_tma_shared_layout(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Equal(2, Regex.Matches(generatedKernelsPy, @"tle\.gpu\.copy\(", RegexOptions.CultureInvariant).Count);
        Assert.DoesNotContain("slot.weight.subslice(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Equal(2, Regex.Matches(generatedKernelsPy, @"writer\.acquire\(", RegexOptions.CultureInvariant).Count);
        Assert.Equal(2, Regex.Matches(generatedKernelsPy, @"reader\.wait\(", RegexOptions.CultureInvariant).Count);
        var descriptorBlockShapeCount = Regex.Matches(
            generatedKernelsPy,
            @"'block_shape': \(8, 8, 2, 64\)",
            RegexOptions.CultureInvariant).Count;
        Assert.Equal(2, descriptorBlockShapeCount);
        Assert.Equal(2, Regex.Matches(generatedKernelsPy, @"'kind': 'table'", RegexOptions.CultureInvariant).Count);
        Assert.Equal(2, Regex.Matches(generatedKernelsPy, @"tle\.gpu\.tensor_map_table_entry\(", RegexOptions.CultureInvariant).Count);
        Assert.Equal(2, Regex.Matches(generatedKernelsPy, @"tle\.gpu\.reinterpret_tensor_map\(", RegexOptions.CultureInvariant).Count);
        Assert.Equal(2, Regex.Matches(generatedKernelsPy, @"eviction_policy=""evict_last""", RegexOptions.CultureInvariant).Count);
        Assert.Contains("capacity=4", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("[4, 8, 8, 2, 64]", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tl.make_tensor_descriptor", generatedKernelsPy, StringComparison.Ordinal);

        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(1)",
            $"input = (torch.randn({seq}, {k}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"gate_weight = (((torch.arange({k} * {n}, device='cuda') % 251) - 125).reshape({k}, {n}) * 0.001).to(torch.bfloat16)",
            $"up_weight = (((torch.arange({k} * {n}, device='cuda') % 241) - 120).reshape({k}, {n}) * 0.001).to(torch.bfloat16)",
            "output = module(input)",
            "gate = input @ gate_weight",
            "up = input @ up_weight",
            "expect = (gate.to(torch.float32) * torch.sigmoid(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)",
            $"tail_mask = torch.arange({n}, device='cuda') % 96 >= 64",
            "assert torch.max(torch.abs(expect[:, tail_mask].to(torch.float32))).item() > 1e-2",
            "torch.testing.assert_close(output.to(torch.float32), expect.to(torch.float32), rtol=2e-2, atol=5e-4)");
    }

    [Fact]
    public async Task TestPyNTTRmsNormQwenLikeRun()
    {
        ConfigureAutoDistributedPyNTT();
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.Vectorize = true;

        var seq = 20;
        var hidden = 1024;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { seq, hidden }));
        var scale = new Var("scale", new TensorType(DataTypes.BFloat16, new[] { hidden }));
        var bias = Tensor.Zeros<BFloat16>([hidden]);
        var main = new Function("main", PyNTTTarget.Kind, IR.F.NN.LayerNorm(1, 1e-6f, input, scale, bias, hasMean: false), new[] { input, scale });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_qwen_like_rms_norm_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja NormStats.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja NormApply.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(2)",
            $"input = (torch.randn({seq}, {hidden}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"scale = (1.0 + torch.randn({hidden}, dtype=torch.float32, device='cuda') * 0.01).to(torch.bfloat16)",
            "output = module(input, scale)",
            "expect = input.to(torch.float32) * torch.rsqrt(torch.mean(input.to(torch.float32) * input.to(torch.float32), dim=1, keepdim=True) + 1e-6) * scale.to(torch.float32)",
            "torch.testing.assert_close(output.to(torch.float32), expect.to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public void TestPyNTTGetPositionIdsUsesSplitHierarchyAxisRun()
    {
        ConfigureAutoDistributedPyNTT();

        var config = new PagedAttentionConfig(
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
                PagedKVCacheDimKind.HeadDim,
                PagedKVCacheDimKind.BlockSize,
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
            [SBP.SContiguous([0, 1])]);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var outputType = new TensorType(DataTypes.Float32, new[] { 20 });
        var outputDistributedType = new DistributedType(outputType, new SBP[] { SBP.SContiguous([0], 5) }, placement);
        var kvCacheObjVar = new Var("kvCache", TensorType.Scalar(new ReferenceType(new PagedAttentionKVCacheType { Config = config })));
        var output = CreateOutputVar("output", outputType);
        var outputBuffer = CreateBuffer("position_ids", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [5], [1], outputDistributedType);
        var body = new TIR.Sequential(
            TIR.F.NTT.GetPositionIds(kvCacheObjVar, outputBuffer, outputDistributedType),
            TIR.F.NTT.TensorStore(outputBuffer, output, outputDistributedType.AxisPolicies, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { kvCacheObjVar, output })
        {
            SchedResult =
            {
                DataUsage = 128,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_get_position_ids_split_hierarchy_axis_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        var globalStart = Assert.Single(generatedKernelsPy.Split('\n').Where(line => line.Contains("global_start =", StringComparison.Ordinal)));
        Assert.Contains("shard_coord0", globalStart, StringComparison.Ordinal);
        Assert.DoesNotContain("shard_index //", globalStart, StringComparison.Ordinal);
        Assert.DoesNotContain("tile", globalStart, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "class MockKVCache:",
            "    pass",
            "cache = MockKVCache()",
            "cache.device = torch.device('cuda')",
            "cache.query_start_loc = torch.tensor([0, 20], dtype=torch.int32, device='cuda')",
            "cache.seq_lens = torch.tensor([20], dtype=torch.int32, device='cuda')",
            "output = module(cache)",
            "torch.testing.assert_close(output, torch.arange(20, dtype=torch.float32, device='cuda'), rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTObjectMemcopyMaterializesOutputAlias()
    {
        var objectType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var input = new Var("cache", objectType);
        var output = CreateOutputVar("output", objectType);
        var body = new TIR.Sequential(TIR.T.Memcopy(output, input));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output });

        var outputDirectory = GeneratePyNTTModelDirectory("generated_object_memcopy_alias_model", main);
        RenderGeneratedKernels(outputDirectory);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var function = document.RootElement.GetProperty("functions").EnumerateArray().Single();
        Assert.Equal("cache", function.GetProperty("inputs").EnumerateArray().Single().GetProperty("name").GetString());
        Assert.Equal("object", function.GetProperty("outputs").EnumerateArray().Single().GetProperty("dtype").GetString());
        var kernel = function.GetProperty("generated_kernels").EnumerateArray().Single();
        Assert.Equal("alias", kernel.GetProperty("op_kind").GetString());
        Assert.True(kernel.GetProperty("attrs").GetProperty("pure_alias").GetBoolean());
        Assert.Equal("cache", kernel.GetProperty("attrs").GetProperty("runtime_output_aliases").GetProperty("output0").GetString());
        Assert.False(kernel.GetProperty("attrs").TryGetProperty("output_aliases", out _));

        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        Assert.Contains("outputs[0] = inputs[0]", modelPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTObjectTensorLoadMaterializesOutputAlias()
    {
        var objectType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var input = new Var("cache", objectType);
        var inputBuffer = TIR.T.AttachBuffer(input, objectType, TIR.MemoryLocation.Input, 0, out _, "cache_input");
        var output = CreateOutputVar("output", objectType);
        var placement = new Placement(new[] { 1 }, "b", "b");
        var body = new TIR.Sequential(TIR.F.NTT.TensorLoad(output, inputBuffer, Array.Empty<SBP>(), placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output });

        var outputDirectory = GeneratePyNTTModelDirectory("generated_object_tensor_load_alias_model", main);
        RenderGeneratedKernels(outputDirectory);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var function = document.RootElement.GetProperty("functions").EnumerateArray().Single();
        var kernel = function.GetProperty("generated_kernels").EnumerateArray().Single();
        Assert.Equal("alias", kernel.GetProperty("op_kind").GetString());
        Assert.True(kernel.GetProperty("attrs").GetProperty("pure_alias").GetBoolean());
        Assert.Equal("cache", kernel.GetProperty("attrs").GetProperty("runtime_output_aliases").GetProperty("output0").GetString());
        Assert.False(kernel.GetProperty("attrs").TryGetProperty("output_aliases", out _));

        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        Assert.Contains("outputs[0] = inputs[0]", modelPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTRejectsCompilerScheduledTileOperations()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var input = new Var("input", tensorType);
        var output = CreateOutputVar("output", tensorType);
        var inputBuffer = TIR.T.AttachBuffer(input, tensorType, TIR.MemoryLocation.Input, 0, out _, "input_buffer");
        var outputBuffer = TIR.T.AttachBuffer(output, tensorType, TIR.MemoryLocation.Output, 0, out _, "output_buffer");
        var tileBuffer = CreateDataBuffer("tile", DataTypes.Float32, 0, [4], [1]);
        var outputSubview = TIR.T.Let(
            out var outputTile,
            IR.F.Buffer.BufferSubview(outputBuffer, new RankedShape(0), new RankedShape(4)),
            "output_tile")
            .Body(TIR.T.TileStore(tileBuffer, outputTile))
            .Build();
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.T.TileLoad(tileBuffer, inputBuffer),
                outputSubview),
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 16,
            },
        };

        var exception = Assert.Throws<NotSupportedException>(
            () => GeneratePyNTTModelDirectory("generated_rejected_scheduled_tile_model", main));
        Assert.Contains("TileLoad", exception.Message, StringComparison.Ordinal);
        Assert.Contains("templates", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTEntryOutputSpecDoesNotCollectNestedFunctionOutputs()
    {
        var input = new Var("x", new TensorType(DataTypes.Float32, new[] { 4 }));
        var placement = new Placement(new[] { 1 }, "b", "b");
        var sourceBuffer = CreateBuffer("source", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [4], [1]);
        var publicOutputBuffer = CreateOutputVar("public_output", new TensorType(DataTypes.Float32, new[] { 4 }));
        var nestedOutputBuffer = CreateOutputVar("nested_output", new TensorType(DataTypes.Float32, new[] { 2 }));
        var nested = new TIR.PrimFunction(
            "nested_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(),
            new IVar[] { nestedOutputBuffer });
        var body = new TIR.Sequential(
            nested,
            TIR.F.NTT.TensorStore(sourceBuffer, publicOutputBuffer, new[] { SBP.B }, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, publicOutputBuffer })
        {
            SchedResult =
            {
                DataUsage = 16,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_nested_output_scope_model", main);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var function = document.RootElement.GetProperty("functions").EnumerateArray().Single();
        var outputs = function.GetProperty("outputs").EnumerateArray().ToArray();
        var output = Assert.Single(outputs);
        Assert.Equal("output0", output.GetProperty("name").GetString());
        Assert.Equal("float32", output.GetProperty("dtype").GetString());
        Assert.Equal(new[] { 4L }, output.GetProperty("shape").EnumerateArray().Select(value => value.GetInt64()).ToArray());
    }

    [Fact]
    public void TestPyNTTNestedPrimFunctionUsesCallerWorkspacePointers()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var input = new Var("x", tensorType);
        var publicOutputBuffer = CreateOutputVar("public_output", tensorType);
        var nestedInputBufferVar = new TIR.BufferVar("nested_input", tensorType, TIR.BufferVarRole.Input, TIR.MemoryLocation.Input);
        var nestedOutputBufferVar = CreateOutputVar("nested_output", tensorType);
        var nestedDataVar = new TIR.BufferVar("data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.Data);
        var nestedChipLocalDataVar = new TIR.BufferVar("chip_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.ChipLocalData);
        var nestedBlockLocalDataVar = new TIR.BufferVar("block_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.BlockLocalData);
        var nestedInputBuffer = TIR.T.AttachBuffer(nestedInputBufferVar, tensorType, TIR.MemoryLocation.Input, 0, out _, "nested_input_buffer");
        var nestedTempBuffer = CreateBuffer("nested_temp", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [4], [1]);
        var placement = new Placement(new[] { 1 }, "b", "b");
        var nested = new TIR.PrimFunction(
            "nested_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.TensorLoad(nestedTempBuffer, nestedInputBuffer, new[] { SBP.B }, placement),
                TIR.F.NTT.TensorStore(nestedTempBuffer, nestedOutputBufferVar, new[] { SBP.B }, placement)),
            new IVar[] { nestedInputBufferVar, nestedOutputBufferVar, nestedDataVar, nestedChipLocalDataVar, nestedBlockLocalDataVar })
        {
            SchedResult =
            {
                DataUsage = 128,
            },
        };

        var callerInputBuffer = CreateBuffer("caller_input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [4], [1]);
        var callerOutputBuffer = CreateBuffer("caller_output", DataTypes.Float32, TIR.MemoryLocation.Data, 16, [4], [1]);
        var calleeDataBuffer = CreateBuffer("data_0", DataTypes.UInt8, TIR.MemoryLocation.Data, 64, [128], [1]);
        var calleeChipLocalDataBuffer = CreateBuffer("chip_local_data_0", DataTypes.UInt8, TIR.MemoryLocation.ChipLocalData, 0, [0], [1]);
        var calleeBlockLocalDataBuffer = CreateBuffer("block_local_data_0", DataTypes.UInt8, TIR.MemoryLocation.BlockLocalData, 0, [0], [1]);
        var body = new TIR.Sequential(
            nested,
            TIR.F.NTT.TensorLoad(callerInputBuffer, input, new[] { SBP.B }, placement),
            new Call(nested, callerInputBuffer, callerOutputBuffer, calleeDataBuffer, calleeChipLocalDataBuffer, calleeBlockLocalDataBuffer),
            TIR.F.NTT.TensorStore(callerOutputBuffer, publicOutputBuffer, new[] { SBP.B }, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, publicOutputBuffer })
        {
            SchedResult =
            {
                DataUsage = 192,
            },
        };

        var module = new IRModule(main);
        module.Add(nested);
        var outputDirectory = GeneratePyNTTModelDirectory("generated_nested_call_workspace_model", module);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("def main_prim_nested_prim_device(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("def pyntt_device_", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_nested_prim_device_arg0_nested_input_scalar_stride0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_nested_prim_device_arg1_nested_output_scalar_stride0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(data).to(tl.pointer_type(tl.float32)), 4, 0, 1, 0, 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(data + 16).to(tl.pointer_type(tl.float32)), 4, 0, 1, 0, 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(data + 64).to(tl.pointer_type(tl.uint8))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("pyntt_call_frame", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("data + 192", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTSemanticKernelMaterializesFormalTensorOperands()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var input = new Var("x", tensorType);
        var publicOutput = CreateOutputVar("public_output", tensorType);
        var nestedInput = new TIR.BufferVar(
            "nested_input",
            tensorType,
            TIR.BufferVarRole.Input,
            TIR.MemoryLocation.Input);
        var nestedOutput = CreateOutputVar("nested_output", tensorType);
        var nestedData = new TIR.BufferVar(
            "data",
            TensorType.Scalar(new PointerType(DataTypes.UInt8)),
            TIR.BufferVarRole.Workspace,
            TIR.MemoryLocation.Data);
        var nestedChipLocalData = new TIR.BufferVar(
            "chip_local_data",
            TensorType.Scalar(new PointerType(DataTypes.UInt8)),
            TIR.BufferVarRole.Workspace,
            TIR.MemoryLocation.ChipLocalData);
        var nestedBlockLocalData = new TIR.BufferVar(
            "block_local_data",
            TensorType.Scalar(new PointerType(DataTypes.UInt8)),
            TIR.BufferVarRole.Workspace,
            TIR.MemoryLocation.BlockLocalData);
        var nested = new TIR.PrimFunction(
            "nested_direct_formal_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(TIR.F.NTT.Unary(UnaryOp.Abs, nestedInput, nestedOutput)),
            new IVar[]
            {
                nestedInput,
                nestedOutput,
                nestedData,
                nestedChipLocalData,
                nestedBlockLocalData,
            });

        var placement = new Placement(new[] { 1 }, "b", "b");
        var callerInput = CreateBuffer(
            "caller_input",
            DataTypes.Float32,
            TIR.MemoryLocation.Data,
            0,
            [4],
            [1]);
        var callerOutput = CreateBuffer(
            "caller_output",
            DataTypes.Float32,
            TIR.MemoryLocation.Data,
            16,
            [4],
            [1]);
        var calleeData = CreateBuffer(
            "callee_data",
            DataTypes.UInt8,
            TIR.MemoryLocation.Data,
            32,
            [1],
            [1]);
        var calleeChipLocalData = CreateBuffer(
            "callee_chip_local_data",
            DataTypes.UInt8,
            TIR.MemoryLocation.ChipLocalData,
            0,
            [0],
            [1]);
        var calleeBlockLocalData = CreateBuffer(
            "callee_block_local_data",
            DataTypes.UInt8,
            TIR.MemoryLocation.BlockLocalData,
            0,
            [0],
            [1]);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                nested,
                TIR.F.NTT.TensorLoad(callerInput, input, new[] { SBP.B }, placement),
                new Call(
                    nested,
                    callerInput,
                    callerOutput,
                    calleeData,
                    calleeChipLocalData,
                    calleeBlockLocalData),
                TIR.F.NTT.TensorStore(callerOutput, publicOutput, new[] { SBP.B }, placement)),
            new IVar[] { input, publicOutput })
        {
            SchedResult =
            {
                DataUsage = 64,
            },
        };

        var module = new IRModule(main);
        module.Add(nested);
        var outputDirectory = GeneratePyNTTModelDirectory(
            "generated_direct_formal_semantic_kernel_model",
            module);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains(
            "nested_direct_formal_prim_device__elementwise_unary__0",
            generatedKernelsPy,
            StringComparison.Ordinal);
        Assert.Contains(
            "main_prim_nested_direct_formal_prim_device_arg1_nested_output",
            generatedKernelsPy,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTWorkspaceSizingUsesPhysicalAllocationForDynamicAlias()
    {
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = new(1, 128) } };
        var inputType = new TensorType(DataTypes.Boolean, new Dimension[] { sequenceLength });
        var outputType = new TensorType(DataTypes.Boolean, new Dimension[] { sequenceLength, 1 });
        var input = new TIR.BufferVar("input", inputType, TIR.BufferVarRole.Input, TIR.MemoryLocation.Input);
        var output = CreateOutputVar("output", outputType);
        var physicalBuffer = new TIR.PhysicalBuffer(1, 0, 128, TIR.MemoryLocation.Data);
        var storage = new TIR.Buffer(
            "storage",
            DataTypes.Boolean,
            new TIR.MemSpan(physicalBuffer),
            new Dimension[] { sequenceLength },
            new Dimension[] { 1 },
            null);
        var logicalAlias = TIR.T.CreateBufferView(
            storage,
            DataTypes.Boolean,
            new Dimension[] { sequenceLength, 1 },
            new Dimension[] { 1, 0 },
            0,
            sequenceLength,
            name: "logical_alias");
        var placement = new Placement(new[] { 1 }, "b", "b");
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.TensorLoad(storage, input, new[] { SBP.B }, placement),
                TIR.F.NTT.TensorStore(logicalAlias, output, new[] { SBP.B, SBP.B }, placement)),
            new IVar[] { input, sequenceLength, output })
        {
            SchedResult =
            {
                DataUsage = 128,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_dynamic_alias_workspace_model", main);
        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        var dataAllocation = Assert.Single(modelPy.Split('\n').Where(line => line.TrimStart().StartsWith("data = self.allocate_workspace", StringComparison.Ordinal)));
        Assert.Contains(", 128 * grid[0], \"uint8\")", dataAllocation, StringComparison.Ordinal);
        Assert.DoesNotContain("sequence_length", dataAllocation, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTBufferSubviewUsesFormalAbiStrides()
    {
        var publicType = new TensorType(DataTypes.Float32, new[] { 1 });
        var input = new Var("x", publicType);
        var publicOutput = CreateOutputVar("public_output", publicType);
        var tensorType = new TensorType(DataTypes.Float32, new[] { 8, 16 });
        var nestedInputVar = new TIR.BufferVar(
            "nested_input",
            tensorType,
            TIR.BufferVarRole.Input,
            TIR.MemoryLocation.Input,
            TIR.BufferLayoutAnnotation.RuntimeStrided);
        var nestedOutputVar = new TIR.BufferVar(
            "nested_output",
            tensorType,
            TIR.BufferVarRole.Output,
            TIR.MemoryLocation.Output,
            TIR.BufferLayoutAnnotation.RuntimeStrided);
        var nestedDataVar = new TIR.BufferVar("data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.Data);
        var nestedChipLocalDataVar = new TIR.BufferVar("chip_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.ChipLocalData);
        var nestedBlockLocalDataVar = new TIR.BufferVar("block_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.BlockLocalData);
        var nestedInput = TIR.T.AttachBuffer(nestedInputVar, tensorType, TIR.MemoryLocation.Input, 0, out _, "nested_input_buffer");
        var nestedOutput = TIR.T.AttachBuffer(nestedOutputVar, tensorType, TIR.MemoryLocation.Output, 0, out _, "nested_output_buffer");
        var placement = new Placement(new[] { 1 }, "b", "b");
        var nested = new TIR.PrimFunction(
            "nested_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.Unary(
                    UnaryOp.Abs,
                    IR.F.Buffer.BufferSubview(nestedInput, new RankedShape(2, 0), new RankedShape(2, 16)),
                    IR.F.Buffer.BufferSubview(nestedOutput, new RankedShape(3, 0), new RankedShape(2, 16))),
                TIR.F.NTT.TensorStore(nestedOutput, nestedOutputVar, new[] { SBP.B, SBP.B }, placement)),
            new IVar[] { nestedInputVar, nestedOutputVar, nestedDataVar, nestedChipLocalDataVar, nestedBlockLocalDataVar });

        var callerInput = CreateBuffer("caller_input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [8, 16], [4, 1]);
        var callerOutput = CreateBuffer("caller_output", DataTypes.Float32, TIR.MemoryLocation.Data, 512, [8, 16], [8, 1]);
        var calleeData = CreateBuffer("callee_data", DataTypes.UInt8, TIR.MemoryLocation.Data, 1024, [1], [1]);
        var calleeChipLocalData = CreateBuffer("callee_chip_local_data", DataTypes.UInt8, TIR.MemoryLocation.ChipLocalData, 0, [0], [1]);
        var calleeBlockLocalData = CreateBuffer("callee_block_local_data", DataTypes.UInt8, TIR.MemoryLocation.BlockLocalData, 0, [0], [1]);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                nested,
                new Call(nested, callerInput, callerOutput, calleeData, calleeChipLocalData, calleeBlockLocalData),
                TIR.T.Memcopy(publicOutput, input)),
            new IVar[] { input, publicOutput })
        {
            SchedResult =
            {
                DataUsage = 2048,
            },
        };

        var module = new IRModule(main);
        module.Add(nested);
        var outputDirectory = GeneratePyNTTModelDirectory("generated_formal_subview_stride_model", module);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("main_prim_nested_prim_device_arg0_nested_input_scalar_stride0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_nested_prim_device_arg1_nested_output_scalar_stride0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("* 2", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("* 3", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTDistributedBufferSubviewPreservesFormalAbiStrides()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 2 } };

        var publicType = new TensorType(DataTypes.Float32, new[] { 1 });
        var input = new Var("x", publicType);
        var publicOutput = CreateOutputVar("public_output", publicType);
        var tensorType = new TensorType(DataTypes.Float32, new[] { 8, 8 });
        var placement = new Placement(new[] { 2 }, "b", "b");
        var distributedType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0], 4) },
            placement);
        var nestedInputVar = new TIR.BufferVar(
            "nested_input",
            distributedType,
            TIR.BufferVarRole.Input,
            TIR.MemoryLocation.Input,
            TIR.BufferLayoutAnnotation.RuntimeStrided);
        var nestedOutputVar = new TIR.BufferVar(
            "nested_output",
            distributedType,
            TIR.BufferVarRole.Output,
            TIR.MemoryLocation.Output,
            TIR.BufferLayoutAnnotation.RuntimeStrided);
        var nestedDataVar = new TIR.BufferVar("data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.Data);
        var nestedChipLocalDataVar = new TIR.BufferVar("chip_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.ChipLocalData);
        var nestedBlockLocalDataVar = new TIR.BufferVar("block_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.BlockLocalData);
        var nestedInput = TIR.T.AttachBuffer(nestedInputVar, tensorType, TIR.MemoryLocation.Input, 0, out _, "nested_input_buffer", distributedType);
        var nestedOutput = TIR.T.AttachBuffer(nestedOutputVar, tensorType, TIR.MemoryLocation.Output, 0, out _, "nested_output_buffer", distributedType);
        var nested = new TIR.PrimFunction(
            "nested_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.Unary(
                    UnaryOp.Abs,
                    IR.F.Buffer.BufferSubview(nestedInput, new RankedShape(2, 0), new RankedShape(2, 4)),
                    IR.F.Buffer.BufferSubview(nestedOutput, new RankedShape(3, 0), new RankedShape(2, 4))),
                TIR.F.NTT.TensorStore(nestedOutput, nestedOutputVar, distributedType.AxisPolicies, placement)),
            new IVar[] { nestedInputVar, nestedOutputVar, nestedDataVar, nestedChipLocalDataVar, nestedBlockLocalDataVar });

        // The local shard is [8, 4], but it aliases rows of the chip-global
        // [8, 8] tensor and therefore has a non-contiguous row stride of 8.
        var callerInput = CreateBuffer("caller_input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [8, 4], [8, 1], distributedType);
        var callerOutput = CreateBuffer("caller_output", DataTypes.Float32, TIR.MemoryLocation.Data, 256, [8, 4], [8, 1], distributedType);
        var calleeData = CreateBuffer("callee_data", DataTypes.UInt8, TIR.MemoryLocation.Data, 512, [1], [1]);
        var calleeChipLocalData = CreateBuffer("callee_chip_local_data", DataTypes.UInt8, TIR.MemoryLocation.ChipLocalData, 0, [0], [1]);
        var calleeBlockLocalData = CreateBuffer("callee_block_local_data", DataTypes.UInt8, TIR.MemoryLocation.BlockLocalData, 0, [0], [1]);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                nested,
                new Call(nested, callerInput, callerOutput, calleeData, calleeChipLocalData, calleeBlockLocalData),
                TIR.T.Memcopy(publicOutput, input)),
            new IVar[] { input, publicOutput })
        {
            SchedResult =
            {
                DataUsage = 1024,
            },
        };

        var module = new IRModule(main);
        module.Add(nested);
        var outputDirectory = GeneratePyNTTModelDirectory("generated_distributed_formal_subview_stride_model", module);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        var inputStride0 = "main_prim_nested_prim_device_arg0_nested_input_scalar_stride0";
        var outputStride0 = "main_prim_nested_prim_device_arg1_nested_output_scalar_stride0";
        Assert.Contains($"({inputStride0} * 2)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains($"({outputStride0} * 3)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains($"* ({inputStride0})", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains($"* ({outputStride0})", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTCanonicalGlobalPointerDoesNotRebaseLocalShardTwice()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 2 } };

        var placement = new Placement(new[] { 2 }, "b", "b");
        var globalType = new TensorType(DataTypes.Float32, new[] { 8 });
        var distributedType = new DistributedType(
            globalType,
            new SBP[] { SBP.SBlockCyclic([0], 1) },
            placement);
        var shardCoordinate = new DimVar("__shard_coord_0");
        var physicalInput = new TIR.PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            128,
            32,
            TIR.MemoryLocation.ChipLocalData);
        var physicalOutput = new TIR.PhysicalBuffer(
            DataTypes.Float32.SizeInBytes,
            160,
            32,
            TIR.MemoryLocation.ChipLocalData);
        var canonicalInput = new TIR.Buffer(
            "canonical_input",
            DataTypes.Float32,
            new TIR.MemSpan(physicalInput, shardCoordinate * 4, 16),
            new Dimension[] { 4 },
            new Dimension[] { 1 },
            distributedType,
            distributedStorageKind: TIR.DistributedBufferStorageKind.CanonicalGlobal);
        var canonicalOutput = new TIR.Buffer(
            "canonical_output",
            DataTypes.Float32,
            new TIR.MemSpan(physicalOutput, shardCoordinate * 4, 16),
            new Dimension[] { 4 },
            new Dimension[] { 1 },
            distributedType,
            distributedStorageKind: TIR.DistributedBufferStorageKind.CanonicalGlobal);
        var publicType = new TensorType(DataTypes.Float32, new[] { 1 });
        var input = new Var("x", publicType);
        var output = CreateOutputVar("output", publicType);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.Unary(UnaryOp.Abs, canonicalInput, canonicalOutput),
                TIR.T.Memcopy(output, input)),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                ChipLocalDataPoolSize = 192,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory(
            "generated_canonical_global_pointer_model",
            main);
        using var manifest = JsonDocument.Parse(
            File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var unaryHelper = manifest.RootElement
            .GetProperty("functions")[0]
            .GetProperty("render_kernels")[0]
            .GetProperty("helpers")
            .EnumerateArray()
            .Single(helper => helper.GetProperty("template").GetString() == "triton/kernels/ElementwiseUnary.py.jinja");
        var inputPointer = unaryHelper.GetProperty("model").GetProperty("Input");
        var inputExpression = inputPointer.GetProperty("Expression").GetString();
        Assert.Contains("128", inputExpression, StringComparison.Ordinal);
        Assert.DoesNotContain("shard_coord", inputExpression, StringComparison.Ordinal);
        Assert.All(
            inputPointer.GetProperty("GlobalOffsets").EnumerateArray(),
            offset => Assert.Equal(0, offset.GetProperty("FixedValue").GetInt64()));

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("shard_coord0", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTFormalReshardUsesByteAddressedBackingPool()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 2 } };

        var tensorType = new TensorType(DataTypes.BFloat16, new[] { 4, 8 });
        var placement = new Placement(new[] { 2 }, "b", "b");
        var inputDistributedType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.SContiguous([0], 4) }, placement);
        var outputDistributedType = new DistributedType(tensorType, new SBP[] { SBP.SContiguous([0], 2), SBP.B }, placement);
        var nestedInput = new TIR.BufferVar("nested_input", inputDistributedType, TIR.BufferVarRole.Input, TIR.MemoryLocation.Input);
        var nestedOutput = new TIR.BufferVar("nested_output", outputDistributedType, TIR.BufferVarRole.Output, TIR.MemoryLocation.Output);
        var nestedData = new TIR.BufferVar("data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.Data);
        var nestedChipLocalData = new TIR.BufferVar("chip_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.ChipLocalData);
        var nestedBlockLocalData = new TIR.BufferVar("block_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.BlockLocalData);
        var nestedInputBuffer = TIR.T.AttachBuffer(
            nestedInput,
            DistributedUtility.GetDividedTensorType(inputDistributedType),
            TIR.MemoryLocation.Input,
            0,
            out _,
            "nested_input_buffer",
            inputDistributedType);
        var nestedOutputBuffer = TIR.T.AttachBuffer(
            nestedOutput,
            DistributedUtility.GetDividedTensorType(outputDistributedType),
            TIR.MemoryLocation.Output,
            0,
            out _,
            "nested_output_buffer",
            outputDistributedType);
        var nested = new TIR.PrimFunction(
            "nested_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.GatherReduceScatter(nestedInputBuffer, nestedOutputBuffer, inputDistributedType, outputDistributedType),
                TIR.F.NTT.TensorStore(nestedOutputBuffer, nestedOutput, outputDistributedType.AxisPolicies, placement)),
            new IVar[] { nestedInput, nestedOutput, nestedData, nestedChipLocalData, nestedBlockLocalData });

        var input = new Var("input", tensorType);
        var output = CreateOutputVar("output", tensorType);
        var callerInput = CreateBuffer("caller_input", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [4, 4], [4, 1], inputDistributedType);
        var callerOutput = CreateBuffer("caller_output", DataTypes.BFloat16, TIR.MemoryLocation.Data, 32, [2, 8], [8, 1], outputDistributedType);
        var callerData = CreateBuffer("callee_data", DataTypes.UInt8, TIR.MemoryLocation.Data, 64, [1], [1]);
        var callerChipLocalData = CreateBuffer("callee_chip_local_data", DataTypes.UInt8, TIR.MemoryLocation.ChipLocalData, 0, [1], [1]);
        var callerBlockLocalData = CreateBuffer("callee_block_local_data", DataTypes.UInt8, TIR.MemoryLocation.BlockLocalData, 0, [1], [1]);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                nested,
                TIR.F.NTT.TensorLoad(callerInput, input, inputDistributedType.AxisPolicies, placement),
                new Call(nested, callerInput, callerOutput, callerData, callerChipLocalData, callerBlockLocalData),
                TIR.F.NTT.TensorStore(callerOutput, output, outputDistributedType.AxisPolicies, placement)),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 128,
            },
        };

        var module = new IRModule(main);
        module.Add(nested);
        var outputDirectory = GeneratePyNTTModelDirectory("generated_formal_reshard_byte_address_model", module);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        var lines = generatedKernelsPy.Split('\n');
        var byteOffsetLine = Assert.Single(lines.Where(line => line.Contains("output_byte_offsets =", StringComparison.Ordinal)));
        Assert.Contains("_pool_stride_elements) * 2", byteOffsetLine, StringComparison.Ordinal);
        var storeLine = Assert.Single(lines.Where(line => line.Contains("output_byte_offsets).to(tl.pointer_type(tl.bfloat16))", StringComparison.Ordinal)));
        Assert.Contains("tl.pointer_type(tl.uint8)", storeLine, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTGatherReduceScatterMaterializesCallerAllocatedOutput()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 2 } };

        var tensorType = new TensorType(DataTypes.BFloat16, new[] { 4, 8 });
        var placement = new Placement(new[] { 2 }, "b", "b");
        var inputType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0], 4) },
            placement);
        var outputType = new DistributedType(
            tensorType,
            new SBP[] { SBP.SContiguous([0], 2), SBP.B },
            placement);
        var input = CreateBuffer(
            "input_shard",
            DataTypes.BFloat16,
            TIR.MemoryLocation.Data,
            0,
            [4, 4],
            [4, 1],
            inputType);
        var output = CreateOutputVar("output", outputType);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.GatherReduceScatter(input, output, inputType, outputType)),
            new TIR.Return(new Expr[] { output }),
            new IVar[] { output })
        {
            SchedResult =
            {
                DataUsage = 64,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory(
            "generated_direct_reshard_output_model",
            main);
        using var manifest = JsonDocument.Parse(
            File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var renderKernel = manifest.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
            .Single();
        Assert.Equal(
            new[] { "output0" },
            renderKernel.GetProperty("metadata")
                .GetProperty("outputs")
                .EnumerateArray()
                .Select(value => value.GetString())
                .ToArray());
    }

    [Fact]
    public async Task TestEntryShardedViewUsesCallerAllocatedCanonicalOutput()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 64 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([1]) },
            placement);
        var input = new Var("input", inputType);
        var cast = IR.F.Tensors.Cast(input, DataTypes.BFloat16);
        Assert.True(cast.InferenceType());
        var castType = Assert.IsType<DistributedType>(cast.CheckedType);
        var outputType = new DistributedType(
            castType.TensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var outputView = IR.F.Distributed.ShardedView(cast, outputType);
        var main = new Function("main", PyNTTTarget.Kind, outputView, new[] { input });
        Assert.True(main.InferenceType());

        var mainPrim = Assert.IsType<TIR.PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(main, new()));
        var abi = TIR.PrimFunctionAbi.GetAbiView(mainPrim);
        var outputParameter = Assert.Single(abi.OutputParameters);
        Assert.Equal(outputType, outputParameter.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(mainPrim.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.TensorStore);

        var selectedCast = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Cast));
        var castOutput = Assert.IsType<TIR.Buffer>(selectedCast.Arguments[1]);
        Assert.Equal(TIR.MemoryLocation.Output, castOutput.MemSpan.Buffer.Location);
        Assert.Same(outputParameter, castOutput.MemSpan.Buffer.Start);
        Assert.Equal(castType, castOutput.DistributedType);
        var resultView = Assert.IsType<TIR.Buffer>(Assert.Single(abi.Results).Value);
        Assert.Same(castOutput.MemSpan.Buffer, resultView.MemSpan.Buffer);
        Assert.Equal(outputType, resultView.DistributedType);

        var codegenMain = Assert.IsType<TIR.PrimFunction>(
            new Passes.Mutators.RemoveNop().Rewrite(mainPrim));
        var outputDirectory = GeneratePyNTTModelDirectory(
            "generated_entry_sharded_view_output_model",
            codegenMain);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.DoesNotContain(
            "tl.where(output0_pool_stride_elements == 0",
            generatedKernelsPy,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestShardedViewCanReadCallerAllocatedTupleOutput()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 64 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([1]) },
            placement);
        var input = new Var("input", inputType);
        var cast = IR.F.Tensors.Cast(input, DataTypes.BFloat16);
        Assert.True(cast.InferenceType());
        var castType = Assert.IsType<DistributedType>(cast.CheckedType);
        var broadcastType = new DistributedType(
            castType.TensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var outputView = IR.F.Distributed.ShardedView(cast, broadcastType);
        var stats = IR.F.NN.NormStats(1, outputView, useMean: false);
        var main = new Function("main", PyNTTTarget.Kind, new IR.Tuple(cast, stats), new[] { input });
        Assert.True(main.InferenceType());

        var mainPrim = Assert.IsType<TIR.PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(main, new()));
        var abi = TIR.PrimFunctionAbi.GetAbiView(mainPrim);
        Assert.Equal(2, abi.OutputParameters.Count);
        var valueOutput = abi.OutputParameters[0];
        var selectedCast = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Cast));
        var castOutput = Assert.IsType<TIR.Buffer>(selectedCast.Arguments[1]);
        Assert.Same(valueOutput, castOutput.MemSpan.Buffer.Start);
        Assert.Equal(
            TIR.DistributedBufferStorageKind.CanonicalGlobal,
            castOutput.DistributedStorageKind);

        var selectedStats = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.NormStats));
        var statsInput = Assert.IsType<TIR.Buffer>(selectedStats.Arguments[0]);
        Assert.Same(castOutput.MemSpan.Buffer, statsInput.MemSpan.Buffer);
        Assert.Equal(broadcastType, statsInput.DistributedType);
    }

    [Fact]
    public async Task TestCallerAllocatesCanonicalTupleOutputUsingChipLocalBacking()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 64 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([1]) },
            placement);
        var layerInput = new Var("layer_input", inputType);
        var cast = IR.F.Tensors.Cast(layerInput, DataTypes.BFloat16);
        Assert.True(cast.InferenceType());
        var castType = Assert.IsType<DistributedType>(cast.CheckedType);
        var broadcastType = new DistributedType(
            castType.TensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var outputView = IR.F.Distributed.ShardedView(cast, broadcastType);
        var stats = IR.F.NN.NormStats(1, outputView, useMean: false);
        var layer = new Function("layer", new IR.Tuple(outputView, stats), layerInput);
        Assert.True(layer.InferenceType());

        var input = new Var("input", inputType);
        var layerCall = new Call(layer, input);
        var main = new Function("main", IR.F.Tensors.GetItem(layerCall, 1), input);
        Assert.True(main.InferenceType());
        var module = new IRModule(main);
        module.Add(layer);

        var passManager = CompileSession.CreatePassManager("CanonicalCallerOutputTIRSelection");
        passManager.Add<NTTTIRSelectionPass>();
        await passManager.RunAsync(module);

        var layerPrim = Assert.Single(module.Functions.OfType<PrimFunctionWrapper>()).Target;
        var outputParameter = TIR.PrimFunctionAbi.GetAbiView(layerPrim).OutputParameters[0];
        Assert.Equal(
            TIR.DistributedBufferStorageKind.CanonicalGlobal,
            outputParameter.LayoutAnnotation.DistributedStorageKind);
        var mainPrim = Assert.IsType<TIR.PrimFunction>(module.Entry);
        var selectedCall = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => ReferenceEquals(call.Target, layerPrim)));
        var outputParameterIndex = Array.FindIndex(
            layerPrim.Parameters.ToArray(),
            parameter => ReferenceEquals(parameter, outputParameter));
        var actualOutput = Assert.IsType<TIR.Buffer>(selectedCall.Arguments[outputParameterIndex]);
        Assert.Equal(TIR.MemoryLocation.ChipLocalData, actualOutput.MemSpan.Buffer.Location);
        Assert.Equal(TIR.DistributedBufferStorageKind.CanonicalGlobal, actualOutput.DistributedStorageKind);
        Assert.Equal(4096, actualOutput.MemSpan.Buffer.Size.FixedValue);
    }

    [Fact]
    public async Task TestEntryVectorizedViewChainUsesCallerAllocatedCanonicalOutput()
    {
        var tensorType = new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 1, 16 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([1]) },
            placement);
        var input = new Var("input", inputType);
        var cast = IR.F.NTT.VectorizedCast(
            input,
            new VectorType(DataTypes.Float32, [4]),
            CastMode.KDefault,
            [1],
            None.Default);
        var bitcast = IR.F.Tensors.Bitcast(cast, DataTypes.Float32);
        Assert.True(bitcast.InferenceType());
        var bitcastType = Assert.IsType<DistributedType>(bitcast.CheckedType);
        var outputType = new DistributedType(
            bitcastType.TensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var outputView = IR.F.Distributed.ShardedView(bitcast, outputType);
        var main = new Function("main", PyNTTTarget.Kind, outputView, new[] { input });
        Assert.True(main.InferenceType());

        var mainPrim = Assert.IsType<TIR.PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(main, new()));
        var abi = TIR.PrimFunctionAbi.GetAbiView(mainPrim);
        var outputParameter = Assert.Single(abi.OutputParameters);
        Assert.Equal(outputType, outputParameter.CheckedType);
        Assert.DoesNotContain(
            ExprCollector.Collect(mainPrim.Body).OfType<Call>(),
            call => call.Target is TIR.NTT.TensorStore);

        var selectedCast = Assert.Single(
            ExprCollector.Collect(mainPrim.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.Cast));
        var castOutput = Assert.IsType<TIR.Buffer>(selectedCast.Arguments[1]);
        Assert.Equal(TIR.MemoryLocation.Output, castOutput.MemSpan.Buffer.Location);
        Assert.Same(outputParameter, castOutput.MemSpan.Buffer.Start);
        Assert.Equal(Assert.IsType<DistributedType>(cast.CheckedType), castOutput.DistributedType);

        var resultView = Assert.IsType<TIR.Buffer>(Assert.Single(abi.Results).Value);
        Assert.Same(castOutput.MemSpan.Buffer, resultView.MemSpan.Buffer);
        Assert.Equal(outputType, resultView.DistributedType);
    }

    [Fact]
    public async Task TestVectorizedCastUsesNonUniformActiveShardShape()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 1, 18992 }),
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 594) },
            placement);
        var input = new Var("input", inputType);
        var cast = IR.F.NTT.VectorizedCast(
            input,
            new VectorType(DataTypes.Float32, [4]),
            CastMode.KDefault,
            [1],
            None.Default);
        var bitcast = IR.F.Tensors.Bitcast(cast, DataTypes.Float32);
        var outputType = new DistributedType(
            Assert.IsType<DistributedType>(bitcast.CheckedType).TensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var main = new Function(
            "main",
            PyNTTTarget.Kind,
            IR.F.Distributed.ShardedView(bitcast, outputType),
            new[] { input });
        Assert.True(main.InferenceType());

        var mainPrim = Assert.IsType<TIR.PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, PyNTTTarget.Kind).RunAsync(main, new()));
        var codegenMain = Assert.IsType<TIR.PrimFunction>(
            new Passes.Mutators.RemoveNop().Rewrite(mainPrim));
        var outputDirectory = GeneratePyNTTModelDirectory(
            "generated_non_uniform_vectorized_cast_model",
            codegenMain);

        using var kernelParams = JsonDocument.Parse(
            File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var castModel = Assert.Single(
            kernelParams.RootElement
                .GetProperty("functions")
                .EnumerateArray()
                .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
                .SelectMany(kernel => kernel.GetProperty("helpers").EnumerateArray())
                .Where(helper => helper.GetProperty("template").GetString() == "triton/kernels/ElementwiseCast.py.jinja"))
            .GetProperty("model");
        var inputExtent = castModel.GetProperty("InputShape")[1];
        var outputExtent = castModel.GetProperty("OutputShape")[1];
        Assert.Equal(JsonValueKind.Null, inputExtent.GetProperty("FixedValue").ValueKind);
        Assert.Equal(JsonValueKind.Null, outputExtent.GetProperty("FixedValue").ValueKind);
        Assert.Equal(594, inputExtent.GetProperty("RangeMax").GetInt64());
        Assert.Equal(1188, outputExtent.GetProperty("RangeMax").GetInt64());
        Assert.Contains("tl.minimum", inputExtent.GetProperty("TritonExpression").GetString(), StringComparison.Ordinal);
        Assert.Contains("tl.minimum", outputExtent.GetProperty("TritonExpression").GetString(), StringComparison.Ordinal);

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains(
            "for major_start in tl.range(0, tl.maximum(0, tl.minimum",
            generatedKernelsPy,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTPartialReshardIsSinglePhaseAndWritesEachDestinationOnce()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 4 } };

        var tensorType = new TensorType(DataTypes.BFloat16, new[] { 8, 16 });
        var placement = new Placement(new[] { 4 }, "b", "b");
        var partialType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement,
            SBP.P([0], ReduceOp.Sum));
        var splitType = new DistributedType(
            tensorType,
            new SBP[] { SBP.SContiguous([0], 2), SBP.B },
            placement);
        var partialBuffer = CreateCompactPerOwnerBuffer("partial", DataTypes.BFloat16, 0, [8, 16], [16, 1], partialType);
        var splitBuffer = CreateBuffer("split", DataTypes.BFloat16, TIR.MemoryLocation.Data, 256, [2, 16], [16, 1], splitType);
        var input = new Var("input", tensorType);
        var output = CreateOutputVar("output", tensorType);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.TensorLoad(partialBuffer, input, partialType.AxisPolicies, placement),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Chip),
                TIR.F.NTT.GatherReduceScatter(partialBuffer, splitBuffer, partialType, splitType),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
                TIR.F.NTT.TensorStore(splitBuffer, output, splitType.AxisPolicies, placement)),
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 512,
                ChipLocalDataPoolSize = 1024,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_partial_reshard_single_phase_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("stage=tile_scatter", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(shard_coord0 == destination_shard_coord0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("(shard_coord0 == 0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("partial_lane = partial_start + tl.arange(0, partial_reduction_width)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("partial_reduction_width: tl.constexpr = min(32, 4)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("partial_worker_thread_id = tl.inline_asm_elementwise", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("mask = mask & (partial_worker_thread_id < partial_worker_count)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("acc += tl.sum(source_value, axis=1)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("for reduce_coord", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("source_pool_index = source_shard_index", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Matches(@"destination_shard_coord0 = .*% 4", generatedKernelsPy);
        Assert.DoesNotContain("collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("shared_shard", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Equal(1, generatedKernelsPy.Split("tle.distributed_barrier", StringSplitOptions.None).Length - 1);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(8 * 16, dtype=torch.float32, device='cuda').reshape(8, 16) - 31) * 0.03125).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x * 4, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTTwoDimensionalPartialReduceScatterRuns()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };

        var tensorType = new TensorType(DataTypes.BFloat16, new[] { 1, 256 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var partialType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement,
            SBP.P([0, 1], ReduceOp.Sum));
        var splitType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 8) },
            placement);
        var partialBuffer = CreateCompactPerOwnerBuffer(
            "partial",
            DataTypes.BFloat16,
            0,
            [1, 256],
            [0, 1],
            partialType);
        var splitBuffer = CreateBuffer(
            "split",
            DataTypes.BFloat16,
            TIR.MemoryLocation.Data,
            512,
            [1, 8],
            [0, 1],
            splitType);
        var input = new Var("input", tensorType);
        var output = CreateOutputVar("output", tensorType);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.TensorLoad(partialBuffer, input, partialType.AxisPolicies, placement),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Chip),
                TIR.F.NTT.GatherReduceScatter(partialBuffer, splitBuffer, partialType, splitType),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
                TIR.F.NTT.TensorStore(splitBuffer, output, splitType.AxisPolicies, placement)),
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 528,
                ChipLocalDataPoolSize = 16384,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_two_dimensional_partial_reshard_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("source_shard_coord0 = (partial_lane // 8) % 4", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("source_shard_coord1 = (partial_lane // 1) % 8", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("acc += tl.sum(source_value, axis=1)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(256, dtype=torch.float32, device='cuda').reshape(1, 256) - 127) * 0.03125).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x * 32, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTVectorizedTwoDimensionalPartialReduceScatterRuns()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };

        var scalarTensorType = new TensorType(DataTypes.BFloat16, new[] { 1, 2048 });
        var vectorElementType = new VectorType(DataTypes.BFloat16, [8]);
        var vectorTensorType = new TensorType(vectorElementType, new[] { 1, 256 });
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var scalarPartialType = new DistributedType(
            scalarTensorType,
            new SBP[] { SBP.B, SBP.B },
            placement,
            SBP.P([0, 1], ReduceOp.Sum));
        var vectorPartialType = new DistributedType(
            vectorTensorType,
            new SBP[] { SBP.B, SBP.B },
            placement,
            SBP.P([0, 1], ReduceOp.Sum));
        var vectorSplitType = new DistributedType(
            vectorTensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 8) },
            placement);
        var scalarSplitType = new DistributedType(
            scalarTensorType,
            new SBP[] { SBP.B, SBP.SContiguous([0, 1], 64) },
            placement);
        var scalarPartialBuffer = CreateCompactPerOwnerBuffer(
            "scalar_partial",
            DataTypes.BFloat16,
            0,
            [1, 2048],
            [0, 1],
            scalarPartialType);
        var vectorPartialBuffer = CreateCompactPerOwnerBuffer(
            "vector_partial",
            vectorElementType,
            131072,
            [1, 256],
            [0, 1],
            vectorPartialType);
        var vectorSplitBuffer = CreateBuffer(
            "vector_split",
            vectorElementType,
            TIR.MemoryLocation.Data,
            8192,
            [1, 8],
            [0, 1],
            vectorSplitType);
        var scalarSplitBuffer = CreateBuffer(
            "scalar_split",
            DataTypes.BFloat16,
            TIR.MemoryLocation.Data,
            8320,
            [1, 64],
            [0, 1],
            scalarSplitType);
        var input = new Var("input", scalarTensorType);
        var output = CreateOutputVar("output", scalarTensorType);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.TensorLoad(scalarPartialBuffer, input, scalarPartialType.AxisPolicies, placement),
                TIR.F.NTT.Pack(scalarPartialBuffer, vectorPartialBuffer, new[] { 8 }, new[] { 1 }),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Chip),
                TIR.F.NTT.GatherReduceScatter(vectorPartialBuffer, vectorSplitBuffer, vectorPartialType, vectorSplitType),
                TIR.F.NTT.Unpack(vectorSplitBuffer, scalarSplitBuffer, new[] { 8 }, new[] { 1 }),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
                TIR.F.NTT.TensorStore(scalarSplitBuffer, output, scalarSplitType.AxisPolicies, placement)),
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 8448,
                ChipLocalDataPoolSize = 262144,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_vectorized_two_dimensional_partial_reshard_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("lane=8, stage=tile_scatter", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("source_shard_coord0 = (partial_lane // 8) % 4", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("source_shard_coord1 = (partial_lane // 1) % 8", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("partial_reduction_width: tl.constexpr = min(32, 32)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("partial_worker_count: tl.constexpr = min(256, elementwise_physical_tile_width * 8 * partial_reduction_width)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("acc += tl.sum(source_value, axis=2)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(2048, dtype=torch.float32, device='cuda').reshape(1, 2048) - 1023) * 0.001953125).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x * 32, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTPartialToBroadcastReshardRepeatsReductionPerBlock()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "b";
        targetOptions.HierarchyLevels = "b";
        targetOptions.Hierarchies = new[] { new[] { 4 } };

        var tensorType = new TensorType(DataTypes.BFloat16, new[] { 8, 16 });
        var placement = new Placement(new[] { 4 }, "b", "b");
        var partialType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement,
            SBP.P([0], ReduceOp.Sum));
        var broadcastType = new DistributedType(
            tensorType,
            new SBP[] { SBP.B, SBP.B },
            placement);
        var partialBuffer = CreateCompactPerOwnerBuffer("partial", DataTypes.BFloat16, 0, [8, 16], [16, 1], partialType);
        var broadcastBuffer = CreateBuffer("broadcast", DataTypes.BFloat16, TIR.MemoryLocation.Data, 256, [8, 16], [16, 1], broadcastType);
        var input = new Var("input", tensorType);
        var output = CreateOutputVar("output", tensorType);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.F.NTT.TensorLoad(partialBuffer, input, partialType.AxisPolicies, placement),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Chip),
                TIR.F.NTT.GatherReduceScatter(partialBuffer, broadcastBuffer, partialType, broadcastType),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
                TIR.F.NTT.TensorStore(broadcastBuffer, output, broadcastType.AxisPolicies, placement)),
            new TIR.Return(new Expr[] { output }),
            new IVar[] { input, output })
        {
            SchedResult =
            {
                DataUsage = 512,
                ChipLocalDataPoolSize = 1024,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_partial_to_broadcast_reshard_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("stage=tile_scatter", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("destination_shard_coord0 = shard_coord0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("for destination_shard_coord0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("(shard_coord0 == 0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("partial_lane = partial_start + tl.arange(0, partial_reduction_width)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("acc += tl.sum(source_value, axis=1)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = ((torch.arange(8 * 16, dtype=torch.float32, device='cuda').reshape(8, 16) - 31) * 0.03125).to(torch.bfloat16)",
            "output = module(x)",
            "torch.testing.assert_close(output, x * 4, rtol=0, atol=0)");
    }

    [Fact]
    public void TestPyNTTGridBarrierPreservesLogicalMeshAndRendersAxisGroups()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };

        var tensorType = new TensorType(DataTypes.Float32, new[] { 1 });
        var input = new Var("input", tensorType);
        var output = CreateOutputVar("output", tensorType);
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(
                TIR.T.Memcopy(output, input),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Chip, [0]),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Chip, [1]),
                TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Chip)),
            new IVar[] { input, output });
        var outputDirectory = GeneratePyNTTModelDirectory("generated_grid_axis_group_barrier_model", main);

        using (var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json"))))
        {
            var attrs = document.RootElement
                .GetProperty("functions")
                .EnumerateArray()
                .Single()
                .GetProperty("render_kernels")
                .EnumerateArray()
                .Single()
                .GetProperty("metadata")
                .GetProperty("attrs");
            Assert.True(attrs.GetProperty("requires_grid_barrier").GetBoolean());
            Assert.Collection(
                attrs.GetProperty("grid_barrier_axis_groups").EnumerateArray().ToArray(),
                group => Assert.Equal(new[] { 0 }, group.EnumerateArray().Select(axis => axis.GetInt32()).ToArray()),
                group => Assert.Equal(new[] { 1 }, group.EnumerateArray().Select(axis => axis.GetInt32()).ToArray()));
        }

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains(
            "_PYNTT_GRID_MESH_VALUE = tle.device_mesh({\"block\": [('block_y', 4), ('block_x', 8)]})",
            generatedKernelsPy,
            StringComparison.Ordinal);
        Assert.Contains("PYNTT_GRID_MESH = tl.constexpr(_PYNTT_GRID_MESH_VALUE)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("PYNTT_GRID_AXIS_GROUP_0 = tl.constexpr(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("_PYNTT_GRID_MESH_VALUE.axis_group(('block_y',))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("PYNTT_GRID_AXIS_GROUP_1 = tl.constexpr(", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("_PYNTT_GRID_MESH_VALUE.axis_group(('block_x',))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.distributed_barrier(PYNTT_GRID_AXIS_GROUP_0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.distributed_barrier(PYNTT_GRID_AXIS_GROUP_1)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.distributed_barrier(PYNTT_GRID_MESH)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.shard_id", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("PYNTT_GRID_SUBMESH", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTDeduplicatesSameNameNestedPrimFunctionClones()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var input = new Var("x", tensorType);
        var publicOutputBuffer = CreateOutputVar("public_output", tensorType);
        var placement = new Placement(new[] { 1 }, "b", "b");
        var nestedA = CreateNested("nested_prim");
        var nestedB = CreateNested("nested_prim");
        var callerInputBuffer = CreateBuffer("caller_input", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [4], [1]);
        var callerTempBuffer = CreateBuffer("caller_temp", DataTypes.Float32, TIR.MemoryLocation.Data, 16, [4], [1]);
        var callerOutputBuffer = CreateBuffer("caller_output", DataTypes.Float32, TIR.MemoryLocation.Data, 32, [4], [1]);
        var calleeDataBuffer = CreateBuffer("data_0", DataTypes.UInt8, TIR.MemoryLocation.Data, 64, [128], [1]);
        var calleeChipLocalDataBuffer = CreateBuffer("chip_local_data_0", DataTypes.UInt8, TIR.MemoryLocation.ChipLocalData, 0, [0], [1]);
        var calleeBlockLocalDataBuffer = CreateBuffer("block_local_data_0", DataTypes.UInt8, TIR.MemoryLocation.BlockLocalData, 0, [0], [1]);
        var body = new TIR.Sequential(
            nestedA,
            nestedB,
            TIR.F.NTT.TensorLoad(callerInputBuffer, input, new[] { SBP.B }, placement),
            new Call(nestedA, callerInputBuffer, callerTempBuffer, calleeDataBuffer, calleeChipLocalDataBuffer, calleeBlockLocalDataBuffer),
            new Call(nestedB, callerTempBuffer, callerOutputBuffer, calleeDataBuffer, calleeChipLocalDataBuffer, calleeBlockLocalDataBuffer),
            TIR.F.NTT.TensorStore(callerOutputBuffer, publicOutputBuffer, new[] { SBP.B }, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, publicOutputBuffer })
        {
            SchedResult =
            {
                DataUsage = 192,
            },
        };

        var module = new IRModule(main);
        module.Add(nestedA);
        module.Add(nestedB);
        var outputDirectory = GeneratePyNTTModelDirectory("generated_nested_call_deduplicate_model", module);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var kernel = document.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .Single(function => function.GetProperty("name").GetString() == "main_prim")
            .GetProperty("render_kernels")
            .EnumerateArray()
            .Single();
        var deviceFunction = Assert.Single(kernel.GetProperty("device_functions").EnumerateArray());
        Assert.Equal("main_prim_nested_prim_device", deviceFunction.GetProperty("name").GetString());
        var deviceCallCount = Regex.Matches(
            kernel.GetProperty("body_source").GetString() ?? string.Empty,
            "__pyntt_device_call__main_prim_nested_prim_device\\(").Count;
        Assert.Equal(2, deviceCallCount);

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Single(Regex.Matches(generatedKernelsPy, "^def main_prim_nested_prim_device\\(", RegexOptions.Multiline));

        TIR.PrimFunction CreateNested(string name)
        {
            var nestedInputBufferVar = new TIR.BufferVar("nested_input", tensorType, TIR.BufferVarRole.Input, TIR.MemoryLocation.Input);
            var nestedOutputBufferVar = CreateOutputVar("nested_output", tensorType);
            var nestedDataVar = new TIR.BufferVar("data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.Data);
            var nestedChipLocalDataVar = new TIR.BufferVar("chip_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.ChipLocalData);
            var nestedBlockLocalDataVar = new TIR.BufferVar("block_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.BlockLocalData);
            var nestedInputBuffer = TIR.T.AttachBuffer(nestedInputBufferVar, tensorType, TIR.MemoryLocation.Input, 0, out _, "nested_input_buffer");
            var nestedTempBuffer = CreateBuffer("nested_temp", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [4], [1]);
            return new TIR.PrimFunction(
                name,
                PyNTTTarget.Kind,
                new TIR.Sequential(
                    TIR.F.NTT.TensorLoad(nestedTempBuffer, nestedInputBuffer, new[] { SBP.B }, placement),
                    TIR.F.NTT.TensorStore(nestedTempBuffer, nestedOutputBufferVar, new[] { SBP.B }, placement)),
                new IVar[] { nestedInputBufferVar, nestedOutputBufferVar, nestedDataVar, nestedChipLocalDataVar, nestedBlockLocalDataVar })
            {
                SchedResult =
                {
                    DataUsage = 128,
                },
            };
        }
    }

    [Fact]
    public void TestPyNTTNestedObjectOutputAliasCanFeedNextCall()
    {
        var config = new PagedAttentionConfig(
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
            [SBP.SContiguous([0])]);
        var objectType = TensorType.Scalar(new ReferenceType(new PagedAttentionKVCacheType { Config = config }));
        var outputType = new TensorType(DataTypes.Float32, new[] { 4 });
        var placement = new Placement(new[] { 1 }, "b", "b");
        var outputDistributedType = new DistributedType(outputType, new SBP[] { SBP.B }, placement);
        var cache = new Var("cache", objectType);
        var publicOutput = CreateOutputVar("public_output", outputType);
        var nestedInput = new TIR.BufferVar("cache_in", objectType, TIR.BufferVarRole.Input, TIR.MemoryLocation.Input);
        var nestedOutput = CreateOutputVar("cache_out", objectType);
        var nestedData = new TIR.BufferVar("data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.Data);
        var nestedChipLocalData = new TIR.BufferVar("chip_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.ChipLocalData);
        var nestedBlockLocalData = new TIR.BufferVar("block_local_data", TensorType.Scalar(new PointerType(DataTypes.UInt8)), TIR.BufferVarRole.Workspace, TIR.MemoryLocation.BlockLocalData);
        var nested = new TIR.PrimFunction(
            "nested_object_alias_prim",
            PyNTTTarget.Kind,
            new TIR.Sequential(TIR.T.Memcopy(nestedOutput, nestedInput)),
            new IVar[] { nestedInput, nestedOutput, nestedData, nestedChipLocalData, nestedBlockLocalData });

        var cacheInput = TIR.T.AttachBuffer(cache, objectType, TIR.MemoryLocation.Input, 0, out _, "cache_input");
        var cacheAfterFirstCall = CreateBuffer("cache_after_first_call", objectType.DType, TIR.MemoryLocation.Data, 0, [], []);
        var cacheAfterSecondCall = CreateBuffer("cache_after_second_call", objectType.DType, TIR.MemoryLocation.Data, 2048, [], []);
        var positionIds = CreateBuffer("position_ids", DataTypes.Float32, TIR.MemoryLocation.Data, 4096, [4], [1], outputDistributedType);
        var calleeData0 = CreateBuffer("data_0", DataTypes.UInt8, TIR.MemoryLocation.Data, 6144, [0], [1]);
        var calleeChipLocalData0 = CreateBuffer("chip_local_data_0", DataTypes.UInt8, TIR.MemoryLocation.ChipLocalData, 0, [0], [1]);
        var calleeBlockLocalData0 = CreateBuffer("block_local_data_0", DataTypes.UInt8, TIR.MemoryLocation.BlockLocalData, 0, [0], [1]);
        var calleeData1 = CreateBuffer("data_1", DataTypes.UInt8, TIR.MemoryLocation.Data, 6144, [0], [1]);
        var calleeChipLocalData1 = CreateBuffer("chip_local_data_1", DataTypes.UInt8, TIR.MemoryLocation.ChipLocalData, 0, [0], [1]);
        var calleeBlockLocalData1 = CreateBuffer("block_local_data_1", DataTypes.UInt8, TIR.MemoryLocation.BlockLocalData, 0, [0], [1]);
        var body = new TIR.Sequential(
            nested,
            new Call(nested, cacheInput, cacheAfterFirstCall, calleeData0, calleeChipLocalData0, calleeBlockLocalData0),
            new Call(nested, cacheAfterFirstCall, cacheAfterSecondCall, calleeData1, calleeChipLocalData1, calleeBlockLocalData1),
            TIR.F.NTT.GetPositionIds(cacheAfterSecondCall, positionIds, outputDistributedType),
            TIR.F.NTT.TensorStore(positionIds, publicOutput, new[] { SBP.B }, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { cache, publicOutput })
        {
            SchedResult =
            {
                DataUsage = 8192,
            },
        };

        var module = new IRModule(main);
        module.Add(nested);
        var outputDirectory = GeneratePyNTTModelDirectory("generated_nested_object_alias_call_model", module);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Single(Regex.Matches(generatedKernelsPy, "def main_prim_nested_object_alias_prim_device"));

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json")));
        var kernel = document.RootElement
            .GetProperty("functions")
            .EnumerateArray()
            .Single(function => function.GetProperty("name").GetString() == "main_prim")
            .GetProperty("render_kernels")
            .EnumerateArray()
            .Single();
        var metadata = kernel.GetProperty("metadata");
        var attrs = metadata.GetProperty("attrs");
        var fieldInputs = attrs
            .GetProperty("kv_cache_field_inputs")
            .EnumerateArray()
            .ToArray();
        Assert.All(fieldInputs, fieldInput => Assert.Equal("cache", fieldInput.GetProperty("SourceName").GetString()));
        Assert.Equal(
            new[] { "num_seqs", "query_start_loc", "seq_lens" },
            fieldInputs
                .Select(fieldInput => fieldInput.GetProperty("Field").GetString())
                .OrderBy(field => field, StringComparer.Ordinal)
                .ToArray());
        var scalarInputName = fieldInputs
            .Single(fieldInput => fieldInput.GetProperty("Field").GetString() == "num_seqs")
            .GetProperty("Name")
            .GetString();
        var scalarInputIndex = metadata
            .GetProperty("inputs")
            .EnumerateArray()
            .Select((input, index) => (Name: input.GetString(), Index: index))
            .Single(input => input.Name == scalarInputName)
            .Index;
        Assert.Equal(
            new[] { $"input{scalarInputIndex.ToString(CultureInfo.InvariantCulture)}" },
            attrs.GetProperty("runtime_scalar_input_args")
                .EnumerateArray()
                .Select(argument => argument.GetString())
                .ToArray());
    }

    [Fact]
    public void TestPyNTTPackedMatmulUsesTwoDimensionalNLanes()
    {
        var lhsBuffer = CreateDataBuffer("lhs", DataTypes.Float32, 0, [1, 64], [64, 1]);
        var packedElemType = new VectorType(DataTypes.Float32, 4, 8);
        var rhsBuffer = CreateDataBuffer("rhs", packedElemType, 256, [4, 64], [64, 1]);
        var packedOutputBuffer = CreateDataBuffer("packed_output", packedElemType, 33024, [1, 4], [4, 1]);
        var vectorOutputBuffer = CreateDataBuffer("vector_output", new VectorType(DataTypes.Float32, 8), 33536, [1, 16], [16, 1]);
        var outputBuffer = CreateDataBuffer("output", DataTypes.Float32, 34048, [1, 128], [128, 1]);
        var output = CreateOutputVar("output", new TensorType(DataTypes.Float32, new[] { 1, 128 }));
        var placement = new Placement(new[] { 1 }, "b", "b");
        var body = new TIR.Sequential(
            TIR.F.NTT.PackedMatMul(lhsBuffer, rhsBuffer, packedOutputBuffer, None.Default, 1.0f),
            TIR.F.NTT.Unpack(packedOutputBuffer, vectorOutputBuffer, new[] { 4 }, new[] { 1 }),
            TIR.F.NTT.Unpack(vectorOutputBuffer, outputBuffer, new[] { 8 }, new[] { 1 }),
            TIR.F.NTT.TensorStore(outputBuffer, output, new[] { SBP.B, SBP.B }, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { output })
        {
            SchedResult =
            {
                DataUsage = 65536,
            },
        };
        var outputDirectory = GeneratePyNTTModelDirectory("generated_packed_matmul_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT algorithm triton.matmul/simt_fma", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("rhs_n_packed_lane=4, rhs_n_lane=8", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("output_n_packed_lane=4, output_n_lane=8", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(((offs_n[:, None]) // 8) % 4)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("((offs_n[:, None]) % 8)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(((offs_n) // 8) % 4)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("((offs_n) % 8)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("((offs_n[:, None]) // 32) * 8 + ((offs_n[:, None]) % 32)", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTPackedMatmulRunUsesPackedNLanes()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 1, 64 }));
        var output = CreateOutputVar("output", new TensorType(DataTypes.Float32, new[] { 1, 128 }));
        var packedElemType = new VectorType(DataTypes.Float32, 4, 8);
        var rhs = CreatePackedMatmulRhsConst();
        var rhsSizeBytes = checked((ulong)rhs.Value.Length * (ulong)rhs.Value.ElementType.SizeInBytes);
        var lhsBuffer = CreateBuffer("lhs_buffer", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [1, 64], [64, 1]);
        var rhsBuffer = CreateBuffer("rhs_buffer", packedElemType, TIR.MemoryLocation.Rdata, 0, [4, 64], [64, 1]);
        var packedOutputBuffer = CreateBuffer("packed_output", packedElemType, TIR.MemoryLocation.Data, 256, [1, 4], [4, 1]);
        var vectorOutputBuffer = CreateBuffer("vector_output", new VectorType(DataTypes.Float32, 8), TIR.MemoryLocation.Data, 768, [1, 16], [16, 1]);
        var outputBuffer = CreateBuffer("output_buffer", DataTypes.Float32, TIR.MemoryLocation.Data, 1280, [1, 128], [128, 1]);
        var placement = new Placement(new[] { 1 }, "b", "b");
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(lhsBuffer, lhs, new[] { SBP.B, SBP.B }, placement),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.PackedMatMul(lhsBuffer, rhsBuffer, packedOutputBuffer, None.Default, 1.0f),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Unpack(packedOutputBuffer, vectorOutputBuffer, new[] { 4 }, new[] { 1 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Unpack(vectorOutputBuffer, outputBuffer, new[] { 8 }, new[] { 1 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.TensorStore(outputBuffer, output, new[] { SBP.B, SBP.B }, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { lhs, output })
        {
            SchedResult =
            {
                DataUsage = 2048,
            },
        };
        main.SchedResult.Rdatas.Add(rhs, (0, rhsSizeBytes));

        var outputDirectory = GeneratePyNTTModelDirectory("generated_packed_matmul_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("rhs_n_packed_lane=4, rhs_n_lane=8", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedBlockBarrierChain(
            generatedKernelsPy,
            "main_prim_composite_0",
            "main_prim__tensor_load__0",
            "main_prim__gemv_compute__0",
            "main_prim__unpack_compute__0",
            "main_prim__unpack_compute__1",
            "main_prim__output_tensor_store__0");
        AssertGeneratedModelRuns(
            outputDirectory,
            "lhs = (torch.arange(64, dtype=torch.float32, device='cuda').reshape(1, 64) - 16) * 0.01",
            "rhs = (torch.arange(64 * 128, dtype=torch.float32, device='cuda').reshape(64, 128) - 128) * 0.001",
            "output = module(lhs)",
            "torch.testing.assert_close(output, lhs @ rhs, rtol=1e-4, atol=1e-4)");
    }

    [Fact]
    public void TestPyNTTPackedMatmulRunFusesExactAddend()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 1, 64 }));
        var output = CreateOutputVar("output", new TensorType(DataTypes.Float32, new[] { 1, 128 }));
        var packedElemType = new VectorType(DataTypes.Float32, 4, 8);
        var rhs = CreatePackedMatmulRhsConst();
        var rhsSizeBytes = checked((ulong)rhs.Value.Length * (ulong)rhs.Value.ElementType.SizeInBytes);
        var addendValues = Enumerable.Range(0, 128)
            .Select(index => ((float)index - 31f) * 0.002f)
            .ToArray();
        var addendTensor = new TensorConst(Tensor.From<float>(addendValues, [1, 128]));
        var vectorAddend = IR.F.Tensors.Pack(addendTensor, [8], [1]);
        var packedAddend = new TensorConst(
            IR.F.Tensors.Pack(vectorAddend, [4], [1]).Evaluate().AsTensor());
        var addendSizeBytes = checked((ulong)packedAddend.Value.BytesBuffer.Length);
        var lhsBuffer = CreateBuffer("lhs_buffer", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [1, 64], [64, 1]);
        var rhsBuffer = CreateBuffer("rhs_buffer", packedElemType, TIR.MemoryLocation.Rdata, 0, [4, 64], [64, 1]);
        var addendBuffer = CreateBuffer(
            "addend_buffer",
            packedElemType,
            TIR.MemoryLocation.Rdata,
            checked((long)rhsSizeBytes),
            [1, 4],
            [4, 1]);
        var packedOutputBuffer = CreateBuffer("packed_output", packedElemType, TIR.MemoryLocation.Data, 256, [1, 4], [4, 1]);
        var vectorOutputBuffer = CreateBuffer("vector_output", new VectorType(DataTypes.Float32, 8), TIR.MemoryLocation.Data, 768, [1, 16], [16, 1]);
        var outputBuffer = CreateBuffer("output_buffer", DataTypes.Float32, TIR.MemoryLocation.Data, 1280, [1, 128], [128, 1]);
        var placement = new Placement(new[] { 1 }, "b", "b");
        var packedMatMulCall = TIR.F.NTT.PackedMatMul(
            lhsBuffer,
            rhsBuffer,
            packedOutputBuffer,
            None.Default,
            1.0f,
            addend: addendBuffer);
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        packedMatMulCall.Metadata.TIRMicroKernel = Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(
            targetOptions.TIRMicroKernelSelector.Select(
                new(
                    Assert.IsType<TIR.NTT.PackedMatMul>(packedMatMulCall.Target),
                    packedMatMulCall.Arguments[..^1].ToArray(),
                    targetOptions.TargetMachineModel)));
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(lhsBuffer, lhs, new[] { SBP.B, SBP.B }, placement),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            packedMatMulCall,
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Unpack(packedOutputBuffer, vectorOutputBuffer, new[] { 4 }, new[] { 1 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Unpack(vectorOutputBuffer, outputBuffer, new[] { 8 }, new[] { 1 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.TensorStore(outputBuffer, output, new[] { SBP.B, SBP.B }, placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { lhs, output })
        {
            SchedResult =
            {
                DataUsage = 2048,
            },
        };
        main.SchedResult.Rdatas.Add(rhs, (0, rhsSizeBytes));
        main.SchedResult.Rdatas.Add(
            packedAddend,
            (rhsSizeBytes, checked(rhsSizeBytes + addendSizeBytes)));

        var outputDirectory = GeneratePyNTTModelDirectory(
            "generated_packed_matmul_addend_run_model",
            main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("has_addend=True", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("addend_value = tl.reshape(tl.load", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "lhs = (torch.arange(64, dtype=torch.float32, device='cuda').reshape(1, 64) - 16) * 0.01",
            "rhs = (torch.arange(64 * 128, dtype=torch.float32, device='cuda').reshape(64, 128) - 128) * 0.001",
            "addend = (torch.arange(128, dtype=torch.float32, device='cuda').reshape(1, 128) - 31) * 0.002",
            "output = module(lhs)",
            "torch.testing.assert_close(output, lhs @ rhs + addend, rtol=1e-4, atol=1e-4)");
    }

    [Fact]
    public void TestPyNTTPackedBFloat16MatmulYxBlockLocalRDataRun()
    {
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.HierarchyLevels = "bb";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };

        const int m = 20;
        const int k = 1024;
        const int n = 4096;
        const int nPackedLane = 4;
        const int nLane = 8;
        const int y = 4;
        const int x = 8;
        const int localM = m / y;
        const int packedN = n / (nPackedLane * nLane);
        const int localPackedN = packedN / x;
        const int localVectorN = n / nLane / x;
        const int localN = n / x;

        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new[] { m, k }));
        var output = CreateOutputVar("output", new TensorType(DataTypes.BFloat16, new[] { m, n }));
        var placement = new Placement(new[] { y, x }, "yx", "bb");
        var lhsDistributedType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new[] { m, k }),
            new SBP[] { SBP.SContiguous([0], localM), SBP.B },
            placement);
        var rhsDistributedType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, nPackedLane, nLane), new[] { packedN, k }),
            new SBP[] { SBP.SContiguous([1], localPackedN), SBP.B },
            placement);
        var packedOutputDistributedType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, nPackedLane, nLane), new[] { m, packedN }),
            new SBP[] { SBP.SContiguous([0], localM), SBP.SContiguous([1], localPackedN) },
            placement);
        var vectorOutputDistributedType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, nLane), new[] { m, n / nLane }),
            new SBP[] { SBP.SContiguous([0], localM), SBP.SContiguous([1], localVectorN) },
            placement);
        var outputDistributedType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new[] { m, n }),
            new SBP[] { SBP.SContiguous([0], localM), SBP.SContiguous([1], localN) },
            placement);
        var packedElemType = new VectorType(DataTypes.BFloat16, nPackedLane, nLane);
        var vectorElemType = new VectorType(DataTypes.BFloat16, nLane);
        var rhs = CreatePackedBFloat16MatmulRhsConst(k, n, rhsDistributedType);
        var lhsBytes = checked(localM * k * DataTypes.BFloat16.SizeInBytes);
        var packedOutputBytes = checked(localM * localPackedN * packedElemType.SizeInBytes);
        var vectorOutputBytes = checked(localM * localVectorN * vectorElemType.SizeInBytes);
        var outputBytes = checked(localM * localN * DataTypes.BFloat16.SizeInBytes);
        var lhsBuffer = CreateBuffer("lhs_buffer", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [localM, k], [k, 1], lhsDistributedType);
        var rhsBuffer = CreateBuffer("rhs_buffer", packedElemType, TIR.MemoryLocation.BlockLocalRdata, 0, [localPackedN, k], [k, 1], rhsDistributedType);
        var packedOutputBuffer = CreateBuffer("packed_output", packedElemType, TIR.MemoryLocation.Data, lhsBytes, [localM, localPackedN], [localPackedN, 1], packedOutputDistributedType);
        var vectorOutputBuffer = CreateBuffer("vector_output", vectorElemType, TIR.MemoryLocation.Data, lhsBytes + packedOutputBytes, [localM, localVectorN], [localVectorN, 1], vectorOutputDistributedType);
        var outputBuffer = CreateBuffer("output_buffer", DataTypes.BFloat16, TIR.MemoryLocation.Data, lhsBytes + packedOutputBytes + vectorOutputBytes, [localM, localN], [localN, 1], outputDistributedType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(lhsBuffer, lhs, lhsDistributedType.AxisPolicies.ToArray(), placement),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.PackedMatMul(lhsBuffer, rhsBuffer, packedOutputBuffer, None.Default, 1.0f),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Unpack(packedOutputBuffer, vectorOutputBuffer, new[] { nPackedLane }, new[] { 1 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.Unpack(vectorOutputBuffer, outputBuffer, new[] { nLane }, new[] { 1 }),
            TIR.F.NTT.Barrier(TIR.NTT.BarrierScope.Block),
            TIR.F.NTT.TensorStore(outputBuffer, output, outputDistributedType.AxisPolicies.ToArray(), placement));
        var main = new TIR.PrimFunction(
            "main_prim",
            PyNTTTarget.Kind,
            body,
            new TIR.Return(new Expr[] { output }),
            new IVar[] { lhs, output })
        {
            SchedResult =
            {
                DataUsage = checked((ulong)(lhsBytes + packedOutputBytes + vectorOutputBytes + outputBytes)),
            },
        };
        var rhsLocalBytes = checked((ulong)(localPackedN * k * packedElemType.SizeInBytes));
        main.SchedResult.BlockLocalRdatas.Add(rhs, (0, rhsLocalBytes));

        var outputDirectory = GeneratePyNTTModelDirectory("generated_packed_bf16_yx_block_local_rdata_matmul_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("rhs_n_packed_lane=4, rhs_n_lane=8", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedBlockBarrierChain(
            generatedKernelsPy,
            "main_prim_composite_0",
            "main_prim__tensor_load__0",
            "main_prim__matmul_compute__0",
            "main_prim__unpack_compute__0",
            "main_prim__unpack_compute__1",
            "main_prim__output_tensor_store__0");
        AssertGeneratedModelRuns(
            outputDirectory,
            $"lhs = ((torch.arange({m} * {k}, dtype=torch.float32, device='cuda').reshape({m}, {k}) - 257) * 0.0005).to(torch.bfloat16)",
            $"rhs = ((torch.arange({k} * {n}, dtype=torch.float32, device='cuda').reshape({k}, {n}) - 521) * 0.0002).to(torch.bfloat16)",
            "output = module(lhs)",
            "torch.testing.assert_close(output.to(torch.float32), (lhs @ rhs).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedReduceRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 4, 3 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Tensors.Reduce(ReduceOp.Sum, x, new[] { 1L }, 0.0f, false), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_reduce_run_model", main);
        AssertGeneratedKernel(outputDirectory, "reduce", "Reduce.py.jinja");
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(12, dtype=torch.float32, device='cuda').reshape(4, 3) * 0.25",
            "output = module(x)",
            "torch.testing.assert_close(output, x.sum(dim=1), rtol=1e-5, atol=1e-5)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedReduceAxisZeroRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 3, 4 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Tensors.Reduce(ReduceOp.Sum, x, new[] { 0L }, 0.0f, false), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_reduce_axis_zero_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(12, dtype=torch.float32, device='cuda').reshape(3, 4) * 0.25",
            "output = module(x)",
            "torch.testing.assert_close(output, x.sum(dim=0), rtol=1e-5, atol=1e-5)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedReduceMeanTracksElementCountRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 4, 513 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Tensors.Reduce(ReduceOp.Mean, x, new[] { 1L }, 0.0f, false), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_reduce_mean_count_run_model", main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var reductionLoops = compiler.Module.Functions
            .SelectMany(function => ExprCollector.Collect(function).OfType<TIR.For>())
            .Where(loop => loop.Mode == TIR.LoopMode.Reduction)
            .ToArray();
        Assert.Contains(reductionLoops, loop => loop.Partition == TIR.LoopPartition.Full);
        Assert.Contains(reductionLoops, loop => loop.Partition == TIR.LoopPartition.Tail);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("reduced_element_count +=", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("acc / reduced_element_count.to(tl.float32)", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = (torch.arange(4 * 513, dtype=torch.float32, device='cuda').reshape(4, 513) - 719) * 0.001",
            "output = module(x)",
            "torch.testing.assert_close(output, x.mean(dim=1), rtol=1e-5, atol=1e-5)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedSoftmaxRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 4, 5 }));
        var main = new Function("main", PyNTTTarget.Kind, IR.F.NN.Softmax(x, 1), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_softmax_run_model", main);
        AssertGeneratedKernel(outputDirectory, "softmax", "Softmax.py.jinja");
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(20, dtype=torch.float32, device='cuda').reshape(4, 5) * 0.125",
            "output = module(x)",
            "torch.testing.assert_close(output, torch.softmax(x, dim=1), rtol=1e-5, atol=1e-5)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedRmsNormRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 4, 8 }));
        var scale = Tensor.From<float>(Enumerable.Range(0, 8).Select(i => 1.0f + (i * 0.01f)).ToArray(), [8]);
        var bias = Tensor.Zeros<float>([8]);
        var main = new Function("main", PyNTTTarget.Kind, IR.F.NN.LayerNorm(1, 1e-5f, x, scale, bias, hasMean: false), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_rms_norm_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja NormStats.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja NormApply.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = (torch.arange(32, dtype=torch.float32, device='cuda').reshape(4, 8) - 7) * 0.125",
            "scale = (1.0 + torch.arange(8, dtype=torch.float32, device='cuda') * 0.01)",
            "expect = x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-5) * scale",
            "output = module(x)",
            "torch.testing.assert_close(output, expect, rtol=1e-5, atol=1e-5)");
    }

    [Fact]
    public async Task TestPyNTTBitcastMaterializesCallerAllocatedGlobalOutput()
    {
        ConfigureAutoDistributedPyNTT();
        var inputType = new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 4, 8 });
        var input = new Var("input", inputType);
        var bitcast = IR.F.Tensors.Bitcast(input, DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, bitcast, [input]);

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_buffer_alias_bitcast_model", main);
        using var metadata = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var function = metadata.RootElement.GetProperty("functions").EnumerateArray().Single();
        var output = function.GetProperty("outputs").EnumerateArray().Single();
        Assert.Equal("global", output.GetProperty("memory").GetString());
        Assert.Equal("bfloat16", output.GetProperty("dtype").GetString());
        Assert.Equal(new[] { 4L, 64L }, output.GetProperty("shape").EnumerateArray().Select(value => value.GetInt64()).ToArray());
        Assert.Single(function.GetProperty("generated_kernels").EnumerateArray());
        var result = function.GetProperty("results").EnumerateArray().Single();
        Assert.Equal("output", result.GetProperty("source").GetString());
        Assert.Equal(0, result.GetProperty("source_index").GetInt32());

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja TensorRegionCopy.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja TensorStore.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("generated from PyNTT Jinja Bitcast.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("generated from PyNTT Jinja Reshape.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestPyNTTSameRankReshapeUsesLogicalResultShape()
    {
        ConfigureAutoDistributedPyNTT();
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 4 }));
        var reshape = IR.F.Tensors.Reshape(input, new Dimension[] { 2, 8 });
        var main = new Function("main", PyNTTTarget.Kind, reshape, [input]);

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_buffer_alias_reshape_model", main);
        using var metadata = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var function = metadata.RootElement.GetProperty("functions").EnumerateArray().Single();
        var output = function.GetProperty("outputs").EnumerateArray().Single();
        Assert.Equal("global", output.GetProperty("memory").GetString());
        Assert.Equal(new[] { 2L, 8L }, output.GetProperty("shape").EnumerateArray().Select(value => value.GetInt64()).ToArray());
        Assert.Single(function.GetProperty("generated_kernels").EnumerateArray());
        var result = function.GetProperty("results").EnumerateArray().Single();
        Assert.Equal("output", result.GetProperty("source").GetString());
        Assert.Equal(0, result.GetProperty("source_index").GetInt32());

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja TensorRegionCopy.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja TensorStore.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("generated from PyNTT Jinja Reshape.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)",
            "output = module(x)",
            "assert output.shape == (2, 8)",
            "torch.testing.assert_close(output, x.reshape(2, 8), rtol=0, atol=0)");
    }

    [Fact]
    public async Task TestPyNTTDynamicVectorizedBinaryCodegen()
    {
        ConfigureAutoDistributedPyNTT();
        var sequenceLength = new DimVar("sequence_length")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };
        var inputType = new TensorType(DataTypes.BFloat16, new Dimension[] { sequenceLength, 1024 });
        var lhs = new Var("lhs", inputType);
        var rhs = new Var("rhs", inputType);
        var vectorType = new VectorType(DataTypes.BFloat16, [8]);
        var vectorizedLhs = IR.F.Tensors.Bitcast(lhs, vectorType);
        var vectorizedRhs = IR.F.Tensors.Bitcast(rhs, vectorType);
        var binary = IR.F.NTT.VectorizedBinary(vectorizedLhs, vectorizedRhs, None.Default, BinaryOp.Add);
        var output = IR.F.Tensors.Bitcast(binary, DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, output, new IVar[] { lhs, rhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_dynamic_vectorized_binary_model", main);
        var compiler = Assert.IsType<global::Nncase.Compiler.Compiler>(CompileSession.Compiler);
        var cacheBuffers = compiler.Module.Functions
            .SelectMany(function => ExprCollector.Collect(function).OfType<TIR.Buffer>())
            .Where(buffer => buffer.MemSpan.Buffer.Location == TIR.MemoryLocation.Cache)
            .ToArray();
        Assert.Empty(cacheBuffers);

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja ElementwiseBinary.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
    }

    private void ConfigureAutoDistributedPyNTT()
    {
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.EGraphCost | DumpFlags.CodeGen | DumpFlags.Compile;
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        targetOptions.HierarchyNames = "yx";
        targetOptions.Hierarchies = new[] { new[] { 4, 8 } };
    }

    private Nncase.Schedule.TIRMicroKernelSelection SelectPagedAttentionPartialMicroKernel(
        PrimType cacheDataType,
        TargetMachineModel? machine = null,
        int queryHeads = 16,
        int pageSize = 32)
    {
        const int headDim = 128;
        const int numKvHeads = 8;
        Assert.True(queryHeads > 0 && queryHeads % numKvHeads == 0);
        var localQueryHeads = queryHeads / numKvHeads;
        var cacheLayout = new[]
        {
            PagedKVCacheDimKind.NumBlocks,
            PagedKVCacheDimKind.KV,
            PagedKVCacheDimKind.NumLayers,
            PagedKVCacheDimKind.BlockSize,
            PagedKVCacheDimKind.NumKVHeads,
            PagedKVCacheDimKind.HeadDim,
        };
        var config = new PagedAttentionConfig(
            1,
            numKvHeads,
            headDim,
            cacheDataType,
            pageSize,
            cacheLayout,
            cacheLayout,
            [PagedKVCacheDimKind.HeadDim],
            [PagedKVCacheDimKind.HeadDim],
            [8],
            [8],
            [],
            []);
        var query = CreateBuffer(
            "query",
            new VectorType(DataTypes.BFloat16, [8]),
            TIR.MemoryLocation.Data,
            0,
            [1, localQueryHeads, headDim / 8],
            [localQueryHeads * headDim / 8, headDim / 8, 1]);
        var cache = new Var(
            "cache",
            TensorType.Scalar(
                new ReferenceType(
                    new PagedAttentionKVCacheType { Config = config })));
        var op = new TIR.NTT.PagedAttentionPartial(
            [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim],
            queryHeads * headDim,
            0,
            4,
            0);
        var targetOptions = Assert.IsType<PyNTTTargetOptions>(CompileOptions.TargetOptions);
        return Assert.IsType<Nncase.Schedule.TIRMicroKernelSelection>(
            targetOptions.TIRMicroKernelSelector.Select(
                new(op, [query, cache], machine ?? targetOptions.TargetMachineModel)));
    }

    private async Task<string> GeneratePyNTTModelDirectoryWithCompilerPipeline(string directoryName, BaseFunction function)
    {
        var outputDirectory = Path.Join(CompileOptions.DumpDir, directoryName);
        if (Directory.Exists(outputDirectory))
        {
            Directory.Delete(outputDirectory, recursive: true);
        }

        ((PyNTTTargetOptions)CompileOptions.TargetOptions).OutputDirectory = outputDirectory;

        CompileSession.Compiler.ImportIRModule(new IRModule(function));
        await CompileSession.Compiler.CompileAsync();
        using var stream = new MemoryStream();
        CompileSession.Compiler.Gencode(stream);
        Assert.NotEqual(0, stream.Length);
        return outputDirectory;
    }

    private void AssertTIRPipelineDump()
    {
        var tirDumpFiles = Directory.GetFiles(Dumpper.Directory, "*.script", SearchOption.AllDirectories)
            .Where(path => path.Contains("TIRPass", StringComparison.Ordinal))
            .ToArray();
        Assert.NotEmpty(tirDumpFiles);

        var tirDump = string.Join(Environment.NewLine, tirDumpFiles.Select(File.ReadAllText));
        Assert.Contains("TensorLoad", tirDump, StringComparison.Ordinal);
        Assert.Contains("VectorizedBinary", tirDump, StringComparison.Ordinal);
        Assert.Contains("TensorStore", tirDump, StringComparison.Ordinal);
    }

    private void AssertGeneratedKernel(string outputDirectory, string opKind, string templateFileName)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var kernel = document.RootElement.GetProperty("functions").EnumerateArray().Single()
            .GetProperty("generated_kernels").EnumerateArray().Single();
        Assert.Equal(opKind, kernel.GetProperty("op_kind").GetString());
        Assert.True(kernel.GetProperty("attrs").GetProperty("tir").GetBoolean());

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        var marker = templateFileName.Contains('/', StringComparison.Ordinal)
            ? $"generated from PyNTT algorithm triton.{templateFileName[..^".py.jinja".Length]}"
            : $"generated from PyNTT Jinja {templateFileName}";
        Assert.Contains(marker, generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("from pyntt.backends.triton.kernels", generatedKernelsPy, StringComparison.Ordinal);
    }

    private void RenderGeneratedKernels(string outputDirectory)
    {
        var packageRoot = Path.Join(SolutionDirectory, "pyntt");
        var script = string.Join(
            "; ",
            "import sys",
            $"sys.path.insert(0, {PythonString(packageRoot)})",
            "from pyntt.codegen.render import render_generated_kernels",
            $"render_generated_kernels({PythonString(outputDirectory)})");
        var (exitCode, stdout, stderr) = RunPythonScript(script);
        Assert.True(
            exitCode == 0,
            $"PyNTT Jinja kernel rendering failed.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
    }

    private string GeneratePyNTTModelDirectory(string directoryName, BaseFunction function)
    {
        var module = new IRModule(function);
        return GeneratePyNTTModelDirectory(directoryName, module);
    }

    private string GeneratePyNTTModelDirectory(string directoryName, IRModule module)
    {
        var outputDirectory = Path.Join(CompileOptions.DumpDir, directoryName);
        if (Directory.Exists(outputDirectory))
        {
            Directory.Delete(outputDirectory, recursive: true);
        }

        ((PyNTTTargetOptions)CompileOptions.TargetOptions).OutputDirectory = outputDirectory;

        var linkedModel = new ModelBuilder(CompileSession.Target, CompileOptions).Build(module);

        using var stream = new MemoryStream();
        linkedModel.Serialize(stream);
        Assert.NotEqual(0, stream.Length);
        return outputDirectory;
    }

    private TIR.Buffer CreateDataBuffer(string name, DataType elemType, long startBytes, long[] dimensions, long[] strides)
        => CreateBuffer(name, elemType, TIR.MemoryLocation.Data, startBytes, dimensions, strides);

    private TIR.BufferVar CreateOutputVar(string name, IRType type)
        => new(name, type, TIR.BufferVarRole.Output, TIR.MemoryLocation.Output);

    private TIR.Buffer CreateBuffer(string name, DataType elemType, TIR.MemoryLocation location, long startBytes, long[] dimensions, long[] strides, DistributedType? distributedType = null)
    {
        var physicalElementCount = dimensions.Aggregate(1L, (acc, dim) => checked(acc * dim));
        var sizeBytes = checked(physicalElementCount * elemType.SizeInBytes);
        return new TIR.Buffer(
            name,
            elemType,
            new TIR.MemSpan(new TIR.PhysicalBuffer(elemType.SizeInBytes, startBytes, sizeBytes, location)),
            dimensions.Select(dim => (Dimension)dim).ToArray(),
            strides.Select(stride => (Dimension)stride).ToArray(),
            distributedType);
    }

    private TIR.Buffer CreateCompactPerOwnerBuffer(
        string name,
        DataType elemType,
        long startBytes,
        long[] dimensions,
        long[] strides,
        DistributedType distributedType)
    {
        var componentElementCount = dimensions.Aggregate(1L, (acc, dim) => checked(acc * dim));
        var componentSizeBytes = checked(componentElementCount * elemType.SizeInBytes);
        var ownerCount = distributedType.Placement.Hierarchy.Aggregate(
            1L,
            (product, extent) => checked(product * extent));
        var physicalSizeBytes = checked(componentSizeBytes * ownerCount);
        var physical = new TIR.PhysicalBuffer(
            elemType.SizeInBytes,
            startBytes,
            physicalSizeBytes,
            TIR.MemoryLocation.ChipLocalData);
        return new TIR.Buffer(
            name,
            elemType,
            new TIR.MemSpan(physical, 0, componentSizeBytes),
            dimensions.Select(dim => (Dimension)dim).ToArray(),
            strides.Select(stride => (Dimension)stride).ToArray(),
            distributedType,
            distributedStorageKind: TIR.DistributedBufferStorageKind.CompactPerOwner);
    }

    private TensorConst CreatePackedMatmulRhsConst()
    {
        var rhsValues = Enumerable.Range(0, 64 * 128)
            .Select(i => ((float)i - 128f) * 0.001f)
            .ToArray();
        var rhs = new TensorConst(Tensor.From<float>(rhsValues, [64, 128]));
        var vectorized = IR.F.Tensors.Pack(rhs, [8], [1]);
        var transposed = IR.F.Tensors.Transpose(vectorized, new[] { 1, 0 });
        var packed = IR.F.Tensors.Pack(transposed, [4], [0]).Evaluate().AsTensor();
        Assert.Equal(new VectorType(DataTypes.Float32, 4, 8), packed.ElementType);
        Assert.Equal(new[] { 4L, 64L }, packed.Dimensions.ToArray());
        return new TensorConst(packed);
    }

    private TensorConst CreatePackedBFloat16MatmulRhsConst(int k, int n, DistributedType distributedType)
    {
        var rhsValues = Enumerable.Range(0, k * n)
            .Select(i => (BFloat16)(((float)i - 521f) * 0.0002f))
            .ToArray();
        var rhs = new TensorConst(Tensor.From<BFloat16>(rhsValues, [k, n]));
        var vectorized = IR.F.Tensors.Pack(rhs, [8], [1]);
        var transposed = IR.F.Tensors.Transpose(vectorized, new[] { 1, 0 });
        var packed = IR.F.Tensors.Pack(transposed, [4], [0]).Evaluate().AsTensor();
        Assert.Equal(distributedType.TensorType.DType, packed.ElementType);
        Assert.Equal(distributedType.TensorType.Shape.ToValueArray(), packed.Dimensions.ToArray());
        return new TensorConst(packed, distributedType.AxisPolicies, distributedType.Placement);
    }

    private void AssertGeneratedModelImports(string outputDirectory)
    {
        var packageRoot = Path.Join(SolutionDirectory, "pyntt");
        var modelParent = Path.GetDirectoryName(outputDirectory)!;
        var modelPackage = Path.GetFileName(outputDirectory);
        var script = string.Join(
            "; ",
            "import sys",
            $"sys.path.insert(0, {PythonString(packageRoot)})",
            $"sys.path.insert(0, {PythonString(modelParent)})",
            $"import {modelPackage}",
            $"module = {modelPackage}.load_model()",
            "assert module.spec.backend == 'triton'",
            "assert module.spec.entry is not None",
            "assert module.spec.entry.name == 'main'",
            "assert len(module.spec.entry.inputs) > 0",
            "assert len(module.spec.entry.outputs) > 0");

        var (exitCode, stdout, stderr) = RunPythonScript(script);
        Assert.True(
            exitCode == 0,
            $"Generated PyNTT model import failed.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
    }

    private void AssertGeneratedModelRunsBinaryAdd(string outputDirectory)
    {
        AssertGeneratedModelRuns(
            outputDirectory,
            "os.environ['PYNTT_TUNE_MAIN_PRIM_BINARY_0_BLOCK_SIZE'] = '128'",
            "lhs = torch.arange(32, dtype=torch.float32, device='cuda').reshape(32, 1)",
            "rhs = torch.arange(32, dtype=torch.float32, device='cuda').reshape(32, 1) * 0.25",
            "output = module(lhs, rhs)",
            "torch.testing.assert_close(output, lhs + rhs, rtol=0, atol=1e-6)",
            "assert output.is_cuda");
    }

    private void AssertGeneratedModelRuns(string outputDirectory, params string[] bodyLines)
    {
        var packageRoot = Path.Join(SolutionDirectory, "pyntt");
        var modelParent = Path.GetDirectoryName(outputDirectory)!;
        var modelPackage = Path.GetFileName(outputDirectory);
        var scriptLines = new List<string>
        {
            "import os",
            "import sys",
            "try:",
            "    import torch",
            "    import triton",
            "except ImportError as ex:",
            "    print(f'missing runtime dependency: {ex}', file=sys.stderr)",
            "    raise SystemExit(77)",
            "if not torch.cuda.is_available():",
            "    print('CUDA is not available', file=sys.stderr)",
            "    raise SystemExit(77)",
            $"sys.path.insert(0, {PythonString(packageRoot)})",
            $"sys.path.insert(0, {PythonString(modelParent)})",
            $"import {modelPackage}",
            $"module = {modelPackage}.load_model()",
        };
        scriptLines.AddRange(bodyLines);
        scriptLines.AddRange(new[]
        {
            "torch.cuda.synchronize()",
            "print('pyntt end-to-end output ok')",
        });
        var script = string.Join(Environment.NewLine, scriptLines);

        var (exitCode, stdout, stderr) = RunPythonScript(script);
        if (exitCode == 77)
        {
            Assert.Skip($"PyNTT end-to-end runtime test requires torch, triton, and CUDA.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        }

        Assert.True(
            exitCode == 0,
            $"Generated PyNTT model execution failed.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
    }

    private void AssertGeneratedBlockBarrierChain(
        string generatedSource,
        string topKernelName,
        params string[] helperNames)
    {
        Assert.True(helperNames.Length >= 2, "A barrier chain requires at least two helpers.");
        var topKernelMarker = $"def {topKernelName}(";
        var topKernelStart = generatedSource.LastIndexOf(topKernelMarker, StringComparison.Ordinal);
        Assert.True(topKernelStart >= 0, $"Generated source does not define top kernel {topKernelName}.");
        var topKernelSource = generatedSource[topKernelStart..];
        Assert.Equal(
            helperNames.Length - 1,
            Regex.Matches(topKernelSource, @"^    tl\.debug_barrier\(\)$", RegexOptions.Multiline).Count);

        for (var index = 0; index + 1 < helperNames.Length; index++)
        {
            var producer = Regex.Escape(helperNames[index]);
            var consumer = Regex.Escape(helperNames[index + 1]);
            Assert.Matches(
                $@"    {producer}\([^\r\n]*\)\r?\n    tl\.debug_barrier\(\)\r?\n    {consumer}\(",
                topKernelSource);
        }
    }

    private (int ExitCode, string Stdout, string Stderr) RunPythonScript(string script)
    {
        var python = Environment.GetEnvironmentVariable("PYTHON") ?? "python";
        using var process = new Process();
        process.StartInfo.FileName = python;
        process.StartInfo.ArgumentList.Add("-c");
        process.StartInfo.ArgumentList.Add(script);
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.Start();
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        process.WaitForExit();
        Task.WaitAll(stdout, stderr);

        return (process.ExitCode, stdout.Result, stderr.Result);
    }

    private string PythonString(string value) => JsonSerializer.Serialize(value, PythonStringLiteralOptions);

    private string RemovedLocalName(string scope, string name) => string.Join('_', scope, "local", name);

    private string RemovedLocalMeta(string scope, string name, string metric) => string.Join('_', RemovedLocalName(scope, name), metric);

    private IReadOnlyList<DistributedType> CollectDistributedTypes(BaseFunction function)
    {
        var types = new List<DistributedType>();
        var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);

        void Visit(BaseExpr expr)
        {
            if (!visited.Add(expr))
            {
                return;
            }

            if (expr.CheckedType is DistributedType distributedType)
            {
                types.Add(distributedType);
            }

            foreach (var operand in expr.Operands)
            {
                Visit(operand);
            }
        }

        if (function is Function f)
        {
            Visit(f.Body);
        }

        return types;
    }
}
