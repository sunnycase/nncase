// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.Passes.Distributed;
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
        var function = root.GetProperty("functions").EnumerateArray().Single();
        Assert.Equal("main", function.GetProperty("name").GetString());
        Assert.Equal("x", function.GetProperty("inputs").EnumerateArray().Single().GetProperty("name").GetString());
        Assert.Equal("float32", function.GetProperty("inputs").EnumerateArray().Single().GetProperty("dtype").GetString());
        Assert.Equal(1, function.GetProperty("outputs").EnumerateArray().Single().GetProperty("shape").EnumerateArray().Single().GetInt64());

        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        Assert.Contains("PyNTTGeneratedModel", modelPy, StringComparison.Ordinal);
        var specsPy = File.ReadAllText(Path.Join(outputDirectory, "specs.py"));
        Assert.Contains("TensorSpec", specsPy, StringComparison.Ordinal);
        Assert.Contains("outputs=", specsPy, StringComparison.Ordinal);

        AssertGeneratedModelImports(outputDirectory);
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
            .FirstOrDefault(type => type.Placement.Name == "yx" && type.AxisPolicies.Any(policy => policy is SBPSplit split && split.Axes.ToArray().SequenceEqual(new[] { 1 })));
        Assert.NotNull(distributedType);
        Assert.Equal(new[] { 28L, 0L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 3, 7 }).Offset).ToValueArray());
        Assert.Equal(new[] { 4L, 1L }, new RankedShape(DistributedUtility.GetLocalOffsetAndShape(distributedType, new[] { 3, 7 }).Shape).ToValueArray());

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
        Assert.Contains("generated from PyNTT Jinja TensorLoad.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja ElementwiseBinary.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("generated from PyNTT Jinja TensorStore.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim_tensor_load_0(source, data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim_tensor_load_1(source, data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim_elementwise_binary_0(data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim_tensor_store_0(destination, data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_tensor_load_0(input0, data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tl.debug_barrier()", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_tensor_load_1(input1, data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_elementwise_binary_0(data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_tensor_store_0(output0, data, rdata, chip_local_rdata, block_local_rdata, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shard_coord1 = tmp_shard % 8", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shard_coord0 = tmp_shard % 4", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("local_dim0 = 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("split_linear0 = shard_coord0 * 8 + shard_coord1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("global_idx0 = idx0 + split_linear0 * local_dim0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("source_offsets = 0 + lane * 0 + global_idx0 * (1 * 1) + global_idx1 * 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("result = value0 + value1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tl.store(destination + destination_offsets, value, mask=mask)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("data, rdata, chip_local_rdata, block_local_rdata, block_local_data", generatedKernelsPy, StringComparison.Ordinal);
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
        Assert.Contains("main_prim_binary_0[grid](", modelPy, StringComparison.Ordinal);
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
        CompileOptions.ShapeBucketOptions.SegmentsCount = 2;

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

        var finalMainTir = Directory.GetFiles(Dumpper.Directory, "main_prim.script", SearchOption.AllDirectories)
            .Select(File.ReadAllText)
            .Last(text => text.Contains("T.PrimFunc(\"main_prim\"", StringComparison.Ordinal) && text.Contains("-> ()", StringComparison.Ordinal));
        Assert.Contains("%out_", finalMainTir, StringComparison.Ordinal);
        Assert.Contains("main_segment_", finalMainTir, StringComparison.Ordinal);
        Assert.DoesNotContain("Return(", finalMainTir, StringComparison.Ordinal);
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
        var splitByFeatureType = new DistributedType(inputType, new SBP[] { SBP.B, SBP.S([0, 1], 4) }, placement);
        var splitByTokenType = new DistributedType(inputType, new SBP[] { SBP.S([0], 1), SBP.B }, placement);
        var featureShard = CreateBuffer("feature_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [3, 128], [4, 1], splitByFeatureType);
        var tokenShard = CreateBuffer("token_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 1024, [3, 128], [128, 1], splitByTokenType);
        var output = CreateOutputVar("output", inputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(featureShard, input, splitByFeatureType.AxisPolicies, placement),
            TIR.F.NTT.GatherReduceScatter(featureShard, tokenShard, splitByFeatureType, splitByTokenType),
            TIR.F.NTT.TensorStore(tokenShard, output, splitByTokenType.AxisPolicies, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
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
        Assert.Contains("stage=to_collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("stage=from_collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tle.distributed_barrier(pyntt_grid_mesh)", generatedKernelsPy, StringComparison.Ordinal);
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
            TIR.F.NTT.Pack(scalarInput, vector32Input, new[] { 4, 8 }, new[] { 2, 2 }),
            TIR.F.NTT.Transpose(vector32Input, vector32Output, new[] { 1, 0, 2 }),
            TIR.F.NTT.Unpack(vector32Output, scalarOutput, new[] { 4, 8 }, new[] { 2, 2 }),
            TIR.F.NTT.TensorStore(scalarOutput, output, outputDistributedType.AxisPolicies, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
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
        Assert.Contains("lane_flat = linear % 32", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("in_idx2 = out_idx2 * 32 + lane0 * 8 + lane1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("out_idx2 = in_idx2 * 32 + lane0 * 8 + lane1", generatedKernelsPy, StringComparison.Ordinal);
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
        var scalarDistributedType = new DistributedType(scalarType, new SBP[] { SBP.B, SBP.S([0, 1], 4) }, placement);
        var packedDistributedType = new DistributedType(packedType, new SBP[] { SBP.B, SBP.S([0, 1], 2) }, placement);
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
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
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
        var splitByTokenType = new DistributedType(inputType, new SBP[] { SBP.S([0], 1), SBP.B }, placement);
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
        Assert.Contains("stage=to_collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("in_non_split_linear = shard_coord1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("in_local_offset1 = tl.where(in_in_bound1, in_global_base1, 0)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("in_local_idx1 = in_idx1 + in_local_offset1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("writer_active = writer_active & (shard_coord1 == 0)", generatedKernelsPy, StringComparison.Ordinal);
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
        var scalarSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.B, SBP.S([1], 2), SBP.B }, placement);
        var vectorSplitType = new DistributedType(vectorType, new SBP[] { SBP.B, SBP.S([1], 2), SBP.B }, placement);
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
        Assert.Contains("stage=to_collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("lane=8, stage=to_collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("in_non_split_linear = shard_coord0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("in_local_idx2 = in_idx2 + in_local_offset2", generatedKernelsPy, StringComparison.Ordinal);
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
        var scalarFeatureSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.B, SBP.S([0, 1], 32) }, placement);
        var vectorFeatureSplitType = new DistributedType(vectorType, new SBP[] { SBP.B, SBP.S([0, 1], 4) }, placement);
        var vectorTokenSplitType = new DistributedType(vectorType, new SBP[] { SBP.S([0], 1), SBP.B }, placement);
        var scalarTokenSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.S([0], 1), SBP.B }, placement);
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
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
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
        Assert.Contains("lane=8, stage=to_collective", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("in_split_linear1 = shard_coord0 * 8 + shard_coord1", generatedKernelsPy, StringComparison.Ordinal);
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
        var scalarSplitType = new DistributedType(scalarInputType, new SBP[] { SBP.S([1], 2), SBP.B, SBP.S([0], 40) }, placement);
        var vectorSplitType = new DistributedType(vectorType, new SBP[] { SBP.S([1], 2), SBP.B, SBP.S([0], 5) }, placement);
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
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
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
        Assert.Contains("input_local_shape=(2, 16, 5)", generatedKernelsPy, StringComparison.Ordinal);
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
        var splitFeatureType = new DistributedType(inputType, new SBP[] { SBP.S([0], 5), SBP.S([1], 384) }, placement);
        var broadcastFeatureType = new DistributedType(inputType, new SBP[] { SBP.S([0], 5), SBP.B }, placement);
        var featureShard = CreateBuffer("feature_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 0, [5, 384], [384, 1], splitFeatureType);
        var broadcastShard = CreateBuffer("broadcast_shard", DataTypes.BFloat16, TIR.MemoryLocation.Data, 4096, [5, 3072], [3072, 1], broadcastFeatureType);
        var output = CreateOutputVar("output", inputType);
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(featureShard, input, splitFeatureType.AxisPolicies, placement),
            TIR.F.NTT.GatherReduceScatter(featureShard, broadcastShard, splitFeatureType, broadcastFeatureType),
            TIR.F.NTT.TensorStore(broadcastShard, output, broadcastFeatureType.AxisPolicies, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { input, output })
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
        Assert.Contains("input_local_shape=(5, 384)", generatedKernelsPy, StringComparison.Ordinal);
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
        AssertGeneratedKernel(outputDirectory, "composite", "Matmul.py.jinja");
        AssertGeneratedModelRuns(
            outputDirectory,
            "lhs = torch.arange(256, dtype=torch.float32, device='cuda').reshape(16, 16) * 0.01",
            "rhs = torch.arange(256, dtype=torch.float32, device='cuda').reshape(16, 16) * 0.02",
            "output = module(lhs, rhs)",
            "torch.testing.assert_close(output, lhs @ rhs, rtol=1e-5, atol=1e-5)");
    }

    [Fact]
    public async Task TestPyNTTIRAutoDistributedPackedBFloat16MatmulRun()
    {
        ConfigureAutoDistributedPyNTT();
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new[] { 1, 1024 }));
        var rhsValues = Enumerable.Range(0, 1024 * 2048)
            .Select(i => (BFloat16)(((float)i - 128f) * 0.0001f))
            .ToArray();
        var rhs = Tensor.From<BFloat16>(rhsValues, [1024, 2048]);
        var matmul = IR.F.Tensors.MatMul(lhs, rhs, DataTypes.BFloat16);
        var reshaped = IR.F.Tensors.Reshape(matmul, [1, 16, 128]);
        var main = new Function("main", PyNTTTarget.Kind, reshaped, new[] { lhs });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_bf16_packed_matmul_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("rhs_n_packed_lane=4, rhs_n_lane=8", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "lhs = ((torch.arange(1024, dtype=torch.float32, device='cuda').reshape(1, 1024) - 16) * 0.001).to(torch.bfloat16)",
            "rhs = ((torch.arange(1024 * 2048, dtype=torch.float32, device='cuda').reshape(1024, 2048) - 128) * 0.0001).to(torch.bfloat16)",
            "output = module(lhs)",
            "torch.testing.assert_close(output, (lhs @ rhs).reshape(1, 16, 128), rtol=2e-2, atol=2e-2)");
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
            [SBP.S([0, 1])]);
        var (root, queryVar, kvVars, kvCacheObjVar) = Nncase.Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(
            [20],
            [20],
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
        Assert.Contains("generated from PyNTT Jinja PagedAttention.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        using (var kernelParams = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "kernel_params.json"))))
        {
            var kernelInputs = kernelParams.RootElement.GetProperty("functions")
                .EnumerateArray()
                .SelectMany(function => function.GetProperty("render_kernels").EnumerateArray())
                .SelectMany(kernel => kernel.GetProperty("metadata").GetProperty("inputs").EnumerateArray())
                .Select(input => input.GetString())
                .ToArray();
            Assert.DoesNotContain(kvCacheObjVar.Name, kernelInputs);
            Assert.Contains($"{kvCacheObjVar.Name}.__metadata", kernelInputs);
        }

        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            "seq_len = 20",
            "num_q_heads = 16",
            "num_kv_heads = 8",
            "head_dim = 128",
            "query = (torch.randn(seq_len, num_q_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "key = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "value = (torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32) * 0.05).to(torch.bfloat16)",
            "class MockKVCache:",
            "    pass",
            "cache = MockKVCache()",
            "cache.context_lens = torch.tensor([0], dtype=torch.int64)",
            "cache.seq_lens = torch.tensor([seq_len], dtype=torch.int64)",
            "cache.block_tables = torch.tensor([[[0, 0]]], dtype=torch.int64)",
            "cache.slot_mapping = torch.stack([torch.zeros(seq_len, dtype=torch.int64), torch.arange(seq_len, dtype=torch.int64)], dim=1)",
            "cache.num_blocks = 32",
            "cache.kv_caches = torch.zeros((4, 8, 1, 2 * num_kv_heads * (head_dim // 8) * 256 * 8), dtype=torch.bfloat16, device='cuda')",
            "output = module(query, key, value, cache)",
            "ref = torch.empty((seq_len, num_q_heads, head_dim), dtype=torch.float32, device='cuda')",
            "for token in range(seq_len):",
            "    for q_head in range(num_q_heads):",
            "        kv_head = q_head // (num_q_heads // num_kv_heads)",
            "        scores = torch.matmul(key[:token + 1, kv_head, :].to(torch.float32), query[token, q_head, :].to(torch.float32))",
            "        probs = torch.softmax(scores, dim=0)",
            "        ref[token, q_head, :] = torch.matmul(probs, value[:token + 1, kv_head, :].to(torch.float32))",
            "torch.testing.assert_close(output.to(torch.float32), ref.to(torch.bfloat16).to(torch.float32), rtol=3e-2, atol=3e-2)");
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
            [SBP.S([0, 1])]);
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
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("layer_id_value = tl.load(layer_id + 0).to(tl.int64)", generatedKernelsPy, StringComparison.Ordinal);
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
            "cache.context_lens = torch.tensor([0], dtype=torch.int64)",
            "cache.seq_lens = torch.tensor([seq_len], dtype=torch.int64)",
            "cache.block_tables = torch.tensor([[[0, 0]]], dtype=torch.int64)",
            "cache.slot_mapping = torch.stack([torch.zeros(seq_len, dtype=torch.int64), torch.arange(seq_len, dtype=torch.int64)], dim=1)",
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
            [SBP.S([0, 1])]);
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
            "prefill_cache.context_lens = torch.tensor([0], dtype=torch.int64)",
            "prefill_cache.seq_lens = torch.tensor([prefill_len], dtype=torch.int64)",
            "prefill_cache.block_tables = torch.tensor([[[0, 0]]], dtype=torch.int64)",
            "prefill_cache.slot_mapping = torch.stack([torch.zeros(prefill_len, dtype=torch.int64), torch.arange(prefill_len, dtype=torch.int64)], dim=1)",
            "prefill_cache.num_blocks = 32",
            "prefill_cache.kv_caches = storage",
            "_ = module(prefill_query, prefill_key, prefill_value, prefill_cache)",
            "decode_cache = MockKVCache()",
            "decode_cache.context_lens = torch.tensor([prefill_len], dtype=torch.int64)",
            "decode_cache.seq_lens = torch.tensor([prefill_len + 1], dtype=torch.int64)",
            "decode_cache.block_tables = torch.tensor([[[0, 0]]], dtype=torch.int64)",
            "decode_cache.slot_mapping = torch.tensor([[0, prefill_len]], dtype=torch.int64)",
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
        var qWeight = new Var("q_weight", new TensorType(DataTypes.BFloat16, new[] { k, qn }));
        var kWeight = new Var("k_weight", new TensorType(DataTypes.BFloat16, new[] { k, kvn }));
        var vWeight = new Var("v_weight", new TensorType(DataTypes.BFloat16, new[] { k, kvn }));
        var qkv = IR.F.NN.QKVParallelLinear(
            input,
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
        var main = new Function("main", PyNTTTarget.Kind, qkv, new[] { input, qWeight, kWeight, vWeight });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_qwen_like_qkv_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja PackedQKVParallelLinear.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "torch.manual_seed(0)",
            $"input = (torch.randn({seq}, {k}, dtype=torch.float32, device='cuda') * 0.05).to(torch.bfloat16)",
            $"q_weight = (torch.randn({k}, {qn}, dtype=torch.float32, device='cuda') * 0.03).to(torch.bfloat16)",
            $"k_weight = (torch.randn({k}, {kvn}, dtype=torch.float32, device='cuda') * 0.03).to(torch.bfloat16)",
            $"v_weight = (torch.randn({k}, {kvn}, dtype=torch.float32, device='cuda') * 0.03).to(torch.bfloat16)",
            "q, k_out, v_out = module(input, q_weight, k_weight, v_weight)",
            "torch.testing.assert_close(q.to(torch.float32), (input @ q_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)",
            "torch.testing.assert_close(k_out.to(torch.float32), (input @ k_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)",
            "torch.testing.assert_close(v_out.to(torch.float32), (input @ v_weight).to(torch.bfloat16).to(torch.float32), rtol=2e-2, atol=2e-2)");
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
        Assert.Contains("generated from PyNTT Jinja PackedMatMulGlu.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
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
            [SBP.S([0, 1])]);
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var outputType = new TensorType(DataTypes.Float32, new[] { 20 });
        var outputDistributedType = new DistributedType(outputType, new SBP[] { SBP.S([0], 5) }, placement);
        var kvCacheObjVar = new Var("kvCache", TensorType.Scalar(new ReferenceType(new PagedAttentionKVCacheType { Config = config })));
        var output = CreateOutputVar("output", outputType);
        var outputBuffer = CreateBuffer("position_ids", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [5], [1], outputDistributedType);
        var body = new TIR.Sequential(
            TIR.F.NTT.GetPositionIds(kvCacheObjVar, outputBuffer, outputDistributedType),
            TIR.F.NTT.TensorStore(outputBuffer, output, outputDistributedType.AxisPolicies, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { kvCacheObjVar, output })
        {
            SchedResult =
            {
                DataUsage = 128,
            },
        };

        var outputDirectory = GeneratePyNTTModelDirectory("generated_get_position_ids_split_hierarchy_axis_run_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("global_start = (shard_coord0) * shard_axis_extent", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("global_start = shard_index * shard_axis_extent", generatedKernelsPy, StringComparison.Ordinal);
        AssertGeneratedModelRuns(
            outputDirectory,
            "class MockKVCache:",
            "    pass",
            "cache = MockKVCache()",
            "cache.device = torch.device('cuda')",
            "cache.context_lens = torch.tensor([0], dtype=torch.int64)",
            "cache.seq_lens = torch.tensor([20], dtype=torch.int64)",
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
        Assert.Contains("def main_prim_nested_prim_device", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("(data + 64).to(tl.pointer_type(tl.uint8))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("source = (main_prim_nested_prim_device_arg0_nested_input).to(tl.pointer_type(tl.float32))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("destination = (main_prim_nested_prim_device_arg1_nested_output).to(tl.pointer_type(tl.float32))", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("data + 192", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPyNTTBufferSubviewUsesFormalAbiStrides()
    {
        var publicType = new TensorType(DataTypes.Float32, new[] { 1 });
        var input = new Var("x", publicType);
        var publicOutput = CreateOutputVar("public_output", publicType);
        var tensorType = new TensorType(DataTypes.Float32, new[] { 8, 16 });
        var nestedInputVar = new TIR.BufferVar("nested_input", tensorType, TIR.BufferVarRole.Input, TIR.MemoryLocation.Input);
        var nestedOutputVar = CreateOutputVar("nested_output", tensorType);
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
                TIR.F.NTT.Reshape(
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
            [SBP.S([0])]);
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
        var fieldInput = kernel
            .GetProperty("metadata")
            .GetProperty("attrs")
            .GetProperty("kv_cache_field_inputs")
            .EnumerateArray()
            .Single();
        Assert.Equal("cache", fieldInput.GetProperty("SourceName").GetString());
        Assert.Equal("metadata", fieldInput.GetProperty("Field").GetString());
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
        Assert.Contains("generated from PyNTT Jinja Gemv.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
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
            TIR.F.NTT.PackedMatMul(lhsBuffer, rhsBuffer, packedOutputBuffer, None.Default, 1.0f),
            TIR.F.NTT.Unpack(packedOutputBuffer, vectorOutputBuffer, new[] { 4 }, new[] { 1 }),
            TIR.F.NTT.Unpack(vectorOutputBuffer, outputBuffer, new[] { 8 }, new[] { 1 }),
            TIR.F.NTT.TensorStore(outputBuffer, output, new[] { SBP.B, SBP.B }, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { lhs, output })
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
        AssertGeneratedModelRuns(
            outputDirectory,
            "lhs = (torch.arange(64, dtype=torch.float32, device='cuda').reshape(1, 64) - 16) * 0.01",
            "rhs = (torch.arange(64 * 128, dtype=torch.float32, device='cuda').reshape(64, 128) - 128) * 0.001",
            "output = module(lhs)",
            "torch.testing.assert_close(output, lhs @ rhs, rtol=1e-4, atol=1e-4)");
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
            new SBP[] { SBP.S([0], localM), SBP.B },
            placement);
        var rhsDistributedType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, nPackedLane, nLane), new[] { packedN, k }),
            new SBP[] { SBP.S([1], localPackedN), SBP.B },
            placement);
        var packedOutputDistributedType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, nPackedLane, nLane), new[] { m, packedN }),
            new SBP[] { SBP.S([0], localM), SBP.S([1], localPackedN) },
            placement);
        var vectorOutputDistributedType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, nLane), new[] { m, n / nLane }),
            new SBP[] { SBP.S([0], localM), SBP.S([1], localVectorN) },
            placement);
        var outputDistributedType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new[] { m, n }),
            new SBP[] { SBP.S([0], localM), SBP.S([1], localN) },
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
            TIR.F.NTT.PackedMatMul(lhsBuffer, rhsBuffer, packedOutputBuffer, None.Default, 1.0f),
            TIR.F.NTT.Unpack(packedOutputBuffer, vectorOutputBuffer, new[] { nPackedLane }, new[] { 1 }),
            TIR.F.NTT.Unpack(vectorOutputBuffer, outputBuffer, new[] { nLane }, new[] { 1 }),
            TIR.F.NTT.TensorStore(outputBuffer, output, outputDistributedType.AxisPolicies.ToArray(), placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new IVar[] { lhs, output })
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
    public async Task TestPyNTTAffineBitcastUsesDedicatedTemplate()
    {
        ConfigureAutoDistributedPyNTT();
        var inputType = new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 4, 8 });
        var input = new Var("input", inputType);
        var bitcast = IR.F.Tensors.Bitcast(input, DataTypes.BFloat16);
        var main = new Function("main", PyNTTTarget.Kind, bitcast, [input]);

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_affine_bitcast_model", main);
        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("generated from PyNTT Jinja Bitcast.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("generated from PyNTT Jinja Reshape.py.jinja", generatedKernelsPy, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TestPyNTTAffineDynamicVectorizedBinaryCodegen()
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
        Assert.NotEmpty(cacheBuffers);
        Assert.All(cacheBuffers, buffer => Assert.NotNull(buffer.DistributedType));

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
        Assert.Contains($"generated from PyNTT Jinja {templateFileName}", generatedKernelsPy, StringComparison.Ordinal);
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
        var result = RunPythonScript(script);
        Assert.True(
            result.ExitCode == 0,
            $"PyNTT Jinja kernel rendering failed.{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");
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

        var result = RunPythonScript(script);
        Assert.True(
            result.ExitCode == 0,
            $"Generated PyNTT model import failed.{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");
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

        var result = RunPythonScript(script);
        if (result.ExitCode == 77)
        {
            Assert.Skip($"PyNTT end-to-end runtime test requires torch, triton, and CUDA.{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");
        }

        Assert.True(
            result.ExitCode == 0,
            $"Generated PyNTT model execution failed.{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");
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
        process.WaitForExit();

        return (process.ExitCode, process.StandardOutput.ReadToEnd(), process.StandardError.ReadToEnd());
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
