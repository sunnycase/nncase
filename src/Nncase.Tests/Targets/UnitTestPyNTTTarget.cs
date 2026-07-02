// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
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
        var main = new Function("main", PyNTTTarget.Kind, x, new[] { x });
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
        Assert.Equal(0, launchMeta.GetProperty("thread_local_rdata_pool_bytes").GetInt64());
        Assert.Equal(0, launchMeta.GetProperty("warp_local_rdata_pool_bytes").GetInt64());
        Assert.Equal(0, launchMeta.GetProperty("block_local_rdata_pool_bytes").GetInt64());
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
        Assert.Contains("def main_prim_tensor_load_0(source, data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim_tensor_load_1(source, data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim_elementwise_binary_0(data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("def main_prim_tensor_store_0(destination, data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size: tl.constexpr):", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_tensor_load_0(input0, data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tl.debug_barrier()", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_tensor_load_1(input1, data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_elementwise_binary_0(data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_tensor_store_0(output0, data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata, warp_local_data, block_local_data, block_size)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shard_coord1 = tmp_shard % 8", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("shard_coord0 = tmp_shard % 4", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("local_dim0 = 4", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("split_linear0 = shard_coord1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("global_idx0 = lane + split_linear0 * local_dim0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("source_offsets = 0 + lane * 0 + global_idx0 * (1 * 1) + global_idx1 * 1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("result = value0 + value1", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("tl.store(destination + destination_offsets, value, mask=mask)", generatedKernelsPy, StringComparison.Ordinal);
        Assert.Contains("data, rdata, thread_local_rdata, warp_local_rdata, block_local_rdata", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("from pyntt.backends.triton.kernels", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("elementwise_binary(input0, input1, output0", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("make_data_tensor_view", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("elementwise_binary_tensor", generatedKernelsPy, StringComparison.Ordinal);
        Assert.DoesNotContain("tle.distributed_barrier()", generatedKernelsPy, StringComparison.Ordinal);

        var rdataPy = File.ReadAllText(Path.Join(outputDirectory, "rdata.py"));
        Assert.Contains("RDATA_BUNDLES", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"main_prim\"", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"thread_local_rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"warp_local_rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);
        Assert.Contains("\"block_local_rdata_bytes\": 0", rdataPy, StringComparison.Ordinal);

        var modelPy = File.ReadAllText(Path.Join(outputDirectory, "model.py"));
        Assert.Contains("grid = (32,)", modelPy, StringComparison.Ordinal);
        Assert.Contains("from .generated_kernels import main_prim_binary_0", modelPy, StringComparison.Ordinal);
        Assert.Contains("main_prim_binary_0[grid](", modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain("from . import generated_kernels as _generated_kernels", modelPy, StringComparison.Ordinal);
        Assert.DoesNotContain("_generated_kernels.", modelPy, StringComparison.Ordinal);
        Assert.Contains("data = self.allocate_workspace(inputs, ", modelPy, StringComparison.Ordinal);
        Assert.Contains("warp_local_data = self.allocate_workspace(inputs, ", modelPy, StringComparison.Ordinal);
        Assert.Contains("block_local_data = self.allocate_workspace(inputs, ", modelPy, StringComparison.Ordinal);
        Assert.Contains("rdata, thread_local_rdata, warp_local_rdata, block_local_rdata = self.materialize_rdata_bundle(inputs, \"main_prim\")", modelPy, StringComparison.Ordinal);
        AssertGeneratedModelRunsBinaryAdd(outputDirectory);
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
    public async Task TestPyNTTIRAutoDistributedThreadLocalRDataRun()
    {
        ConfigureAutoDistributedPyNTT();
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 32, 1 }));
        var bias = Tensor.From<float>(Enumerable.Range(0, 32).Select(i => i * 0.5f).ToArray(), [32, 1]);
        var main = new Function("main", PyNTTTarget.Kind, IR.F.Math.Binary(BinaryOp.Add, x, bias), new[] { x });

        var outputDirectory = await GeneratePyNTTModelDirectoryWithCompilerPipeline("generated_thread_local_rdata_run_model", main);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Join(outputDirectory, "metadata.json")));
        var launchMeta = document.RootElement.GetProperty("functions").EnumerateArray().Single()
            .GetProperty("generated_kernels").EnumerateArray().Single()
            .GetProperty("launch").GetProperty("meta");
        Assert.True(launchMeta.GetProperty("thread_local_rdata_pool_bytes").GetInt64() > 0);
        Assert.True(launchMeta.GetProperty("thread_local_rdata_stride_bytes").GetInt64() > 0);

        RenderGeneratedKernels(outputDirectory);
        var generatedKernelsPy = File.ReadAllText(Path.Join(outputDirectory, "generated_kernels.py"));
        Assert.Contains("thread_local_rdata + shard_index *", generatedKernelsPy, StringComparison.Ordinal);
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
    public void TestPyNTTPackedMatmulUsesTwoDimensionalNLanes()
    {
        var lhsBuffer = CreateDataBuffer("lhs", DataTypes.Float32, 0, [1, 64], [64, 1]);
        var packedElemType = new VectorType(DataTypes.Float32, 4, 8);
        var rhsBuffer = CreateDataBuffer("rhs", packedElemType, 256, [4, 64], [64, 1]);
        var outputBuffer = CreateDataBuffer("output", packedElemType, 33024, [1, 4], [4, 1]);
        var outputAlias = new Var("output_alias", new TensorType(DataTypes.Float32, new[] { 1 }));
        var body = new TIR.Sequential(
            TIR.F.NTT.PackedMatMul(lhsBuffer, rhsBuffer, outputBuffer, None.Default, 1.0f),
            TIR.T.Return(outputAlias));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new[] { outputAlias })
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
        var output = new Var("output", new TensorType(DataTypes.Float32, new[] { 1, 128 }));
        var packedElemType = new VectorType(DataTypes.Float32, 4, 8);
        var rhs = CreatePackedMatmulRhsConst();
        var rhsSizeBytes = checked((ulong)rhs.Value.Length * (ulong)rhs.Value.ElementType.SizeInBytes);
        var lhsBuffer = CreateBuffer("lhs_buffer", DataTypes.Float32, TIR.MemoryLocation.Data, 0, [1, 64], [64, 1]);
        var rhsBuffer = CreateBuffer("rhs_buffer", packedElemType, TIR.MemoryLocation.Rdata, 0, [4, 64], [64, 1]);
        var packedOutputBuffer = CreateBuffer("packed_output", packedElemType, TIR.MemoryLocation.Data, 256, [1, 4], [4, 1]);
        var vectorOutputBuffer = CreateBuffer("vector_output", new VectorType(DataTypes.Float32, 8), TIR.MemoryLocation.Data, 768, [1, 16], [16, 1]);
        var outputBuffer = CreateBuffer("output_buffer", DataTypes.Float32, TIR.MemoryLocation.Data, 1280, [1, 128], [128, 1]);
        var placement = new Placement(new[] { 1 }, "t");
        var body = new TIR.Sequential(
            TIR.F.NTT.TensorLoad(lhsBuffer, lhs, new[] { SBP.B, SBP.B }, placement),
            TIR.F.NTT.PackedMatMul(lhsBuffer, rhsBuffer, packedOutputBuffer, None.Default, 1.0f),
            TIR.F.NTT.Unpack(packedOutputBuffer, vectorOutputBuffer, new[] { 4 }, new[] { 1 }),
            TIR.F.NTT.Unpack(vectorOutputBuffer, outputBuffer, new[] { 8 }, new[] { 1 }),
            TIR.F.NTT.TensorStore(outputBuffer, output, new[] { SBP.B, SBP.B }, placement));
        var main = new TIR.PrimFunction("main_prim", PyNTTTarget.Kind, body, new[] { lhs })
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
        var outputDirectory = Path.Join(CompileOptions.DumpDir, directoryName);
        if (Directory.Exists(outputDirectory))
        {
            Directory.Delete(outputDirectory, recursive: true);
        }

        ((PyNTTTargetOptions)CompileOptions.TargetOptions).OutputDirectory = outputDirectory;

        var module = new IRModule(function);
        var linkedModel = new ModelBuilder(CompileSession.Target, CompileOptions).Build(module);

        using var stream = new MemoryStream();
        linkedModel.Serialize(stream);
        Assert.NotEqual(0, stream.Length);
        return outputDirectory;
    }

    private TIR.Buffer CreateDataBuffer(string name, DataType elemType, long startBytes, long[] dimensions, long[] strides)
        => CreateBuffer(name, elemType, TIR.MemoryLocation.Data, startBytes, dimensions, strides);

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
