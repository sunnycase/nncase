// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestInlineSingleCallPrimFunctionsPass : TestClassBase
{
    private const string ModuleKind = "test_tir_inline";

    [Fact]
    public async Task TestTIRSelectionPreservesNonTensorParameterIdentity()
    {
        var layerId = new DimVar("layer_id") { Metadata = { Range = new(1, 16) } };
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 16 }));
        var slice = IR.F.Tensors.Slice(
            input,
            new Dimension[] { 0 },
            new Dimension[] { layerId },
            new Dimension[] { 0 },
            new Dimension[] { 1 });
        var function = new Function("main", ModuleKind, slice, new IVar[] { input, layerId });
        Assert.True(function.InferenceType());

        var lowered = Assert.IsType<PrimFunction>(
            await new NTTTIRSelectionPass(CompileOptions, ModuleKind).RunAsync(function, new()));

        Assert.Same(layerId, lowered.Parameters[1]);
        var bodyLayerIds = ExprCollector.Collect(lowered.Body)
            .OfType<DimVar>()
            .Where(dimVar => dimVar.Name == layerId.Name)
            .ToArray();
        Assert.NotEmpty(bodyLayerIds);
        Assert.All(bodyLayerIds, bodyLayerId => Assert.Same(layerId, bodyLayerId));
    }

    [Fact]
    public async Task TestInlineSingleCallSubstitutesActualBufferDescriptor()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var formal = new BufferVar("formal", tensorType, BufferVarRole.Input, MemoryLocation.Data);
        var callee = MakeLoadFunction("callee", formal);
        var actual = MakeStridedBuffer("actual", stride: 2);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee, actual)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        Assert.Single(module.Functions);
        var load = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(caller.Body)));
        Assert.IsType<LoadT>(load.Target);
        var inlinedBuffer = Assert.IsType<Buffer>(load.Arguments[0]);
        Assert.Same(actual, inlinedBuffer);
        Assert.Equal(2L, inlinedBuffer.Strides[0].FixedValue);
    }

    [Fact]
    public async Task TestInlineSingleCallSubstitutesAttachedFormalBufferDescriptor()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var formal = new BufferVar("formal", tensorType, BufferVarRole.Input, MemoryLocation.Data);
        var formalBuffer = T.AttachBuffer(formal, tensorType, MemoryLocation.Data, 0, out _, "formal_buffer");
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(new LoadT(), formalBuffer, formalBuffer)),
            new IVar[] { formal });
        var actual = MakeStridedBuffer("actual", stride: 2);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee, actual)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        Assert.Single(module.Functions);
        var traceScope = Assert.IsType<Sequential>(Assert.Single(caller.Body.Fields.ToArray()));
        Assert.Equal(callee.Name, traceScope.TraceScopeName);
        Assert.True(traceScope.PreserveCodegenBoundary);
        var load = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(caller.Body)));
        Assert.Same(actual, load.Arguments[0]);
        Assert.Same(actual, load.Arguments[1]);
        Assert.Equal(2L, Assert.IsType<Buffer>(load.Arguments[0]).Strides[0].FixedValue);
    }

    [Fact]
    public async Task TestInlineSingleCallSubstitutesDimensionArgument()
    {
        var sinkParameter = new DimVar("sink_value");
        var sink = new PrimFunction(
            "sink",
            ModuleKind,
            new Sequential(),
            new IVar[] { sinkParameter });
        var calleeParameter = new DimVar("layer_id");
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(sink, calleeParameter)),
            new IVar[] { calleeParameter });
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(
                new Call(callee, new DimConst(0)),
                new Call(sink, new DimConst(1))),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);
        module.Add(sink);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        Assert.Equal(2, module.Functions.Count);
        var calls = GetExecutableStatements(caller.Body).Select(Assert.IsType<Call>).ToArray();
        Assert.All(calls, call => Assert.Same(sink, call.Target));
        Assert.Equal(0L, Assert.IsType<DimConst>(calls[0].Arguments[0]).Value);
        Assert.Equal(1L, Assert.IsType<DimConst>(calls[1].Arguments[0]).Value);
    }

    [Fact]
    public async Task TestInlineSingleCallComposesFormalAliasWithActualStorage()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var formal = new BufferVar("formal", tensorType, BufferVarRole.Input, MemoryLocation.Data);
        var formalRoot = T.AttachBuffer(formal, tensorType, MemoryLocation.Data, 0, out _, "formal_root");
        var formalAlias = T.CreateBufferView(
            formalRoot,
            DataTypes.Float32,
            new Dimension[] { 2 },
            new Dimension[] { 1 },
            4,
            8,
            name: "formal_alias");
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(new LoadT(), formalAlias, formalAlias)),
            new IVar[] { formal });
        var actualRoot = MakeStridedBuffer("actual_root", stride: 1);
        var actual = T.CreateBufferView(
            actualRoot,
            DataTypes.Float32,
            new Dimension[] { 4 },
            new Dimension[] { 1 },
            8,
            16,
            name: "actual");
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee, actual)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        var load = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(caller.Body)));
        var inlinedAlias = Assert.IsType<Buffer>(load.Arguments[0]);
        Assert.Same(actualRoot.MemSpan.Buffer, inlinedAlias.MemSpan.Buffer);
        Assert.Equal(12L, inlinedAlias.MemSpan.Start.FixedValue);
        Assert.Equal(8L, inlinedAlias.MemSpan.Size.FixedValue);
    }

    [Fact]
    public async Task TestInlineSingleCallSubstitutesDimensionsInLocalMemSpan()
    {
        var offset = new DimVar("offset");
        var size = new DimVar("size");
        var localBuffer = new Buffer(
            "local",
            DataTypes.Float32,
            new MemSpan(
                new PhysicalBuffer(DataTypes.Float32.SizeInBytes, 64, MemoryLocation.Data),
                offset,
                size),
            new Dimension[] { 4 },
            new Dimension[] { 1 },
            null);
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(new LoadT(), localBuffer, localBuffer)),
            new IVar[] { offset, size });
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee, new DimConst(12), new DimConst(16))),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        var load = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(caller.Body)));
        var inlinedBuffer = Assert.IsType<Buffer>(load.Arguments[0]);
        Assert.Equal(12L, inlinedBuffer.MemSpan.Start.FixedValue);
        Assert.Equal(16L, inlinedBuffer.MemSpan.Size.FixedValue);
    }

    [Fact]
    public async Task TestInlineSingleCallRebasesBufferizedWorkspace()
    {
        var workspace = new BufferVar(
            "data",
            TensorType.Scalar(new PointerType(DataTypes.UInt8)),
            BufferVarRole.Workspace,
            MemoryLocation.Data);
        var localBuffer = new Buffer(
            "local",
            DataTypes.Float32,
            new MemSpan(
                new PhysicalBuffer(DataTypes.Float32.SizeInBytes, 16, 16, MemoryLocation.Data),
                0,
                16),
            new Dimension[] { 4 },
            new Dimension[] { 1 },
            null);
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(new LoadT(), localBuffer, localBuffer)),
            new IVar[] { workspace });
        var actualWorkspace = new Buffer(
            "actual_workspace",
            DataTypes.UInt8,
            new MemSpan(
                new PhysicalBuffer(DataTypes.UInt8.SizeInBytes, 100, 64, MemoryLocation.Data),
                4,
                32),
            new Dimension[] { 32 },
            new Dimension[] { 1 },
            null);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee, actualWorkspace)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        var load = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(caller.Body)));
        var inlinedBuffer = Assert.IsType<Buffer>(load.Arguments[0]);
        var start = Assert.IsType<TensorConst>(inlinedBuffer.MemSpan.Buffer.Start);
        Assert.Equal(120L, System.Convert.ToInt64(start.Value[System.Array.Empty<long>()]));
        Assert.Equal(16L, inlinedBuffer.MemSpan.Buffer.Size.FixedValue);
    }

    [Fact]
    public async Task TestInlineSingleCallRebasesPointerBackedWorkspace()
    {
        var workspace = new BufferVar(
            "data",
            TensorType.Scalar(new PointerType(DataTypes.UInt8)),
            BufferVarRole.Workspace,
            MemoryLocation.Data);
        var localBuffer = new Buffer(
            "local",
            DataTypes.Float32,
            new MemSpan(
                new PhysicalBuffer(DataTypes.Float32.SizeInBytes, 16, 16, MemoryLocation.Data),
                0,
                16),
            new Dimension[] { 4 },
            new Dimension[] { 1 },
            null);
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(new LoadT(), localBuffer, localBuffer)),
            new IVar[] { workspace });
        var pointerStart = new TensorConst(Tensor.FromScalar(new Pointer<byte>(100)));
        var actualWorkspace = new Buffer(
            "actual_workspace",
            DataTypes.UInt8,
            new MemSpan(
                new PhysicalBuffer(DataTypes.UInt8.SizeInBytes, pointerStart, 64, MemoryLocation.Data),
                4,
                32),
            new Dimension[] { 32 },
            new Dimension[] { 1 },
            null);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee, actualWorkspace)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        var load = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(caller.Body)));
        var inlinedBuffer = Assert.IsType<Buffer>(load.Arguments[0]);
        var start = Assert.IsType<TensorConst>(inlinedBuffer.MemSpan.Buffer.Start);
        Assert.Equal(120L, start.Value.ToScalar<long>());
        Assert.Equal(16L, inlinedBuffer.MemSpan.Buffer.Size.FixedValue);
    }

    [Fact]
    public async Task TestInlineSingleCallTransfersReadOnlyDataAllocations()
    {
        var rdata = new TensorConst(Tensor.FromScalar(1.0f));
        var chipLocalRdata = new TensorConst(Tensor.FromScalar(2.0f));
        var blockLocalRdata = new TensorConst(Tensor.FromScalar(3.0f));
        var source = MakeStridedBuffer("source", stride: 1);
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(new LoadT(), source, source)),
            System.Array.Empty<IVar>());
        callee.SchedResult.Rdatas.Add(rdata, new(0, 4));
        callee.SchedResult.ChipLocalRdatas.Add(chipLocalRdata, new(16, 20));
        callee.SchedResult.BlockLocalRdatas.Add(blockLocalRdata, new(32, 36));
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        Assert.Single(module.Functions);
        Assert.Equal(new ValueRange<ulong>(0, 4), caller.SchedResult.Rdatas[rdata]);
        Assert.Equal(new ValueRange<ulong>(16, 20), caller.SchedResult.ChipLocalRdatas[chipLocalRdata]);
        Assert.Equal(new ValueRange<ulong>(32, 36), caller.SchedResult.BlockLocalRdatas[blockLocalRdata]);
    }

    [Fact]
    public async Task TestKeepPrimFunctionWithMultipleCallSites()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var formal = new BufferVar("formal", tensorType, BufferVarRole.Input, MemoryLocation.Data);
        var callee = MakeLoadFunction("callee", formal);
        var actual = MakeStridedBuffer("actual", stride: 1);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(
                new Call(callee, actual),
                new Call(callee, actual)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        Assert.Equal(2, module.Functions.Count);
        var calls = GetExecutableStatements(caller.Body).Select(Assert.IsType<Call>).ToArray();
        Assert.All(calls, call => Assert.Same(callee, call.Target));
    }

    [Fact]
    public async Task TestBufferizePreservesCanonicalSharedCalleeForInlining()
    {
        T.CreateBuffer(
            new TensorType(DataTypes.UInt8, new[] { 4 }),
            MemoryLocation.Data,
            out var localBuffer,
            "leaf_local");
        var leaf = new PrimFunction(
            "leaf",
            ModuleKind,
            new Sequential(new Call(new LoadT(), localBuffer, localBuffer)),
            System.Array.Empty<IVar>());
        var decoder = new PrimFunction(
            "decoder",
            ModuleKind,
            new Sequential(new Call(leaf)),
            System.Array.Empty<IVar>());
        var main = new PrimFunction(
            "main",
            ModuleKind,
            new Sequential(new Call(decoder), new Call(decoder)),
            System.Array.Empty<IVar>());
        var module = new IRModule(main);
        module.Add(decoder);
        module.Add(leaf);

        var bufferizePass = new BufferizePass();
        await bufferizePass.RunAsync(module, new());
        await bufferizePass.RunAsync(module, new());

        var bufferedDecoder = module.Functions.OfType<PrimFunction>().Single(function => function.Name == decoder.Name);
        var bufferedLeaf = module.Functions.OfType<PrimFunction>().Single(function => function.Name == leaf.Name);
        var mainCalls = GetExecutableStatements(main.Body).Select(Assert.IsType<Call>).ToArray();
        Assert.Equal(2, mainCalls.Length);
        Assert.All(mainCalls, call => Assert.Same(bufferedDecoder, call.Target));

        var decoderWorkspaces = bufferedDecoder.Parameters
            .ToArray()
            .OfType<BufferVar>()
            .Where(parameter => parameter.Role == BufferVarRole.Workspace)
            .ToArray();
        Assert.Collection(
            decoderWorkspaces,
            parameter => Assert.Equal(MemoryLocation.Data, parameter.Location),
            parameter => Assert.Equal(MemoryLocation.ChipLocalData, parameter.Location),
            parameter => Assert.Equal(MemoryLocation.BlockLocalData, parameter.Location));
        var decoderCall = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(bufferedDecoder.Body)));
        Assert.Same(bufferedLeaf, decoderCall.Target);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        Assert.Equal(2, module.Functions.Count);
        Assert.Contains(module.Functions, function => ReferenceEquals(function, bufferedDecoder));
        Assert.DoesNotContain(module.Functions, function => ReferenceEquals(function, bufferedLeaf));
        Assert.DoesNotContain(
            GetExecutableStatements(bufferedDecoder.Body).OfType<Call>(),
            call => call.Target is PrimFunction);
        Assert.All(mainCalls, call => Assert.Same(bufferedDecoder, call.Target));
    }

    [Fact]
    public async Task TestInlineTransitiveSingleCallChain()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var leafFormal = new BufferVar("leaf_formal", tensorType, BufferVarRole.Input, MemoryLocation.Data);
        var leaf = MakeLoadFunction("leaf", leafFormal);
        var middleFormal = new BufferVar("middle_formal", tensorType, BufferVarRole.Input, MemoryLocation.Data);
        var middle = new PrimFunction(
            "middle",
            ModuleKind,
            new Sequential(new Call(leaf, middleFormal)),
            new IVar[] { middleFormal });
        var actual = MakeStridedBuffer("actual", stride: 3);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(middle, actual)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(middle);
        module.Add(leaf);

        await new InlineSingleCallPrimFunctionsPass(ModuleKind).RunAsync(module, new());

        Assert.Single(module.Functions);
        var load = Assert.IsType<Call>(Assert.Single(GetExecutableStatements(caller.Body)));
        Assert.IsType<LoadT>(load.Target);
        Assert.Same(actual, load.Arguments[0]);
    }

    [Fact]
    public async Task TestSpecializePrimFunctionBufferLayoutUpdatesFormalDescriptor()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var formal = new BufferVar("formal", tensorType, BufferVarRole.Input, MemoryLocation.Input);
        var formalBuffer = T.AttachBuffer(formal, tensorType, MemoryLocation.Input, 0, out _, "formal_buffer");
        var callee = new PrimFunction(
            "callee",
            ModuleKind,
            new Sequential(new Call(new LoadT(), formalBuffer, formalBuffer)),
            new IVar[] { formal });
        var actual = MakeStridedBuffer("actual", stride: 3);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(new Call(callee, actual)),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        var specializedModule = await new SpecializePrimFunctionBufferLayoutsPass().RunAsync(module, new());

        var specializedCallee = specializedModule.Functions.OfType<PrimFunction>().Single(function => function.Name == "callee");
        var specializedParameter = Assert.IsType<BufferVar>(Assert.Single(specializedCallee.Parameters.ToArray()));
        Assert.Equal(BufferLayoutKind.ExactStrided, specializedParameter.LayoutAnnotation.Kind);
        Assert.Equal(3L, specializedParameter.LayoutAnnotation.Strides[0].FixedValue);
        var specializedBuffer = ExprCollector.Collect(specializedCallee.Body)
            .OfType<Buffer>()
            .Single(buffer => ReferenceEquals(buffer.MemSpan.Buffer.Start, specializedParameter));
        Assert.Equal(3L, specializedBuffer.Strides[0].FixedValue);
        Assert.Equal(40L, specializedBuffer.MemSpan.Size.FixedValue);
    }

    [Fact]
    public async Task TestSpecializePrimFunctionBufferLayoutCreatesDistinctVariants()
    {
        var tensorType = new TensorType(DataTypes.Float32, new[] { 4 });
        var formal = new BufferVar("formal", tensorType, BufferVarRole.Input, MemoryLocation.Input);
        var callee = MakeLoadFunction("callee", formal);
        var caller = new PrimFunction(
            "caller",
            ModuleKind,
            new Sequential(
                new Call(callee, MakeStridedBuffer("actual_1", stride: 1)),
                new Call(callee, MakeStridedBuffer("actual_2", stride: 2))),
            System.Array.Empty<IVar>());
        var module = new IRModule(caller);
        module.Add(callee);

        var specializedModule = await new SpecializePrimFunctionBufferLayoutsPass().RunAsync(module, new());

        var variants = specializedModule.Functions.OfType<PrimFunction>()
            .Where(function => function.Name.StartsWith("callee", System.StringComparison.Ordinal))
            .OrderBy(function => function.Name, System.StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(2, variants.Length);
        Assert.Equal(new long[] { 1, 2 }, variants
            .Select(function => Assert.IsType<BufferVar>(function.Parameters[0]).LayoutAnnotation.Strides[0].FixedValue)
            .OrderBy(stride => stride)
            .ToArray());
        var calls = GetExecutableStatements(Assert.IsType<PrimFunction>(specializedModule.Entry).Body)
            .Select(Assert.IsType<Call>)
            .ToArray();
        Assert.NotSame(calls[0].Target, calls[1].Target);
    }

    private static PrimFunction MakeLoadFunction(string name, BufferVar formal)
        => new(
            name,
            ModuleKind,
            new Sequential(new Call(new LoadT(), formal, formal)),
            new IVar[] { formal });

    private static Expr[] GetExecutableStatements(Sequential sequential)
        => sequential.Fields.ToArray()
            .SelectMany(field => field is Sequential nested
                ? GetExecutableStatements(nested)
                : new[] { field })
            .ToArray();

    private static Buffer MakeStridedBuffer(string name, long stride)
        => new(
            name,
            DataTypes.Float32,
            new MemSpan(new PhysicalBuffer(DataTypes.Float32.SizeInBytes, 32, MemoryLocation.Data)),
            new Dimension[] { 4 },
            new Dimension[] { stride },
            null);
}
