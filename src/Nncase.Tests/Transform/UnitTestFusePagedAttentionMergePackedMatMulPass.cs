// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.NTT;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFusePagedAttentionMergePackedMatMulPass : TestClassBase
{
    [Fact]
    public async Task TestPreservesSplitAxisBarrierAtFusedConsumer()
    {
        var maxState = CreateBuffer("max_state", DataTypes.Float32, 0, [8]);
        var sumState = CreateBuffer("sum_state", DataTypes.Float32, 32, [8]);
        var accState = CreateBuffer("acc_state", DataTypes.Float32, 64, [8, 128]);
        var mergedPhysical = CreatePhysicalBuffer(DataTypes.BFloat16, 4160, 256);
        var mergedOutput = CreateBuffer("merged_output", DataTypes.BFloat16, mergedPhysical, [1, 128]);
        var lhs = CreateBuffer("lhs", DataTypes.BFloat16, mergedPhysical, [1, 128]);
        var rhs = CreateBuffer("rhs", DataTypes.BFloat16, 4416, [128, 256]);
        var output = CreateBuffer("output", DataTypes.BFloat16, 69952, [1, 256]);
        var merge = TIR.F.NTT.PagedAttentionMerge(
            maxState,
            sumState,
            accState,
            None.Default,
            mergedOutput,
            new IRArray<AttentionDimKind>(
                new[] { AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim }),
            2048,
            splitHierarchyAxis: 0,
            splitCount: 8);
        var matmul = TIR.F.NTT.PackedMatMul(
            lhs,
            rhs,
            output,
            false,
            1.0f,
            rhsLayout: PackedMatMulRhsLayout.KMajor);
        var function = new PrimFunction(
            "main",
            PyNTTTarget.Kind,
            new Sequential(
                new Sequential(
                    new Expr[]
                    {
                        TIR.F.NTT.Barrier(
                            TIR.NTT.BarrierScope.Chip,
                            new IRArray<int>(new[] { 0 })),
                        merge,
                    },
                    traceScopeName: "paged_attention_split_kv_merge"),
                matmul),
            System.Array.Empty<IVar>());
        var module = new IRModule(function);

        var rewritten = await new FusePagedAttentionMergePackedMatMulPass(PyNTTTarget.Kind)
            .RunAsync(module, new());
        var rewrittenMain = Assert.IsType<PrimFunction>(rewritten.Entry);
        var calls = ExprCollector.Collect(rewrittenMain.Body).OfType<Call>().ToArray();
        Assert.DoesNotContain(calls, call => call.Target is TIR.NTT.PagedAttentionMerge);
        Assert.DoesNotContain(calls, call => call.Target is TIR.NTT.PackedMatMul);
        Assert.Single(calls.Where(call => call.Target is TIR.NTT.PagedAttentionMergePackedMatMul));

        var fields = rewrittenMain.Body.Fields.ToArray();
        var fusedIndex = System.Array.FindIndex(
            fields,
            field => field is Call { Target: TIR.NTT.PagedAttentionMergePackedMatMul });
        Assert.True(fusedIndex > 0);
        var barrier = Assert.IsType<TIR.NTT.Barrier>(Assert.IsType<Call>(fields[fusedIndex - 1]).Target);
        Assert.Equal(TIR.NTT.BarrierScope.Chip, barrier.Scope);
        Assert.Equal(new[] { 0 }, barrier.AxisGroupAxes.ToArray());
    }

    private static Nncase.TIR.Buffer CreateBuffer(
        string name,
        DataType dataType,
        ulong offset,
        Dimension[] shape)
    {
        var sizeBytes = shape.Aggregate(1L, (size, dimension) => size * dimension.FixedValue) * dataType.SizeInBytes;
        return CreateBuffer(name, dataType, CreatePhysicalBuffer(dataType, offset, sizeBytes), shape);
    }

    private static PhysicalBuffer CreatePhysicalBuffer(DataType dataType, ulong offset, long sizeBytes)
        => new(
            dataType.SizeInBytes,
            Tensor.FromPointer(offset, dataType),
            sizeBytes,
            MemoryLocation.ChipLocalData);

    private static Nncase.TIR.Buffer CreateBuffer(
        string name,
        DataType dataType,
        PhysicalBuffer physical,
        Dimension[] shape)
        => new(
            name,
            dataType,
            new MemSpan(physical, 0, physical.Size),
            shape,
            TensorUtilities.GetDefaultStrides(shape.Select(dimension => checked((int)dimension.FixedValue)).ToArray())
                .Select(stride => (Dimension)stride)
                .ToArray(),
            null);
}
