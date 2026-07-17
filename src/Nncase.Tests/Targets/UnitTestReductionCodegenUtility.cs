// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TargetTest;

public sealed class UnitTestReductionCodegenUtility
{
    [Fact]
    public void TestRecognizesAdjacentLoopPartitionPair()
    {
        var full = CreateLoop("i", 0, 8, LoopPartition.Full);
        var tail = CreateLoop("i_tail", 8, 10, LoopPartition.Tail);

        Assert.True(ReductionCodegenUtility.TryGetAdjacentReductionLoopPartitionPair(
            new Expr[] { full, tail },
            0,
            out var actualFull,
            out var actualTail));
        Assert.Same(full, actualFull);
        Assert.Same(tail, actualTail);
    }

    [Fact]
    public void TestDoesNotMergeSeparatedLoopPartitions()
    {
        var full = CreateLoop("i", 0, 8, LoopPartition.Full);
        var tail = CreateLoop("i_tail", 8, 10, LoopPartition.Tail);

        Assert.False(ReductionCodegenUtility.TryGetAdjacentReductionLoopPartitionPair(
            new Expr[] { full, T.Nop(), tail },
            0,
            out _,
            out _));
        Assert.False(ReductionCodegenUtility.TryGetAdjacentReductionLoopPartitionPair(
            new Expr[] { full, T.Nop(), tail },
            2,
            out _,
            out _));
    }

    [Fact]
    public void TestRejectsMalformedAdjacentLoopPartitionPair()
    {
        var full = CreateLoop("i", 0, 8, LoopPartition.Full);
        var tail = CreateLoop("i_tail", 7, 10, LoopPartition.Tail);

        Assert.Throws<InvalidOperationException>(() =>
            ReductionCodegenUtility.TryGetAdjacentReductionLoopPartitionPair(
                new Expr[] { full, tail },
                0,
                out _,
                out _));
    }

    [Fact]
    public void TestGroupsClonedSubviewsByTheirSharedAccumulatorSource()
    {
        var accumulator = new BufferVar(
            "accumulator",
            new TensorType(DataTypes.Float32, new[] { 1 }),
            BufferVarRole.Output,
            MemoryLocation.Output);
        var fullOutput = IR.F.Buffer.BufferSubview(accumulator, new RankedShape(0), new RankedShape(1));
        var tailOutput = IR.F.Buffer.BufferSubview(accumulator, new RankedShape(0), new RankedShape(1));
        var fullUpdate = TIR.F.NTT.Reduce(
            accumulator,
            fullOutput,
            0,
            Array.Empty<int>(),
            Array.Empty<Dimension>(),
            new IRArray<int>(),
            false,
            ReduceOp.Sum);
        var tailUpdate = TIR.F.NTT.Reduce(
            accumulator,
            tailOutput,
            1,
            Array.Empty<int>(),
            Array.Empty<Dimension>(),
            new IRArray<int>(),
            false,
            ReduceOp.Sum);

        var group = Assert.Single(ReductionCodegenUtility.CollectReductionCallGroups(fullUpdate, tailUpdate));
        Assert.Equal(2, group.Calls.Length);
    }

    [Fact]
    public void TestDoesNotGroupDistinctAccumulatorRegionsSharingPhysicalStorage()
    {
        var owner = new BufferVar(
            "owner",
            new TensorType(DataTypes.Float32, new[] { 2 }),
            BufferVarRole.Output,
            MemoryLocation.Output);
        var physicalBuffer = new PhysicalBuffer(8, owner, 8, MemoryLocation.Output);
        var firstOutput = new TIR.Buffer(
            "first_output",
            DataTypes.Float32,
            new MemSpan(physicalBuffer, 0, 4),
            new Dimension[] { 1 },
            new Dimension[] { 1 },
            null);
        var secondOutput = new TIR.Buffer(
            "second_output",
            DataTypes.Float32,
            new MemSpan(physicalBuffer, 4, 4),
            new Dimension[] { 1 },
            new Dimension[] { 1 },
            null);
        var fullUpdate = TIR.F.NTT.Reduce(
            owner,
            firstOutput,
            0,
            Array.Empty<int>(),
            Array.Empty<Dimension>(),
            new IRArray<int>(),
            false,
            ReduceOp.Sum);
        var tailUpdate = TIR.F.NTT.Reduce(
            owner,
            secondOutput,
            1,
            Array.Empty<int>(),
            Array.Empty<Dimension>(),
            new IRArray<int>(),
            false,
            ReduceOp.Sum);

        Assert.Throws<InvalidOperationException>(() =>
            ReductionCodegenUtility.CollectReductionCallGroups(fullUpdate, tailUpdate));
    }

    private static For CreateLoop(string name, int start, int stop, LoopPartition partition)
        => new(
            new DimVar(name),
            new Nncase.TIR.Range(start, stop, 4),
            LoopMode.Reduction,
            new Sequential(T.Nop()),
            partition);
}
