// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.CostModelTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTritonTargetCostModel : TestClassBase
{
    [Fact]
    public void TestDynamicMatMulCostUsesMaxShape()
    {
        CompileOptions.TargetOptions = CreateOptions(CreateGpuMachine());
        var seq = new DimVar("seq").With(range: new ValueRange<double>(1, 20));
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape((Dimension)1, seq, (Dimension)64)));
        var rhs = new Var("rhs", new TensorType(DataTypes.BFloat16, new RankedShape((Dimension)64, (Dimension)128)));
        var matmul = IR.F.Tensors.MatMul(lhs, rhs);
        CompilerServices.InferenceType(matmul);

        var cost = CompilerServices.EvaluateCost(matmul, CompileOptions);

        Assert.Equal((UInt128)4_096, cost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestScalarMatMulCostDoesNotUseMatrixPrimitive()
    {
        var costModel = new TritonTargetOpCostModel(CreateGpuMachine(wgmmaInstructionsPerCycle: 4));
        var query = new MatMulOpCostQuery(
            new TargetCostTensor(DataTypes.BFloat16, new RankedShape(64, 64)),
            new TargetCostTensor(DataTypes.BFloat16, new RankedShape(64, 128)),
            new TargetCostTensor(DataTypes.BFloat16, new RankedShape(64, 128)),
            DataTypes.BFloat16);

        Assert.True(costModel.TryGetMatMulCost(query, out var cost));
        Assert.Equal((UInt128)8_192, cost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestVectorizedMatMulCanSelectWgmma()
    {
        var costModel = new TritonTargetOpCostModel(CreateGpuMachine(wgmmaInstructionsPerCycle: 4));
        var vectorBf16 = new VectorType(DataTypes.BFloat16, [8]);
        var query = new MatMulOpCostQuery(
            new TargetCostTensor(DataTypes.BFloat16, new RankedShape(64, 64)),
            new TargetCostTensor(vectorBf16, new RankedShape(64, 16)),
            new TargetCostTensor(vectorBf16, new RankedShape(64, 16)),
            DataTypes.BFloat16);

        Assert.True(costModel.TryGetMatMulCost(query, out var cost));
        Assert.Equal((UInt128)16, cost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestElementwiseCostCountsVectorLanes()
    {
        var costModel = new TritonTargetOpCostModel(CreateGpuMachine(rootBytesPerCycle: 128));
        var dtype = new VectorType(DataTypes.BFloat16, [8]);
        var tensor = new TargetCostTensor(dtype, new RankedShape(128));
        var query = new ElementwiseOpCostQuery("vectorized_cast", [tensor], tensor);

        Assert.True(costModel.TryGetElementwiseCost(query, out var cost));
        Assert.Equal((UInt128)8, cost[CostFactorNames.CPUCycles]);
        Assert.Equal((UInt128)2048, cost[CostFactorNames.BlockLocalMemoryLoadBytes]);
        Assert.Equal((UInt128)2048, cost[CostFactorNames.BlockLocalMemoryStoreBytes]);
        Assert.Equal((UInt128)332, costModel.GetLatency(cost));
    }

    [Fact]
    public void TestBoxingTensorStoreUsesLocalCopyCostWithoutSynchronization()
    {
        var machine = CreateGpuMachine(rootBytesPerCycle: 250);
        CompileOptions.TargetOptions = CreateOptions(machine);
        var costModel = new TritonTargetOpCostModel(machine);
        var placement = new Placement([4, 8], "y,x", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(128, 151936));
        var distributedType = new DistributedType(tensorType, [SBP.S([0]), SBP.S([1])], placement);
        var input = new Var("input", distributedType);
        var boxing = IR.F.Distributed.Boxing(input, tensorType);
        CompilerServices.InferenceType(boxing);

        var cost = CompilerServices.EvaluateCost(boxing, CompileOptions);

        Assert.False(cost.Factors.ContainsKey(CostFactorNames.CPUCycles));
        Assert.Equal((UInt128)2_430_976, cost[CostFactorNames.BlockLocalMemoryLoadBytes]);
        Assert.Equal((UInt128)2_430_976, cost[CostFactorNames.BlockLocalMemoryStoreBytes]);
        Assert.False(cost.Factors.ContainsKey(CostFactorNames.GridSynchronization));
        Assert.Equal((UInt128)19_748, costModel.GetLatency(cost));
    }

    [Fact]
    public void TestDistributedReshardIncludesGridSynchronization()
    {
        CompileOptions.TargetOptions = CreateOptions(CreateGpuMachine());
        var placement = new Placement([4, 8], "y,x", "bb");
        var tensorType = new TensorType(DataTypes.BFloat16, new RankedShape(16, 128));
        var inputType = new DistributedType(tensorType, [SBP.B, SBP.S([1])], placement);
        var outputType = new DistributedType(tensorType, [SBP.B, SBP.B], placement);
        var input = new Var("input", inputType);
        var boxing = IR.F.Distributed.Boxing(input, outputType);
        CompilerServices.InferenceType(boxing);

        var cost = CompilerServices.EvaluateCost(boxing, CompileOptions);

        Assert.True(cost[CostFactorNames.BlockLocalMemoryLoadBytes] > 0);
        Assert.True(cost[CostFactorNames.BlockLocalMemoryStoreBytes] > 0);
        Assert.Equal((UInt128)1, cost[CostFactorNames.GridSynchronization]);
    }

    [Fact]
    public void TestPackedMatMulUsesLocalSimtModel()
    {
        CompileOptions.TargetOptions = CreateOptions(CreateGpuMachine(rootBytesPerCycle: 1_000_000));
        var packedBf16 = new VectorType(DataTypes.BFloat16, [4, 8]);
        var broadcastN = IR.F.NTT.PackedMatMul(
            new Var("lhs_broadcast_n", new TensorType(DataTypes.BFloat16, new RankedShape(1, 1024))),
            new Var("rhs_broadcast_n", new TensorType(packedBf16, new RankedShape(96, 1024))),
            outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(broadcastN);
        var broadcastNCost = CompilerServices.EvaluateCost(broadcastN, CompileOptions);

        var splitN = IR.F.NTT.PackedMatMul(
            new Var("lhs_n_split", new TensorType(DataTypes.BFloat16, new RankedShape(1, 1024))),
            new Var("rhs_n_split", new TensorType(packedBf16, new RankedShape(12, 1024))),
            outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(splitN);
        var splitNCost = CompilerServices.EvaluateCost(splitN, CompileOptions);

        Assert.Equal((UInt128)49_152, broadcastNCost[CostFactorNames.CPUCycles]);
        Assert.Equal((UInt128)6_144, splitNCost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestNamedTargetMachineProfilesAreResolved()
    {
        var rtx = NTTTargetMachineCatalog.Resolve("rtx5060");
        var h800 = NTTTargetMachineCatalog.Resolve("h800");

        Assert.Equal(NTTTargetMachineCatalog.Rtx5060Ti16Gb, rtx.Id);
        Assert.Equal(36, rtx.Execution.ComputeUnitCount);
        Assert.Equal(8, rtx.Execution.WorkersPerBlock);
        Assert.Equal(8 * 1024, rtx.Execution.BackendPrivateAccumulatorCapacityBytes);
        Assert.Equal(16, rtx.Execution.BackendPrivateMatrixAccumulatorMinM);
        Assert.Equal(16, rtx.Execution.BackendPrivateMatrixAccumulatorMinN);
        Assert.Equal(32, rtx.Execution.BackendPrivateGemvAccumulatorMinN);
        Assert.Equal(
            [MemoryLocation.Shared, MemoryLocation.BlockLocalData],
            rtx.TilingMemorySpaces.Select(space => space.TIRBinding!.Value.Location).ToArray());
        var rtxShared = rtx.TilingMemorySpaces.Single(space => space.TIRBinding?.Location == MemoryLocation.Shared);
        Assert.Equal(32 * 1024, rtxShared.MaxAllocationBytesPerScope);
        Assert.Equal(101_376, rtx.GetMemoryResource(rtxShared).CapacityBytes);
        Assert.Equal(
            16L * 1024 * 1024 * 1024,
            rtx.GetMemoryResource(rtx.GetMemorySpace(rtx.RootMemorySpace)).CapacityBytes);
        Assert.Equal(
            rtx.GetMemorySpace(rtx.RootMemorySpace).ResourceId,
            rtx.TilingMemorySpaces.Single(space => space.TIRBinding?.Location == MemoryLocation.BlockLocalData).ResourceId);
        Assert.True(rtx.RequiresExplicitTransfer(0));
        Assert.False(rtx.RequiresExplicitTransfer(1));
        Assert.DoesNotContain(rtx.Compute.MatrixPrimitives, primitive => primitive.Name == "wgmma" && primitive.IsSupported);
        Assert.Equal(NTTTargetMachineCatalog.H800Sxm80Gb, h800.Id);
        Assert.Equal(8 * 1024, h800.Execution.BackendPrivateAccumulatorCapacityBytes);
        var h800Shared = h800.TilingMemorySpaces.Single(space => space.TIRBinding?.Location == MemoryLocation.Shared);
        Assert.Equal(64 * 1024, h800Shared.MaxAllocationBytesPerScope);
        Assert.Equal(227 * 1024, h800.GetMemoryResource(h800Shared).CapacityBytes);
        Assert.Equal(
            80L * 1024 * 1024 * 1024,
            h800.GetMemoryResource(h800.GetMemorySpace(h800.RootMemorySpace)).CapacityBytes);
        Assert.Contains(h800.Compute.MatrixPrimitives, primitive => primitive.Name == "wgmma" && primitive.IsSupported);
    }

    [Fact]
    public void TestHierarchyLatencyAggregatesChipTrafficAcrossBlocks()
    {
        var costModel = new TritonTargetOpCostModel(CreateGpuMachine(rootBytesPerCycle: 100));
        var cost = new Cost
        {
            Factors =
            {
                [CostFactorNames.BlockLocalMemoryLoadBytes] = (UInt128)100,
            },
        };
        var placement = new Placement([4, 8], "y,x", "bb");
        var distributedType = new DistributedType(new TensorType(DataTypes.Float32, new RankedShape(1)), [SBP.B], placement);

        Assert.Equal((UInt128)301, costModel.GetLatency(cost));
        Assert.Equal((UInt128)332, TargetOpCostModelUtility.GetCostLatency(costModel, cost, distributedType));
    }

    [Fact]
    public void TestLatencyUsesMachineSynchronizationCycles()
    {
        var costModel = new TritonTargetOpCostModel(CreateGpuMachine(gridSynchronizationCycles: 1500));
        var cost = new Cost
        {
            Factors =
            {
                [CostFactorNames.CPUCycles] = (UInt128)10,
                [CostFactorNames.GridSynchronization] = (UInt128)2,
            },
        };

        Assert.Equal((UInt128)3010, costModel.GetLatency(cost));
    }

    private static PyNTTTargetOptions CreateOptions(TargetMachineModel machine)
        => new() { TargetMachineModel = machine };

    private static TargetMachineModel CreateGpuMachine(
        long rootBytesPerCycle = 1_000_000,
        long blockBytesPerCycle = 512,
        double elementwiseElementsPerCycle = 128,
        double simtFmaPerCycle = 64,
        double mmaInstructionsPerCycle = 1,
        double wgmmaInstructionsPerCycle = 1,
        long gridSynchronizationCycles = 2200)
    {
        var sharedResource = new TargetMemoryResourceId("test.shared-memory");
        var globalResource = new TargetMemoryResourceId("test.global-memory");
        var shared = new TargetMemorySpaceId("test.shared");
        var blockGlobal = new TargetMemorySpaceId("test.block-global");
        var global = new TargetMemorySpaceId("test.global");
        var operandTypes = ImmutableArray.Create<DataType>(DataTypes.Float16, DataTypes.BFloat16, DataTypes.Float32, DataTypes.Int8);
        return new TargetMachineModel(
            "test-gpu",
            new(BlockExecutionKind.PersistentGpuBlock, 128, 8, 32, 1.0, 128, 4, 8 * 1024, 16, 16, 32),
            new(
                elementwiseElementsPerCycle,
                simtFmaPerCycle,
                ImmutableArray.Create(
                    new MatrixComputePrimitiveSpec("mma", 16, 8, 16, mmaInstructionsPerCycle, operandTypes),
                    new MatrixComputePrimitiveSpec("wgmma", 64, 8, 16, wgmmaInstructionsPerCycle, operandTypes))),
            new(25, gridSynchronizationCycles),
            [
                new(sharedResource, TargetMemorySpaceKind.Shared, 48 * 1024, blockBytesPerCycle, blockBytesPerCycle, 20, 16),
                new(globalResource, TargetMemorySpaceKind.Global, int.MaxValue, rootBytesPerCycle, rootBytesPerCycle, 300, 128),
            ],
            [
                new(shared, sharedResource, MemorySharingScope.Block, new(MemoryLocation.Shared), 48 * 1024, TargetMemoryAllocationSizePolicy.PowerOfTwo, true, 0, true, true, true),
                new(blockGlobal, globalResource, MemorySharingScope.Block, new(MemoryLocation.BlockLocalData), 64 * 1024 * 1024, TargetMemoryAllocationSizePolicy.GranularityAligned, true, 1, true, true, true),
                new(global, globalResource, MemorySharingScope.Chip, null, int.MaxValue, TargetMemoryAllocationSizePolicy.GranularityAligned, false, -1, true, true, false),
            ],
            global,
            [
                new(global, blockGlobal, rootBytesPerCycle, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, global, rootBytesPerCycle, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, shared, blockBytesPerCycle, 300, TargetMemoryTransferMode.ExplicitCopy),
                new(shared, blockGlobal, blockBytesPerCycle, 300, TargetMemoryTransferMode.ExplicitCopy),
            ],
            new Dictionary<MemoryLocation, TargetMemorySpaceId>
            {
                [MemoryLocation.BlockLocalData] = blockGlobal,
            });
    }
}
