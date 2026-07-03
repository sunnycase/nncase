// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CostModelTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTritonTargetCostModel : TestClassBase
{
    [Fact]
    public void TestDynamicMatMulCostUsesMaxShape()
    {
        var capability = TritonTargetCapability.ForComputeCapability(8, 0) with
        {
            MultiprocessorCount = 128,
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1_000_000_000.0,
            Mma = new TritonDotInstructionCapability("mma", 16, 8, 16, 1.0),
            Wgmma = new TritonDotInstructionCapability("wgmma", 64, 8, 16, 1.0, isSupported: false),
        };
        CompileOptions.TargetOptions = new PyNTTTargetOptions { TritonCapability = capability };

        var seq = new DimVar("seq").With(range: new ValueRange<double>(1, 20));
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape((Dimension)1, seq, (Dimension)64)));
        var rhs = new Var("rhs", new TensorType(DataTypes.BFloat16, new RankedShape((Dimension)64, (Dimension)128)));
        var matmul = IR.F.Tensors.MatMul(lhs, rhs);
        CompilerServices.InferenceType(matmul);

        var cost = CompilerServices.EvaluateCost(matmul, CompileOptions);

        Assert.Equal((UInt128)163840, cost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestScalarMatMulCostDoesNotUseDotInstructions()
    {
        var capability = TritonTargetCapability.ForComputeCapability(9, 0) with
        {
            MultiprocessorCount = 128,
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1_000_000_000.0,
            Mma = new TritonDotInstructionCapability("mma", 16, 8, 16, 1.0),
            Wgmma = new TritonDotInstructionCapability("wgmma", 64, 8, 16, 4.0),
        };
        var costModel = new TritonTargetOpCostModel(capability);
        var query = new MatMulOpCostQuery(
            new TargetCostTensor(DataTypes.BFloat16, new RankedShape(64, 64)),
            new TargetCostTensor(DataTypes.BFloat16, new RankedShape(64, 128)),
            new TargetCostTensor(DataTypes.BFloat16, new RankedShape(64, 128)),
            DataTypes.BFloat16);

        Assert.True(costModel.TryGetMatMulCost(query, out var cost));
        Assert.Equal((UInt128)524288, cost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestMatMulCostCanSelectWgmma()
    {
        var capability = TritonTargetCapability.ForComputeCapability(9, 0) with
        {
            MultiprocessorCount = 128,
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1_000_000_000.0,
            Mma = new TritonDotInstructionCapability("mma", 16, 8, 16, 1.0),
            Wgmma = new TritonDotInstructionCapability("wgmma", 64, 8, 16, 4.0),
        };
        var costModel = new TritonTargetOpCostModel(capability);
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
    public void TestVectorizedMatMulCostUsesDotInstructionsFromVectorDType()
    {
        var capability = TritonTargetCapability.ForComputeCapability(9, 0) with
        {
            MultiprocessorCount = 128,
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1_000_000_000.0,
            Mma = new TritonDotInstructionCapability("mma", 16, 8, 16, 1.0),
            Wgmma = new TritonDotInstructionCapability("wgmma", 64, 8, 16, 4.0),
        };
        CompileOptions.TargetOptions = new PyNTTTargetOptions { TritonCapability = capability };

        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(64, 64)));
        var rhs = new Var("rhs", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(64, 16)));
        var matmul = IR.F.NTT.VectorizedMatMul(lhs, rhs, [], [1], outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(matmul);

        var cost = CompilerServices.EvaluateCost(matmul, CompileOptions);

        Assert.Equal((UInt128)16, cost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestElementwiseCostCountsVectorLanes()
    {
        var capability = TritonTargetCapability.ForComputeCapability(8, 0) with
        {
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1_000_000_000.0,
            GlobalMemoryElementsPerCyclePerCta = 128.0,
            ElementwiseElementsPerCyclePerCta = 128.0,
        };
        var costModel = new TritonTargetOpCostModel(capability);
        var dtype = new VectorType(DataTypes.BFloat16, [8]);
        var tensor = new TargetCostTensor(dtype, new RankedShape(128));
        var query = new ElementwiseOpCostQuery("vectorized_cast", [tensor], tensor);

        Assert.True(costModel.TryGetElementwiseCost(query, out var cost));
        Assert.Equal((UInt128)8, cost[CostFactorNames.CPUCycles]);
        Assert.Equal((UInt128)1024, cost[CostFactorNames.MemoryLoad]);
        Assert.Equal((UInt128)1024, cost[CostFactorNames.MemoryStore]);
        Assert.Equal((UInt128)16, costModel.GetLatency(cost));
    }

    [Fact]
    public void TestBoxingTensorStoreUsesLocalTargetCopyCostWithoutSynchronization()
    {
        var capability = TritonTargetCapability.ForComputeCapability(8, 0) with
        {
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1000.0,
            GlobalMemoryElementsPerCyclePerCta = 250.0,
        };
        CompileOptions.TargetOptions = new PyNTTTargetOptions { TritonCapability = capability };
        var costModel = new TritonTargetOpCostModel(capability);

        var placement = new Placement([4, 8], "y,x", "bb");
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(128, 151936));
        var distributedType = new DistributedType(tensorType, [SBP.S([0]), SBP.S([1])], placement);
        var input = new Var("input", distributedType);
        var boxing = IR.F.Distributed.Boxing(input, tensorType);
        CompilerServices.InferenceType(boxing);

        var cost = CompilerServices.EvaluateCost(boxing, CompileOptions);

        Assert.False(cost.Factors.ContainsKey(CostFactorNames.CPUCycles));
        Assert.Equal((UInt128)607744, cost[CostFactorNames.MemoryLoad]);
        Assert.Equal((UInt128)607744, cost[CostFactorNames.MemoryStore]);
        Assert.False(cost.Factors.ContainsKey(CostFactorNames.Synchronization));
        Assert.Equal((UInt128)4862, costModel.GetLatency(cost));
    }

    [Fact]
    public void TestBoxingTensorLoadUsesLocalTargetCopyCostWithoutSynchronization()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();

        var placement = new Placement([4, 8], "y,x", "bb");
        var tensorType = new TensorType(DataTypes.BFloat16, new RankedShape(16, 128));
        var distributedType = new DistributedType(tensorType, [SBP.B, SBP.S([1])], placement);
        var input = new Var("input", tensorType);
        var boxing = IR.F.Distributed.Boxing(input, distributedType);
        CompilerServices.InferenceType(boxing);

        var cost = CompilerServices.EvaluateCost(boxing, CompileOptions);

        Assert.True(cost[CostFactorNames.MemoryLoad] > 0);
        Assert.True(cost[CostFactorNames.MemoryStore] > 0);
        Assert.False(cost.Factors.ContainsKey(CostFactorNames.Synchronization));
    }

    [Fact]
    public void TestSmallReshardTargetCostIncludesSynchronization()
    {
        CompileOptions.TargetOptions = new PyNTTTargetOptions();

        var placement = new Placement([4, 8], "y,x", "bb");
        var tensorType = new TensorType(DataTypes.BFloat16, new RankedShape(16, 128));
        var inputType = new DistributedType(tensorType, [SBP.B, SBP.S([1])], placement);
        var outputType = new DistributedType(tensorType, [SBP.B, SBP.B], placement);
        var input = new Var("input", inputType);
        var boxing = IR.F.Distributed.Boxing(input, outputType);
        CompilerServices.InferenceType(boxing);

        var cost = CompilerServices.EvaluateCost(boxing, CompileOptions);

        Assert.True(cost[CostFactorNames.MemoryLoad] > 0);
        Assert.True(cost[CostFactorNames.MemoryStore] > 0);
        Assert.Equal((UInt128)1, cost[CostFactorNames.Synchronization]);
    }

    [Fact]
    public void TestPackedMatMulCostUsesLocalSimtModel()
    {
        var capability = TritonTargetCapability.ForComputeCapability(8, 0) with
        {
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1_000_000_000.0,
            GlobalMemoryElementsPerCyclePerCta = 1_000_000.0,
            SimtFmaPerCyclePerCta = 64.0,
            Mma = new TritonDotInstructionCapability("mma", 16, 8, 16, 1.0),
            Wgmma = new TritonDotInstructionCapability("wgmma", 64, 8, 16, 1.0, isSupported: false),
        };
        CompileOptions.TargetOptions = new PyNTTTargetOptions { TritonCapability = capability };

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
        Assert.True(broadcastNCost[CostFactorNames.CPUCycles] > splitNCost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestDistributedPackedMatMulCostUsesLocalShardShape()
    {
        var capability = TritonTargetCapability.ForComputeCapability(8, 0) with
        {
            ClockRateGHz = 1.0,
            GlobalMemoryBandwidthGBps = 1_000_000_000.0,
            GlobalMemoryElementsPerCyclePerCta = 1_000_000.0,
            SimtFmaPerCyclePerCta = 64.0,
            Mma = new TritonDotInstructionCapability("mma", 16, 8, 16, 1.0),
            Wgmma = new TritonDotInstructionCapability("wgmma", 64, 8, 16, 1.0, isSupported: false),
        };
        CompileOptions.TargetOptions = new PyNTTTargetOptions { TritonCapability = capability };

        var placement = new Placement([4, 8], "y,x", "bb");
        var packedBf16 = new VectorType(DataTypes.BFloat16, [4, 8]);
        var broadcastN = IR.F.NTT.PackedMatMul(
            new Var("lhs_broadcast_n", new DistributedType(new TensorType(DataTypes.BFloat16, new RankedShape(1, 1024)), [SBP.B, SBP.B], placement)),
            new Var("rhs_broadcast_n", new DistributedType(new TensorType(packedBf16, new RankedShape(96, 1024)), [SBP.B, SBP.B], placement)),
            outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(broadcastN);
        var broadcastNCost = CompilerServices.EvaluateCost(broadcastN, CompileOptions);

        var splitN = IR.F.NTT.PackedMatMul(
            new Var("lhs_n_split", new DistributedType(new TensorType(DataTypes.BFloat16, new RankedShape(1, 1024)), [SBP.B, SBP.B], placement)),
            new Var("rhs_n_split", new DistributedType(new TensorType(packedBf16, new RankedShape(96, 1024)), [SBP.S([0, 1]), SBP.B], placement)),
            outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(splitN);
        var splitNCost = CompilerServices.EvaluateCost(splitN, CompileOptions);

        Assert.True(broadcastNCost[CostFactorNames.CPUCycles] > splitNCost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public void TestPyNTTDefaultTargetCostModelIsTriton()
    {
        var options = new PyNTTTargetOptions();

        var costModel = Assert.IsType<TritonTargetOpCostModel>(options.TargetCostModel);
        Assert.Equal(TritonTargetCapability.Default, costModel.Capability);
        Assert.Equal(2200.0, costModel.Capability.SynchronizationCyclesPerEvent);
    }

    [Fact]
    public void TestParseTritonCapability()
    {
        var capability = TritonTargetCapability.Parse("cc=90,num_sms=120,clock_ghz=2.0,mem_bw_gbps=2500,memory_epc=256,sync_us=2,mma=16x8x16,mma_ipc=2,wgmma=64x16x16,wgmma_ipc=8");

        Assert.Equal(9, capability.ComputeCapabilityMajor);
        Assert.Equal(0, capability.ComputeCapabilityMinor);
        Assert.Equal(120, capability.MultiprocessorCount);
        Assert.Equal(2.0, capability.ClockRateGHz);
        Assert.Equal(2500.0, capability.GlobalMemoryBandwidthGBps);
        Assert.Equal(256.0, capability.GlobalMemoryElementsPerCyclePerCta);
        Assert.Equal(4000.0, capability.SynchronizationCyclesPerEvent);
        Assert.Equal(2.0, capability.SynchronizationLatencyUs);
        Assert.Equal(16, capability.Mma.M);
        Assert.Equal(8, capability.Mma.N);
        Assert.Equal(16, capability.Mma.K);
        Assert.Equal(2.0, capability.Mma.InstructionsPerCyclePerCta);
        Assert.Equal(64, capability.Wgmma.M);
        Assert.Equal(16, capability.Wgmma.N);
        Assert.Equal(16, capability.Wgmma.K);
        Assert.Equal(8.0, capability.Wgmma.InstructionsPerCyclePerCta);
        Assert.True(capability.Wgmma.IsSupported);
    }

    [Fact]
    public void TestLatencyCanDeriveMemoryElementsPerCycle()
    {
        var capability = TritonTargetCapability.Parse("cc=80,clock_ghz=2.0,mem_bw_gbps=1000,memory_epc=0,memory_element_bytes=4,memory_efficiency=0.5");
        var costModel = new TritonTargetOpCostModel(capability);
        var cost = new Cost
        {
            Factors =
            {
                [CostFactorNames.CPUCycles] = (UInt128)1,
                [CostFactorNames.MemoryLoad] = (UInt128)125,
            },
        };

        Assert.Equal(0.0, capability.GlobalMemoryElementsPerCyclePerCta);
        Assert.Equal(62.5, capability.EffectiveGlobalMemoryElementsPerCyclePerCta, precision: 1);
        Assert.Equal((UInt128)2, costModel.GetLatency(cost));
    }

    [Fact]
    public void TestLatencyConvertsSynchronizationEventsWithTritonCapability()
    {
        var capability = TritonTargetCapability.Parse("cc=80,clock_ghz=1.0,memory_epc=1024,sync_us=1.5");
        var costModel = new TritonTargetOpCostModel(capability);
        var cost = new Cost
        {
            Factors =
            {
                [CostFactorNames.CPUCycles] = (UInt128)10,
                [CostFactorNames.Synchronization] = (UInt128)2,
            },
        };

        Assert.Equal((UInt128)3010, costModel.GetLatency(cost));
    }
}
