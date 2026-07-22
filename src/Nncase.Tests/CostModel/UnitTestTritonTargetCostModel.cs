// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Schedule;
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

        Assert.Equal((UInt128)163_840, cost[CostFactorNames.CPUCycles]);
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
        Assert.Equal((UInt128)524_288, cost[CostFactorNames.CPUCycles]);
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
        Assert.Equal((UInt128)264, cost[CostFactorNames.CPUCycles]);
    }

    [Fact]
    public async Task TestAutoVectorizeExtractsVectorizedMatMul()
    {
        CompileOptions.TargetOptions = CreateOptions(CreateGpuMachine(rootBytesPerCycle: 174));
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(1, 3072)));
        var rhs = new Var("rhs", new TensorType(DataTypes.BFloat16, new RankedShape(3072, 1024)));
        var function = new Function("main", IR.F.Tensors.MatMul(lhs, rhs), lhs, rhs);
        Assert.True(function.InferenceType());

        var module = new IRModule(function);
        var passManager = CompileSession.CreatePassManager("AutoVectorizeMatMulCost");
        passManager.AddWithName<EGraphRulesPass>("AutoVectorize").Configure(pass =>
        {
            pass.Add<Nncase.Passes.Rules.NTT.VectorizeMatMul>(1, 16);
        });

        await passManager.RunAsync(module);

        var post = Assert.IsType<Function>(module.Entry);
        Assert.Contains(
            ExprCollector.Collect(post.Body).OfType<Call>(),
            call => call.Target is Nncase.IR.NTT.VectorizedMatMul);
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
        var rtxRegisters = rtx.GetPrivateResource(NTTTargetMachineCatalog.GpuRegisterFile);
        Assert.Equal(TargetPrivateResourceUnit.Register32, rtxRegisters.Unit);
        Assert.Equal(255L * 8 * 32, rtxRegisters.CapacityUnits);
        var rtxBackendShared = rtx.GetPrivateResource(NTTTargetMachineCatalog.GpuBackendSharedMemory);
        Assert.Equal(TargetPrivateResourceUnit.Bytes, rtxBackendShared.Unit);
        Assert.Equal(101_376, rtxBackendShared.CapacityUnits);
        Assert.Equal(
            [MemoryLocation.Shared, MemoryLocation.BlockLocalData],
            rtx.TilingMemorySpaces.Select(space => space.TIRBinding!.Value.Location).ToArray());
        var rtxShared = rtx.TilingMemorySpaces.Single(space => space.TIRBinding?.Location == MemoryLocation.Shared);
        Assert.Equal(64 * 1024, rtxShared.MaxAllocationBytesPerScope);
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
        var rtxMma = Assert.Single(rtx.Compute.MatrixPrimitives, primitive => primitive.Name == "mma");
        Assert.Equal(29, rtxMma.DependencyLatencyCycles);
        Assert.Equal(23, rtxMma.ReciprocalThroughputCyclesPerWorker);
        Assert.Equal(0.25, rtxMma.MaxInstructionsPerCyclePerBlock);
        Assert.Equal(1, rtxMma.CooperativeWorkers);
        Assert.Equal(
            537,
            MatrixComputeCostModel.EstimateCycles(
                rtxMma,
                accumulatorChains: 4,
                dependentInstructionsPerChain: 16,
                rtx.Execution));
        var rtxAsyncTransfer = rtx.GetTransfer(
            rtx.GetTilingParentMemorySpace(rtxShared.TilingLevel).Id,
            rtxShared.Id).Asynchronous;
        Assert.Equal(new[] { 2 }, Assert.IsType<TargetAsynchronousTransferSpec>(rtxAsyncTransfer).SupportedStageCounts);
        Assert.Equal(NTTTargetMachineCatalog.H800Sxm80Gb, h800.Id);
        Assert.Equal(255L * 8 * 32, h800.GetPrivateResource(NTTTargetMachineCatalog.GpuRegisterFile).CapacityUnits);
        var h800Shared = h800.TilingMemorySpaces.Single(space => space.TIRBinding?.Location == MemoryLocation.Shared);
        Assert.Equal(128 * 1024, h800Shared.MaxAllocationBytesPerScope);
        Assert.Equal(227 * 1024, h800.GetMemoryResource(h800Shared).CapacityBytes);
        Assert.Equal(
            80L * 1024 * 1024 * 1024,
            h800.GetMemoryResource(h800.GetMemorySpace(h800.RootMemorySpace)).CapacityBytes);
        Assert.Contains(h800.Compute.MatrixPrimitives, primitive => primitive.Name == "wgmma" && primitive.IsSupported);
        var h800AsyncTransfer = h800.GetTransfer(
            h800.GetTilingParentMemorySpace(h800Shared.TilingLevel).Id,
            h800Shared.Id).Asynchronous;
        Assert.Equal(new[] { 2 }, Assert.IsType<TargetAsynchronousTransferSpec>(h800AsyncTransfer).SupportedStageCounts);
    }

    [Fact]
    public void TestMatrixComputeCostUsesLatencyAndThroughputBounds()
    {
        var machine = NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb);
        var primitive = Assert.Single(machine.Compute.MatrixPrimitives);
        var singleWorker = machine.Execution with { WorkersPerBlock = 1 };

        Assert.Equal(
            464,
            MatrixComputeCostModel.EstimateCycles(
                primitive,
                accumulatorChains: 1,
                dependentInstructionsPerChain: 16,
                singleWorker));
        Assert.Equal(
            98,
            MatrixComputeCostModel.EstimateCycles(
                primitive,
                accumulatorChains: 4,
                dependentInstructionsPerChain: 1,
                singleWorker));
        Assert.Equal(
            57,
            MatrixComputeCostModel.EstimateCycles(
                primitive,
                accumulatorChains: 8,
                dependentInstructionsPerChain: 1,
                machine.Execution));
    }

    [Fact]
    public void TestTritonGemvMicroKernelCandidatesModelRegisterState()
    {
        var solver = new Google.OrTools.ConstraintSolver.Solver("triton-gemv-microkernels");
        var op = new Nncase.TIR.NTT.Matmul(
            new IRArray<int>(),
            new IRArray<int>(),
            false,
            true,
            false,
            null,
            null);
        var workload = new MatrixTileWorkload(
            static (shapes, localSolver, _) => new(
                shapes[2][0],
                shapes[2][1],
                shapes[0][1],
                localSolver.MakeIntConst(1)),
            DataTypes.Float32.SizeInBytes);
        var fullShapes = new long[][]
        {
            [1, 1024],
            [2048, 1024],
            [1, 2048],
        };
        var context = new TileWorkloadContext(
            op,
            fullShapes.Select(shape => shape.ToImmutableArray()).ToImmutableArray(),
            ImmutableArray.Create<DataType>(DataTypes.BFloat16, DataTypes.BFloat16, DataTypes.BFloat16));
        var localShapes = new Google.OrTools.ConstraintSolver.IntExpr[][]
        {
            [solver.MakeIntConst(1), solver.MakeIntConst(4)],
            [solver.MakeIntConst(2048), solver.MakeIntConst(4)],
            [solver.MakeIntConst(1), solver.MakeIntConst(2048)],
        };
        var symbolicFullShapes = fullShapes
            .Select(shape => shape.Select(extent => (Google.OrTools.ConstraintSolver.IntExpr)solver.MakeIntConst(extent)).ToArray())
            .ToArray();
        var machine = NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb);

        var candidates = new TritonBlockMicroKernelModel().GetCandidates(new(
            op,
            workload,
            context,
            localShapes,
            symbolicFullShapes,
            solver.MakeIntConst(100),
            machine,
            solver));

        var register = Assert.Single(candidates, candidate => IsCandidate(candidate, "register_simt_accumulator"));
        var registerMma = Assert.Single(candidates, candidate => IsCandidate(candidate, "register_mma_accumulator"));
        Assert.Equal(2, candidates.Count);
        Assert.DoesNotContain(candidates, candidate => candidate.Variant.Contains("shared", StringComparison.Ordinal));
        Assert.Equal(1, register.IsLegal.Var().Max());
        Assert.Equal(0, registerMma.IsLegal.Var().Max());
        Assert.Contains(register.Resources, usage => usage.Resource == NTTTargetMachineCatalog.GpuRegisterFile);
        Assert.Equal(
            (2048 / machine.Execution.WorkerWidth) *
                machine.Execution.ThreadsPerBlock * DataTypes.Float32.SizeInBytes,
            register.Resources.Single(usage =>
                usage.Resource == NTTTargetMachineCatalog.GpuBackendSharedMemory).Units.Var().Min());
        Assert.Equal(
            TritonBlockMicroKernelContract.Version,
            register.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.VersionParameter).Value.Var().Max());
        Assert.DoesNotContain(
            register.Parameters,
            parameter => parameter.Name.StartsWith("simt_", StringComparison.Ordinal));
        var mmaLocalShapes = new Google.OrTools.ConstraintSolver.IntExpr[][]
        {
            [solver.MakeIntConst(1), solver.MakeIntConst(128)],
            [solver.MakeIntConst(128), solver.MakeIntConst(128)],
            [solver.MakeIntConst(1), solver.MakeIntConst(128)],
        };
        var mmaCandidates = new TritonBlockMicroKernelModel().GetCandidates(new(
            op,
            workload,
            context,
            mmaLocalShapes,
            symbolicFullShapes,
            solver.MakeIntConst(100),
            NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb),
            solver));
        var legalRegisterMma = Assert.Single(mmaCandidates, candidate => IsCandidate(candidate, "register_mma_accumulator"));
        var legalSimt = Assert.Single(mmaCandidates, candidate => IsCandidate(candidate, "register_simt_accumulator"));
        Assert.Equal(2, mmaCandidates.Count);
        Assert.DoesNotContain(mmaCandidates, candidate => candidate.Variant.Contains("shared", StringComparison.Ordinal));
        Assert.Equal(1, legalRegisterMma.IsLegal.Var().Min());
        Assert.Equal(8, legalRegisterMma.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.InnerMParameter).Value.Var().Min());
        Assert.Equal(16, legalRegisterMma.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.MmaMParameter).Value.Var().Min());
        Assert.Equal(128, legalRegisterMma.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.InnerNParameter).Value.Var().Min());
        Assert.True(
            legalRegisterMma.ExecutionCost.RegionCycles.Var().Max() <
            legalSimt.ExecutionCost.RegionCycles.Var().Min(),
            $"Expected MMA cycles {legalRegisterMma.ExecutionCost.RegionCycles.Var().Max()} " +
            $"to be lower than SIMT cycles {legalSimt.ExecutionCost.RegionCycles.Var().Min()}.");
        Assert.DoesNotContain(
            legalRegisterMma.Resources,
            usage => usage.Resource == NTTTargetMachineCatalog.GpuBackendSharedMemory);
        Assert.Equal(2, legalRegisterMma.BufferEncodingRequirements.Length);
        Assert.All(
            legalRegisterMma.BufferEncodingRequirements,
            requirement =>
            {
                Assert.Equal("gpu.shared", requirement.MemorySpace.Value);
                Assert.Equal(
                    TritonTargetStorageEncodingModel.NvidiaMmaShared,
                    Assert.Single(requirement.AcceptedEncodings));
            });

        var n32LocalShapes = new Google.OrTools.ConstraintSolver.IntExpr[][]
        {
            [solver.MakeIntConst(1), solver.MakeIntConst(128)],
            [solver.MakeIntConst(32), solver.MakeIntConst(128)],
            [solver.MakeIntConst(1), solver.MakeIntConst(32)],
        };
        var n32Candidates = new TritonBlockMicroKernelModel().GetCandidates(new(
            op,
            workload,
            context,
            n32LocalShapes,
            symbolicFullShapes,
            solver.MakeIntConst(100),
            NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb),
            solver));
        var n32Mma = Assert.Single(n32Candidates, candidate => IsCandidate(candidate, "register_mma_accumulator"));
        var n32Simt = Assert.Single(n32Candidates, candidate => IsCandidate(candidate, "register_simt_accumulator"));
        Assert.Equal(1, n32Mma.IsLegal.Var().Min());
        Assert.True(
            n32Mma.ExecutionCost.RegionCycles.Var().Min() >
            n32Simt.ExecutionCost.RegionCycles.Var().Max(),
            $"Expected narrow-N MMA cycles {n32Mma.ExecutionCost.RegionCycles.Var().Min()} " +
            $"to exceed SIMT cycles {n32Simt.ExecutionCost.RegionCycles.Var().Max()}.");
        Assert.DoesNotContain(
            n32Mma.Resources,
            usage => usage.Resource == NTTTargetMachineCatalog.GpuBackendSharedMemory);

        var n32LhsReads = n32Mma.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 0 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        var n128LhsReads = legalRegisterMma.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 0 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        var n32RhsReads = n32Mma.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 1 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        var n128RhsReads = legalRegisterMma.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 1 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        Assert.Equal(4 * n128LhsReads, n32LhsReads);
        Assert.Equal(n128RhsReads, n32RhsReads);

        var k256LocalShapes = new Google.OrTools.ConstraintSolver.IntExpr[][]
        {
            [solver.MakeIntConst(1), solver.MakeIntConst(256)],
            [solver.MakeIntConst(128), solver.MakeIntConst(256)],
            [solver.MakeIntConst(1), solver.MakeIntConst(128)],
        };
        var k256Candidates = new TritonBlockMicroKernelModel().GetCandidates(new(
            op,
            workload,
            context,
            k256LocalShapes,
            symbolicFullShapes,
            solver.MakeIntConst(100),
            NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb),
            solver));
        var k256Mma = Assert.Single(k256Candidates, candidate => IsCandidate(candidate, "register_mma_accumulator"));
        var k128Registers = legalRegisterMma.Resources.Single(usage => usage.Resource == NTTTargetMachineCatalog.GpuRegisterFile).Units.Var().Max();
        var k256Registers = k256Mma.Resources.Single(usage => usage.Resource == NTTTargetMachineCatalog.GpuRegisterFile).Units.Var().Max();
        Assert.Equal(k128Registers, k256Registers);

        var equalWorkN32K256 = new Google.OrTools.ConstraintSolver.IntExpr[][]
        {
            [solver.MakeIntConst(1), solver.MakeIntConst(256)],
            [solver.MakeIntConst(32), solver.MakeIntConst(256)],
            [solver.MakeIntConst(1), solver.MakeIntConst(32)],
        };
        var equalWorkN256K32 = new Google.OrTools.ConstraintSolver.IntExpr[][]
        {
            [solver.MakeIntConst(1), solver.MakeIntConst(32)],
            [solver.MakeIntConst(256), solver.MakeIntConst(32)],
            [solver.MakeIntConst(1), solver.MakeIntConst(256)],
        };
        var equalWorkN32 = Assert.Single(
            new TritonBlockMicroKernelModel().GetCandidates(new(
                op,
                workload,
                context,
                equalWorkN32K256,
                symbolicFullShapes,
                solver.MakeIntConst(100),
                NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb),
                solver)),
            candidate => IsCandidate(candidate, "register_simt_accumulator"));
        var equalWorkN256 = Assert.Single(
            new TritonBlockMicroKernelModel().GetCandidates(new(
                op,
                workload,
                context,
                equalWorkN256K32,
                symbolicFullShapes,
                solver.MakeIntConst(100),
                NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb),
                solver)),
            candidate => IsCandidate(candidate, "register_simt_accumulator"));
        Assert.Equal(
            equalWorkN32.ExecutionCost.RegionCycles.Var().Min(),
            equalWorkN256.ExecutionCost.RegionCycles.Var().Max());
        Assert.Equal(
            256,
            equalWorkN32.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.InnerKParameter).Value.Var().Max());
        Assert.Equal(
            32,
            equalWorkN256.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.InnerKParameter).Value.Var().Max());
        var equalWorkN32Registers = equalWorkN32.Resources.Single(
            usage => usage.Resource == NTTTargetMachineCatalog.GpuRegisterFile).Units.Var().Max();
        Assert.Equal(
            (56 * 8 * 32) + 256 + (32 * 256 * 2) + 32,
            equalWorkN32Registers);
        var equalWorkN32LhsReads = equalWorkN32.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 0 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        var equalWorkN256LhsReads = equalWorkN256.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 0 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        var equalWorkN32RhsReads = equalWorkN32.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 1 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        var equalWorkN256RhsReads = equalWorkN256.MemoryAccesses.Single(access =>
            access.BufferAccessIndex == 1 && access.Mode == MemoryAccessMode.Read).Bytes.Var().Max();
        Assert.Equal(8 * equalWorkN256LhsReads, equalWorkN32LhsReads);
        Assert.Equal(equalWorkN32RhsReads, equalWorkN256RhsReads);

        var n4LocalShapes = new Google.OrTools.ConstraintSolver.IntExpr[][]
        {
            [solver.MakeIntConst(1), solver.MakeIntConst(32)],
            [solver.MakeIntConst(4), solver.MakeIntConst(32)],
            [solver.MakeIntConst(1), solver.MakeIntConst(4)],
        };
        var n4Register = Assert.Single(
            new TritonBlockMicroKernelModel().GetCandidates(new(
                op,
                workload,
                context,
                n4LocalShapes,
                symbolicFullShapes,
                solver.MakeIntConst(100),
                NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb),
                solver)),
            candidate => IsCandidate(candidate, "register_simt_accumulator"));
        Assert.Equal(
            32,
            n4Register.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.StateBlockNParameter).Value.Var().Max());
        Assert.Equal(
            32,
            n4Register.Parameters.Single(parameter => parameter.Name == TritonBlockMicroKernelContract.InnerNParameter).Value.Var().Max());

        var packedOp = new Nncase.TIR.NTT.PackedMatMul(false);
        var packedContext = new TileWorkloadContext(
            packedOp,
            fullShapes.Select(shape => shape.ToImmutableArray()).ToImmutableArray(),
            ImmutableArray.Create<DataType>(
                DataTypes.BFloat16,
                new VectorType(DataTypes.BFloat16, 4, 8),
                new VectorType(DataTypes.BFloat16, 4, 8)));
        var packedCandidates = new TritonBlockMicroKernelModel().GetCandidates(new(
            packedOp,
            workload,
            packedContext,
            n4LocalShapes,
            symbolicFullShapes,
            solver.MakeIntConst(100),
            NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb),
            solver));
        Assert.All(
            packedCandidates,
            candidate =>
            {
                Assert.Equal(
                    new[] { 0, 1 },
                    candidate.BufferEncodingRequirements
                        .Select(requirement => requirement.BufferAccessIndex)
                        .Order()
                        .ToArray());
                Assert.All(
                    candidate.BufferEncodingRequirements,
                    requirement => Assert.Equal("gpu.shared", requirement.MemorySpace.Value));
            });
        var packedRegister = Assert.Single(
            packedCandidates,
            candidate => IsCandidate(candidate, "register_simt_accumulator"));
        Assert.Equal(2, packedRegister.BufferEncodingRequirements.Length);
        var packedLhsRequirement = Assert.Single(
            packedRegister.BufferEncodingRequirements,
            requirement => requirement.BufferAccessIndex == 0);
        Assert.Equal("gpu.shared", packedLhsRequirement.MemorySpace.Value);
        Assert.Equal(
            TritonTargetStorageEncodingModel.SwizzledShared,
            Assert.Single(packedLhsRequirement.AcceptedEncodings));
        var packedRhsRequirement = Assert.Single(
            packedRegister.BufferEncodingRequirements,
            requirement => requirement.BufferAccessIndex == 1);
        Assert.Equal(1, packedRhsRequirement.BufferAccessIndex);
        Assert.Equal("gpu.shared", packedRhsRequirement.MemorySpace.Value);
        Assert.Equal(
            TritonTargetStorageEncodingModel.KMajorPackedN,
            Assert.Single(packedRhsRequirement.AcceptedEncodings));
    }

    [Fact]
    public void TestTritonStorageEncodingModelExposesMmaSharedOnlyAtSharedLevel()
    {
        using var solver = new Google.OrTools.ConstraintSolver.Solver("storage_encoding");
        var machine = NTTTargetMachineCatalog.Resolve(NTTTargetMachineCatalog.Rtx5060Ti16Gb);
        var model = new TritonTargetStorageEncodingModel();
        var shared = machine.TilingMemorySpaces.Single(space =>
            machine.GetMemoryResource(space).Kind == TargetMemorySpaceKind.Shared);
        var sharedCandidates = model.GetCandidates(new(
            shared,
            DataTypes.BFloat16,
            [solver.MakeIntConst(16), solver.MakeIntConst(16)],
            solver.MakeIntConst(512),
            machine,
            solver));

        var mma = Assert.Single(
            sharedCandidates,
            candidate => candidate.Id == TritonTargetStorageEncodingModel.NvidiaMmaShared);
        Assert.Equal(1, mma.IsLegal.Var().Min());
        Assert.Equal(512, mma.PhysicalBytes.Var().Min());
        Assert.Equal(16, mma.AlignmentBytes);
        Assert.Equal(2, mma.SelectionPriority);
        var swizzled = Assert.Single(
            sharedCandidates,
            candidate => candidate.Id == TritonTargetStorageEncodingModel.SwizzledShared);
        Assert.Equal(0, swizzled.SelectionPriority);
        Assert.All(sharedCandidates, candidate => Assert.Equal(0, candidate.EstimatedCycles.Var().Min()));
        Assert.DoesNotContain(
            sharedCandidates,
            candidate => candidate.Id == TritonTargetStorageEncodingModel.KMajorPackedN);

        var paddedSharedCandidates = model.GetCandidates(new(
            shared,
            new VectorType(DataTypes.BFloat16, 4, 8),
            [solver.MakeIntConst(1), solver.MakeIntConst(149)],
            solver.MakeIntConst(9536),
            machine,
            solver));
        Assert.All(
            paddedSharedCandidates,
            candidate => Assert.Equal(16384, candidate.PhysicalBytes.Var().Min()));

        var stagedSharedCandidates = model.GetCandidates(new TargetStorageEncodingModelContext(
            shared,
            new VectorType(DataTypes.BFloat16, 4, 8),
            [solver.MakeIntConst(1), solver.MakeIntConst(149)],
            solver.MakeIntConst(9536),
            machine,
            solver)
        {
            StagedAllocation = new StagedAllocationContext("matrix-inputs", solver.MakeIntConst(2)),
        });
        Assert.All(
            stagedSharedCandidates,
            candidate =>
            {
                Assert.NotNull(candidate.StageStrideBytes);
                Assert.Equal(16384, candidate.StageStrideBytes!.Var().Min());
                Assert.Equal(16384, candidate.StageStrideBytes.Var().Max());
            });
        var packed = Assert.Single(
            paddedSharedCandidates,
            candidate => candidate.Id == TritonTargetStorageEncodingModel.KMajorPackedN);
        Assert.Equal(1, packed.IsLegal.Var().Min());
        Assert.Equal(1, packed.SelectionPriority);

        var tinyStagedSharedCandidates = model.GetCandidates(new TargetStorageEncodingModelContext(
            shared,
            DataTypes.BFloat16,
            [solver.MakeIntConst(4)],
            solver.MakeIntConst(8),
            machine,
            solver)
        {
            StagedAllocation = new StagedAllocationContext("tiny-input", solver.MakeIntConst(2)),
        });
        Assert.All(
            tinyStagedSharedCandidates,
            candidate =>
            {
                Assert.Equal(16, candidate.PhysicalBytes.Var().Min());
                Assert.Equal(16, candidate.StageStrideBytes!.Var().Min());
                Assert.Equal(0, candidate.StageStrideBytes.Var().Min() % candidate.AlignmentBytes);
            });

        var groupedPackedCandidates = model.GetCandidates(new(
            shared,
            new VectorType(DataTypes.BFloat16, 4, 8),
            [solver.MakeIntConst(2), solver.MakeIntConst(149)],
            solver.MakeIntConst(19072),
            machine,
            solver));
        var groupedPacked = Assert.Single(
            groupedPackedCandidates,
            candidate => candidate.Id == TritonTargetStorageEncodingModel.KMajorPackedN);
        Assert.Equal(1, groupedPacked.IsLegal.Var().Min());

        var blockGlobal = machine.TilingMemorySpaces.Single(space =>
            machine.GetMemoryResource(space).Kind == TargetMemorySpaceKind.Global);
        var globalCandidate = Assert.Single(model.GetCandidates(new(
            blockGlobal,
            DataTypes.BFloat16,
            [solver.MakeIntConst(16), solver.MakeIntConst(16)],
            solver.MakeIntConst(512),
            machine,
            solver)));
        Assert.Equal(TargetStorageEncodingIds.Linear, globalCandidate.Id);
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

    private static bool IsCandidate(BlockMicroKernelCandidate candidate, string variant)
        => candidate.Variant == variant;

    private static TargetMachineModel CreateGpuMachine(
        long rootBytesPerCycle = 1_000_000,
        long blockBytesPerCycle = 512,
        double elementwiseElementsPerCycle = 128,
        double simtFmaPerCycle = 64,
        double mmaInstructionsPerCycle = 1,
        double wgmmaInstructionsPerCycle = 1,
        double matrixDependencyLatencyCycles = 16,
        double matrixReciprocalThroughputCycles = 8,
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
            new(BlockExecutionKind.PersistentGpuBlock, 128, 8, 32, 1.0, 128, 4),
            new(
                elementwiseElementsPerCycle,
                simtFmaPerCycle,
                ImmutableArray.Create(
                    new MatrixComputePrimitiveSpec(
                        "mma",
                        16,
                        8,
                        16,
                        matrixDependencyLatencyCycles,
                        matrixReciprocalThroughputCycles,
                        mmaInstructionsPerCycle,
                        1,
                        operandTypes),
                    new MatrixComputePrimitiveSpec(
                        "wgmma",
                        64,
                        8,
                        16,
                        matrixDependencyLatencyCycles,
                        matrixReciprocalThroughputCycles,
                        wgmmaInstructionsPerCycle,
                        4,
                        operandTypes))),
            new(25, gridSynchronizationCycles),
            [
                new(NTTTargetMachineCatalog.GpuRegisterFile, TargetPrivateResourceUnit.Register32, 255L * 8 * 32, 8 * 32),
                new(NTTTargetMachineCatalog.GpuBackendSharedMemory, TargetPrivateResourceUnit.Bytes, 48 * 1024, 16, sharedResource),
            ],
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
            ]);
    }
}
