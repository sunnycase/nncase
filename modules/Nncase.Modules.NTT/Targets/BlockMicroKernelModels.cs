// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Schedule;

namespace Nncase.Targets;

/// <summary>
/// Stable parameter names shared by the Triton block cost model and PyNTT
/// code generation. These parameters describe a backend-private CTA
/// implementation; they do not add warp or thread levels to the TIR hierarchy.
/// </summary>
public static class TritonBlockMicroKernelContract
{
    public const long Version = 3;

    public const string VersionParameter = "contract_version";

    public const string StateBlockMParameter = "state_block_m";

    public const string StateBlockNParameter = "state_block_n";

    public const string StateBlockKParameter = "state_block_k";

    public const string InnerMParameter = "inner_m";

    public const string InnerNParameter = "inner_n";

    public const string InnerKParameter = "inner_k";

    public const string PipelineStagesParameter = "pipeline_stages";

    public const string MmaMParameter = "mma_m";

    public const string MmaNParameter = "mma_n";

    public const string MmaKParameter = "mma_k";
}

/// <summary>
/// Conservative backend-private state model used by native NTT targets.
/// </summary>
public sealed class DefaultBlockMicroKernelModel : IBlockMicroKernelModelProvider
{
    public IReadOnlyList<BlockMicroKernelCandidate> GetCandidates(BlockMicroKernelModelContext context)
    {
        if (context.Workload is not IReductionStateTileWorkload stateful)
        {
            return Array.Empty<BlockMicroKernelCandidate>();
        }

        var stateBytes = SumLogicalStateBytes(stateful, context);
        var resource = context.Machine.PrivateResources.Values
            .OrderBy(spec => spec.Id.Value, StringComparer.Ordinal)
            .FirstOrDefault(spec => spec.BackingMemoryResource is null)
            ?? throw new InvalidOperationException(
                $"Target {context.Machine.Id} has no unbacked private resource for {context.Op.GetType().Name}.");
        var units = resource.Unit switch
        {
            TargetPrivateResourceUnit.Bytes => stateBytes,
            TargetPrivateResourceUnit.Register32 => CeilDiv(stateBytes, 4, context.Solver),
            _ => throw new InvalidOperationException(
                $"Target-private resource {resource.Id} uses unsupported unit {resource.Unit}."),
        };
        return
        [
            new(
                "default.block",
                "backend_private",
                context.Solver.MakeIntConst(1),
                context.BaseComputeCycles,
                [new(resource.Id, units)],
                ImmutableArray<BlockMicroKernelMemoryAccess>.Empty,
                ImmutableArray<BlockMicroKernelParameter>.Empty),
        ];
    }

    internal static IntExpr SumLogicalStateBytes(
        IReductionStateTileWorkload stateful,
        BlockMicroKernelModelContext context)
    {
        var states = stateful.GetReductionStates(
            context.LocalBufferShapes,
            context.Solver,
            context.WorkloadContext);
        if (states.Count == 0)
        {
            throw new InvalidOperationException(
                $"Reduction workload {context.Op.GetType().Name} returned no logical state descriptors.");
        }

        foreach (var state in states)
        {
            if (string.IsNullOrWhiteSpace(state.Name) || state.ElementSizeBytes <= 0)
            {
                throw new InvalidOperationException(
                    $"Reduction workload {context.Op.GetType().Name} returned an invalid state descriptor.");
            }
        }

        return states.Aggregate(
            (IntExpr)context.Solver.MakeIntConst(0),
            static (sum, state) => sum + state.GetLogicalBytes());
    }

    internal static IntExpr CeilDiv(IntExpr value, long divisor, Solver solver)
        => solver.MakeDiv(value + (divisor - 1), divisor);
}

/// <summary>
/// Triton block microkernel model. It models only CTA-visible aggregate
/// resources. Triton remains responsible for warp/thread layouts and exact
/// register allocation, which is verified from the compiled specialization.
/// </summary>
public sealed class TritonBlockMicroKernelModel : IBlockMicroKernelModelProvider
{
    private const int MatrixAccumulatorMinM = 16;
    private const int MatrixAccumulatorMinN = 16;
    private const int FixedRegistersPerThread = 48;
    private const int GemvFixedRegistersPerThread = 56;
    private const int GemvMmaCompilerOverheadRegistersPerThread = 216;
    private const int SimtComputeElementSizeBytes = 4;
    private const int SimtRhsLiveValueCount = 2;
    private const int MmaPipelineStages = 2;

    public IReadOnlyList<BlockMicroKernelCandidate> GetCandidates(BlockMicroKernelModelContext context)
    {
        if (context.Workload is not IReductionStateTileWorkload stateful)
        {
            return Array.Empty<BlockMicroKernelCandidate>();
        }

        if (context.Workload is not MatrixTileWorkload matrix)
        {
            return GetGenericReductionCandidates(stateful, context);
        }

        return GetMatrixCandidates(matrix, context);
    }

    private static IReadOnlyList<BlockMicroKernelCandidate> GetGenericReductionCandidates(
        IReductionStateTileWorkload stateful,
        BlockMicroKernelModelContext context)
    {
        var registerResource = GetResource(
            context.Machine,
            NTTTargetMachineCatalog.GpuRegisterFile,
            TargetPrivateResourceUnit.Register32);
        var stateBytes = DefaultBlockMicroKernelModel.SumLogicalStateBytes(stateful, context);
        var stateRegisters = DefaultBlockMicroKernelModel.CeilDiv(stateBytes, 4, context.Solver);
        var fixedRegisters = context.Solver.MakeIntConst(
            checked((long)FixedRegistersPerThread * context.Machine.Execution.ThreadsPerBlock));
        return
        [
            new(
                "triton.reduce",
                "register_accumulator",
                context.Solver.MakeIntConst(1),
                context.BaseComputeCycles,
                [new(registerResource.Id, fixedRegisters + stateRegisters)],
                ImmutableArray<BlockMicroKernelMemoryAccess>.Empty,
                ImmutableArray<BlockMicroKernelParameter>.Empty),
        ];
    }

    private static IReadOnlyList<BlockMicroKernelCandidate> GetMatrixCandidates(
        MatrixTileWorkload matrix,
        BlockMicroKernelModelContext context)
    {
        var solver = context.Solver;
        var local = matrix.GetShape(context.LocalBufferShapes, solver, context.WorkloadContext);
        var full = matrix.GetShape(context.FullBufferShapes, solver, context.WorkloadContext);
        var fullM = full.M.Var();
        if (fullM.Min() != fullM.Max())
        {
            throw new InvalidOperationException(
                $"Full matrix M extent for {context.Op.GetType().Name} must be constant.");
        }

        var useGemv = fullM.Max() == 1;
        var simtWorkerWidth = context.Machine.Execution.WorkerWidth;
        if (!System.Numerics.BitOperations.IsPow2((uint)simtWorkerWidth))
        {
            throw new InvalidOperationException(
                $"Triton SIMT microkernels require a power-of-two worker width, got " +
                $"{simtWorkerWidth} lanes on {context.Machine.Id}.");
        }

        var simtComputeCycles = DivideByRate(
            full.GetWork(),
            context.Machine.Compute.SimtFmaPerCycle,
            solver);
        var alignedM = useGemv ? local.M : AlignUp(local.M, MatrixAccumulatorMinM, solver);
        var alignedN = useGemv
            ? AlignUp(solver.MakeMax(local.N, solver.MakeIntConst(simtWorkerWidth)), simtWorkerWidth, solver)
            : AlignUp(local.N, MatrixAccumulatorMinN, solver);
        var registerM = useGemv ? solver.MakeIntConst(1) : alignedM;

        // The SIMT helper materializes one complete block-K reduction tile.
        // Splitting K inside the helper repeats Triton layout propagation and
        // warp reductions, so block-K itself is the backend primitive extent.
        // GraphTiler remains responsible for shrinking block-K when aggregate
        // CTA register or shared-memory capacity requires it.
        var simtReductionK = local.K;
        var simtFixedRegisters = solver.MakeIntConst(
            checked((long)(useGemv ? GemvFixedRegistersPerThread : FixedRegistersPerThread)
                * context.Machine.Execution.ThreadsPerBlock));
        var accumulatorRegisters = DefaultBlockMicroKernelModel.CeilDiv(
            registerM * alignedN * local.Multiplicity * matrix.AccumulatorElementSizeBytes,
            4,
            solver);

        // The template computes in fp32 and keeps both the converted RHS tile
        // and its multiply/reduction intermediate live across the reduction.
        var simtOperandBytes = (registerM * simtReductionK * SimtComputeElementSizeBytes)
            + (alignedN * simtReductionK * local.Multiplicity
                * SimtComputeElementSizeBytes * SimtRhsLiveValueCount);
        var simtOperandRegisters = DefaultBlockMicroKernelModel.CeilDiv(simtOperandBytes, 4, solver);
        var registerSimtUsage = simtFixedRegisters + accumulatorRegisters + simtOperandRegisters;
        var registerResource = GetResource(
            context.Machine,
            NTTTargetMachineCatalog.GpuRegisterFile,
            TargetPrivateResourceUnit.Register32);

        var family = useGemv ? "triton.gemv" : "triton.matmul";
        var isSupportedGemv = useGemv && context.Op is (Nncase.TIR.NTT.Matmul or Nncase.TIR.NTT.PackedMatMul);
        var mmaSharedMemorySpace = GetMmaSharedMemorySpace(context.Machine);
        var dotOperandRequirements = !useGemv
            ? GetDotOperandAccessIndices(context.Op)
                .Select(index => new BlockMicroKernelBufferEncodingRequirement(
                    index,
                    mmaSharedMemorySpace.Id,
                    [TritonTargetStorageEncodingModel.NvidiaMmaShared]))
                .ToImmutableArray()
            : ImmutableArray<BlockMicroKernelBufferEncodingRequirement>.Empty;
        var simtOperandRequirements = useGemv
            ? GetDirectPackedGemvRhsAccessIndices(context.Op)
                .Select(index => new BlockMicroKernelBufferEncodingRequirement(
                    index,
                    mmaSharedMemorySpace.Id,
                    [TritonTargetStorageEncodingModel.KMajorPackedN]))
                .ToImmutableArray()
            : dotOperandRequirements;
        var sharedResource = GetResource(
            context.Machine,
            NTTTargetMachineCatalog.GpuBackendSharedMemory,
            TargetPrivateResourceUnit.Bytes);
        var sharedMemoryResource = sharedResource.BackingMemoryResource is { } backing
            ? context.Machine.GetMemoryResource(backing)
            : throw new InvalidOperationException(
                "Triton backend-private shared memory must name its physical backing resource.");
        var gemvMemoryAccesses = isSupportedGemv
            ? GetGemvMemoryAccesses(context, local, full, sharedMemoryResource.Id)
            : ImmutableArray<BlockMicroKernelMemoryAccess>.Empty;
        var hiddenDotStagingBytes = useGemv
            ? solver.MakeIntConst(0)
            : GetHiddenDotOperandStagingBytes(context, GetDotOperandAccessIndices(context.Op));
        var registerSimtResources = ImmutableArray.Create(
            new BlockMicroKernelResourceUsage(registerResource.Id, registerSimtUsage));
        if (hiddenDotStagingBytes.Var().Max() > 0)
        {
            registerSimtResources = registerSimtResources.Add(
                new(sharedResource.Id, hiddenDotStagingBytes));
        }

        var registerSimtCandidate = new BlockMicroKernelCandidate(
            family,
            "register_simt_accumulator",
            solver.MakeIntConst(1),
            useGemv ? simtComputeCycles : context.BaseComputeCycles,
            registerSimtResources,
            gemvMemoryAccesses,
            [
                new(TritonBlockMicroKernelContract.VersionParameter, solver.MakeIntConst(TritonBlockMicroKernelContract.Version)),
                new(TritonBlockMicroKernelContract.StateBlockMParameter, alignedM),
                new(TritonBlockMicroKernelContract.StateBlockNParameter, alignedN),
                new(TritonBlockMicroKernelContract.StateBlockKParameter, local.K),
                new(TritonBlockMicroKernelContract.InnerMParameter, registerM),
                new(TritonBlockMicroKernelContract.InnerNParameter, alignedN),
                new(TritonBlockMicroKernelContract.InnerKParameter, simtReductionK),
                new(TritonBlockMicroKernelContract.PipelineStagesParameter, solver.MakeIntConst(1)),
            ])
        {
            BufferEncodingRequirements = simtOperandRequirements,
        };
        var candidates = new List<BlockMicroKernelCandidate> { registerSimtCandidate };

        if (isSupportedGemv)
        {
            var mmaPrimitives = context.Machine.Compute.MatrixPrimitives
                .Where(primitive => primitive.Supports(
                    context.WorkloadContext.BufferDataTypes[0],
                    context.WorkloadContext.BufferDataTypes[1]))
                .OrderBy(primitive => primitive.M)
                .ThenBy(primitive => primitive.N)
                .ThenBy(primitive => primitive.K)
                .ToArray();
            if (mmaPrimitives.Length > 0)
            {
                var mma = mmaPrimitives[0];
                var lhsScalarBytes = GetScalarElementSizeBytes(context.WorkloadContext.BufferDataTypes[0]);
                var rhsScalarBytes = GetScalarElementSizeBytes(context.WorkloadContext.BufferDataTypes[1]);
                var nIsLargeEnough = 1 - solver.MakeIsLessVar(local.N, solver.MakeIntConst(mma.M));
                var nIsAligned = solver.MakeIsEqualCstVar(solver.MakeModulo(local.N, mma.M), 0);
                var kIsLargeEnough = 1 - solver.MakeIsLessVar(local.K, solver.MakeIntConst(mma.K));
                var kIsAligned = solver.MakeIsEqualCstVar(solver.MakeModulo(local.K, mma.K), 0);

                var fullNBound = full.N.Var().Max();
                var fullNTileCount = solver.MakeDiv(full.N, local.N).Var();
                fullNTileCount.SetRange(0, fullNBound);
                var tailN = (full.N - (fullNTileCount * local.N)).Var();
                tailN.SetRange(0, fullNBound);
                var tailUsesMma = 1 - solver.MakeIsLessVar(tailN, solver.MakeIntConst(mma.M));
                var mmaNTileCount = (fullNTileCount * solver.MakeDiv(local.N, mma.M))
                    + (tailUsesMma * DefaultBlockMicroKernelModel.CeilDiv(tailN, mma.M, solver));
                var simtTailNElements = (1 - tailUsesMma) * tailN;
                var mmaInstructionCount = mmaNTileCount
                    * DefaultBlockMicroKernelModel.CeilDiv(full.M, mma.N, solver)
                    * GetTotalPrimitiveTileCount(full.K, local.K, mma.K, solver)
                    * full.Multiplicity;
                var mmaComputeCycles = DivideByRate(
                    mmaInstructionCount,
                    mma.InstructionsPerCyclePerBlock,
                    solver);
                var simtTailCycles = DivideByRate(
                    full.M * simtTailNElements * full.K * full.Multiplicity,
                    context.Machine.Compute.SimtFmaPerCycle,
                    solver);
                var totalMmaComputeCycles = mmaComputeCycles + simtTailCycles;

                // Registers are modeled from simultaneously live instruction
                // fragments and pipeline stages. Full reduction extent K does
                // not multiply the live set because K fragments are consumed
                // sequentially by the backend microkernel.
                var nFragments = DefaultBlockMicroKernelModel.CeilDiv(alignedN, mma.M, solver);
                var mFragments = DefaultBlockMicroKernelModel.CeilDiv(alignedM, mma.N, solver);
                var mmaAccumulatorRegisters = nFragments
                    * mFragments
                    * mma.M
                    * mma.N
                    * local.Multiplicity;
                var mmaOperandBytesPerStage = nFragments
                    * mFragments
                    * local.Multiplicity
                    * ((mma.M * mma.K * rhsScalarBytes) + (mma.K * mma.N * lhsScalarBytes));
                var mmaOperandRegisters = DefaultBlockMicroKernelModel.CeilDiv(
                    mmaOperandBytesPerStage,
                    4,
                    solver) * MmaPipelineStages;

                // Calibrated backend overhead for the fixed eight-warp persistent
                // execution model. Shape-dependent state remains represented by
                // the live accumulator and operand fragments separately. Dot
                // operand staging is explicit TIR storage selected through the
                // target storage-encoding contract.
                var mmaFixedRegisters = solver.MakeIntConst(
                    checked((long)GemvMmaCompilerOverheadRegistersPerThread * context.Machine.Execution.ThreadsPerBlock));
                var registerMmaUsage = mmaFixedRegisters
                    + accumulatorRegisters
                    + mmaAccumulatorRegisters
                    + mmaOperandRegisters;
                var mmaLegality = nIsLargeEnough * nIsAligned * kIsLargeEnough * kIsAligned;
                var mmaParameters = ImmutableArray.Create(
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.VersionParameter, solver.MakeIntConst(TritonBlockMicroKernelContract.Version)),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.StateBlockMParameter, alignedM),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.StateBlockNParameter, alignedN),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.StateBlockKParameter, local.K),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.InnerMParameter, solver.MakeIntConst(mma.N)),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.InnerNParameter, alignedN),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.InnerKParameter, solver.MakeIntConst(mma.K)),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.PipelineStagesParameter, solver.MakeIntConst(MmaPipelineStages)),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.MmaMParameter, solver.MakeIntConst(mma.M)),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.MmaNParameter, solver.MakeIntConst(mma.N)),
                    new BlockMicroKernelParameter(TritonBlockMicroKernelContract.MmaKParameter, solver.MakeIntConst(mma.K)));
                var mmaBufferRequirements = ImmutableArray.Create(
                    new BlockMicroKernelBufferEncodingRequirement(
                        0,
                        mmaSharedMemorySpace.Id,
                        [TritonTargetStorageEncodingModel.NvidiaMmaShared]),
                    new BlockMicroKernelBufferEncodingRequirement(
                        1,
                        mmaSharedMemorySpace.Id,
                        [TritonTargetStorageEncodingModel.NvidiaMmaShared]));

                candidates.Add(
                    new(
                        family,
                        "register_mma_accumulator",
                        mmaLegality,
                        totalMmaComputeCycles,
                        [
                            new(registerResource.Id, registerMmaUsage),
                        ],
                        gemvMemoryAccesses,
                        mmaParameters)
                    {
                        BufferEncodingRequirements = mmaBufferRequirements,
                    });
            }
        }

        return candidates;
    }

    private static TargetMemorySpaceSpec GetMmaSharedMemorySpace(TargetMachineModel machine)
    {
        var candidates = machine.TilingMemorySpaces
            .Where(space => machine.GetMemoryResource(space).Kind == TargetMemorySpaceKind.Shared)
            .ToArray();
        return candidates.Length == 1
            ? candidates[0]
            : throw new InvalidOperationException(
                $"Triton matrix microkernels require exactly one tiling shared-memory space, found " +
                $"[{string.Join(", ", candidates.Select(space => space.Id))}] on {machine.Id}.");
    }

    private static int[] GetDotOperandAccessIndices(Op op)
        => op switch
        {
            Nncase.TIR.NTT.Matmul or Nncase.TIR.NTT.PackedMatMul => [0, 1],
            Nncase.TIR.NTT.QKVParallelLinear or Nncase.TIR.NTT.PackedQKVParallelLinear => [0, 1, 2, 3],
            Nncase.TIR.NTT.MatMulGlu or Nncase.TIR.NTT.PackedMatMulGlu => [0, 1, 2],
            _ => throw new NotSupportedException(
                $"Triton matrix workload {op.GetType().Name} must declare its dot operand accesses."),
        };

    private static int[] GetDirectPackedGemvRhsAccessIndices(Op op)
        => op switch
        {
            Nncase.TIR.NTT.PackedMatMul => [1],
            Nncase.TIR.NTT.Matmul or
                Nncase.TIR.NTT.QKVParallelLinear or
                Nncase.TIR.NTT.PackedQKVParallelLinear or
                Nncase.TIR.NTT.MatMulGlu or
                Nncase.TIR.NTT.PackedMatMulGlu => [],
            _ => throw new NotSupportedException(
                $"Triton matrix workload {op.GetType().Name} must declare its direct packed GEMV RHS accesses."),
        };

    private static IntExpr GetHiddenDotOperandStagingBytes(
        BlockMicroKernelModelContext context,
        IReadOnlyList<int> accessIndices)
    {
        var solver = context.Solver;
        var total = (IntExpr)solver.MakeIntConst(0);
        foreach (var accessIndex in accessIndices)
        {
            var dataType = context.WorkloadContext.BufferDataTypes[accessIndex];
            if (dataType is not VectorType)
            {
                continue;
            }

            var shape = context.LocalBufferShapes[accessIndex];
            var bytes = shape.Aggregate(
                (IntExpr)solver.MakeIntConst(dataType.SizeInBytes),
                (product, extent) => product * extent);
            var directlyConsumable = shape.Length == 2
                ? solver.MakeIsEqualCstVar(shape[0], 1)
                : solver.MakeIntConst(0);
            total += (1 - directlyConsumable) * bytes;
        }

        return total;
    }

    private static ImmutableArray<BlockMicroKernelMemoryAccess> GetGemvMemoryAccesses(
        BlockMicroKernelModelContext context,
        MatrixTileWorkloadShape local,
        MatrixTileWorkloadShape full,
        TargetMemoryResourceId resource)
    {
        if (context.WorkloadContext.BufferDataTypes.Length < 3)
        {
            throw new InvalidOperationException(
                $"Triton GEMV {context.Op.GetType().Name} requires lhs, rhs, and output buffers.");
        }

        var solver = context.Solver;
        var nTileCount = solver.MakeDiv(full.N + local.N - 1, local.N);
        var lhsBytes = full.M
            * full.K
            * full.Multiplicity
            * nTileCount
            * GetScalarElementSizeBytes(context.WorkloadContext.BufferDataTypes[0]);
        var rhsBytes = full.M
            * full.N
            * full.K
            * full.Multiplicity
            * GetScalarElementSizeBytes(context.WorkloadContext.BufferDataTypes[1]);
        var outputBytes = full.M
            * full.N
            * full.Multiplicity
            * GetScalarElementSizeBytes(context.WorkloadContext.BufferDataTypes[2]);
        return
        [
            new(0, resource, MemoryAccessMode.Read, lhsBytes),
            new(1, resource, MemoryAccessMode.Read, rhsBytes),
            new(2, resource, MemoryAccessMode.Write, outputBytes),
        ];
    }

    private static IntExpr GetTotalPrimitiveTileCount(
        IntExpr fullExtent,
        IntExpr localExtent,
        int primitiveExtent,
        Solver solver)
    {
        var fullExtentBound = fullExtent.Var().Max();
        var fullLocalTileCount = solver.MakeDiv(fullExtent, localExtent).Var();
        fullLocalTileCount.SetRange(0, fullExtentBound);
        var tailExtent = (fullExtent - (fullLocalTileCount * localExtent)).Var();
        tailExtent.SetRange(0, fullExtentBound);
        return (fullLocalTileCount * solver.MakeDiv(localExtent, primitiveExtent))
            + DefaultBlockMicroKernelModel.CeilDiv(tailExtent, primitiveExtent, solver);
    }

    private static int GetScalarElementSizeBytes(DataType dataType)
        => dataType is VectorType vectorType
            ? GetScalarElementSizeBytes(vectorType.ElemType)
            : dataType.SizeInBytes;

    private static TargetPrivateResourceSpec GetResource(
        TargetMachineModel machine,
        TargetPrivateResourceId id,
        TargetPrivateResourceUnit expectedUnit)
    {
        var resource = machine.GetPrivateResource(id);
        if (resource.Unit != expectedUnit)
        {
            throw new InvalidOperationException(
                $"Target-private resource {id} uses {resource.Unit}, expected {expectedUnit}.");
        }

        return resource;
    }

    private static IntExpr AlignUp(IntExpr value, int alignment, Solver solver)
        => solver.MakeDiv(value + (alignment - 1), alignment) * alignment;

    private static IntExpr DivideByRate(IntExpr work, double unitsPerCycle, Solver solver)
    {
        const long scale = 1024;
        if (!double.IsFinite(unitsPerCycle) || unitsPerCycle <= 0)
        {
            throw new InvalidOperationException(
                $"Target throughput must be finite and positive, got {unitsPerCycle}.");
        }

        var scaledRate = checked((long)Math.Round(unitsPerCycle * scale));
        return DefaultBlockMicroKernelModel.CeilDiv(work * scale, scaledRate, solver);
    }
}
