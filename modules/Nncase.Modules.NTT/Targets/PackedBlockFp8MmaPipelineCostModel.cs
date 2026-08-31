// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.Targets;

/// <summary>
/// Cost model shared by block-FP8 MMA microkernel selection and graph-level target costing.
/// </summary>
internal static class PackedBlockFp8MmaPipelineCostModel
{
    private const int MinimumBlockK = 128;
    private const int MaximumBlockK = 1024;
    private const int MaximumBlockN = 128;
    private const int MinimumLogicalStages = 2;
    private const int NTilesPerActivationBatch = 2;
    private const int SharedVectorAlignmentBytes = 16;

    internal readonly record struct Estimate(
        long Cycles,
        long LhsLoadBytes,
        long RhsLoadBytes);

    public static bool TryEstimateBestLocalCycles(
        TargetMachineModel machine,
        DataType lhsComputeType,
        DataType rhsComputeType,
        long n,
        long k,
        int reductionGroup,
        int rhsTilesPerGroup,
        out Estimate estimate)
    {
        estimate = default;
        if (lhsComputeType != DataTypes.Float8E4M3
            || rhsComputeType != DataTypes.Float8E4M3
            || n < 8
            || k <= 0
            || reductionGroup <= 0
            || rhsTilesPerGroup <= 0
            || k % MinimumBlockK != 0
            || machine.Execution.Kind != BlockExecutionKind.PersistentGpuBlock
            || machine.Execution.WorkersPerBlock != 8
            || machine.Execution.WorkerWidth != 32)
        {
            return false;
        }

        var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared);
        if (sharedSpace is null)
        {
            return false;
        }

        var parentSpace = machine.GetTilingParentMemorySpace(sharedSpace.TilingLevel);
        var transfer = machine.GetTransfer(parentSpace.Id, sharedSpace.Id);
        if (transfer.Asynchronous is not { } asynchronousTransfer)
        {
            return false;
        }

        var sharedMemory = machine.GetMemoryResource(sharedSpace);
        Estimate? bestEstimate = null;
        foreach (var primitive in GetMatrixPrimitives(machine, lhsComputeType, rhsComputeType))
        {
            var primitiveEstimate = EstimateBestForPrimitive(
                machine,
                transfer,
                sharedMemory,
                sharedSpace,
                asynchronousTransfer,
                primitive,
                rhsComputeType,
                n,
                k,
                reductionGroup,
                rhsTilesPerGroup);
            if (primitiveEstimate is { } candidate &&
                (bestEstimate is null || candidate.Cycles < bestEstimate.Value.Cycles))
            {
                bestEstimate = candidate;
            }
        }

        if (bestEstimate is null)
        {
            return false;
        }

        estimate = bestEstimate.Value;
        return true;
    }

    private static Estimate? EstimateBestForPrimitive(
        TargetMachineModel machine,
        TargetMemoryTransferSpec transfer,
        TargetMemoryResourceSpec sharedMemory,
        TargetMemorySpaceSpec sharedSpace,
        TargetAsynchronousTransferSpec asynchronousTransfer,
        MatrixComputePrimitiveSpec primitive,
        DataType rhsComputeType,
        long n,
        long k,
        int reductionGroup,
        int rhsTilesPerGroup)
    {
        var materializeCompleteLhs = primitive.CooperativeWorkers > 1;
        var maximumSupportedBlockN = Math.Min(
            MaximumBlockN,
            checked(
                primitive.M *
                Math.Max(
                    1,
                    machine.Execution.WorkersPerBlock /
                    checked(primitive.CooperativeWorkers * rhsTilesPerGroup))));
        var maximumCandidateBlockK = Math.Max(
            MinimumBlockK,
            MaximumBlockK / rhsTilesPerGroup);
        Estimate? bestEstimate = null;
        for (var blockN = primitive.M;
             blockN <= GetMaximumBlockN(n, primitive.M, maximumSupportedBlockN);
             blockN *= 2)
        {
            for (var blockK = MinimumBlockK;
                 blockK <= Math.Min(k, maximumCandidateBlockK);
                 blockK *= 2)
            {
                if (k % blockK != 0 || blockK % reductionGroup != 0)
                {
                    continue;
                }

                var stageBytes = checked((long)blockN * blockK * rhsComputeType.SizeInBytes);
                foreach (var numStages in asynchronousTransfer.SupportedStageCounts)
                {
                    if (numStages % rhsTilesPerGroup != 0
                        || numStages / rhsTilesPerGroup < MinimumLogicalStages)
                    {
                        continue;
                    }

                    var requiredSharedBytes = checked(
                        ((long)numStages * stageBytes) +
                        GetActivationSharedBytes(
                            materializeCompleteLhs ? k : blockK,
                            reductionGroup,
                            materializeCompleteLhs));
                    var allocatedSharedBytes = machine.GetAllocationSizeBytes(
                        sharedSpace,
                        requiredSharedBytes);
                    if (allocatedSharedBytes > machine.GetMaximumUsableAllocationBytes(sharedSpace))
                    {
                        continue;
                    }

                    var candidateCycles = EstimateCycles(
                        machine,
                        transfer,
                        sharedMemory,
                        primitive,
                        n,
                        k,
                        rhsComputeType.SizeInBytes,
                        reductionGroup,
                        rhsTilesPerGroup,
                        stageCompleteLhs: materializeCompleteLhs,
                        [n],
                        blockN,
                        blockK,
                        numStages);
                    var nTileCount = DivideRoundUp(n, blockN);
                    var nBatchCount = DivideRoundUp(nTileCount, NTilesPerActivationBatch);
                    var candidateEstimate = new Estimate(
                        candidateCycles,
                        checked(
                            (materializeCompleteLhs ? 1 : nBatchCount) *
                            k * DataTypes.BFloat16.SizeInBytes),
                        checked(nTileCount * k * blockN * rhsComputeType.SizeInBytes * rhsTilesPerGroup));
                    if (bestEstimate is null || candidateCycles < bestEstimate.Value.Cycles)
                    {
                        bestEstimate = candidateEstimate;
                    }
                }
            }
        }

        return bestEstimate;
    }

    public static long EstimateCycles(
        TargetMachineModel machine,
        TargetMemoryTransferSpec transfer,
        TargetMemoryResourceSpec sharedMemory,
        MatrixComputePrimitiveSpec primitive,
        long n,
        long k,
        int elementBytes,
        int reductionGroup,
        int rhsTilesPerGroup,
        bool stageCompleteLhs,
        IReadOnlyList<long> localNExtents,
        int blockN,
        int blockK,
        int numStages)
    {
        var (nTileCount, totalNTileCount) = GetNTileCounts(localNExtents, blockN);
        var nBatchCount = DivideRoundUp(nTileCount, NTilesPerActivationBatch);
        var kTileCount = k / blockK;
        var logicalTileCount = checked(nTileCount * kTileCount);
        var reductionGroupsPerTile = blockK / reductionGroup;
        var bytesPerLogicalTile = checked(
            (long)blockN * blockK * elementBytes * rhsTilesPerGroup);

        var producerTransferCycles = DivideRoundUp(bytesPerLogicalTile, transfer.BytesPerCycle);
        var producerControlCycles = checked(
            machine.Synchronization.BlockCycles +
            (rhsTilesPerGroup * transfer.Asynchronous!.CommitCycles));
        var producerServiceCycles = checked(producerTransferCycles + producerControlCycles);

        // GEMV maps output N to the primitive M axis and pads its singleton
        // column to primitive N. MatrixComputeCostModel charges idle workers.
        var accumulatorChains = checked(
            (double)DivideRoundUp(blockN, primitive.M) *
            DivideRoundUp(1, primitive.N));
        var dependentInstructions = DivideRoundUp(reductionGroup, primitive.K);
        var matrixCyclesPerGroup = checked((long)Math.Ceiling(
            MatrixComputeCostModel.EstimateCycles(
                primitive,
                accumulatorChains,
                dependentInstructions,
                machine.Execution)));
        var sharedLoadCyclesPerGroup = DivideRoundUp(
            checked((long)blockN * reductionGroup * elementBytes),
            sharedMemory.ReadBytesPerCycle);
        var activationComputeCyclesPerGroup = DivideRoundUp(
            reductionGroup,
            machine.Compute.ElementwiseElementsPerCycle);
        var activationLoadCyclesPerKTile = DivideRoundUp(
            checked((long)blockK * DataTypes.BFloat16.SizeInBytes),
            transfer.BytesPerCycle);
        var activationSharedStoreCyclesPerKTile = checked(
            DivideRoundUp(
                checked(
                    (long)blockK *
                    DataTypes.Float8E4M3.SizeInBytes),
                sharedMemory.WriteBytesPerCycle) +
            DivideRoundUp(
                checked((long)reductionGroupsPerTile * DataTypes.Float32.SizeInBytes),
                sharedMemory.WriteBytesPerCycle));
        var activationSharedLoadCyclesPerGroup = DivideRoundUp(
            checked(
                (long)reductionGroup *
                DataTypes.Float8E4M3.SizeInBytes),
            sharedMemory.ReadBytesPerCycle);
        var epilogueCyclesPerGroup = DivideRoundUp(
            blockN,
            machine.Compute.ElementwiseElementsPerCycle);

        var activationPreparationCyclesPerKTile = checked(
            activationLoadCyclesPerKTile +
            activationSharedStoreCyclesPerKTile +
            ((long)reductionGroupsPerTile * activationComputeCyclesPerGroup) +
            ((long)(reductionGroupsPerTile + 1) * machine.Synchronization.BlockCycles));
        var activationPreparationPasses = stageCompleteLhs ? 1 : nBatchCount;
        var activationPreparationCycles = checked(
            activationPreparationPasses * kTileCount * activationPreparationCyclesPerKTile);
        var matrixAndEpilogueCyclesPerLogicalTile = checked(
            (long)reductionGroupsPerTile *
            (activationSharedLoadCyclesPerGroup +
             (rhsTilesPerGroup *
              (Math.Max(sharedLoadCyclesPerGroup, matrixCyclesPerGroup) +
               epilogueCyclesPerGroup))));
        var consumerWorkCycles = DivideRoundUp(
            checked(
                activationPreparationCycles +
                (logicalTileCount * matrixAndEpilogueCyclesPerLogicalTile)),
            logicalTileCount);
        var consumerServiceCycles = checked(
            consumerWorkCycles + transfer.Asynchronous.WaitCycles);

        var slotLifetimeCycles = checked(
            producerServiceCycles + transfer.LatencyCycles + consumerServiceCycles);
        var logicalStageCount = numStages / rhsTilesPerGroup;
        var recurrenceCycles = DivideRoundUp(slotLifetimeCycles, logicalStageCount);
        var initiationIntervalCycles = Math.Max(
            Math.Max(producerServiceCycles, consumerServiceCycles),
            recurrenceCycles);
        var localPipelineCycles = checked(
            producerServiceCycles +
            consumerServiceCycles +
            ((logicalTileCount - 1) * initiationIntervalCycles));
        var rootMemory = machine.GetMemoryResource(machine.GetMemorySpace(machine.RootMemorySpace));
        var remainingBlockBytes = checked(
            (totalNTileCount - nTileCount) * kTileCount * bytesPerLogicalTile);
        var remainingBlockCycles = DivideRoundUp(remainingBlockBytes, rootMemory.ReadBytesPerCycle);
        return checked(localPipelineCycles + remainingBlockCycles);
    }

    private static IEnumerable<MatrixComputePrimitiveSpec> GetMatrixPrimitives(
        TargetMachineModel machine,
        DataType lhsType,
        DataType rhsType)
        => machine.Compute.MatrixPrimitives
            .Where(primitive => primitive.Supports(lhsType, rhsType))
            .Where(primitive => primitive.Name == "mma" && primitive.CooperativeWorkers == 1)
            .OrderBy(primitive => primitive.CooperativeWorkers)
            .ThenBy(primitive => primitive.M)
            .ThenBy(primitive => primitive.N);

    private static long GetActivationSharedBytes(
        long blockK,
        int reductionGroup,
        bool materializeCompleteLhs)
    {
        var logicalGroupCount = blockK / reductionGroup;
        var allocatedGroupCount = materializeCompleteLhs
            ? RoundUpPowerOfTwo(logicalGroupCount)
            : logicalGroupCount;
        var quantizedBytes = RoundUp(
            checked(
                allocatedGroupCount * reductionGroup *
                DataTypes.Float8E4M3.SizeInBytes),
            SharedVectorAlignmentBytes);
        var scaleBytes = RoundUp(
            checked(allocatedGroupCount * DataTypes.Float32.SizeInBytes),
            SharedVectorAlignmentBytes);
        return checked(quantizedBytes + scaleBytes);
    }

    private static long RoundUpPowerOfTwo(long value)
    {
        var rounded = System.Numerics.BitOperations.RoundUpToPowerOf2(
            checked((ulong)value));
        if (rounded == 0 || rounded > long.MaxValue)
        {
            throw new OverflowException(
                $"Shared workspace element capacity {value} cannot be represented as a power of two.");
        }

        return checked((long)rounded);
    }

    private static (long Maximum, long Total) GetNTileCounts(
        IReadOnlyList<long> localNExtents,
        int blockN)
    {
        long maximum = 0;
        long total = 0;
        foreach (var extent in localNExtents)
        {
            var tiles = extent == 0 ? 0 : DivideRoundUp(extent, blockN);
            maximum = Math.Max(maximum, tiles);
            total = checked(total + tiles);
        }

        return (maximum, total);
    }

    private static int GetMaximumBlockN(long n, int minimumBlockN, int maximumBlockN)
    {
        var roundedN = minimumBlockN;
        while (roundedN < n && roundedN < maximumBlockN)
        {
            roundedN *= 2;
        }

        return Math.Min(roundedN, maximumBlockN);
    }

    private static long RoundUp(long value, long alignment)
        => checked(DivideRoundUp(value, alignment) * alignment);

    private static long DivideRoundUp(long value, long divisor)
        => value == 0 ? 0 : checked(((value - 1) / divisor) + 1);

    private static long DivideRoundUp(long value, double divisor)
        => checked((long)Math.Ceiling(value / divisor));
}
