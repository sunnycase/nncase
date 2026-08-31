// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Targets;

/// <summary>
/// Deterministic block-microkernel selection for direct PyNTT TIR. Triton owns
/// warp/thread scheduling and instruction lowering; this selector only fixes
/// the template family, block tile, and compiler-managed shared workspaces.
/// </summary>
public sealed class TritonTIRMicroKernelSelector : ITIRMicroKernelSelector
{
    private const int NvidiaNvmmaSharedAlignmentBytes = 1024;
    private const int TritonSharedVectorAlignmentBytes = 16;
    private const int SimtPagedAttentionMaximumBlockN = 128;
    private const int MmaPagedAttentionMaximumBlockN = 64;
    private const int PackedGemvMinimumBlockN = 8;
    private const int SimtPackedGemvMaximumBlockN = 64;
    private const int MmaPackedGemvMaximumBlockN = 128;
    private const int SimtPagedAttentionNumStages = 1;
    private const int MmaPagedAttentionNumStages = 2;
    private const int PackedGemvMinimumBlockK = 128;

    // The SIMT stage helper statically expands one 32-element reduction group.
    // Keep a stage within 32 groups; larger bodies delay first-tile consumption
    // and underutilize the asynchronous double buffer despite fitting in Shared.
    private const int PackedGemvMaximumBlockK = 1024;
    private const int BlockFp8MmaMaximumTransferBlockK = 128;
    private const int BlockFp8MmaMaximumMergedTransferBlockK = 256;
    private const int NVFP4BlockK = 512;
    private const int NVFP4MaximumBlockN = 128;
    private const int PackedGemvMinimumLogicalStages = 2;
    private const int BlockFp8MmaNTilesPerActivationBatch = 2;
    private const int GatedDeltaNetConvolutionMaximumBlockN = 256;
    private const int GatedDeltaNetRecurrentCoreBlockN = 128;
    private const int GatedDeltaNetProjectionMaximumBlockK = 2048;
    private const int GatedDeltaNetProjectionTmaKAtom = 64;
    private const int GatedDeltaNetStateValueTile = 8;
    private const int SparseExpertsDownBlockM = 16;
    private const int SparseExpertsDownMaximumBlockN = 64;
    private const int SparseExpertsDownMaximumStageK = 128;
    private const int SparseExpertsDownMaximumRoutesPerStage = 8;
    private const int SparseExpertsDownMinimumExpertBlockK = 16;
    private const int PackedQkvSplitKMmaBlockN = 256;
    private const int PackedQkvSplitKMmaBlockK = 64;
    private const int PackedQkvSplitKMmaNumStages = 2;
    private const int PackedQkvDirectMmaBlockN = 32;
    private const int PackedQkvDirectMmaInputK = 2048;
    private const int PackedQkvDirectMmaBlockK = 1024;
    private const int PackedQkvDirectMmaNumStages = 2;

    private enum ConsumerLhsStagingKind
    {
        None,
        PerKTile,
        CompleteK,
    }

    public TIRMicroKernelSelection? Select(TIRMicroKernelSelectionContext context)
    {
        return context.Op switch
        {
            Nncase.TIR.NTT.Matmul matmul => SelectMatmul(
                context,
                matmul.TransposeA,
                matmul.TransposeB,
                kMajorPacked: false,
                lhsIndex: 0,
                rhsIndex: 1,
                outputIndex: 2),
            Nncase.TIR.NTT.PackedMatMul packedMatmul => SelectMatmul(
                context,
                transposeA: false,
                transposeB: packedMatmul.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajor,
                kMajorPacked: packedMatmul.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
                lhsIndex: 0,
                rhsIndex: 1,
                outputIndex: 2),
            Nncase.TIR.NTT.PackedScaledMatMul packedScaledMatmul => SelectMatmul(
                context,
                transposeA: false,
                transposeB: packedScaledMatmul.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajor,
                kMajorPacked: packedScaledMatmul.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
                lhsIndex: Nncase.TIR.NTT.PackedScaledMatMul.Lhs.Index,
                rhsIndex: Nncase.TIR.NTT.PackedScaledMatMul.Rhs.Index,
                outputIndex: Nncase.TIR.NTT.PackedScaledMatMul.Output.Index),
            Nncase.TIR.NTT.PackedBlockScaledMatMul packedBlockScaledMatmul => SelectMatmul(
                context,
                transposeA: false,
                transposeB: packedBlockScaledMatmul.RhsLayout != IR.NTT.PackedMatMulRhsLayout.KMajor,
                kMajorPacked: packedBlockScaledMatmul.RhsLayout is
                    IR.NTT.PackedMatMulRhsLayout.KMajor or
                    IR.NTT.PackedMatMulRhsLayout.NMajorKPacked,
                lhsIndex: Nncase.TIR.NTT.PackedBlockScaledMatMul.Lhs.Index,
                rhsIndex: Nncase.TIR.NTT.PackedBlockScaledMatMul.Rhs.Index,
                outputIndex: Nncase.TIR.NTT.PackedBlockScaledMatMul.Output.Index,
                fp8Variant: packedBlockScaledMatmul.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajorKPacked
                    ? "mma_block_fp8_smem_pipeline"
                    : "simt_block_fp8_fma_smem_pipeline",
                blockFp8ReductionGroup: checked((int)packedBlockScaledMatmul.WeightBlockK),
                enableCompleteConsumerLhsStage: true,
                minimumCompleteConsumerLhsK: 2048),
            Nncase.TIR.NTT.PackedBlockScaledMatMulNormStats packedBlockScaledMatmulNormStats => SelectMatmul(
                context,
                transposeA: false,
                transposeB: packedBlockScaledMatmulNormStats.RhsLayout != IR.NTT.PackedMatMulRhsLayout.KMajor,
                kMajorPacked: packedBlockScaledMatmulNormStats.RhsLayout is
                    IR.NTT.PackedMatMulRhsLayout.KMajor or
                    IR.NTT.PackedMatMulRhsLayout.NMajorKPacked,
                lhsIndex: Nncase.TIR.NTT.PackedBlockScaledMatMulNormStats.Lhs.Index,
                rhsIndex: Nncase.TIR.NTT.PackedBlockScaledMatMulNormStats.Rhs.Index,
                outputIndex: Nncase.TIR.NTT.PackedBlockScaledMatMulNormStats.Output.Index,
                fp8Variant: packedBlockScaledMatmulNormStats.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajorKPacked
                    ? "mma_block_fp8_smem_pipeline"
                    : "simt_block_fp8_fma_smem_pipeline",
                blockFp8ReductionGroup: checked((int)packedBlockScaledMatmulNormStats.WeightBlockK)),
            Nncase.TIR.NTT.NVFP4MatMul nvfp4MatMul => SelectNVFP4MatMul(
                context,
                nvfp4MatMul.GroupSize,
                Nncase.TIR.NTT.NVFP4MatMul.Lhs.Index,
                Nncase.TIR.NTT.NVFP4MatMul.RhsPacked.Index,
                Nncase.TIR.NTT.NVFP4MatMul.RhsScale.Index,
                Nncase.TIR.NTT.NVFP4MatMul.Output.Index,
                "triton.nvfp4_matmul",
                [
                    Nncase.TIR.NTT.NVFP4MatMul.RhsPacked.Index,
                ]),
            Nncase.TIR.NTT.PagedAttentionMergePackedMatMul fused => SelectMatmul(
                context,
                transposeA: false,
                transposeB: fused.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajor,
                kMajorPacked: fused.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
                lhsIndex: 4,
                rhsIndex: 5,
                outputIndex: 6,
                family: "triton.paged_attention_merge_matmul"),
            Nncase.TIR.NTT.PackedMatMulNormStats packedMatmulNormStats => SelectMatmul(
                context,
                transposeA: false,
                transposeB: packedMatmulNormStats.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajor,
                kMajorPacked: packedMatmulNormStats.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
                lhsIndex: 0,
                rhsIndex: 1,
                outputIndex: 2,
                enableCompleteConsumerLhsStage: true),
            Nncase.TIR.NTT.PackedMatMulSamplingPartial packedMatmulSamplingPartial =>
                SelectPackedMatMulSamplingPartial(context, packedMatmulSamplingPartial),
            Nncase.TIR.NTT.SUMMA summa => SelectSumma(context, summa),
            Nncase.TIR.NTT.QKVParallelLinear => SelectFusedLinear(
                context,
                "triton.qkv_parallel_linear",
                inputIndex: 0,
                weightIndex: 1,
                outputIndex: 13),
            Nncase.TIR.NTT.PackedQKVParallelLinearFusedRhs packedQkv => SelectPackedQkv(
                context,
                packedQkv),
            Nncase.TIR.NTT.MatMulGlu => SelectFusedLinear(
                context,
                "triton.matmul_glu",
                inputIndex: 0,
                weightIndex: 1,
                outputIndex: 9),
            Nncase.TIR.NTT.PackedMatMulGlu packedMatmulGlu => SelectPackedMatMulGlu(
                context,
                packedMatmulGlu),
            Nncase.TIR.NTT.NVFP4MatMulGlu nvfp4MatMulGlu => SelectNVFP4MatMulGlu(
                context,
                nvfp4MatMulGlu),
            Nncase.TIR.NTT.PagedAttentionPartial pagedAttention =>
                SelectPagedAttentionPartial(context, pagedAttention),
            Nncase.TIR.NTT.GatedDeltaNetConvolution convolution =>
                SelectGatedDeltaNetConvolution(context, convolution),
            Nncase.TIR.NTT.GatedDeltaNetRecurrentCore recurrentCore =>
                SelectGatedDeltaNetRecurrentCore(context, recurrentCore),
            Nncase.TIR.NTT.SparseExpertsDown sparseExpertsDown =>
                SelectSparseExpertsDown(context, sparseExpertsDown),
            _ => null,
        };
    }

    private static TIRMicroKernelSelection SelectSparseExpertsDown(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.SparseExpertsDown down)
    {
        var activations = GetBuffer(context, 0, "activations");
        var expertIds = GetBuffer(context, 1, "expert ids");
        var downWeight = GetBuffer(context, 4, "down weight");
        var output = GetBuffer(context, 6, "output");
        RequireRank(activations, 3, down, "activations");
        RequireRank(expertIds, 2, down, "expert ids");
        RequireRank(downWeight, 3, down, "down weight");
        RequireRank(output, 2, down, "output");

        var activationType = GetScalarDataType(activations.ElemType);
        var weightType = GetScalarDataType(downWeight.ElemType);
        var outputType = GetScalarDataType(output.ElemType);
        if (activationType != DataTypes.BFloat16 ||
            weightType != DataTypes.BFloat16 ||
            outputType != DataTypes.BFloat16)
        {
            throw new NotSupportedException(
                $"SparseExpertsDown concatenated MMA requires BF16 activation, weight, and output, got " +
                $"{activations.ElemType}/{downWeight.ElemType}/{output.ElemType}.");
        }

        RequirePersistentBFloat16Mma(context.Machine);
        var activationDimensions = GetLocalDimensions(activations);
        var weightDimensions = GetLocalDimensions(downWeight);
        var outputDimensions = GetLocalDimensions(output);
        var localK = GetScalarExtent(activationDimensions[2], activations.ElemType);
        var localN = GetScalarExtent(outputDimensions[1], output.ElemType);
        if (GetMax(activationDimensions[1]) != down.NumTopK ||
            GetMax(GetLocalDimensions(expertIds)[1]) != down.NumTopK ||
            GetScalarExtent(weightDimensions[1], downWeight.ElemType) != localN ||
            GetScalarExtent(weightDimensions[2], downWeight.ElemType) != localK)
        {
            throw new InvalidOperationException(
                "SparseExpertsDown microkernel selection requires matching local route, N, and K extents.");
        }

        var expertBlockK = SelectPowerOfTwoDivisor(
            localK,
            Math.Min(localK, SparseExpertsDownMaximumStageK),
            "SparseExpertsDown expert K tile");
        if (expertBlockK < SparseExpertsDownMinimumExpertBlockK)
        {
            throw new NotSupportedException(
                $"SparseExpertsDown requires a local K tile of at least " +
                $"{SparseExpertsDownMinimumExpertBlockK}, got local K {localK} and tile {expertBlockK}.");
        }

        var routesPerStage = SelectPowerOfTwoDivisor(
            down.NumTopK,
            Math.Min(
                Math.Min(down.NumTopK, SparseExpertsDownMaximumRoutesPerStage),
                SparseExpertsDownMaximumStageK / expertBlockK),
            "SparseExpertsDown routes per stage");

        var blockN = SelectPowerOfTwoDivisor(
            localN,
            Math.Min(localN, SparseExpertsDownMaximumBlockN),
            "SparseExpertsDown N tile");
        if (blockN < 8)
        {
            throw new NotSupportedException(
                $"SparseExpertsDown requires a local N tile of at least 8, got local N {localN}.");
        }

        var stageK = checked(routesPerStage * expertBlockK);
        var numStages = SelectMaximumFittingAsyncStageCount(
            context.Machine,
            checked((long)blockN * stageK * weightType.SizeInBytes),
            "SparseExpertsDown");
        var workspace = new TIRSharedWorkspaceDescriptor(
            "weight_stage",
            new TensorType(
                weightType,
                new RankedShape(numStages, blockN, stageK)),
            NvidiaNvmmaSharedAlignmentBytes);
        return new(
            "triton.sparse_experts_down",
            "concatenated_mma_smem_pipeline",
            CreateParameters(
                    SparseExpertsDownBlockM,
                    blockN,
                    stageK,
                    numStages)
                .Add("routes_per_stage", routesPerStage)
                .Add("expert_block_k", expertBlockK),
            ImmutableArray.Create(workspace),
            new TIRTransferPipelineContract(
            [
                new TIRTransferPipelineChannel("weight", [1, 4], [0]),
            ]));
    }

    private static int SelectPowerOfTwoDivisor(long extent, long maximum, string context)
    {
        if (extent <= 0 || maximum <= 0)
        {
            throw new InvalidOperationException(
                $"{context} requires positive extent and maximum, got {extent}/{maximum}.");
        }

        var candidate = 1L;
        while (candidate <= maximum / 2)
        {
            candidate *= 2;
        }

        while (candidate > 1 && extent % candidate != 0)
        {
            candidate /= 2;
        }

        return checked((int)candidate);
    }

    private static int SelectPowerOfTwoAtMost(long extent, int maximum)
    {
        if (extent <= 0 || maximum <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(extent),
                $"Power-of-two tile bounds must be positive, got extent={extent}, maximum={maximum}.");
        }

        var bounded = Math.Min(extent, maximum);
        return checked((int)(1L << (63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)bounded))));
    }

    private static int SelectMaximumFittingAsyncStageCount(
        TargetMachineModel machine,
        long stageBytes,
        string context,
        long reservedSharedBytes = 0,
        int maximumUsefulStageCount = int.MaxValue)
    {
        var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared)
            ?? throw new NotSupportedException($"{context} requires Shared memory.");
        var parentSpace = machine.GetTilingParentMemorySpace(sharedSpace.TilingLevel);
        var transfer = machine.GetTransfer(parentSpace.Id, sharedSpace.Id);
        var asynchronous = transfer.Asynchronous ?? throw new NotSupportedException(
            $"{context} requires an asynchronous parent-to-Shared transfer.");
        var capacity = machine.GetMaximumUsableAllocationBytes(sharedSpace);
        foreach (var candidate in asynchronous.SupportedStageCounts
                     .Where(stageCount => stageCount <= maximumUsefulStageCount)
                     .OrderDescending())
        {
            var pipelineBytes = machine.GetAllocationSizeBytes(
                sharedSpace,
                checked(stageBytes * candidate));
            var reservedBytes = reservedSharedBytes == 0
                ? 0
                : machine.GetAllocationSizeBytes(sharedSpace, reservedSharedBytes);
            var requiredBytes = checked(pipelineBytes + reservedBytes);
            if (requiredBytes <= capacity)
            {
                return candidate;
            }
        }

        throw new NotSupportedException(
            $"{context} cannot fit one supported async pipeline: stage_bytes={stageBytes}, " +
            $"reserved_shared_bytes={reservedSharedBytes}, " +
            $"maximum_useful_stages={maximumUsefulStageCount}, " +
            $"supported_stages=[{string.Join(',', asynchronous.SupportedStageCounts)}], " +
            $"shared_capacity={capacity}.");
    }

    private static TIRMicroKernelSelection SelectGatedDeltaNetConvolution(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.GatedDeltaNetConvolution convolution)
    {
        var qkv = GetBuffer(context, 0, "QKV");
        var output = GetBuffer(context, 3, "QKV output");
        RequireRank(qkv, 2, convolution, "QKV");
        RequireRank(output, 2, convolution, "QKV output");
        if (GetScalarDataType(qkv.ElemType) != DataTypes.BFloat16 ||
            qkv.ElemType != output.ElemType)
        {
            throw new NotSupportedException(
                $"GatedDeltaNet convolution requires matching packed BF16 QKV and output, got " +
                $"{qkv.ElemType}/{output.ElemType}.");
        }

        var localN = GetScalarExtent(GetLocalDimensions(output)[1], output.ElemType);
        var consumerThreads = context.Machine.Execution.ThreadsPerBlock;
        if (context.Machine.Execution.Kind != BlockExecutionKind.PersistentGpuBlock ||
            consumerThreads > GatedDeltaNetConvolutionMaximumBlockN)
        {
            throw new NotSupportedException(
                $"GatedDeltaNet convolution requires a persistent GPU block with at most " +
                $"{GatedDeltaNetConvolutionMaximumBlockN} consumer threads, got " +
                $"{context.Machine.Execution.Kind}/{consumerThreads}.");
        }

        // Triton distributes a one-dimensional tensor over every consumer
        // warp in the enclosing persistent block. Keep at least one element
        // per thread; an undersized tile can leave only a subset of warps with
        // side-effecting state stores after warp-specialization lowering.
        var blockN = checked((int)Math.Min(
            GatedDeltaNetConvolutionMaximumBlockN,
            RoundUpPowerOfTwo(Math.Max(localN, consumerThreads))));
        return new(
            "triton.gated_delta_net",
            "convolution",
            CreateParameters(blockM: 1, blockN, blockK: 1, numStages: 1),
            ImmutableArray<TIRSharedWorkspaceDescriptor>.Empty,
            TransferPipeline: null);
    }

    private static TIRMicroKernelSelection SelectGatedDeltaNetRecurrentCore(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.GatedDeltaNetRecurrentCore recurrentCore)
    {
        var qkv = GetBuffer(context, 1, "QKV");
        var z = GetBuffer(context, 2, "Z");
        var projectionInput = GetBuffer(context, 3, "projection input");
        var bWeight = GetBuffer(context, 4, "B weight");
        var aWeight = GetBuffer(context, 5, "A weight");
        var coreScratch = GetBuffer(context, 9, "core scratch");
        var gatedOutput = GetBuffer(context, 10, "gated output");
        RequireRank(qkv, 2, recurrentCore, "QKV");
        RequireRank(z, 2, recurrentCore, "Z");
        RequireRank(projectionInput, 2, recurrentCore, "projection input");
        RequireRank(bWeight, 2, recurrentCore, "B weight");
        RequireRank(aWeight, 2, recurrentCore, "A weight");
        RequireRank(coreScratch, 2, recurrentCore, "core scratch");
        RequireRank(gatedOutput, 2, recurrentCore, "gated output");
        var inputType = GetScalarDataType(qkv.ElemType);
        var outputType = GetScalarDataType(gatedOutput.ElemType);
        if (inputType != DataTypes.BFloat16 ||
            GetScalarDataType(z.ElemType) != DataTypes.BFloat16 ||
            GetScalarDataType(projectionInput.ElemType) != DataTypes.BFloat16 ||
            GetScalarDataType(bWeight.ElemType) != DataTypes.BFloat16 ||
            GetScalarDataType(aWeight.ElemType) != DataTypes.BFloat16 ||
            coreScratch.ElemType != DataTypes.Float32 ||
            outputType != DataTypes.BFloat16)
        {
            throw new NotSupportedException(
                $"GatedDeltaNet recurrent core requires BF16 QKV/Z/projection input/A/B weights and output, got " +
                $"{qkv.ElemType}/{z.ElemType}/{projectionInput.ElemType}/" +
                $"{aWeight.ElemType}/{bWeight.ElemType}/{coreScratch.ElemType}/{gatedOutput.ElemType}.");
        }

        var projectionInputDimensions = GetLocalDimensions(projectionInput);
        var bWeightDimensions = GetLocalDimensions(bWeight);
        var aWeightDimensions = GetLocalDimensions(aWeight);
        var coreScratchDimensions = GetLocalDimensions(coreScratch);
        var hiddenSize = GetScalarExtent(projectionInputDimensions[1], projectionInput.ElemType);
        if (GetScalarExtent(bWeightDimensions[0], bWeight.ElemType) != recurrentCore.NumValueHeads ||
            GetScalarExtent(aWeightDimensions[0], aWeight.ElemType) != recurrentCore.NumValueHeads ||
            GetScalarExtent(bWeightDimensions[1], bWeight.ElemType) != hiddenSize ||
            GetScalarExtent(aWeightDimensions[1], aWeight.ElemType) != hiddenSize ||
            GetMax(coreScratchDimensions[0]) != recurrentCore.NumValueHeads ||
            GetMax(coreScratchDimensions[1]) != recurrentCore.ValueHeadDim)
        {
            throw new NotSupportedException(
                $"GatedDeltaNet recurrent A/B weights must have scalar shape " +
                $"[{recurrentCore.NumValueHeads}, {hiddenSize}].");
        }

        if (context.Machine.Execution.Kind != BlockExecutionKind.PersistentGpuBlock ||
            context.Machine.Execution.WorkersPerBlock != 8 ||
            context.Machine.Execution.WorkerWidth != 32)
        {
            throw new NotSupportedException(
                "GatedDeltaNet recurrent_core requires a persistent eight-warp GPU block.");
        }

        var localValueElements = GetScalarExtent(
            GetLocalDimensions(gatedOutput)[1],
            gatedOutput.ElemType);
        var blockN = SelectPowerOfTwoAtMost(
            localValueElements,
            GatedDeltaNetRecurrentCoreBlockN);
        if (localValueElements <= 0 ||
            localValueElements % GatedDeltaNetStateValueTile != 0)
        {
            throw new InvalidOperationException(
                "GatedDeltaNet recurrent core requires a non-empty local output aligned to state value tiles.");
        }

        if (recurrentCore.NumValueHeads % recurrentCore.NumKeyHeads != 0)
        {
            throw new InvalidOperationException(
                "GatedDeltaNet recurrent core requires an integral value-head to key-head ratio.");
        }

        var valueHeadsPerKeyHead = recurrentCore.NumValueHeads / recurrentCore.NumKeyHeads;
        var projectionBlockK = SelectPowerOfTwoAtMost(
            hiddenSize,
            GatedDeltaNetProjectionMaximumBlockK);
        if (projectionBlockK % GatedDeltaNetProjectionTmaKAtom != 0 ||
            hiddenSize % GatedDeltaNetProjectionTmaKAtom != 0)
        {
            throw new NotSupportedException(
                $"GatedDeltaNet recurrent projection requires hidden size and block K aligned to " +
                $"{GatedDeltaNetProjectionTmaKAtom} BF16 elements, got {hiddenSize}/{projectionBlockK}.");
        }

        var projectionHeadCapacity = checked((int)Math.Min(
            valueHeadsPerKeyHead,
            DivideRoundUp(
                checked(localValueElements + recurrentCore.ValueHeadDim -
                    GreatestCommonDivisor(localValueElements, recurrentCore.ValueHeadDim)),
                recurrentCore.ValueHeadDim)));
        if (projectionHeadCapacity <= 0)
        {
            throw new InvalidOperationException(
                "GatedDeltaNet recurrent projection requires at least one local value head.");
        }

        var projectionStageCapacity = RoundUpPowerOfTwo(valueHeadsPerKeyHead);
        var projectionScratchBytes = checked(2L * projectionStageCapacity * DataTypes.Float32.SizeInBytes);
        var projectionWeightStageBytes = checked(
            2L * projectionHeadCapacity * projectionBlockK * DataTypes.BFloat16.SizeInBytes);
        var numStages = SelectMaximumFittingAsyncStageCount(
            context.Machine,
            projectionWeightStageBytes,
            "GatedDeltaNet recurrent projection",
            projectionScratchBytes,
            maximumUsefulStageCount: checked(
                (int)DivideRoundUp(hiddenSize, projectionBlockK) + 1));
        var projectionWeightStageShape = new RankedShape(
            numStages,
            projectionHeadCapacity,
            projectionBlockK / GatedDeltaNetProjectionTmaKAtom,
            GatedDeltaNetProjectionTmaKAtom);
        var bProjectionStage = new TIRSharedWorkspaceDescriptor(
            "b_projection_stage",
            new TensorType(DataTypes.BFloat16, projectionWeightStageShape),
            NvidiaNvmmaSharedAlignmentBytes);
        var aProjectionStage = new TIRSharedWorkspaceDescriptor(
            "a_projection_stage",
            new TensorType(DataTypes.BFloat16, projectionWeightStageShape),
            NvidiaNvmmaSharedAlignmentBytes);
        var projectionStage = new TIRSharedWorkspaceDescriptor(
            "projection_stage",
            new TensorType(
                DataTypes.Float32,
                new RankedShape(2, projectionStageCapacity)),
            16);

        return new(
            "triton.gated_delta_net",
            "recurrent_core",
            CreateParameters(
                blockM: 1,
                blockN,
                blockK: projectionBlockK,
                numStages)
                .Add("state_value_tile", GatedDeltaNetStateValueTile)
                .Add("projection_head_capacity", projectionHeadCapacity)
                .Add("projection_tma_k_atom", GatedDeltaNetProjectionTmaKAtom),
            ImmutableArray.Create(bProjectionStage, aProjectionStage, projectionStage),
            new TIRTransferPipelineContract(
            [
                new TIRTransferPipelineChannel("projection", [4, 5], [0, 1]),
            ],
            [2]));
    }

    private static void RequirePersistentBFloat16Mma(TargetMachineModel machine)
    {
        if (machine.Execution.Kind != BlockExecutionKind.PersistentGpuBlock ||
            machine.Execution.WorkersPerBlock != 8 ||
            machine.Execution.WorkerWidth != 32 ||
            !machine.Compute.MatrixPrimitives.Any(
                primitive => primitive.Supports(DataTypes.BFloat16, DataTypes.BFloat16)))
        {
            throw new NotSupportedException(
                "The selected BF16 MMA microkernel requires a persistent GPU block with eight 32-lane " +
                "consumer warps and BF16 matrix-compute support.");
        }
    }

    private static long RoundUp(long value, long alignment)
        => checked(DivideRoundUp(value, alignment) * alignment);

    private static long GreatestCommonDivisor(long lhs, long rhs)
    {
        while (rhs != 0)
        {
            (lhs, rhs) = (rhs, lhs % rhs);
        }

        return Math.Abs(lhs);
    }

    private static TIRMicroKernelSelection SelectPagedAttentionPartial(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PagedAttentionPartial pagedAttention)
    {
        var config = GetPagedAttentionConfig(context, 1);
        var candidate = SelectPagedAttentionMicroKernelCandidate(
            context,
            pagedAttention,
            config);
        var parameters = CreatePagedAttentionParameters(
            config,
            candidate.BlockN,
            candidate.NumStages);
        if (!candidate.UsesTma)
        {
            return new(
                "triton.paged_attention_partial",
                candidate.Variant,
                parameters,
                ImmutableArray<TIRSharedWorkspaceDescriptor>.Empty,
                TransferPipeline: null);
        }

        var stageType = new TensorType(
            config.KVPrimType,
            new RankedShape(new[] { candidate.NumStages, 1, 1, candidate.BlockN, 1, config.HeadDim }));
        return new(
            "triton.paged_attention_partial",
            candidate.Variant,
            parameters,
            ImmutableArray.Create(
                new TIRSharedWorkspaceDescriptor(
                    "key_stage",
                    stageType,
                    NvidiaNvmmaSharedAlignmentBytes),
                new TIRSharedWorkspaceDescriptor(
                    "value_stage",
                    stageType,
                    NvidiaNvmmaSharedAlignmentBytes)),
            new TIRTransferPipelineContract(
            [
                new TIRTransferPipelineChannel("key", [1], [0]),
                new TIRTransferPipelineChannel("value", [1], [1]),
            ]));
    }

    private static PagedAttentionMicroKernelCandidate SelectPagedAttentionMicroKernelCandidate(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PagedAttentionPartial pagedAttention,
        IR.NN.IPagedAttentionConfig config)
    {
        var mma = CreatePagedAttentionMicroKernelCandidate(
            context.Machine,
            config,
            useSimt: false);
        if (!TryGetPagedAttentionDecodeGqaSimtGroupTile(
                context,
                pagedAttention,
                config,
                out var groupTile))
        {
            return mma;
        }

        var simt = CreatePagedAttentionMicroKernelCandidate(
            context.Machine,
            config,
            useSimt: true);
        var query = GetBuffer(context, 0, "query");
        var mmaCyclesPerTile = EstimatePagedAttentionMmaCyclesPerTile(
            context.Machine,
            query.ElemType,
            config.KVPrimType,
            groupTile,
            mma.BlockN,
            config.HeadDim);
        if (!double.IsFinite(mmaCyclesPerTile))
        {
            return simt;
        }

        // Compare equal logical KV spans. The transfer volume and softmax work
        // are then equal; this isolates the target-dependent QK/PV arithmetic
        // and reduction cost that differs between the two implementations.
        var comparisonSpan = LeastCommonMultiple(simt.BlockN, mma.BlockN);
        var simtCycles = EstimatePagedAttentionSimtCyclesPerTile(
                context.Machine,
                config.KVPrimType,
                groupTile,
                simt.BlockN,
                config.HeadDim) *
            (comparisonSpan / simt.BlockN);
        var mmaCycles = mmaCyclesPerTile * (comparisonSpan / mma.BlockN);
        return simtCycles < mmaCycles ? simt : mma;
    }

    private static PagedAttentionMicroKernelCandidate CreatePagedAttentionMicroKernelCandidate(
        TargetMachineModel machine,
        IR.NN.IPagedAttentionConfig config,
        bool useSimt)
    {
        var numStages = useSimt
            ? SimtPagedAttentionNumStages
            : MmaPagedAttentionNumStages;
        var maximumBlockN = useSimt
            ? SimtPagedAttentionMaximumBlockN
            : MmaPagedAttentionMaximumBlockN;
        var tmaBlockN = SelectPagedAttentionTmaBlockN(
            machine,
            config,
            numStages,
            maximumBlockN);
        var usesTma = tmaBlockN is not null;
        var blockN = tmaBlockN ?? SelectPagedAttentionPageLocalBlockN(config.BlockSize);
        var variant = (useSimt, usesTma) switch
        {
            (true, true) => "simt_tma_smem_pipeline",
            (true, false) => "simt_direct",
            (false, true) => "mma_tma_smem_pipeline",
            (false, false) => "mma_direct",
        };
        return new(variant, blockN, numStages, usesTma);
    }

    private static ImmutableDictionary<string, long> CreatePagedAttentionParameters(
        IR.NN.IPagedAttentionConfig config,
        int blockN,
        int numStages)
        => new Dictionary<string, long>(StringComparer.Ordinal)
        {
            ["block_m"] = 1,
            ["block_n"] = blockN,
            ["block_k"] = config.HeadDim,
            ["num_stages"] = numStages,
            ["head_dim"] = config.HeadDim,
            ["page_size"] = config.BlockSize,
        }.ToImmutableDictionary(StringComparer.Ordinal);

    private static int SelectPagedAttentionPageLocalBlockN(int pageSize)
    {
        if (pageSize <= 0)
        {
            throw new InvalidOperationException(
                $"PagedAttention cache page size must be positive, got {pageSize}.");
        }

        for (var candidate = 64; candidate >= 1; candidate /= 2)
        {
            if (pageSize % candidate == 0)
            {
                return candidate;
            }
        }

        throw new InvalidOperationException(
            $"PagedAttention cache page size {pageSize} has no legal block-N tile.");
    }

    private static int? SelectPagedAttentionTmaBlockN(
        TargetMachineModel machine,
        IR.NN.IPagedAttentionConfig config,
        int numStages,
        int maximumBlockN)
    {
        if (config.BlockSize <= 0)
        {
            throw new InvalidOperationException(
                $"PagedAttention cache page size must be positive, got {config.BlockSize}.");
        }

        if (maximumBlockN <= 0 || (maximumBlockN & (maximumBlockN - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maximumBlockN),
                maximumBlockN,
                "PagedAttention maximum block-N must be a positive power of two.");
        }

        for (var candidate = maximumBlockN; candidate >= 1; candidate /= 2)
        {
            if (!CanPartitionPagedAttentionTile(config.BlockSize, candidate))
            {
                continue;
            }

            if (CanUsePagedAttentionTmaPipeline(machine, config, candidate, numStages))
            {
                return candidate;
            }
        }

        return null;
    }

    private static bool CanPartitionPagedAttentionTile(int pageSize, int blockN)
        => pageSize % blockN == 0 || blockN % pageSize == 0;

    private static bool TryGetPagedAttentionDecodeGqaSimtGroupTile(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PagedAttentionPartial pagedAttention,
        IR.NN.IPagedAttentionConfig config,
        out int groupTile)
    {
        groupTile = 0;
        var query = GetBuffer(context, 0, "query");
        var layout = pagedAttention.Layout.ToArray();
        var seqAxis = Array.IndexOf(layout, IR.NN.AttentionDimKind.Seq);
        if (seqAxis < 0 || seqAxis >= query.Rank)
        {
            throw new InvalidOperationException(
                "PagedAttentionPartial layout must contain a valid sequence axis.");
        }

        if (pagedAttention.HiddenSize <= 0 ||
            pagedAttention.HiddenSize % config.HeadDim != 0 ||
            GetScalarDataType(query.ElemType) != DataTypes.BFloat16 ||
            GetVectorLaneCount(query.ElemType) != 8 ||
            !PagedAttentionCacheLayoutUtility.Analyze(
                config,
                IR.NN.AttentionCacheKind.Key,
                "Triton PagedAttention key cache").HasContiguousHeadDimension(8) ||
            !PagedAttentionCacheLayoutUtility.Analyze(
                config,
                IR.NN.AttentionCacheKind.Value,
                "Triton PagedAttention value cache").HasContiguousHeadDimension(8) ||
            GetMax(GetLocalDimensions(query)[seqAxis]) != 1)
        {
            return false;
        }

        var queryHeads = pagedAttention.HiddenSize / config.HeadDim;
        if (queryHeads % config.NumKVHeads != 0)
        {
            return false;
        }

        var groupSize = queryHeads / config.NumKVHeads;
        if (groupSize <= 1)
        {
            return false;
        }

        groupTile = 1;
        while (groupTile < groupSize)
        {
            groupTile = checked(groupTile * 2);
        }

        var consumerWarps = context.Machine.Execution.WorkersPerBlock;
        return groupTile <= consumerWarps && consumerWarps % groupTile == 0;
    }

    private static double EstimatePagedAttentionSimtCyclesPerTile(
        TargetMachineModel machine,
        DataType kvType,
        int groupTile,
        int blockN,
        int headDim)
    {
        var fmaCount = checked(2L * groupTile * blockN * headDim);
        var fmaCycles = fmaCount / Math.Max(1.0, machine.Compute.SimtFmaPerCycle);

        // QK reduces headDim values for each score and PV reduces blockN
        // probabilities for each output. Triton maps each reduction to a
        // power-of-two lane tree in the explicit SIMT product layout.
        var reductionSteps = Math.Ceiling(
            Math.Log2(Math.Max(2, machine.Execution.WorkerWidth)));
        var reductionOutputs = checked((long)groupTile * (blockN + headDim));
        var reductionCycles = reductionOutputs * reductionSteps /
            Math.Max(1.0, machine.Compute.ElementwiseElementsPerCycle);
        var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(
            space => machine.GetMemoryResource(space).Kind == TargetMemorySpaceKind.Shared);
        if (sharedSpace is null)
        {
            return double.PositiveInfinity;
        }

        // The explicit SIMT layout expands both K and V over the GQA head
        // group before loading them from Shared. Account for those physical
        // reads; global transfer traffic is equal between candidates and is
        // intentionally excluded from this block-local comparison.
        var sharedReadBytes = checked(
            2L * groupTile * blockN * headDim * kvType.SizeInBytes);
        var sharedReadCycles = sharedReadBytes /
            Math.Max(1.0, machine.GetMemoryResource(sharedSpace).ReadBytesPerCycle);
        var convertedElements = checked(2L * groupTile * blockN * headDim);
        var conversionCycles = convertedElements /
            Math.Max(1.0, machine.Compute.ElementwiseElementsPerCycle);
        return Math.Max(fmaCycles, sharedReadCycles) + conversionCycles + reductionCycles;
    }

    private static double EstimatePagedAttentionMmaCyclesPerTile(
        TargetMachineModel machine,
        DataType queryType,
        DataType kvType,
        int groupTile,
        int blockN,
        int headDim)
    {
        var candidates = machine.Compute.MatrixPrimitives
            .Where(primitive => primitive.Supports(queryType, kvType))
            .Select(primitive =>
                EstimatePagedAttentionMatrixCycles(
                    machine,
                    primitive,
                    groupTile,
                    blockN,
                    headDim) +
                EstimatePagedAttentionMatrixCycles(
                    machine,
                    primitive,
                    groupTile,
                    headDim,
                    blockN))
            .ToArray();
        return candidates.Length == 0
            ? double.PositiveInfinity
            : candidates.Min();
    }

    private static double EstimatePagedAttentionMatrixCycles(
        TargetMachineModel machine,
        MatrixComputePrimitiveSpec primitive,
        int m,
        int n,
        int k)
    {
        var accumulatorChains = checked(
            (double)DivideRoundUp(m, primitive.M) *
            DivideRoundUp(n, primitive.N));
        var dependentInstructions = DivideRoundUp(k, primitive.K);
        return MatrixComputeCostModel.EstimateCycles(
            primitive,
            accumulatorChains,
            dependentInstructions,
            machine.Execution);
    }

    private static int LeastCommonMultiple(int lhs, int rhs)
    {
        if (lhs <= 0 || rhs <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lhs),
                $"PagedAttention block-N values must be positive, got {lhs} and {rhs}.");
        }

        var a = lhs;
        var b = rhs;
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }

        return checked(lhs / a * rhs);
    }

    private static bool CanUsePagedAttentionTmaPipeline(
        TargetMachineModel machine,
        IR.NN.IPagedAttentionConfig config,
        int blockN,
        int numStages)
    {
        var keyLayout = PagedAttentionCacheLayoutUtility.Analyze(
            config,
            IR.NN.AttentionCacheKind.Key,
            "Triton PagedAttention key cache");
        var valueLayout = PagedAttentionCacheLayoutUtility.Analyze(
            config,
            IR.NN.AttentionCacheKind.Value,
            "Triton PagedAttention value cache");
        if (config.KVPrimType != DataTypes.BFloat16 ||
            config.HeadDim != 128 ||
            !CanPartitionPagedAttentionTile(config.BlockSize, blockN) ||
            !keyLayout.HasContiguousHeadDimension(8) ||
            !valueLayout.HasContiguousHeadDimension(8))
        {
            return false;
        }

        var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared);
        if (sharedSpace is null)
        {
            return false;
        }

        var stageBytes = checked(
            (long)numStages * blockN * config.HeadDim *
            config.KVPrimType.SizeInBytes);
        var requiredBytes = checked(stageBytes * 2);
        return machine.GetAllocationSizeBytes(sharedSpace, requiredBytes) <=
            machine.GetMaximumUsableAllocationBytes(sharedSpace);
    }

    private static IR.NN.IPagedAttentionConfig GetPagedAttentionConfig(
        TIRMicroKernelSelectionContext context,
        int index)
    {
        if ((uint)index < (uint)context.Arguments.Count &&
            context.Arguments[index].CheckedType is TensorType
            {
                DType: ReferenceType
                {
                    ElemType: IR.NN.PagedAttentionKVCacheType
                    {
                        Config: { } config,
                    },
                },
            })
        {
            return config;
        }

        throw new InvalidOperationException(
            $"TIR microkernel selector for {context.Op.GetType().Name} expects a " +
            $"paged-attention KV-cache object at argument {index}.");
    }

    private static TIRMicroKernelSelection SelectMatmul(
        TIRMicroKernelSelectionContext context,
        bool transposeA,
        bool transposeB,
        bool kMajorPacked,
        int lhsIndex,
        int rhsIndex,
        int outputIndex,
        bool enableCompleteConsumerLhsStage = false,
        long minimumCompleteConsumerLhsK = 4096,
        string family = "triton.matmul",
        string? fp8Variant = null,
        int blockFp8ReductionGroup = 0)
    {
        var lhs = GetBuffer(context, lhsIndex, "lhs");
        var rhs = GetBuffer(context, rhsIndex, "rhs");
        var output = GetBuffer(context, outputIndex, "output");
        RequireRank(lhs, 2, context.Op, "lhs");
        RequireRank(rhs, 2, context.Op, "rhs");
        RequireRank(output, 2, context.Op, "output");

        var lhsDimensions = GetLocalDimensions(lhs);
        var outputDimensions = GetLocalDimensions(output);
        var m = GetMax(outputDimensions[^2]);
        var n = GetScalarExtent(outputDimensions[^1], output.ElemType);
        var kDimension = lhsDimensions[transposeA ? ^2 : ^1];
        var k = GetScalarExtent(kDimension, lhs.ElemType);
        var consumerLhsStaging = enableCompleteConsumerLhsStage &&
            CanStageCompleteConsumerLhs(
                lhs,
                transposeA,
                m,
                kDimension,
                k,
                minimumCompleteConsumerLhsK,
                blockFp8ReductionGroup > 0
                    ? blockFp8ReductionGroup
                    : PackedGemvMaximumBlockK)
            ? ConsumerLhsStagingKind.CompleteK
            : ConsumerLhsStagingKind.None;
        var useAuxiliaryConsumer =
            consumerLhsStaging == ConsumerLhsStagingKind.CompleteK &&
            fp8Variant == "mma_block_fp8_smem_pipeline" &&
            n > PackedGemvMinimumBlockN;
        return CreateMatrixSelection(
            context.Machine,
            family,
            GetScalarDataType(lhs.ElemType),
            GetScalarDataType(rhs.ElemType),
            GetScalarDataType(output.ElemType),
            m,
            n,
            k,
            kDimension.IsFixed,
            kMajorPacked,
            sourceArgumentIndices: [rhsIndex],
            consumerLhsStaging: consumerLhsStaging,
            fp8Variant: fp8Variant,
            blockFp8ReductionGroup: blockFp8ReductionGroup,
            localNExtents: GetScalarLocalExtentProfile(output, output.Rank - 1),
            useAuxiliaryConsumer: useAuxiliaryConsumer);
    }

    private static TIRMicroKernelSelection SelectPackedMatMulSamplingPartial(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PackedMatMulSamplingPartial sampling)
    {
        var lhs = GetBuffer(context, 0, "lhs");
        var rhs = GetBuffer(context, 1, "rhs");
        var logits = GetBuffer(context, 3, "logits");
        var processedLogits = GetBuffer(context, 4, "processed_logits");
        var argMaxState = GetBuffer(context, 5, "argmax_state");
        RequireRank(lhs, 2, context.Op, "lhs");
        RequireRank(rhs, 2, context.Op, "rhs");
        RequireRank(logits, 2, context.Op, "logits");
        RequireRank(processedLogits, 2, context.Op, "processed_logits");
        RequireRank(argMaxState, 1, context.Op, "argmax_state");
        if (sampling.RhsLayout != IR.NTT.PackedMatMulRhsLayout.KMajor)
        {
            throw new NotSupportedException(
                "PackedMatMulSamplingPartial currently requires K-major packed weights.");
        }

        var localLogits = DistributedUtility.GetDividedTensorType(
            sampling.LogitsType,
            DistributedUtility.DivideFlags.MaxShape);
        if (localLogits.Shape is not RankedShape { Rank: 2 } logitsShape ||
            localLogits.DType != GetScalarDataType(sampling.PackedOutputType.TensorType.DType))
        {
            throw new InvalidOperationException(
                $"PackedMatMulSamplingPartial selector requires rank-2 scalar logits, got {localLogits}.");
        }

        if (GetScalarDataType(logits.ElemType) != localLogits.DType ||
            processedLogits.ElemType != DataTypes.Float32 ||
            argMaxState.ElemType != DataTypes.UInt64)
        {
            throw new InvalidOperationException(
                "PackedMatMulSamplingPartial requires scalar logits, FP32 processed logits, and UInt64 partial argmax state outputs.");
        }

        var localLogitsDimensions = logitsShape.Dimensions.ToArray();
        var m = GetMax(localLogitsDimensions[^2]);
        var n = GetMax(localLogitsDimensions[^1]);
        if (m != 1)
        {
            throw new NotSupportedException(
                $"PackedMatMulSamplingPartial currently requires GEMV with local M=1, got {m}.");
        }

        var lhsDimensions = GetLocalDimensions(lhs);
        var kDimension = lhsDimensions[^1];
        var k = GetScalarExtent(kDimension, lhs.ElemType);
        if (!TryGetPackedGemvPipelineConfiguration(
                context.Machine,
                "triton.matmul_sampling_partial",
                GetScalarDataType(lhs.ElemType),
                GetScalarDataType(rhs.ElemType),
                n,
                k,
                kDimension.IsFixed,
                kMajorPacked: true,
                rhsTilesPerGroup: 1,
                reservedSharedBytes: 0,
                variant: "simt_fma_smem_pipeline",
                blockFp8ReductionGroup: 0,
                consumerLhsStaging: ConsumerLhsStagingKind.None,
                localNExtents: [n],
                useAuxiliaryConsumer: false,
                out var pipelineCandidate))
        {
            throw new InvalidOperationException(
                $"PackedMatMulSamplingPartial cannot select a spill-free BF16 Shared-staged GEMV for " +
                $"local shape M={m}, N={n}, K={k}.");
        }

        var pipeline = pipelineCandidate.Configuration;

        const int nVector = 8;
        const int kAtom = 16;
        var rhsStage = new TIRSharedWorkspaceDescriptor(
            "rhs_stage",
            new TensorType(
                GetScalarDataType(rhs.ElemType),
                new RankedShape(
                    pipeline.NumStages,
                    pipeline.BlockK / kAtom * (pipeline.BlockN / nVector),
                    nVector * kAtom)),
            NvidiaNvmmaSharedAlignmentBytes);
        return new(
            "triton.matmul_sampling_partial",
            "simt_fma_smem_pipeline",
            CreateParameters(1, pipeline.BlockN, pipeline.BlockK, pipeline.NumStages),
            ImmutableArray.Create(rhsStage),
            new TIRTransferPipelineContract(
            [
                new TIRTransferPipelineChannel("weight", [1], [0]),
            ]));
    }

    private static TIRMicroKernelSelection SelectFusedLinear(
        TIRMicroKernelSelectionContext context,
        string family,
        int inputIndex,
        int weightIndex,
        int outputIndex)
    {
        var input = GetBuffer(context, inputIndex, "input");
        var weight = GetBuffer(context, weightIndex, "weight");
        var output = GetBuffer(context, outputIndex, "output");
        RequireRank(input, 2, context.Op, "input");
        RequireRank(weight, 2, context.Op, "weight");
        RequireRank(output, 2, context.Op, "output");

        var inputDimensions = GetLocalDimensions(input);
        var outputDimensions = GetLocalDimensions(output);
        var m = GetMax(outputDimensions[^2]);
        var n = GetScalarExtent(outputDimensions[^1], output.ElemType);
        var kDimension = inputDimensions[^1];
        var k = GetScalarExtent(kDimension, input.ElemType);
        return CreateMatrixSelection(
            context.Machine,
            family,
            GetScalarDataType(input.ElemType),
            GetScalarDataType(weight.ElemType),
            GetScalarDataType(output.ElemType),
            m,
            n,
            k,
            kDimension.IsFixed,
            kMajorPacked: false,
            sourceArgumentIndices: [weightIndex],
            localNExtents: GetScalarLocalExtentProfile(output, output.Rank - 1));
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

    private static TIRMicroKernelSelection SelectPackedQkv(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PackedQKVParallelLinearFusedRhs qkv)
    {
        var input = GetBuffer(context, 0, "input");
        var weight = GetBuffer(context, 1, "fused qkv weight");
        var qOutput = GetBuffer(context, 11, "q output");
        var kOutput = GetBuffer(context, 12, "k output");
        var vOutput = GetBuffer(context, 13, "v output");
        RequireRank(input, 2, context.Op, "input");
        RequireRank(weight, 2, context.Op, "fused qkv weight");
        RequireRank(qOutput, 2, context.Op, "q output");
        RequireRank(kOutput, 2, context.Op, "k output");
        RequireRank(vOutput, 2, context.Op, "v output");

        var inputDimensions = GetLocalDimensions(input);
        var outputDimensions = new[]
        {
            GetLocalDimensions(qOutput),
            GetLocalDimensions(kOutput),
            GetLocalDimensions(vOutput),
        };
        var m = GetMax(outputDimensions[0][^2]);
        if (outputDimensions.Skip(1).Any(dimensions => GetMax(dimensions[^2]) != m))
        {
            throw new InvalidOperationException(
                "PackedQKVParallelLinear microkernel selection requires Q/K/V to have the same local M extent.");
        }

        if (qkv.ProjectionNCapacities.Count != 3 || qkv.ProjectionNCapacities.Any(x => x <= 0))
        {
            throw new InvalidOperationException(
                "PackedQKVParallelLinearFusedRhs requires three positive projection N capacities.");
        }

        var n = checked(qkv.ProjectionNCapacities.Sum());
        var kDimension = inputDimensions[^1];
        var k = GetScalarExtent(kDimension, input.ElemType);
        if (TryGetPackedQkvMmaConfiguration(
                context,
                qkv,
                input,
                weight,
                qOutput,
                kOutput,
                vOutput,
                m,
                n,
                kDimension,
                k,
                out var mmaPipeline))
        {
            const int nVector = 8;
            const int kAtom = 16;
            var rhsStageKAtoms = mmaPipeline.BlockK / kAtom;
            var rhsStageNAtoms = mmaPipeline.BlockN / nVector;
            var rhsStageAtoms = checked(rhsStageKAtoms * rhsStageNAtoms);
            var rhsStage = new TIRSharedWorkspaceDescriptor(
                "rhs_stage",
                new TensorType(
                    DataTypes.BFloat16,
                    new RankedShape(
                        mmaPipeline.NumStages,
                        rhsStageAtoms,
                        nVector * kAtom)),
                NvidiaNvmmaSharedAlignmentBytes);
            var lhsStage = new TIRSharedWorkspaceDescriptor(
                "lhs_stage",
                new TensorType(
                    DataTypes.BFloat16,
                    new RankedShape(1, checked((int)k))),
                NvidiaNvmmaSharedAlignmentBytes);
            return new(
                "triton.qkv_parallel_linear",
                "mma_smem_pipeline",
                CreateParameters(
                    1,
                    mmaPipeline.BlockN,
                    mmaPipeline.BlockK,
                    mmaPipeline.NumStages),
                ImmutableArray.Create(rhsStage, lhsStage),
                new TIRTransferPipelineContract(
                [
                    new TIRTransferPipelineChannel("weight", [1], [0]),
                ],
                [1]));
        }

        return CreateMatrixSelection(
            context.Machine,
            "triton.qkv_parallel_linear",
            GetScalarDataType(input.ElemType),
            GetScalarDataType(weight.ElemType),
            GetScalarDataType(qOutput.ElemType),
            m,
            n,
            k,
            kDimension.IsFixed,
            qkv.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
            sourceArgumentIndices: [1],
            consumerLhsStaging: ConsumerLhsStagingKind.PerKTile);
    }

    private static bool TryGetPackedQkvMmaConfiguration(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PackedQKVParallelLinearFusedRhs qkv,
        TensorBufferOperand input,
        TensorBufferOperand weight,
        TensorBufferOperand qOutput,
        TensorBufferOperand kOutput,
        TensorBufferOperand vOutput,
        long m,
        long n,
        Dimension kDimension,
        long k,
        out PackedGemvPipelineConfiguration configuration)
    {
        configuration = default;
        var partial = qOutput.DistributedType?.Partial;
        var hasCanonicalPacking =
            input.ElemType == DataTypes.BFloat16 &&
            weight.ElemType is VectorType weightVector &&
            weightVector.Lanes.SequenceEqual([8, 2, 8]) &&
            qOutput.ElemType is VectorType qVector &&
            qVector.Lanes.SequenceEqual([8]) &&
            kOutput.ElemType is VectorType kVector &&
            kVector.Lanes.SequenceEqual([8]) &&
            vOutput.ElemType is VectorType vVector &&
            vVector.Lanes.SequenceEqual([8]);
        var hasContiguousReductionAxis =
            input.Strides is { } inputStrides &&
            inputStrides[^1].IsFixed &&
            inputStrides[^1].FixedValue == 1;
        var supportsWarpMma = context.Machine.Compute.MatrixPrimitives.Any(
            primitive =>
                primitive.M == 16 &&
                primitive.N == 8 &&
                primitive.K == 16 &&
                primitive.CooperativeWorkers == 1 &&
                primitive.Supports(DataTypes.BFloat16, DataTypes.BFloat16));
        var commonRequirements =
            qkv.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor &&
            hasCanonicalPacking &&
            hasContiguousReductionAxis &&
            supportsWarpMma &&
            GetScalarDataType(input.ElemType) == DataTypes.BFloat16 &&
            GetScalarDataType(weight.ElemType) == DataTypes.BFloat16 &&
            GetScalarDataType(qOutput.ElemType) == DataTypes.BFloat16 &&
            GetScalarDataType(kOutput.ElemType) == DataTypes.BFloat16 &&
            GetScalarDataType(vOutput.ElemType) == DataTypes.BFloat16 &&
            m == 1 &&
            kDimension.IsFixed &&
            context.Arguments.Skip(2).Take(9).All(argument => argument is None) &&
            context.Machine.Execution.Kind == BlockExecutionKind.PersistentGpuBlock &&
            context.Machine.Execution.WorkersPerBlock == 8 &&
            context.Machine.Execution.WorkerWidth == 32;
        if (!commonRequirements)
        {
            return false;
        }

        if (n == PackedQkvSplitKMmaBlockN &&
            k == 256 &&
            partial is { Op: ReduceOp.Sum } &&
            HasSameSumPartial(kOutput, partial) &&
            HasSameSumPartial(vOutput, partial))
        {
            configuration = new(
                PackedQkvSplitKMmaBlockN,
                PackedQkvSplitKMmaBlockK,
                PackedQkvSplitKMmaNumStages);
        }
        else if (n == PackedQkvDirectMmaBlockN &&
                 k == PackedQkvDirectMmaInputK &&
                 partial is null &&
                 kOutput.DistributedType?.Partial is null &&
                 vOutput.DistributedType?.Partial is null)
        {
            configuration = new(
                PackedQkvDirectMmaBlockN,
                PackedQkvDirectMmaBlockK,
                PackedQkvDirectMmaNumStages);
        }
        else
        {
            return false;
        }

        var sharedSpace = context.Machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared);
        if (sharedSpace is null)
        {
            configuration = default;
            return false;
        }

        var parentSpace = context.Machine.GetTilingParentMemorySpace(sharedSpace.TilingLevel);
        var transfer = context.Machine.GetTransfer(parentSpace.Id, sharedSpace.Id);
        if (transfer.Asynchronous is not { } asynchronousTransfer ||
            (configuration.NumStages > 1 &&
             !asynchronousTransfer.SupportsStageCount(configuration.NumStages)))
        {
            configuration = default;
            return false;
        }

        const long elementBytes = 2;
        var requiredSharedBytes = checked(
            ((long)configuration.NumStages * configuration.BlockN * configuration.BlockK * elementBytes) +
            (k * elementBytes));
        var allocatedSharedBytes = context.Machine.GetAllocationSizeBytes(
            sharedSpace,
            requiredSharedBytes);
        if (allocatedSharedBytes > context.Machine.GetMaximumUsableAllocationBytes(sharedSpace))
        {
            configuration = default;
            return false;
        }

        return true;
    }

    private static bool HasSameSumPartial(
        TensorBufferOperand output,
        SBPPartial expected)
        => output.DistributedType?.Partial is { Op: ReduceOp.Sum } actual &&
            actual.Axes.SequenceEqual(expected.Axes);

    private static TIRMicroKernelSelection SelectPackedMatMulGlu(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PackedMatMulGlu matmulGlu)
    {
        var input = GetBuffer(context, 0, "input");
        var gateWeight = GetBuffer(context, 1, "gate weight");
        var upWeight = GetBuffer(context, 2, "up weight");
        var outputIndex = matmulGlu.EmitPartialResults
            ? Nncase.TIR.NTT.PackedMatMulGlu.GateOutput.Index
            : Nncase.TIR.NTT.PackedMatMulGlu.Output.Index;
        var output = GetBuffer(context, outputIndex, "output");
        RequireRank(input, 2, context.Op, "input");
        RequireRank(gateWeight, 2, context.Op, "gate weight");
        RequireRank(upWeight, 2, context.Op, "up weight");
        RequireRank(output, 2, context.Op, "output");
        if (matmulGlu.EmitPartialResults)
        {
            var upOutput = GetBuffer(
                context,
                Nncase.TIR.NTT.PackedMatMulGlu.UpOutput.Index,
                "up output");
            RequireRank(upOutput, 2, context.Op, "up output");
            if (!SameLocalShape(output, upOutput) || output.ElemType != upOutput.ElemType)
            {
                throw new InvalidOperationException(
                    "PackedMatMulGlu split-K gate/up outputs must have matching local shape and dtype.");
            }
        }

        var gateDimensions = GetLocalDimensions(gateWeight);
        var upDimensions = GetLocalDimensions(upWeight);
        if (gateDimensions.Length != upDimensions.Length ||
            gateDimensions.Where((dimension, index) => GetMax(dimension) != GetMax(upDimensions[index])).Any())
        {
            throw new InvalidOperationException(
                "PackedMatMulGlu microkernel selection requires gate/up weights to have the same local shape.");
        }

        if (GetScalarDataType(gateWeight.ElemType) != GetScalarDataType(upWeight.ElemType))
        {
            throw new InvalidOperationException(
                "PackedMatMulGlu microkernel selection requires gate/up weights to have the same scalar dtype.");
        }

        var inputDimensions = GetLocalDimensions(input);
        var outputDimensions = GetLocalDimensions(output);
        var m = GetMax(outputDimensions[^2]);
        var n = GetScalarExtent(outputDimensions[^1], output.ElemType);
        var kDimension = inputDimensions[^1];
        var k = GetScalarExtent(kDimension, input.ElemType);
        var consumerLhsStaging = n > PackedGemvMinimumBlockN &&
            CanStageCompleteConsumerLhs(
                input,
                false,
                m,
                kDimension,
                k,
                minimumK: PackedGemvMaximumBlockK)
            ? ConsumerLhsStagingKind.CompleteK
            : ConsumerLhsStagingKind.None;
        var useAuxiliaryConsumer =
            !matmulGlu.EmitPartialResults &&
            matmulGlu.QuantizationMode == IR.Math.MatMulQuantizationMode.DynamicBlock &&
            matmulGlu.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajorKPacked &&
            consumerLhsStaging == ConsumerLhsStagingKind.CompleteK &&
            n > PackedGemvMinimumBlockN;
        return CreateMatrixSelection(
            context.Machine,
            "triton.matmul_glu",
            GetScalarDataType(input.ElemType),
            GetScalarDataType(gateWeight.ElemType),
            GetScalarDataType(output.ElemType),
            m,
            n,
            k,
            kDimension.IsFixed,
            matmulGlu.RhsLayout is
                IR.NTT.PackedMatMulRhsLayout.KMajor or
                IR.NTT.PackedMatMulRhsLayout.NMajorKPacked,
            simultaneousRhsTileCount: 2,
            sourceArgumentIndices: [1, 2],
            fp8Variant: matmulGlu.QuantizationMode == IR.Math.MatMulQuantizationMode.DynamicBlock
                ? matmulGlu.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajorKPacked
                    ? "mma_n_major_block_fp8_smem_pipeline"
                    : "mma_block_fp8_smem_pipeline"
                : null,
            consumerLhsStaging: consumerLhsStaging,
            blockFp8ReductionGroup: matmulGlu.QuantizationMode == IR.Math.MatMulQuantizationMode.DynamicBlock
                ? checked((int)matmulGlu.WeightBlockK)
                : 0,
            nMajorKPacked: matmulGlu.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajorKPacked,
            localNExtents: GetScalarLocalExtentProfile(output, output.Rank - 1),
            useAuxiliaryConsumer: useAuxiliaryConsumer);
    }

    private static TIRMicroKernelSelection SelectNVFP4MatMulGlu(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.NVFP4MatMulGlu op)
    {
        var selection = SelectNVFP4MatMul(
            context,
            op.GroupSize,
            Nncase.TIR.NTT.NVFP4MatMulGlu.Input.Index,
            Nncase.TIR.NTT.NVFP4MatMulGlu.GateWeightPacked.Index,
            Nncase.TIR.NTT.NVFP4MatMulGlu.GateWeightScale.Index,
            Nncase.TIR.NTT.NVFP4MatMulGlu.Output.Index,
            "triton.nvfp4_matmul_glu",
            [
                Nncase.TIR.NTT.NVFP4MatMulGlu.GateWeightPacked.Index,
                Nncase.TIR.NTT.NVFP4MatMulGlu.UpWeightPacked.Index,
            ]);
        var upWeight = GetBuffer(
            context,
            Nncase.TIR.NTT.NVFP4MatMulGlu.UpWeightPacked.Index,
            "up packed weight");
        var upScale = GetBuffer(
            context,
            Nncase.TIR.NTT.NVFP4MatMulGlu.UpWeightScale.Index,
            "up weight scale");
        var gateWeight = GetBuffer(
            context,
            Nncase.TIR.NTT.NVFP4MatMulGlu.GateWeightPacked.Index,
            "gate packed weight");
        var gateScale = GetBuffer(
            context,
            Nncase.TIR.NTT.NVFP4MatMulGlu.GateWeightScale.Index,
            "gate weight scale");
        if (!SameLocalShape(gateWeight, upWeight) || !SameLocalShape(gateScale, upScale))
        {
            throw new InvalidOperationException(
                "NVFP4MatMulGlu requires gate/up packed weights and block scales to have matching local shapes.");
        }

        return selection;
    }

    private static TIRMicroKernelSelection SelectNVFP4MatMul(
        TIRMicroKernelSelectionContext context,
        long groupSize,
        int lhsIndex,
        int rhsIndex,
        int rhsScaleIndex,
        int outputIndex,
        string family,
        int[] transferSourceArgumentIndices)
    {
        if (groupSize != 16)
        {
            throw new NotSupportedException(
                $"{family} supports the NVFP4 group size 16, got {groupSize}.");
        }

        RequirePersistentBFloat16Mma(context.Machine);
        var lhs = GetBuffer(context, lhsIndex, "lhs");
        var rhs = GetBuffer(context, rhsIndex, "packed rhs");
        var rhsScale = GetBuffer(context, rhsScaleIndex, "rhs block scale");
        var output = GetBuffer(context, outputIndex, "output");
        RequireRank(lhs, 2, context.Op, "lhs");
        RequireRank(rhs, 2, context.Op, "packed rhs");
        RequireRank(rhsScale, 2, context.Op, "rhs block scale");
        RequireRank(output, 2, context.Op, "output");
        if (GetScalarDataType(lhs.ElemType) != DataTypes.BFloat16 ||
            GetScalarDataType(rhs.ElemType) != DataTypes.UInt8 ||
            GetScalarDataType(rhsScale.ElemType) != DataTypes.Float8E4M3 ||
            GetScalarDataType(output.ElemType) != DataTypes.BFloat16)
        {
            throw new NotSupportedException(
                $"{family} requires BF16 lhs/output, U8 packed E2M1 rhs, and E4M3 block scales, got " +
                $"{lhs.ElemType}/{rhs.ElemType}/{rhsScale.ElemType}/{output.ElemType}.");
        }

        RequireVectorLanes(lhs.ElemType, [8], family, "lhs");
        RequireVectorLanes(rhs.ElemType, [2, 16], family, "packed rhs");
        RequireVectorLanes(rhsScale.ElemType, [], family, "rhs block scale");
        RequireVectorLanes(output.ElemType, [8], family, "output");

        var lhsShape = GetLocalDimensions(lhs);
        var rhsShape = GetLocalDimensions(rhs);
        var scaleShape = GetLocalDimensions(rhsScale);
        var outputShape = GetLocalDimensions(output);
        var m = GetMax(outputShape[0]);
        var n = GetScalarExtent(outputShape[1], output.ElemType);
        var k = GetScalarExtent(lhsShape[1], lhs.ElemType);
        if (m != 1)
        {
            throw new NotSupportedException(
                $"{family} currently selects the decode GEMV algorithm only, got local M={m}.");
        }

        if (k % groupSize != 0 ||
            GetMax(lhsShape[0]) != m ||
            GetMax(rhsShape[0]) != n || GetScalarExtent(rhsShape[1], rhs.ElemType) * 2 != k ||
            GetMax(scaleShape[0]) != n || GetMax(scaleShape[1]) * groupSize != k)
        {
            throw new InvalidOperationException(
                $"{family} local storage contract must be lhs=bf16<8>[1,K/8], " +
                $"rhs=u8<2,16>[N,K/64], scale=[N,K/{groupSize}], " +
                $"output=bf16<8>[1,N/8].");
        }

        var selectedBlockN = SelectPowerOfTwoAtMost(n, NVFP4MaximumBlockN);
        var sharedSpace = context.Machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared)
            ?? throw new NotSupportedException($"{family} requires Shared memory.");
        var parentSpace = context.Machine.GetTilingParentMemorySpace(sharedSpace.TilingLevel);
        var asynchronousTransfer = context.Machine
            .GetTransfer(parentSpace.Id, sharedSpace.Id)
            .Asynchronous
            ?? throw new NotSupportedException($"{family} requires an asynchronous parent-to-Shared transfer.");
        var maximumTransactionBlockK = checked(
            (int)(asynchronousTransfer.MaximumTransactionBytes * 2L / selectedBlockN));
        var selectedBlockK = SelectPowerOfTwoAtMost(
            k,
            Math.Min(NVFP4BlockK, maximumTransactionBlockK));
        if (selectedBlockK < groupSize)
        {
            throw new NotSupportedException(
                $"{family} requires local K to admit at least one group-{groupSize} MMA tile, got K={k}.");
        }

        var stageBytes = checked(
            (long)selectedBlockN *
            (selectedBlockK / 2L));
        var numStages = SelectMaximumFittingAsyncStageCount(
            context.Machine,
            stageBytes,
            family);
        var packedWeightStage = new TIRSharedWorkspaceDescriptor(
            "packed_weight_stage",
            new TensorType(
                DataTypes.UInt8,
                new RankedShape(numStages, selectedBlockN, selectedBlockK / 2)),
            NvidiaNvmmaSharedAlignmentBytes);
        return new(
            family,
            "mma_tma_smem_pipeline",
            CreateParameters(1, selectedBlockN, selectedBlockK, numStages)
                .Add("group_size", groupSize),
            ImmutableArray.Create(packedWeightStage),
            new TIRTransferPipelineContract(
            [
                new TIRTransferPipelineChannel(
                    "weight",
                    transferSourceArgumentIndices,
                    [0],
                    sourceAlignmentBytes: 16),
            ]));
    }

    private static bool SameLocalShape(TensorBufferOperand lhs, TensorBufferOperand rhs)
    {
        var lhsShape = GetLocalDimensions(lhs);
        var rhsShape = GetLocalDimensions(rhs);
        return lhsShape.Length == rhsShape.Length &&
            lhsShape.Where((dimension, index) => GetMax(dimension) != GetMax(rhsShape[index])).Any() is false;
    }

    private static TIRMicroKernelSelection SelectSumma(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.SUMMA summa)
    {
        var lhs = GetBuffer(context, 0, "lhs");
        var rhs = GetBuffer(context, 1, "rhs");
        RequireRank(lhs, 2, context.Op, "lhs");
        RequireRank(rhs, 2, context.Op, "rhs");
        return CreateSelection(
            "triton.summa",
            "dot",
            blockM: 16,
            blockN: 16,
            blockK: 32,
            GetScalarDataType(lhs.ElemType),
            GetScalarDataType(rhs.ElemType),
            reserveMatrixOperands: true);
    }

    private static TIRMicroKernelSelection CreateMatrixSelection(
        TargetMachineModel machine,
        string family,
        DataType lhsType,
        DataType rhsType,
        DataType outputType,
        long m,
        long n,
        long k,
        bool fixedK,
        bool kMajorPacked,
        int simultaneousRhsTileCount = 1,
        IReadOnlyList<int>? sourceArgumentIndices = null,
        ConsumerLhsStagingKind consumerLhsStaging = ConsumerLhsStagingKind.None,
        string? fp8Variant = null,
        int blockFp8ReductionGroup = 0,
        bool nMajorKPacked = false,
        IReadOnlyList<long>? localNExtents = null,
        bool useAuxiliaryConsumer = false)
    {
        var gemv = m == 1;
        if (gemv)
        {
            var requestedVariant = rhsType == DataTypes.BFloat16
                ? "simt_fma_smem_pipeline"
                : fp8Variant ?? "simt_fp8_fma_smem_pipeline";
            if (TryGetPackedGemvPipelineConfiguration(
                    machine,
                    family,
                    lhsType,
                    rhsType,
                    n,
                    k,
                    fixedK,
                    kMajorPacked,
                    simultaneousRhsTileCount,
                    reservedSharedBytes: 0,
                    variant: requestedVariant,
                    blockFp8ReductionGroup: blockFp8ReductionGroup,
                    consumerLhsStaging: consumerLhsStaging,
                    localNExtents: localNExtents ?? [n],
                    useAuxiliaryConsumer: useAuxiliaryConsumer,
                    out var selectedPipeline))
            {
                var variant = requestedVariant;
                var pipeline = selectedPipeline.Configuration;
                if (outputType != DataTypes.BFloat16)
                {
                    throw new NotSupportedException(
                        $"{family} Shared-staged GEMV requires BF16 output, got {outputType}.");
                }

                var nVector = 16 / outputType.SizeInBytes;
                var kVector = 16 / rhsType.SizeInBytes;
                const int kPack = 2;
                var kAtom = kPack * kVector;
                var blockFp8TransferBlockK = 0;
                var directBlockFp8Mma = IsBlockFp8MmaVariant(variant);
                var materializeCompleteBlockFp8Lhs =
                    pipeline.PrequantizeLhs ||
                    consumerLhsStaging == ConsumerLhsStagingKind.CompleteK;
                var mergeBlockFp8ScaleGroups =
                    variant == "mma_n_major_block_fp8_smem_pipeline" ||
                    (family == "triton.matmul" &&
                     variant == "mma_block_fp8_smem_pipeline");
                if (directBlockFp8Mma &&
                    (blockFp8ReductionGroup <= 0 ||
                     pipeline.BlockK % blockFp8ReductionGroup != 0))
                {
                    throw new InvalidOperationException(
                        $"{family}/{variant} requires block_k divisible by its positive " +
                        $"block-FP8 reduction group, got block_k={pipeline.BlockK}, " +
                        $"group={blockFp8ReductionGroup}.");
                }

                if (directBlockFp8Mma)
                {
                    blockFp8TransferBlockK = mergeBlockFp8ScaleGroups
                        ? GetMergedBlockFp8MmaTransferBlockK(
                            machine,
                            pipeline.BlockN,
                            pipeline.BlockK,
                            blockFp8ReductionGroup,
                            rhsType.SizeInBytes)
                        : Math.Min(
                            blockFp8ReductionGroup,
                            BlockFp8MmaMaximumTransferBlockK);
                    if (useAuxiliaryConsumer)
                    {
                        // Auxiliary roles cannot consume a merged scale group
                        // until the whole transaction completes. Keep the first
                        // ready unit bounded to one reduction group.
                        blockFp8TransferBlockK = Math.Min(
                            blockFp8TransferBlockK,
                            BlockFp8MmaMaximumTransferBlockK);
                    }

                    if ((!mergeBlockFp8ScaleGroups &&
                         blockFp8ReductionGroup % blockFp8TransferBlockK != 0) ||
                        (mergeBlockFp8ScaleGroups &&
                         (blockFp8TransferBlockK % blockFp8ReductionGroup != 0 ||
                          pipeline.BlockK % blockFp8TransferBlockK != 0)))
                    {
                        throw new InvalidOperationException(
                            $"{family}/{variant} selected incompatible block/scale/transfer K " +
                            $"{pipeline.BlockK}/{blockFp8ReductionGroup}/{blockFp8TransferBlockK}.");
                    }
                }

                var useKMajorBlockFp8MmaStage =
                    family == "triton.matmul_glu" &&
                    variant == "mma_block_fp8_smem_pipeline" &&
                    kMajorPacked &&
                    !nMajorKPacked;
                var rhsShape = new TensorType(
                    rhsType,
                    new RankedShape(
                        directBlockFp8Mma &&
                        !useKMajorBlockFp8MmaStage
                            ? new[]
                            {
                                pipeline.NumStages,
                                mergeBlockFp8ScaleGroups
                                    ? pipeline.BlockK / blockFp8TransferBlockK
                                    : pipeline.BlockK / blockFp8ReductionGroup,
                                mergeBlockFp8ScaleGroups
                                    ? 1
                                    : blockFp8ReductionGroup / blockFp8TransferBlockK,
                                pipeline.BlockN,
                                blockFp8TransferBlockK,
                            }
                            : new[]
                            {
                                pipeline.NumStages,
                                pipeline.BlockK / kAtom * (pipeline.BlockN / nVector),
                                nVector * kAtom,
                            }));
                var workspaces = ImmutableArray.CreateBuilder<TIRSharedWorkspaceDescriptor>();
                workspaces.Add(
                    new TIRSharedWorkspaceDescriptor(
                        "rhs_stage",
                        rhsShape,
                        NvidiaNvmmaSharedAlignmentBytes));
                var lhsStageExtent = GetConsumerLhsStageExtent(
                    consumerLhsStaging,
                    k,
                    pipeline.BlockK);
                if (directBlockFp8Mma)
                {
                    var activationExtent = materializeCompleteBlockFp8Lhs
                        ? k
                        : pipeline.BlockK;
                    var logicalActivationGroupCount = checked(
                        activationExtent / blockFp8ReductionGroup);
                    var activationGroupCount = materializeCompleteBlockFp8Lhs
                        ? RoundUpPowerOfTwo(logicalActivationGroupCount)
                        : logicalActivationGroupCount;
                    workspaces.Add(
                        new TIRSharedWorkspaceDescriptor(
                            "lhs_quantized",
                            new TensorType(
                                DataTypes.Float8E4M3,
                                new RankedShape(new[]
                                {
                                    activationGroupCount,
                                    blockFp8ReductionGroup,
                                })),
                            TritonSharedVectorAlignmentBytes));
                    workspaces.Add(
                        new TIRSharedWorkspaceDescriptor(
                            "lhs_scale",
                            new TensorType(
                                DataTypes.Float32,
                                new RankedShape(new[] { activationGroupCount, 1L })),
                            TritonSharedVectorAlignmentBytes));
                }

                var lhsStageWorkspaceIndex = -1;
                if (lhsStageExtent > 0)
                {
                    lhsStageWorkspaceIndex = workspaces.Count;
                    workspaces.Add(new TIRSharedWorkspaceDescriptor(
                        "lhs_stage",
                        new TensorType(
                            lhsType,
                            new RankedShape(new[] { 1, lhsStageExtent })),
                        NvidiaNvmmaSharedAlignmentBytes));
                }

                var parameters = CreateParameters(
                    1,
                    pipeline.BlockN,
                    pipeline.BlockK,
                    pipeline.NumStages);
                if (directBlockFp8Mma)
                {
                    parameters = parameters.Add("transfer_block_k", blockFp8TransferBlockK);
                }

                if (pipeline.PrequantizeLhs)
                {
                    parameters = parameters.Add("prequantize_lhs", 1);
                }

                if (consumerLhsStaging == ConsumerLhsStagingKind.CompleteK)
                {
                    parameters = parameters.Add("lhs_stage_extent", lhsStageExtent);
                }

                if (useAuxiliaryConsumer)
                {
                    parameters = parameters.Add("fragment_accumulate_mma", 1);
                }

                var consumerSharedWorkspaceIndices = ImmutableArray.CreateBuilder<int>();
                if (directBlockFp8Mma)
                {
                    consumerSharedWorkspaceIndices.Add(1);
                    consumerSharedWorkspaceIndices.Add(2);
                }

                if (lhsStageWorkspaceIndex >= 0)
                {
                    consumerSharedWorkspaceIndices.Add(lhsStageWorkspaceIndex);
                }

                var auxiliarySharedWorkspaceIndices = family == "triton.matmul_glu"
                    ? ImmutableArray.Create(1, 2)
                    : consumerSharedWorkspaceIndices.ToImmutable();
                return new(
                    family,
                    variant,
                    parameters,
                    workspaces.ToImmutable(),
                    new TIRTransferPipelineContract(
                    [
                        new TIRTransferPipelineChannel(
                            "weight",
                            sourceArgumentIndices ?? throw new InvalidOperationException(
                                $"{family}/{variant} is missing transfer source operand indexes."),
                            [0]),
                    ],
                    consumerSharedWorkspaceIndices.ToImmutable(),
                    useAuxiliaryConsumer
                        ? new TIRAuxiliaryConsumerContract([0], auxiliarySharedWorkspaceIndices)
                        : null));
            }

            if (rhsType == DataTypes.Float8E4M3 || rhsType == DataTypes.Float8E5M2)
            {
                throw new NotSupportedException(
                    $"{family} has no legal Shared-staged FP8 GEMV configuration for M={m}, N={n}, K={k}.");
            }

            const int blockK = 256;
            var blockN = k <= blockK && n >= 4096 ? 128 : 32;
            return CreateSelection(
                family,
                "simt_fma",
                blockM: 1,
                blockN,
                blockK,
                lhsType,
                rhsType,
                reserveMatrixOperands: false);
        }

        if (rhsType == DataTypes.Float8E4M3 || rhsType == DataTypes.Float8E5M2)
        {
            throw new NotSupportedException(
                $"{family} FP8 prefill is not implemented; only decode GEMV (M=1) is supported.");
        }

        return CreateSelection(
            family,
            "mma",
            blockM: 16,
            blockN: 64,
            blockK: 64,
            lhsType,
            rhsType,
            reserveMatrixOperands: true);
    }

    private static TIRMicroKernelSelection CreateSelection(
        string family,
        string variant,
        int blockM,
        int blockN,
        int blockK,
        DataType lhsType,
        DataType rhsType,
        bool reserveMatrixOperands)
    {
        if (!reserveMatrixOperands)
        {
            return new(
                family,
                variant,
                CreateParameters(blockM, blockN, blockK, numStages: 1),
                ImmutableArray<TIRSharedWorkspaceDescriptor>.Empty,
                TransferPipeline: null);
        }

        var gemv = blockM == 1;
        var lhsShape = gemv
            ? new TensorType(lhsType, new RankedShape(new[] { blockK }))
            : new TensorType(lhsType, new RankedShape(new[] { blockM, blockK }));
        var rhsShape = gemv
            ? new TensorType(rhsType, new RankedShape(new[] { blockN, blockK }))
            : new TensorType(rhsType, new RankedShape(new[] { blockK, blockN }));
        return new(
            family,
            variant,
            CreateParameters(blockM, blockN, blockK, numStages: 1),
            ImmutableArray.Create(
                new TIRSharedWorkspaceDescriptor("lhs_stage", lhsShape, NvidiaNvmmaSharedAlignmentBytes),
                new TIRSharedWorkspaceDescriptor("rhs_stage", rhsShape, NvidiaNvmmaSharedAlignmentBytes)),
            TransferPipeline: null);
    }

    private static ImmutableDictionary<string, long> CreateParameters(
        int blockM,
        int blockN,
        int blockK,
        int numStages)
        => new Dictionary<string, long>(StringComparer.Ordinal)
        {
            ["block_m"] = blockM,
            ["block_n"] = blockN,
            ["block_k"] = blockK,
            ["num_stages"] = numStages,
        }.ToImmutableDictionary(StringComparer.Ordinal);

    private static bool CanStageCompleteConsumerLhs(
        TensorBufferOperand lhs,
        bool transposeA,
        long m,
        Dimension kDimension,
        long k,
        long minimumK = 4096,
        long requiredKMultiple = PackedGemvMaximumBlockK)
    {
        var reductionAxis = transposeA ? lhs.Rank - 2 : lhs.Rank - 1;
        if (lhs.Strides is not { } strides)
        {
            return false;
        }

        var reductionStride = strides[reductionAxis];
        return m == 1 &&
            kDimension.IsFixed &&
            k >= minimumK &&
            k % requiredKMultiple == 0 &&
            reductionStride.IsFixed &&
            reductionStride.FixedValue == 1 &&
            TryRoundUpToPowerOfTwo(k, out _);
    }

    private static int GetConsumerLhsStageExtent(
        ConsumerLhsStagingKind kind,
        long k,
        int blockK)
    {
        return kind switch
        {
            ConsumerLhsStagingKind.None => 0,
            ConsumerLhsStagingKind.PerKTile => blockK,
            ConsumerLhsStagingKind.CompleteK when TryRoundUpToPowerOfTwo(k, out var extent) => extent,
            ConsumerLhsStagingKind.CompleteK => throw new InvalidOperationException(
                $"Complete consumer LHS staging cannot represent K={k} as a positive int32 power-of-two extent."),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
        };
    }

    private static bool TryRoundUpToPowerOfTwo(long value, out int result)
    {
        result = 0;
        if (value <= 0 || value > (1L << 30))
        {
            return false;
        }

        var extent = 1;
        while (extent < value)
        {
            extent = checked(extent * 2);
        }

        result = extent;
        return true;
    }

    private static bool TryGetPackedGemvPipelineConfiguration(
        TargetMachineModel machine,
        string family,
        DataType lhsType,
        DataType rhsType,
        long n,
        long k,
        bool fixedK,
        bool kMajorPacked,
        int rhsTilesPerGroup,
        long reservedSharedBytes,
        string variant,
        int blockFp8ReductionGroup,
        ConsumerLhsStagingKind consumerLhsStaging,
        IReadOnlyList<long> localNExtents,
        bool useAuxiliaryConsumer,
        out PackedGemvPipelineCandidate result)
    {
        result = default;
        if ((family != "triton.matmul" &&
             family != "triton.paged_attention_merge_matmul" &&
             family != "triton.matmul_sampling_partial" &&
             family != "triton.qkv_parallel_linear" &&
             family != "triton.matmul_glu") ||
            rhsTilesPerGroup <= 0 ||
            !kMajorPacked ||
            lhsType != DataTypes.BFloat16 ||
            (rhsType != DataTypes.BFloat16 &&
             rhsType != DataTypes.Float8E4M3) ||
            n < 8 ||
            !fixedK ||
            k <= 0 ||
            k % PackedGemvMinimumBlockK != 0 ||
            machine.Execution.Kind != BlockExecutionKind.PersistentGpuBlock ||
            machine.Execution.WorkersPerBlock != 8 ||
            machine.Execution.WorkerWidth != 32 ||
            localNExtents.Count == 0 ||
            localNExtents.Any(extent => extent < 0) ||
            localNExtents.Max() != n)
        {
            return false;
        }

        var directBlockFp8Mma = IsBlockFp8MmaVariant(variant);
        MatrixComputePrimitiveSpec? matrixPrimitive = null;
        if (directBlockFp8Mma)
        {
            if (blockFp8ReductionGroup <= 0)
            {
                return false;
            }

            matrixPrimitive = machine.Compute.MatrixPrimitives
                .Where(primitive => primitive.Supports(rhsType, rhsType))
                .Where(primitive => primitive.Name == "mma" && primitive.CooperativeWorkers == 1)
                .OrderBy(primitive => primitive.M)
                .ThenBy(primitive => primitive.N)
                .FirstOrDefault();
            if (matrixPrimitive is null)
            {
                return false;
            }
        }

        var minimumBlockN = directBlockFp8Mma
            ? matrixPrimitive!.M
            : PackedGemvMinimumBlockN;
        var maximumSupportedBlockN = directBlockFp8Mma
            ? Math.Min(
                MmaPackedGemvMaximumBlockN,
                checked(
                    matrixPrimitive!.M *
                    Math.Max(
                        1,
                        machine.Execution.WorkersPerBlock /
                        checked(matrixPrimitive.CooperativeWorkers * rhsTilesPerGroup))))
            : SimtPackedGemvMaximumBlockN;
        var maximumBlockN = GetPackedGemvPipelineMaximumBlockN(
            n,
            minimumBlockN,
            maximumSupportedBlockN);
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

        PackedGemvPipelineCandidate? bestCandidate = null;
        var elementBytes = rhsType.SizeInBytes;

        // Search N and K jointly. The cycle model below charges ceil(N / blockN)
        // tiles, including padded transfer and FMA work, so small local N shards
        // can trade additional TMA issues for less tail computation.
        for (var candidateBlockN = minimumBlockN;
             candidateBlockN <= maximumBlockN;
             candidateBlockN *= 2)
        {
            if (variant == "mma_n_major_block_fp8_smem_pipeline" &&
                candidateBlockN < 64)
            {
                continue;
            }

            // Keep the statically lowered reduction body bounded across fused
            // projections. One 1024-K MMA tile is spill-free for a single RHS;
            // paired GLU projections have twice the dot/epilogue body and use a
            // proportionally smaller K tile under the same register contract.
            var maximumCandidateBlockK = directBlockFp8Mma
                ? Math.Max(
                    PackedGemvMinimumBlockK,
                    PackedGemvMaximumBlockK / rhsTilesPerGroup)
                : PackedGemvMaximumBlockK;
            if (useAuxiliaryConsumer)
            {
                // Auxiliary consumers start on the first ready slot. Keep that
                // first-wave transfer bounded; larger K tiles delay useful MMA
                // work even when their steady-state byte/cycle estimate ties.
                maximumCandidateBlockK = Math.Min(maximumCandidateBlockK, 256);
            }

            for (var candidateBlockK = PackedGemvMinimumBlockK;
                 candidateBlockK <= Math.Min(k, maximumCandidateBlockK);
                 candidateBlockK *= 2)
            {
                if (k % candidateBlockK != 0)
                {
                    continue;
                }

                if (directBlockFp8Mma &&
                    candidateBlockK % blockFp8ReductionGroup != 0)
                {
                    continue;
                }

                var stageBytes = checked((long)candidateBlockN * candidateBlockK * elementBytes);
                foreach (var physicalStageCount in asynchronousTransfer.SupportedStageCounts)
                {
                    // A physical pipe slot owns one staged RHS tile. Fused projections
                    // consume adjacent slots, so double buffering applies to complete
                    // RHS groups rather than individual slots.
                    if (physicalStageCount % rhsTilesPerGroup != 0 ||
                        physicalStageCount / rhsTilesPerGroup < PackedGemvMinimumLogicalStages)
                    {
                        continue;
                    }

                    var kTileCount = checked((int)(k / candidateBlockK));
                    var maximumUsefulPhysicalStages = checked(
                        rhsTilesPerGroup *
                        (useAuxiliaryConsumer
                            ? machine.Execution.WorkersPerBlock
                            : kTileCount + 1));
                    if (physicalStageCount > maximumUsefulPhysicalStages)
                    {
                        continue;
                    }

                    var localNTileCount = GetPackedGemvNTileCounts(
                        localNExtents,
                        candidateBlockN).Maximum;
                    var canPrequantizeLhs =
                        directBlockFp8Mma &&
                        consumerLhsStaging != ConsumerLhsStagingKind.CompleteK &&
                        localNTileCount > 1;
                    var prequantizationCandidates = canPrequantizeLhs
                        ? localNTileCount >= 3
                            ? new[] { true }
                            : new[] { false, true }
                        : new[] { false };
                    foreach (var prequantizeLhs in prequantizationCandidates)
                    {
                        var materializeCompleteBlockFp8Lhs =
                            prequantizeLhs ||
                            consumerLhsStaging == ConsumerLhsStagingKind.CompleteK;
                        var candidateReservedSharedBytes = checked(
                            reservedSharedBytes +
                            (directBlockFp8Mma
                                ? GetPackedBlockFp8MmaActivationSharedBytes(
                                    materializeCompleteBlockFp8Lhs
                                        ? k
                                        : candidateBlockK,
                                    blockFp8ReductionGroup,
                                    materializeCompleteBlockFp8Lhs)
                                : 0));
                        var lhsStageExtent = GetConsumerLhsStageExtent(
                            consumerLhsStaging,
                            k,
                            candidateBlockK);
                        var lhsStageBytes = checked((long)lhsStageExtent * lhsType.SizeInBytes);
                        var requiredSharedBytes = checked(
                            (physicalStageCount * stageBytes) + lhsStageBytes);
                        var allocatedSharedBytes = machine.GetAllocationSizeBytes(
                            sharedSpace,
                            checked(requiredSharedBytes + candidateReservedSharedBytes));
                        if (allocatedSharedBytes >
                            machine.GetMaximumUsableAllocationBytes(sharedSpace))
                        {
                            continue;
                        }

                        var candidateConfiguration = new PackedGemvPipelineConfiguration(
                            candidateBlockN,
                            candidateBlockK,
                            physicalStageCount,
                            prequantizeLhs);
                        var candidate = new PackedGemvPipelineCandidate(
                            candidateConfiguration,
                            directBlockFp8Mma
                                ? EstimatePackedBlockFp8MmaPipelineCycles(
                                    machine,
                                    transfer,
                                    machine.GetMemoryResource(sharedSpace),
                                    matrixPrimitive!,
                                    n,
                                    k,
                                    rhsType.SizeInBytes,
                                    blockFp8ReductionGroup,
                                    rhsTilesPerGroup,
                                    materializeCompleteBlockFp8Lhs,
                                    localNExtents,
                                    candidateConfiguration)
                                : EstimatePackedGemvPipelineCycles(
                                    machine,
                                    transfer,
                                    machine.GetMemoryResource(sharedSpace),
                                    n,
                                    k,
                                    rhsTilesPerGroup,
                                    rhsType.SizeInBytes,
                                    localNExtents,
                                    candidateConfiguration),
                            allocatedSharedBytes);
                        if (bestCandidate is null || IsBetterPackedGemvPipelineCandidate(
                                candidate,
                                bestCandidate.Value,
                                useAuxiliaryConsumer))
                        {
                            bestCandidate = candidate;
                        }
                    }
                }
            }
        }

        if (bestCandidate is null)
        {
            return false;
        }

        result = bestCandidate.Value;
        return true;
    }

    private static bool IsBlockFp8MmaVariant(string variant)
        => variant is "mma_block_fp8_smem_pipeline" or
            "mma_n_major_block_fp8_smem_pipeline";

    private static int GetMergedBlockFp8MmaTransferBlockK(
        TargetMachineModel machine,
        int blockN,
        int blockK,
        int reductionGroup,
        int elementBytes)
    {
        var sharedSpace = machine.MemorySpaces.Values.Single(
            space => space.TIRBinding?.Location == MemoryLocation.Shared);
        var parentSpace = machine.GetTilingParentMemorySpace(sharedSpace.TilingLevel);
        var asynchronous = machine.GetTransfer(parentSpace.Id, sharedSpace.Id).Asynchronous
            ?? throw new InvalidOperationException(
                "Block-FP8 MMA transfer merging requires an asynchronous Shared transfer.");
        var transactionLimitedK = checked(
            (int)(asynchronous.MaximumTransactionBytes / ((long)blockN * elementBytes)));
        var maximumTransferK = Math.Min(
            Math.Min(blockK, BlockFp8MmaMaximumMergedTransferBlockK),
            transactionLimitedK);
        var transferK = reductionGroup;
        while (transferK <= maximumTransferK / 2 && blockK % (transferK * 2) == 0)
        {
            transferK *= 2;
        }

        return transferK;
    }

    private static long GetPackedBlockFp8MmaActivationSharedBytes(
        long blockK,
        int reductionGroup,
        bool materializeCompleteLhs)
    {
        if (blockK <= 0 || reductionGroup <= 0 || blockK % reductionGroup != 0)
        {
            throw new InvalidOperationException(
                $"Block-FP8 MMA activation staging requires positive block K divisible by " +
                $"the reduction group, got block_k={blockK}, group={reductionGroup}.");
        }

        var logicalGroupCount = blockK / reductionGroup;

        // Complete activation materialization is indexed by a Triton tensor of
        // group ids. TLE local_ptr can only expose that leading Shared axis as
        // a tensor dimension when its physical extent is a power of two.
        var allocatedGroupCount = materializeCompleteLhs
            ? RoundUpPowerOfTwo(logicalGroupCount)
            : logicalGroupCount;
        var quantizedBytes = RoundUp(
            checked(
                allocatedGroupCount * reductionGroup *
                DataTypes.Float8E4M3.SizeInBytes),
            TritonSharedVectorAlignmentBytes);
        var scaleBytes = RoundUp(
            checked(allocatedGroupCount * DataTypes.Float32.SizeInBytes),
            TritonSharedVectorAlignmentBytes);
        return checked(quantizedBytes + scaleBytes);
    }

    private static long EstimatePackedGemvPipelineCycles(
        TargetMachineModel machine,
        TargetMemoryTransferSpec transfer,
        TargetMemoryResourceSpec sharedMemory,
        long n,
        long k,
        int rhsTilesPerGroup,
        int elementBytes,
        IReadOnlyList<long> localNExtents,
        PackedGemvPipelineConfiguration configuration)
    {
        var (nTileCount, totalNTileCount) = GetPackedGemvNTileCounts(
            localNExtents,
            configuration.BlockN);
        var kTileCount = k / configuration.BlockK;
        var logicalTileCount = checked(nTileCount * kTileCount);
        var bytesPerRhsTile = checked(
            (long)configuration.BlockN *
            configuration.BlockK *
            elementBytes);
        var bytesPerLogicalTile = checked(bytesPerRhsTile * rhsTilesPerGroup);

        // Model the block-local modulo schedule here. Chip-global contention is
        // common to all blocks and belongs to the graph-level cost model; folding
        // it into each producer service serializes latency that the stage ring is
        // specifically intended to overlap.
        var producerTransferCycles = DivideRoundUp(
            bytesPerLogicalTile,
            transfer.BytesPerCycle);
        var producerControlCycles = checked(
            (long)rhsTilesPerGroup *
            (machine.Synchronization.BlockCycles +
             transfer.Asynchronous!.CommitCycles));
        var producerServiceCycles = checked(
            producerTransferCycles + producerControlCycles);

        var fmaCount = checked(
            (long)configuration.BlockN *
            configuration.BlockK *
            rhsTilesPerGroup);
        var sharedLoadCycles = DivideRoundUp(
            bytesPerLogicalTile,
            sharedMemory.ReadBytesPerCycle);
        var fmaCycles = DivideRoundUp(fmaCount, machine.Compute.SimtFmaPerCycle);

        // The template partitions N across all consumer warps and K within a
        // warp. With one reduction group per warp width, block_n / workers is
        // also the number of contiguous K values loaded by each thread. Narrow
        // vectors reduce shared-load instruction efficiency, but they do not
        // create inactive FMA lanes: the FMA count above already contains only
        // the work issued by the tile.
        var vectorLanes = machine.Execution.VectorWidthBits /
            checked(elementBytes * 8);
        var kValuesPerThread = configuration.BlockN /
            machine.Execution.WorkersPerBlock;
        if (vectorLanes <= 0 || kValuesPerThread <= 0)
        {
            throw new InvalidOperationException(
                $"Packed GEMV vector utilization requires positive vector and per-thread K extents, " +
                $"got vector_lanes={vectorLanes}, block_n={configuration.BlockN}, " +
                $"consumer_warps={machine.Execution.WorkersPerBlock}.");
        }

        var usefulVectorLanes = Math.Min(vectorLanes, kValuesPerThread);
        var vectorizedSharedLoadCycles = DivideRoundUp(
            checked(sharedLoadCycles * vectorLanes),
            usefulVectorLanes);
        var consumerWorkCycles = Math.Max(vectorizedSharedLoadCycles, fmaCycles);
        var consumerControlCycles = checked(
            (long)rhsTilesPerGroup *
            (transfer.Asynchronous.WaitCycles +
             machine.Synchronization.BlockCycles));
        var consumerServiceCycles = checked(
            consumerWorkCycles + consumerControlCycles);

        var logicalStageCount = configuration.NumStages / rhsTilesPerGroup;
        var slotLifetimeCycles = checked(
            producerServiceCycles +
            transfer.LatencyCycles +
            consumerServiceCycles);
        var recurrenceCycles = DivideRoundUp(
            slotLifetimeCycles,
            logicalStageCount);
        var initiationIntervalCycles = Math.Max(
            Math.Max(producerServiceCycles, consumerServiceCycles),
            recurrenceCycles);
        var localPipelineCycles = checked(
            producerServiceCycles +
            consumerServiceCycles +
            ((logicalTileCount - 1) * initiationIntervalCycles));
        var chipGlobalCycles = EstimatePackedGemvChipGlobalCycles(
            machine,
            checked(totalNTileCount - nTileCount),
            kTileCount,
            bytesPerLogicalTile);
        return checked(localPipelineCycles + chipGlobalCycles);
    }

    private static long EstimatePackedBlockFp8MmaPipelineCycles(
        TargetMachineModel machine,
        TargetMemoryTransferSpec transfer,
        TargetMemoryResourceSpec sharedMemory,
        MatrixComputePrimitiveSpec primitive,
        long n,
        long k,
        int elementBytes,
        int reductionGroup,
        int rhsTilesPerGroup,
        bool materializeCompleteLhs,
        IReadOnlyList<long> localNExtents,
        PackedGemvPipelineConfiguration configuration)
        => PackedBlockFp8MmaPipelineCostModel.EstimateCycles(
            machine,
            transfer,
            sharedMemory,
            primitive,
            n,
            k,
            elementBytes,
            reductionGroup,
            rhsTilesPerGroup,
            materializeCompleteLhs || primitive.CooperativeWorkers > 1,
            localNExtents,
            configuration.BlockN,
            configuration.BlockK,
            configuration.NumStages);

    private static (long Maximum, long Total) GetPackedGemvNTileCounts(
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

    private static long EstimatePackedGemvChipGlobalCycles(
        TargetMachineModel machine,
        long totalNTileCount,
        long kTileCount,
        long bytesPerLogicalTile)
    {
        var rootMemory = machine.GetMemoryResource(
            machine.GetMemorySpace(machine.RootMemorySpace));
        var totalBytes = checked(totalNTileCount * kTileCount * bytesPerLogicalTile);
        return DivideRoundUp(totalBytes, rootMemory.ReadBytesPerCycle);
    }

    private static bool IsBetterPackedGemvPipelineCandidate(
        PackedGemvPipelineCandidate candidate,
        PackedGemvPipelineCandidate current,
        bool preferDeeperPipeline)
    {
        if (candidate.EstimatedCycles != current.EstimatedCycles)
        {
            return candidate.EstimatedCycles < current.EstimatedCycles;
        }

        if (preferDeeperPipeline &&
            candidate.Configuration.NumStages != current.Configuration.NumStages)
        {
            return candidate.Configuration.NumStages > current.Configuration.NumStages;
        }

        if (candidate.AllocatedSharedBytes != current.AllocatedSharedBytes)
        {
            return candidate.AllocatedSharedBytes < current.AllocatedSharedBytes;
        }

        if (candidate.Configuration.NumStages != current.Configuration.NumStages)
        {
            return candidate.Configuration.NumStages < current.Configuration.NumStages;
        }

        // Equal-cost tiles move the same bytes and execute the same FMA work.
        // Prefer partitioning that work across N: it assigns more warp lanes
        // to N, preserves a wider contiguous K vector per lane, and emits fewer
        // static K-reduction groups per stage. This matters for large-N GEMV
        // such as lm_head, where a larger K tile only grows the unrolled body.
        if (candidate.Configuration.BlockN != current.Configuration.BlockN)
        {
            return candidate.Configuration.BlockN > current.Configuration.BlockN;
        }

        if (candidate.Configuration.BlockK != current.Configuration.BlockK)
        {
            return candidate.Configuration.BlockK < current.Configuration.BlockK;
        }

        return false;
    }

    private static long DivideRoundUp(long value, long divisor)
        => checked(((value - 1) / divisor) + 1);

    private static long DivideRoundUp(long value, double divisor)
        => checked((long)Math.Ceiling(value / divisor));

    private static int GetPackedGemvPipelineMaximumBlockN(
        long n,
        int minimumBlockN,
        int maximumBlockN)
    {
        if (minimumBlockN <= 0 || maximumBlockN < minimumBlockN)
        {
            throw new ArgumentOutOfRangeException(
                nameof(minimumBlockN),
                $"Packed GEMV block-N bounds are invalid: [{minimumBlockN}, {maximumBlockN}].");
        }

        var boundedN = Math.Min(n, maximumBlockN);
        var blockN = minimumBlockN;
        while (blockN < boundedN)
        {
            blockN *= 2;
        }

        return blockN;
    }

    private static TensorBufferOperand GetBuffer(
        TIRMicroKernelSelectionContext context,
        int index,
        string name)
    {
        if ((uint)index >= (uint)context.Arguments.Count)
        {
            throw new InvalidOperationException(
                $"TIR microkernel selector for {context.Op.GetType().Name} expects {name} buffer at argument {index}.");
        }

        return context.Arguments[index] switch
        {
            Nncase.TIR.Buffer buffer => new TensorBufferOperand(
                buffer.Name,
                buffer.ElemType,
                buffer.Dimensions.ToArray(),
                buffer.DistributedType,
                buffer.Strides.ToArray()),
            BufferVar bufferVar => CreateBufferOperand(context.Op, index, name, bufferVar),
            _ => throw new InvalidOperationException(
                $"TIR microkernel selector for {context.Op.GetType().Name} expects {name} " +
                $"buffer at argument {index}, got {context.Arguments[index].GetType().Name}."),
        };
    }

    private static TensorBufferOperand CreateBufferOperand(
        Op op,
        int index,
        string name,
        BufferVar bufferVar)
    {
        var (tensorType, distributedType) = bufferVar.TypeAnnotation switch
        {
            DistributedType distributed => (distributed.TensorType, distributed),
            TensorType tensor => (tensor, null),
            _ => throw new InvalidOperationException(
                $"TIR microkernel selector for {op.GetType().Name} expects {name} tensor " +
                $"buffer at argument {index}, got {bufferVar.TypeAnnotation}."),
        };
        if (tensorType.Shape is not RankedShape shape ||
            tensorType.DType is PointerType or ReferenceType)
        {
            throw new InvalidOperationException(
                $"TIR microkernel selector for {op.GetType().Name} expects {name} ranked " +
                $"tensor buffer at argument {index}, got {tensorType}.");
        }

        return new TensorBufferOperand(
            bufferVar.Name,
            tensorType.DType,
            shape.Dimensions.ToArray(),
            distributedType,
            Strides: null);
    }

    private static void RequireRank(TensorBufferOperand buffer, int minimumRank, Op op, string name)
    {
        if (buffer.Rank < minimumRank)
        {
            throw new InvalidOperationException(
                $"TIR microkernel selector for {op.GetType().Name} expects {name} rank >= {minimumRank}, got {buffer.Rank}.");
        }
    }

    private static long GetMax(Dimension dimension)
        => CompilerServices.GetMaxShape([dimension])[0];

    private static long GetScalarExtent(Dimension dimension, DataType dataType)
        => checked(GetMax(dimension) * GetVectorLaneCount(dataType));

    private static Dimension[] GetLocalDimensions(TensorBufferOperand buffer)
    {
        if (buffer.DistributedType is not { } distributedType)
        {
            return buffer.Dimensions.ToArray();
        }

        var localType = Nncase.Utilities.DistributedUtility.GetDividedTensorType(distributedType);
        if (localType.Shape is not RankedShape localShape)
        {
            throw new InvalidOperationException(
                $"TIR microkernel selection requires a ranked local shape for buffer {buffer.Name}.");
        }

        return localShape.Dimensions.ToArray();
    }

    private static ImmutableArray<long> GetScalarLocalExtentProfile(
        TensorBufferOperand buffer,
        int axis)
    {
        if ((uint)axis >= (uint)buffer.Rank)
        {
            throw new ArgumentOutOfRangeException(nameof(axis));
        }

        var lanes = GetVectorLaneCount(buffer.ElemType);
        if (buffer.DistributedType is not { } distributedType)
        {
            return [checked(GetMax(buffer.Dimensions[axis]) * lanes)];
        }

        var hierarchy = distributedType.Placement.Hierarchy.ToArray();
        var ownerCount = hierarchy.Aggregate(
            1,
            static (product, extent) => checked(product * extent));
        var extents = ImmutableArray.CreateBuilder<long>(ownerCount);
        for (var owner = 0; owner < ownerCount; owner++)
        {
            var coordinates = DistributedUtility.GetUnraveledIndex(owner, hierarchy);
            var descriptor = DistributedUtility.GetLocalShardDescriptor(
                distributedType,
                coordinates,
                DistributedUtility.DivideFlags.MaxShape);
            extents.Add(checked(GetMax(descriptor.Axes[axis].ActiveExtent) * lanes));
        }

        return extents.MoveToImmutable();
    }

    private static int GetVectorLaneCount(DataType dataType)
        => dataType is VectorType vectorType
            ? checked(vectorType.Lanes.Aggregate(1, static (product, lane) => product * lane) *
                GetVectorLaneCount(vectorType.ElemType))
            : 1;

    private static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vectorType
            ? GetScalarDataType(vectorType.ElemType)
            : dataType;

    private static void RequireVectorLanes(
        DataType dataType,
        int[] expectedLanes,
        string family,
        string operand)
    {
        var actualLanes = dataType is VectorType vectorType
            ? vectorType.Lanes.ToArray()
            : Array.Empty<int>();
        if (!actualLanes.SequenceEqual(expectedLanes))
        {
            throw new NotSupportedException(
                $"{family} requires {operand} vector lanes [{string.Join(",", expectedLanes)}], " +
                $"got [{string.Join(",", actualLanes)}].");
        }
    }

    private readonly record struct PackedGemvPipelineConfiguration(
        int BlockN,
        int BlockK,
        int NumStages,
        bool PrequantizeLhs = false);

    private sealed record TensorBufferOperand(
        string Name,
        DataType ElemType,
        Dimension[] Dimensions,
        DistributedType? DistributedType,
        Dimension[]? Strides)
    {
        public int Rank => Dimensions.Length;
    }

    private readonly record struct PagedAttentionMicroKernelCandidate(
        string Variant,
        int BlockN,
        int NumStages,
        bool UsesTma);

    private readonly record struct PackedGemvPipelineCandidate(
        PackedGemvPipelineConfiguration Configuration,
        long EstimatedCycles,
        long AllocatedSharedBytes);
}
