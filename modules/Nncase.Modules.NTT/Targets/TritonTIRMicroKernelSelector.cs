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
    private const int SimtPagedAttentionMaximumBlockN = 128;
    private const int MmaPagedAttentionMaximumBlockN = 64;
    private const int PackedGemvMinimumBlockN = 8;
    private const int PackedGemvMaximumBlockN = 64;
    private const int SimtPagedAttentionNumStages = 1;
    private const int MmaPagedAttentionNumStages = 2;
    private const int PackedGemvMinimumBlockK = 128;

    // The SIMT stage helper statically expands one 32-element reduction group.
    // Keep a stage within 32 groups; larger bodies delay first-tile consumption
    // and underutilize the asynchronous double buffer despite fitting in Shared.
    private const int PackedGemvMaximumBlockK = 1024;
    private const int PackedGemvMinimumLogicalStages = 2;
    private const int GatedDeltaNetConvolutionMaximumBlockN = 256;
    private const int GatedDeltaNetRecurrentCoreBlockN = 128;
    private const int SparseExpertsDownBlockM = 16;
    private const int SparseExpertsDownMaximumBlockN = 64;
    private const int SparseExpertsDownMaximumStageK = 128;
    private const int SparseExpertsDownMaximumRoutesPerStage = 8;
    private const int SparseExpertsDownMinimumExpertBlockK = 16;

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
                transposeB: packedBlockScaledMatmul.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajor,
                kMajorPacked: packedBlockScaledMatmul.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
                lhsIndex: Nncase.TIR.NTT.PackedBlockScaledMatMul.Lhs.Index,
                rhsIndex: Nncase.TIR.NTT.PackedBlockScaledMatMul.Rhs.Index,
                outputIndex: Nncase.TIR.NTT.PackedBlockScaledMatMul.Output.Index,
                fp8Variant: "simt_block_fp8_fma_smem_pipeline"),
            Nncase.TIR.NTT.PackedMatMulNormStats packedMatmulNormStats => SelectMatmul(
                context,
                transposeA: false,
                transposeB: packedMatmulNormStats.RhsLayout == IR.NTT.PackedMatMulRhsLayout.NMajor,
                kMajorPacked: packedMatmulNormStats.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
                lhsIndex: 0,
                rhsIndex: 1,
                outputIndex: 2),
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
        string context)
    {
        var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared)
            ?? throw new NotSupportedException($"{context} requires Shared memory.");
        var parentSpace = machine.GetTilingParentMemorySpace(sharedSpace.TilingLevel);
        var transfer = machine.GetTransfer(parentSpace.Id, sharedSpace.Id);
        var asynchronous = transfer.Asynchronous ?? throw new NotSupportedException(
            $"{context} requires an asynchronous parent-to-Shared transfer.");
        var capacity = machine.GetMaximumUsableAllocationBytes(sharedSpace);
        foreach (var candidate in asynchronous.SupportedStageCounts.OrderDescending())
        {
            var requiredBytes = machine.GetAllocationSizeBytes(
                sharedSpace,
                checked(stageBytes * candidate));
            if (requiredBytes <= capacity)
            {
                return candidate;
            }
        }

        throw new NotSupportedException(
            $"{context} cannot fit one supported async pipeline: stage_bytes={stageBytes}, " +
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
        var blockN = SelectPowerOfTwoAtMost(
            localN,
            GatedDeltaNetConvolutionMaximumBlockN);
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
        var bProjection = GetBuffer(context, 3, "B projection");
        var aProjection = GetBuffer(context, 4, "A projection");
        var gatedOutput = GetBuffer(context, 8, "gated output");
        RequireRank(qkv, 2, recurrentCore, "QKV");
        RequireRank(z, 2, recurrentCore, "Z");
        RequireRank(bProjection, 2, recurrentCore, "B projection");
        RequireRank(aProjection, 2, recurrentCore, "A projection");
        RequireRank(gatedOutput, 2, recurrentCore, "gated output");
        var inputType = GetScalarDataType(qkv.ElemType);
        var outputType = GetScalarDataType(gatedOutput.ElemType);
        if (inputType != DataTypes.BFloat16 ||
            GetScalarDataType(z.ElemType) != DataTypes.BFloat16 ||
            GetScalarDataType(bProjection.ElemType) != DataTypes.BFloat16 ||
            GetScalarDataType(aProjection.ElemType) != DataTypes.BFloat16 ||
            outputType != DataTypes.BFloat16)
        {
            throw new NotSupportedException(
                $"GatedDeltaNet recurrent core requires BF16 QKV/Z/A/B projections and output, got " +
                $"{qkv.ElemType}/{z.ElemType}/{aProjection.ElemType}/" +
                $"{bProjection.ElemType}/{gatedOutput.ElemType}.");
        }

        var blockN = SelectPowerOfTwoAtMost(
            GetMax(GetLocalDimensions(gatedOutput)[1]),
            GatedDeltaNetRecurrentCoreBlockN);
        var localValueElements = GetMax(GetLocalDimensions(gatedOutput)[1]);
        if (localValueElements <= 0 || localValueElements % recurrentCore.ValueHeadDim != 0)
        {
            throw new InvalidOperationException(
                "GatedDeltaNet recurrent core requires a non-empty local output aligned to complete value heads.");
        }

        return new(
            "triton.gated_delta_net",
            "recurrent_core",
            CreateParameters(
                blockM: 1,
                blockN,
                blockK: 1,
                numStages: 1),
            ImmutableArray<TIRSharedWorkspaceDescriptor>.Empty,
            TransferPipeline: null);
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

    private static void ValidateGatedDeltaNetSharedCapacity(
        TargetMachineModel machine,
        ImmutableArray<TIRSharedWorkspaceDescriptor> workspaces,
        string context)
    {
        var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared)
            ?? throw new NotSupportedException($"{context} requires Shared memory.");
        long requiredBytes = 0;
        foreach (var workspace in workspaces)
        {
            if (workspace.Type.Shape is not RankedShape shape)
            {
                throw new InvalidOperationException(
                    $"{context} workspace {workspace.Name} must have a ranked shape.");
            }

            var elements = 1L;
            foreach (var dimension in shape.Dimensions)
            {
                elements = checked(elements * GetMax(dimension));
            }

            var bytes = checked(elements * workspace.Type.DType.SizeInBytes);
            requiredBytes = checked(
                requiredBytes + RoundUp(bytes, workspace.AlignmentBytes));
        }

        var allocatedBytes = machine.GetAllocationSizeBytes(sharedSpace, requiredBytes);
        var capacityBytes = machine.GetMaximumUsableAllocationBytes(sharedSpace);
        if (allocatedBytes > capacityBytes)
        {
            throw new NotSupportedException(
                $"{context} requires {allocatedBytes} Shared bytes, exceeding target capacity " +
                $"{capacityBytes} bytes.");
        }
    }

    private static long RoundUp(long value, long alignment)
        => checked(DivideRoundUp(value, alignment) * alignment);

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

        return checked((lhs / a) * rhs);
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
        string? fp8Variant = null)
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
        return CreateMatrixSelection(
            context.Machine,
            "triton.matmul",
            GetScalarDataType(lhs.ElemType),
            GetScalarDataType(rhs.ElemType),
            GetScalarDataType(output.ElemType),
            m,
            n,
            k,
            kDimension.IsFixed,
            kMajorPacked,
            sourceArgumentIndices: [rhsIndex],
            fp8Variant: fp8Variant);
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
                out var pipeline))
        {
            throw new InvalidOperationException(
                $"PackedMatMulSamplingPartial cannot select a spill-free BF16 Shared-staged GEMV for " +
                $"local shape M={m}, N={n}, K={k}.");
        }

        const int nVector = 8;
        const int kAtom = 16;
        var rhsStage = new TIRSharedWorkspaceDescriptor(
            "rhs_stage",
            new TensorType(
                GetScalarDataType(rhs.ElemType),
                new RankedShape(
                    pipeline.NumStages,
                    (pipeline.BlockK / kAtom) * (pipeline.BlockN / nVector),
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
            sourceArgumentIndices: [weightIndex]);
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
        return CreateMatrixSelection(
            context.Machine,
            "triton.qkv_parallel_linear",
            GetScalarDataType(input.ElemType),
            GetScalarDataType(weight.ElemType),
            GetScalarDataType(qOutput.ElemType),
            m,
            n,
            GetScalarExtent(kDimension, input.ElemType),
            kDimension.IsFixed,
            qkv.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
            sourceArgumentIndices: [1]);
    }

    private static TIRMicroKernelSelection SelectPackedMatMulGlu(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PackedMatMulGlu matmulGlu)
    {
        var input = GetBuffer(context, 0, "input");
        var gateWeight = GetBuffer(context, 1, "gate weight");
        var upWeight = GetBuffer(context, 2, "up weight");
        var output = GetBuffer(context, 9, "output");
        RequireRank(input, 2, context.Op, "input");
        RequireRank(gateWeight, 2, context.Op, "gate weight");
        RequireRank(upWeight, 2, context.Op, "up weight");
        RequireRank(output, 2, context.Op, "output");

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
        return CreateMatrixSelection(
            context.Machine,
            "triton.matmul_glu",
            GetScalarDataType(input.ElemType),
            GetScalarDataType(gateWeight.ElemType),
            GetScalarDataType(output.ElemType),
            m,
            n,
            GetScalarExtent(kDimension, input.ElemType),
            kDimension.IsFixed,
            matmulGlu.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
            simultaneousRhsTileCount: 2,
            sourceArgumentIndices: [1, 2],
            fp8Variant: matmulGlu.QuantizationMode == IR.Math.MatMulQuantizationMode.DynamicBlock
                ? "simt_block_fp8_fma_smem_pipeline"
                : null);
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
        string? fp8Variant = null)
    {
        var gemv = m == 1;
        if (gemv)
        {
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
                    out var pipeline))
            {
                if (outputType != DataTypes.BFloat16)
                {
                    throw new NotSupportedException(
                        $"{family} Shared-staged GEMV requires BF16 output, got {outputType}.");
                }

                var nVector = 16 / outputType.SizeInBytes;
                var kVector = 16 / rhsType.SizeInBytes;
                const int kPack = 2;
                var kAtom = kPack * kVector;
                var variant = rhsType == DataTypes.BFloat16
                    ? "simt_fma_smem_pipeline"
                    : fp8Variant ?? "simt_fp8_fma_smem_pipeline";
                var rhsShape = new TensorType(
                    rhsType,
                    new RankedShape(
                        new[]
                        {
                            pipeline.NumStages,
                            (pipeline.BlockK / kAtom) * (pipeline.BlockN / nVector),
                            nVector * kAtom,
                        }));
                return new(
                    family,
                    variant,
                    CreateParameters(1, pipeline.BlockN, pipeline.BlockK, pipeline.NumStages),
                    ImmutableArray.Create(
                        new TIRSharedWorkspaceDescriptor(
                            "rhs_stage",
                            rhsShape,
                            NvidiaNvmmaSharedAlignmentBytes)),
                    new TIRTransferPipelineContract(
                    [
                        new TIRTransferPipelineChannel(
                            "weight",
                            sourceArgumentIndices ?? throw new InvalidOperationException(
                                $"{family}/{variant} is missing transfer source operand indexes."),
                            [0]),
                    ]));
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
        out PackedGemvPipelineConfiguration configuration)
    {
        configuration = default;
        if ((family != "triton.matmul" &&
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
            machine.Execution.WorkerWidth != 32)
        {
            return false;
        }

        var maximumBlockN = GetPackedGemvPipelineMaximumBlockN(n);
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
        for (var candidateBlockN = PackedGemvMinimumBlockN;
             candidateBlockN <= maximumBlockN;
             candidateBlockN *= 2)
        {
            for (var candidateBlockK = PackedGemvMinimumBlockK;
                 candidateBlockK <= Math.Min(k, PackedGemvMaximumBlockK);
                 candidateBlockK *= 2)
            {
                if (k % candidateBlockK != 0)
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

                    var requiredSharedBytes = checked(physicalStageCount * stageBytes);
                    var allocatedSharedBytes = machine.GetAllocationSizeBytes(
                        sharedSpace,
                        requiredSharedBytes);
                    if (checked(allocatedSharedBytes + reservedSharedBytes) >
                        machine.GetMaximumUsableAllocationBytes(sharedSpace))
                    {
                        continue;
                    }

                    var candidateConfiguration = new PackedGemvPipelineConfiguration(
                        candidateBlockN,
                        candidateBlockK,
                        physicalStageCount);
                    var candidate = new PackedGemvPipelineCandidate(
                        candidateConfiguration,
                        EstimatePackedGemvPipelineCycles(
                            machine,
                            transfer,
                            machine.GetMemoryResource(sharedSpace),
                            n,
                            k,
                            rhsTilesPerGroup,
                            rhsType.SizeInBytes,
                            candidateConfiguration),
                        allocatedSharedBytes);
                    if (bestCandidate is null || IsBetterPackedGemvPipelineCandidate(candidate, bestCandidate.Value))
                    {
                        bestCandidate = candidate;
                    }
                }
            }
        }

        if (bestCandidate is null)
        {
            return false;
        }

        configuration = bestCandidate.Value.Configuration;
        return true;
    }

    private static long EstimatePackedGemvPipelineCycles(
        TargetMachineModel machine,
        TargetMemoryTransferSpec transfer,
        TargetMemoryResourceSpec sharedMemory,
        long n,
        long k,
        int rhsTilesPerGroup,
        int elementBytes,
        PackedGemvPipelineConfiguration configuration)
    {
        var nTileCount = DivideRoundUp(n, configuration.BlockN);
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
        var consumerWorkCycles = Math.Max(sharedLoadCycles, fmaCycles);

        // The template partitions N across all consumer warps and K within a
        // warp. A tile narrower than one target vector per thread still issues
        // the same warp instructions with proportionally fewer useful lanes.
        var vectorLanes = machine.Execution.VectorWidthBits /
            checked(elementBytes * 8);
        var nValuesPerWarp = configuration.BlockN /
            machine.Execution.WorkersPerBlock;
        if (vectorLanes <= 0 || nValuesPerWarp <= 0)
        {
            throw new InvalidOperationException(
                $"Packed GEMV vector utilization requires positive vector and per-warp N extents, " +
                $"got vector_lanes={vectorLanes}, block_n={configuration.BlockN}, " +
                $"consumer_warps={machine.Execution.WorkersPerBlock}.");
        }

        var usefulVectorLanes = Math.Min(vectorLanes, nValuesPerWarp);
        consumerWorkCycles = DivideRoundUp(
            checked(consumerWorkCycles * vectorLanes),
            usefulVectorLanes);
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
        return checked(
            producerServiceCycles +
            consumerServiceCycles +
            ((logicalTileCount - 1) * initiationIntervalCycles));
    }

    private static bool IsBetterPackedGemvPipelineCandidate(
        PackedGemvPipelineCandidate candidate,
        PackedGemvPipelineCandidate current)
    {
        if (candidate.EstimatedCycles != current.EstimatedCycles)
        {
            return candidate.EstimatedCycles < current.EstimatedCycles;
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

    private static int GetPackedGemvPipelineMaximumBlockN(long n)
    {
        var boundedN = Math.Min(n, PackedGemvMaximumBlockN);
        var blockN = PackedGemvMinimumBlockN;
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
                buffer.DistributedType),
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
            distributedType);
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

    private static int GetVectorLaneCount(DataType dataType)
        => dataType is VectorType vectorType
            ? checked(vectorType.Lanes.Aggregate(1, static (product, lane) => product * lane) *
                GetVectorLaneCount(vectorType.ElemType))
            : 1;

    private static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vectorType
            ? GetScalarDataType(vectorType.ElemType)
            : dataType;

    private readonly record struct PackedGemvPipelineConfiguration(
        int BlockN,
        int BlockK,
        int NumStages);

    private sealed record TensorBufferOperand(
        string Name,
        DataType ElemType,
        Dimension[] Dimensions,
        DistributedType? DistributedType)
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
