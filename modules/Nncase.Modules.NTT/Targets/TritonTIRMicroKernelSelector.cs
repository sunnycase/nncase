// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;

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
            Nncase.TIR.NTT.SUMMA summa => SelectSumma(context, summa),
            Nncase.TIR.NTT.QKVParallelLinear => SelectFusedLinear(
                context,
                "triton.qkv_parallel_linear",
                inputIndex: 0,
                weightIndex: 1,
                outputIndex: 13),
            Nncase.TIR.NTT.PackedQKVParallelLinear packedQkv => SelectPackedQkv(
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
            _ => null,
        };
    }

    private static TIRMicroKernelSelection SelectPagedAttentionPartial(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PagedAttentionPartial pagedAttention)
    {
        const int numStages = 1;
        var config = GetPagedAttentionConfig(context, 1);
        var useDecodeGqaSimt = CanUsePagedAttentionDecodeGqaSimt(
            context,
            pagedAttention,
            config);
        var maximumBlockN = useDecodeGqaSimt
            ? SimtPagedAttentionMaximumBlockN
            : MmaPagedAttentionMaximumBlockN;
        var tmaBlockN = SelectPagedAttentionTmaBlockN(
            context.Machine,
            config,
            numStages,
            maximumBlockN);
        if (tmaBlockN is null)
        {
            var blockN = SelectPagedAttentionPageLocalBlockN(config.BlockSize);
            return new(
                "triton.paged_attention_partial",
                useDecodeGqaSimt ? "simt_direct" : "mma_direct",
                CreatePagedAttentionParameters(config, blockN, numStages),
                ImmutableArray<TIRSharedWorkspaceDescriptor>.Empty,
                TransferPipeline: null);
        }

        var selectedBlockN = tmaBlockN.Value;
        var stageType = new TensorType(
            config.KVPrimType,
            new RankedShape(new[] { numStages, 1, 1, selectedBlockN, 1, config.HeadDim }));
        var variant = useDecodeGqaSimt
            ? "simt_tma_smem_pipeline"
            : "mma_tma_smem_pipeline";
        return new(
            "triton.paged_attention_partial",
            variant,
            CreatePagedAttentionParameters(config, selectedBlockN, numStages),
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

    private static bool CanUsePagedAttentionDecodeGqaSimt(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PagedAttentionPartial pagedAttention,
        IR.NN.IPagedAttentionConfig config)
    {
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

        var groupTile = 1;
        while (groupTile < groupSize)
        {
            groupTile *= 2;
        }

        const int consumerWarps = 8;
        return groupTile <= consumerWarps && consumerWarps % groupTile == 0;
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
        int outputIndex)
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
            m,
            n,
            k,
            kDimension.IsFixed,
            kMajorPacked,
            sourceArgumentIndices: [rhsIndex]);
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
            m,
            n,
            k,
            kDimension.IsFixed,
            kMajorPacked: false,
            sourceArgumentIndices: [weightIndex]);
    }

    private static TIRMicroKernelSelection SelectPackedQkv(
        TIRMicroKernelSelectionContext context,
        Nncase.TIR.NTT.PackedQKVParallelLinear qkv)
    {
        var input = GetBuffer(context, 0, "input");
        var qWeight = GetBuffer(context, 1, "q weight");
        var kWeight = GetBuffer(context, 2, "k weight");
        var vWeight = GetBuffer(context, 3, "v weight");
        var qOutput = GetBuffer(context, 13, "q output");
        var kOutput = GetBuffer(context, 14, "k output");
        var vOutput = GetBuffer(context, 15, "v output");
        RequireRank(input, 2, context.Op, "input");
        RequireRank(qWeight, 2, context.Op, "q weight");
        RequireRank(kWeight, 2, context.Op, "k weight");
        RequireRank(vWeight, 2, context.Op, "v weight");
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

        var n = checked(
            GetScalarExtent(outputDimensions[0][^1], qOutput.ElemType) +
            GetScalarExtent(outputDimensions[1][^1], kOutput.ElemType) +
            GetScalarExtent(outputDimensions[2][^1], vOutput.ElemType));
        var kDimension = inputDimensions[^1];
        return CreateMatrixSelection(
            context.Machine,
            "triton.qkv_parallel_linear",
            GetScalarDataType(input.ElemType),
            GetScalarDataType(qWeight.ElemType),
            m,
            n,
            GetScalarExtent(kDimension, input.ElemType),
            kDimension.IsFixed,
            qkv.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
            sourceArgumentIndices: [1, 2, 3]);
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
            m,
            n,
            GetScalarExtent(kDimension, input.ElemType),
            kDimension.IsFixed,
            matmulGlu.RhsLayout == IR.NTT.PackedMatMulRhsLayout.KMajor,
            simultaneousRhsTileCount: 2,
            sourceArgumentIndices: [1, 2]);
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
        long m,
        long n,
        long k,
        bool fixedK,
        bool kMajorPacked,
        int simultaneousRhsTileCount = 1,
        IReadOnlyList<int>? sourceArgumentIndices = null)
    {
        var gemv = m == 1;
        if (gemv)
        {
            if (TryGetPackedBFloat16GemvPipelineConfiguration(
                    machine,
                    family,
                    lhsType,
                    rhsType,
                    n,
                    k,
                    fixedK,
                    kMajorPacked,
                    simultaneousRhsTileCount,
                    out var pipeline))
            {
                const int nVector = 8;
                const int kAtom = 16;
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
                    "simt_fma_smem_pipeline",
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
                                $"{family}/simt_fma_smem_pipeline is missing transfer source operand indexes."),
                            [0]),
                    ]));
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

    private static bool TryGetPackedBFloat16GemvPipelineConfiguration(
        TargetMachineModel machine,
        string family,
        DataType lhsType,
        DataType rhsType,
        long n,
        long k,
        bool fixedK,
        bool kMajorPacked,
        int simultaneousRhsTileCount,
        out PackedGemvPipelineConfiguration configuration)
    {
        configuration = default;
        if ((family != "triton.matmul" &&
             family != "triton.qkv_parallel_linear" &&
             family != "triton.matmul_glu") ||
            simultaneousRhsTileCount <= 0 ||
            !kMajorPacked ||
            lhsType != DataTypes.BFloat16 ||
            rhsType != DataTypes.BFloat16 ||
            n < 8 ||
            !fixedK ||
            k <= 0 ||
            k % 128 != 0 ||
            machine.Execution.Kind != BlockExecutionKind.PersistentGpuBlock ||
            machine.Execution.WorkersPerBlock != 8 ||
            machine.Execution.WorkerWidth != 32)
        {
            return false;
        }

        const int blockK = 128;
        var blockN = GetPackedGemvPipelineBlockN(n);
        var sharedSpace = machine.MemorySpaces.Values.SingleOrDefault(
            space => space.TIRBinding?.Location == MemoryLocation.Shared);
        if (sharedSpace is null)
        {
            return false;
        }

        const long elementBytes = 2;
        var stageBytes = checked((long)blockN * blockK * elementBytes);
        for (var candidateStages = 4; candidateStages >= 2; candidateStages--)
        {
            // A physical pipe slot owns one RHS transfer. Fused projections
            // consume adjacent slots and require at least two logical tile groups.
            if (candidateStages % simultaneousRhsTileCount != 0 ||
                candidateStages / simultaneousRhsTileCount < 2)
            {
                continue;
            }

            var requiredSharedBytes = checked(candidateStages * stageBytes);
            if (machine.GetAllocationSizeBytes(sharedSpace, requiredSharedBytes) <=
                machine.GetMaximumUsableAllocationBytes(sharedSpace))
            {
                configuration = new(blockN, blockK, candidateStages);
                return true;
            }
        }

        return false;
    }

    private static int GetPackedGemvPipelineBlockN(long n)
    {
        const int minimumBlockN = 8;
        const int maximumBlockN = 64;
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
}
