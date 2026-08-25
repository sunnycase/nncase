// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.PyNTT;

public sealed record PyNTTBufferPointerTemplateModel(
    string Expression,
    int AddressSpace = 1)
{
    public string DistributedStorageKind { get; init; } = "CompactLocal";

    public PyNTTDimExpression[] GlobalShape { get; init; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] GlobalOffsets { get; init; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] Strides { get; init; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTShardAxisTemplateModel[] ShardAxes { get; init; } = Array.Empty<PyNTTShardAxisTemplateModel>();

    public int[] Hierarchy { get; init; } = Array.Empty<int>();
}

public sealed record PyNTTSplitStageTemplateModel(
    int[] HierarchyAxes,
    string Distribution,
    PyNTTDimExpression? Granularity,
    long BlockSize);

public sealed record PyNTTShardAxisTemplateModel(
    PyNTTSplitStageTemplateModel[] Stages);

public sealed record PyNTTTransferPipelineChannelTemplateModel(
    string Name,
    string[] SharedWorkspaceNames);

public sealed record PyNTTMicroKernelTemplateModel(
    string Family,
    string Variant,
    IReadOnlyDictionary<string, long> Parameters,
    IReadOnlyDictionary<string, string> SharedWorkspaceOffsets,
    IReadOnlyDictionary<string, long[]> SharedWorkspaceShapes,
    PyNTTTransferPipelineChannelTemplateModel[] TransferPipelineChannels)
{
    public bool HasTransferPipeline => TransferPipelineChannels.Length != 0;
}

public sealed record PyNTTPipelineStageTemplateModel(
    string StageId,
    string StageName,
    string ChannelName,
    string HelperName,
    string Template,
    object Model,
    IReadOnlyDictionary<string, string> SharedWorkspaceOffsets,
    string PipeName,
    string ReaderName,
    string WriterName);

public sealed record PyNTTPipelineHandoffTemplateModel(
    string HandoffId,
    string HandoffName,
    string PipeName,
    string ReaderName,
    string WriterName);

public sealed record PyNTTProducerConsumerRegionTemplateModel(
    string FunctionName,
    string ProducerFunctionName,
    string ConsumerFunctionName,
    string ProducerBodySource,
    string ConsumerBodySource,
    PyNTTPipelineStageTemplateModel[] Stages,
    PyNTTPipelineHandoffTemplateModel[] Handoffs,
    string[] ProducerEndpointNames,
    string[] ConsumerEndpointNames)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTPooledByteAddressTemplateModel(
    string BaseName,
    string OffsetBytes,
    string PoolStrideBytes,
    string PoolScopeSize,
    int AddressSpace);

public sealed record PyNTTTensorLoadTemplateModel(
    string FunctionName,
    string SourceName,
    long SourceOffset,
    PyNTTBufferPointerTemplateModel Destination,
    string DType,
    string TritonDType,
    PyNTTDimExpression[] LocalShape,
    PyNTTDimExpression[] DestinationStrides,
    PyNTTDimExpression[] GlobalShape,
    PyNTTDimExpression[] GlobalOffsets,
    int[] Hierarchy,
    PyNTTShardAxisTemplateModel[] ShardAxes,
    int VectorLaneCount,
    int[] VectorLaneShape,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public PyNTTBufferPointerTemplateModel? Source { get; set; }

    public PyNTTDimExpression[]? SourceStrides { get; set; }
}

public sealed record PyNTTTensorStoreTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Source,
    string DestinationName,
    long DestinationOffset,
    string DType,
    string TritonDType,
    PyNTTDimExpression[] LocalShape,
    PyNTTDimExpression[] SourceStrides,
    PyNTTDimExpression[] GlobalShape,
    PyNTTDimExpression[] GlobalOffsets,
    int[] Hierarchy,
    PyNTTShardAxisTemplateModel[] ShardAxes,
    int VectorLaneCount,
    int[] VectorLaneShape,
    bool OwnerOnly,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public PyNTTBufferPointerTemplateModel? Destination { get; set; }

    public PyNTTDimExpression[]? DestinationStrides { get; set; }
}

public sealed record PyNTTMemcopyTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Source,
    PyNTTBufferPointerTemplateModel Destination,
    string DType,
    string TritonDType,
    PyNTTDimExpression[] Shape,
    PyNTTDimExpression[] SourceStrides,
    PyNTTDimExpression[] DestinationStrides,
    int VectorLaneCount,
    int[] VectorLaneShape,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTRegionCopyPlanTemplateModel(
    PyNTTDimExpression[] SourceOrigins,
    PyNTTDimExpression[] DestinationOrigins,
    PyNTTDimExpression[] Extents,
    bool CoversWholeSource,
    bool CoversWholeDestination);

public sealed record PyNTTRegionCopyTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Source,
    PyNTTBufferPointerTemplateModel Destination,
    string DType,
    string TritonDType,
    PyNTTDimExpression[] SourceShape,
    PyNTTDimExpression[] DestinationShape,
    PyNTTDimExpression[] SourceGlobalOffsets,
    PyNTTDimExpression[] DestinationGlobalOffsets,
    PyNTTDimExpression[] SourceStrides,
    PyNTTDimExpression[] DestinationStrides,
    int[] VectorLaneShape,
    string OperationKind,
    PyNTTRegionCopyPlanTemplateModel CopyPlan,
    bool OwnerOnly,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTElementwiseBinaryTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Lhs,
    PyNTTBufferPointerTemplateModel Rhs,
    PyNTTBufferPointerTemplateModel Output,
    string LhsDType,
    string RhsDType,
    string OutputDType,
    string LhsTritonDType,
    string RhsTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] LhsShape,
    PyNTTDimExpression[] RhsShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] LhsStrides,
    PyNTTDimExpression[] RhsStrides,
    PyNTTDimExpression[] OutputStrides,
    int LhsVectorLaneCount,
    int RhsVectorLaneCount,
    int OutputVectorLaneCount,
    int[] LhsVectorLaneShape,
    int[] RhsVectorLaneShape,
    int[] OutputVectorLaneShape,
    PyNTTDimExpression[] Shape,
    string BinaryExpression,
    string Op,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTElementwiseUnaryTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int InputVectorLaneCount,
    int OutputVectorLaneCount,
    int[] InputVectorLaneShape,
    int[] OutputVectorLaneShape,
    PyNTTDimExpression[] Shape,
    string UnaryExpression,
    string Op,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTElementwiseCastTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int InputVectorLaneCount,
    int OutputVectorLaneCount,
    int[] InputVectorLaneShape,
    int[] OutputVectorLaneShape,
    int[] VectorizedAxes,
    PyNTTDimExpression[] Shape,
    string CastExpression,
    string CastMode,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTElementwiseWhereTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Cond,
    PyNTTBufferPointerTemplateModel TrueValue,
    PyNTTBufferPointerTemplateModel FalseValue,
    PyNTTBufferPointerTemplateModel Output,
    string CondDType,
    string ValueDType,
    string OutputDType,
    string CondTritonDType,
    string ValueTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] CondShape,
    PyNTTDimExpression[] TrueShape,
    PyNTTDimExpression[] FalseShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] CondStrides,
    PyNTTDimExpression[] TrueStrides,
    PyNTTDimExpression[] FalseStrides,
    PyNTTDimExpression[] OutputStrides,
    int CondVectorLaneCount,
    int TrueVectorLaneCount,
    int FalseVectorLaneCount,
    int OutputVectorLaneCount,
    int[] CondVectorLaneShape,
    int[] TrueVectorLaneShape,
    int[] FalseVectorLaneShape,
    int[] OutputVectorLaneShape,
    PyNTTDimExpression[] Shape,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTVectorLayoutTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int[] InputLanes,
    int[] OutputLanes,
    int[] Axes,
    int[] Lanes,
    bool IsPack,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTConcatTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel[] Inputs,
    PyNTTBufferPointerTemplateModel Output,
    string OutputDType,
    string OutputTritonDType,
    PyNTTDimExpression[][] InputShapes,
    PyNTTDimExpression[][] InputStrides,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] OutputStrides,
    int Axis,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTGatherTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Index,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string IndexDType,
    string OutputDType,
    string InputTritonDType,
    string IndexTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] InputGlobalShape,
    PyNTTDimExpression[] IndexShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] IndexStrides,
    PyNTTDimExpression[] OutputStrides,
    int Axis,
    int ValueVectorLaneCount,
    int[] ValueVectorLaneShape,
    int[] Hierarchy,
    PyNTTShardAxisTemplateModel[] InputShardAxes,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTReshardTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    PyNTTPooledByteAddressTemplateModel? PartialInputAddress,
    PyNTTPooledByteAddressTemplateModel OutputAddress,
    int ScalarElementSizeBytes,
    string DType,
    string TritonDType,
    PyNTTDimExpression[] GlobalShape,
    PyNTTDimExpression[] InputLocalShape,
    PyNTTDimExpression[] InputActiveShape,
    PyNTTDimExpression[] InputGlobalOffsets,
    PyNTTDimExpression[] OutputLocalShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int VectorLaneCount,
    int[] VectorLaneShape,
    int[] Hierarchy,
    PyNTTShardAxisTemplateModel[] InputShardAxes,
    int[] InputPartialAxes,
    PyNTTShardAxisTemplateModel[] OutputShardAxes,
    string Stage,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTPadTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    long[][] Pads,
    string PadValue,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTScatterNDTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Indices,
    PyNTTBufferPointerTemplateModel Updates,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string IndicesDType,
    string UpdatesDType,
    string OutputDType,
    string InputTritonDType,
    string IndicesTritonDType,
    string UpdatesTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] IndicesShape,
    PyNTTDimExpression[] UpdatesShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] IndicesStrides,
    PyNTTDimExpression[] UpdatesStrides,
    PyNTTDimExpression[] OutputStrides,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTSliceTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    long[] Starts,
    long[] Strides,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTRoPETemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Cos,
    PyNTTBufferPointerTemplateModel Sin,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string CosDType,
    string SinDType,
    string OutputDType,
    string InputTritonDType,
    string CosTritonDType,
    string SinTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] CosShape,
    PyNTTDimExpression[] SinShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] CosStrides,
    PyNTTDimExpression[] SinStrides,
    PyNTTDimExpression[] OutputStrides,
    int[] InputVectorLaneShape,
    int[] CosVectorLaneShape,
    int[] SinVectorLaneShape,
    int[] OutputVectorLaneShape,
    int InputVectorLaneCount,
    int CosVectorLaneCount,
    int SinVectorLaneCount,
    int OutputVectorLaneCount,
    int SinCosVectorPackFactor,
    int RotaryAxis,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTLayerNormTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Scale,
    PyNTTBufferPointerTemplateModel Bias,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string ScaleDType,
    string BiasDType,
    string OutputDType,
    string InputTritonDType,
    string ScaleTritonDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] ScaleShape,
    PyNTTDimExpression[] BiasShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] ScaleStrides,
    PyNTTDimExpression[] BiasStrides,
    PyNTTDimExpression[] OutputStrides,
    int InputVectorLaneCount,
    int ScaleVectorLaneCount,
    int BiasVectorLaneCount,
    int OutputVectorLaneCount,
    int[] InputVectorLaneShape,
    int[] ScaleVectorLaneShape,
    int[] BiasVectorLaneShape,
    int[] OutputVectorLaneShape,
    int Axis,
    float Epsilon,
    bool UseMean,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTNormStatsTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int InputVectorLaneCount,
    int OutputVectorLaneCount,
    int[] InputVectorLaneShape,
    int[] OutputVectorLaneShape,
    int Axis,
    bool UseMean,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTNormApplyTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Stats,
    PyNTTBufferPointerTemplateModel Scale,
    PyNTTBufferPointerTemplateModel Bias,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string StatsDType,
    string ScaleDType,
    string BiasDType,
    string OutputDType,
    string InputTritonDType,
    string StatsTritonDType,
    string ScaleTritonDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] InputGlobalShape,
    PyNTTDimExpression[] StatsShape,
    PyNTTDimExpression[] ScaleShape,
    PyNTTDimExpression[] BiasShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] StatsStrides,
    PyNTTDimExpression[] ScaleStrides,
    PyNTTDimExpression[] BiasStrides,
    PyNTTDimExpression[] OutputStrides,
    int InputVectorLaneCount,
    int StatsVectorLaneCount,
    int ScaleVectorLaneCount,
    int BiasVectorLaneCount,
    int OutputVectorLaneCount,
    int[] InputVectorLaneShape,
    int[] StatsVectorLaneShape,
    int[] ScaleVectorLaneShape,
    int[] BiasVectorLaneShape,
    int[] OutputVectorLaneShape,
    int Axis,
    float Epsilon,
    bool UseMean,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTGatherReduceNormApplyTemplateModel(
    string FunctionName,
    PyNTTNormApplyTemplateModel NormApply,
    PyNTTPooledByteAddressTemplateModel PartialStatsAddress,
    int[] Hierarchy,
    int[] PartialAxes,
    bool HasBias,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTQKVRoPENormTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Scale,
    PyNTTBufferPointerTemplateModel Bias,
    string InputDType,
    string ScaleDType,
    string BiasDType,
    string InputTritonDType,
    string ScaleTritonDType,
    string BiasTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] InputGlobalShape,
    PyNTTDimExpression[] ScaleShape,
    PyNTTDimExpression[] BiasShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] ScaleStrides,
    PyNTTDimExpression[] BiasStrides,
    int InputVectorLaneCount,
    int ScaleVectorLaneCount,
    int BiasVectorLaneCount,
    int[] InputVectorLaneShape,
    int[] ScaleVectorLaneShape,
    int[] BiasVectorLaneShape,
    int Axis,
    float Epsilon,
    bool UseMean,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTGetPositionIdsTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Output,
    string OutputDType,
    string OutputTritonDType,
    PyNTTDimExpression[] LocalShape,
    PyNTTDimExpression[] GlobalShape,
    PyNTTDimExpression[] OutputGlobalOffsets,
    PyNTTDimExpression[] OutputStrides,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

internal sealed record PyNTTObjectFieldInputMetadata(
    string Name,
    string SourceName,
    string ObjectKind,
    string Field,
    string Materialization,
    PyNTTKVCacheStorageMetadata? Storage,
    string? DType,
    int[] Shape);

internal sealed record PyNTTObjectFieldBindingMetadata(
    string ObjectKind,
    string Field,
    string Materialization,
    PyNTTKVCacheStorageMetadata? Storage,
    string? DType,
    int[] Shape);

internal sealed record PyNTTKVCacheStorageMetadata(
    string DType,
    int[] TopologyShape,
    int[] KeyTailShape,
    int[] ValueTailShape,
    int KeySectionElements,
    int ValueSectionElements,
    int BlockElements,
    int BlockSize);

public sealed record PyNTTPagedAttentionCacheTemplateModel(
    string DType,
    string TritonDType,
    int NumLayers,
    int NumKVHeads,
    int HeadDim,
    int BlockSize,
    int KeyLaneCount,
    int ValueLaneCount,
    int KeyVectorizedDim,
    int ValueVectorizedDim,
    int KeyHeadDimBlocks,
    int ValueHeadDimBlocks,
    int KeySectionOffset,
    int ValueSectionOffset,
    int KeySectionElements,
    int ValueSectionElements,
    int BlockElements,
    int KeyLayerStride,
    int KeyHeadStride,
    int KeyDimBlockStride,
    int KeyBlockOffsetStride,
    int ValueLayerStride,
    int ValueHeadStride,
    int ValueDimBlockStride,
    int ValueBlockOffsetStride,
    int[] KeyTailShape,
    int[] ValueTailShape,
    int IdLength,
    int[] TopologyShape,
    int[] NumBlocksHierarchyAxes);

public sealed record PyNTTUpdatePagedAttentionKVCacheTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Slots,
    string SlotsDType,
    string SlotsTritonDType,
    PyNTTDimExpression[] SlotsShape,
    PyNTTDimExpression[] SlotsGlobalShape,
    PyNTTDimExpression[] SlotsGlobalOffsets,
    PyNTTDimExpression[] SlotsStrides,
    PyNTTShardAxisTemplateModel[] SlotsShardAxes,
    PyNTTShardAxisTemplateModel[] SlotsSourceShardAxes,
    int[] Hierarchy,
    int SeqAxis,
    int HeadAxis,
    int DimAxis,
    string LayerIdExpression,
    int CacheKind,
    int SlotsVectorLaneCount,
    int[] SlotsVectorLaneShape,
    PyNTTPagedAttentionCacheTemplateModel Cache,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTQKVRoPEWithCacheTemplateModel(
    string FunctionName,
    PyNTTQKVRoPENormTemplateModel QNorm,
    PyNTTQKVRoPENormTemplateModel KNorm,
    PyNTTRoPETemplateModel QRoPE,
    PyNTTRoPETemplateModel KRoPE,
    PyNTTUpdatePagedAttentionKVCacheTemplateModel KUpdate,
    PyNTTUpdatePagedAttentionKVCacheTemplateModel VUpdate,
    PyNTTBufferPointerTemplateModel QOutput,
    string QOutputDType,
    string QOutputTritonDType,
    PyNTTDimExpression[] QOutputShape,
    PyNTTDimExpression[] QOutputStrides,
    int QOutputVectorLaneCount,
    int[] QOutputVectorLaneShape,
    int[] QKVLayout,
    int[] AttentionLayout,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTPagedAttentionTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Query,
    PyNTTBufferPointerTemplateModel Scale,
    PyNTTBufferPointerTemplateModel Output,
    string QueryDType,
    string QueryTritonDType,
    string OutputDType,
    string OutputTritonDType,
    PyNTTDimExpression[] QueryShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] OutputGlobalShape,
    PyNTTDimExpression[] QueryStrides,
    PyNTTDimExpression[] OutputStrides,
    int[] QueryVectorLaneShape,
    int[] OutputVectorLaneShape,
    PyNTTShardAxisTemplateModel[] OutputShardAxes,
    int[] Hierarchy,
    int SeqAxis,
    int HeadAxis,
    int DimAxis,
    int GlobalNumQueryHeads,
    string LayerIdExpression,
    PyNTTPagedAttentionCacheTemplateModel Cache,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTPagedAttentionPartialTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Query,
    PyNTTBufferPointerTemplateModel Scale,
    PyNTTBufferPointerTemplateModel MaxState,
    PyNTTBufferPointerTemplateModel SumState,
    PyNTTBufferPointerTemplateModel AccState,
    PyNTTBufferPointerTemplateModel? Output,
    bool HasDirectOutput,
    string QueryDType,
    string QueryTritonDType,
    PyNTTDimExpression[] QueryShape,
    PyNTTDimExpression[] QueryStrides,
    int[] QueryVectorLaneShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] OutputGlobalShape,
    PyNTTDimExpression[] OutputStrides,
    int[] OutputVectorLaneShape,
    PyNTTShardAxisTemplateModel[] OutputShardAxes,
    PyNTTDimExpression[] MaxStateShape,
    PyNTTDimExpression[] MaxStateStrides,
    PyNTTDimExpression[] SumStateShape,
    PyNTTDimExpression[] SumStateStrides,
    PyNTTDimExpression[] AccStateShape,
    PyNTTDimExpression[] AccStateStrides,
    int[] Hierarchy,
    int SeqAxis,
    int HeadAxis,
    int DimAxis,
    int GlobalNumQueryHeads,
    string LayerIdExpression,
    int SplitHierarchyAxis,
    int SplitCount,
    long DirectContextThreshold,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string? KeyDescriptorName,
    string? ValueDescriptorName,
    string BlockTableArgument,
    string? KVCacheArgument,
    string NumBlocksPerShardArgument,
    string QueryStartLocArgument,
    string SeqLensArgument,
    string NumSeqsArgument,
    PyNTTPagedAttentionCacheTemplateModel Cache,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTPagedAttentionMergeTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel MaxState,
    PyNTTPooledByteAddressTemplateModel MaxStateAddress,
    PyNTTBufferPointerTemplateModel SumState,
    PyNTTPooledByteAddressTemplateModel SumStateAddress,
    PyNTTBufferPointerTemplateModel AccState,
    PyNTTPooledByteAddressTemplateModel AccStateAddress,
    PyNTTBufferPointerTemplateModel Output,
    string OutputDType,
    string OutputTritonDType,
    PyNTTDimExpression[] MaxStateShape,
    PyNTTDimExpression[] MaxStateStrides,
    PyNTTDimExpression[] SumStateShape,
    PyNTTDimExpression[] SumStateStrides,
    PyNTTDimExpression[] AccStateShape,
    PyNTTDimExpression[] AccStateStrides,
    PyNTTDimExpression[] StateGlobalShape,
    PyNTTShardAxisTemplateModel[] StateShardAxes,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] OutputGlobalShape,
    PyNTTDimExpression[] OutputStrides,
    int[] OutputVectorLaneShape,
    PyNTTShardAxisTemplateModel[] OutputShardAxes,
    int[] Hierarchy,
    int SeqAxis,
    int HeadAxis,
    int DimAxis,
    int HeadDimension,
    int GlobalNumQueryHeads,
    int SplitHierarchyAxis,
    int SplitCount,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTConv2DTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Weights,
    PyNTTBufferPointerTemplateModel Bias,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string WeightsDType,
    string BiasDType,
    string OutputDType,
    string InputTritonDType,
    string WeightsTritonDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] WeightsShape,
    PyNTTDimExpression[] BiasShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] WeightsStrides,
    PyNTTDimExpression[] BiasStrides,
    PyNTTDimExpression[] OutputStrides,
    long[] Stride,
    long[] Padding,
    long[] Dilation,
    long Groups,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTTransposeTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int InputVectorLaneCount,
    int OutputVectorLaneCount,
    int[] InputVectorLaneShape,
    int[] OutputVectorLaneShape,
    int[] Perm,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTMatmulTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Lhs,
    PyNTTBufferPointerTemplateModel Rhs,
    PyNTTBufferPointerTemplateModel Output,
    string LhsDType,
    string RhsDType,
    string OutputDType,
    string LhsTritonDType,
    string RhsTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] LhsShape,
    PyNTTDimExpression[] RhsShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] LhsStrides,
    PyNTTDimExpression[] RhsStrides,
    PyNTTDimExpression[] OutputStrides,
    bool TransposeA,
    bool TransposeB,
    int[] Hierarchy,
    int RhsNVectorLaneCount,
    int OutputNVectorLaneCount,
    string Scale,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public PyNTTBufferPointerTemplateModel? LhsScale { get; set; }

    public PyNTTBufferPointerTemplateModel? RhsScale { get; set; }

    public bool HasOperandScales => LhsScale is not null && RhsScale is not null;

    public PyNTTDimExpression[] LhsGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] OutputGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] RhsScaleShape { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] RhsScaleStrides { get; set; } = Array.Empty<PyNTTDimExpression>();

    public long WeightBlockN { get; set; }

    public long WeightBlockK { get; set; }

    public bool HasRhsBlockScale =>
        LhsScale is null && RhsScale is not null && WeightBlockN > 0 && WeightBlockK > 0;

    public PyNTTDimExpression[] RhsGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public string? RhsDescriptorName { get; set; }

    public string RhsDescriptorOriginElements { get; set; } = "0";

    public int RhsNPackedLaneCount { get; set; } = 1;

    public int OutputNPackedLaneCount { get; set; } = 1;

    public string RhsLayout { get; set; } = "n_major";

    public int RhsKPackLaneCount { get; set; } = 1;

    public int RhsKVectorLaneCount { get; set; } = 1;

    public string LoadCExpression { get; set; } = "False";

    public PyNTTBufferPointerTemplateModel? Addend { get; set; }

    public PyNTTDimExpression[] AddendShape { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] AddendStrides { get; set; } = Array.Empty<PyNTTDimExpression>();

    public bool HasAddend => Addend is not null;

    public PyNTTBufferPointerTemplateModel? Stats { get; set; }

    public string StatsDType { get; set; } = "float32";

    public string StatsTritonDType { get; set; } = "tl.float32";

    public PyNTTDimExpression[] StatsShape { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] StatsStrides { get; set; } = Array.Empty<PyNTTDimExpression>();

    public int NormAxis { get; set; } = -1;

    public bool UseMean { get; set; }

    public bool HasNormStats => Stats is not null;
}

public sealed record PyNTTQKVParallelLinearTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel QWeight,
    PyNTTBufferPointerTemplateModel KWeight,
    PyNTTBufferPointerTemplateModel VWeight,
    PyNTTBufferPointerTemplateModel QBias,
    PyNTTBufferPointerTemplateModel KBias,
    PyNTTBufferPointerTemplateModel VBias,
    PyNTTBufferPointerTemplateModel QOutput,
    PyNTTBufferPointerTemplateModel KOutput,
    PyNTTBufferPointerTemplateModel VOutput,
    bool HasQBias,
    bool HasKBias,
    bool HasVBias,
    string InputDType,
    string WeightDType,
    string BiasDType,
    string OutputDType,
    string InputTritonDType,
    string WeightTritonDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] QWeightShape,
    PyNTTDimExpression[] KWeightShape,
    PyNTTDimExpression[] VWeightShape,
    PyNTTDimExpression[] QBiasShape,
    PyNTTDimExpression[] KBiasShape,
    PyNTTDimExpression[] VBiasShape,
    PyNTTDimExpression[] QOutputShape,
    PyNTTDimExpression[] KOutputShape,
    PyNTTDimExpression[] VOutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] QWeightStrides,
    PyNTTDimExpression[] KWeightStrides,
    PyNTTDimExpression[] VWeightStrides,
    PyNTTDimExpression[] QBiasStrides,
    PyNTTDimExpression[] KBiasStrides,
    PyNTTDimExpression[] VBiasStrides,
    PyNTTDimExpression[] QOutputStrides,
    PyNTTDimExpression[] KOutputStrides,
    PyNTTDimExpression[] VOutputStrides,
    int[] Hierarchy,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public bool PackedN { get; set; }

    public int NPackedLaneCount { get; set; } = 1;

    public int NVectorLaneCount { get; set; } = 1;

    public string RhsLayout { get; set; } = "n_major";

    public int KPackLaneCount { get; set; } = 1;

    public int KVectorLaneCount { get; set; } = 1;

    public PyNTTDimExpression[] QWeightGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] KWeightGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] VWeightGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public string? QWeightDescriptorName { get; set; }

    public string QWeightDescriptorOriginElements { get; set; } = "0";

    public string? KWeightDescriptorName { get; set; }

    public string KWeightDescriptorOriginElements { get; set; } = "0";

    public string? VWeightDescriptorName { get; set; }

    public string VWeightDescriptorOriginElements { get; set; } = "0";
}

public sealed record PyNTTPackedQKVParallelLinearTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Weight,
    PyNTTBufferPointerTemplateModel QBias,
    PyNTTBufferPointerTemplateModel KBias,
    PyNTTBufferPointerTemplateModel VBias,
    PyNTTBufferPointerTemplateModel QOutput,
    PyNTTBufferPointerTemplateModel KOutput,
    PyNTTBufferPointerTemplateModel VOutput,
    bool HasQBias,
    bool HasKBias,
    bool HasVBias,
    string InputDType,
    string WeightDType,
    string BiasDType,
    string OutputDType,
    string InputTritonDType,
    string WeightTritonDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] WeightShape,
    PyNTTDimExpression[] QBiasShape,
    PyNTTDimExpression[] KBiasShape,
    PyNTTDimExpression[] VBiasShape,
    PyNTTDimExpression[] QOutputShape,
    PyNTTDimExpression[] KOutputShape,
    PyNTTDimExpression[] VOutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] WeightStrides,
    PyNTTDimExpression[] QBiasStrides,
    PyNTTDimExpression[] KBiasStrides,
    PyNTTDimExpression[] VBiasStrides,
    PyNTTDimExpression[] QOutputStrides,
    PyNTTDimExpression[] KOutputStrides,
    PyNTTDimExpression[] VOutputStrides,
    long[] ProjectionNCapacities,
    int[] Hierarchy,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public PyNTTBufferPointerTemplateModel? QInputScale { get; set; }

    public PyNTTBufferPointerTemplateModel? KInputScale { get; set; }

    public PyNTTBufferPointerTemplateModel? VInputScale { get; set; }

    public PyNTTBufferPointerTemplateModel? QWeightScale { get; set; }

    public PyNTTBufferPointerTemplateModel? KWeightScale { get; set; }

    public PyNTTBufferPointerTemplateModel? VWeightScale { get; set; }

    public bool HasOperandScales =>
        QInputScale is not null && KInputScale is not null && VInputScale is not null &&
        QWeightScale is not null && KWeightScale is not null && VWeightScale is not null;

    public bool PackedN => true;

    public int NPackedLaneCount { get; set; } = 1;

    public int NVectorLaneCount { get; set; } = 1;

    public string RhsLayout { get; set; } = "k_major";

    public int KPackLaneCount { get; set; } = 1;

    public int KVectorLaneCount { get; set; } = 1;

    public PyNTTDimExpression[] WeightGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public string? WeightDescriptorName { get; set; }

    public string WeightDescriptorOriginElements { get; set; } = "0";
}

public sealed record PyNTTMatMulGluTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel GateWeight,
    PyNTTBufferPointerTemplateModel UpWeight,
    PyNTTBufferPointerTemplateModel GateBias,
    PyNTTBufferPointerTemplateModel UpBias,
    PyNTTBufferPointerTemplateModel Output,
    bool HasGateBias,
    bool HasUpBias,
    string GluType,
    string InputDType,
    string WeightDType,
    string BiasDType,
    string OutputDType,
    string InputTritonDType,
    string WeightTritonDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] GateWeightShape,
    PyNTTDimExpression[] UpWeightShape,
    PyNTTDimExpression[] GateBiasShape,
    PyNTTDimExpression[] UpBiasShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] GateWeightStrides,
    PyNTTDimExpression[] UpWeightStrides,
    PyNTTDimExpression[] GateBiasStrides,
    PyNTTDimExpression[] UpBiasStrides,
    PyNTTDimExpression[] OutputStrides,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public PyNTTBufferPointerTemplateModel? GateInputScale { get; set; }

    public PyNTTBufferPointerTemplateModel? UpInputScale { get; set; }

    public PyNTTBufferPointerTemplateModel? GateWeightScale { get; set; }

    public PyNTTBufferPointerTemplateModel? UpWeightScale { get; set; }

    public bool HasOperandScales =>
        GateInputScale is not null && UpInputScale is not null &&
        GateWeightScale is not null && UpWeightScale is not null;

    public bool HasWeightScales =>
        GateWeightScale is not null && UpWeightScale is not null;

    public string QuantizationMode { get; set; } = "none";

    public int WeightBlockN { get; set; }

    public int WeightBlockK { get; set; }

    public PyNTTDimExpression[] GateWeightScaleShape { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] UpWeightScaleShape { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] GateWeightScaleStrides { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] UpWeightScaleStrides { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] InputGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] OutputGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public bool PackedN { get; set; }

    public int NPackedLaneCount { get; set; } = 1;

    public int NVectorLaneCount { get; set; } = 1;

    public string RhsLayout { get; set; } = "n_major";

    public int KPackLaneCount { get; set; } = 1;

    public int KVectorLaneCount { get; set; } = 1;

    public PyNTTDimExpression[] GateWeightGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public PyNTTDimExpression[] UpWeightGlobalOffsets { get; set; } = Array.Empty<PyNTTDimExpression>();

    public string? GateWeightDescriptorName { get; set; }

    public string GateWeightDescriptorOriginElements { get; set; } = "0";

    public string? UpWeightDescriptorName { get; set; }

    public string UpWeightDescriptorOriginElements { get; set; } = "0";
}

public sealed record PyNTTSummaTemplateModel(
    string FunctionName,
    string LhsBaseName,
    string LhsOffsetBytes,
    string LhsPoolBytes,
    int LhsAddressSpace,
    string RhsBaseName,
    string RhsOffsetBytes,
    string RhsPoolBytes,
    int RhsAddressSpace,
    string OutputBaseName,
    string OutputOffsetBytes,
    string OutputPoolBytes,
    int OutputAddressSpace,
    string LhsDType,
    string RhsDType,
    string OutputDType,
    string LhsTritonDType,
    string RhsTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] LhsShape,
    PyNTTDimExpression[] RhsShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] LhsGlobalShape,
    PyNTTDimExpression[] RhsGlobalShape,
    PyNTTDimExpression[] OutputGlobalShape,
    PyNTTDimExpression[] LhsStrides,
    PyNTTDimExpression[] RhsStrides,
    PyNTTDimExpression[] OutputStrides,
    PyNTTShardAxisTemplateModel[] LhsShardAxes,
    PyNTTShardAxisTemplateModel[] RhsShardAxes,
    PyNTTShardAxisTemplateModel[] OutputShardAxes,
    int[] Hierarchy,
    int RhsNVectorLaneCount,
    int OutputNVectorLaneCount,
    int[] RhsNVectorLaneShape,
    int[] OutputNVectorLaneShape,
    string Scale,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTReduceTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int[] Axes,
    bool KeepDims,
    string ReduceOp,
    string InitValue,
    string UpdateExpression,
    string FinalizeExpression,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTSoftmaxTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Output,
    string InputDType,
    string OutputDType,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] Shape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int Axis,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTTopKTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel Values,
    PyNTTBufferPointerTemplateModel Indices,
    string InputTritonDType,
    string ValuesTritonDType,
    string IndicesTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] ValuesShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] ValuesStrides,
    PyNTTDimExpression[] IndicesStrides,
    int Axis,
    int K,
    int AxisBlockSize,
    bool Largest,
    bool Sorted,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTSparseExpertsGateUpTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Input,
    PyNTTBufferPointerTemplateModel ExpertIds,
    PyNTTBufferPointerTemplateModel GateInputScale,
    PyNTTBufferPointerTemplateModel GateWeight,
    PyNTTBufferPointerTemplateModel GateScale,
    PyNTTBufferPointerTemplateModel UpInputScale,
    PyNTTBufferPointerTemplateModel UpWeight,
    PyNTTBufferPointerTemplateModel UpScale,
    PyNTTBufferPointerTemplateModel Output,
    string InputTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] InputShape,
    PyNTTDimExpression[] GateWeightShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] ExpertIdsStrides,
    PyNTTDimExpression[] GateInputScaleStrides,
    PyNTTDimExpression[] GateWeightStrides,
    PyNTTDimExpression[] GateScaleStrides,
    PyNTTDimExpression[] UpInputScaleStrides,
    PyNTTDimExpression[] UpWeightStrides,
    PyNTTDimExpression[] UpScaleStrides,
    PyNTTDimExpression[] OutputStrides,
    int HiddenSize,
    int IntermediateSize,
    int NumExperts,
    int NumTopK,
    int InputLaneCount,
    int OutputLaneCount,
    int LocalIntermediateSize,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTSparseExpertsDownTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Activations,
    PyNTTBufferPointerTemplateModel ExpertIds,
    PyNTTBufferPointerTemplateModel ExpertWeights,
    PyNTTBufferPointerTemplateModel DownInputScale,
    PyNTTBufferPointerTemplateModel DownWeight,
    PyNTTBufferPointerTemplateModel DownScale,
    PyNTTBufferPointerTemplateModel Output,
    string ActivationTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] ActivationShape,
    PyNTTDimExpression[] DownWeightShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] ActivationStrides,
    PyNTTDimExpression[] ExpertIdsStrides,
    PyNTTDimExpression[] ExpertWeightsStrides,
    PyNTTDimExpression[] DownInputScaleStrides,
    PyNTTDimExpression[] DownWeightStrides,
    PyNTTDimExpression[] DownScaleStrides,
    PyNTTDimExpression[] OutputStrides,
    int HiddenSize,
    int IntermediateSize,
    int NumExperts,
    int NumTopK,
    int ActivationLaneCount,
    int OutputLaneCount,
    int LocalIntermediateSize,
    int LocalOutputSize,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTGatedDeltaNetConvolutionTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel QKV,
    PyNTTBufferPointerTemplateModel ConvState,
    PyNTTBufferPointerTemplateModel ConvWeight,
    PyNTTBufferPointerTemplateModel QKVOutput,
    string ActivationTritonDType,
    PyNTTDimExpression[] QKVStrides,
    PyNTTDimExpression[] ConvWeightStrides,
    PyNTTDimExpression[] QKVOutputStrides,
    PyNTTGatedDeltaNetStateAxisTemplateModel ConvStateLayerAxis,
    PyNTTGatedDeltaNetStateAxisTemplateModel ConvStateChannelAxis,
    PyNTTGatedDeltaNetStateAxisTemplateModel ConvStateHistoryAxis,
    string LayerId,
    int ConvKernelSize,
    int LocalConvDim,
    string ActiveLocalConvDim,
    int ActivationLaneCount,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTGatedDeltaNetRecurrentCoreTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel RecurrentState,
    PyNTTBufferPointerTemplateModel QKV,
    PyNTTBufferPointerTemplateModel Z,
    PyNTTBufferPointerTemplateModel ProjectionInput,
    PyNTTBufferPointerTemplateModel BWeight,
    PyNTTBufferPointerTemplateModel AWeight,
    PyNTTBufferPointerTemplateModel ALog,
    PyNTTBufferPointerTemplateModel DtBias,
    PyNTTBufferPointerTemplateModel NormWeight,
    PyNTTBufferPointerTemplateModel CoreScratch,
    PyNTTBufferPointerTemplateModel GatedOutput,
    string ActivationTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] QKVStrides,
    PyNTTDimExpression[] ZStrides,
    PyNTTDimExpression[] ProjectionInputStrides,
    PyNTTDimExpression[] BWeightStrides,
    PyNTTDimExpression[] AWeightStrides,
    PyNTTDimExpression[] ALogStrides,
    PyNTTDimExpression[] DtBiasStrides,
    PyNTTDimExpression[] NormWeightStrides,
    PyNTTDimExpression[] CoreScratchStrides,
    PyNTTDimExpression[] GatedOutputStrides,
    PyNTTGatedDeltaNetStateAxisTemplateModel RecurrentStateLayerAxis,
    PyNTTGatedDeltaNetStateAxisTemplateModel RecurrentStateHeadAxis,
    PyNTTGatedDeltaNetStateAxisTemplateModel RecurrentStateKeyAxis,
    PyNTTGatedDeltaNetStateAxisTemplateModel RecurrentStateValueAxis,
    string LayerId,
    int HiddenSize,
    int NumKeyHeads,
    int NumValueHeads,
    int KeyHeadDim,
    int ValueHeadDim,
    int ConvDim,
    int ValueDim,
    int LocalValueCapacity,
    string ActiveLocalValueDim,
    int QKVLaneCount,
    int ZLaneCount,
    int ProjectionInputLaneCount,
    int KeyBlockSize,
    int ValueBlockSize,
    string StateBarrier,
    float Epsilon,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public string? BWeightDescriptorName { get; set; }

    public string BWeightDescriptorOriginElements { get; set; } = "0";

    public bool BWeightDescriptorOwnerIndexed { get; set; }

    public string? AWeightDescriptorName { get; set; }

    public string AWeightDescriptorOriginElements { get; set; } = "0";

    public bool AWeightDescriptorOwnerIndexed { get; set; }
}

public sealed record PyNTTGatedDeltaNetStateAxisTemplateModel(
    long Extent,
    long BlockStride,
    int LaneCount,
    long LaneStride);

public sealed record PyNTTSamplingPartialTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Logits,
    PyNTTBufferPointerTemplateModel ProcessedLogits,
    PyNTTBufferPointerTemplateModel ArgMaxState,
    string Active,
    string ProcessorFlags,
    PyNTTSamplerProcessorFlagsTemplateModel ProcessorFlagValues,
    string Temperature,
    string RequestedLogprobs,
    string FrequencyPenalty,
    string PresencePenalty,
    string RepetitionPenalty,
    string PromptTokenMask,
    string OutputTokenCounts,
    string AllowedTokenMask,
    string ForbiddenTokenMask,
    string LogitBias,
    string LogitsDType,
    string LogitsTritonDType,
    PyNTTDimExpression[] LogitsShape,
    PyNTTDimExpression[] LogitsGlobalShape,
    PyNTTDimExpression[] LogitsStrides,
    PyNTTDimExpression[] ProcessedLogitsStrides,
    PyNTTDimExpression[] ArgMaxStateStrides,
    PyNTTShardAxisTemplateModel[] LogitsShardAxes,
    int[] Hierarchy,
    int VocabSize,
    int MaxBatchSize,
    string LogprobsMode,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTSamplingCombineTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Logits,
    PyNTTBufferPointerTemplateModel ProcessedLogits,
    PyNTTBufferPointerTemplateModel ArgMaxState,
    PyNTTPooledByteAddressTemplateModel ArgMaxStateAddress,
    PyNTTBufferPointerTemplateModel Summary,
    PyNTTBufferPointerTemplateModel SampledIds,
    PyNTTBufferPointerTemplateModel LogprobIds,
    PyNTTBufferPointerTemplateModel Logprobs,
    PyNTTBufferPointerTemplateModel Ranks,
    PyNTTBufferPointerTemplateModel Counts,
    string Active,
    string Temperature,
    string TopP,
    string TopK,
    string MinP,
    string RequestedLogprobs,
    string Seeds,
    string Counters,
    string OutputTokenCounts,
    string LogitsDType,
    string LogitsTritonDType,
    PyNTTDimExpression[] LogitsShape,
    PyNTTDimExpression[] LogitsGlobalShape,
    PyNTTDimExpression[] LogitsStrides,
    PyNTTDimExpression[] ProcessedLogitsStrides,
    PyNTTDimExpression[] ArgMaxStateStrides,
    PyNTTShardAxisTemplateModel[] LogitsShardAxes,
    int[] Hierarchy,
    int VocabSize,
    int MaxBatchSize,
    int MaxLogprobs,
    string LogprobsMode,
    int BlockCount,
    int RadixBits,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTSamplerProcessorFlagsTemplateModel(
    uint AllowedTokenMask,
    uint ForbiddenTokenMask,
    uint LogitBias,
    uint RepetitionPenalty,
    uint FrequencyPenalty,
    uint PresencePenalty);

public sealed record PyNTTPackedMatMulSamplingPartialTemplateModel(
    string FunctionName,
    PyNTTMatmulTemplateModel Matmul,
    PyNTTSamplingPartialTemplateModel Sampling,
    PyNTTMicroKernelTemplateModel MicroKernel,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}
