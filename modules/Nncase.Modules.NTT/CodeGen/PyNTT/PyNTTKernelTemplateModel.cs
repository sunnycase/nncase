// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.PyNTT;

public interface IPyNTTBlockMicroKernelTemplateModel
{
    string FunctionName { get; }

    string MicroKernelFamily { get; set; }

    string MicroKernelVariant { get; set; }

    Dictionary<string, long> MicroKernelParameters { get; set; }
}

public interface IPyNTTMatrixMicroKernelTemplateModel : IPyNTTBlockMicroKernelTemplateModel
{
}

public sealed record PyNTTBufferPointerTemplateModel(
    string Expression,
    int[]? ShardCoordHierarchy = null,
    int AddressSpace = 1,
    PyNTTLocalBufferTemplateModel? LocalBuffer = null);

public sealed record PyNTTLocalBufferTemplateModel(
    string DescriptorExpression,
    long[] DescriptorShape,
    long[] LogicalShape,
    long[] LogicalStrides,
    PyNTTDimExpression[] BaseCoordinates,
    int[] VectorLaneShape,
    long AvailableBytes,
    int ScalarElementSizeBytes,
    string StorageEncoding);

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
    int[][] SplitAxes,
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
    int[][] SplitAxes,
    int VectorLaneCount,
    int[] VectorLaneShape,
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
    int[][] InputSplitAxes,
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
    int[][] InputSplitAxes,
    int[] InputPartialAxes,
    int[][] OutputSplitAxes,
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

internal sealed record PyNTTKVCacheFieldInputMetadata(
    string Name,
    string SourceName,
    string Field,
    PyNTTKVCacheStorageMetadata? Storage);

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
    int[] NumBlocksSplitAxes);

public sealed record PyNTTUpdatePagedAttentionKVCacheTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Slots,
    string SlotsDType,
    string SlotsTritonDType,
    PyNTTDimExpression[] SlotsShape,
    PyNTTDimExpression[] SlotsGlobalShape,
    PyNTTDimExpression[] SlotsGlobalOffsets,
    PyNTTDimExpression[] SlotsStrides,
    int[][] SlotsSplitAxes,
    int[][] SlotsSourceSplitAxes,
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
    int[][] OutputSplitAxes,
    int[] Hierarchy,
    int SeqAxis,
    int HeadAxis,
    int DimAxis,
    int GlobalNumQueryHeads,
    string LayerIdExpression,
    int AttentionBlockSize,
    PyNTTPagedAttentionCacheTemplateModel Cache,
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
    string Comment) : IPyNTTMatrixMicroKernelTemplateModel
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public int RhsNPackedLaneCount { get; set; } = 1;

    public int OutputNPackedLaneCount { get; set; } = 1;

    public string LoadCExpression { get; set; } = "False";

    public string ReductionPhase { get; set; } = "complete";

    public int ReductionBlockM { get; set; }

    public int ReductionBlockN { get; set; }

    public int ReductionBlockK { get; set; }

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
}

public sealed record PyNTTMatmulReductionFinalizeTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Output,
    string OutputDType,
    string OutputTritonDType,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] OutputStrides,
    int OutputNPackedLaneCount,
    int OutputNVectorLaneCount,
    string Scale,
    bool Gemv,
    int ReductionBlockM,
    int ReductionBlockN,
    string Comment) : IPyNTTMatrixMicroKernelTemplateModel
{
    public string ReductionPhase { get; } = "finalize";

    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
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
    string Comment) : IPyNTTMatrixMicroKernelTemplateModel
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public bool PackedN { get; set; }

    public int NPackedLaneCount { get; set; } = 1;

    public int NVectorLaneCount { get; set; } = 1;

    public string ReductionPhase { get; set; } = "complete";

    public int ReductionBlockM { get; set; }

    public int ReductionBlockN { get; set; }

    public int ReductionBlockK { get; set; }

    public int ReductionQBlockN { get; set; }

    public int ReductionKBlockN { get; set; }

    public int ReductionVBlockN { get; set; }

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
}

public sealed record PyNTTQKVParallelLinearReductionFinalizeTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel QBias,
    PyNTTBufferPointerTemplateModel KBias,
    PyNTTBufferPointerTemplateModel VBias,
    PyNTTBufferPointerTemplateModel QOutput,
    PyNTTBufferPointerTemplateModel KOutput,
    PyNTTBufferPointerTemplateModel VOutput,
    bool HasQBias,
    bool HasKBias,
    bool HasVBias,
    string BiasDType,
    string OutputDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] QBiasShape,
    PyNTTDimExpression[] KBiasShape,
    PyNTTDimExpression[] VBiasShape,
    PyNTTDimExpression[] QOutputShape,
    PyNTTDimExpression[] KOutputShape,
    PyNTTDimExpression[] VOutputShape,
    PyNTTDimExpression[] QBiasStrides,
    PyNTTDimExpression[] KBiasStrides,
    PyNTTDimExpression[] VBiasStrides,
    PyNTTDimExpression[] QOutputStrides,
    PyNTTDimExpression[] KOutputStrides,
    PyNTTDimExpression[] VOutputStrides,
    bool PackedN,
    int NPackedLaneCount,
    int NVectorLaneCount,
    int ReductionBlockM,
    int ReductionQBlockN,
    int ReductionKBlockN,
    int ReductionVBlockN,
    string Comment) : IPyNTTMatrixMicroKernelTemplateModel
{
    public string ReductionPhase { get; } = "finalize";

    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
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
    string Comment) : IPyNTTMatrixMicroKernelTemplateModel
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public bool PackedN { get; set; }

    public int NPackedLaneCount { get; set; } = 1;

    public int NVectorLaneCount { get; set; } = 1;

    public string ReductionPhase { get; set; } = "complete";

    public int ReductionBlockM { get; set; }

    public int ReductionBlockN { get; set; }

    public int ReductionBlockK { get; set; }

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
}

public sealed record PyNTTMatMulGluReductionFinalizeTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel GateBias,
    PyNTTBufferPointerTemplateModel UpBias,
    PyNTTBufferPointerTemplateModel Output,
    bool HasGateBias,
    bool HasUpBias,
    string GluType,
    string BiasDType,
    string OutputDType,
    string BiasTritonDType,
    string OutputTritonDType,
    PyNTTDimExpression[] GateBiasShape,
    PyNTTDimExpression[] UpBiasShape,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] GateBiasStrides,
    PyNTTDimExpression[] UpBiasStrides,
    PyNTTDimExpression[] OutputStrides,
    bool PackedN,
    int NPackedLaneCount,
    int NVectorLaneCount,
    int ReductionBlockM,
    int ReductionBlockN,
    string Comment) : IPyNTTMatrixMicroKernelTemplateModel
{
    public string ReductionPhase { get; } = "finalize";

    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
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
    int[][] LhsSplitAxes,
    int[][] RhsSplitAxes,
    int[][] OutputSplitAxes,
    int[] Hierarchy,
    int RhsNVectorLaneCount,
    int OutputNVectorLaneCount,
    int[] RhsNVectorLaneShape,
    int[] OutputNVectorLaneShape,
    string Scale,
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
    string Comment) : IPyNTTBlockMicroKernelTemplateModel
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public string ReductionPhase { get; set; } = "complete";

    public int ReductionBlockSize { get; set; }

    public string AccumulatorTritonDType { get; set; } = "tl.float32";

    public bool TrackReductionElementCount { get; set; }

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
}

public sealed record PyNTTReduceReductionFinalizeTemplateModel(
    string FunctionName,
    PyNTTBufferPointerTemplateModel Output,
    string OutputDType,
    string OutputTritonDType,
    PyNTTDimExpression[] OutputShape,
    PyNTTDimExpression[] OutputStrides,
    string FinalizeExpression,
    int ReductionBlockSize,
    bool TrackReductionElementCount,
    string Comment) : IPyNTTBlockMicroKernelTemplateModel
{
    public string ReductionPhase { get; } = "finalize";

    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public string MicroKernelFamily { get; set; } = string.Empty;

    public string MicroKernelVariant { get; set; } = string.Empty;

    public Dictionary<string, long> MicroKernelParameters { get; set; } = new(StringComparer.Ordinal);
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
