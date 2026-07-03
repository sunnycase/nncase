// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.PyNTT;

public sealed record PyNTTBufferPointerTemplateModel(string Expression);

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
    int[] Hierarchy,
    int[][] SplitAxes,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
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
    int[] Hierarchy,
    int[][] SplitAxes,
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
    string InputBaseName,
    string InputOffsetBytes,
    long InputPoolBytes,
    string OutputBaseName,
    string OutputOffsetBytes,
    long OutputPoolBytes,
    int ScalarElementSizeBytes,
    string DType,
    string TritonDType,
    PyNTTDimExpression[] GlobalShape,
    PyNTTDimExpression[] InputLocalShape,
    PyNTTDimExpression[] OutputLocalShape,
    PyNTTDimExpression[] InputStrides,
    PyNTTDimExpression[] OutputStrides,
    int VectorLaneCount,
    int[] Hierarchy,
    int[][] InputSplitAxes,
    int[][] OutputSplitAxes,
    long CollectiveOffsetBytes,
    long CollectivePoolBytes,
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
    int InputVectorLaneCount,
    int CosVectorLaneCount,
    int SinVectorLaneCount,
    int OutputVectorLaneCount,
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
    PyNTTDimExpression[] OutputStrides,
    int[] Hierarchy,
    int[][] SplitAxes,
    int? ShardAxis,
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
    int[] TailShape,
    int BlockSize);

public sealed record PyNTTPagedAttentionCacheTemplateModel(
    string DType,
    string TritonDType,
    int NumLayers,
    int NumKVHeads,
    int HeadDim,
    int BlockSize,
    int LaneCount,
    int HeadDimBlocks,
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
    PyNTTDimExpression[] SlotsStrides,
    int[][] SlotsSplitAxes,
    int[] Hierarchy,
    string SlotsBaseName,
    string SlotsOffsetBytes,
    long SlotsPoolBytes,
    int ScalarElementSizeBytes,
    int SeqAxis,
    int HeadAxis,
    int DimAxis,
    int LayerId,
    int CacheKind,
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
    int[][] OutputSplitAxes,
    int[] Hierarchy,
    int SeqAxis,
    int HeadAxis,
    int DimAxis,
    int GlobalNumQueryHeads,
    int LayerId,
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
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();

    public int RhsNPackedLaneCount { get; set; } = 1;

    public int OutputNPackedLaneCount { get; set; } = 1;
}

public sealed record PyNTTSummaTemplateModel(
    string FunctionName,
    string LhsBaseName,
    string LhsOffsetBytes,
    long LhsPoolBytes,
    string RhsBaseName,
    string RhsOffsetBytes,
    long RhsPoolBytes,
    string OutputBaseName,
    string OutputOffsetBytes,
    long OutputPoolBytes,
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
    string Scale,
    string Comment)
{
    public string[] RuntimeShapeArgs { get; set; } = Array.Empty<string>();
}

public sealed record PyNTTShardReduceTemplateModel(
    string FunctionName,
    string BaseName,
    string DType,
    string TritonDType,
    PyNTTDimExpression[] LocalShape,
    PyNTTDimExpression[] Strides,
    int VectorLaneCount,
    int[] Hierarchy,
    int[] ReduceAxes,
    bool Broadcast,
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
