// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Nncase.IR.NN;

public enum GatedDeltaNetStateKind : byte
{
    Convolution = 0,
    Recurrent = 1,
}

public enum GatedDeltaNetStateDimKind : byte
{
    NumLayers = 0,
    ConvChannels = 1,
    ConvHistory = 2,
    NumValueHeads = 3,
    KeyHeadDim = 4,
    ValueHeadDim = 5,
}

/// <summary>
/// Physical storage contract for the persistent state of a gated delta network.
/// Tensor shapes keep semantic dimensions; vector lanes describe only packed storage.
/// </summary>
public sealed record GatedDeltaNetStateConfig
{
    private static readonly GatedDeltaNetStateDimKind[] RequiredConvolutionAxes =
    [
        GatedDeltaNetStateDimKind.NumLayers,
        GatedDeltaNetStateDimKind.ConvChannels,
        GatedDeltaNetStateDimKind.ConvHistory,
    ];

    private static readonly GatedDeltaNetStateDimKind[] RequiredRecurrentAxes =
    [
        GatedDeltaNetStateDimKind.NumLayers,
        GatedDeltaNetStateDimKind.NumValueHeads,
        GatedDeltaNetStateDimKind.KeyHeadDim,
        GatedDeltaNetStateDimKind.ValueHeadDim,
    ];

    public GatedDeltaNetStateConfig(
        int numLayers,
        int numKeyHeads,
        int numValueHeads,
        int keyHeadDim,
        int valueHeadDim,
        int convKernelSize,
        int hiddenSize,
        PrimType activationPrimType,
        IRArray<int> activationLanes,
        IRArray<GatedDeltaNetStateDimKind> convolutionLayout,
        IRArray<GatedDeltaNetStateDimKind> convolutionVectorizedAxes,
        IRArray<int> convolutionLanes,
        IRArray<GatedDeltaNetStateDimKind> recurrentLayout,
        IRArray<GatedDeltaNetStateDimKind> recurrentVectorizedAxes,
        IRArray<int> recurrentLanes)
    {
        NumLayers = RequirePositive(numLayers, nameof(numLayers));
        NumKeyHeads = RequirePositive(numKeyHeads, nameof(numKeyHeads));
        NumValueHeads = RequirePositive(numValueHeads, nameof(numValueHeads));
        KeyHeadDim = RequirePositive(keyHeadDim, nameof(keyHeadDim));
        ValueHeadDim = RequirePositive(valueHeadDim, nameof(valueHeadDim));
        ConvKernelSize = RequirePositive(convKernelSize, nameof(convKernelSize));
        HiddenSize = RequirePositive(hiddenSize, nameof(hiddenSize));
        ActivationPrimType = activationPrimType ?? throw new ArgumentNullException(nameof(activationPrimType));
        ActivationLanes = ValidateLanes(activationLanes, nameof(activationLanes));
        ConvolutionLayout = ValidateLayout(
            convolutionLayout,
            RequiredConvolutionAxes,
            nameof(convolutionLayout));
        ConvolutionVectorizedAxes = ValidateVectorizedAxes(
            convolutionVectorizedAxes,
            convolutionLanes,
            RequiredConvolutionAxes,
            nameof(convolutionVectorizedAxes));
        ConvolutionLanes = ValidateLanes(convolutionLanes, nameof(convolutionLanes));
        RecurrentLayout = ValidateLayout(
            recurrentLayout,
            RequiredRecurrentAxes,
            nameof(recurrentLayout));
        RecurrentVectorizedAxes = ValidateVectorizedAxes(
            recurrentVectorizedAxes,
            recurrentLanes,
            RequiredRecurrentAxes,
            nameof(recurrentVectorizedAxes));
        RecurrentLanes = ValidateLanes(recurrentLanes, nameof(recurrentLanes));

        if (NumValueHeads % NumKeyHeads != 0)
        {
            throw new ArgumentException(
                $"GDN value-head count {NumValueHeads} must be divisible by key-head count {NumKeyHeads}.");
        }

        ValidatePacking(GatedDeltaNetStateKind.Convolution);
        ValidatePacking(GatedDeltaNetStateKind.Recurrent);
        ValidateActivationPacking();
    }

    public int NumLayers { get; }

    public int NumKeyHeads { get; }

    public int NumValueHeads { get; }

    public int KeyHeadDim { get; }

    public int ValueHeadDim { get; }

    public int ConvKernelSize { get; }

    public int HiddenSize { get; }

    public PrimType ActivationPrimType { get; }

    public IRArray<int> ActivationLanes { get; }

    public IRArray<GatedDeltaNetStateDimKind> ConvolutionLayout { get; }

    public IRArray<GatedDeltaNetStateDimKind> ConvolutionVectorizedAxes { get; }

    public IRArray<int> ConvolutionLanes { get; }

    public IRArray<GatedDeltaNetStateDimKind> RecurrentLayout { get; }

    public IRArray<GatedDeltaNetStateDimKind> RecurrentVectorizedAxes { get; }

    public IRArray<int> RecurrentLanes { get; }

    public DataType ActivationType => ActivationLanes.Count == 0
        ? ActivationPrimType
        : new VectorType(ActivationPrimType, ActivationLanes);

    public DataType GetStateType(GatedDeltaNetStateKind kind)
    {
        var lanes = GetLanes(kind);
        var primType = kind == GatedDeltaNetStateKind.Convolution
            ? ActivationPrimType
            : DataTypes.Float32;
        return lanes.Count == 0 ? primType : new VectorType(primType, lanes);
    }

    public IRArray<GatedDeltaNetStateDimKind> GetLayout(GatedDeltaNetStateKind kind) => kind switch
    {
        GatedDeltaNetStateKind.Convolution => ConvolutionLayout,
        GatedDeltaNetStateKind.Recurrent => RecurrentLayout,
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    public IRArray<GatedDeltaNetStateDimKind> GetVectorizedAxes(GatedDeltaNetStateKind kind) => kind switch
    {
        GatedDeltaNetStateKind.Convolution => ConvolutionVectorizedAxes,
        GatedDeltaNetStateKind.Recurrent => RecurrentVectorizedAxes,
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    public IRArray<int> GetLanes(GatedDeltaNetStateKind kind) => kind switch
    {
        GatedDeltaNetStateKind.Convolution => ConvolutionLanes,
        GatedDeltaNetStateKind.Recurrent => RecurrentLanes,
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    public IRArray<int> GetLanes(
        GatedDeltaNetStateKind kind,
        GatedDeltaNetStateDimKind axis) =>
        GetVectorizedAxes(kind)
            .Select((vectorizedAxis, index) => (vectorizedAxis, lane: GetLanes(kind)[index]))
            .Where(item => item.vectorizedAxis == axis)
            .Select(item => item.lane)
            .ToArray();

    public long GetDimension(GatedDeltaNetStateDimKind axis) => axis switch
    {
        GatedDeltaNetStateDimKind.NumLayers => NumLayers,
        GatedDeltaNetStateDimKind.ConvChannels => checked(
            (NumKeyHeads * KeyHeadDim * 2L) + (NumValueHeads * ValueHeadDim)),
        GatedDeltaNetStateDimKind.ConvHistory => ConvKernelSize - 1L,
        GatedDeltaNetStateDimKind.NumValueHeads => NumValueHeads,
        GatedDeltaNetStateDimKind.KeyHeadDim => KeyHeadDim,
        GatedDeltaNetStateDimKind.ValueHeadDim => ValueHeadDim,
        _ => throw new ArgumentOutOfRangeException(nameof(axis)),
    };

    /// <summary>
    /// Gets the packed logical tensor type seen by compiler kernels.
    /// </summary>
    public TensorType GetLogicalTensorType(GatedDeltaNetStateKind kind)
    {
        var dimensions = GetLayout(kind).Select(GetDimension).ToArray();
        foreach (var (axis, lane) in GetVectorizedAxes(kind).Zip(GetLanes(kind)))
        {
            var physicalAxis = GetLayout(kind).IndexOf(axis);
            dimensions[physicalAxis] /= lane;
        }

        return new TensorType(GetStateType(kind), dimensions);
    }

    /// <summary>
    /// Gets the scalar torch storage shape. Logical tensor dimensions precede the
    /// trailing physical vector-lane dimensions, matching nncase Pack storage.
    /// </summary>
    public long[] GetStorageShape(GatedDeltaNetStateKind kind)
    {
        var logicalShape = GetLogicalTensorType(kind).Shape;
        if (!logicalShape.IsFixed)
        {
            throw new InvalidOperationException("GDN state storage requires fixed dimensions.");
        }

        return logicalShape.ToValueArray().Concat(GetLanes(kind).Select(lane => (long)lane)).ToArray();
    }

    public override string ToString() =>
        $"GatedDeltaNetStateConfig(Layers={NumLayers}, KeyHeads={NumKeyHeads}, " +
        $"ValueHeads={NumValueHeads}, KeyHeadDim={KeyHeadDim}, ValueHeadDim={ValueHeadDim}, " +
        $"ConvKernel={ConvKernelSize}, Hidden={HiddenSize}, ActivationType={ActivationType}, " +
        $"ConvLayout=[{string.Join(',', ConvolutionLayout)}], ConvType={GetStateType(GatedDeltaNetStateKind.Convolution)}, " +
        $"RecurrentLayout=[{string.Join(',', RecurrentLayout)}], RecurrentType={GetStateType(GatedDeltaNetStateKind.Recurrent)})";

    private static int RequirePositive(int value, string name) => value > 0
        ? value
        : throw new ArgumentOutOfRangeException(name, value, "GDN state dimensions must be positive.");

    private static IRArray<int> ValidateLanes(IRArray<int> lanes, string name)
    {
        if (lanes.Any(lane => lane <= 1))
        {
            throw new ArgumentException("GDN packed lanes must be greater than one.", name);
        }

        return lanes;
    }

    private static IRArray<GatedDeltaNetStateDimKind> ValidateLayout(
        IRArray<GatedDeltaNetStateDimKind> layout,
        IReadOnlyList<GatedDeltaNetStateDimKind> requiredAxes,
        string name)
    {
        if (layout.Count != requiredAxes.Count ||
            layout.Distinct().Count() != layout.Count ||
            requiredAxes.Any(axis => !layout.Contains(axis)))
        {
            throw new ArgumentException(
                $"GDN state layout must contain [{string.Join(',', requiredAxes)}] exactly once.",
                name);
        }

        return layout;
    }

    private static IRArray<GatedDeltaNetStateDimKind> ValidateVectorizedAxes(
        IRArray<GatedDeltaNetStateDimKind> axes,
        IRArray<int> lanes,
        IReadOnlyList<GatedDeltaNetStateDimKind> allowedAxes,
        string name)
    {
        if (axes.Count != lanes.Count)
        {
            throw new ArgumentException("GDN vectorized axes and lanes must have the same length.", name);
        }

        if (axes.Any(axis => !allowedAxes.Contains(axis)))
        {
            throw new ArgumentException("GDN vectorized axis is not present in the corresponding state.", name);
        }

        if (axes.Distinct().Count() != axes.Count)
        {
            throw new ArgumentException("GDN vectorized axes must be unique.", name);
        }

        return axes;
    }

    private void ValidatePacking(GatedDeltaNetStateKind kind)
    {
        var remaining = GetLayout(kind).ToDictionary(axis => axis, GetDimension);
        foreach (var (axis, lane) in GetVectorizedAxes(kind).Zip(GetLanes(kind)))
        {
            if (remaining[axis] % lane != 0)
            {
                throw new ArgumentException(
                    $"GDN {kind} axis {axis} extent {remaining[axis]} is not divisible by lane {lane}.");
            }

            remaining[axis] /= lane;
        }
    }

    private void ValidateActivationPacking()
    {
        var remaining = (long)HiddenSize;
        foreach (var lane in ActivationLanes)
        {
            if (remaining % lane != 0)
            {
                throw new ArgumentException(
                    $"GDN hidden extent {remaining} is not divisible by activation lane {lane}.");
            }

            remaining /= lane;
        }
    }
}

public interface IGatedDeltaNetState
{
    GatedDeltaNetStateConfig Config { get; }

    Tensor GetState(GatedDeltaNetStateKind kind, int layerId);

    void UpdateState(GatedDeltaNetStateKind kind, int layerId, Tensor value);
}

public sealed record GatedDeltaNetStateType : ValueType
{
    public GatedDeltaNetStateConfig Config { get; init; } = null!;

    public override Type CLRType => typeof(IGatedDeltaNetState);

    public override int SizeInBytes => IntPtr.Size;

    public override Guid Uuid { get; } = new("628a2436-c7c4-4aa9-9d66-d9e2f1f6a0f4");

    public override string ToString() => "GatedDeltaNetState";
}
