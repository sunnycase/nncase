// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.CostModel;

/// <summary>
/// Target matmul implementation kind.
/// </summary>
public enum MatMulOpCostKind
{
    /// <summary>
    /// Target decides the implementation from tensor types and shapes.
    /// </summary>
    Auto,

    /// <summary>
    /// SIMT implementation, without tensor-core dot instructions.
    /// </summary>
    Simt,
}

/// <summary>
/// Target-provided op cost model. Evaluators build these generic queries and do
/// not depend on concrete hardware details.
/// </summary>
public interface ITargetOpCostModel
{
    UInt128 GetLatency(Cost cost);

    bool TryGetUnaryCost(UnaryOpCostQuery query, out Cost cost);

    bool TryGetBinaryCost(BinaryOpCostQuery query, out Cost cost);

    bool TryGetElementwiseCost(ElementwiseOpCostQuery query, out Cost cost);

    bool TryGetMatMulCost(MatMulOpCostQuery query, out Cost cost);
}

/// <summary>
/// Optional target cost model extension for aggregating local shard costs over a physical hierarchy.
/// </summary>
public interface IHierarchicalTargetOpCostModel
{
    UInt128 GetLatency(Cost cost, TargetCostAggregationContext context);
}

/// <summary>
/// Optional target cost model extension for explaining latency aggregation.
/// </summary>
public interface ITargetOpCostBreakdownModel
{
    TargetCostLatencyBreakdown GetLatencyBreakdown(Cost cost, TargetCostAggregationContext context);
}

/// <summary>
/// Optional target options extension for registering an op cost model.
/// </summary>
public interface ITargetOpCostModelProvider
{
    ITargetOpCostModel TargetCostModel { get; }
}

/// <summary>
/// Latency aggregation breakdown in target cycles.
/// </summary>
public sealed record TargetCostLatencyBreakdown(
    long ActiveBlockCount,
    double CPUCycles,
    double BlockLocalMemoryCycles,
    double ChipGlobalMemoryCycles,
    double OverlappedCycles,
    double BlockSynchronizationCycles,
    double GridSynchronizationCycles,
    double CommCycles,
    double OtherCycles,
    UInt128 Latency);

/// <summary>
/// Context used to aggregate a local block cost into candidate latency.
/// </summary>
public sealed record TargetCostAggregationContext(long ActiveBlockCount)
{
    public static readonly TargetCostAggregationContext Local = new(1);

    public static TargetCostAggregationContext FromResultType(IRType? resultType)
    {
        return new(Math.Max(1, GetActiveBlockCount(resultType)));
    }

    private static long GetActiveBlockCount(IRType? type)
    {
        return type switch
        {
            DistributedType distributedType => Product(distributedType.Placement.Hierarchy.Select(static x => (long)x)),
            TupleType tupleType => tupleType.Fields.Select(GetActiveBlockCount).DefaultIfEmpty(1).Max(),
            _ => 1,
        };
    }

    private static long Product(IEnumerable<long> values)
    {
        long product = 1;
        foreach (var value in values)
        {
            var factor = Math.Max(1, value);
            if (product > long.MaxValue / factor)
            {
                return long.MaxValue;
            }

            product *= factor;
        }

        return product;
    }
}

/// <summary>
/// Target op cost model helpers.
/// </summary>
public static class TargetOpCostModelUtility
{
    public static ITargetOpCostModel GetTargetCostModel(CompileOptions compileOptions)
    {
        return compileOptions.TargetOptions is ITargetOpCostModelProvider provider
            ? provider.TargetCostModel
            : DefaultTargetOpCostModel.Instance;
    }

    public static bool TryGetTargetElementwiseCost(ITargetOpCostModel targetCostModel, string op, IReadOnlyList<IRType> inputs, IRType output, double workPerElement, out Cost cost)
    {
        cost = Cost.Zero;
        var inputTensors = new TargetCostTensor[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            if (!TargetCostTensor.TryFromType(inputs[i], out inputTensors[i]))
            {
                return false;
            }
        }

        if (!TargetCostTensor.TryFromType(output, out var outputTensor))
        {
            return false;
        }

        return targetCostModel.TryGetElementwiseCost(new(op, inputTensors, outputTensor, workPerElement), out cost);
    }

    public static UInt128 GetCostLatency(ITargetOpCostModel targetCostModel, Cost cost)
    {
        return targetCostModel is IHierarchicalTargetOpCostModel hierarchicalTargetCostModel
            ? hierarchicalTargetCostModel.GetLatency(cost, TargetCostAggregationContext.Local)
            : targetCostModel.GetLatency(cost);
    }

    public static UInt128 GetCostLatency(ITargetOpCostModel targetCostModel, Cost cost, IRType? resultType)
    {
        return targetCostModel is IHierarchicalTargetOpCostModel hierarchicalTargetCostModel
            ? hierarchicalTargetCostModel.GetLatency(cost, TargetCostAggregationContext.FromResultType(resultType))
            : targetCostModel.GetLatency(cost);
    }

    public static TargetCostLatencyBreakdown GetCostLatencyBreakdown(ITargetOpCostModel targetCostModel, Cost cost, IRType? resultType)
    {
        var context = targetCostModel is IHierarchicalTargetOpCostModel
            ? TargetCostAggregationContext.FromResultType(resultType)
            : TargetCostAggregationContext.Local;
        if (targetCostModel is ITargetOpCostBreakdownModel breakdownModel)
        {
            return breakdownModel.GetLatencyBreakdown(cost, context);
        }

        return new TargetCostLatencyBreakdown(
            context.ActiveBlockCount,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            GetCostLatency(targetCostModel, cost, resultType));
    }
}

/// <summary>
/// Target-independent tensor description for target cost queries.
/// </summary>
public sealed record TargetCostTensor(DataType DType, Shape Shape)
{
    public static bool TryFromType(IRType type, out TargetCostTensor tensor)
    {
        var tensorType = type switch
        {
            TensorType tt => tt,
            DistributedType dt => DistributedUtility.GetDividedTensorType(dt, DistributedUtility.DivideFlags.MaxShape),
            _ => null,
        };

        if (tensorType is null)
        {
            tensor = null!;
            return false;
        }

        tensor = new TargetCostTensor(tensorType.DType, tensorType.Shape);
        return true;
    }
}

/// <summary>
/// Target cost query for unary operators.
/// </summary>
public sealed record UnaryOpCostQuery(UnaryOp Op, TargetCostTensor Input, TargetCostTensor Output);

/// <summary>
/// Target cost query for binary operators.
/// </summary>
public sealed record BinaryOpCostQuery(BinaryOp Op, TargetCostTensor Lhs, TargetCostTensor Rhs, TargetCostTensor Output);

/// <summary>
/// Target cost query for generic elementwise-like operators.
/// </summary>
public sealed record ElementwiseOpCostQuery(string Op, IReadOnlyList<TargetCostTensor> Inputs, TargetCostTensor Output, double WorkPerElement = 1.0);

/// <summary>
/// Target cost query for matmul operators.
/// </summary>
public sealed record MatMulOpCostQuery(TargetCostTensor Lhs, TargetCostTensor Rhs, TargetCostTensor Output, DataType OutputDataType, MatMulOpCostKind Kind = MatMulOpCostKind.Auto);

/// <summary>
/// Default target cost model used when a target does not provide one.
/// </summary>
public sealed class DefaultTargetOpCostModel : ITargetOpCostModel, IHierarchicalTargetOpCostModel, ITargetOpCostBreakdownModel
{
    public static readonly DefaultTargetOpCostModel Instance = new();

    private readonly TargetMachineModel? _machine;

    public DefaultTargetOpCostModel(TargetMachineModel machine)
    {
        _machine = machine ?? throw new ArgumentNullException(nameof(machine));
    }

    private DefaultTargetOpCostModel()
    {
    }

    public UInt128 GetLatency(Cost cost)
    {
        return GetLatencyBreakdown(cost, TargetCostAggregationContext.Local).Latency;
    }

    public UInt128 GetLatency(Cost cost, TargetCostAggregationContext context)
    {
        return GetLatencyBreakdown(cost, context).Latency;
    }

    public TargetCostLatencyBreakdown GetLatencyBreakdown(Cost cost, TargetCostAggregationContext context)
    {
        var cpuCycles = ToDouble(GetFactor(cost, CostFactorNames.CPUCycles));
        var blockLocalLoadBytes = ToDouble(GetFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes));
        var blockLocalStoreBytes = ToDouble(GetFactor(cost, CostFactorNames.BlockLocalMemoryStoreBytes));
        var chipGlobalLoadBytes = ToDouble(GetFactor(cost, CostFactorNames.ChipGlobalMemoryLoadBytes));
        var chipGlobalStoreBytes = ToDouble(GetFactor(cost, CostFactorNames.ChipGlobalMemoryStoreBytes));
        var activeBlocks = Math.Max(1L, context.ActiveBlockCount);
        var blockSpace = _machine?.TilingMemorySpaces
            .LastOrDefault(space => _machine.GetMemoryResource(space).Kind != TargetMemorySpaceKind.Global)
            ?? _machine?.TilingMemorySpaces[^1];
        var blockMemory = blockSpace is null ? null : _machine?.GetMemoryResource(blockSpace);
        var chipMemory = _machine is null ? null : _machine.GetMemoryResource(_machine.GetMemorySpace(_machine.RootMemorySpace));
        var blockLocalMemoryCycles = GetMemoryCycles(blockLocalLoadBytes, blockLocalStoreBytes, blockMemory);
        var chipGlobalMemoryCycles = GetMemoryCycles(
            (blockLocalLoadBytes + chipGlobalLoadBytes) * activeBlocks,
            (blockLocalStoreBytes + chipGlobalStoreBytes) * activeBlocks,
            chipMemory);
        var overlappedCycles = Math.Max(cpuCycles, Math.Max(blockLocalMemoryCycles, chipGlobalMemoryCycles));
        var blockSynchronizationCycles = ToDouble(GetFactor(cost, CostFactorNames.BlockSynchronization)) * (_machine?.Synchronization.BlockCycles ?? 25);
        var gridSynchronizationCycles = ToDouble(GetFactor(cost, CostFactorNames.GridSynchronization)) * (_machine?.Synchronization.GridCycles ?? 25_000);
        var commCycles = ToDouble(GetFactor(cost, CostFactorNames.Comm));
        var otherCycles = ToDouble(GetOtherCost(cost));
        var latency = overlappedCycles + blockSynchronizationCycles + gridSynchronizationCycles + commCycles + otherCycles;
        return new TargetCostLatencyBreakdown(
            activeBlocks,
            cpuCycles,
            blockLocalMemoryCycles,
            chipGlobalMemoryCycles,
            overlappedCycles,
            blockSynchronizationCycles,
            gridSynchronizationCycles,
            commCycles,
            otherCycles,
            ToCostFactor(latency));
    }

    public bool TryGetUnaryCost(UnaryOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxVectorElementCount(query.Output, out var elements))
        {
            cost = Cost.Zero;
            return false;
        }

        cost = ElementwiseCost(
            EstimateElementwiseCycles(
                elements * CostUtility.GetCPUCyclesOfUnary(query.Op),
                query.Input.DType,
                query.Output.DType),
            GetTensorByteCount(query.Input),
            GetTensorByteCount(query.Output));
        return true;
    }

    public bool TryGetBinaryCost(BinaryOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxVectorElementCount(query.Output, out var elements))
        {
            cost = Cost.Zero;
            return false;
        }

        cost = ElementwiseCost(
            EstimateElementwiseCycles(
                elements * CostUtility.GetCPUCyclesOfBinary(query.Op),
                query.Lhs.DType,
                query.Rhs.DType,
                query.Output.DType),
            GetTensorByteCount(query.Lhs) + GetTensorByteCount(query.Rhs),
            GetTensorByteCount(query.Output));
        return true;
    }

    public bool TryGetElementwiseCost(ElementwiseOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxVectorElementCount(query.Output, out var elements))
        {
            cost = Cost.Zero;
            return false;
        }

        cost = ElementwiseCost(
            EstimateElementwiseCycles(
                elements * Math.Max(0.0, query.WorkPerElement),
                query.Inputs.Select(input => input.DType).Append(query.Output.DType).ToArray()),
            query.Inputs.Sum(GetTensorByteCount),
            GetTensorByteCount(query.Output));
        return true;
    }

    public bool TryGetMatMulCost(MatMulOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxShape(query.Lhs, out var lhsShape)
            || !TryGetMaxShape(query.Output, out var outputShape)
            || lhsShape.Length < 2
            || outputShape.Length < 2)
        {
            cost = Cost.Zero;
            return false;
        }

        var m = outputShape[^2];
        var n = checked(outputShape[^1] * GetVectorLaneCount(query.Output.DType));
        var k = checked(lhsShape[^1] * GetVectorLaneCount(query.Lhs.DType));
        var batch = Product(outputShape.AsSpan(0, outputShape.Length - 2));
        var work = (double)m * n * k * batch;
        var computeCycles = work / Math.Max(1.0, _machine?.Compute.SimtFmaPerCycle ?? 1.0);
        cost = ElementwiseCost(
            computeCycles,
            GetTensorByteCount(query.Lhs) + GetTensorByteCount(query.Rhs),
            GetTensorByteCount(query.Output));
        return true;
    }

    private static int GetVectorLaneCount(DataType dataType)
        => dataType is VectorType vectorType
            ? vectorType.Lanes.Aggregate(1, static (product, lane) => checked(product * lane)) * GetVectorLaneCount(vectorType.ElemType)
            : 1;

    private static UInt128 GetFactor(Cost cost, string name)
    {
        return cost.Factors.TryGetValue(name, out var value) ? value : 0;
    }

    private static double GetMemoryCycles(double loadBytes, double storeBytes, TargetMemoryResourceSpec? memorySpace)
    {
        if (memorySpace is null)
        {
            return loadBytes + storeBytes;
        }

        return (loadBytes / memorySpace.ReadBytesPerCycle)
            + (storeBytes / memorySpace.WriteBytesPerCycle)
            + ((loadBytes + storeBytes) > 0 ? memorySpace.LatencyCycles : 0);
    }

    private static Cost ElementwiseCost(double cpuCycles, double blockLocalMemoryLoadBytes, double blockLocalMemoryStoreBytes)
    {
        var cost = new Cost();
        AddCostFactor(cost, CostFactorNames.CPUCycles, cpuCycles);
        AddCostFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes, blockLocalMemoryLoadBytes);
        AddCostFactor(cost, CostFactorNames.BlockLocalMemoryStoreBytes, blockLocalMemoryStoreBytes);
        return cost;
    }

    private static void AddCostFactor(Cost cost, string name, double value)
    {
        var factor = ToCostFactor(value);
        if (factor > 0)
        {
            cost[name] = factor;
        }
    }

    private static bool TryGetMaxVectorElementCount(TargetCostTensor tensor, out double elements)
    {
        if (TryGetMaxShape(tensor, out var shape))
        {
            elements = Product(shape) * GetVectorLaneCount(tensor.DType);
            return true;
        }

        elements = 0;
        return false;
    }

    private static bool TryGetMaxShape(TargetCostTensor tensor, out long[] shape)
    {
        if (tensor.Shape is RankedShape && CompilerServices.TryGetMaxShape(tensor.Shape, out var maxShape))
        {
            shape = maxShape;
            return true;
        }

        shape = Array.Empty<long>();
        return false;
    }

    private static double GetTensorByteCount(TargetCostTensor tensor)
    {
        return TryGetMaxShape(tensor, out var shape) ? Product(shape) * tensor.DType.SizeInBytes : 0;
    }

    private static double Product(ReadOnlySpan<long> values)
    {
        var product = 1.0;
        foreach (var value in values)
        {
            product *= Math.Max(0, value);
        }

        return product;
    }

    private static UInt128 ToCostFactor(double value)
    {
        if (!double.IsFinite(value) || value <= 0)
        {
            return 0;
        }

        return (UInt128)(ulong)Math.Ceiling(Math.Min(value, ulong.MaxValue));
    }

    private static UInt128 GetOtherCost(Cost cost)
    {
        UInt128 otherCost = 0;
        foreach (var (name, value) in cost.Factors)
        {
            if (name != CostFactorNames.CPUCycles
                && name != CostFactorNames.BlockLocalMemoryLoadBytes
                && name != CostFactorNames.BlockLocalMemoryStoreBytes
                && name != CostFactorNames.ChipGlobalMemoryLoadBytes
                && name != CostFactorNames.ChipGlobalMemoryStoreBytes
                && name != CostFactorNames.BlockSynchronization
                && name != CostFactorNames.GridSynchronization
                && name != CostFactorNames.Comm)
            {
                otherCost += value;
            }
        }

        return otherCost;
    }

    private static double ToDouble(UInt128 value)
    {
        return value > ulong.MaxValue ? ulong.MaxValue : (ulong)value;
    }

    private double EstimateElementwiseCycles(double work, params DataType[] dataTypes)
    {
        var vectorLanes = dataTypes.Select(GetVectorLaneCount).DefaultIfEmpty(1).Max();
        var elementsPerCycle = vectorLanes > 1
            ? _machine?.Compute.ElementwiseElementsPerCycle ?? vectorLanes
            : 1.0;
        return work / Math.Max(1.0, elementsPerCycle);
    }
}
