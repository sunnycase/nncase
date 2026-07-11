// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;

namespace Nncase.Targets;

/// <summary>
/// Triton backend op cost model.
/// </summary>
public sealed class TritonTargetOpCostModel : ITargetOpCostModel, IHierarchicalTargetOpCostModel, ITargetOpCostBreakdownModel
{
    private readonly TargetMachineModel _machine;
    private readonly TargetMemorySpaceSpec _blockMemory;
    private readonly TargetMemorySpaceSpec _rootMemory;

    public TritonTargetOpCostModel(TargetMachineModel machine)
    {
        _machine = machine ?? throw new ArgumentNullException(nameof(machine));
        if (machine.Execution.Kind != BlockExecutionKind.PersistentGpuBlock)
        {
            throw new ArgumentException($"Triton cost model requires a persistent GPU target machine, got {machine.Id} ({machine.Execution.Kind}).", nameof(machine));
        }

        _blockMemory = machine.TilingMemorySpaces[^1];
        _rootMemory = machine.GetMemorySpace(machine.RootMemorySpace);
    }

    public UInt128 GetLatency(Cost cost) => GetLatency(cost, TargetCostAggregationContext.Local);

    public UInt128 GetLatency(Cost cost, TargetCostAggregationContext context) => GetLatencyBreakdown(cost, context).Latency;

    public TargetCostLatencyBreakdown GetLatencyBreakdown(Cost cost, TargetCostAggregationContext context)
    {
        var cpuCycles = ToDouble(GetFactor(cost, CostFactorNames.CPUCycles));
        var blockLocalLoadBytes = ToDouble(GetFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes));
        var blockLocalStoreBytes = ToDouble(GetFactor(cost, CostFactorNames.BlockLocalMemoryStoreBytes));
        var chipGlobalLoadBytes = ToDouble(GetFactor(cost, CostFactorNames.ChipGlobalMemoryLoadBytes));
        var chipGlobalStoreBytes = ToDouble(GetFactor(cost, CostFactorNames.ChipGlobalMemoryStoreBytes));
        var activeBlocks = Math.Max(1.0, context.ActiveBlockCount);
        var blockLocalMemoryCycles = (blockLocalLoadBytes / _blockMemory.ReadBytesPerCycle)
            + (blockLocalStoreBytes / _blockMemory.WriteBytesPerCycle)
            + GetMemoryLatency(blockLocalLoadBytes + blockLocalStoreBytes, _blockMemory);
        var chipGlobalMemoryCycles = (((blockLocalLoadBytes + chipGlobalLoadBytes) * activeBlocks) / _rootMemory.ReadBytesPerCycle)
            + (((blockLocalStoreBytes + chipGlobalStoreBytes) * activeBlocks) / _rootMemory.WriteBytesPerCycle)
            + GetMemoryLatency(blockLocalLoadBytes + blockLocalStoreBytes + chipGlobalLoadBytes + chipGlobalStoreBytes, _rootMemory);
        var overlappedCycles = Math.Max(cpuCycles, Math.Max(blockLocalMemoryCycles, chipGlobalMemoryCycles));
        var blockSynchronizationCycles = ToDouble(GetFactor(cost, CostFactorNames.BlockSynchronization)) * _machine.Synchronization.BlockCycles;
        var gridSynchronizationCycles = ToDouble(GetFactor(cost, CostFactorNames.GridSynchronization)) * _machine.Synchronization.GridCycles;
        var commCycles = ToDouble(GetFactor(cost, CostFactorNames.Comm));
        var otherCycles = ToDouble(GetOtherCost(cost));
        var latency = overlappedCycles
            + blockSynchronizationCycles
            + gridSynchronizationCycles
            + commCycles
            + otherCycles;
        return new TargetCostLatencyBreakdown(
            Math.Max(1L, context.ActiveBlockCount),
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
            EstimateElementwiseComputeCycles(elements, CostUtility.GetCPUCyclesOfUnary(query.Op)),
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
            EstimateElementwiseComputeCycles(elements, CostUtility.GetCPUCyclesOfBinary(query.Op)),
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

        var loadElements = 0.0;
        foreach (var input in query.Inputs)
        {
            loadElements += GetTensorByteCount(input);
        }

        cost = ElementwiseCost(
            EstimateElementwiseComputeCycles(elements, Math.Max(0.0, query.WorkPerElement)),
            loadElements,
            GetTensorByteCount(query.Output));
        return true;
    }

    public bool TryGetMatMulCost(MatMulOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxShape(query.Output, out var outputShape)
            || !TryGetMaxShape(query.Lhs, out var lhsShape)
            || !TryGetMaxShape(query.Rhs, out var rhsShape)
            || outputShape.Length < 2
            || lhsShape.Length < 2
            || rhsShape.Length < 2)
        {
            cost = Cost.Zero;
            return false;
        }

        var (m, n, k) = GetMatMulLogicalShape(query, lhsShape, outputShape);
        var batch = Product(outputShape.AsSpan(0, outputShape.Length - 2));
        var outputDType = query.OutputDataType ?? query.Output.DType;
        var useDotInstructions = HasVectorDType(query.Lhs.DType)
            || HasVectorDType(query.Rhs.DType)
            || HasVectorDType(query.Output.DType)
            || HasVectorDType(outputDType);
        var computeCycles = query.Kind switch
        {
            MatMulOpCostKind.Simt => EstimateSimtMatMulComputeCycles(m, n, k, batch),
            _ => EstimateMatMulComputeCycles(m, n, k, batch, query.Lhs.DType, query.Rhs.DType, outputDType, useDotInstructions),
        };

        cost = ElementwiseCost(
            computeCycles,
            GetTensorByteCount(query.Lhs) + GetTensorByteCount(query.Rhs),
            GetTensorByteCount(query.Output));
        return true;
    }

    private double EstimateElementwiseComputeCycles(double elements, double workPerElement)
    {
        return (elements * workPerElement) / Math.Max(1.0, _machine.Compute.ElementwiseElementsPerCycle);
    }

    private (long M, long N, long K) GetMatMulLogicalShape(MatMulOpCostQuery query, long[] lhsShape, long[] outputShape)
    {
        var m = outputShape[^2];
        var n = outputShape[^1];
        var k = lhsShape[^1];
        var lhsLanes = GetVectorLaneCount(query.Lhs.DType);
        var rhsLanes = GetVectorLaneCount(query.Rhs.DType);
        var outputLanes = GetVectorLaneCount(query.Output.DType);
        var reductionLanes = EstimateReductionLaneCount(lhsLanes, rhsLanes, outputLanes);
        var lhsNonReductionLanes = lhsLanes / reductionLanes;
        var rhsNonReductionLanes = rhsLanes / reductionLanes;

        m = MultiplyDimension(m, lhsNonReductionLanes);
        n = MultiplyDimension(n, rhsNonReductionLanes);
        k = MultiplyDimension(k, reductionLanes);
        return (m, n, k);
    }

    private double EstimateMatMulComputeCycles(long m, long n, long k, double batch, DataType lhsDType, DataType rhsDType, DataType outputDType, bool useDotInstructions)
    {
        var simtCycles = EstimateSimtMatMulComputeCycles(m, n, k, batch);
        if (useDotInstructions)
        {
            var candidates = _machine.Compute.MatrixPrimitives
                .Where(instruction => instruction.Supports(lhsDType, rhsDType))
                .Select(instruction => EstimateDotInstructionCycles(instruction, m, n, k, batch))
                .Where(double.IsFinite)
                .ToArray();
            if (candidates.Length > 0)
            {
                return Math.Min(simtCycles, candidates.Min());
            }
        }

        return simtCycles;
    }

    private double EstimateSimtMatMulComputeCycles(long m, long n, long k, double batch)
    {
        double paddedM;
        double paddedN;
        double paddedK;
        if (m <= 1)
        {
            paddedM = Math.Max(0, m);
            paddedN = CeilDiv(Math.Max(0, n), 32) * 32;
            paddedK = CeilDiv(Math.Max(0, k), 256) * 256;
        }
        else
        {
            paddedM = CeilDiv(Math.Max(0, m), 16) * 16;
            paddedN = CeilDiv(Math.Max(0, n), 64) * 64;
            paddedK = CeilDiv(Math.Max(0, k), 64) * 64;
        }

        var workItems = paddedM * paddedN * paddedK * batch;
        return workItems / Math.Max(1.0, _machine.Compute.SimtFmaPerCycle);
    }

    private long EstimateReductionLaneCount(long lhsLanes, long rhsLanes, long outputLanes)
    {
        if (lhsLanes <= 1 || rhsLanes <= 1)
        {
            return 1;
        }

        outputLanes = Math.Max(1, outputLanes);
        var product = lhsLanes * rhsLanes;
        if (product % outputLanes != 0)
        {
            return 1;
        }

        var quotient = product / outputLanes;
        var root = (long)Math.Round(Math.Sqrt(quotient));
        return root > 1 && (root * root) == quotient && (lhsLanes % root) == 0 && (rhsLanes % root) == 0
            ? root
            : 1;
    }

    private long MultiplyDimension(long dimension, long lanes)
    {
        lanes = Math.Max(1, lanes);
        if (dimension <= 0)
        {
            return 0;
        }

        return dimension > (long.MaxValue / lanes) ? long.MaxValue : dimension * lanes;
    }

    private double EstimateDotInstructionCycles(MatrixComputePrimitiveSpec instruction, long m, long n, long k, double batch)
    {
        if (instruction.M <= 0 || instruction.N <= 0 || instruction.K <= 0 || instruction.InstructionsPerCyclePerBlock <= 0)
        {
            return double.PositiveInfinity;
        }

        var mTiles = CeilDiv(m, instruction.M);
        var nTiles = CeilDiv(n, instruction.N);
        var kTiles = CeilDiv(k, instruction.K);
        var blockTiles = Math.Max(1.0, mTiles * nTiles * batch);
        return blockTiles * kTiles / instruction.InstructionsPerCyclePerBlock;
    }

    private static double GetMemoryLatency(double bytes, TargetMemorySpaceSpec memorySpace)
        => bytes > 0 ? memorySpace.LatencyCycles : 0;

    private DataType GetScalarDType(DataType dtype) => dtype switch
    {
        VectorType vectorType => GetScalarDType(vectorType.ElemType),
        _ => dtype,
    };

    private Cost ElementwiseCost(double cpuCycles, double blockLocalMemoryLoadBytes, double blockLocalMemoryStoreBytes)
    {
        var cost = new Cost();
        AddCostFactor(cost, CostFactorNames.CPUCycles, cpuCycles);
        AddCostFactor(cost, CostFactorNames.BlockLocalMemoryLoadBytes, blockLocalMemoryLoadBytes);
        AddCostFactor(cost, CostFactorNames.BlockLocalMemoryStoreBytes, blockLocalMemoryStoreBytes);
        return cost;
    }

    private void AddCostFactor(Cost cost, string name, double value)
    {
        var factor = ToCostFactor(value);
        if (factor > 0)
        {
            cost[name] = factor;
        }
    }

    private UInt128 ToCostFactor(double value)
    {
        if (!double.IsFinite(value) || value <= 0)
        {
            return 0;
        }

        return (UInt128)(ulong)Math.Ceiling(Math.Min(value, ulong.MaxValue));
    }

    private bool TryGetMaxShape(TargetCostTensor tensor, out long[] shape)
    {
        if (tensor.Shape is RankedShape && CompilerServices.TryGetMaxShape(tensor.Shape, out var maxShape))
        {
            shape = maxShape;
            return true;
        }

        shape = Array.Empty<long>();
        return false;
    }

    private bool TryGetMaxVectorElementCount(TargetCostTensor tensor, out double elements)
    {
        if (TryGetMaxShape(tensor, out var shape))
        {
            elements = Product(shape) * GetVectorLaneCount(tensor.DType);
            return true;
        }

        elements = 0;
        return false;
    }

    private bool TryGetMaxScalarElementCount(TargetCostTensor tensor, out double elements)
    {
        if (TryGetMaxShape(tensor, out var shape))
        {
            elements = Product(shape) * GetVectorLaneCount(tensor.DType);
            return true;
        }

        elements = 0;
        return false;
    }

    private double GetTensorScalarElementCount(TargetCostTensor tensor)
    {
        return TryGetMaxScalarElementCount(tensor, out var elements) ? elements : 0;
    }

    private double GetTensorByteCount(TargetCostTensor tensor)
    {
        return GetTensorScalarElementCount(tensor) * GetScalarDType(tensor.DType).SizeInBytes;
    }

    private bool HasVectorDType(DataType dtype) => dtype switch
    {
        VectorType => true,
        _ => false,
    };

    private long GetVectorLaneCount(DataType dtype) => dtype switch
    {
        VectorType vectorType => vectorType.Lanes.Aggregate(1, static (acc, lane) => acc * lane) * GetVectorLaneCount(vectorType.ElemType),
        _ => 1,
    };

    private UInt128 GetFactor(Cost cost, string name)
    {
        return cost.Factors.TryGetValue(name, out var value) ? value : 0;
    }

    private UInt128 GetOtherCost(Cost cost)
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

    private double ToDouble(UInt128 value)
    {
        return value > ulong.MaxValue ? ulong.MaxValue : (ulong)value;
    }

    private double Product(ReadOnlySpan<long> values)
    {
        var product = 1.0;
        foreach (var value in values)
        {
            product *= Math.Max(0, value);
        }

        return product;
    }

    private double CeilDiv(long lhs, int rhs)
    {
        return Math.Ceiling(Math.Max(0, lhs) / (double)rhs);
    }
}
