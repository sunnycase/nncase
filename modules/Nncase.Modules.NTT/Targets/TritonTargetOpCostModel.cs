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
public sealed class TritonTargetOpCostModel : ITargetOpCostModel
{
    private readonly TritonTargetCapability _capability;

    public TritonTargetOpCostModel(TritonTargetCapability capability)
    {
        _capability = capability;
    }

    public TritonTargetCapability Capability => _capability;

    public UInt128 GetLatency(Cost cost)
    {
        var cpuCycles = ToDouble(GetFactor(cost, CostFactorNames.CPUCycles));
        var memoryElements = ToDouble(GetFactor(cost, CostFactorNames.MemoryLoad) + GetFactor(cost, CostFactorNames.MemoryStore));
        var memoryCycles = memoryElements / Math.Max(1.0, _capability.EffectiveGlobalMemoryElementsPerCyclePerCta);
        var overlappedCycles = Math.Max(cpuCycles, memoryCycles);
        var latency = overlappedCycles
            + (ToDouble(GetFactor(cost, CostFactorNames.Synchronization)) * _capability.SynchronizationCycles)
            + ToDouble(GetFactor(cost, CostFactorNames.Comm))
            + ToDouble(GetOtherCost(cost));
        return ToCostFactor(latency);
    }

    public bool TryGetUnaryCost(UnaryOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxScalarElementCount(query.Output, out var elements))
        {
            cost = Cost.Zero;
            return false;
        }

        cost = ElementwiseCost(
            EstimateElementwiseComputeCycles(elements, workPerElement: 1.0),
            GetTensorScalarElementCount(query.Input),
            GetTensorScalarElementCount(query.Output));
        return true;
    }

    public bool TryGetBinaryCost(BinaryOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxScalarElementCount(query.Output, out var elements))
        {
            cost = Cost.Zero;
            return false;
        }

        cost = ElementwiseCost(
            EstimateElementwiseComputeCycles(elements, workPerElement: 1.0),
            GetTensorScalarElementCount(query.Lhs) + GetTensorScalarElementCount(query.Rhs),
            GetTensorScalarElementCount(query.Output));
        return true;
    }

    public bool TryGetElementwiseCost(ElementwiseOpCostQuery query, out Cost cost)
    {
        if (!TryGetMaxScalarElementCount(query.Output, out var elements))
        {
            cost = Cost.Zero;
            return false;
        }

        var loadElements = 0.0;
        foreach (var input in query.Inputs)
        {
            loadElements += GetTensorScalarElementCount(input);
        }

        cost = ElementwiseCost(
            EstimateElementwiseComputeCycles(elements, Math.Max(0.0, query.WorkPerElement)),
            loadElements,
            GetTensorScalarElementCount(query.Output));
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
            computeCycles + _capability.FixedOverheadCycles,
            GetTensorScalarElementCount(query.Lhs) + GetTensorScalarElementCount(query.Rhs),
            GetTensorScalarElementCount(query.Output));
        return true;
    }

    private double EstimateElementwiseComputeCycles(double elements, double workPerElement)
    {
        return ((elements * workPerElement) / Math.Max(1.0, _capability.ElementwiseElementsPerCyclePerCta)) + _capability.FixedOverheadCycles;
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
        if (useDotInstructions && CanUseDotInstruction(lhsDType, rhsDType, outputDType))
        {
            var candidates = new[] { _capability.Mma, _capability.Wgmma }
                .Where(instruction => instruction.IsSupported)
                .Select(instruction => EstimateDotInstructionCycles(instruction, m, n, k, batch))
                .Where(double.IsFinite)
                .ToArray();
            if (candidates.Length > 0)
            {
                return candidates.Min();
            }
        }

        return Math.Max(0, m) * Math.Max(0, n) * Math.Max(0, k) * batch;
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
        return workItems / Math.Max(1.0, _capability.SimtFmaPerCyclePerCta);
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

    private double EstimateDotInstructionCycles(TritonDotInstructionCapability instruction, long m, long n, long k, double batch)
    {
        if (instruction.M <= 0 || instruction.N <= 0 || instruction.K <= 0 || instruction.InstructionsPerCyclePerCta <= 0)
        {
            return double.PositiveInfinity;
        }

        var mTiles = CeilDiv(m, instruction.M);
        var nTiles = CeilDiv(n, instruction.N);
        var kTiles = CeilDiv(k, instruction.K);
        var ctaTiles = Math.Max(1.0, mTiles * nTiles * batch);
        return ctaTiles * kTiles / instruction.InstructionsPerCyclePerCta;
    }

    private bool CanUseDotInstruction(DataType lhsDType, DataType rhsDType, DataType outputDType)
    {
        lhsDType = GetScalarDType(lhsDType);
        rhsDType = GetScalarDType(rhsDType);
        outputDType = GetScalarDType(outputDType);

        if (lhsDType != rhsDType)
        {
            return false;
        }

        return lhsDType == DataTypes.Float16
            || lhsDType == DataTypes.BFloat16
            || lhsDType == DataTypes.Float8E4M3
            || lhsDType == DataTypes.Float8E5M2
            || lhsDType == DataTypes.Int8
            || (lhsDType == DataTypes.Float32 && outputDType == DataTypes.Float32 && _capability.UseTensorCoresForFloat32);
    }

    private DataType GetScalarDType(DataType dtype) => dtype switch
    {
        VectorType vectorType => GetScalarDType(vectorType.ElemType),
        _ => dtype,
    };

    private Cost ElementwiseCost(double cpuCycles, double memoryLoadElements, double memoryStoreElements)
    {
        var cost = new Cost();
        AddCostFactor(cost, CostFactorNames.CPUCycles, cpuCycles);
        AddCostFactor(cost, CostFactorNames.MemoryLoad, memoryLoadElements);
        AddCostFactor(cost, CostFactorNames.MemoryStore, memoryStoreElements);
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
                && name != CostFactorNames.MemoryLoad
                && name != CostFactorNames.MemoryStore
                && name != CostFactorNames.Synchronization
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
