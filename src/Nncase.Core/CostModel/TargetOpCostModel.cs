// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
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
/// Optional target options extension for registering an op cost model.
/// </summary>
public interface ITargetOpCostModelProvider
{
    ITargetOpCostModel TargetCostModel { get; }
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
        return targetCostModel.GetLatency(cost);
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
public sealed class DefaultTargetOpCostModel : ITargetOpCostModel
{
    public static readonly DefaultTargetOpCostModel Instance = new();

    private DefaultTargetOpCostModel()
    {
    }

    public UInt128 GetLatency(Cost cost)
    {
        var cpuCycles = GetFactor(cost, CostFactorNames.CPUCycles);
        var memoryCycles = GetFactor(cost, CostFactorNames.MemoryLoad) + GetFactor(cost, CostFactorNames.MemoryStore);
        var overlappedCycles = cpuCycles > memoryCycles ? cpuCycles : memoryCycles;
        return overlappedCycles
            + (GetFactor(cost, CostFactorNames.Synchronization) * (UInt128)25_000)
            + GetFactor(cost, CostFactorNames.Comm)
            + GetOtherCost(cost);
    }

    public bool TryGetUnaryCost(UnaryOpCostQuery query, out Cost cost)
    {
        cost = Cost.Zero;
        return false;
    }

    public bool TryGetBinaryCost(BinaryOpCostQuery query, out Cost cost)
    {
        cost = Cost.Zero;
        return false;
    }

    public bool TryGetElementwiseCost(ElementwiseOpCostQuery query, out Cost cost)
    {
        cost = Cost.Zero;
        return false;
    }

    public bool TryGetMatMulCost(MatMulOpCostQuery query, out Cost cost)
    {
        cost = Cost.Zero;
        return false;
    }

    private static UInt128 GetFactor(Cost cost, string name)
    {
        return cost.Factors.TryGetValue(name, out var value) ? value : 0;
    }

    private static UInt128 GetOtherCost(Cost cost)
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
}
