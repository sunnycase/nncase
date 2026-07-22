// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule;

/// <summary>
/// Converts a matrix primitive's latency and throughput contract into one
/// block-level execution service estimate.
/// </summary>
public static class MatrixComputeCostModel
{
    private const long TimingScale = 1000;

    /// <summary>
    /// Estimates cycles for accumulator chains distributed over a fixed block.
    /// Instructions in one chain are dependent along the reduction dimension;
    /// distinct chains may overlap subject to worker and block throughput.
    /// </summary>
    public static double EstimateCycles(
        MatrixComputePrimitiveSpec primitive,
        double accumulatorChains,
        double dependentInstructionsPerChain,
        BlockExecutionSpec execution)
    {
        ArgumentNullException.ThrowIfNull(primitive);
        ArgumentNullException.ThrowIfNull(execution);
        if (!double.IsFinite(accumulatorChains) || accumulatorChains < 0 ||
            !double.IsFinite(dependentInstructionsPerChain) || dependentInstructionsPerChain < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(accumulatorChains),
                "Matrix accumulator chains and dependent instruction counts must be finite and non-negative.");
        }

        if (accumulatorChains == 0 || dependentInstructionsPerChain == 0)
        {
            return 0;
        }

        var executionGroups = GetExecutionGroupCount(primitive, execution);
        var chainsPerGroup = Math.Max(1.0, Math.Ceiling(accumulatorChains / executionGroups));
        var instructionsPerGroup = chainsPerGroup * dependentInstructionsPerChain;
        var dynamicInstructions = instructionsPerGroup * executionGroups;
        var dependencyCycles = dependentInstructionsPerChain * primitive.DependencyLatencyCycles;
        var workerThroughputCycles = primitive.DependencyLatencyCycles
            + ((instructionsPerGroup - 1) * primitive.ReciprocalThroughputCyclesPerWorker);
        var blockThroughputCycles = primitive.DependencyLatencyCycles
            + ((dynamicInstructions - 1) / primitive.MaxInstructionsPerCyclePerBlock);
        return Math.Ceiling(Math.Max(dependencyCycles, Math.Max(workerThroughputCycles, blockThroughputCycles)));
    }

    /// <summary>
    /// Symbolic counterpart of <see cref="EstimateCycles(MatrixComputePrimitiveSpec,double,double,BlockExecutionSpec)"/>.
    /// </summary>
    public static IntExpr EstimateCycles(
        MatrixComputePrimitiveSpec primitive,
        IntExpr accumulatorChains,
        IntExpr dependentInstructionsPerChain,
        BlockExecutionSpec execution,
        Solver solver)
    {
        ArgumentNullException.ThrowIfNull(primitive);
        ArgumentNullException.ThrowIfNull(accumulatorChains);
        ArgumentNullException.ThrowIfNull(dependentInstructionsPerChain);
        ArgumentNullException.ThrowIfNull(execution);
        ArgumentNullException.ThrowIfNull(solver);
        if (accumulatorChains.Var().Min() < 0 || dependentInstructionsPerChain.Var().Min() < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(accumulatorChains),
                "Matrix accumulator chains and dependent instruction counts must be non-negative.");
        }

        var hasChains = solver.MakeIsGreaterCstVar(accumulatorChains, 0);
        var hasInstructions = solver.MakeIsGreaterCstVar(dependentInstructionsPerChain, 0);
        var hasWork = hasChains * hasInstructions;
        var executionGroups = GetExecutionGroupCount(primitive, execution);
        var safeChains = solver.MakeMax(accumulatorChains, solver.MakeIntConst(1));
        var safeInstructions = solver.MakeMax(dependentInstructionsPerChain, solver.MakeIntConst(1));
        var chainsPerGroup = CeilDiv(safeChains, executionGroups, solver);
        var instructionsPerGroup = chainsPerGroup * safeInstructions;
        var dynamicInstructions = instructionsPerGroup * executionGroups;

        var dependencyCycles = ScaleCycles(safeInstructions, primitive.DependencyLatencyCycles, solver);
        var workerThroughputCycles = ScaleCycles(
            solver.MakeIntConst(1),
            primitive.DependencyLatencyCycles,
            solver)
            + ScaleCycles(
                instructionsPerGroup - 1,
                primitive.ReciprocalThroughputCyclesPerWorker,
                solver);
        var blockThroughputCycles = ScaleCycles(
            solver.MakeIntConst(1),
            primitive.DependencyLatencyCycles,
            solver)
            + DivideByRate(
                dynamicInstructions - 1,
                primitive.MaxInstructionsPerCyclePerBlock,
                solver);
        return hasWork * solver.MakeMax(
            dependencyCycles,
            solver.MakeMax(workerThroughputCycles, blockThroughputCycles));
    }

    private static int GetExecutionGroupCount(
        MatrixComputePrimitiveSpec primitive,
        BlockExecutionSpec execution)
    {
        if (primitive.CooperativeWorkers <= 0 || execution.WorkersPerBlock % primitive.CooperativeWorkers != 0)
        {
            throw new InvalidOperationException(
                $"Matrix primitive {primitive.Name} requires {primitive.CooperativeWorkers} cooperative workers per instruction, " +
                $"which does not divide block worker count {execution.WorkersPerBlock}.");
        }

        return execution.WorkersPerBlock / primitive.CooperativeWorkers;
    }

    private static IntExpr ScaleCycles(IntExpr count, double cyclesPerUnit, Solver solver)
    {
        var scaledCycles = ScalePositive(cyclesPerUnit, nameof(cyclesPerUnit));
        return CeilDiv(count * scaledCycles, TimingScale, solver);
    }

    private static IntExpr DivideByRate(IntExpr count, double unitsPerCycle, Solver solver)
    {
        var scaledRate = ScalePositive(unitsPerCycle, nameof(unitsPerCycle));
        return CeilDiv(count * TimingScale, scaledRate, solver);
    }

    private static IntExpr CeilDiv(IntExpr value, long divisor, Solver solver)
    {
        if (divisor <= 0 || value.Var().Min() < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(divisor),
                divisor,
                "Symbolic matrix timing ceil-div requires a non-negative value and positive divisor.");
        }

        return solver.MakeDiv(value + divisor - 1, divisor);
    }

    private static long ScalePositive(double value, string parameterName)
    {
        if (!double.IsFinite(value) || value <= 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, value, "Matrix timing values must be finite and positive.");
        }

        return Math.Max(1L, checked((long)Math.Round(value * TimingScale)));
    }
}
