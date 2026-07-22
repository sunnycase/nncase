// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule;

/// <summary>
/// Cost of one emitted block-microkernel call. Materialization, transfer,
/// synchronization, and loop overlap are owned by AutoTiling's loop schedule.
/// </summary>
public sealed record BlockMicroKernelExecutionCost(IntExpr RegionCycles)
{
    public static BlockMicroKernelExecutionCost ComputeOnly(IntExpr cycles) => new(cycles);
}

/// <summary>
/// Symbolic fill/steady/drain estimate for one lexical loop schedule.
/// </summary>
public sealed class LoopPipelineScheduleEstimate
{
    private LoopPipelineScheduleEstimate(
        IntExpr iterationCount,
        IntExpr invocationCount,
        IntExpr producerCycles,
        IntExpr consumerCycles,
        IntExpr initiationIntervalCycles,
        IntExpr serialRegionCycles,
        IntExpr pipelinedRegionCycles)
    {
        IterationCount = iterationCount;
        InvocationCount = invocationCount;
        ProducerCycles = producerCycles;
        ConsumerCycles = consumerCycles;
        InitiationIntervalCycles = initiationIntervalCycles;
        SerialRegionCycles = serialRegionCycles;
        PipelinedRegionCycles = pipelinedRegionCycles;
    }

    public IntExpr IterationCount { get; }

    public IntExpr InvocationCount { get; }

    public IntExpr ProducerCycles { get; }

    public IntExpr ConsumerCycles { get; }

    public IntExpr InitiationIntervalCycles { get; }

    public IntExpr SerialRegionCycles { get; }

    public IntExpr PipelinedRegionCycles { get; }

    /// <summary>
    /// Builds the two alternatives from the same producer and consumer
    /// services. Stage one is serial by definition; stage two alone overlaps
    /// the steady-state services.
    /// </summary>
    public static LoopPipelineScheduleEstimate Create(
        Solver solver,
        IntExpr iterationCount,
        IntExpr invocationCount,
        IntExpr producerCycles,
        IntExpr consumerCycles,
        IntExpr producerCommitCycles,
        IntExpr consumerWaitAcquireCycles,
        IntExpr consumerReleaseCycles)
    {
        ArgumentNullException.ThrowIfNull(solver);
        ValidatePositive(iterationCount, nameof(iterationCount));
        ValidatePositive(invocationCount, nameof(invocationCount));
        ValidateNonNegative(producerCycles, nameof(producerCycles));
        ValidateNonNegative(consumerCycles, nameof(consumerCycles));
        ValidateNonNegative(producerCommitCycles, nameof(producerCommitCycles));
        ValidateNonNegative(consumerWaitAcquireCycles, nameof(consumerWaitAcquireCycles));
        ValidateNonNegative(consumerReleaseCycles, nameof(consumerReleaseCycles));

        var serialPerIteration = producerCycles + consumerCycles;
        var serialRegionCycles = invocationCount * iterationCount * serialPerIteration;
        var producerService = producerCycles + producerCommitCycles;
        var consumerService = consumerWaitAcquireCycles + consumerCycles + consumerReleaseCycles;
        var initiationInterval = solver.MakeMax(producerService, consumerService);
        var steadyIterations = solver.MakeMax(solver.MakeIntConst(0), iterationCount - 1);
        var pipelinedPerInvocation = producerService
            + (steadyIterations * initiationInterval)
            + consumerService;
        var pipelinedRegionCycles = invocationCount * pipelinedPerInvocation;
        return new(
            iterationCount,
            invocationCount,
            producerCycles,
            consumerCycles,
            initiationInterval,
            serialRegionCycles,
            pipelinedRegionCycles);
    }

    private static void ValidatePositive(IntExpr expression, string name)
    {
        ArgumentNullException.ThrowIfNull(expression);
        if (expression.Var().Min() < 1)
        {
            throw new ArgumentOutOfRangeException(name, $"{name} must be positive.");
        }
    }

    private static void ValidateNonNegative(IntExpr expression, string name)
    {
        ArgumentNullException.ThrowIfNull(expression);
        if (expression.Var().Min() < 0)
        {
            throw new ArgumentOutOfRangeException(name, $"{name} must not be negative.");
        }
    }
}

public sealed record SelectedLoopPipelineScheduleEstimate(
    long IterationCount,
    long InvocationCount,
    long ProducerCycles,
    long ConsumerCycles,
    long InitiationIntervalCycles,
    long SerialRegionCycles,
    long PipelinedRegionCycles);
