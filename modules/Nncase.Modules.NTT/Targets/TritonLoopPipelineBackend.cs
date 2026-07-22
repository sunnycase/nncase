// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Schedule;

namespace Nncase.Targets;

/// <summary>
/// Triton lowering contract for loop-owned global-to-shared pipelines.
/// </summary>
public sealed class TritonLoopPipelineBackend : ILoopPipelineBackend
{
    public const int DoubleBufferStageCount = 2;

    public static readonly PipelineTemplateId CpAsyncN2TemplateId = new("triton.loop.cp_async.n2.v1");

    public static readonly PipelineSynchronizationProtocol CpAsyncN2Synchronization = new(
        asynchronousProduce: true,
        requiresProducerCommit: true,
        requiresConsumerWait: true,
        waitProvidesConsumerAcquire: false,
        requiresConsumerRelease: true);

    private TritonLoopPipelineBackend()
    {
    }

    public static TritonLoopPipelineBackend Instance { get; } = new();

    public bool SupportsStageCount(int stageCount) => stageCount == DoubleBufferStageCount;

    public IntExpr GetChannelLegality(LoopPipelineChannelModelContext context, int stageCount)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (!SupportsStageCount(stageCount))
        {
            return context.Solver.MakeIntConst(0);
        }

        var source = context.SourceMemorySpace;
        var destination = context.DestinationMemorySpace;
        var transfer = context.Machine.GetTransfer(source.Id, destination.Id);
        var fixedContractIsLegal =
            context.Machine.GetMemoryResource(source).Kind == TargetMemorySpaceKind.Global &&
            context.Machine.GetMemoryResource(destination).Kind == TargetMemorySpaceKind.Shared &&
            destination.IsTilingCandidate &&
            destination.IsAddressable &&
            destination.SupportsDynamicIndexing &&
            destination.RequiresExplicitSynchronization &&
            context.Machine.GetTilingParentMemorySpace(destination.TilingLevel).Id == source.Id &&
            transfer.Mode == TargetMemoryTransferMode.ExplicitCopy &&
            transfer.Asynchronous?.SupportsStageCount(stageCount) == true &&
            GetScalarDataType(context.DataType) is PrimType;
        if (!fixedContractIsLegal)
        {
            return context.Solver.MakeIntConst(0);
        }

        if (context.LocalShape.IsDefaultOrEmpty ||
            context.LocalShape.Length != context.FullShape.Length ||
            context.LocalShape.Any(extent => extent.Var().Min() < 1) ||
            context.FullShape.Any(extent => extent.Var().Min() < 1))
        {
            return context.Solver.MakeIntConst(0);
        }

        // tle.gpu.copy consumes one compact rectangular source region. An
        // axis may be tiled only when all physically inner axes span their
        // complete backing region; singleton outer axes remain legal.
        IntExpr legality = context.Solver.MakeIntConst(1);
        IntExpr localInnerSpan = context.Solver.MakeIntConst(1);
        IntExpr fullInnerSpan = context.Solver.MakeIntConst(1);
        for (var axis = context.LocalShape.Length - 1; axis >= 0; axis--)
        {
            var singleton = context.Solver.MakeIsEqualCstVar(context.LocalShape[axis], 1);
            var compactInner = context.Solver.MakeIsEqualVar(localInnerSpan, fullInnerSpan);
            legality *= context.Solver.MakeMax(singleton, compactInner);
            localInnerSpan *= context.LocalShape[axis];
            fullInnerSpan *= context.FullShape[axis];
        }

        var result = legality.Var();
        result.SetName("triton_loop_pipeline_channel_legal");
        result.SetRange(0, 1);
        return result;
    }

    public PipelineTemplateDescriptor GetTemplate(int stageCount, TargetMachineModel machine)
    {
        ArgumentNullException.ThrowIfNull(machine);
        if (!SupportsStageCount(stageCount))
        {
            throw new NotSupportedException(
                $"Triton does not implement a {stageCount}-stage loop pipeline.");
        }

        return new(CpAsyncN2TemplateId, CpAsyncN2Synchronization);
    }

    private static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vector ? GetScalarDataType(vector.ElemType) : dataType;
}
