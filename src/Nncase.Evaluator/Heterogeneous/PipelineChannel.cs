// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Heterogeneous;

namespace Nncase.Evaluator.Heterogeneous;

public sealed partial class CreatePipelineChannelEvaluator :
    ITypeInferencer<CreatePipelineChannel>,
    ICostEvaluator<CreatePipelineChannel>
{
    public IRType Visit(ITypeInferenceContext context, CreatePipelineChannel target)
    {
        if (string.IsNullOrWhiteSpace(target.ChannelId))
        {
            return new InvalidType("Pipeline channel ID must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(target.ProducerModuleKind) ||
            string.IsNullOrWhiteSpace(target.ConsumerModuleKind))
        {
            return new InvalidType("Pipeline channel module kinds must not be empty.");
        }

        if (target.ProducerModuleKind == target.ConsumerModuleKind)
        {
            return new InvalidType("Pipeline channel endpoints must belong to different modules.");
        }

        if (target.Capacity <= 0)
        {
            return new InvalidType($"Pipeline channel capacity must be positive, got {target.Capacity}.");
        }

        return TensorType.Scalar(new ReferenceType(new PipelineChannelType()));
    }

    public Cost Visit(ICostEvaluateContext context, CreatePipelineChannel target) => Cost.Zero;
}

public sealed partial class ProduceEvaluator : ITypeInferencer<Produce>, ICostEvaluator<Produce>
{
    public IRType Visit(ITypeInferenceContext context, Produce target)
    {
        var channelType = context.GetArgumentType(target, Produce.Channel);
        if (!IsChannel(channelType))
        {
            return new InvalidType($"Produce expects a PipelineChannel reference, got {channelType}.");
        }

        var dependencyType = context.GetArgumentType(target, Produce.Dependency);
        if (dependencyType is not NoneType)
        {
            return new InvalidType($"Produce dependency must be None, got {dependencyType}.");
        }

        return NoneType.Default;
    }

    public Cost Visit(ICostEvaluateContext context, Produce target) => Cost.Zero;

    internal static bool IsChannel(IRType type)
        => type is TensorType { IsScalar: true, DType: ReferenceType { ElemType: PipelineChannelType } };
}

public sealed partial class ConsumeEvaluator : ITypeInferencer<Consume>, ICostEvaluator<Consume>
{
    public IRType Visit(ITypeInferenceContext context, Consume target)
    {
        var channelType = context.GetArgumentType(target, Consume.Channel);
        if (!ProduceEvaluator.IsChannel(channelType))
        {
            return new InvalidType($"Consume expects a PipelineChannel reference, got {channelType}.");
        }

        var dependencyType = context.GetArgumentType(target, Consume.Dependency);
        return dependencyType is NoneType
            ? target.PayloadType
            : new InvalidType($"Consume dependency must be None, got {dependencyType}.");
    }

    public Cost Visit(ICostEvaluateContext context, Consume target)
    {
        var outputType = context.GetReturnType<IRType>();
        var bytes = CostUtility.GetMemoryAccess(outputType);
        return new Cost
        {
            [CostFactorNames.BlockLocalMemoryLoadBytes] = bytes,
            [CostFactorNames.BlockLocalMemoryStoreBytes] = bytes,
        };
    }
}

public sealed partial class PipelineTokenEvaluator :
    ITypeInferencer<PipelineToken>,
    ICostEvaluator<PipelineToken>
{
    public IRType Visit(ITypeInferenceContext context, PipelineToken target) => NoneType.Default;

    public Cost Visit(ICostEvaluateContext context, PipelineToken target) => Cost.Zero;
}

public sealed partial class PipelineYieldEvaluator :
    ITypeInferencer<PipelineYield>,
    ICostEvaluator<PipelineYield>
{
    public IRType Visit(ITypeInferenceContext context, PipelineYield target)
    {
        var dependencyType = context.GetArgumentType(target, PipelineYield.Dependency);
        return dependencyType is NoneType
            ? context.GetArgumentType(target, PipelineYield.Value)
            : new InvalidType($"PipelineYield dependency must be None, got {dependencyType}.");
    }

    public Cost Visit(ICostEvaluateContext context, PipelineYield target) => Cost.Zero;
}

public sealed partial class PipelineLaunchEvaluator :
    ITypeInferencer<PipelineLaunch>,
    ICostEvaluator<PipelineLaunch>
{
    public IRType Visit(ITypeInferenceContext context, PipelineLaunch target)
    {
        var workers = context.GetArgumentType(target, PipelineLaunch.Workers);
        if (workers is not TupleType tuple)
        {
            return new InvalidType($"PipelineLaunch workers must be a tuple, got {workers}.");
        }

        return target.ResultWorkerIndex >= 0 && target.ResultWorkerIndex < tuple.Count
            ? tuple[target.ResultWorkerIndex]
            : new InvalidType(
                $"PipelineLaunch result worker index {target.ResultWorkerIndex} is outside [0, {tuple.Count}).");
    }

    public Cost Visit(ICostEvaluateContext context, PipelineLaunch target) => Cost.Zero;
}
