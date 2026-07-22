// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Reactive;
using Nncase.IR;
using Nncase.TIR;
using Nncase.TIR.Builders;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Constructs one selected producer/consumer loop directly while TileGraph is
/// lowered. The two phases and every staged allocation are explicit inputs;
/// this builder never classifies an already-built body.
/// </summary>
internal sealed class PipelineForBuilder : ISequentialBuilder<Expr>
{
    private readonly List<object> _consumeBody = new();
    private readonly List<object> _consumeTail = new();
    private readonly List<object> _produceBody = new();
    private readonly Dictionary<string, StagedBinding> _bindings = new(StringComparer.Ordinal);
    private readonly DimVar _loopVar;
    private readonly TIR.Range _domain;
    private readonly LoopMode _mode;
    private readonly PipelineRegionPlan _plan;
    private readonly PipelineRegionId _regionId;

    public PipelineForBuilder(
        DimVar loopVar,
        TIR.Range domain,
        LoopMode mode,
        PipelineRegionPlan plan,
        PipelineRegionId regionId)
    {
        ArgumentNullException.ThrowIfNull(loopVar);
        ArgumentNullException.ThrowIfNull(domain);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(regionId);
        if (mode is not (LoopMode.Serial or LoopMode.Reduction))
        {
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "A pipeline requires a serial or reduction loop.");
        }

        _loopVar = loopVar;
        _domain = domain;
        _mode = mode;
        _plan = plan;
        _regionId = regionId;
    }

    public ISequentialBuilder<Expr> Body(params object[] exprOrBuilders)
    {
        _consumeBody.AddRange(exprOrBuilders);
        return this;
    }

    public ISequentialBuilder<Expr> Tail(params object[] exprOrBuilders)
    {
        _consumeTail.AddRange(exprOrBuilders);
        return this;
    }

    public void Produce(params object[] exprOrBuilders) => _produceBody.AddRange(exprOrBuilders);

    public void Bind(
        PipelineBufferBindingDescriptor descriptor,
        IVar access,
        BaseExpr allocation,
        TIR.Buffer buffer)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        ArgumentNullException.ThrowIfNull(access);
        ArgumentNullException.ThrowIfNull(allocation);
        ArgumentNullException.ThrowIfNull(buffer);
        var planChannel = _plan.Channels.SingleOrDefault(channel =>
            string.Equals(channel.ChannelId, descriptor.ChannelId, StringComparison.Ordinal));
        if (planChannel is null ||
            planChannel.SourceMemorySpace != descriptor.SourceMemorySpace ||
            planChannel.DestinationMemorySpace != descriptor.DestinationMemorySpace)
        {
            throw new InvalidOperationException(
                $"Staged binding {descriptor.ChannelId} does not belong to pipeline {_plan.ScheduleId}.");
        }

        if (buffer.StagedLayout is not { } layout || layout.StageCount != _plan.StageCount)
        {
            throw new InvalidOperationException(
                $"Staged binding {descriptor.ChannelId} requires a {_plan.StageCount}-stage allocation.");
        }

        if (!ExprCollector.Collect(allocation).Any(expr => ReferenceEquals(expr, buffer)))
        {
            throw new InvalidOperationException(
                $"Staged binding {descriptor.ChannelId} allocation does not reference buffer {buffer.Name}.");
        }

        if (!_bindings.TryAdd(descriptor.ChannelId, new(descriptor, access, allocation, buffer)))
        {
            throw new InvalidOperationException(
                $"Pipeline {_plan.ScheduleId} binds channel {descriptor.ChannelId} more than once.");
        }
    }

    public Expr Build()
    {
        var missingChannels = _plan.Channels
            .Where(channel => !_bindings.ContainsKey(channel.ChannelId))
            .Select(channel => channel.ChannelId)
            .ToArray();
        if (missingChannels.Length != 0 || _bindings.Count != _plan.Channels.Length)
        {
            throw new InvalidOperationException(
                $"Pipeline {_plan.ScheduleId} has incomplete staged bindings; missing=[{string.Join(",", missingChannels)}].");
        }

        var producer = Sequential.Flatten(_produceBody.ToArray());
        var consumer = Sequential.Flatten(_consumeBody.Concat(_consumeTail).ToArray());
        if (producer.Count == 0 || consumer.Count == 0)
        {
            throw new InvalidOperationException(
                $"Pipeline {_plan.ScheduleId} requires non-empty producer and consumer bodies.");
        }

        var orderedBindings = _plan.Channels
            .Select(channel => _bindings[channel.ChannelId])
            .ToImmutableArray();
        Dimension logicalStage = (_loopVar - _domain.Start) / _domain.Step;
        return new PipelineFor(
            _loopVar,
            _domain,
            _mode,
            LoopPartition.Unpartitioned,
            BuildPhase("produce", producer, orderedBindings, logicalStage),
            BuildPhase("consume", consumer, orderedBindings, logicalStage),
            _plan,
            _regionId,
            orderedBindings.Select(binding => binding.Descriptor),
            orderedBindings.Select(binding => binding.Access),
            orderedBindings.Select(binding => binding.Allocation),
            orderedBindings.Select(binding => binding.Buffer));
    }

    private static Sequential BuildPhase(
        string phase,
        Sequential body,
        ImmutableArray<StagedBinding> bindings,
        Dimension logicalStage)
    {
        var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        var stageAliases = new List<(IVar Access, TIR.Buffer Buffer)>();
        foreach (var binding in bindings)
        {
            var layout = binding.Buffer.StagedLayout!;
            var stage = logicalStage % layout.StageCount;
            var stageAccess = binding.Access.With(name: $"{binding.Access.Name}_{phase}_stage");
            var stageBuffer = new TIR.Buffer(
                $"{binding.Buffer.Name}_{phase}_stage",
                binding.Buffer.ElemType,
                binding.Buffer.MemSpan.With(
                    start: binding.Buffer.MemSpan.Start + (stage * layout.StageStrideBytes),
                    size: layout.StagePhysicalBytes),
                binding.Buffer.Dimensions.ToArray(),
                binding.Buffer.Strides.ToArray(),
                binding.Buffer.DistributedType,
                binding.Buffer.StorageEncoding);
            replacements.Add((BaseExpr)binding.Access, (BaseExpr)stageAccess);
            stageAliases.Add((stageAccess, stageBuffer));
        }

        var cloner = new PhaseCloner(replacements);
        var fields = body.Fields
            .ToArray()
            .Select(field => (Expr)cloner.Clone(field, default))
            .ToArray();
        Sequential result = new(fields);
        for (var index = stageAliases.Count - 1; index >= 0; index--)
        {
            result = new Sequential(new Let(stageAliases[index].Access, stageAliases[index].Buffer, result));
        }

        return result;
    }

    private sealed class PhaseCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;

        public PhaseCloner(IReadOnlyDictionary<BaseExpr, BaseExpr> replacements)
        {
            _replacements = replacements;
            CloneUnmutated = false;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
            => _replacements.TryGetValue(expr, out var replacement)
                ? replacement
                : expr is IVar ? expr : base.DispatchVisit(expr, context);

        protected override BaseExpr VisitLeafBuffer(TIR.Buffer expr, Unit context) => expr;

        protected override BaseExpr VisitLeafPhysicalBuffer(PhysicalBuffer expr, Unit context) => expr;
    }

    private sealed record StagedBinding(
        PipelineBufferBindingDescriptor Descriptor,
        IVar Access,
        BaseExpr Allocation,
        TIR.Buffer Buffer);
}
