// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Splits selected transfer-pipelined kernels into explicit producer and
/// consumer tasks after Bufferize has fixed Shared allocation ranges.
/// </summary>
public sealed class LowerTransferPipelineRegionsPass : ModulePass
{
    private readonly string _moduleKind;

    public LowerTransferPipelineRegionsPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var functions = input.Functions
            .Select((function, index) => (Function: function as PrimFunction, Index: index))
            .Where(item => item.Function?.ModuleKind == _moduleKind)
            .Select(item => (Function: item.Function!, item.Index))
            .ToArray();
        var pipelineFunctions = FindPipelineFunctions(functions.Select(item => item.Function));
        if (pipelineFunctions.Count == 0)
        {
            return Task.FromResult(input);
        }

        var effectAnalyzer = new MemoryEffectAnalyzer(
            functions.Select(item => item.Function));
        effectAnalyzer.AnalyzeAll();
        var replacements = new Dictionary<PrimFunction, PrimFunction>(
            ReferenceEqualityComparer.Instance);

        foreach (var (function, _) in functions)
        {
            if (!pipelineFunctions.Contains(function))
            {
                continue;
            }

            if (ContainsRegion(function.Body))
            {
                throw new InvalidOperationException(
                    $"PrimFunction {function.Name} already contains a producer/consumer region.");
            }

            var loweredBody = LowerFunctionBody(
                function,
                pipelineFunctions,
                effectAnalyzer);
            var lowered = function.With(body: loweredBody);
            lowered.Metadata = function.Metadata.Clone();
            replacements.Add(function, lowered);
        }

        if (replacements.Count == 0)
        {
            return Task.FromResult(input);
        }

        var targetRewriter = new FunctionTargetRewriter(replacements);
        foreach (var (function, index) in functions)
        {
            var replacement = replacements.TryGetValue(function, out var lowered)
                ? lowered
                : function;
            replacement = (PrimFunction)targetRewriter.Rewrite(replacement);
            if (!CompilerServices.InferenceType(replacement))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after lowering transfer pipelines in {function.Name}.");
            }

            input.Replace(index, replacement);
        }

        return Task.FromResult(input);
    }

    private static HashSet<PrimFunction> FindPipelineFunctions(
        IEnumerable<PrimFunction> functions)
    {
        var allFunctions = functions.ToArray();
        var result = new HashSet<PrimFunction>(ReferenceEqualityComparer.Instance);
        foreach (var function in allFunctions)
        {
            if (EnumerateCalls(function.Body).Any(IsTransferPipelineCall))
            {
                result.Add(function);
            }
        }

        var changed = true;
        while (changed)
        {
            changed = false;
            foreach (var function in allFunctions)
            {
                if (!result.Contains(function) &&
                    EnumerateCalls(function.Body)
                        .Any(call => call.Target is PrimFunction callee && result.Contains(callee)))
                {
                    result.Add(function);
                    changed = true;
                }
            }
        }

        return result;
    }

    private static Sequential LowerFunctionBody(
        PrimFunction function,
        IReadOnlySet<PrimFunction> pipelineFunctions,
        MemoryEffectAnalyzer effectAnalyzer)
    {
        ValidateStructuredPipelinePlacement(
            function.Body,
            pipelineFunctions,
            function.Name);
        var stageIndex = 0;
        var consumer = WrapPipelineStages(
            function.Body,
            pipelineFunctions,
            function.Name,
            ref stageIndex);
        return new Sequential(BuildPipelineRegion(
            consumer,
            function.Name,
            effectAnalyzer));
    }

    private static ProducerConsumerRegion BuildPipelineRegion(
        Sequential consumer,
        string functionName,
        MemoryEffectAnalyzer effectAnalyzer)
    {
        var synchronization = BuildSynchronizationPlan(
            consumer,
            functionName,
            effectAnalyzer);
        var consumeBody = InsertConsumerSynchronization(consumer, synchronization);
        var produceBody = BuildProducerBody(consumer, synchronization);
        return new ProducerConsumerRegion(produceBody, consumeBody);
    }

    private static PipelineSynchronizationPlan BuildSynchronizationPlan(
        Sequential consumer,
        string functionName,
        MemoryEffectAnalyzer effectAnalyzer)
    {
        var owners = EnumerateSharedOwners(
            consumer,
            functionName,
            effectAnalyzer).ToArray();
        var boundaries = owners
            .SelectMany(owner => owner.Ranges)
            .SelectMany(range => new[] { range.Start, range.End })
            .Distinct()
            .OrderBy(value => value)
            .ToArray();
        var frontier = new Dictionary<int, SharedOwner>();
        var drainedStages = new HashSet<string>(StringComparer.Ordinal);
        var consumerMarkers = new Dictionary<Expr, List<PipelineHandoff>>(
            ReferenceEqualityComparer.Instance);
        var producerMarkers = new Dictionary<Expr, List<PipelineHandoff>>(
            ReferenceEqualityComparer.Instance);
        var handoffIndex = 0;

        foreach (var owner in owners)
        {
            var predecessorOffsets = new Dictionary<SharedOwner, long>(
                ReferenceEqualityComparer.Instance);
            for (var segment = 0; segment + 1 < boundaries.Length; segment++)
            {
                var segmentRange = new MemoryByteRange(
                    boundaries[segment],
                    boundaries[segment + 1]);
                if (!owner.Ranges.Any(range => range.Overlaps(segmentRange)))
                {
                    continue;
                }

                if (frontier.TryGetValue(segment, out var predecessor) &&
                    !ReferenceEquals(predecessor, owner))
                {
                    predecessorOffsets[predecessor] = predecessorOffsets.TryGetValue(
                        predecessor,
                        out var currentOffset)
                        ? Math.Min(currentOffset, segmentRange.Start)
                        : segmentRange.Start;
                }

                frontier[segment] = owner;
            }

            foreach (var (predecessor, sharedOffsetBytes) in predecessorOffsets)
            {
                if (predecessor.Stage is { } predecessorStage)
                {
                    drainedStages.Add(predecessorStage.StageId);
                    continue;
                }

                if (owner.Stage is null)
                {
                    continue;
                }

                var handoff = new PipelineHandoff(
                    $"{functionName}_shared_handoff_{handoffIndex++}_at_{sharedOffsetBytes}");
                AddMarker(consumerMarkers, predecessor.Expression, handoff);
                AddMarker(producerMarkers, owner.Expression, handoff);
            }
        }

        AddTransferSourceHandoffs(
            consumer,
            functionName,
            effectAnalyzer,
            consumerMarkers,
            producerMarkers,
            ref handoffIndex);

        return new(
            drainedStages,
            consumerMarkers,
            producerMarkers);
    }

    private static void AddTransferSourceHandoffs(
        Sequential consumer,
        string functionName,
        MemoryEffectAnalyzer effectAnalyzer,
        IDictionary<Expr, List<PipelineHandoff>> consumerMarkers,
        IDictionary<Expr, List<PipelineHandoff>> producerMarkers,
        ref int handoffIndex)
    {
        var executionOrder = EnumerateExecutionOrder(consumer).ToArray();
        for (var stageIndex = 0; stageIndex < executionOrder.Length; stageIndex++)
        {
            if (executionOrder[stageIndex] is not PipelineStage stage)
            {
                continue;
            }

            var dependency = effectAnalyzer.AnalyzeTransferSourceDependency(
                executionOrder,
                stageIndex,
                stage.Operation);
            if (!dependency.SourceEffects.Items.Any())
            {
                if (stage.Operation.Target is PrimFunction)
                {
                    // The callee may have no entry-ready source because every
                    // transfer is released by one of its internal handoffs.
                    continue;
                }

                throw new InvalidOperationException(
                    $"Transfer-pipeline stage {stage.StageId} in {functionName} has no " +
                    "physical source effects.");
            }

            if (dependency.UnsynchronizedRequirement is { } unsynchronized)
            {
                throw new InvalidOperationException(
                    $"Transfer-pipeline stage {stage.StageId} in {functionName} reads a " +
                    $"source after a conflicting write without an effective {unsynchronized.Scope} barrier.");
            }

            if (dependency.ReleaseBoundary is not { } releaseBoundary)
            {
                continue;
            }

            var handoff = new PipelineHandoff(
                $"{functionName}_source_handoff_{handoffIndex++}");
            AddMarker(consumerMarkers, releaseBoundary, handoff);
            AddMarker(producerMarkers, stage, handoff);
        }
    }

    private static IEnumerable<SharedOwner> EnumerateSharedOwners(
        Sequential body,
        string functionName,
        MemoryEffectAnalyzer effectAnalyzer)
    {
        foreach (var expression in EnumerateExecutionOrder(body))
        {
            var stage = expression as PipelineStage;
            var ranges = stage is not null
                ? CollectSharedRanges(
                    GetPipelineSharedOperands(stage.Operation, functionName),
                    functionName)
                : CollectOrdinarySharedRanges(
                    expression,
                    effectAnalyzer,
                    functionName);
            if (ranges.Count == 0)
            {
                if (stage is not null)
                {
                    throw new InvalidOperationException(
                        $"Transfer-pipeline stage {stage.StageId} in {functionName} has no " +
                        "physical Shared workspace after Bufferize.");
                }

                continue;
            }

            yield return new SharedOwner(
                expression,
                ranges,
                stage);
        }
    }

    private static IReadOnlyList<BaseExpr> GetPipelineSharedOperands(
        Call call,
        string functionName)
    {
        switch (call.Target)
        {
            case TIR.NTT.NTTKernelOp:
                var selection = call.Metadata.TIRMicroKernel
                    ?? throw new InvalidOperationException(
                        $"Transfer-pipeline kernel in {functionName} has no TIR microkernel selection.");
                var contract = selection.TransferPipeline
                    ?? throw new InvalidOperationException(
                        $"Transfer-pipeline kernel {selection.Family}/{selection.Variant} in " +
                        $"{functionName} has no transfer-pipeline contract.");
                if (call.Arguments.Length == 0)
                {
                    throw new InvalidOperationException(
                        $"Transfer-pipeline kernel {selection.Family}/{selection.Variant} in " +
                        $"{functionName} has no Shared workspace operand.");
                }

                var workspaces = TIRSharedWorkspace.Unpack(call.Arguments[^1]);
                return contract.SharedWorkspaceIndices
                    .Select(index => (uint)index < (uint)workspaces.Length
                        ? workspaces[index]
                        : throw new InvalidOperationException(
                            $"Transfer-pipeline kernel {selection.Family}/{selection.Variant} in " +
                            $"{functionName} references missing Shared workspace {index}."))
                    .ToArray();
            case PrimFunction callee:
                var parameters = callee.Parameters.ToArray();
                var arguments = call.Arguments.ToArray();
                if (parameters.Length != arguments.Length)
                {
                    throw new InvalidOperationException(
                        $"Pipeline PrimFunction {callee.Name} in {functionName} expects " +
                        $"{parameters.Length} arguments, got {arguments.Length}.");
                }

                var sharedWorkspaces = parameters
                    .Select((parameter, index) => (Parameter: parameter, Argument: arguments[index]))
                    .Where(item => item.Parameter is BufferVar
                    {
                        Role: BufferVarRole.Workspace,
                        Location: MemoryLocation.Shared,
                    })
                    .Select(item => item.Argument)
                    .ToArray();
                if (sharedWorkspaces.Length == 0)
                {
                    throw new InvalidOperationException(
                        $"Pipeline PrimFunction {callee.Name} in {functionName} has no " +
                        "caller-allocated Shared workspace parameter.");
                }

                return sharedWorkspaces;
            default:
                throw new InvalidOperationException(
                    $"Pipeline stage in {functionName} must call an NTT kernel or PrimFunction, " +
                    $"got {call.Target.GetType().Name}.");
        }
    }

    private static IReadOnlyList<MemoryByteRange> CollectSharedRanges(
        IEnumerable<BaseExpr> operands,
        string functionName)
    {
        var buffers = new HashSet<TIR.Buffer>(ReferenceEqualityComparer.Instance);
        foreach (var operand in operands)
        {
            CollectSharedBuffers(operand, buffers);
        }

        var ranges = new HashSet<MemoryByteRange>();
        foreach (var buffer in buffers)
        {
            AddSharedRange(
                ranges,
                MemoryEffectAnalyzer.TryGetAbsoluteAccessByteRange(buffer.MemSpan),
                buffer.Name,
                functionName);
        }

        return ranges.OrderBy(range => range.Start).ThenBy(range => range.End).ToArray();
    }

    private static IReadOnlyList<MemoryByteRange> CollectOrdinarySharedRanges(
        Expr expression,
        MemoryEffectAnalyzer effectAnalyzer,
        string functionName)
    {
        var ranges = new HashSet<MemoryByteRange>();
        foreach (var item in effectAnalyzer.GetEffects(expression).Items)
        {
            if (!IsSharedResource(item.Resource))
            {
                continue;
            }

            AddSharedRange(
                ranges,
                item.Resource.AccessRange,
                GetResourceName(item.Resource),
                functionName);
        }

        foreach (var call in EnumerateCalls(expression))
        {
            if (call.Target is not PrimFunction callee)
            {
                continue;
            }

            var parameters = callee.Parameters.ToArray();
            var arguments = call.Arguments.ToArray();
            if (parameters.Length != arguments.Length)
            {
                throw new InvalidOperationException(
                    $"PrimFunction {callee.Name} in {functionName} expects " +
                    $"{parameters.Length} arguments, got {arguments.Length}.");
            }

            foreach (var (parameter, index) in parameters.Select(
                         (parameter, index) => (parameter, index)))
            {
                if (parameter is not BufferVar
                    {
                        Role: BufferVarRole.Workspace,
                        Location: MemoryLocation.Shared,
                    })
                {
                    continue;
                }

                foreach (var range in CollectSharedRanges(
                             new[] { arguments[index] },
                             functionName))
                {
                    ranges.Add(range);
                }
            }
        }

        return ranges.OrderBy(range => range.Start).ThenBy(range => range.End).ToArray();
    }

    private static bool IsSharedResource(MemoryResource resource)
        => resource.Arena?.Location == MemoryLocation.Shared ||
            (resource.LogicalIdentity is TIR.Buffer buffer &&
             buffer.MemSpan.Buffer.Location == MemoryLocation.Shared);

    private static string GetResourceName(MemoryResource resource)
        => resource.LogicalIdentity switch
        {
            TIR.Buffer buffer => buffer.Name,
            IVar variable => variable.Name,
            { } expression => expression.GetType().Name,
            null => "unknown",
        };

    private static void AddSharedRange(
        ISet<MemoryByteRange> ranges,
        MemoryByteRange? range,
        string resourceName,
        string functionName)
    {
        var value = range
            ?? throw new InvalidOperationException(
                $"Shared resource {resourceName} in {functionName} has no fixed " +
                "post-Bufferize byte range.");
        if (value.End <= value.Start)
        {
            throw new InvalidOperationException(
                $"Shared resource {resourceName} in {functionName} has empty range " +
                $"[{value.Start}, {value.End}).");
        }

        ranges.Add(value);
    }

    private static void CollectSharedBuffers(
        BaseExpr expression,
        HashSet<TIR.Buffer> buffers)
    {
        switch (expression)
        {
            case TIR.Buffer buffer when
                buffer.MemSpan.Buffer.Location == MemoryLocation.Shared:
                buffers.Add(buffer);
                break;
            case IR.Tuple tuple:
                foreach (var field in tuple.Fields)
                {
                    CollectSharedBuffers(field, buffers);
                }

                break;
            case Call { Target: IR.Buffers.BufferSubview or IR.Buffers.AllocateBufferView } view
                when view.Arguments.Length > 0:
                CollectSharedBuffers(view.Arguments[0], buffers);
                break;
        }
    }

    private static Sequential InsertConsumerSynchronization(
        Sequential body,
        PipelineSynchronizationPlan synchronization)
        => RewriteSequential(
            body,
            field =>
            {
                var fields = new List<Expr> { RewriteConsumerExpression(field, synchronization) };
                if (field is PipelineStage stage &&
                    synchronization.DrainedStages.Contains(stage.StageId))
                {
                    fields.Add(new PipelineDrain(stage.StageId));
                }

                if (synchronization.ConsumerHandoffs.TryGetValue(field, out var handoffs))
                {
                    fields.AddRange(handoffs);
                }

                return fields;
            });

    private static Expr RewriteConsumerExpression(
        Expr expression,
        PipelineSynchronizationPlan synchronization)
        => expression switch
        {
            Sequential sequential => InsertConsumerSynchronization(sequential, synchronization),
            _ => expression,
        };

    private static Sequential BuildProducerBody(
        Sequential consumer,
        PipelineSynchronizationPlan synchronization)
        => RewriteSequential(
            consumer,
            field =>
            {
                var fields = new List<Expr>();
                if (synchronization.ProducerHandoffs.TryGetValue(field, out var handoffs))
                {
                    fields.AddRange(handoffs);
                }

                switch (field)
                {
                    case PipelineStage stage:
                        fields.Add(stage);
                        if (synchronization.DrainedStages.Contains(stage.StageId))
                        {
                            fields.Add(new PipelineDrain(stage.StageId));
                        }

                        break;
                    case Sequential sequential:
                        var nested = BuildProducerBody(
                            sequential,
                            synchronization);
                        if (nested.Count > 0)
                        {
                            fields.Add(nested);
                        }

                        break;
                }

                return fields;
            });

    private static Sequential RewriteSequential(
        Sequential source,
        Func<Expr, IEnumerable<Expr>> rewriteField)
    {
        var fields = source.Fields
            .ToArray()
            .SelectMany(rewriteField)
            .ToArray();
        return source.With(fields: fields);
    }

    private static Sequential WrapPipelineStages(
        Sequential body,
        IReadOnlySet<PrimFunction> pipelineFunctions,
        string functionName,
        ref int stageIndex)
    {
        var fields = new List<Expr>();
        foreach (var field in body.Fields)
        {
            switch (field)
            {
                case Sequential sequential:
                    fields.Add(WrapPipelineStages(
                        sequential,
                        pipelineFunctions,
                        functionName,
                        ref stageIndex));
                    break;
                case Call call when IsPipelineInvocation(call, pipelineFunctions):
                    fields.Add(new PipelineStage(
                        $"{functionName}_transfer_stage_{stageIndex++}",
                        call));
                    break;
                default:
                    fields.Add(field);
                    break;
            }
        }

        return body.With(fields: fields.ToArray());
    }

    private static void ValidateStructuredPipelinePlacement(
        Sequential body,
        IReadOnlySet<PrimFunction> pipelineFunctions,
        string functionName)
    {
        foreach (var expression in EnumerateStructuralExpressions(body))
        {
            switch (expression)
            {
                case IfThenElse conditional
                    when ContainsPipelineInvocation(
                        conditional.Then,
                        pipelineFunctions) ||
                         ContainsPipelineInvocation(
                             conditional.Else,
                             pipelineFunctions):
                    throw UnsupportedControlFlow(functionName, nameof(IfThenElse));
                case For loop when ContainsPipelineInvocation(
                    loop.Body,
                    pipelineFunctions):
                    throw UnsupportedControlFlow(functionName, nameof(For));
                case PipelineFor pipelineFor
                    when ContainsPipelineInvocation(
                        pipelineFor.ProduceBody,
                        pipelineFunctions) ||
                         ContainsPipelineInvocation(
                             pipelineFor.ConsumeBody,
                             pipelineFunctions):
                    throw UnsupportedControlFlow(functionName, nameof(PipelineFor));
                case Let let when ContainsPipelineInvocation(
                    let.Body,
                    pipelineFunctions):
                    throw UnsupportedControlFlow(functionName, nameof(Let));
                case Block block
                    when ContainsPipelineInvocation(
                        block.InitBody,
                        pipelineFunctions) ||
                         ContainsPipelineInvocation(
                             block.Body,
                             pipelineFunctions):
                    throw UnsupportedControlFlow(functionName, nameof(Block));
            }
        }
    }

    private static NotSupportedException UnsupportedControlFlow(
        string functionName,
        string construct)
        => new(
            $"PrimFunction {functionName} contains a transfer-pipeline stage under " +
            $"{construct}. Producer/consumer lowering requires a straight-line " +
            "stage order until structured task control flow is represented in TIR.");

    private static bool ContainsPipelineInvocation(
        Expr expression,
        IReadOnlySet<PrimFunction> pipelineFunctions)
        => EnumerateCalls(expression)
            .Any(call => IsPipelineInvocation(call, pipelineFunctions));

    private static bool IsPipelineInvocation(
        Call call,
        IReadOnlySet<PrimFunction> pipelineFunctions)
        => IsTransferPipelineCall(call) ||
            (call.Target is PrimFunction callee &&
             pipelineFunctions.Contains(callee));

    private static bool IsTransferPipelineCall(Call call)
        => call.Metadata.TIRMicroKernel?.TransferPipeline is not null;

    private static IEnumerable<Call> EnumerateCalls(BaseExpr expression)
    {
        switch (expression)
        {
            case Call call:
                yield return call;
                yield break;
            case Sequential sequential:
                foreach (var field in sequential.Fields.ToArray())
                {
                    foreach (var call in EnumerateCalls(field))
                    {
                        yield return call;
                    }
                }

                yield break;
            case IfThenElse conditional:
                foreach (var call in EnumerateCalls(conditional.Then))
                {
                    yield return call;
                }

                foreach (var call in EnumerateCalls(conditional.Else))
                {
                    yield return call;
                }

                yield break;
            case For loop:
                foreach (var call in EnumerateCalls(loop.Body))
                {
                    yield return call;
                }

                yield break;
            case PipelineFor pipeline:
                foreach (var call in EnumerateCalls(pipeline.ProduceBody))
                {
                    yield return call;
                }

                foreach (var call in EnumerateCalls(pipeline.ConsumeBody))
                {
                    yield return call;
                }

                yield break;
            case Let let:
                foreach (var call in EnumerateCalls(let.Body))
                {
                    yield return call;
                }

                yield break;
            case Block block:
                foreach (var call in EnumerateCalls(block.InitBody))
                {
                    yield return call;
                }

                foreach (var call in EnumerateCalls(block.Body))
                {
                    yield return call;
                }

                yield break;
        }
    }

    private static IEnumerable<Expr> EnumerateStructuralExpressions(Expr expression)
    {
        yield return expression;
        switch (expression)
        {
            case Sequential sequential:
                foreach (var field in sequential.Fields.ToArray())
                {
                    foreach (var nested in EnumerateStructuralExpressions(field))
                    {
                        yield return nested;
                    }
                }

                break;
            case IfThenElse conditional:
                foreach (var nested in EnumerateStructuralExpressions(conditional.Then))
                {
                    yield return nested;
                }

                foreach (var nested in EnumerateStructuralExpressions(conditional.Else))
                {
                    yield return nested;
                }

                break;
            case For loop:
                foreach (var nested in EnumerateStructuralExpressions(loop.Body))
                {
                    yield return nested;
                }

                break;
            case PipelineFor pipeline:
                foreach (var nested in EnumerateStructuralExpressions(pipeline.ProduceBody))
                {
                    yield return nested;
                }

                foreach (var nested in EnumerateStructuralExpressions(pipeline.ConsumeBody))
                {
                    yield return nested;
                }

                break;
            case Let let:
                foreach (var nested in EnumerateStructuralExpressions(let.Body))
                {
                    yield return nested;
                }

                break;
            case Block block:
                foreach (var nested in EnumerateStructuralExpressions(block.InitBody))
                {
                    yield return nested;
                }

                foreach (var nested in EnumerateStructuralExpressions(block.Body))
                {
                    yield return nested;
                }

                break;
        }
    }

    private static IEnumerable<Expr> EnumerateExecutionOrder(Sequential body)
    {
        foreach (var field in body.Fields.ToArray())
        {
            if (field is Sequential nested)
            {
                foreach (var expression in EnumerateExecutionOrder(nested))
                {
                    yield return expression;
                }
            }
            else
            {
                yield return field;
            }
        }
    }

    private static bool ContainsRegion(BaseExpr expression)
        => expression switch
        {
            ProducerConsumerRegion => true,
            Sequential sequential => sequential.Fields.ToArray().Any(ContainsRegion),
            _ => false,
        };

    private static void AddMarker(
        IDictionary<Expr, List<PipelineHandoff>> markers,
        Expr expression,
        PipelineHandoff handoff)
    {
        if (!markers.TryGetValue(expression, out var values))
        {
            values = new List<PipelineHandoff>();
            markers.Add(expression, values);
        }

        values.Add(handoff);
    }

    private sealed record SharedOwner(
        Expr Expression,
        IReadOnlyList<MemoryByteRange> Ranges,
        PipelineStage? Stage);

    private sealed record PipelineSynchronizationPlan(
        IReadOnlySet<string> DrainedStages,
        IReadOnlyDictionary<Expr, List<PipelineHandoff>> ConsumerHandoffs,
        IReadOnlyDictionary<Expr, List<PipelineHandoff>> ProducerHandoffs);

    private sealed class FunctionTargetRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<PrimFunction, PrimFunction> _replacements;

        public FunctionTargetRewriter(
            IReadOnlyDictionary<PrimFunction, PrimFunction> replacements)
            : base(visitOtherFunctions: false)
        {
            _replacements = replacements;
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
            => expr.Target is PrimFunction target &&
                _replacements.TryGetValue(target, out var replacement)
                ? expr.With(target: replacement)
                : expr;
    }
}
