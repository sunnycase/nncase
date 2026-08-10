// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// A block-local asynchronous execution region with one producer task and one
/// semantic consumer task. Both bodies are constructed by the owning lowering
/// pass; code generation must not recover either role from memory effects.
/// </summary>
public sealed class ProducerConsumerRegion : Expr
{
    public ProducerConsumerRegion(Sequential produceBody, Sequential consumeBody)
        : base([produceBody, consumeBody])
    {
        Validate(produceBody, consumeBody);
    }

    public Sequential ProduceBody => (Sequential)Operands[0];

    public Sequential ConsumeBody => (Sequential)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(
        ExprFunctor<TExprResult, TTypeResult, TContext> functor,
        TContext context)
        => functor.VisitProducerConsumerRegion(this, context);

    public ProducerConsumerRegion With(
        Sequential? produceBody = null,
        Sequential? consumeBody = null)
        => new(produceBody ?? ProduceBody, consumeBody ?? ConsumeBody);

    private static void Validate(Sequential produceBody, Sequential consumeBody)
    {
        ArgumentNullException.ThrowIfNull(produceBody);
        ArgumentNullException.ThrowIfNull(consumeBody);

        var produce = RegionStructure.Collect(produceBody, "producer");
        var consume = RegionStructure.Collect(consumeBody, "consumer");
        if (produce.StageIds.Count == 0 || consume.StageIds.Count == 0)
        {
            throw new ArgumentException(
                "Producer/consumer regions must contain at least one pipeline stage.");
        }

        if (!produce.StageIds.SequenceEqual(consume.StageIds, StringComparer.Ordinal))
        {
            throw new ArgumentException(
                "Producer and consumer pipeline stages must have identical IDs and order.");
        }

        if (!produce.DrainIds.SetEquals(consume.DrainIds))
        {
            throw new ArgumentException(
                "Producer and consumer pipeline drain sets must be identical.");
        }

        if (!produce.HandoffIds.SetEquals(consume.HandoffIds))
        {
            throw new ArgumentException(
                "Producer and consumer handoffs must have identical IDs.");
        }
    }

    private sealed record RegionStructure(
        IReadOnlyList<string> StageIds,
        HashSet<string> DrainIds,
        HashSet<string> HandoffIds)
    {
        public static RegionStructure Collect(Sequential body, string role)
        {
            var stages = new List<string>();
            var stageSet = new HashSet<string>(StringComparer.Ordinal);
            var drains = new HashSet<string>(StringComparer.Ordinal);
            var handoffs = new HashSet<string>(StringComparer.Ordinal);
            Visit(body);
            return new(stages, drains, handoffs);

            void Visit(Expr expression)
            {
                switch (expression)
                {
                    case PipelineStage stage:
                        if (!stageSet.Add(stage.StageId))
                        {
                            throw new ArgumentException(
                                $"{role} body contains duplicate pipeline stage {stage.StageId}.");
                        }

                        stages.Add(stage.StageId);
                        break;
                    case PipelineDrain drain:
                        if (!stageSet.Contains(drain.StageId))
                        {
                            throw new ArgumentException(
                                $"{role} body drains pipeline stage {drain.StageId} before executing it.");
                        }

                        if (!drains.Add(drain.StageId))
                        {
                            throw new ArgumentException(
                                $"{role} body drains pipeline stage {drain.StageId} more than once.");
                        }

                        break;
                    case PipelineHandoff handoff:
                        if (!handoffs.Add(handoff.HandoffId))
                        {
                            throw new ArgumentException(
                                $"{role} body contains duplicate pipeline handoff {handoff.HandoffId}.");
                        }

                        break;
                    case Sequential sequential:
                        foreach (var field in sequential.Fields)
                        {
                            Visit(field);
                        }

                        break;
                    case IfThenElse conditional:
                        Visit(conditional.Then);
                        Visit(conditional.Else);
                        break;
                    case For loop:
                        Visit(loop.Body);
                        break;
                    case Let let:
                        Visit(let.Body);
                        break;
                }
            }
        }
    }
}
