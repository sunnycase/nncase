// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.NTT;

internal static class ReductionCodegenUtility
{
    public static Call[] CollectReductionCalls(BaseExpr expression)
    {
        var calls = new List<Call>();
        var stack = new Stack<BaseExpr>();
        var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
        stack.Push(expression);
        while (stack.Count > 0)
        {
            var current = stack.Pop();
            if (!visited.Add(current))
            {
                continue;
            }

            if (current is Call { Target: Op } call && GetAccumulatorOperands(call).Length > 0)
            {
                calls.Add(call);
                continue;
            }

            if (current is BaseFunction)
            {
                continue;
            }

            foreach (var operand in current.Operands)
            {
                stack.Push(operand);
            }
        }

        return calls.ToArray();
    }

    public static ReductionAccumulatorOperand[] GetAccumulatorOperands(Call call)
    {
        if (call.Target is not Op op)
        {
            return Array.Empty<ReductionAccumulatorOperand>();
        }

        var operands = new List<ReductionAccumulatorOperand>();
        call.ParametersForeach((argument, parameter) =>
        {
            var effect = op is IOpMemoryEffectProvider provider
                ? provider.GetMemoryEffect(parameter)
                : parameter.MemoryEffect ?? MemoryEffect.None;
            if (effect.Kind == MemoryEffectKind.ReductionAccumulator)
            {
                operands.Add(new ReductionAccumulatorOperand(
                    parameter,
                    argument,
                    effect));
            }
        });

        return operands.ToArray();
    }

    internal static bool TryGetReductionLoopPartitionPair(
        ReadOnlySpan<Expr> fields,
        int index,
        out ReductionLoopPartitionPair pair)
    {
        pair = null!;
        if ((uint)index >= (uint)fields.Length)
        {
            return false;
        }

        var fullLoop = fields[index];
        var fullDomain = fullLoop switch
        {
            For { Partition: LoopPartition.Full, Mode: LoopMode.Reduction } loop => loop.Domain,
            PipelineFor { Partition: LoopPartition.Full, Mode: LoopMode.Reduction } loop => loop.Domain,
            _ => null,
        };
        if (fullDomain is null)
        {
            return false;
        }

        var synchronizationFields = new List<Expr>();
        var tailIndex = index + 1;
        while (tailIndex < fields.Length && IsInterPartitionSynchronization(fields[tailIndex]))
        {
            synchronizationFields.Add(fields[tailIndex]);
            tailIndex++;
        }

        if (tailIndex >= fields.Length)
        {
            return false;
        }

        var tailLoop = fields[tailIndex];
        var tailDomain = tailLoop switch
        {
            For { Partition: LoopPartition.Tail, Mode: LoopMode.Reduction } loop => loop.Domain,
            PipelineFor { Partition: LoopPartition.Tail, Mode: LoopMode.Reduction } loop => loop.Domain,
            _ => null,
        };
        if (tailDomain is null)
        {
            return false;
        }

        if ((fullLoop is PipelineFor) != (tailLoop is PipelineFor))
        {
            throw new InvalidOperationException(
                $"Malformed full/tail reduction loop pair at fields {index}/{tailIndex}: " +
                "both partitions must use the same TIR loop representation.");
        }

        if (fullLoop is PipelineFor fullPipeline && tailLoop is PipelineFor tailPipeline &&
            (fullPipeline.Plan != tailPipeline.Plan ||
             fullPipeline.BindingDescriptors.Length != tailPipeline.BindingDescriptors.Length ||
             !fullPipeline.BindingDescriptors.SequenceEqual(tailPipeline.BindingDescriptors)))
        {
            throw new InvalidOperationException(
                $"Malformed pipeline reduction pair at fields {index}/{tailIndex}: " +
                "full and tail partitions must share one schedule and channel contract.");
        }

        if (!fullDomain.Stop.Equals(tailDomain.Start) ||
            !fullDomain.Step.Equals(tailDomain.Step))
        {
            throw new InvalidOperationException(
                $"Malformed full/tail reduction loop pair at fields {index}/{tailIndex}: " +
                "the loops must have the same step and share the partition boundary.");
        }

        pair = new(
            fullLoop,
            tailLoop,
            synchronizationFields.ToArray(),
            tailIndex);
        return true;
    }

    internal static ReductionCallGroup[] CollectReductionCallGroups(params BaseExpr[] expressions)
    {
        if (expressions.Length == 0)
        {
            return Array.Empty<ReductionCallGroup>();
        }

        var groups = new List<MutableReductionCallGroup>();
        for (var expressionIndex = 0; expressionIndex < expressions.Length; expressionIndex++)
        {
            foreach (var call in CollectReductionCalls(expressions[expressionIndex]))
            {
                var accumulatorIdentities = GetAccumulatorOperands(call)
                    .Select(operand => operand.Argument)
                    .ToArray();
                var group = groups.FirstOrDefault(candidate =>
                    candidate.Prototype.Target.Equals(call.Target) &&
                    SameAccumulatorIdentities(candidate.AccumulatorIdentities, accumulatorIdentities));
                if (group is null)
                {
                    groups.Add(new MutableReductionCallGroup(
                        call,
                        accumulatorIdentities,
                        expressionIndex,
                        expressions.Length));
                }
                else
                {
                    group.Add(call, expressionIndex);
                }
            }
        }

        if (expressions.Length > 1)
        {
            var incompleteGroup = groups.FirstOrDefault(group => !group.HasOneCallPerExpression);
            if (incompleteGroup is not null)
            {
                throw new InvalidOperationException(
                    $"Structured reduction partitions for {incompleteGroup.Prototype.Target.GetType().Name} " +
                    "must each update the same accumulator exactly once.");
            }
        }

        return groups
            .Select(group => new ReductionCallGroup(
                group.Prototype,
                group.Calls.ToArray(),
                group.ExpectedUpdateCount))
            .ToArray();
    }

    private static bool IsInterPartitionSynchronization(Expr expression)
        => expression is Call { Target: TIR.NTT.Barrier };

    private static bool SameAccumulatorIdentities(
        IReadOnlyList<BaseExpr> lhs,
        IReadOnlyList<BaseExpr> rhs)
    {
        // Peeling clones logical views, so reference identity is too strict.
        // Comparing the complete view also keeps distinct regions of one
        // backing allocation from sharing backend-private accumulator state.
        return lhs.Count == rhs.Count &&
            lhs.Zip(rhs).All(pair => pair.First.Equals(pair.Second));
    }

    private sealed class MutableReductionCallGroup
    {
        private readonly int[] _callsPerExpression;

        public MutableReductionCallGroup(
            Call prototype,
            BaseExpr[] accumulatorIdentities,
            int expressionIndex,
            int expressionCount)
        {
            Prototype = prototype;
            AccumulatorIdentities = accumulatorIdentities;
            _callsPerExpression = new int[expressionCount];
            Calls.Add(prototype);
            _callsPerExpression[expressionIndex] = 1;
        }

        public Call Prototype { get; }

        public BaseExpr[] AccumulatorIdentities { get; }

        public List<Call> Calls { get; } = new();

        public bool HasOneCallPerExpression => _callsPerExpression.All(count => count == 1);

        public int ExpectedUpdateCount => _callsPerExpression.Sum();

        public void Add(Call call, int expressionIndex)
        {
            Calls.Add(call);
            _callsPerExpression[expressionIndex]++;
        }
    }
}

internal sealed record ReductionLoopPartitionPair(
    Expr FullLoop,
    Expr TailLoop,
    IReadOnlyList<Expr> SynchronizationFields,
    int TailFieldIndex)
{
    public BaseExpr FullBody => FullLoop switch
    {
        For loop => loop.Body,
        PipelineFor loop => loop.ConsumeBody,
        _ => throw new InvalidOperationException(
            $"Unsupported full reduction partition {FullLoop.GetType().Name}."),
    };

    public BaseExpr TailBody => TailLoop switch
    {
        For loop => loop.Body,
        PipelineFor loop => loop.ConsumeBody,
        _ => throw new InvalidOperationException(
            $"Unsupported tail reduction partition {TailLoop.GetType().Name}."),
    };
}

internal sealed record ReductionCallGroup(
    Call Prototype,
    Call[] Calls,
    int ExpectedUpdateCount);

internal sealed record ReductionAccumulatorOperand(
    ParameterInfo Parameter,
    BaseExpr Argument,
    MemoryEffect Effect);
