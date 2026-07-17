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

    internal static bool TryGetAdjacentReductionLoopPartitionPair(
        ReadOnlySpan<Expr> fields,
        int index,
        out For fullLoop,
        out For tailLoop)
    {
        fullLoop = null!;
        tailLoop = null!;
        if (fields[index] is not For { Partition: LoopPartition.Full, Mode: LoopMode.Reduction } loop ||
            index + 1 >= fields.Length ||
            fields[index + 1] is not For { Partition: LoopPartition.Tail } candidateTail)
        {
            return false;
        }

        if (loop.Mode != candidateTail.Mode ||
            !loop.Domain.Stop.Equals(candidateTail.Domain.Start) ||
            !loop.Domain.Step.Equals(candidateTail.Domain.Step))
        {
            throw new InvalidOperationException(
                $"Malformed full/tail loop pair {loop.LoopVar.Name}/{candidateTail.LoopVar.Name}: " +
                "the loops must have the same mode and step, and share the partition boundary.");
        }

        fullLoop = loop;
        tailLoop = candidateTail;
        return true;
    }

    internal static ReductionCallGroup[] CollectReductionCallGroups(params BaseExpr[] expressions)
    {
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
            .Select(group => new ReductionCallGroup(group.Prototype, group.Calls.ToArray()))
            .ToArray();
    }

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

        public void Add(Call call, int expressionIndex)
        {
            Calls.Add(call);
            _callsPerExpression[expressionIndex]++;
        }
    }
}

internal sealed record ReductionCallGroup(Call Prototype, Call[] Calls);

internal sealed record ReductionAccumulatorOperand(
    ParameterInfo Parameter,
    BaseExpr Argument,
    MemoryEffect Effect);
