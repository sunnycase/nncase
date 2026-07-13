// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

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
}

internal sealed record ReductionAccumulatorOperand(
    ParameterInfo Parameter,
    BaseExpr Argument,
    MemoryEffect Effect);
