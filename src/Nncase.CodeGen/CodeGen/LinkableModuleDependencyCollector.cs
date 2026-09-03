// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// Collects direct cross-module call dependencies without traversing into the
/// callee's lexical scope.
/// </summary>
public static class LinkableModuleDependencyCollector
{
    public static IReadOnlySet<string> Collect(string moduleKind, IEnumerable<BaseFunction> functions)
    {
        var dependencies = new HashSet<string>(StringComparer.Ordinal);
        foreach (var function in functions)
        {
            var body = function switch
            {
                Function value => value.Body,
                Fusion value => value.Body,
                TIR.PrimFunction value => value.Body,
                _ => throw new NotSupportedException(
                    $"Cannot collect module dependencies from {function.GetType().Name} {function.Name}."),
            };
            Collect(body, moduleKind, dependencies);
        }

        return dependencies;
    }

    private static void Collect(BaseExpr expr, string moduleKind, HashSet<string> dependencies)
    {
        if (expr is Call { Target: BaseFunction callee } call)
        {
            if (!string.Equals(callee.ModuleKind, moduleKind, StringComparison.Ordinal))
            {
                dependencies.Add(callee.ModuleKind);
            }

            foreach (var argument in call.Arguments)
            {
                Collect(argument, moduleKind, dependencies);
            }

            return;
        }

        if (expr is BaseFunction)
        {
            return;
        }

        foreach (var operand in expr.Operands)
        {
            Collect(operand, moduleKind, dependencies);
        }
    }
}
