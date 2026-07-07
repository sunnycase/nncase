// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using System.Xml;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.ShapeBucket;

public sealed class AddFunctionToModule : ModulePass
{
    public AddFunctionToModule(CompileOptions compileOptions)
    {
        CompileOptions = compileOptions;
    }

    public CompileOptions CompileOptions { get; }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        while (true)
        {
            var funcs = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
            foreach (var func in input.Functions)
            {
                var collector = new FuncCollector(funcs);
                collector.Visit(func);
            }

            var toAdd = funcs.Except(input.Functions).ToArray();
            if (toAdd.Length == 0)
            {
                break;
            }

            foreach (var ifToAdd in toAdd)
            {
                input.Add(ifToAdd);
            }
        }

        ModuleGraphValidator.ValidateAcyclic(input);
        return Task.FromResult(input);
    }

    private sealed class ModuleGraphValidator
    {
        public static void ValidateAcyclic(IRModule module)
        {
            var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            var active = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            var path = new List<BaseExpr>();
            foreach (var function in module.Functions)
            {
                Visit(function, visited, active, path);
            }
        }

        private static void Visit(BaseExpr expr, HashSet<BaseExpr> visited, HashSet<BaseExpr> active, List<BaseExpr> path)
        {
            if (active.Contains(expr))
            {
                var cycleStart = path.FindIndex(item => ReferenceEquals(item, expr));
                var cycle = path.Skip(cycleStart).Append(expr).Select(Describe);
                throw new InvalidOperationException($"IR module contains an operand cycle: {string.Join(" -> ", cycle)}.");
            }

            if (!visited.Add(expr))
            {
                return;
            }

            active.Add(expr);
            path.Add(expr);
            var operands = expr.Operands;
            for (int i = 0; i < operands.Length; i++)
            {
                Visit(operands[i], visited, active, path);
            }

            path.RemoveAt(path.Count - 1);
            active.Remove(expr);
        }

        private static string Describe(BaseExpr expr)
        {
            return expr switch
            {
                BaseFunction function => $"{expr.GetType().Name}({function.Name})",
                Call call => $"Call({DescribeCallTarget(call.Target)})",
                Op op => $"Op({op.GetType().Name})",
                Var var => $"Var({var.Name})",
                _ => expr.GetType().Name,
            };
        }

        private static string DescribeCallTarget(Expr target)
        {
            return target switch
            {
                BaseFunction function => function.Name,
                Op op => op.GetType().Name,
                _ => target.GetType().Name,
            };
        }
    }

    private class FuncCollector : ExprWalker
    {
        public FuncCollector(HashSet<BaseFunction> funcs)
            : base(true)
        {
            Funcs = funcs;
        }

        public HashSet<BaseFunction> Funcs { get; }

        protected override Unit VisitLeafBaseFunction(BaseFunction expr)
        {
            Funcs.Add(expr);
            return default;
        }
    }
}
