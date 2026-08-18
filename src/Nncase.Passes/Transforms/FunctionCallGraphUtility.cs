// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

internal static class FunctionCallGraphUtility
{
    public static Dictionary<Function, List<Call>> CollectCallSites(IEnumerable<Function> functions)
    {
        var callSites = new Dictionary<Function, List<Call>>(ReferenceEqualityComparer.Instance);
        foreach (var function in functions)
        {
            foreach (var call in ExprCollector.Collect(function.Body).OfType<Call>())
            {
                if (call.Target is not Function target)
                {
                    continue;
                }

                if (!callSites.TryGetValue(target, out var calls))
                {
                    calls = new List<Call>();
                    callSites.Add(target, calls);
                }

                calls.Add(call);
            }
        }

        return callSites;
    }

    public static Function[] GetCalleeFirstOrder(IEnumerable<Function> functions)
    {
        var functionArray = functions.ToArray();
        var moduleFunctions = new HashSet<Function>(functionArray, ReferenceEqualityComparer.Instance);
        var visited = new HashSet<Function>(ReferenceEqualityComparer.Instance);
        var active = new HashSet<Function>(ReferenceEqualityComparer.Instance);
        var order = new List<Function>();

        void Visit(Function function)
        {
            if (visited.Contains(function))
            {
                return;
            }

            if (!active.Add(function))
            {
                throw new InvalidOperationException(
                    $"Recursive internal function calls are not supported: {function.Name}.");
            }

            foreach (var callee in ExprCollector.Collect(function.Body)
                         .OfType<Call>()
                         .Select(call => call.Target)
                         .OfType<Function>()
                         .Where(moduleFunctions.Contains))
            {
                Visit(callee);
            }

            active.Remove(function);
            visited.Add(function);
            order.Add(function);
        }

        foreach (var function in functionArray)
        {
            Visit(function);
        }

        return order.ToArray();
    }
}
