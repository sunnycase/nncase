// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

[Flags]
public enum MemorySynchronizationScopes
{
    None = 0,
    Block = 1 << 0,
    Chip = 1 << 1,
    All = Block | Chip,
}

/// <summary>
/// Materializes synchronization implied by structured TIR memory effects.
/// </summary>
public sealed class PlanMemorySynchronizationPass : ModulePass
{
    private readonly string _moduleKind;
    private readonly MemorySynchronizationScopes _materializedScopes;

    public PlanMemorySynchronizationPass(
        string moduleKind,
        MemorySynchronizationScopes materializedScopes)
    {
        _moduleKind = moduleKind;
        _materializedScopes = materializedScopes;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var functions = input.Functions
            .Select((function, index) => (Function: function as PrimFunction, Index: index))
            .Where(item => item.Function is { ModuleKind: var moduleKind } && moduleKind == _moduleKind)
            .Select(item => (Function: item.Function!, item.Index))
            .ToArray();
        var analyzer = new MemoryEffectAnalyzer(functions.Select(item => item.Function));
        analyzer.AnalyzeAll();

        var rewrittenFunctions = functions
            .Select(item => (Function: new MemorySynchronizationPlanner(analyzer, _materializedScopes).Rewrite(item.Function), item.Index))
            .ToArray();
        foreach (var (function, index) in rewrittenFunctions)
        {
            if (!CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException($"Type inference failed after memory synchronization planning in {function.Name}.");
            }

            input.Replace(index, function);
        }

        return Task.FromResult(input);
    }
}
