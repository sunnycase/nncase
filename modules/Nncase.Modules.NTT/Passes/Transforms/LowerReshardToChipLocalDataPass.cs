// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Lowers copy-like reshard operations to chip-local sharded buffer views before bufferization.
/// </summary>
public sealed class LowerReshardToChipLocalDataPass : ModulePass
{
    private readonly string _moduleKind;

    public LowerReshardToChipLocalDataPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var functions = input.Functions
            .Select((function, index) => (Function: function, Index: index))
            .Where(item => item.Function is PrimFunction { ModuleKind: var moduleKind } && moduleKind == _moduleKind)
            .ToArray();

        foreach (var (function, index) in functions)
        {
            var primFunction = (PrimFunction)function;
            var rewritten = RewritePrimFunction(primFunction);
            if (!ReferenceEquals(primFunction, rewritten))
            {
                input.Replace(index, rewritten);
            }
        }

        return Task.FromResult(input);
    }

    private static PrimFunction RewritePrimFunction(PrimFunction function)
    {
        var post = (PrimFunction)new RemoveNop().Rewrite(function);
        post = (PrimFunction)new LowerReshardToChipLocalData().Rewrite(post);
        post = (PrimFunction)new RemoveNop().Rewrite(post);

        if (!ReferenceEquals(function, post) && !CompilerServices.InferenceType(post))
        {
            throw new InvalidOperationException($"Type inference failed after lowering reshard to chip-local data in {function.Name}.");
        }

        return post;
    }
}
