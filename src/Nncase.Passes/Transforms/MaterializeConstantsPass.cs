// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Materializes constant expressions after alternative-producing rewrites have been extracted.
/// </summary>
public sealed class MaterializeConstantsPass : FunctionPass
{
    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        var rewriter = new Mutators.FoldConstCall(deduplicateConstants: false);
        var rewritten = (BaseFunction)rewriter.Rewrite(input);
        if (!rewriter.IsMutated && ReferenceEquals(input, rewritten))
        {
            return Task.FromResult(input);
        }

        context.IsMutated = true;
        if (!CompilerServices.InferenceType(rewritten))
        {
            throw new InvalidOperationException(
                $"Type inference failed after materializing constants in function {input.Name}.");
        }

        return Task.FromResult(rewritten);
    }
}
