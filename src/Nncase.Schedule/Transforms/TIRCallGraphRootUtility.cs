// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Resolves executable TIR call-graph roots and maintains their host wrappers.
/// </summary>
public static class TIRCallGraphRootUtility
{
    /// <summary>
    /// Collects the executable TIR roots owned by <paramref name="moduleKind"/>.
    /// </summary>
    public static PrimFunction[] Collect(IRModule module, string? moduleKind = null)
    {
        if (module.Entry is PrimFunction entry)
        {
            return moduleKind is null || string.Equals(entry.ModuleKind, moduleKind, StringComparison.Ordinal)
                ? new[] { entry }
                : Array.Empty<PrimFunction>();
        }

        if (module.Entry is not Function { Role: FunctionRole.ModuleDispatch } dispatch)
        {
            throw new InvalidOperationException(
                $"TIR call graph requires a PrimFunction entry or heterogeneous ModuleDispatch entry, " +
                $"but found {module.Entry?.GetType().Name ?? "null"}.");
        }

        return ExprCollector.Collect(dispatch.Body)
            .OfType<Call>()
            .Select(call => call.Target switch
            {
                PrimFunctionWrapper wrapper => wrapper.Target,
                FunctionWrapper { Target: PrimFunctionWrapper wrapper } => wrapper.Target,
                _ => null,
            })
            .Where(function =>
                function?.Role == FunctionRole.PipelineWorker &&
                (moduleKind is null || string.Equals(function.ModuleKind, moduleKind, StringComparison.Ordinal)))
            .Cast<PrimFunction>()
            .Distinct(new ReferenceEqualityComparer<PrimFunction>())
            .ToArray();
    }

    /// <summary>
    /// Rebinds host-visible wrappers after their TIR roots are replaced.
    /// </summary>
    public static IReadOnlyDictionary<PrimFunctionWrapper, PrimFunctionWrapper> RebindWrappers(
        IRModule module,
        IReadOnlyDictionary<PrimFunction, PrimFunction> rootReplacements)
    {
        var replacements = new Dictionary<PrimFunctionWrapper, PrimFunctionWrapper>(ReferenceEqualityComparer.Instance);
        foreach (var wrapper in module.Functions.OfType<PrimFunctionWrapper>())
        {
            if (!rootReplacements.TryGetValue(wrapper.Target, out var target))
            {
                continue;
            }

            var replacement = wrapper.With(target: target);
            replacement.Metadata = wrapper.Metadata.Clone();
            replacements.Add(wrapper, replacement);
        }

        foreach (var (wrapper, replacement) in replacements)
        {
            ReplaceUtility.ReplaceAllUsesWith(wrapper, replacement);
        }

        return replacements;
    }
}
