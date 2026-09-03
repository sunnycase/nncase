// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// The Kmodel Builder.
/// </summary>
public sealed class ModelBuilder : IModelBuilder
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ModelBuilder"/> class.
    /// default ctor.
    /// </summary>
    public ModelBuilder(ITarget target, CompileOptions compileOptions)
    {
        Target = target;
        CompileOptions = compileOptions;
    }

    /// <summary>
    /// Gets get the Target.
    /// </summary>
    public ITarget Target { get; }

    /// <summary>
    /// Gets get the CompileOptions.
    /// </summary>
    public CompileOptions CompileOptions { get; }

    public ILinkedModel Build(IRModule module)
    {
        var functionsByKind = module.Functions
            .Where(function => function is not PrimFunctionWrapper { Target.Role: FunctionRole.PipelineWorker })
            .GroupBy(x => x.ModuleKind)
            .ToList();
        var linkableModules = functionsByKind.Select(x => GetModuleBuilder(x.Key).Build(x.ToList())).ToList();
        var functionIds = MakeFunctionsIds(linkableModules);
        if (DumpScope.Current.IsEnabled(DumpFlags.CodeGen))
        {
            CodeGenDumper.DumpIdMap(functionIds);
        }

        var linkContext = new LinkContext(functionIds);
        var linkedModules = new List<ILinkedModule>(linkableModules.Count);
        foreach (var linkableModule in TopologicallyOrderModules(linkableModules))
        {
            var linkedModule = linkableModule.Link(linkContext);
            linkContext.RegisterLinkedModule(linkedModule);
            linkedModules.Add(linkedModule);
        }

        linkedModules = linkableModules
            .Select(module => linkedModules.Single(linked => linked.ModuleKind == module.ModuleKind))
            .ToList();
        var entryFunctionId = module.Entry == null ? null : functionIds[module.Entry];
        return new LinkedModel(entryFunctionId, linkedModules);
    }

    private static IReadOnlyList<ILinkableModule> TopologicallyOrderModules(IReadOnlyList<ILinkableModule> modules)
    {
        var modulesByKind = modules.ToDictionary(module => module.ModuleKind, StringComparer.Ordinal);
        foreach (var module in modules)
        {
            var missing = module.DependencyModuleKinds.FirstOrDefault(kind => !modulesByKind.ContainsKey(kind));
            if (missing is not null)
            {
                throw new InvalidOperationException(
                    $"Module {module.ModuleKind} depends on missing module {missing}.");
            }
        }

        var result = new List<ILinkableModule>(modules.Count);
        var states = new Dictionary<string, int>(StringComparer.Ordinal);
        void Visit(ILinkableModule module)
        {
            if (states.TryGetValue(module.ModuleKind, out var state))
            {
                if (state == 1)
                {
                    throw new InvalidOperationException(
                        $"Module link dependency graph contains a cycle involving {module.ModuleKind}.");
                }

                return;
            }

            states.Add(module.ModuleKind, 1);
            foreach (var dependency in module.DependencyModuleKinds.OrderBy(kind => kind, StringComparer.Ordinal))
            {
                Visit(modulesByKind[dependency]);
            }

            states[module.ModuleKind] = 2;
            result.Add(module);
        }

        foreach (var module in modules)
        {
            Visit(module);
        }

        return result;
    }

    private IModuleBuilder GetModuleBuilder(string kind)
    {
        var moduleOptions = Target.GetModuleCompileOptions(kind, CompileOptions);
        return Target.GetModuleCompiler(kind).CreateModuleBuilder(moduleOptions);
    }

    private Dictionary<BaseFunction, FunctionId> MakeFunctionsIds(IReadOnlyList<ILinkableModule> modules)
    {
        var ids = new Dictionary<BaseFunction, FunctionId>(ReferenceEqualityComparer.Instance);
        uint moduleId = 0;
        foreach (var mod in modules)
        {
            uint funcId = 0;
            foreach (var func in mod.PublicFunctions)
            {
                ids.Add(func.SourceFunction, new FunctionId(funcId++, moduleId));
            }

            moduleId++;
        }

        return ids;
    }
}

internal class LinkContext : ILinkContext
{
    private readonly IDictionary<BaseFunction, FunctionId> _functionIds;
    private readonly Dictionary<string, ILinkedModule> _linkedModules = new(StringComparer.Ordinal);

    public LinkContext(IDictionary<BaseFunction, FunctionId> functionIds)
    {
        _functionIds = functionIds;
    }

    public FunctionId GetFunctionId(BaseFunction function)
    {
        return _functionIds[function];
    }

    public ILinkedModule GetLinkedModule(string moduleKind)
    {
        if (!_linkedModules.TryGetValue(moduleKind, out var module))
        {
            throw new InvalidOperationException(
                $"Module {moduleKind} has not been linked. Declare it as a link dependency.");
        }

        return module;
    }

    public void RegisterLinkedModule(ILinkedModule module)
    {
        if (!_linkedModules.TryAdd(module.ModuleKind, module))
        {
            throw new InvalidOperationException($"Module {module.ModuleKind} was linked more than once.");
        }
    }
}
