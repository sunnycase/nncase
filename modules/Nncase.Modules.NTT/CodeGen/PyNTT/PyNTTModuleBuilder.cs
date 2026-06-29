// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.CodeGen.PyNTT;

/// <summary>
/// PyNTT module builder.
/// </summary>
public sealed class PyNTTModuleBuilder : IModuleBuilder
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PyNTTModuleBuilder"/> class.
    /// </summary>
    /// <param name="moduleKind">Module kind.</param>
    /// <param name="options">Compile options.</param>
    public PyNTTModuleBuilder(string moduleKind, CompileOptions options)
    {
        ModuleKind = moduleKind;
        CompileOptions = options;
    }

    /// <inheritdoc/>
    public string ModuleKind { get; }

    /// <summary>
    /// Gets compile options.
    /// </summary>
    public CompileOptions CompileOptions { get; }

    /// <inheritdoc/>
    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var linkableFunctions = functions
            .Select((function, index) => new PyNTTFunctionBuilder((uint)index, CompileOptions).Build(function))
            .ToArray();
        return new PyNTTLinkableModule(ModuleKind, linkableFunctions, CompileOptions);
    }
}
