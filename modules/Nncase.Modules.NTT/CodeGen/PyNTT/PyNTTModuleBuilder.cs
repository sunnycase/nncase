// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

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
        var primFunctions = functions.Select(RequirePrimFunction).ToArray();
        var linkableFunctions = primFunctions
            .Select((function, index) => new PyNTTFunctionBuilder((uint)index, CompileOptions).Build(function))
            .ToArray();
        return new PyNTTLinkableModule(ModuleKind, linkableFunctions, CompileOptions);
    }

    private static PrimFunction RequirePrimFunction(BaseFunction function)
    {
        if (function is PrimFunction primFunction)
        {
            return primFunction;
        }

        throw new NotSupportedException($"PyNTT module builder expects lowered PrimFunction inputs, got {function.GetType().Name} {function.Name}. Run TIR selection and RemoveFunctionWrapperPass before PyNTT codegen.");
    }
}
