// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.DependencyInjection;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase;

/// <summary>
/// Compile session.
/// </summary>
public sealed class CompileSession : IServiceProvider, IDisposable
{
    private readonly IResolverContext _serviceProvider;

    private bool _disposedValue;
    private ICompiler? _compiler;

    /// <summary>
    /// Initializes a new instance of the <see cref="CompileSession"/> class.
    /// </summary>
    /// <param name="serviceProvider">Service provider.</param>
    /// <param name="target">Target.</param>
    /// <param name="compileOptions">Compile options.</param>
    /// <param name="activeModuleKind">Optional backend module selected by this session.</param>
    /// <param name="activeFunctionRole">Optional function role selected by this session.</param>
    internal CompileSession(
        IResolverContext serviceProvider,
        ITarget target,
        CompileOptions compileOptions,
        string? activeModuleKind = null,
        FunctionRole? activeFunctionRole = null)
    {
        _serviceProvider = serviceProvider;
        Target = target;
        CompileOptions = compileOptions;
        ActiveModuleKind = activeModuleKind;
        ActiveFunctionRole = activeFunctionRole;
    }

    /// <summary>
    /// Gets target.
    /// </summary>
    public ITarget Target { get; }

    /// <summary>
    /// Gets compile options.
    /// </summary>
    public CompileOptions CompileOptions { get; }

    /// <summary>
    /// Gets the module kind selected for a target-specific compilation unit.
    /// Null denotes a whole-module, target-independent session.
    /// </summary>
    public string? ActiveModuleKind { get; }

    /// <summary>
    /// Gets the function role selected for a role-specific compilation unit.
    /// Null denotes that the session does not select functions by role.
    /// </summary>
    public FunctionRole? ActiveFunctionRole { get; }

    /// <summary>
    /// Gets compiler.
    /// </summary>
    public ICompiler Compiler => _compiler ??= this.GetRequiredService<ICompiler>();

    /// <summary>
    /// Tests whether a function belongs to this compilation unit.
    /// </summary>
    /// <param name="function">Function to test.</param>
    /// <returns>True when the function belongs to the active compilation unit.</returns>
    public bool IsFunctionActive(BaseFunction function)
    {
        if (ActiveFunctionRole is { } activeFunctionRole)
        {
            return function.Role == activeFunctionRole &&
                (ActiveModuleKind is null ||
                 string.Equals(function.ModuleKind, ActiveModuleKind, StringComparison.Ordinal));
        }

        if (ActiveModuleKind is { } activeModuleKind)
        {
            return function.Role != FunctionRole.ModuleDispatch &&
                string.Equals(function.ModuleKind, activeModuleKind, StringComparison.Ordinal);
        }

        return true;
    }

    /// <summary>
    /// Create new compile session.
    /// </summary>
    /// <param name="target">Compile target.</param>
    /// <param name="compileOptions">Compile options.</param>
    /// <param name="activeModuleKind">Optional backend module selected by this session.</param>
    /// <param name="activeFunctionRole">Optional function role selected by this session.</param>
    /// <returns>Created compile session.</returns>
    public static CompileSession Create(
        ITarget target,
        CompileOptions compileOptions,
        string? activeModuleKind = null,
        FunctionRole? activeFunctionRole = null)
    {
        var childContainer = CompilerServices.CreateScope();
        childContainer.RegisterInstance(target);
        childContainer.RegisterInstance(compileOptions);

        var session = new CompileSession(
            childContainer,
            target,
            compileOptions,
            activeModuleKind,
            activeFunctionRole);
        childContainer.RegisterInstance(session);
        return session;
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType) => _serviceProvider.GetService(serviceType);

    /// <summary>
    /// Create new pass manager.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <returns>Created pass manager.</returns>
    public IPassManager CreatePassManager(string name)
        => _serviceProvider.GetRequiredService<IPassManagerFactory>().Create(name, this);

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
    }

    private void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                _serviceProvider.Dispose();
            }

            _disposedValue = true;
        }
    }
}
