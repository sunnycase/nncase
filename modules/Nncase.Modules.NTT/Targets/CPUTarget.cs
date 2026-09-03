// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CodeGen.NTT;
using Nncase.Passes;

namespace Nncase.Targets;

/// <summary>
/// Target for CPU.
/// </summary>
public class CPUTarget : NTTTarget
{
    public const string Kind = "cpu";

    protected override INTTModuleCompiler NTTModuleCompiler { get; } = new CPUModuleCompiler();

    /// <inheritdoc/>
    public override void RegisterTargetDependentPass(IPassManager passManager, CompileOptions options)
    {
        ValidateOptions(options);
    }

    internal static NTTTargetOptions ValidateOptions(CompileOptions options)
    {
        var targetOptions = options.TargetOptions as NTTTargetOptions
            ?? throw new InvalidOperationException(
                $"CPU NTT requires {nameof(NTTTargetOptions)}, got " +
                $"{options.TargetOptions?.GetType().Name ?? "null"}.");
        if (targetOptions.HierarchyLevels.Length == 0 ||
            targetOptions.HierarchyLevels.Any(level => level != 'b'))
        {
            throw new InvalidOperationException(
                "CPU NTT supports only physical block hierarchy levels. " +
                $"Logical axes may form a mesh, but HierarchyLevels must contain only 'b'; got '{targetOptions.HierarchyLevels}'.");
        }

        return targetOptions;
    }
}
