// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public abstract class NTTKernelOp : Op, IOpMemoryEffectProvider
{
    private ParameterInfo[]? _kernelParameters;

    /// <summary>
    /// Gets the final operand that carries zero or more target-private shared
    /// workspace buffers in canonical None/value/tuple form.
    /// </summary>
    public ParameterInfo SharedWorkspaceParameter => Parameters[^1];

    public override IReadOnlyList<ParameterInfo> Parameters
    {
        get
        {
            if (_kernelParameters is not null)
            {
                return _kernelParameters;
            }

            var semanticParameters = base.Parameters;
            var missing = semanticParameters.Where(parameter => parameter.MemoryEffect is null).ToArray();
            if (missing.Length != 0)
            {
                throw new InvalidOperationException(
                    $"{GetType().Name} must declare a memory effect for every operand. Missing: {string.Join(", ", missing.Select(parameter => parameter.Name))}.");
            }

            _kernelParameters =
            [
                .. semanticParameters,
                new ParameterInfo(
                    GetType(),
                    semanticParameters.Count,
                    "shared_workspace",
                    memoryEffect: MemoryEffect.ReadWrite),
            ];
            return _kernelParameters;
        }
    }

    /// <summary>
    /// Creates a well-formed semantic TIR call with no shared workspace. Target
    /// TIR Selection replaces None after choosing a concrete microkernel.
    /// </summary>
    public Call CreateCall(params BaseExpr[] arguments)
        => new(this, [.. arguments, None.Default]);

    /// <inheritdoc/>
    public virtual MemoryEffect GetMemoryEffect(
        ParameterInfo parameter,
        IReadOnlyList<BaseExpr> arguments)
        => parameter.MemoryEffect
            ?? throw new ArgumentOutOfRangeException(
                nameof(parameter),
                parameter,
                $"Unknown {GetType().Name} operand.");
}
