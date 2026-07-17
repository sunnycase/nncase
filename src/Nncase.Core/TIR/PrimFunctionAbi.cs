// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// One ordered logical PrimFunction result and the ABI storage that owns its bytes.
/// </summary>
public sealed record PrimFunctionResultBinding(Expr Value, IVar Storage)
{
    /// <summary>
    /// Gets the logical result type, including distributed metadata when present.
    /// </summary>
    public IRType Type => Value switch
    {
        Buffer buffer => buffer.DistributedType ?? buffer.CheckedType,
        _ => Value.CheckedType,
    };
}

/// <summary>
/// Read-only prim function ABI view computed from parameters and explicit results.
/// </summary>
public sealed record PrimFunctionAbiView(
    IReadOnlyList<IVar> Inputs,
    IReadOnlyList<BufferVar> OutputParameters,
    IReadOnlyList<PrimFunctionResultBinding> Results,
    IReadOnlyList<BufferVar> Workspaces)
{
    /// <summary>
    /// Gets the logical runtime parameter types. Caller-allocated outputs and
    /// workspaces are physical ABI details and are not model inputs.
    /// </summary>
    public IReadOnlyList<IRType> RuntimeParameterTypes => Inputs.Select(input => ((BaseExpr)input).CheckedType).ToArray();

    /// <summary>
    /// Gets the logical runtime return type represented by the explicit result bindings.
    /// </summary>
    public IRType RuntimeReturnType => Results.Count switch
    {
        0 => TupleType.Void,
        1 => Results[0].Type,
        _ => new TupleType(Results.Select(result => result.Type).ToArray()),
    };
}

/// <summary>
/// Prim function ABI helpers.
/// </summary>
public static class PrimFunctionAbi
{
    private enum AbiParameterPhase
    {
        Inputs,
        Outputs,
        Workspaces,
    }

    /// <summary>
    /// Builds an ABI view from parameters and explicit logical results.
    /// </summary>
    public static PrimFunctionAbiView GetAbiView(this PrimFunction function)
    {
        var inputs = new List<IVar>();
        var outputs = new List<BufferVar>();
        var workspaces = new List<BufferVar>();
        var phase = AbiParameterPhase.Inputs;

        foreach (var parameter in function.Parameters)
        {
            if (parameter is not BufferVar bufferVar)
            {
                if (phase != AbiParameterPhase.Inputs)
                {
                    throw new InvalidOperationException($"PrimFunction {function.Name} has regular parameter {parameter.Name} after output/workspace parameters.");
                }

                inputs.Add(parameter);
                continue;
            }

            switch (bufferVar.Role)
            {
                case BufferVarRole.Input:
                case BufferVarRole.InOut:
                    if (phase != AbiParameterPhase.Inputs)
                    {
                        throw new InvalidOperationException($"PrimFunction {function.Name} has input parameter {bufferVar.Name} after output/workspace parameters.");
                    }

                    inputs.Add(bufferVar);
                    break;
                case BufferVarRole.Output:
                    if (phase == AbiParameterPhase.Workspaces)
                    {
                        throw new InvalidOperationException($"PrimFunction {function.Name} has output parameter {bufferVar.Name} after workspace parameters.");
                    }

                    phase = AbiParameterPhase.Outputs;
                    outputs.Add(bufferVar);
                    break;
                case BufferVarRole.Workspace:
                    phase = AbiParameterPhase.Workspaces;
                    workspaces.Add(bufferVar);
                    break;
                default:
                    throw new InvalidOperationException($"PrimFunction {function.Name} has unsupported BufferVar role {bufferVar.Role} on {bufferVar.Name}.");
            }
        }

        var storages = inputs.Concat<IVar>(outputs).ToHashSet(ReferenceEqualityComparer.Instance);
        var results = new PrimFunctionResultBinding[function.Results.Values.Length];
        for (var resultIndex = 0; resultIndex < results.Length; resultIndex++)
        {
            var value = function.Results.Values[resultIndex];
            var storage = ResolveResultStorage(function, resultIndex, value);
            if (!storages.Contains(storage))
            {
                throw new InvalidOperationException(
                    $"PrimFunction {function.Name} result {resultIndex} is backed by {storage.Name}, which is not an input or caller-allocated output parameter.");
            }

            results[resultIndex] = new PrimFunctionResultBinding(value, storage);
        }

        return new PrimFunctionAbiView(inputs, outputs, results, workspaces);
    }

    private static IVar ResolveResultStorage(PrimFunction function, int resultIndex, Expr value)
    {
        if (value is IVar variable)
        {
            return variable;
        }

        if (value is Buffer buffer && buffer.MemSpan.Buffer.Start is IVar storage)
        {
            return storage;
        }

        throw new InvalidOperationException(
            $"PrimFunction {function.Name} result {resultIndex} must be a BufferVar or a TIR.Buffer view backed by an ABI parameter, got {value.GetType().Name}.");
    }
}
