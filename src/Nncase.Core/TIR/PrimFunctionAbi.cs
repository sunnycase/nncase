// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// Read-only prim function ABI view computed from the parameter list.
/// </summary>
public sealed record PrimFunctionAbiView(
    IReadOnlyList<IVar> Inputs,
    IReadOnlyList<BufferVar> Outputs,
    IReadOnlyList<BufferVar> Workspaces);

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
    /// Builds an ABI view from <see cref="PrimFunction.Parameters"/>.
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
                    if (phase != AbiParameterPhase.Inputs)
                    {
                        throw new InvalidOperationException($"PrimFunction {function.Name} has input parameter {bufferVar.Name} after output/workspace parameters.");
                    }

                    inputs.Add(bufferVar);
                    break;
                case BufferVarRole.InOut:
                    if (phase != AbiParameterPhase.Inputs)
                    {
                        throw new InvalidOperationException($"PrimFunction {function.Name} has input/output parameter {bufferVar.Name} after output/workspace parameters.");
                    }

                    inputs.Add(bufferVar);
                    outputs.Add(bufferVar);
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

        return new PrimFunctionAbiView(inputs, outputs, workspaces);
    }
}
