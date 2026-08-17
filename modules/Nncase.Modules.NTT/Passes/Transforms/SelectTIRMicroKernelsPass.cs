// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.Passes.Mutators;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Selects target block microkernels after semantic TIR canonicalization and
/// reserves the Shared resources required by the selected implementation.
/// </summary>
public sealed class SelectTIRMicroKernelsPass : ModulePass
{
    private readonly string _moduleKind;

    public SelectTIRMicroKernelsPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        if (CompileSession.CompileOptions.TargetOptions is not INTTTargetOptions targetOptions)
        {
            throw new InvalidOperationException(
                $"TIR microkernel selection requires {nameof(INTTTargetOptions)}.");
        }

        foreach (var function in input.Functions.OfType<PrimFunction>().Where(x => x.ModuleKind == _moduleKind))
        {
            var rewriter = new SelectorRewriter(targetOptions);
            rewriter.Rewrite(function);
            if (rewriter.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after selecting TIR microkernels in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private sealed class SelectorRewriter : ExprRewriter
    {
        private readonly INTTTargetOptions _targetOptions;
        private int _bufferIndex;

        public SelectorRewriter(INTTTargetOptions targetOptions)
            : base(visitOtherFunctions: false)
        {
            _targetOptions = targetOptions;
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
        {
            if (expr.Target is not TIR.NTT.NTTKernelOp kernelOp)
            {
                return expr;
            }

            var arguments = expr.Arguments.ToArray();
            if (arguments.Length == 0 || arguments[^1] is not None)
            {
                throw new InvalidOperationException(
                    $"TIR kernel {kernelOp.GetType().Name} must carry an initially None shared_workspace operand before microkernel selection.");
            }

            var semanticArguments = arguments[..^1];
            var selection = _targetOptions.TIRMicroKernelSelector.Select(
                new TIRMicroKernelSelectionContext(
                    kernelOp,
                    semanticArguments,
                    _targetOptions.TargetMachineModel));
            if (selection?.TransferPipeline is { } transferPipeline)
            {
                ValidateTransferPipelineContract(kernelOp, semanticArguments, selection, transferPipeline);
            }

            var workspaces = selection is null
                ? Array.Empty<BaseExpr>()
                : selection.SharedWorkspaces
                    .Select(descriptor => (BaseExpr)CreateSharedWorkspaceBuffer(kernelOp, descriptor))
                    .ToArray();
            var result = expr.With(arguments: [.. semanticArguments, TIRSharedWorkspace.Pack(workspaces)]);
            result.Metadata.TIRMicroKernel = selection;
            SetMutated();
            return result;
        }

        private static void ValidateTransferPipelineContract(
            TIR.NTT.NTTKernelOp kernelOp,
            IReadOnlyList<BaseExpr> semanticArguments,
            TIRMicroKernelSelection selection,
            TIRTransferPipelineContract contract)
        {
            foreach (var channel in contract.Channels)
            {
                foreach (var argumentIndex in channel.SourceArgumentIndices)
                {
                    if ((uint)argumentIndex >= (uint)semanticArguments.Count)
                    {
                        throw new InvalidOperationException(
                            $"TIR microkernel {selection.Family}/{selection.Variant} for " +
                            $"{kernelOp.GetType().Name} transfer channel {channel.Name} declares " +
                            $"invalid source operand {argumentIndex}.");
                    }

                    var parameter = kernelOp.Parameters[argumentIndex];
                    var effect = kernelOp.GetMemoryEffect(parameter);
                    if (MemoryEffectUtility.GetPhysicalBufferAccessMode(effect) != MemoryAccessMode.Read)
                    {
                        throw new InvalidOperationException(
                            $"TIR microkernel {selection.Family}/{selection.Variant} for " +
                            $"{kernelOp.GetType().Name} transfer channel {channel.Name} declares " +
                            $"source operand {argumentIndex} ({parameter.Name}) with non-read-only " +
                            $"memory effect {effect.Mode}.");
                    }
                }

                foreach (var workspaceIndex in channel.SharedWorkspaceIndices)
                {
                    if ((uint)workspaceIndex >= (uint)selection.SharedWorkspaces.Length)
                    {
                        throw new InvalidOperationException(
                            $"TIR microkernel {selection.Family}/{selection.Variant} for " +
                            $"{kernelOp.GetType().Name} transfer channel {channel.Name} declares " +
                            $"invalid Shared workspace {workspaceIndex}.");
                    }
                }
            }
        }

        private TIR.Buffer CreateSharedWorkspaceBuffer(
            TIR.NTT.NTTKernelOp kernelOp,
            TIRSharedWorkspaceDescriptor descriptor)
        {
            if (string.IsNullOrWhiteSpace(descriptor.Name))
            {
                throw new InvalidOperationException(
                    $"TIR microkernel {kernelOp.GetType().Name} declared an unnamed shared workspace.");
            }

            if (descriptor.Type.Shape is not RankedShape shape)
            {
                throw new InvalidOperationException(
                    $"TIR microkernel {kernelOp.GetType().Name} shared workspace {descriptor.Name} must have a ranked shape.");
            }

            if (descriptor.AlignmentBytes <= 0 ||
                (descriptor.AlignmentBytes & (descriptor.AlignmentBytes - 1)) != 0 ||
                descriptor.AlignmentBytes < descriptor.Type.DType.SizeInBytes)
            {
                throw new InvalidOperationException(
                    $"TIR microkernel {kernelOp.GetType().Name} shared workspace {descriptor.Name} has invalid " +
                    $"alignment {descriptor.AlignmentBytes} for {descriptor.Type.DType}.");
            }

            (var size, var strides) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(
                descriptor.Type,
                distributedType: null);
            if (CompilerServices.GetMaxShape([size])[0] <= 0)
            {
                throw new InvalidOperationException(
                    $"TIR microkernel {kernelOp.GetType().Name} shared workspace {descriptor.Name} must contain at least one byte.");
            }

            var storage = new PhysicalBuffer(
                descriptor.AlignmentBytes,
                size,
                MemoryLocation.Shared);
            return new TIR.Buffer(
                $"{kernelOp.GetType().Name}_{descriptor.Name}_shared_{_bufferIndex++}",
                descriptor.Type.DType,
                new MemSpan(storage),
                shape.Dimensions.ToArray(),
                strides,
                distributedType: null);
        }
    }
}
