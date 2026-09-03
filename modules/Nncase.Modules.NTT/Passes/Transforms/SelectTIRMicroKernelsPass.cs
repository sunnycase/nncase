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

        var functions = input.Functions
            .OfType<PrimFunction>()
            .Where(x => x.ModuleKind == _moduleKind && CompileSession.IsFunctionActive(x))
            .ToArray();
        var alignmentRequirements = new OperandAlignmentRequirements();
        foreach (var function in functions)
        {
            var rewriter = new SelectorRewriter(
                targetOptions,
                alignmentRequirements,
                function.Name);
            rewriter.Rewrite(function);
            if (rewriter.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after selecting TIR microkernels in {function.Name}.");
            }
        }

        if (alignmentRequirements.HasRequirements)
        {
            foreach (var function in functions)
            {
                new PhysicalBufferAlignmentRewriter(alignmentRequirements).Rewrite(function);
            }
        }

        return Task.FromResult(input);
    }

    private sealed class SelectorRewriter : ExprRewriter
    {
        private readonly INTTTargetOptions _targetOptions;
        private readonly OperandAlignmentRequirements _alignmentRequirements;
        private readonly string _functionName;
        private int _bufferIndex;

        public SelectorRewriter(
            INTTTargetOptions targetOptions,
            OperandAlignmentRequirements alignmentRequirements,
            string functionName)
            : base(visitOtherFunctions: false)
        {
            _targetOptions = targetOptions;
            _alignmentRequirements = alignmentRequirements;
            _functionName = functionName;
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
            TIRMicroKernelSelection? selection;
            try
            {
                selection = _targetOptions.TIRMicroKernelSelector.Select(
                    new TIRMicroKernelSelectionContext(
                        kernelOp,
                        semanticArguments,
                        _targetOptions.TargetMachineModel));
            }
            catch (Exception exception) when (exception is InvalidOperationException or NotSupportedException)
            {
                throw new NotSupportedException(
                    $"Failed to select a TIR microkernel for {kernelOp.GetType().Name} " +
                    $"in prim function {_functionName}: {exception.Message}",
                    exception);
            }
            if (selection?.TransferPipeline is { } transferPipeline)
            {
                ValidateTransferPipelineContract(kernelOp, semanticArguments, selection, transferPipeline);
                RecordTransferSourceAlignments(semanticArguments, transferPipeline);
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

        private void RecordTransferSourceAlignments(
            IReadOnlyList<BaseExpr> semanticArguments,
            TIRTransferPipelineContract contract)
        {
            foreach (var channel in contract.Channels)
            {
                if (channel.SourceAlignmentBytes == 1)
                {
                    continue;
                }

                foreach (var argumentIndex in channel.SourceArgumentIndices)
                {
                    _alignmentRequirements.Add(
                        semanticArguments[argumentIndex],
                        channel.SourceAlignmentBytes,
                        channel.Name);
                }
            }
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
                    var effect = kernelOp.GetMemoryEffect(parameter, semanticArguments);
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

            foreach (var workspaceIndex in contract.ConsumerSharedWorkspaceIndices)
            {
                if ((uint)workspaceIndex >= (uint)selection.SharedWorkspaces.Length)
                {
                    throw new InvalidOperationException(
                        $"TIR microkernel {selection.Family}/{selection.Variant} for " +
                        $"{kernelOp.GetType().Name} declares invalid consumer Shared workspace {workspaceIndex}.");
                }
            }

            var ownedWorkspaceIndices = contract.SharedWorkspaceIndices
                .Concat(contract.ConsumerSharedWorkspaceIndices)
                .OrderBy(index => index)
                .ToArray();
            if (!ownedWorkspaceIndices.SequenceEqual(Enumerable.Range(0, selection.SharedWorkspaces.Length)))
            {
                throw new InvalidOperationException(
                    $"TIR microkernel {selection.Family}/{selection.Variant} for {kernelOp.GetType().Name} " +
                    "must assign every Shared workspace to one transfer channel or the consumer role.");
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

    private sealed class OperandAlignmentRequirements
    {
        private readonly Dictionary<PhysicalBuffer, int> _physicalBuffers =
            new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<Const, int> _constants =
            new(ReferenceEqualityComparer.Instance);

        public bool HasRequirements => _physicalBuffers.Count != 0;

        public void Add(BaseExpr operand, int alignmentBytes, string channelName)
        {
            switch (operand)
            {
                case TIR.Buffer buffer:
                    Add(buffer, alignmentBytes, channelName);
                    return;
                case IR.Tuple tuple:
                    foreach (var field in tuple.Fields)
                    {
                        Add(field, alignmentBytes, channelName);
                    }

                    return;
                default:
                    throw new InvalidOperationException(
                        $"Transfer channel {channelName} requires {alignmentBytes}-byte aligned " +
                        $"TIR buffer operands, got {operand.GetType().Name}.");
            }
        }

        public int GetRequiredAlignment(PhysicalBuffer buffer)
        {
            var alignment = _physicalBuffers.TryGetValue(buffer, out var physicalAlignment)
                ? physicalAlignment
                : 1;
            if (TryGetAddressedConst(buffer, out var constValue) &&
                _constants.TryGetValue(constValue, out var constAlignment))
            {
                alignment = Math.Max(alignment, constAlignment);
            }

            return alignment;
        }

        private void Add(TIR.Buffer buffer, int alignmentBytes, string channelName)
        {
            if (!Dimension.TryDivExactly(buffer.MemSpan.Start, alignmentBytes, out _))
            {
                throw new InvalidOperationException(
                    $"Transfer channel {channelName} requires {alignmentBytes}-byte aligned " +
                    $"buffer views, but {buffer.Name} starts at byte offset {buffer.MemSpan.Start}.");
            }

            _physicalBuffers[buffer.MemSpan.Buffer] = Math.Max(
                _physicalBuffers.GetValueOrDefault(buffer.MemSpan.Buffer, 1),
                alignmentBytes);
            if (TryGetAddressedConst(buffer.MemSpan.Buffer, out var constValue))
            {
                _constants[constValue] = Math.Max(
                    _constants.GetValueOrDefault(constValue, 1),
                    alignmentBytes);
            }
        }

        private static bool TryGetAddressedConst(
            PhysicalBuffer buffer,
            [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out Const? constValue)
        {
            if (buffer.Start is Call { Target: IR.Buffers.AddressOf } addressOf &&
                addressOf[IR.Buffers.AddressOf.Input] is Const addressedConst)
            {
                constValue = addressedConst;
                return true;
            }

            constValue = null;
            return false;
        }
    }

    private sealed class PhysicalBufferAlignmentRewriter : ExprRewriter
    {
        private readonly OperandAlignmentRequirements _requirements;

        public PhysicalBufferAlignmentRewriter(OperandAlignmentRequirements requirements)
            : base(visitOtherFunctions: false)
        {
            _requirements = requirements;
        }

        protected override BaseExpr RewriteLeafPhysicalBuffer(PhysicalBuffer expr)
        {
            var requiredAlignment = _requirements.GetRequiredAlignment(expr);
            return requiredAlignment > expr.Alignment
                ? expr.With(alignment: requiredAlignment)
                : expr;
        }
    }
}
