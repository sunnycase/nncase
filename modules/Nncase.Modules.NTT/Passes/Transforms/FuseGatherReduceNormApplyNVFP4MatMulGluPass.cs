// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Passes.Mutators;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses a single-use distributed normalization apply into the activation load
/// of an immediately following NVFP4 MatMulGlu.
/// </summary>
public sealed class FuseGatherReduceNormApplyNVFP4MatMulGluPass : ModulePass
{
    private readonly string _moduleKind;

    public FuseGatherReduceNormApplyNVFP4MatMulGluPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var function in input.Functions
                     .OfType<PrimFunction>()
                     .Where(function =>
                         function.ModuleKind == _moduleKind &&
                         CompileSession.IsFunctionActive(function)))
        {
            var storageUseCounts = CollectStorageCallUseCounts(function.Body);
            var rewriter = new FusionRewriter(storageUseCounts);
            rewriter.Rewrite(function);
            if (rewriter.IsMutated && !CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after fusing GatherReduceNormApply with " +
                    $"NVFP4MatMulGlu in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private static IReadOnlyDictionary<PhysicalBuffer, int> CollectStorageCallUseCounts(BaseExpr body)
    {
        var counts = new Dictionary<PhysicalBuffer, int>(ReferenceEqualityComparer.Instance);
        foreach (var buffer in ExprCollector.Collect(body)
                     .OfType<Call>()
                     .SelectMany(call => call.Arguments.ToArray().OfType<TIR.Buffer>()))
        {
            counts.TryGetValue(buffer.MemSpan.Buffer, out var count);
            counts[buffer.MemSpan.Buffer] = count + 1;
        }

        return counts;
    }

    private sealed class FusionRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<PhysicalBuffer, int> _storageUseCounts;

        public FusionRewriter(IReadOnlyDictionary<PhysicalBuffer, int> storageUseCounts)
            : base(visitOtherFunctions: false)
        {
            _storageUseCounts = storageUseCounts;
        }

        protected override BaseExpr RewriteLeafSequential(Sequential expr)
        {
            var source = expr.Fields.ToArray();
            var result = new List<Expr>(source.Length);
            for (var index = 0; index < source.Length; index++)
            {
                var consumerIndex = index + 1;
                while (consumerIndex < source.Length && IsNop(source[consumerIndex]))
                {
                    consumerIndex++;
                }

                if (consumerIndex < source.Length &&
                    TryFuse(source[index], source[consumerIndex], out var fused))
                {
                    result.Add(fused);
                    index = consumerIndex;
                    SetMutated();
                    continue;
                }

                result.Add(source[index]);
            }

            return result.Count == source.Length
                ? expr
                : expr.With(fields: result.ToArray());
        }

        private static bool IsNop(Expr expression)
            => expression is Call { Target: Nop };

        private bool TryFuse(Expr producer, Expr consumer, out Expr fused)
        {
            fused = null!;
            if (producer is not Call
                {
                    Target: TIR.NTT.GatherReduceNormApply norm,
                } normCall ||
                consumer is not Call
                {
                    Target: TIR.NTT.NVFP4MatMulGlu matMulGlu,
                } matMulGluCall)
            {
                return false;
            }

            var normArguments = normCall.Arguments.ToArray();
            var matMulArguments = matMulGluCall.Arguments.ToArray();
            if (normArguments.Length < 5 || matMulArguments.Length < 10 ||
                normArguments[0] is not TIR.Buffer partialStats ||
                normArguments[1] is not TIR.Buffer input ||
                normArguments[2] is not TIR.Buffer normScale ||
                normArguments[3] is not TIR.Buffer normBias ||
                normArguments[4] is not TIR.Buffer normOutput ||
                matMulArguments[0] is not TIR.Buffer matMulInput ||
                matMulInput.DistributedType is not DistributedType normalizedInputType ||
                !ReferenceEquals(normOutput.MemSpan.Buffer, matMulInput.MemSpan.Buffer) ||
                !normOutput.MemSpan.Start.Equals(matMulInput.MemSpan.Start) ||
                !normOutput.MemSpan.Size.Equals(matMulInput.MemSpan.Size) ||
                !_storageUseCounts.TryGetValue(normOutput.MemSpan.Buffer, out var useCount) ||
                useCount != 2)
            {
                return false;
            }

            if (!CanReadDistributedInputDirectly(norm, input, matMulInput))
            {
                return false;
            }

            if (!BufferViewUtility.TryCreateCanonicalGlobalReadAlias(
                    normScale,
                    $"{normScale.Name}_canonical_global",
                    out var canonicalNormScale,
                    out _) ||
                !BufferViewUtility.TryCreateCanonicalGlobalReadAlias(
                    normBias,
                    $"{normBias.Name}_canonical_global",
                    out var canonicalNormBias,
                    out _))
            {
                return false;
            }

            fused = TIR.F.NTT.GatherReduceNormApplyNVFP4MatMulGlu(
                    partialStats,
                    input,
                    canonicalNormScale,
                    canonicalNormBias,
                    (Expr)matMulArguments[1],
                    (Expr)matMulArguments[2],
                    (Expr)matMulArguments[3],
                    (Expr)matMulArguments[4],
                    (Expr)matMulArguments[5],
                    (Expr)matMulArguments[6],
                    (Expr)matMulArguments[7],
                    (Expr)matMulArguments[8],
                    (Expr)matMulArguments[9],
                    norm.InStatsType,
                    norm.OutStatsType,
                    normalizedInputType,
                    norm.Axis,
                    norm.Epsilon,
                    norm.UseMean,
                    norm.HasBias,
                    matMulGlu.GluType,
                    matMulGlu.GroupSize)
                .InheritMetaData(matMulGluCall);
            return true;
        }

        private static bool CanReadDistributedInputDirectly(
            TIR.NTT.GatherReduceNormApply norm,
            TIR.Buffer input,
            TIR.Buffer matMulInput)
        {
            if (norm.InStatsType.Partial is not { Op: ReduceOp.Sum } partial ||
                partial.Axes.Count == 0 ||
                norm.OutStatsType.Partial is not null ||
                !norm.InStatsType.Placement.Equals(norm.OutStatsType.Placement) ||
                input.DistributedType is not { } inputType ||
                matMulInput.DistributedType is not { } matMulInputType ||
                input.DistributedStorageKind is not (
                    DistributedBufferStorageKind.CompactLocal or
                    DistributedBufferStorageKind.CompactPerOwner) ||
                input.MemSpan.Buffer.Location is not (
                    MemoryLocation.Data or
                    MemoryLocation.ChipLocalData))
            {
                return false;
            }

            if (!inputType.Placement.Equals(matMulInputType.Placement) ||
                !inputType.TensorType.Equals(matMulInputType.TensorType) ||
                matMulInputType.AxisPolicies.Any(policy => policy is not SBPBroadCast) ||
                matMulInputType.Partial is not null)
            {
                return false;
            }

            var rank = inputType.TensorType.Shape.Rank;
            var axis = norm.Axis < 0 ? norm.Axis + rank : norm.Axis;
            return rank == 2 && axis == rank - 1;
        }
    }
}
