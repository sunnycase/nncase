// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses a single-use paged-attention merge into the packed matmul that
/// consumes its physical output. The partial-attention grid barrier remains
/// explicit and the packed-matmul transfer pipeline is selected afterwards.
/// </summary>
public sealed class FusePagedAttentionMergePackedMatMulPass : ModulePass
{
    private readonly string _moduleKind;

    public FusePagedAttentionMergePackedMatMulPass(string moduleKind)
    {
        _moduleKind = moduleKind;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var function in input.Functions.OfType<PrimFunction>().Where(function => function.ModuleKind == _moduleKind))
        {
            var calls = ExprCollector.Collect(function.Body).OfType<Call>().ToArray();
            var uses = CollectBufferRegionCallUses(calls);
            var replacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
            foreach (var matmulCall in calls.Where(call => call.Target is TIR.NTT.PackedMatMul))
            {
                if (TryFuse(matmulCall, uses, out var mergeCall, out var fused))
                {
                    replacements[mergeCall] = TIR.T.Nop().InheritMetaData(mergeCall);
                    var merge = (TIR.NTT.PagedAttentionMerge)mergeCall.Target;
                    replacements[matmulCall] = new Sequential(
                        TIR.F.NTT.Barrier(
                            TIR.NTT.BarrierScope.Chip,
                            new IRArray<int>(new[] { merge.SplitHierarchyAxis })),
                        fused.InheritMetaData(matmulCall));
                }
            }

            if (replacements.Count == 0)
            {
                continue;
            }

            new ReplacementRewriter(replacements).Rewrite(function);
            if (!CompilerServices.InferenceType(function))
            {
                throw new InvalidOperationException(
                    $"Type inference failed after fusing PagedAttentionMerge with PackedMatMul in {function.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private static IReadOnlyDictionary<TIR.Buffer, Call[]> CollectBufferRegionCallUses(IEnumerable<Call> calls)
    {
        var uses = new Dictionary<TIR.Buffer, HashSet<Call>>(ExactBufferRegionComparer.Instance);
        foreach (var call in calls)
        {
            foreach (var buffer in call.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                if (!uses.TryGetValue(buffer, out var bufferUses))
                {
                    bufferUses = new HashSet<Call>(ReferenceEqualityComparer.Instance);
                    uses.Add(buffer, bufferUses);
                }

                bufferUses.Add(call);
            }
        }

        var result = new Dictionary<TIR.Buffer, Call[]>(ExactBufferRegionComparer.Instance);
        foreach (var (buffer, bufferUses) in uses)
        {
            result.Add(buffer, bufferUses.ToArray());
        }

        return result;
    }

    private static bool TryFuse(
        Call matmulCall,
        IReadOnlyDictionary<TIR.Buffer, Call[]> uses,
        out Call mergeCall,
        out Call fused)
    {
        mergeCall = null!;
        fused = null!;
        var matmul = (TIR.NTT.PackedMatMul)matmulCall.Target;
        var arguments = matmulCall.Arguments.ToArray();
        if (matmul.FusedReduce ||
            arguments.Length < 6 ||
            arguments[0] is not TIR.Buffer lhs ||
            arguments[1] is not TIR.Buffer rhs ||
            arguments[2] is not TIR.Buffer output ||
            !uses.TryGetValue(lhs, out var lhsUsers) ||
            lhsUsers.Length != 2)
        {
            return false;
        }

        mergeCall = lhsUsers.SingleOrDefault(call => call.Target is TIR.NTT.PagedAttentionMerge)!;
        if (mergeCall is null ||
            !lhsUsers.Any(call => ReferenceEquals(call, matmulCall)) ||
            mergeCall.Target is not TIR.NTT.PagedAttentionMerge merge)
        {
            return false;
        }

        var mergeArguments = mergeCall.Arguments.ToArray();
        if (mergeArguments.Length < 5 ||
            mergeArguments[0] is not TIR.Buffer maxState ||
            mergeArguments[1] is not TIR.Buffer sumState ||
            mergeArguments[2] is not TIR.Buffer accState ||
            mergeArguments[3] is not None ||
            mergeArguments[4] is not TIR.Buffer mergedOutput ||
            !ExactBufferRegionComparer.Instance.Equals(mergedOutput, lhs) ||
            lhs.Rank != 2 ||
            output.Rank != 2 ||
            matmul.RhsLayout != IR.NTT.PackedMatMulRhsLayout.KMajor)
        {
            return false;
        }

        fused = TIR.F.NTT.PagedAttentionMergePackedMatMul(
            maxState,
            sumState,
            accState,
            mergedOutput,
            lhs,
            rhs,
            output,
            (Expr)arguments[3],
            (Expr)arguments[4],
            (Expr)arguments[5],
            merge.Layout,
            merge.HiddenSize,
            merge.SplitHierarchyAxis,
            merge.SplitCount,
            matmul.RhsLayout);
        return true;
    }

    private sealed class ExactBufferRegionComparer : IEqualityComparer<TIR.Buffer>
    {
        public static readonly ExactBufferRegionComparer Instance = new();

        public bool Equals(TIR.Buffer? lhs, TIR.Buffer? rhs)
        {
            if (ReferenceEquals(lhs, rhs))
            {
                return true;
            }

            return lhs is not null &&
                rhs is not null &&
                HasSameBacking(lhs.MemSpan.Buffer, rhs.MemSpan.Buffer) &&
                lhs.MemSpan.Start.Equals(rhs.MemSpan.Start) &&
                lhs.MemSpan.Size.Equals(rhs.MemSpan.Size);
        }

        public int GetHashCode(TIR.Buffer buffer)
        {
            var physical = buffer.MemSpan.Buffer;
            var backingHash = HasStableBackingAddress(physical)
                ? HashCode.Combine(
                    physical.Location,
                    physical.Hierarchy,
                    physical.Start,
                    physical.Size,
                    physical.BlockLocalRDataMaterialization is null
                        ? 0
                        : RuntimeHelpers.GetHashCode(physical.BlockLocalRDataMaterialization))
                : RuntimeHelpers.GetHashCode(physical);
            return HashCode.Combine(backingHash, buffer.MemSpan.Start, buffer.MemSpan.Size);
        }

        private static bool HasSameBacking(PhysicalBuffer lhs, PhysicalBuffer rhs)
        {
            if (ReferenceEquals(lhs, rhs))
            {
                return true;
            }

            return HasStableBackingAddress(lhs) &&
                HasStableBackingAddress(rhs) &&
                lhs.Location == rhs.Location &&
                lhs.Hierarchy == rhs.Hierarchy &&
                ReferenceEquals(lhs.BlockLocalRDataMaterialization, rhs.BlockLocalRDataMaterialization) &&
                lhs.Start.Equals(rhs.Start) &&
                lhs.Size.Equals(rhs.Size);
        }

        private static bool HasStableBackingAddress(PhysicalBuffer buffer)
            => buffer.Start is not None and not Var;
    }

    private sealed class ReplacementRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;

        public ReplacementRewriter(IReadOnlyDictionary<BaseExpr, BaseExpr> replacements)
            : base(visitOtherFunctions: false)
        {
            _replacements = replacements;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
            => _replacements.TryGetValue(expr, out var replacement)
                ? Visit(replacement, context)
                : base.DispatchVisit(expr, context);

        protected override BaseExpr VisitSequential(Sequential expr, Unit context)
        {
            var rewritten = (Sequential)base.VisitSequential(expr, context);
            if (!rewritten.Fields.ToArray().Any(field => field is Sequential { CanFlatten: true }))
            {
                return rewritten;
            }

            var fields = new List<Expr>();
            foreach (var field in rewritten.Fields)
            {
                if (field is Sequential { CanFlatten: true } nested)
                {
                    fields.AddRange(nested.Fields.ToArray());
                }
                else
                {
                    fields.Add(field);
                }
            }

            return rewritten.With(fields: fields.ToArray());
        }
    }
}
