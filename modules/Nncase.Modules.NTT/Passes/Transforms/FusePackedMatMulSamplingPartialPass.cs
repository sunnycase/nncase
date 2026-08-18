// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Fuses the post-distribution LM-head with token-local sampling processors
/// after vocabulary sharding has been fixed. The cross-shard combine remains
/// explicit and consumes materialized raw/processed logits and partial state.
/// </summary>
public sealed class FusePackedMatMulSamplingPartialPass : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function)
        {
            return Task.FromResult(input);
        }

        var replacements = new Dictionary<BaseExpr, BaseExpr>(
            ReferenceEqualityComparer.Instance);
        foreach (var combineCall in ExprCollector.Collect(function.Body)
                     .OfType<Call>()
                     .Where(call => call.Target is SamplingCombine))
        {
            if (TryCreateFusion(combineCall, out var fused))
            {
                replacements.Add(combineCall, fused.InheritMetaData(combineCall));
            }
        }

        if (replacements.Count == 0)
        {
            return Task.FromResult(input);
        }

        var rewritten = (BaseFunction)new ReplacementRewriter(replacements).Rewrite(function);
        if (!CompilerServices.InferenceType(rewritten))
        {
            throw new InvalidOperationException(
                $"PackedMatMulSamplingPartial fusion could not infer function {function.Name}.");
        }

        if (rewritten.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException(
                $"PackedMatMulSamplingPartial fusion produced an invalid function {function.Name}: {invalid}.");
        }

        return Task.FromResult(rewritten);
    }

    private static bool TryCreateFusion(Call combineCall, out Expr fused)
    {
        fused = null!;
        var combine = (SamplingCombine)combineCall.Target;
        if (combineCall[SamplingCombine.Logits] is not Call
            {
                Target: Bitcast bitcast,
            } bitcastCall ||
            bitcastCall[Bitcast.Input] is not Call
            {
                Target: PackedMatMul packed,
            } packedCall ||
            packed.FusedReduce ||
            packedCall.CheckedType is not DistributedType { Partial: null } ||
            bitcast.NewType != packed.OutputDataType ||
            !TryGetPartialOutput(
                combineCall[SamplingCombine.ProcessedLogits],
                expectedIndex: 0,
                out var partialCall) ||
            !TryGetPartialOutput(
                combineCall[SamplingCombine.ArgMaxState],
                expectedIndex: 1,
                out var argMaxPartialCall) ||
            !ReferenceEquals(partialCall, argMaxPartialCall) ||
            partialCall.Target is not SamplingPartial partial ||
            partial.Config != combine.Config ||
            !ReferenceEquals(partialCall[SamplingPartial.Logits], bitcastCall) ||
            !ReferenceEquals(partialCall[SamplingPartial.State], combineCall[SamplingCombine.State]) ||
            !HasOnlyExpectedUsers(packedCall, bitcastCall) ||
            !HasOnlyExpectedUsers(bitcastCall, partialCall, combineCall) ||
            !HasOnlyGetItemUsers(partialCall, combineCall))
        {
            return false;
        }

        var partialFusion = IR.F.NTT.PackedMatMulSamplingPartial(
            (Expr)packedCall[PackedMatMul.Lhs],
            (Expr)packedCall[PackedMatMul.Rhs],
            (Expr)combineCall[SamplingCombine.State],
            packed.OutputDataType,
            packed.RhsLayout,
            combine.Config,
            (Expr)packedCall[PackedMatMul.Scale],
            (Expr)packedCall[PackedMatMul.Addend]);
        if (!CompilerServices.InferenceType(partialFusion) ||
            partialFusion.CheckedType is not TupleType { Fields.Count: 3 })
        {
            return false;
        }

        var candidate = IR.F.NTT.SamplingCombine(
            partialFusion[0],
            partialFusion[1],
            partialFusion[2],
            (Expr)combineCall[SamplingCombine.State],
            combine.Config);
        if (!CompilerServices.InferenceType(candidate) ||
            candidate.CheckedType is InvalidType ||
            !Equals(candidate.CheckedType, combineCall.CheckedType))
        {
            return false;
        }

        fused = candidate;
        return true;
    }

    private static bool TryGetPartialOutput(
        BaseExpr expression,
        long expectedIndex,
        out Call partialCall)
    {
        if (expression is Call { Target: GetItem } getItemCall &&
            getItemCall[GetItem.Input] is Call { Target: SamplingPartial } input &&
            getItemCall[GetItem.Index] is DimConst { Value: var index } &&
            index == expectedIndex)
        {
            partialCall = input;
            return true;
        }

        partialCall = null!;
        return false;
    }

    private static bool HasOnlyExpectedUsers(BaseExpr expression, params BaseExpr[] expected)
    {
        var actual = expression.Users.ToArray();
        return actual.Length == expected.Length &&
               actual.All(user => expected.Any(item => ReferenceEquals(item, user)));
    }

    private static bool HasOnlyGetItemUsers(Call partialCall, Call combineCall)
    {
        var users = partialCall.Users.OfType<Call>().ToArray();
        return users.Length == 2 &&
               users.All(user =>
                   user.Target is GetItem &&
                   user.Users.Count() == 1 &&
                   ReferenceEquals(user.Users.Single(), combineCall));
    }

    private sealed class ReplacementRewriter : ExprRewriter
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replacements;

        public ReplacementRewriter(IReadOnlyDictionary<BaseExpr, BaseExpr> replacements)
        {
            _replacements = replacements;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
            => _replacements.TryGetValue(expr, out var replacement)
                ? Visit(replacement, context)
                : base.DispatchVisit(expr, context);
    }
}
