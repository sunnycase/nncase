// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
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
        if (!TryGetPartialOutput(
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
            !ReferenceEquals(
                partialCall[SamplingPartial.Logits],
                combineCall[SamplingCombine.Logits]) ||
            !ReferenceEquals(partialCall[SamplingPartial.State], combineCall[SamplingCombine.State]) ||
            !HasOnlyGetItemUsers(partialCall, combineCall) ||
            !TryMatchLogitsProducer(
                combineCall[SamplingCombine.Logits],
                partialCall,
                combineCall,
                out var producer))
        {
            return false;
        }

        var packed = (PackedMatMul)producer.PackedCall.Target;
        var partialFusion = IR.F.NTT.PackedMatMulSamplingPartial(
            (Expr)producer.PackedCall[PackedMatMul.Lhs],
            (Expr)producer.PackedCall[PackedMatMul.Rhs],
            (Expr)combineCall[SamplingCombine.State],
            packed.OutputDataType,
            producer.OutputDataType,
            packed.RhsLayout,
            combine.Config,
            (Expr)producer.PackedCall[PackedMatMul.Scale],
            (Expr)producer.PackedCall[PackedMatMul.Addend],
            producer.LhsScale,
            producer.RhsScale);
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

    private static bool TryMatchLogitsProducer(
        BaseExpr logits,
        Call partialCall,
        Call combineCall,
        out LogitsProducer producer)
    {
        producer = null!;
        if (logits is not Call { Target: Bitcast bitcast } bitcastCall ||
            bitcast.NewType is not PrimType outputDataType)
        {
            return false;
        }

        BaseExpr product = bitcastCall[Bitcast.Input];
        var internalCalls = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance)
        {
            bitcastCall,
        };
        if (product is Call { Target: VectorizedCast cast } castCall)
        {
            if (cast.NewType is not VectorType { ElemType: var castOutputType } ||
                castOutputType != outputDataType ||
                cast.CastMode != CastMode.KDefault ||
                castCall[VectorizedCast.PostOps] is not None)
            {
                return false;
            }

            internalCalls.Add(castCall);
            product = castCall[VectorizedCast.Input];
        }

        var factors = new List<Expr>();
        if (!TryCollectScaledPackedMatMul(
                product,
                factors,
                internalCalls,
                out var packedCall) ||
            packedCall.Target is not PackedMatMul
            {
                FusedReduce: false,
            } packed ||
            packedCall.CheckedType is not DistributedType { Partial: null })
        {
            return false;
        }

        Expr lhsScale;
        Expr rhsScale;
        switch (factors.Count)
        {
            case 0 when outputDataType == packed.OutputDataType:
                lhsScale = None.Default;
                rhsScale = None.Default;
                break;
            case 2 when TryOrderScales(factors[0], factors[1], out lhsScale, out rhsScale):
                break;
            default:
                return false;
        }

        internalCalls.Add(packedCall);
        foreach (var call in internalCalls)
        {
            foreach (var user in call.Users)
            {
                if (internalCalls.Contains(user) ||
                    ReferenceEquals(call, bitcastCall) &&
                    (ReferenceEquals(user, partialCall) || ReferenceEquals(user, combineCall)))
                {
                    continue;
                }

                return false;
            }
        }

        producer = new LogitsProducer(packedCall, outputDataType, lhsScale, rhsScale);
        return true;
    }

    private static bool TryCollectScaledPackedMatMul(
        BaseExpr expression,
        List<Expr> factors,
        HashSet<BaseExpr> internalCalls,
        out Call packedCall)
    {
        while (expression is Call { Target: ShardedView } viewCall &&
               ContainsPackedMatMul(viewCall[ShardedView.Input]))
        {
            internalCalls.Add(viewCall);
            expression = viewCall[ShardedView.Input];
        }

        if (expression is Call { Target: PackedMatMul } direct)
        {
            packedCall = direct;
            return true;
        }

        if (expression is not Call
            {
                Target: Binary
                {
                    BinaryOp: BinaryOp.Mul,
                },
            } multiply)
        {
            packedCall = null!;
            return false;
        }

        var lhsContainsPacked = ContainsPackedMatMul(multiply[Binary.Lhs]);
        var rhsContainsPacked = ContainsPackedMatMul(multiply[Binary.Rhs]);
        if (lhsContainsPacked == rhsContainsPacked)
        {
            packedCall = null!;
            return false;
        }

        internalCalls.Add(multiply);
        factors.Add((Expr)multiply[lhsContainsPacked ? Binary.Rhs : Binary.Lhs]);
        return TryCollectScaledPackedMatMul(
            multiply[lhsContainsPacked ? Binary.Lhs : Binary.Rhs],
            factors,
            internalCalls,
            out packedCall);
    }

    private static bool ContainsPackedMatMul(BaseExpr expression)
        => expression switch
        {
            Call { Target: PackedMatMul } => true,
            Call { Target: ShardedView } view => ContainsPackedMatMul(view[ShardedView.Input]),
            Call
            {
                Target: Binary
                {
                    BinaryOp: BinaryOp.Mul,
                },
            } binary =>
                ContainsPackedMatMul(binary[Binary.Lhs]) ||
                ContainsPackedMatMul(binary[Binary.Rhs]),
            _ => false,
        };

    private static bool TryOrderScales(
        Expr first,
        Expr second,
        out Expr lhsScale,
        out Expr rhsScale)
    {
        var firstScalar = IsSingleElementTensor(first.CheckedType);
        var secondScalar = IsSingleElementTensor(second.CheckedType);
        if (firstScalar == secondScalar)
        {
            lhsScale = null!;
            rhsScale = null!;
            return false;
        }

        lhsScale = firstScalar ? first : second;
        rhsScale = firstScalar ? second : first;
        return true;
    }

    private static bool IsSingleElementTensor(IRType type)
    {
        var tensor = type switch
        {
            TensorType value => value,
            DistributedType value => value.TensorType,
            _ => null,
        };
        return tensor?.Shape is RankedShape shape &&
               shape.All(dimension => dimension.IsFixed && dimension.FixedValue == 1);
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

    private static bool HasOnlyGetItemUsers(Call partialCall, Call combineCall)
    {
        var users = partialCall.Users.OfType<Call>().ToArray();
        return users.Length == 2 &&
               users.All(user =>
                   user.Target is GetItem &&
                   user.Users.Count() == 1 &&
                   ReferenceEquals(user.Users.Single(), combineCall));
    }

    private sealed record LogitsProducer(
        Call PackedCall,
        PrimType OutputDataType,
        Expr LhsScale,
        Expr RhsScale);

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
