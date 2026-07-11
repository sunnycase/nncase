// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Deterministically composes storage-preserving affine views into compatible Grid reads.
/// </summary>
public sealed class AffineViewCompositionPass : FunctionPass
{
    public AffineViewCompositionPass(string moduleKind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function || input.ModuleKind != ModuleKind)
        {
            return Task.FromResult(input);
        }

        var body = (BaseExpr)new AffineViewCompositionRewriter().Rewrite(function.Body);
        VerifyNoComposableGridViews(body);
        VerifyNoUnobservedDistributionRoundTrips(body);
        return Task.FromResult((BaseFunction)function.With(body: body));
    }

    private static void VerifyNoUnobservedDistributionRoundTrips(BaseExpr root)
    {
        foreach (var call in ExprCollector.Collect(root).OfType<Call>())
        {
            if (TryGetUnobservedDistributionRoundTrip(call, out _, out _, out _))
            {
                throw new InvalidOperationException("Unobserved distributed AffineView round trip remains after affine-view composition.");
            }
        }
    }

    private static void VerifyNoComposableGridViews(BaseExpr root)
    {
        foreach (var grid in ExprCollector.Collect(root).OfType<Grid>())
        {
            for (var index = 0; index < grid.Accesses.Length; index++)
            {
                var access = grid.Accesses[index];
                if (access.IsRead &&
                    access.Value is Call { Target: AffineView } viewCall &&
                    CanCompose((AffineView)viewCall.Target, access.Value.CheckedType, ((Expr)viewCall[AffineView.Input]).CheckedType))
                {
                    throw new InvalidOperationException($"Composable AffineView remains on Grid read {index} after affine-view composition.");
                }
            }
        }
    }

    private static bool CanCompose(AffineView view, IRType resultType, IRType sourceType)
    {
        var resultTensorType = resultType switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => null,
        };
        var sourceTensorType = sourceType switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => null,
        };
        if (resultTensorType is null || sourceTensorType is null ||
            resultTensorType.DType != sourceTensorType.DType ||
            resultTensorType.Shape.Rank != sourceTensorType.Shape.Rank ||
            view.Transform.ResultMap.Domains.Length != resultTensorType.Shape.Rank)
        {
            return false;
        }

        return (resultType, sourceType) switch
        {
            (TensorType, TensorType) => true,
            (DistributedType resultDistributed, DistributedType sourceDistributed) =>
                resultDistributed.Placement == sourceDistributed.Placement &&
                resultDistributed.Partial == sourceDistributed.Partial &&
                DistributedUtility.AreSamePolicies(resultDistributed.AxisPolicies, sourceDistributed.AxisPolicies),
            _ => false,
        };
    }

    private static bool TryGetUnobservedDistributionRoundTrip(
        Call call,
        out Expr source,
        out TensorType resultType,
        out AffineViewTransform transform)
    {
        source = null!;
        resultType = null!;
        transform = null!;
        if (call.Target is not Boxing { NewType: TensorType outerType } ||
            call.Arguments[Boxing.Input.Index] is not Call { Target: AffineView view } viewCall ||
            view.NewType is not DistributedType { Partial: null } ||
            viewCall.Arguments[AffineView.Input.Index] is not Call { Target: Boxing { NewType: DistributedType { Partial: null } } } innerBoxing ||
            innerBoxing.Arguments[Boxing.Input.Index] is not Expr innerSource ||
            innerSource.CheckedType is not TensorType)
        {
            return false;
        }

        if (AffineViewUtility.Verify(innerSource.CheckedType, outerType, view.Transform) is not null)
        {
            return false;
        }

        source = innerSource;
        resultType = outerType;
        transform = view.Transform;
        return true;
    }

    private sealed class AffineViewCompositionRewriter : ExprRewriter
    {
        protected override BaseExpr RewriteLeafCall(Call call)
        {
            if (!TryGetUnobservedDistributionRoundTrip(call, out var source, out var resultType, out var transform))
            {
                return call;
            }

            var view = IR.F.Affine.View(source, resultType, transform);
            if (!CompilerServices.InferenceType(view))
            {
                throw new InvalidOperationException("Type inference failed while eliminating an unobserved distributed AffineView round trip.");
            }

            return view;
        }

        protected override BaseExpr RewriteLeafGrid(Grid grid)
        {
            var accesses = grid.Accesses.ToArray();
            var changed = false;

            for (var accessIndex = 0; accessIndex < accesses.Length; accessIndex++)
            {
                var access = accesses[accessIndex];
                if (!access.IsRead || !access.IsAffine)
                {
                    continue;
                }

                while (access.Value is Call { Target: AffineView view } viewCall)
                {
                    var source = (Expr)viewCall[AffineView.Input];
                    if (!CanCompose(view, access.Value.CheckedType, source.CheckedType))
                    {
                        break;
                    }

                    access = access.With(
                        value: source,
                        buffer: IR.F.Buffer.BufferOf(source),
                        region: view.Transform.ComposeResultAccess(access.AffineMap));
                    changed = true;
                }

                accesses[accessIndex] = access;
            }

            return changed
                ? grid.With(accesses: accesses)
                : grid;
        }
    }
}
