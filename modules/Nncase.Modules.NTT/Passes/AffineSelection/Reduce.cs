// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectReduce(IR.NTT.VectorizedReduce reduce, Call call, Expr output)
        => SelectReduce(
            call,
            (Expr)call[IR.NTT.VectorizedReduce.Input],
            output,
            reduce.Axes.ToArray(),
            reduce.KeepDims,
            reduce.ReduceOp,
            reduce.VectorizedAxes.ToArray(),
            ((RankedShape)call[IR.NTT.VectorizedReduce.PadedNums]).Dimensions.ToArray());

    public Expr SelectReduce(IR.Math.Reduce reduce, Call call, Expr output)
    {
        var input = (Expr)call[IR.Math.Reduce.Input];
        var axes = ((RankedShape)call[IR.Math.Reduce.Axes])
            .ToValueArray()
            .Select(axis => (int)Util.PositiveIndex(axis, input.CheckedTensorType))
            .OrderBy(axis => axis)
            .ToArray();
        var keepDims = ((TensorConst)call[IR.Math.Reduce.KeepDims]).Value.ToScalar<bool>();
        return SelectReduce(
            call,
            input,
            output,
            axes,
            keepDims,
            reduce.ReduceOp,
            Array.Empty<int>(),
            Array.Empty<Dimension>());
    }

    private Expr SelectReduce(
        Call call,
        Expr input,
        Expr output,
        int[] reductionAxes,
        bool keepDims,
        ReduceOp reduceOp,
        int[] vectorizedAxes,
        Dimension[] padedNums)
    {
        if (output.CheckedShape is not { IsFixed: true, Rank: > 0 })
        {
            return call;
        }

        var rank = input.CheckedShape.Rank;
        if (reductionAxes.Distinct().Count() != reductionAxes.Length ||
            reductionAxes.Any(axis => axis < 0 || axis >= rank))
        {
            throw new InvalidOperationException(
                $"Reduce axes [{string.Join(", ", reductionAxes)}] are invalid for rank {rank}.");
        }

        // Backend-private accumulators must be created after every parallel
        // tile coordinate is in scope. Canonicalize the affine domain so all
        // reduction loops are innermost without changing tensor-axis order.
        var domainInputAxes = Enumerable.Range(0, rank)
            .Where(axis => !reductionAxes.Contains(axis))
            .Concat(reductionAxes)
            .ToArray();
        var inputAxisToDomainAxis = new int[rank];
        for (var domainAxis = 0; domainAxis < rank; domainAxis++)
        {
            inputAxisToDomainAxis[domainInputAxes[domainAxis]] = domainAxis;
        }

        var domains = IR.F.Affine.Domains(rank);
        var inputResults = Enumerable.Range(0, rank)
            .Select(inputAxis => new AffineRange(
                domains[inputAxisToDomainAxis[inputAxis]].Offset,
                domains[inputAxisToDomainAxis[inputAxis]].Extent))
            .ToArray();
        var outrank = call.CheckedShape.Rank;
        var results = new AffineRange[outrank];
        {
            var j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (reductionAxes.Contains(i))
                {
                    if (keepDims)
                    {
                        results[j++] = new AffineRange(0, 1);
                    }
                }
                else
                {
                    var domain = domains[inputAxisToDomainAxis[i]];
                    results[j++] = new AffineRange(domain.Offset, domain.Extent);
                }
            }
        }

        var inputMap = new AffineMap(domains, default, inputResults);
        var affinemap = new AffineMap(domains, default, results);
        var tileAxisPolicies = domainInputAxes
            .Select(inputAxis => reductionAxes.Contains(inputAxis)
                ? GridTileAxisPolicy.Reduction()
                : GridTileAxisPolicy.Search())
            .ToArray();
        var reductionDomainAxes = Enumerable.Range(rank - reductionAxes.Length, reductionAxes.Length).ToArray();
        return IR.F.Affine.Grid()
            .Domain(tileAxisPolicies, out var domainVar)
            .Read(input, inputMap, out var intile)
            .Write(output, affinemap, out var outTile)
            .Body(TIR.F.NTT.Reduce(intile, outTile, GetLoadPreviousExpr(reductionDomainAxes, domainVar), vectorizedAxes, padedNums, reductionAxes, keepDims, reduceOp))
            .Build();
    }

    private Expr GetLoadPreviousExpr(IReadOnlyList<int> axes, Expr domainVar)
    {
        Expr? outExpr = null;
        foreach (var axis in axes)
        {
            var domainAxisVar = (Expr)domainVar[axis][0];
            if (outExpr is null)
            {
                outExpr = IR.F.Math.NotEqual(domainAxisVar, 0L);
            }
            else
            {
                outExpr = IR.F.Math.LogicalOr(outExpr, IR.F.Math.NotEqual(domainAxisVar, 0L));
            }
        }

        if (outExpr is null)
        {
            throw new NotSupportedException("reduce axes is empty");
        }

        return outExpr;
    }
}
