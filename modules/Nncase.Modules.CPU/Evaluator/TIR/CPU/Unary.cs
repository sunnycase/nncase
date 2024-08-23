﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class UnaryEvaluator : ITypeInferencer<Unary>, IKernelInfoEvaluator<Unary>
{
    public IRType Visit(ITypeInferenceContext context, Unary target)
    {
        context.CheckArgumentType<TensorType>(target, Unary.Input);
        context.CheckArgumentType<TensorType>(target, Unary.Output);
        return TupleType.Void;
    }

    public MicroKernelInfo Visit(Unary op, AffineDim[] domain, AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions)
    {
        var primitives = new int[bufferShapes[0].Length];
        var multipliers = new ValueRange<int>[bufferShapes[0].Length];
        for (int i = 0; i < bufferShapes[0].Length; i++)
        {
            if (Utilities.DistributedUtility.IsDivideExactly(bufferShapes[0][i], 4))
            {
                primitives[i] = 4;
                multipliers[i] = new(1, bufferShapes[0][i] / 4);
            }
            else
            {
                primitives[i] = 1;
                multipliers[i] = new(1, bufferShapes[0][i]);
            }
        }

        return new MicroKernelInfo(primitives, multipliers, 128, 128);
    }
}
