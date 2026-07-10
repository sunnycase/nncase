// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class BitcastEvaluator : ITypeInferencer<Bitcast>, IKernelInfoEvaluator<Bitcast>
{
    public IRType Visit(ITypeInferenceContext context, Bitcast target)
    {
        var inputType = context.CheckArgumentType<TensorType>(target, Bitcast.Input);
        var outputType = context.CheckArgumentType<TensorType>(target, Bitcast.Output);
        if (GetScalarDataType(inputType.DType) != GetScalarDataType(outputType.DType))
        {
            throw new TypeInferenceInterruptException(
                new InvalidType($"TIR Bitcast only supports lane reinterpretation of one scalar dtype, got input={inputType.DType}, output={outputType.DType}."));
        }

        return TupleType.Void;
    }

    public MicroKernelInfo Visit(Bitcast op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var tileBounds = Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(tileBounds, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        var outputSize = bufferShapes[^1].Aggregate((IntExpr)solver.MakeIntConst(1), (acc, dim) => solver.MakeProd(acc, dim));
        return outputSize;
    }

    private static DataType GetScalarDataType(DataType dataType) => dataType switch
    {
        VectorType vectorType => vectorType.ElemType,
        MaskVectorType => DataTypes.Boolean,
        _ => dataType,
    };
}
