// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Google.OrTools.ConstraintSolver;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass : AffineSelectionPass
{
    private readonly CompileOptions _compileOptions;

    public NTTAffineSelectionPass(CompileOptions compileOptions, string moduleKind = CPUTarget.Kind)
        : base(moduleKind)
    {
        _compileOptions = compileOptions;
    }

    protected override Expr SelectCall(Call call, BaseExpr output)
    {
        switch (call.Target)
        {
            case IR.NTT.PackedQKVParallelLinear op:
                return SelectPackedQKVParallelLinear(op, call, output);
        }

        if (output is not Expr exprOutput)
        {
            return call;
        }

        switch (call.Target)
        {
            case IR.Distributed.Boxing op:
                return SelectBoxing(op, call, exprOutput);
            case IR.NTT.PackedMatMul op:
                return SelectMatMul(op, call, exprOutput);
            case IR.NTT.PackedMatMulGlu op:
                return SelectPackedMatMulGlu(op, call, exprOutput);
            case IR.NTT.VectorizedBinary op:
                return SelectVectorizedBinary(op, call, exprOutput);
            case IR.NTT.VectorizedMatMul:
                return SelectMatMul((Op)call.Target, call, exprOutput);
            case IR.Tensors.Pack op:
                return SelectVectorize(op, call, exprOutput);
            case IR.NTT.VectorizedReduce op:
                return SelectReduce(op, call, exprOutput);
            case IR.NTT.VectorizedRoPE op:
                return SelectRoPE(op, call, exprOutput);
            case IR.Tensors.Unpack op:
                return SelectDevectorize(op, call, exprOutput);
            case IR.Math.Binary op:
                return SelectBinary(op, call, exprOutput);
            case IR.Math.MatMul:
                return SelectMatMul((Op)call.Target, call, exprOutput);
            case IR.Math.Unary op:
                return SelectUnaryLike((Expr)call[IR.Math.Unary.Input], new TIR.NTT.Unary(op.UnaryOp), call, exprOutput);
            case IR.NN.Swish op:
                return SelectSwish(op, call, exprOutput);
            case IR.NTT.VectorizedLayerNorm op:
                return SelectLayerNorm(op, call, exprOutput);
            case IR.NN.LayerNorm op:
                return SelectLayerNorm(op, call, exprOutput);
            case IR.NN.NormStats op:
                return SelectNormStats(op, call, exprOutput);
            case IR.NN.NormApply op:
                return SelectNormApply(op, call, exprOutput);
            case IR.Tensors.Cast op:
                return SelectCast(op, call, exprOutput);
            case IR.NTT.VectorizedCast op:
                return SelectVectorizedCast(op, call, exprOutput);
            case IR.Tensors.Bitcast:
                return SelectBitcast(call, exprOutput);
            case IR.Tensors.Reshape:
                return SelectReshape(call, exprOutput);
            case IR.Tensors.Transpose op:
                return SelectTranspose(op, call, exprOutput);
            case IR.Tensors.Where op:
                return SelectWhere(op, call, exprOutput);
            case IR.Math.Compare op:
                return SelectCompare(op, call, exprOutput);
            case IR.NN.GetPositionIds op:
                return SelectGetPositionIds(op, call, exprOutput);
            case IR.NN.UpdatePagedAttentionKVCache op:
                return SelectUpdatePagedAttentionKVCache(op, call);
            default:
                return call;
        }
    }
}
