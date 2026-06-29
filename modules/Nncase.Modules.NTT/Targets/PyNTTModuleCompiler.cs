// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.CodeGen.PyNTT;
using Nncase.IR;

namespace Nncase.Targets;

/// <summary>
/// PyNTT module compiler.
/// </summary>
public sealed class PyNTTModuleCompiler : INTTModuleCompiler
{
    /// <inheritdoc/>
    public string ModuleKind => PyNTTTarget.Kind;

    /// <inheritdoc/>
    public MaskVectorStyle MaskVectorStyle => MaskVectorStyle.Fat;

    /// <inheritdoc/>
    public int Lane => 1;

    /// <inheritdoc/>
    public int Nr => 1;

    /// <inheritdoc/>
    public IModuleBuilder CreateModuleBuilder(CompileOptions options) => new PyNTTModuleBuilder(ModuleKind, options);

    /// <inheritdoc/>
    public bool IsSupportedCall(Call call, CompileOptions options)
    {
        return call.Target switch
        {
            Op op when IsSupportedOp(op) => call.Arguments.AsValueEnumerable().All(IsSupportedArgument),
            _ => false,
        };
    }

    private static bool IsSupportedArgument(BaseExpr argument)
    {
        return argument.CheckedType switch
        {
            TensorType tensorType => tensorType.Shape.IsRanked,
            DistributedType distributedType => distributedType.TensorType.Shape.IsRanked,
            _ => true,
        };
    }

    private static bool IsSupportedOp(Op op)
    {
        return op switch
        {
            IR.Math.Unary unary => IsSupportedUnaryOp(unary.UnaryOp),
            IR.CustomNTT.Unary customUnary => IsSupportedUnaryOp(customUnary.UnaryOp),
            IR.Math.Binary { BinaryOp: BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div or BinaryOp.Mod or BinaryOp.Min or BinaryOp.Max } => true,
            IR.Math.Clamp => true,
            IR.Math.Compare => true,
            IR.Tensors.Cast { CastMode: CastMode.KDefault } => true,
            IR.Tensors.Concat
                or IR.Tensors.Expand
                or IR.Tensors.Gather
                or IR.Tensors.ScatterND
                or IR.Tensors.Slice
                or IR.Tensors.Transpose
                or IR.Tensors.Where
                or IR.NN.Pad { PadMode: PadMode.Constant }
                or IR.NN.Conv2D { PadMode: PadMode.Constant }
                or IR.NN.Erf
                or IR.NN.RoPE
                or IR.NN.Swish
                or IR.Math.MatMul
                or IR.NTT.VectorizedMatMul
                or IR.Math.Reduce { ReduceOp: ReduceOp.Sum or ReduceOp.Mean or ReduceOp.Max or ReduceOp.Min }
                or IR.NTT.VectorizedReduce { ReduceOp: ReduceOp.Sum or ReduceOp.Mean or ReduceOp.Max or ReduceOp.Min }
                or IR.NN.Softmax
                or IR.NTT.VectorizedSoftmax => true,
            _ => false,
        };
    }

    private static bool IsSupportedUnaryOp(UnaryOp op)
    {
        return op is UnaryOp.Abs
            or UnaryOp.Acos
            or UnaryOp.Acosh
            or UnaryOp.Asin
            or UnaryOp.Asinh
            or UnaryOp.Ceil
            or UnaryOp.Cos
            or UnaryOp.Cosh
            or UnaryOp.Erf
            or UnaryOp.Exp
            or UnaryOp.Floor
            or UnaryOp.Log
            or UnaryOp.Neg
            or UnaryOp.Round
            or UnaryOp.Rsqrt
            or UnaryOp.Sin
            or UnaryOp.Sinh
            or UnaryOp.Sqrt
            or UnaryOp.Square
            or UnaryOp.Tanh
            or UnaryOp.LogicalNot
            or UnaryOp.Sign;
    }
}
