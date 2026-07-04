// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CostModelTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTargetOpCostModel : TestClassBase
{
    private readonly TestTargetOpCostModel _costModel = new();

    public UnitTestTargetOpCostModel()
    {
        CompileOptions.TargetOptions = new TestTargetOptions(_costModel);
    }

    [Fact]
    public void TestUnaryBinaryElementwiseAndMatMulUseTargetCostModel()
    {
        var unaryInput = new Var("unary_input", new TensorType(DataTypes.Float32, [2, 3]));
        var unary = IR.F.Math.Unary(UnaryOp.Abs, unaryInput);
        CompilerServices.InferenceType(unary);
        var unaryCost = CompilerServices.EvaluateCost(unary, CompileOptions);

        Assert.Equal((UInt128)101, unaryCost[CostFactorNames.CPUCycles]);
        Assert.Equal(new long[] { 2, 3 }, ((RankedShape)_costModel.UnaryQuery!.Input.Shape).ToValueArray());

        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [2, 3]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [2, 3]));
        var binary = lhs + rhs;
        CompilerServices.InferenceType(binary);
        var binaryCost = CompilerServices.EvaluateCost(binary, CompileOptions);

        Assert.Equal((UInt128)202, binaryCost[CostFactorNames.CPUCycles]);
        Assert.Equal(new long[] { 2, 3 }, ((RankedShape)_costModel.BinaryQuery!.Output.Shape).ToValueArray());

        var castInput = new Var("cast_input", new TensorType(DataTypes.Float32, [2, 3]));
        var cast = IR.F.Tensors.Cast(castInput, DataTypes.Float16);
        CompilerServices.InferenceType(cast);
        var castCost = CompilerServices.EvaluateCost(cast, CompileOptions);

        Assert.Equal((UInt128)404, castCost[CostFactorNames.CPUCycles]);
        Assert.Equal(DataTypes.Float32, _costModel.ElementwiseQuery!.Inputs[0].DType);
        Assert.Equal(DataTypes.Float16, _costModel.ElementwiseQuery!.Output.DType);
        Assert.Equal(new long[] { 2, 3 }, ((RankedShape)_costModel.ElementwiseQuery!.Output.Shape).ToValueArray());

        var matMulLhs = new Var("matmul_lhs", new TensorType(DataTypes.Float32, [2, 8]));
        var matMulRhs = new Var("matmul_rhs", new TensorType(DataTypes.Float32, [8, 4]));
        var matmul = IR.F.Tensors.MatMul(matMulLhs, matMulRhs);
        CompilerServices.InferenceType(matmul);
        var matmulCost = CompilerServices.EvaluateCost(matmul, CompileOptions);

        Assert.Equal((UInt128)303, matmulCost[CostFactorNames.CPUCycles]);
        Assert.False(_costModel.MatMulQuery!.Lhs.DType is VectorType);
        Assert.False(_costModel.MatMulQuery!.Rhs.DType is VectorType);
        Assert.False(_costModel.MatMulQuery!.Output.DType is VectorType);
        Assert.Equal(new long[] { 2, 4 }, ((RankedShape)_costModel.MatMulQuery!.Output.Shape).ToValueArray());
    }

    [Fact]
    public void TestGatherCostUsesGatheredSliceNotWholeInput()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, [100, 32]));
        var indices = new Var("indices", new TensorType(DataTypes.Int64, [2]));
        var gather = IR.F.Tensors.Gather(input, 0, indices);
        CompilerServices.InferenceType(gather);

        var cost = CompilerServices.EvaluateCost(gather, CompileOptions);

        Assert.Equal((UInt128)272, cost[CostFactorNames.BlockLocalMemoryLoadBytes]);
        Assert.Equal((UInt128)256, cost[CostFactorNames.BlockLocalMemoryStoreBytes]);
    }

    private sealed record TestTargetOptions(ITargetOpCostModel TargetCostModel) : ITargetOptions, ITargetOpCostModelProvider;

    private sealed class TestTargetOpCostModel : ITargetOpCostModel
    {
        public UnaryOpCostQuery? UnaryQuery { get; private set; }

        public BinaryOpCostQuery? BinaryQuery { get; private set; }

        public MatMulOpCostQuery? MatMulQuery { get; private set; }

        public ElementwiseOpCostQuery? ElementwiseQuery { get; private set; }

        public UInt128 GetLatency(Cost cost)
        {
            return cost.Score;
        }

        public bool TryGetUnaryCost(UnaryOpCostQuery query, out Cost cost)
        {
            UnaryQuery = query;
            cost = Cycles(101);
            return true;
        }

        public bool TryGetBinaryCost(BinaryOpCostQuery query, out Cost cost)
        {
            BinaryQuery = query;
            cost = Cycles(202);
            return true;
        }

        public bool TryGetElementwiseCost(ElementwiseOpCostQuery query, out Cost cost)
        {
            ElementwiseQuery = query;
            cost = Cycles(404);
            return true;
        }

        public bool TryGetMatMulCost(MatMulOpCostQuery query, out Cost cost)
        {
            MatMulQuery = query;
            cost = Cycles(303);
            return true;
        }

        private static Cost Cycles(UInt128 cycles)
        {
            return new Cost { [CostFactorNames.CPUCycles] = cycles };
        }
    }
}
