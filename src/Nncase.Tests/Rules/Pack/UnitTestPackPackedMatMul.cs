// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.NTT;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestVectorizeVectorizedMatMul : TransformTestBase
{
    [Fact]
    public void TestPackMatMulByNDoesNotRequireOuterUnpack()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(8, 32)));
        var rhs = new Var("rhs", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(32, 8)));
        var expr = VectorizedMatMul(lhs, rhs, [], new[] { 1 }, outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = (Expr)CompilerServices.Rewrite(expr, [new PackMatMulByN(4)], context);
        CompilerServices.InferenceType(post);
        var printed = CompilerServices.Print(post);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Contains("PackedMatMul", printed, System.StringComparison.Ordinal);
        Assert.Contains("Unpack(Lanes: {4}", printed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("Unpack(Lanes: {8}", printed, System.StringComparison.Ordinal);
        var packedMatMul = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(call => call.Target is IR.NTT.PackedMatMul));
        Assert.Equal(
            IR.NTT.PackedMatMulRhsLayout.NMajor,
            Assert.IsType<IR.NTT.PackedMatMul>(packedMatMul.Target).RhsLayout);
    }

    [Fact]
    public void TestPackMatMulRhsKMajorPreservesVectorizedResultType()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var rhs = new Var("rhs", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(64, 16)));
        var expr = VectorizedMatMul(lhs, rhs, [], new[] { 1 }, outDataType: DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = (Expr)CompilerServices.Rewrite(expr, [new PackMatMulRhsKMajor(16, 2)], context);
        CompilerServices.InferenceType(post);
        var packedMatMul = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(call => call.Target is IR.NTT.PackedMatMul));
        var packedOp = Assert.IsType<IR.NTT.PackedMatMul>(packedMatMul.Target);
        var packedRhs = (Expr)packedMatMul.Arguments[IR.NTT.PackedMatMul.Rhs.Index];

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, packedOp.RhsLayout);
        Assert.Equal(new RankedShape(4, 16), packedRhs.CheckedShape);
        Assert.Equal(new VectorType(DataTypes.BFloat16, [8, 2, 8]), packedRhs.CheckedDataType);
    }

    [Fact]
    public void TestPackFp8MatMulRhsKMajorPreservesFp32ResultType()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float8E4M3, new RankedShape(1, 64)));
        var rhs = new Var(
            "rhs",
            new TensorType(
                new VectorType(DataTypes.Float8E4M3, [16]),
                new RankedShape(64, 8)));
        var expr = VectorizedMatMul(
            lhs,
            rhs,
            [],
            new[] { 1 },
            outDataType: DataTypes.Float32);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = (Expr)CompilerServices.Rewrite(
            expr,
            [new PackMatMulRhsKMajor(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedMatMul = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedMatMul));
        var packedOp = Assert.IsType<IR.NTT.PackedMatMul>(packedMatMul.Target);
        var packedRhs = (Expr)packedMatMul.Arguments[IR.NTT.PackedMatMul.Rhs.Index];

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, packedOp.RhsLayout);
        Assert.Equal(DataTypes.Float32, packedOp.OutputDataType);
        Assert.Equal(new RankedShape(2, 8), packedRhs.CheckedShape);
        Assert.Equal(
            new VectorType(DataTypes.Float8E4M3, [16, 2, 16]),
            packedRhs.CheckedDataType);
        Assert.Equal(
            new TensorType(
                new VectorType(DataTypes.Float32, [16]),
                new RankedShape(1, 8)),
            packedMatMul.CheckedType);
    }

    [Fact]
    public void TestPackBlockScaledMatMulPreservesSemanticRegionOnCompute()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float8E4M3, new RankedShape(64, 128)));
        var rhsScale = new Var("rhs_scale", new TensorType(DataTypes.BFloat16, new RankedShape(128, 1)));
        var region = new SemanticRegion(
            SemanticRegionKinds.Attention,
            "model.layers.0.linear_attn");
        var expr = IR.F.Math.BlockScaledMatMul(
            lhs,
            rhs,
            rhsScale,
            DataTypes.BFloat16,
            1,
            64);
        expr.Metadata.SemanticRegion = region;
        CompilerServices.InferenceType(expr);

        var post = (Expr)CompilerServices.Rewrite(
            expr,
            [new PackBlockScaledMatMulRhsNMajorKPacked(16, 2)],
            new Nncase.Passes.RunPassContext());
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedBlockScaledMatMul));

        Assert.Same(region, packedCall.Metadata.SemanticRegion);
    }

    [Fact]
    public void TestPackQKVParallelLinearByNUsesSinglePackedQKVOp()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var qWeight = new Var("q_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 64)));
        var kWeight = new Var("k_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var vWeight = new Var("v_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var expr = QKVParallelLinear(
            input,
            qWeight,
            kWeight,
            vWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            4,
            2,
            DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(expr, [new PackQKVParallelLinearByN(4, 16)], context);
        CompilerServices.InferenceType(post);
        var printed = CompilerServices.Print(post);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Contains("PackedQKVParallelLinear", printed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("PackedMatMul", printed, System.StringComparison.Ordinal);
        var combine = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedQKVParallelLinearCombine));
        Assert.IsType<IR.NTT.PackedQKVParallelLinear>(
            Assert.IsType<Call>(combine.Arguments[IR.NTT.PackedQKVParallelLinearCombine.QKV.Index]).Target);
    }

    [Fact]
    public void TestPackQKVParallelLinearRhsKMajorMatchesPackedMatMulLayout()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var qWeight = new Var("q_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 64)));
        var kWeight = new Var("k_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var vWeight = new Var("v_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 32)));
        var qBias = new Var("q_bias", new TensorType(DataTypes.BFloat16, new RankedShape(64)));
        var kBias = new Var("k_bias", new TensorType(DataTypes.BFloat16, new RankedShape(32)));
        var vBias = new Var("v_bias", new TensorType(DataTypes.BFloat16, new RankedShape(32)));
        var expr = QKVParallelLinear(
            input,
            qWeight,
            kWeight,
            vWeight,
            qBias,
            kBias,
            vBias,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            4,
            2,
            DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(
            expr,
            [new PackQKVParallelLinearRhsForGpu(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedQKVParallelLinear));
        var packedOp = Assert.IsType<IR.NTT.PackedQKVParallelLinear>(packedCall.Target);
        var combineCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedQKVParallelLinearCombine));
        Assert.Same(packedCall, combineCall.Arguments[IR.NTT.PackedQKVParallelLinearCombine.QKV.Index]);
        Assert.Equal(
            packedCall.CheckedType,
            Assert.IsType<IR.NTT.PackedQKVParallelLinearCombine>(combineCall.Target).OutputType);
        var expectedWeightType = new VectorType(DataTypes.BFloat16, [8, 2, 8]);
        var expectedOutputType = new VectorType(DataTypes.BFloat16, [8]);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, packedOp.RhsLayout);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.QWeight.Index, new RankedShape(4, 8), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.KWeight.Index, new RankedShape(4, 4), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.VWeight.Index, new RankedShape(4, 4), expectedWeightType);
        Assert.Equal(
            new TupleType(
            [
                new TensorType(expectedOutputType, new RankedShape(1, 8)),
                new TensorType(expectedOutputType, new RankedShape(1, 4)),
                new TensorType(expectedOutputType, new RankedShape(1, 4)),
            ]),
            packedCall.CheckedType);

        void AssertPackedBuffer(int index, RankedShape shape, DataType dataType)
        {
            var argument = (Expr)packedCall.Arguments[index];
            Assert.Equal(shape, argument.CheckedShape);
            Assert.Equal(dataType, argument.CheckedDataType);
        }
    }

    [Fact]
    public void TestPackDynamicTensorQKVParallelLinearUsesNMajorKPackedWeights()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var qWeight = new Var("q_weight", new TensorType(DataTypes.Float8E4M3, new RankedShape(64, 64)));
        var kWeight = new Var("k_weight", new TensorType(DataTypes.Float8E4M3, new RankedShape(64, 32)));
        var vWeight = new Var("v_weight", new TensorType(DataTypes.Float8E4M3, new RankedShape(64, 32)));
        var qWeightScale = new Var("q_weight_scale", new TensorType(DataTypes.BFloat16, new RankedShape(64)));
        var kWeightScale = new Var("k_weight_scale", new TensorType(DataTypes.BFloat16, new RankedShape(32)));
        var vWeightScale = new Var("v_weight_scale", new TensorType(DataTypes.BFloat16, new RankedShape(32)));
        var expr = QKVParallelLinear(
            input,
            qWeight,
            kWeight,
            vWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            qWeightScale,
            kWeightScale,
            vWeightScale,
            4,
            2,
            DataTypes.BFloat16,
            MatMulQuantizationMode.DynamicTensor);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(
            expr,
            [new PackQKVParallelLinearRhsForGpu(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedQKVParallelLinear));
        var packedOp = Assert.IsType<IR.NTT.PackedQKVParallelLinear>(packedCall.Target);
        var expectedWeightType = new VectorType(DataTypes.Float8E4M3, [2, 16]);
        var expectedScaleType = new VectorType(DataTypes.BFloat16, [8]);

        Assert.Equal(MatMulQuantizationMode.DynamicTensor, packedOp.QuantizationMode);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.NMajorKPacked, packedOp.RhsLayout);
        Assert.Equal(8, packedOp.OutputNVectorLaneCount);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.QWeight.Index, new RankedShape(64, 2), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.KWeight.Index, new RankedShape(32, 2), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.VWeight.Index, new RankedShape(32, 2), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.QWeightScale.Index, new RankedShape(8), expectedScaleType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.KWeightScale.Index, new RankedShape(4), expectedScaleType);
        AssertPackedBuffer(IR.NTT.PackedQKVParallelLinear.VWeightScale.Index, new RankedShape(4), expectedScaleType);

        void AssertPackedBuffer(int index, RankedShape shape, DataType dataType)
        {
            var argument = (Expr)packedCall.Arguments[index];
            Assert.Equal(shape, argument.CheckedShape);
            Assert.Equal(dataType, argument.CheckedDataType);
        }
    }

    [Fact]
    public void TestPackMatMulGluByNUsesSinglePackedMatMulGluOp()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var gateWeight = new Var("gate_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var upWeight = new Var("up_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var expr = MatMulGlu(
            input,
            gateWeight,
            upWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(expr, [new PackMatMulGluByN(4, 16)], context);
        CompilerServices.InferenceType(post);
        var printed = CompilerServices.Print(post);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Contains("PackedMatMulGlu", printed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("PackedMatMul(", printed, System.StringComparison.Ordinal);
    }

    [Fact]
    public void TestPackMatMulGluRhsKMajorMatchesPackedMatMulLayout()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var gateWeight = new Var("gate_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var upWeight = new Var("up_weight", new TensorType(DataTypes.BFloat16, new RankedShape(64, 128)));
        var gateBias = new Var("gate_bias", new TensorType(DataTypes.BFloat16, new RankedShape(128)));
        var upBias = new Var("up_bias", new TensorType(DataTypes.BFloat16, new RankedShape(128)));
        var expr = MatMulGlu(
            input,
            gateWeight,
            upWeight,
            gateBias,
            upBias,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = CompilerServices.Rewrite(
            expr,
            [new PackMatMulGluRhsKMajor(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedMatMulGlu));
        var packedOp = Assert.IsType<IR.NTT.PackedMatMulGlu>(packedCall.Target);
        var expectedWeightType = new VectorType(DataTypes.BFloat16, [8, 2, 8]);
        var expectedOutputType = new VectorType(DataTypes.BFloat16, [8]);

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Equal(IR.NTT.PackedMatMulRhsLayout.KMajor, packedOp.RhsLayout);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.GateWeight.Index, new RankedShape(4, 16), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.UpWeight.Index, new RankedShape(4, 16), expectedWeightType);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.GateBias.Index, new RankedShape(16), expectedOutputType);
        AssertPackedBuffer(IR.NTT.PackedMatMulGlu.UpBias.Index, new RankedShape(16), expectedOutputType);
        Assert.Equal(new TensorType(expectedOutputType, new RankedShape(1, 16)), packedCall.CheckedType);

        void AssertPackedBuffer(int index, RankedShape shape, DataType dataType)
        {
            var argument = (Expr)packedCall.Arguments[index];
            Assert.Equal(shape, argument.CheckedShape);
            Assert.Equal(dataType, argument.CheckedDataType);
        }
    }

    [Fact]
    public void TestPackNVFP4MatMulRhsKMajorCreatesTargetPackedAbi()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.BFloat16, new RankedShape(1, 64)));
        var rhs = new Var("rhs", new TensorType(DataTypes.UInt8, new RankedShape(128, 32)));
        var rhsScale = new Var("rhs_scale", new TensorType(DataTypes.Float8E4M3, new RankedShape(128, 4)));
        var lhsGlobalScale = new Var("lhs_global_scale", new TensorType(DataTypes.Float32, new RankedShape(1)));
        var rhsGlobalScale = new Var("rhs_global_scale", new TensorType(DataTypes.Float32, new RankedShape(1)));
        var expr = IR.F.Math.NVFP4MatMul(
            lhs,
            rhs,
            rhsScale,
            lhsGlobalScale,
            rhsGlobalScale,
            DataTypes.BFloat16,
            16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = (Expr)CompilerServices.Rewrite(
            expr,
            [new PackNVFP4MatMulRhsKMajor(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedNVFP4MatMul));

        Assert.False(ReferenceEquals(expr, post));
        Assert.Equal(expr.CheckedType, post.CheckedType);
        AssertPackedBuffer(
            packedCall,
            IR.NTT.PackedNVFP4MatMul.Lhs.Index,
            new RankedShape(1, 8),
            new VectorType(DataTypes.BFloat16, [8]));
        AssertPackedBuffer(
            packedCall,
            IR.NTT.PackedNVFP4MatMul.RhsPacked.Index,
            new RankedShape(128, 1),
            new VectorType(DataTypes.UInt8, [2, 16]));
        Assert.Equal(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(1, 16)),
            packedCall.CheckedType);
    }

    [Fact]
    public void TestPackNVFP4MatMulGluConsumesExistingPackedInput()
    {
        var packedInput = new Var(
            "packed_input",
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new RankedShape(1, 8)));
        var input = Unpack(packedInput, [8], [1]);
        var gateWeight = new Var("gate_weight", new TensorType(DataTypes.UInt8, new RankedShape(128, 32)));
        var upWeight = new Var("up_weight", new TensorType(DataTypes.UInt8, new RankedShape(128, 32)));
        var gateScale = new Var("gate_scale", new TensorType(DataTypes.Float8E4M3, new RankedShape(128, 4)));
        var upScale = new Var("up_scale", new TensorType(DataTypes.Float8E4M3, new RankedShape(128, 4)));
        var inputGlobalScale = new Var("input_global_scale", new TensorType(DataTypes.Float32, new RankedShape(1)));
        var weightGlobalScale = new Var("weight_global_scale", new TensorType(DataTypes.Float32, new RankedShape(1)));
        var expr = IR.F.NN.NVFP4MatMulGlu(
            input,
            gateWeight,
            upWeight,
            gateScale,
            upScale,
            inputGlobalScale,
            inputGlobalScale,
            weightGlobalScale,
            weightGlobalScale,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16,
            16);
        CompilerServices.InferenceType(expr);

        var context = new Nncase.Passes.RunPassContext();
        var post = (Expr)CompilerServices.Rewrite(
            expr,
            [new PackNVFP4MatMulGluRhsKMajor(16, 2)],
            context);
        CompilerServices.InferenceType(post);
        var packedCall = Assert.Single(
            ExprCollector.Collect(post).OfType<Call>().Where(
                call => call.Target is IR.NTT.PackedNVFP4MatMulGlu));

        Assert.Equal(expr.CheckedType, post.CheckedType);
        Assert.Same(
            packedInput,
            packedCall.Arguments[IR.NTT.PackedNVFP4MatMulGlu.Input.Index]);
        AssertPackedBuffer(
            packedCall,
            IR.NTT.PackedNVFP4MatMulGlu.GateWeightPacked.Index,
            new RankedShape(128, 1),
            new VectorType(DataTypes.UInt8, [2, 16]));
        AssertPackedBuffer(
            packedCall,
            IR.NTT.PackedNVFP4MatMulGlu.UpWeightPacked.Index,
            new RankedShape(128, 1),
            new VectorType(DataTypes.UInt8, [2, 16]));
    }

    [Fact]
    public void TestVectorizedMatMulDevectorizePropagation()
    {
        var lhs = Pack(Testing.Rand<float>(3, 24), [8], [1]).Evaluate().AsTensor();
        var lhsVar = new Var(new TensorType(lhs.ElementType, lhs.Shape));
        var rhs = Pack(Testing.Rand<float>(24, 24), [8], [1]).Evaluate().AsTensor();
        var expr = Unpack(lhsVar, [8], [1]);
        expr = VectorizedMatMul(expr, rhs, [], new int[] { 1 });
        expr = Unpack(expr, [8], [1]);
        TestMatched<VectorizedMatMulDevectorizePropagation>(
            expr,
            new Dictionary<IVar, IValue> {
                { lhsVar, Value.FromTensor(lhs) },
            });
    }

    private static void AssertPackedBuffer(
        Call call,
        int index,
        RankedShape shape,
        DataType dataType)
    {
        var argument = Assert.IsAssignableFrom<Expr>(call.Arguments[index]);
        Assert.Equal(shape, argument.CheckedShape);
        Assert.Equal(dataType, argument.CheckedDataType);
    }
}
