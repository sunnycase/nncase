// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;

namespace Nncase.Importer;

/// <summary>
/// Qwen3.5-architecture dense text decoder importer, including Qwen3.8 checkpoints.
/// </summary>
public class Qwen3_5 : HuggingFaceModel
{
    private readonly string _textModelPrefix;
    private Expr? _state;

    public Qwen3_5(string textModelPrefix)
    {
        _textModelPrefix = textModelPrefix;
    }

    protected override bool RequiresPagedAttentionKVCache => Enumerable.Range(
        0,
        checked((int)Config.GetNestedValue<long>("num_hidden_layers")))
        .Any(index => GetLayerType(index) == "full_attention");

    protected override bool RequiresRotaryEmbedding => RequiresPagedAttentionKVCache;

    protected override bool DecoderLayerUsesRotaryEmbedding(int layerIndex)
        => GetLayerType(layerIndex) == "full_attention";

    public override (IEnumerable<IVar> Inputs, Dictionary<IVar, Dimension[]> VarMap) CreateInputs()
    {
        ValidateSupportedConfiguration();
        var (inputs, varMap) = base.CreateInputs();
        if (!Context!.FixVarMap!.TryGetValue("sequence_length", out var sequenceLength) || sequenceLength != 1)
        {
            throw new NotSupportedException(
                "Qwen3.5 single-layer support requires shape_bucket_fix_var_map.sequence_length=1.");
        }

        var stateType = GetRequestedTensorType() ?? DataTypes.BFloat16;
        if (stateType is not PrimType statePrimType)
        {
            throw new NotSupportedException($"Qwen3.5 GDN state requires a primitive activation type, got {stateType}.");
        }

        var numLayers = Config.GetNestedValue<long>("num_hidden_layers");
        var numKeyHeads = Config.GetNestedValue<long>("linear_num_key_heads");
        var numValueHeads = Config.GetNestedValue<long>("linear_num_value_heads");
        var keyHeadDim = Config.GetNestedValue<long>("linear_key_head_dim");
        var valueHeadDim = Config.GetNestedValue<long>("linear_value_head_dim");
        var convKernelSize = Config.GetNestedValue<long>("linear_conv_kernel_dim");
        var hiddenSize = Config.GetNestedValue<long>("hidden_size");
        var activationLane = checked(16 / statePrimType.SizeInBytes);
        var recurrentLane = checked(16 / DataTypes.Float32.SizeInBytes);
        var stateConfig = new GatedDeltaNetStateConfig(
            checked((int)numLayers),
            checked((int)numKeyHeads),
            checked((int)numValueHeads),
            checked((int)keyHeadDim),
            checked((int)valueHeadDim),
            checked((int)convKernelSize),
            checked((int)hiddenSize),
            statePrimType,
            [activationLane],
            [
                GatedDeltaNetStateDimKind.NumLayers,
                GatedDeltaNetStateDimKind.ConvChannels,
                GatedDeltaNetStateDimKind.ConvHistory,
            ],
            [GatedDeltaNetStateDimKind.ConvChannels],
            [activationLane],
            [
                GatedDeltaNetStateDimKind.NumLayers,
                GatedDeltaNetStateDimKind.NumValueHeads,
                GatedDeltaNetStateDimKind.ValueHeadDim,
                GatedDeltaNetStateDimKind.KeyHeadDim,
            ],
            [GatedDeltaNetStateDimKind.KeyHeadDim],
            [recurrentLane]);
        var state = new Var(
            "gated_delta_net_state",
            TensorType.Scalar(new ReferenceType(new GatedDeltaNetStateType { Config = stateConfig })));
        _state = state;
        Context.Inputs!.Add(state);

        return (inputs.Concat(new IVar[] { state }), varMap);
    }

    public override System.Tuple<Expr, Expr?> LLMModel(Expr inputIds, Expr pastKeyValues)
        => base.LLMModel(inputIds, pastKeyValues);

    public override System.Tuple<Expr, Expr> DecodeLayer(
        int count,
        Expr hiddenStates,
        Expr pastKeyValues,
        System.Tuple<Expr, Expr> positionEmbeddings,
        Dimension layerId)
    {
        if (GetLayerType(count) == "full_attention")
        {
            return base.DecodeLayer(count, hiddenStates, pastKeyValues, positionEmbeddings, layerId);
        }

        var residual = hiddenStates;
        hiddenStates = LLMLayerNorm(hiddenStates, $"model.layers.{count}.input_layernorm.weight");
        hiddenStates = BuildGatedDeltaNet(count, hiddenStates, layerId);
        hiddenStates = residual + hiddenStates;

        residual = hiddenStates;
        hiddenStates = LLMLayerNorm(hiddenStates, $"model.layers.{count}.post_attention_layernorm.weight");
        hiddenStates = residual + LLMMlp(count, hiddenStates);
        return System.Tuple.Create(hiddenStates, pastKeyValues);
    }

    protected override QKVProjection QKVCompute(
        int count,
        Expr hiddenStates,
        Dimension seqLen,
        Dimension headDim)
    {
        var hiddenShape = new RankedShape(seqLen, -1L, headDim);
        var (queryAndGateStates, keyStates, valueStates) = BuildQKVParallelLinear(
            count,
            hiddenStates,
            hiddenShape);
        var numHeads = Config.GetNestedValue<long>("num_attention_heads");
        var queryAndGateShape = new RankedShape(seqLen, numHeads, 2, headDim);
        queryAndGateStates = Tensors.Reshape(queryAndGateStates, queryAndGateShape);
        var queryStates = Tensors.Reshape(
            Tensors.Slice(queryAndGateStates, [0L], [1L], [2L], [1L]),
            new RankedShape(seqLen, numHeads, headDim));
        var outputGate = Tensors.Reshape(
            Tensors.Slice(queryAndGateStates, [1L], [2L], [2L], [1L]),
            new RankedShape(seqLen, numHeads, headDim));
        queryStates = LLMLayerNorm(
            queryStates,
            $"model.layers.{count}.self_attn.q_norm.weight");
        keyStates = LLMLayerNorm(
            keyStates,
            $"model.layers.{count}.self_attn.k_norm.weight");
        return new(queryStates, keyStates, valueStates, outputGate);
    }

    public override Call LLMLayerNorm(Expr hiddenStates, string layerName)
    {
        var originType = hiddenStates.CheckedDataType;
        var weightTensor = GetWeight(layerName)
            ?? throw new InvalidOperationException($"Required weight {layerName} is missing.");
        var scale = GetLayerWeightExpr(layerName, tensor => PrepareRmsNormScale(tensor, originType));
        var bias = Tensor.Zeros(originType, weightTensor.Dimensions);
        var epsilon = (float)Config.GetNestedValue<double>("rms_norm_eps");
        return NN.LayerNorm(-1, epsilon, hiddenStates, scale, bias, false)
            .With(metadata: new IRMetadata { OutputNames = new[] { layerName[..^7] } });
    }

    private static Tensor PrepareRmsNormScale(Tensor weight, DataType dataType) => dataType switch
    {
        var type when type == DataTypes.Float16 => AddOne(weight.Cast<Half>()),
        var type when type == DataTypes.BFloat16 => AddOne(weight.Cast<BFloat16>()),
        var type when type == DataTypes.Float32 => AddOne(weight.Cast<float>()),
        _ => throw new NotSupportedException(
            $"Qwen3.5 RMSNorm scale preparation does not support {dataType}."),
    };

    private static Tensor<T> AddOne<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, INumber<T>
    {
        var output = input.Clone();
        var values = output.Buffer.Span;
        for (var index = 0; index < values.Length; index++)
        {
            values[index] += T.One;
        }

        return output;
    }

    public override Call LLMMlp(int count, Expr hiddenStates)
    {
        if (TryGetNVFP4Config(out var groupSize) && HasNVFP4Projection($"model.layers.{count}.mlp.down_proj"))
        {
            var prefix = $"model.layers.{count}.mlp.down_proj";
            return IR.F.Math.NVFP4MatMul(
                    BuildMatMulGlu(count, hiddenStates),
                    RequireLayerWeight($"{prefix}.weight_packed"),
                    RequireLayerWeight($"{prefix}.weight_scale"),
                    RequireLayerWeight($"{prefix}.input_global_scale"),
                    RequireLayerWeight($"{prefix}.weight_global_scale"),
                    hiddenStates.CheckedDataType,
                    groupSize)
                .With(metadata: new IRMetadata { OutputNames = new[] { prefix } });
        }

        if (!TryGetFp8ProjectionConfig(out var blockN, out var blockK, out var compressed))
        {
            return base.LLMMlp(count, hiddenStates);
        }

        return BlockScaledLinearByName(
            BuildMatMulGlu(count, hiddenStates),
            $"model.layers.{count}.mlp.down_proj.weight",
            $"model.layers.{count}.mlp.down_proj.{(compressed ? "weight_scale" : "weight_scale_inv")}",
            blockN,
            blockK,
            compressed,
            $"model.layers.{count}.mlp.down_proj");
    }

    protected override string ResolveWeightName(string canonicalWeightName)
    {
        const string canonicalModelPrefix = "model.";
        return canonicalWeightName.StartsWith(canonicalModelPrefix, StringComparison.Ordinal)
            ? $"{_textModelPrefix}.{canonicalWeightName[canonicalModelPrefix.Length..]}"
            : canonicalWeightName;
    }

    protected override Call BuildAttentionOutputProjection(int count, Expr input)
    {
        if (!TryGetFp8ProjectionConfig(out var blockN, out var blockK, out var compressed))
        {
            return base.BuildAttentionOutputProjection(count, input);
        }

        var prefix = $"model.layers.{count}.self_attn.o_proj";
        var scaleSuffix = compressed ? "weight_scale" : "weight_scale_inv";
        return BlockScaledLinearByName(
            input,
            $"{prefix}.weight",
            $"{prefix}.{scaleSuffix}",
            blockN,
            blockK,
            compressed,
            prefix);
    }

    protected override Call BuildMatMulGlu(int count, Expr hiddenStates)
    {
        if (TryGetNVFP4Config(out var groupSize) &&
            HasNVFP4Projection($"model.layers.{count}.mlp.gate_proj") &&
            HasNVFP4Projection($"model.layers.{count}.mlp.up_proj"))
        {
            var gatePrefix = $"model.layers.{count}.mlp.gate_proj";
            var upPrefix = $"model.layers.{count}.mlp.up_proj";
            return IR.F.NN.NVFP4MatMulGlu(
                    hiddenStates,
                    RequireLayerWeight($"{gatePrefix}.weight_packed"),
                    RequireLayerWeight($"{upPrefix}.weight_packed"),
                    RequireLayerWeight($"{gatePrefix}.weight_scale"),
                    RequireLayerWeight($"{upPrefix}.weight_scale"),
                    RequireLayerWeight($"{gatePrefix}.input_global_scale"),
                    RequireLayerWeight($"{upPrefix}.input_global_scale"),
                    RequireLayerWeight($"{gatePrefix}.weight_global_scale"),
                    RequireLayerWeight($"{upPrefix}.weight_global_scale"),
                    GetMlpGluType(),
                    hiddenStates.CheckedDataType,
                    groupSize)
                .With(metadata: new IRMetadata
                {
                    OutputNames = new[] { $"model.layers.{count}.mlp.gate_up_proj" },
                });
        }

        if (!TryGetFp8ProjectionConfig(out var blockN, out var blockK, out var compressed))
        {
            return base.BuildMatMulGlu(count, hiddenStates);
        }

        var gateWeightName = $"model.layers.{count}.mlp.gate_proj.weight";
        var upWeightName = $"model.layers.{count}.mlp.up_proj.weight";
        var scaleSuffix = compressed ? "weight_scale" : "weight_scale_inv";
        var gateScaleName = $"model.layers.{count}.mlp.gate_proj.{scaleSuffix}";
        var upScaleName = $"model.layers.{count}.mlp.up_proj.{scaleSuffix}";
        Expr PrepareWeight(string name) =>
            GetLayerWeightExpr(name, tensor => PrepareLinearWeightTensor(tensor, DataTypes.Float8E4M3));
        Expr RequireScale(string name) => GetLayerWeightExpr(
            name,
            tensor => PrepareFp8WeightScale(tensor, name, hiddenStates.CheckedShape[^1].FixedValue, blockK, compressed));

        return IR.F.NN.MatMulGlu(
            hiddenStates,
            PrepareWeight(gateWeightName),
            PrepareWeight(upWeightName),
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            RequireScale(gateScaleName),
            RequireScale(upScaleName),
            GetMlpGluType(),
            hiddenStates.CheckedDataType,
            MatMulQuantizationMode.DynamicBlock,
            blockN,
            blockK)
            .With(metadata: new IRMetadata
            {
                OutputNames = new[] { $"model.layers.{count}.mlp.gate_up_proj" },
            });
    }

    protected override string? GetDecoderLayerStructureKey(int layerIndex)
    {
        return $"{GetLayerType(layerIndex)}_{GetDecoderPrecisionKey(layerIndex)}";
    }

    private string GetDecoderPrecisionKey(int layerIndex)
        => HasNVFP4Projection($"model.layers.{layerIndex}.mlp.down_proj")
            ? "nvfp4"
            : "fp8";

    private Expr BuildGatedDeltaNet(int count, Expr hiddenStates, Dimension layerId)
    {
        var inputType = hiddenStates.CheckedDataType;
        var numKeyHeads = Config.GetNestedValue<long>("linear_num_key_heads");
        var numValueHeads = Config.GetNestedValue<long>("linear_num_value_heads");
        var keyHeadDim = Config.GetNestedValue<long>("linear_key_head_dim");
        var valueHeadDim = Config.GetNestedValue<long>("linear_value_head_dim");
        var convKernelSize = Config.GetNestedValue<long>("linear_conv_kernel_dim");
        var keyDim = numKeyHeads * keyHeadDim;
        var valueDim = numValueHeads * valueHeadDim;
        var convDim = (keyDim * 2) + valueDim;
        var prefix = $"model.layers.{count}.linear_attn";
        var modelState = _state ?? throw new InvalidOperationException("Qwen3.5 GDN state input is unavailable.");
        var state = GetDecoderLayerResource("gated_delta_net_state", modelState);
        Expr PrepareLinearWeight(string name) =>
            GetLayerWeightExpr(name, tensor => PrepareLinearWeightTensor(tensor, inputType));
        var hasBlockFp8 = TryGetFp8ProjectionConfig(out var blockN, out var blockK, out var hasCompressedFp8);
        Expr PrepareBlockWeight(string name) =>
            GetLayerWeightExpr(name, tensor => PrepareLinearWeightTensor(tensor, DataTypes.Float8E4M3));
        Expr PrepareProjectionWeight(string name) => hasBlockFp8
            ? PrepareBlockWeight(name)
            : PrepareLinearWeight(name);
        Expr PrepareProjectionScale(string name, long reductionExtent) => hasBlockFp8
            ? GetLayerWeightExpr(
                name,
                tensor => PrepareFp8WeightScale(tensor, name, reductionExtent, blockK, hasCompressedFp8))
            : None.Default;

        var convWeightTensor = GetWeight($"{prefix}.conv1d.weight")
            ?? throw new InvalidOperationException($"Required weight {prefix}.conv1d.weight is missing.");
        var convWeight = Tensors.Reshape(
            GetLayerWeightExpr($"{prefix}.conv1d.weight", tensor => tensor.CastTo(inputType)),
            new RankedShape(convDim, convKernelSize));
        var result = NN.GatedDeltaNet(
            hiddenStates,
            state,
            PrepareProjectionWeight($"{prefix}.in_proj_qkv.weight"),
            PrepareProjectionWeight($"{prefix}.in_proj_z.weight"),
            PrepareLinearWeight($"{prefix}.in_proj_b.weight"),
            PrepareLinearWeight($"{prefix}.in_proj_a.weight"),
            convWeight,
            RequireLayerWeight($"{prefix}.A_log"),
            RequireLayerWeight($"{prefix}.dt_bias"),
            RequireLayerWeight($"{prefix}.norm.weight"),
            PrepareProjectionWeight($"{prefix}.out_proj.weight"),
            layerId,
            numKeyHeads,
            numValueHeads,
            keyHeadDim,
            valueHeadDim,
            convKernelSize,
            (float)Config.GetNestedValue<double>("rms_norm_eps"),
            qkvWeightScale: PrepareProjectionScale(
                $"{prefix}.in_proj_qkv.{(hasCompressedFp8 ? "weight_scale" : "weight_scale_inv")}",
                hiddenStates.CheckedShape[^1].FixedValue),
            zWeightScale: PrepareProjectionScale(
                $"{prefix}.in_proj_z.{(hasCompressedFp8 ? "weight_scale" : "weight_scale_inv")}",
                hiddenStates.CheckedShape[^1].FixedValue),
            outputWeightScale: PrepareProjectionScale(
                $"{prefix}.out_proj.{(hasCompressedFp8 ? "weight_scale" : "weight_scale_inv")}",
                valueDim),
            quantizationMode: hasBlockFp8
                ? MatMulQuantizationMode.DynamicBlock
                : MatMulQuantizationMode.None,
            weightBlockN: blockN,
            weightBlockK: blockK)
            .With(metadata: new IRMetadata
            {
                OutputNames = new[] { prefix },
                SemanticRegion = new SemanticRegion(SemanticRegionKinds.Attention, prefix),
            });
        if (result.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException(
                $"Imported {prefix} has an invalid type: {invalid.Reason}");
        }

        return result[0];
    }

    private Call BlockScaledLinearByName(
        Expr input,
        string weightName,
        string weightScaleName,
        long blockN,
        long blockK,
        bool compressed,
        string layerName)
    {
        var weight = GetLayerWeightExpr(
            weightName,
            tensor => PrepareLinearWeightTensor(tensor, DataTypes.Float8E4M3));
        var scale = GetLayerWeightExpr(
            weightScaleName,
            tensor => PrepareFp8WeightScale(
                tensor,
                weightScaleName,
                input.CheckedShape[^1].FixedValue,
                blockK,
                compressed));
        return IR.F.Math.BlockScaledMatMul(
                input,
                weight,
                scale,
                input.CheckedDataType,
                blockN,
                blockK)
            .With(metadata: new IRMetadata { OutputNames = new[] { layerName } });
    }

    private bool TryGetFp8ProjectionConfig(
        out long blockN,
        out long blockK,
        out bool compressed)
    {
        if (TryGetDynamicBlockFp8Config(out blockN, out blockK))
        {
            compressed = false;
            return true;
        }

        compressed = TryGetCompressedTensorFp8Config(out blockN, out blockK);
        return compressed;
    }

    private static Tensor PrepareFp8WeightScale(
        Tensor scale,
        string name,
        long reductionExtent,
        long blockK,
        bool compressed)
    {
        if (!compressed)
        {
            return scale;
        }

        if (scale.Rank != 2 || scale.Dimensions[1] != 1 ||
            blockK <= 0 || reductionExtent % blockK != 0)
        {
            throw new InvalidOperationException(
                $"Compressed-tensors FP8 scale {name} must have shape [N,1], and K={reductionExtent} " +
                $"must be divisible by block K {blockK}; got shape " +
                $"[{string.Join(",", scale.Dimensions.ToArray())}].");
        }

        var repeats = reductionExtent / blockK;
        var expanded = Tensor.Zeros(scale.ElementType, [scale.Dimensions[0], repeats]);
        var elementSize = scale.ElementType.SizeInBytes;
        for (var row = 0L; row < scale.Dimensions[0]; row++)
        {
            var source = scale.BytesBuffer.Slice(checked((int)(row * elementSize)), elementSize);
            for (var column = 0L; column < repeats; column++)
            {
                source.CopyTo(
                    expanded.BytesBuffer.Slice(
                        checked((int)(((row * repeats) + column) * elementSize)),
                        elementSize));
            }
        }

        return expanded;
    }

    private bool TryGetDynamicBlockFp8Config(out long blockN, out long blockK)
    {
        blockN = 0;
        blockK = 0;
        if (!Config.TryGetValue("quantization_config", out var value) ||
            value is not Dictionary<string, object> quantization)
        {
            return false;
        }

        var method = quantization.GetNestedValue<string>("quant_method");
        if (string.Equals(method, "compressed-tensors", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var activationScheme = quantization.GetNestedValue<string>("activation_scheme");
        var format = quantization.GetNestedValue<string>("fmt");
        if (!string.Equals(method, "fp8", StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(activationScheme, "dynamic", StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(format, "e4m3", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Qwen3.5 quantization_config is unsupported: quant_method={method}, " +
                $"activation_scheme={activationScheme}, fmt={format}.");
        }

        blockN = quantization.GetNestedValue<long>("weight_block_size", 0);
        blockK = quantization.GetNestedValue<long>("weight_block_size", 1);
        if (blockN <= 0 || blockK <= 0)
        {
            throw new InvalidOperationException(
                $"Qwen3.5 FP8 weight block size must be positive, got [{blockN}, {blockK}].");
        }

        return true;
    }

    private bool TryGetNVFP4Config(out long groupSize)
    {
        groupSize = 0;
        if (!TryGetCompressedTensorsConfig(out var quantization))
        {
            return false;
        }

        var format = quantization.GetNestedValue<string>("config_groups", "group_1", "format");
        var weightBits = quantization.GetNestedValue<long>("config_groups", "group_1", "weights", "num_bits");
        var activationBits = quantization.GetNestedValue<long>("config_groups", "group_1", "input_activations", "num_bits");
        var weightGroupSize = quantization.GetNestedValue<long>("config_groups", "group_1", "weights", "group_size");
        var activationGroupSize = quantization.GetNestedValue<long>("config_groups", "group_1", "input_activations", "group_size");
        var scaleType = quantization.GetNestedValue<string>("config_groups", "group_1", "weights", "scale_dtype");
        if (!string.Equals(format, "nvfp4-pack-quantized", StringComparison.OrdinalIgnoreCase) ||
            weightBits != 4 || activationBits != 4 ||
            weightGroupSize != activationGroupSize || weightGroupSize != 16 ||
            !string.Equals(scaleType, "torch.float8_e4m3fn", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Qwen3.5 compressed-tensors NVFP4 group is unsupported: format={format}, " +
                $"bits={weightBits}/{activationBits}, group_size={weightGroupSize}/{activationGroupSize}, " +
                $"scale_dtype={scaleType}.");
        }

        groupSize = weightGroupSize;
        return true;
    }

    private bool TryGetCompressedTensorFp8Config(out long blockN, out long blockK)
    {
        blockN = 0;
        blockK = 0;
        if (!TryGetCompressedTensorsConfig(out var quantization))
        {
            return false;
        }

        var format = quantization.GetNestedValue<string>("config_groups", "group_0", "format");
        var weightBits = quantization.GetNestedValue<long>("config_groups", "group_0", "weights", "num_bits");
        var weightStrategy = quantization.GetNestedValue<string>("config_groups", "group_0", "weights", "strategy");
        var activationBits = quantization.GetNestedValue<long>("config_groups", "group_0", "input_activations", "num_bits");
        var activationStrategy = quantization.GetNestedValue<string>("config_groups", "group_0", "input_activations", "strategy");
        if (!string.Equals(format, "float-quantized", StringComparison.OrdinalIgnoreCase) ||
            weightBits != 8 || activationBits != 8 ||
            !string.Equals(weightStrategy, "channel", StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(activationStrategy, "token", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Qwen3.5 compressed-tensors FP8 group is unsupported: format={format}, " +
                $"bits={weightBits}/{activationBits}, strategies={weightStrategy}/{activationStrategy}.");
        }

        var hiddenSize = Config.GetNestedValue<long>("hidden_size");
        var valueProjectionSize = Config.GetNestedValue<long>("linear_num_value_heads") *
            Config.GetNestedValue<long>("linear_value_head_dim");
        blockN = 1;
        blockK = GreatestCommonDivisor(hiddenSize, valueProjectionSize);
        return true;
    }

    private bool TryGetCompressedTensorsConfig(out Dictionary<string, object> quantization)
    {
        quantization = null!;
        if (!Config.TryGetValue("quantization_config", out var value) ||
            value is not Dictionary<string, object> candidate)
        {
            return false;
        }

        var method = candidate.GetNestedValue<string>("quant_method");
        if (!string.Equals(method, "compressed-tensors", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var format = candidate.GetNestedValue<string>("format");
        if (!string.Equals(format, "mixed-precision", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Qwen3.5 compressed-tensors format must be mixed-precision, got {format}.");
        }

        quantization = candidate;
        return true;
    }

    private static long GreatestCommonDivisor(long lhs, long rhs)
    {
        while (rhs != 0)
        {
            (lhs, rhs) = (rhs, lhs % rhs);
        }

        return System.Math.Abs(lhs);
    }

    private Tensor RequireWeight(string name) => GetWeight(name)
        ?? throw new InvalidOperationException($"Required weight {name} is missing.");

    private Expr RequireLayerWeight(string name) => GetLayerWeightExpr(name, tensor => tensor);

    private bool HasNVFP4Projection(string prefix)
        => WeightToFileMap.ContainsKey(ResolveWeightName($"{prefix}.weight_packed"));

    private string GetLayerType(int index)
    {
        var layerType = Config.GetNestedValue<string>("layer_types", index);
        return layerType switch
        {
            "linear_attention" or "full_attention" => layerType,
            _ => throw new NotSupportedException(
                $"Qwen3.5 layer {index} has unsupported layer type {layerType}."),
        };
    }

    private void ValidateSupportedConfiguration()
    {
        var numLayers = checked((int)Config.GetNestedValue<long>("num_hidden_layers"));
        if (numLayers <= 0)
        {
            throw new InvalidOperationException(
                $"Qwen3.5 num_hidden_layers must be positive, got {numLayers}.");
        }

        _ = Enumerable.Range(0, numLayers).Select(GetLayerType).ToArray();

        var numKeyHeads = Config.GetNestedValue<long>("linear_num_key_heads");
        var numValueHeads = Config.GetNestedValue<long>("linear_num_value_heads");
        if (numValueHeads % numKeyHeads != 0)
        {
            throw new InvalidOperationException(
                $"linear_num_value_heads ({numValueHeads}) must be divisible by linear_num_key_heads ({numKeyHeads}).");
        }
    }
}
