// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
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

    protected override bool SupportsDecoderLayerFunctionReuse => false;

    protected override bool RequiresPagedAttentionKVCache => false;

    public override (IEnumerable<IVar> Inputs, Dictionary<IVar, Dimension[]> VarMap) CreateInputs()
    {
        ValidateSupportedConfiguration();
        if (ImportOptions.HuggingFaceOptions.EnableSampler)
        {
            throw new NotSupportedException("Qwen3.5 recurrent-state outputs are not yet compatible with the fused sampler ABI.");
        }

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
    {
        var embedTokensWeight = GetWeight("model.embed_tokens.weight")
            ?? throw new InvalidOperationException("Required weight model.embed_tokens.weight is missing.");
        var requestedTensorType = GetRequestedTensorType();
        if (requestedTensorType is not null)
        {
            embedTokensWeight = embedTokensWeight.CastTo(requestedTensorType);
        }

        Expr hiddenStates;
        if (inputIds.CheckedShape.Rank > 2 && inputIds.CheckedDataType.IsFloat())
        {
            hiddenStates = inputIds;
        }
        else
        {
            long? paddingIndex = Config.TryGetValue("pad_token_id", out var value) && value is long index
                ? index
                : null;
            hiddenStates = Embedding(inputIds, embedTokensWeight, paddingIndex);
        }

        Expr? allHiddenStates = null;
        if (ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            allHiddenStates = Tensors.Unsqueeze(hiddenStates, new long[] { 0 });
        }

        var (layerOutput, _) = DecodeLayer(
            0,
            hiddenStates,
            pastKeyValues,
            System.Tuple.Create(hiddenStates, hiddenStates),
            new DimConst(0));
        var lastHiddenStates = LLMLayerNorm(layerOutput, "model.norm.weight");

        if (ImportOptions.HuggingFaceOptions.OutputHiddenStates)
        {
            allHiddenStates = Tensors.Concat(
                new Nncase.IR.Tuple(allHiddenStates!, Tensors.Unsqueeze(lastHiddenStates, new long[] { 0 })),
                0);
        }

        return System.Tuple.Create<Expr, Expr?>((Expr)lastHiddenStates, allHiddenStates);
    }

    public override System.Tuple<Expr, Expr> DecodeLayer(
        int count,
        Expr hiddenStates,
        Expr pastKeyValues,
        System.Tuple<Expr, Expr> positionEmbeddings,
        Dimension layerId)
    {
        if (count != 0)
        {
            throw new NotSupportedException("Qwen3.5 support currently covers exactly the first decoder layer.");
        }

        var residual = hiddenStates;
        hiddenStates = LLMLayerNorm(hiddenStates, $"model.layers.{count}.input_layernorm.weight");
        hiddenStates = BuildGatedDeltaNet(count, hiddenStates);
        hiddenStates = residual + hiddenStates;

        residual = hiddenStates;
        hiddenStates = LLMLayerNorm(hiddenStates, $"model.layers.{count}.post_attention_layernorm.weight");
        hiddenStates = LLMMlp(count, hiddenStates);
        hiddenStates = residual + hiddenStates;
        return System.Tuple.Create(hiddenStates, pastKeyValues);
    }

    public override Call LLMLayerNorm(Expr hiddenStates, string layerName)
    {
        var originType = hiddenStates.CheckedDataType;
        var weightTensor = GetWeight(layerName)
            ?? throw new InvalidOperationException($"Required weight {layerName} is missing.");
        var weight = weightTensor.CastTo(originType);
        var scale = IR.F.Math.Add(weight, Tensor.FromScalar(1.0f).CastTo(originType));
        var bias = Tensor.Zeros(originType, weightTensor.Dimensions);
        var epsilon = (float)Config.GetNestedValue<double>("rms_norm_eps");
        return NN.LayerNorm(-1, epsilon, hiddenStates, scale, bias, false)
            .With(metadata: new IRMetadata { OutputNames = new[] { layerName[..^7] } });
    }

    public override Call LLMMlp(int count, Expr hiddenStates)
    {
        if (TryGetNVFP4Config(out var groupSize))
        {
            var prefix = $"model.layers.{count}.mlp.down_proj";
            return IR.F.Math.NVFP4MatMul(
                    BuildMatMulGlu(count, hiddenStates),
                    RequireWeight($"{prefix}.weight_packed"),
                    RequireWeight($"{prefix}.weight_scale"),
                    RequireWeight($"{prefix}.input_global_scale"),
                    RequireWeight($"{prefix}.weight_global_scale"),
                    hiddenStates.CheckedDataType,
                    groupSize)
                .With(metadata: new IRMetadata { OutputNames = new[] { prefix } });
        }

        if (!TryGetDynamicBlockFp8Config(out var blockN, out var blockK))
        {
            return base.LLMMlp(count, hiddenStates);
        }

        return BlockScaledLinearByName(
            BuildMatMulGlu(count, hiddenStates),
            $"model.layers.{count}.mlp.down_proj.weight",
            $"model.layers.{count}.mlp.down_proj.weight_scale_inv",
            blockN,
            blockK,
            $"model.layers.{count}.mlp.down_proj");
    }

    protected override string ResolveWeightName(string canonicalWeightName)
    {
        const string canonicalModelPrefix = "model.";
        return canonicalWeightName.StartsWith(canonicalModelPrefix, StringComparison.Ordinal)
            ? $"{_textModelPrefix}.{canonicalWeightName[canonicalModelPrefix.Length..]}"
            : canonicalWeightName;
    }

    protected override Call BuildMatMulGlu(int count, Expr hiddenStates)
    {
        if (TryGetNVFP4Config(out var groupSize))
        {
            var gatePrefix = $"model.layers.{count}.mlp.gate_proj";
            var upPrefix = $"model.layers.{count}.mlp.up_proj";
            return IR.F.NN.NVFP4MatMulGlu(
                    hiddenStates,
                    RequireWeight($"{gatePrefix}.weight_packed"),
                    RequireWeight($"{upPrefix}.weight_packed"),
                    RequireWeight($"{gatePrefix}.weight_scale"),
                    RequireWeight($"{upPrefix}.weight_scale"),
                    RequireWeight($"{gatePrefix}.input_global_scale"),
                    RequireWeight($"{upPrefix}.input_global_scale"),
                    RequireWeight($"{gatePrefix}.weight_global_scale"),
                    RequireWeight($"{upPrefix}.weight_global_scale"),
                    GetMlpGluType(),
                    hiddenStates.CheckedDataType,
                    groupSize)
                .With(metadata: new IRMetadata
                {
                    OutputNames = new[] { $"model.layers.{count}.mlp.gate_up_proj" },
                });
        }

        if (!TryGetDynamicBlockFp8Config(out var blockN, out var blockK))
        {
            return base.BuildMatMulGlu(count, hiddenStates);
        }

        var gateWeightName = $"model.layers.{count}.mlp.gate_proj.weight";
        var upWeightName = $"model.layers.{count}.mlp.up_proj.weight";
        var gateScaleName = $"model.layers.{count}.mlp.gate_proj.weight_scale_inv";
        var upScaleName = $"model.layers.{count}.mlp.up_proj.weight_scale_inv";
        Tensor RequireWeight(string name) => GetWeight(name)
            ?? throw new InvalidOperationException($"Required weight {name} is missing.");
        Expr PrepareWeight(string name) =>
            PrepareLinearWeightTensor(RequireWeight(name), DataTypes.Float8E4M3);
        Expr RequireScale(string name) => RequireWeight(name);

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

    private Expr BuildGatedDeltaNet(int count, Expr hiddenStates)
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
        var state = _state ?? throw new InvalidOperationException("Qwen3.5 GDN state input is unavailable.");
        Tensor PrepareLinearWeight(string name) => PrepareLinearWeightTensor(RequireWeight(name), inputType);
        var hasLegacyBlockFp8 = TryGetDynamicBlockFp8Config(out var blockN, out var blockK);
        var hasCompressedFp8 = !hasLegacyBlockFp8 && TryGetCompressedTensorFp8Config(out blockN, out blockK);
        var hasBlockFp8 = hasLegacyBlockFp8 || hasCompressedFp8;
        Tensor PrepareBlockWeight(string name) => PrepareLinearWeightTensor(
            RequireWeight(name),
            DataTypes.Float8E4M3);
        Expr PrepareProjectionWeight(string name) => hasBlockFp8
            ? PrepareBlockWeight(name)
            : PrepareLinearWeight(name);
        Expr PrepareProjectionScale(string name, long reductionExtent)
        {
            if (!hasBlockFp8)
            {
                return None.Default;
            }

            var scale = RequireWeight(name);
            if (!hasCompressedFp8)
            {
                return scale;
            }

            if (scale.Rank != 2 || scale.Dimensions[1] != 1 || reductionExtent % blockK != 0)
            {
                throw new InvalidOperationException(
                    $"Compressed-tensors FP8 scale {name} must have shape [N,1] and K " +
                    $"must divide block K {blockK}, got shape [{string.Join(",", scale.Dimensions.ToArray())}] " +
                    $"and K={reductionExtent}.");
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

        var convWeightTensor = GetWeight($"{prefix}.conv1d.weight")
            ?? throw new InvalidOperationException($"Required weight {prefix}.conv1d.weight is missing.");
        var convWeight = Tensors.Reshape(
            convWeightTensor.CastTo(inputType),
            new RankedShape(convDim, convKernelSize));
        var result = NN.GatedDeltaNet(
            hiddenStates,
            state,
            PrepareProjectionWeight($"{prefix}.in_proj_qkv.weight"),
            PrepareProjectionWeight($"{prefix}.in_proj_z.weight"),
            PrepareLinearWeight($"{prefix}.in_proj_b.weight"),
            PrepareLinearWeight($"{prefix}.in_proj_a.weight"),
            convWeight,
            RequireWeight($"{prefix}.A_log"),
            RequireWeight($"{prefix}.dt_bias"),
            RequireWeight($"{prefix}.norm.weight"),
            PrepareProjectionWeight($"{prefix}.out_proj.weight"),
            new DimConst(count),
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
            .With(metadata: new IRMetadata { OutputNames = new[] { prefix } });
        if (result.CheckedType is InvalidType invalid)
        {
            throw new InvalidOperationException(
                $"Imported {prefix} has an invalid type: {invalid.Reason}");
        }

        _state = result[1];
        return result[0];
    }

    private Call BlockScaledLinearByName(
        Expr input,
        string weightName,
        string weightScaleName,
        long blockN,
        long blockK,
        string layerName)
    {
        var weightTensor = GetWeight(weightName)
            ?? throw new InvalidOperationException($"Required weight {weightName} is missing.");
        var scale = GetWeight(weightScaleName)
            ?? throw new InvalidOperationException($"Required weight {weightScaleName} is missing.");
        var weight = PrepareLinearWeightTensor(weightTensor, DataTypes.Float8E4M3);
        return IR.F.Math.BlockScaledMatMul(
                input,
                weight,
                scale,
                input.CheckedDataType,
                blockN,
                blockK)
            .With(metadata: new IRMetadata { OutputNames = new[] { layerName } });
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

    private void ValidateSupportedConfiguration()
    {
        if (Config.GetNestedValue<long>("num_hidden_layers") != 1)
        {
            throw new NotSupportedException(
                "Qwen3.5 support currently requires huggingface_options.num_layers=1.");
        }

        if (Config.GetNestedValue<string>("layer_types", 0) != "linear_attention")
        {
            throw new NotSupportedException("The supported Qwen3.5 layer must be the first linear_attention layer.");
        }

        var numKeyHeads = Config.GetNestedValue<long>("linear_num_key_heads");
        var numValueHeads = Config.GetNestedValue<long>("linear_num_value_heads");
        if (numValueHeads % numKeyHeads != 0)
        {
            throw new InvalidOperationException(
                $"linear_num_value_heads ({numValueHeads}) must be divisible by linear_num_key_heads ({numKeyHeads}).");
        }
    }
}
