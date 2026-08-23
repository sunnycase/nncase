// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;

namespace Nncase.Importer;

/// <summary>
/// Qwen3.5 MoE text decoder importer.
/// </summary>
public sealed class Qwen3_5Moe : HuggingFaceModel
{
    private readonly string _textModelPrefix;
    private Expr? _state;

    public Qwen3_5Moe(string textModelPrefix)
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
                GatedDeltaNetStateDimKind.KeyHeadDim,
                GatedDeltaNetStateDimKind.ValueHeadDim,
            ],
            [GatedDeltaNetStateDimKind.ValueHeadDim],
            [recurrentLane]);
        var state = new Var(
            "gated_delta_net_state",
            TensorType.Scalar(new ReferenceType(new GatedDeltaNetStateType { Config = stateConfig })));
        _state = state;
        Context.Inputs!.Add(state);

        return (inputs.Concat(new IVar[] { state }), varMap);
    }

    public override BaseExpr CreateOutputs()
    {
        return base.CreateOutputs();
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
        var numExperts = Config.GetNestedValue<long>("num_experts");
        var numTopK = Config.GetNestedValue<long>("num_experts_per_tok");
        var hiddenSize = Config.GetNestedValue<long>("hidden_size");
        var intermediateSize = Config.GetNestedValue<long>("moe_intermediate_size");

        var routerLogits = LinearByName(
            hiddenStates,
            $"model.layers.{count}.mlp.gate.weight",
            layerName: $"model.layers.{count}.mlp.router_logits");
        var routerProbabilities = NN.Softmax(Tensors.Cast(routerLogits, DataTypes.Float32), -1);
        var topK = Tensors.TopK(
            routerProbabilities,
            Tensor.FromScalar(DataTypes.Int64, numTopK, [1]),
            -1,
            true,
            true);
        var routerWeights = topK[0];
        var selectedExperts = topK[1];
        routerWeights /= Tensors.Reduce(
            ReduceOp.Sum,
            routerWeights,
            new long[] { -1 },
            0.0f,
            true);

        Expr gateWeight;
        Expr upWeight;
        Expr downWeight;
        var stackedGateUp = GetWeight($"model.layers.{count}.mlp.experts.gate_up_proj");
        var stackedDown = GetWeight($"model.layers.{count}.mlp.experts.down_proj");
        if ((stackedGateUp is null) != (stackedDown is null))
        {
            throw new InvalidOperationException(
                "Qwen3.5 expert weights must consistently use either stacked or per-expert serialization.");
        }

        if (stackedGateUp is not null)
        {
            gateWeight = Tensors.Slice(
                stackedGateUp,
                new[] { 0L },
                new RankedShape(intermediateSize),
                new[] { 1L },
                new[] { 1L });
            upWeight = Tensors.Slice(
                stackedGateUp,
                new[] { intermediateSize },
                new RankedShape(intermediateSize * 2),
                new[] { 1L },
                new[] { 1L });
            downWeight = stackedDown!;
        }
        else
        {
            gateWeight = StackExpertWeights(count, numExperts, "gate_proj.weight");
            upWeight = StackExpertWeights(count, numExperts, "up_proj.weight");
            downWeight = StackExpertWeights(count, numExperts, "down_proj.weight");
        }

        var unitScales = Tensor.Ones(DataTypes.Float32, new long[] { numExperts, 1 });
        var expertOutput = NN.SparseExperts(
            hiddenStates,
            selectedExperts,
            routerWeights,
            unitScales,
            gateWeight,
            unitScales,
            unitScales,
            downWeight,
            unitScales,
            unitScales,
            upWeight,
            unitScales,
            hiddenSize,
            intermediateSize,
            numExperts,
            numTopK,
            1);

        var sharedGate = LinearByName(
            hiddenStates,
            $"model.layers.{count}.mlp.shared_expert.gate_proj.weight",
            layerName: $"model.layers.{count}.mlp.shared_expert.gate_proj");
        var sharedUp = LinearByName(
            hiddenStates,
            $"model.layers.{count}.mlp.shared_expert.up_proj.weight",
            layerName: $"model.layers.{count}.mlp.shared_expert.up_proj");
        var sharedHidden = Silu(sharedGate) * sharedUp;
        var sharedOutput = LinearByName(
            sharedHidden,
            $"model.layers.{count}.mlp.shared_expert.down_proj.weight",
            layerName: $"model.layers.{count}.mlp.shared_expert.down_proj");
        var sharedExpertScale = NN.Sigmoid(LinearByName(
            hiddenStates,
            $"model.layers.{count}.mlp.shared_expert_gate.weight",
            layerName: $"model.layers.{count}.mlp.shared_expert_gate"));
        return (expertOutput + (sharedOutput * sharedExpertScale))
            .With(metadata: new IRMetadata { OutputNames = new[] { $"model.layers.{count}.mlp" } });
    }

    protected override string ResolveWeightName(string canonicalWeightName)
    {
        const string canonicalModelPrefix = "model.";
        return canonicalWeightName.StartsWith(canonicalModelPrefix, StringComparison.Ordinal)
            ? $"{_textModelPrefix}.{canonicalWeightName[canonicalModelPrefix.Length..]}"
            : canonicalWeightName;
    }

    private static Call Silu(Expr value) => value * NN.Sigmoid(value);

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
        Tensor RequireWeight(string name) => GetWeight(name)
            ?? throw new InvalidOperationException($"Required weight {name} is missing.");
        Tensor PrepareLinearWeight(string name) => PrepareLinearWeightTensor(RequireWeight(name), inputType);

        var convWeightTensor = GetWeight($"{prefix}.conv1d.weight")
            ?? throw new InvalidOperationException($"Required weight {prefix}.conv1d.weight is missing.");
        var convWeight = Tensors.Reshape(
            convWeightTensor.CastTo(inputType),
            new RankedShape(convDim, convKernelSize));
        var result = NN.GatedDeltaNet(
            hiddenStates,
            state,
            PrepareLinearWeight($"{prefix}.in_proj_qkv.weight"),
            PrepareLinearWeight($"{prefix}.in_proj_z.weight"),
            PrepareLinearWeight($"{prefix}.in_proj_b.weight"),
            PrepareLinearWeight($"{prefix}.in_proj_a.weight"),
            convWeight,
            RequireWeight($"{prefix}.A_log"),
            RequireWeight($"{prefix}.dt_bias"),
            RequireWeight($"{prefix}.norm.weight"),
            PrepareLinearWeight($"{prefix}.out_proj.weight"),
            new DimConst(count),
            numKeyHeads,
            numValueHeads,
            keyHeadDim,
            valueHeadDim,
            convKernelSize,
            (float)Config.GetNestedValue<double>("rms_norm_eps"))
            .With(metadata: new IRMetadata { OutputNames = new[] { prefix } });
        _state = result[1];
        return result[0];
    }

    private Call StackExpertWeights(int layerIndex, long numExperts, string suffix)
    {
        var weights = new Expr[checked((int)numExperts)];
        for (var expertIndex = 0; expertIndex < weights.Length; expertIndex++)
        {
            var name = $"model.layers.{layerIndex}.mlp.experts.{expertIndex}.{suffix}";
            var weight = GetWeight(name)
                ?? throw new InvalidOperationException($"Required Qwen3.5 expert weight {name} is missing.");
            weights[expertIndex] = Tensors.Unsqueeze(weight, new long[] { 0 });
        }

        return Tensors.Concat(new Nncase.IR.Tuple(weights), 0);
    }

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
