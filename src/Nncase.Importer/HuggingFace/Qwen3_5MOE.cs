// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;

namespace Nncase.Importer;

/// <summary>
/// Qwen3.5 MoE text decoder importer.
/// </summary>
public sealed class Qwen3_5Moe : Qwen3_5
{
    public Qwen3_5Moe(string textModelPrefix)
        : base(textModelPrefix)
    {
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

    private static Call Silu(Expr value) => value * NN.Sigmoid(value);

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
}
