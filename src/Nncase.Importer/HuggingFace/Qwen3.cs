// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;

namespace Nncase.Importer
{
    public class Qwen3 : HuggingFaceModel
    {
        public override Tuple<Call, Call, Call> QKVCompute(int count, Expr hiddenStates, Dimension seqLen, Dimension headDim)
        {
            var hidden_shape = new RankedShape(seqLen, -1L, headDim);
            var (queryStates, keyStates, valueStates) = BuildQKVParallelLinear(count, hiddenStates, hidden_shape);
            queryStates = LLMLayerNorm(queryStates, $"model.layers.{count}.self_attn.q_norm.weight");
            keyStates = LLMLayerNorm(keyStates, $"model.layers.{count}.self_attn.k_norm.weight");
            return System.Tuple.Create(queryStates, keyStates, valueStates);
        }
    }
}
