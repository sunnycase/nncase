// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// Computes the selected experts' gate/up projections and SwiGLU activations.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class SparseExpertsGateUp : Op
{
    public static readonly ParameterInfo Q = new(typeof(SparseExpertsGateUp), 0, "q", ParameterKind.Input);

    public static readonly ParameterInfo RouterExpertIds = new(typeof(SparseExpertsGateUp), 1, "router_expert_ids", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertGateInputScale = new(typeof(SparseExpertsGateUp), 2, "moe_expert_gate_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertGateProjW = new(typeof(SparseExpertsGateUp), 3, "moe_expert_gate_proj_w", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertGateProjScale = new(typeof(SparseExpertsGateUp), 4, "moe_expert_gate_proj_scale", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertUpInputScale = new(typeof(SparseExpertsGateUp), 5, "moe_expert_up_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertUpProjW = new(typeof(SparseExpertsGateUp), 6, "moe_expert_up_proj_w", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertUpProjScale = new(typeof(SparseExpertsGateUp), 7, "moe_expert_up_proj_scale", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public long HiddenSize { get; }

    public long MoEIntermediateSize { get; }

    public long NumExpert { get; }

    public long NumTopK { get; }

    public long ChunkSize { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, HiddenSize: {HiddenSize}, MoEIntermediateSize: {MoEIntermediateSize}, NumExpert: {NumExpert}, NumTopK: {NumTopK}, ChunkSize: {ChunkSize}";
}
