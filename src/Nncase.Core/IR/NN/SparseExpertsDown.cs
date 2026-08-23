// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// Projects selected expert activations back to hidden size and applies router weights.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class SparseExpertsDown : Op
{
    public static readonly ParameterInfo Activations = new(typeof(SparseExpertsDown), 0, "activations", ParameterKind.Input);

    public static readonly ParameterInfo RouterExpertIds = new(typeof(SparseExpertsDown), 1, "router_expert_ids", ParameterKind.Input);

    public static readonly ParameterInfo RouterExpertWeights = new(typeof(SparseExpertsDown), 2, "router_expert_weights", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertDownInputScale = new(typeof(SparseExpertsDown), 3, "moe_expert_down_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertDownProjW = new(typeof(SparseExpertsDown), 4, "moe_expert_down_proj_w", ParameterKind.Input);

    public static readonly ParameterInfo MoeExpertDownProjScale = new(typeof(SparseExpertsDown), 5, "moe_expert_down_proj_scale", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public long HiddenSize { get; }

    public long MoEIntermediateSize { get; }

    public long NumExpert { get; }

    public long NumTopK { get; }

    public long ChunkSize { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, HiddenSize: {HiddenSize}, MoEIntermediateSize: {MoEIntermediateSize}, NumExpert: {NumExpert}, NumTopK: {NumTopK}, ChunkSize: {ChunkSize}";
}
