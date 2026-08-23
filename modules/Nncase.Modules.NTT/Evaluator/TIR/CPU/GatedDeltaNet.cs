// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class GatedDeltaNetProjectionEvaluator : ITypeInferencer<GatedDeltaNetProjection>
{
    public IRType Visit(ITypeInferenceContext context, GatedDeltaNetProjection target) => TupleType.Void;
}

public sealed class GatedDeltaNetRecurrentCoreEvaluator : ITypeInferencer<GatedDeltaNetRecurrentCore>
{
    public IRType Visit(ITypeInferenceContext context, GatedDeltaNetRecurrentCore target) => TupleType.Void;
}
