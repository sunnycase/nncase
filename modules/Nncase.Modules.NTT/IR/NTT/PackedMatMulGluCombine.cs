// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Materializes optional split-K gate/up partial sums and applies GLU.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedMatMulGluCombine : Op
{
    public static readonly ParameterInfo Projections = new(typeof(PackedMatMulGluCombine), 0, "projections", ParameterKind.Input);

    public IRType OutputType { get; }

    public GluType GluType { get; }

    public override string DisplayProperty() => $"OutputType: {OutputType}, GluType: {GluType}";
}
