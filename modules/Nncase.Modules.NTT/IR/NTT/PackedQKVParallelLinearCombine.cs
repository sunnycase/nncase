// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Materializes optional split-K Q/K/V partial sums into the selected output layout.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedQKVParallelLinearCombine : Op
{
    public static readonly ParameterInfo QKV = new(typeof(PackedQKVParallelLinearCombine), 0, "qkv", ParameterKind.Input);

    public IRType OutputType { get; }

    public override string DisplayProperty() => $"OutputType: {OutputType}";
}
