// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Passes;

namespace Nncase.CostModel;

internal class EGraphCostModel
{
    private readonly IReadOnlyDictionary<ENode, Cost> _costs;
    private readonly ITargetOpCostModel _targetCostModel;

    public EGraphCostModel(IReadOnlyDictionary<ENode, Cost> costs, ITargetOpCostModel targetCostModel)
    {
        _costs = costs;
        _targetCostModel = targetCostModel;
    }

    public Cost this[ENode enode] => _costs[enode];

    public UInt128 GetLatency(ENode enode)
    {
        return GetLatency(_costs[enode]);
    }

    public UInt128 GetLatency(Cost cost)
    {
        return TargetOpCostModelUtility.GetCostLatency(_targetCostModel, cost);
    }

    public bool TryGet(ENode node, [MaybeNullWhen(false)] out Cost cost)
    {
        return _costs.TryGetValue(node, out cost);
    }
}
