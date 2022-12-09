// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using GiGraph.Dot.Entities.Clusters;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Colors;
using GiGraph.Dot.Types.Edges;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Styling;
using Nncase.PatternMatch;
using Nncase.Transform;

namespace Nncase.Transform;

public partial class EGraphPrinter
{
    private DotGraph AttachEGraphCost(CostModel.EGraphCostModel costModel, EClass entry)
    {
        // 1. display each enode costs.
        foreach (var (enode, (dotnode, table)) in NodesMap)
        {
            if (enode.Expr is (IR.Var or IR.Op or IR.Marker or IR.None))
                continue;
            table.AddRow(row =>
            {
                var cost = costModel[enode];
                foreach (var (k, v) in cost.Factors)
                {
                    row.AddCell($"{k}: {v:F2}");
                };
                row.AddCell($"Score: {cost.Score:F2}");
            });
            dotnode.ToPlainHtmlNode(table);
        }

        dotGraph.Edges.Clear();

        void dfs(EClass curclass, ENode? minCostEnode)
        {
            if (minCostEnode is null)
                return;
            var (minCostDotnode, table) = NodesMap[minCostEnode];
            minCostDotnode.Color = Color.DeepSkyBlue;
            foreach (var (child, i) in minCostEnode.Children.Select((c, i) => (c, i)))
            {
                var childEnode = child.Find().Nodes.MinBy(x => costModel[x]);
                dfs(child.Find(), childEnode);
                if (childEnode is null)
                    continue;
                var (childDotNode, _) = NodesMap[childEnode];
                dotGraph.Edges.Add(childDotNode, minCostDotnode, edge =>
                {
                    edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                    edge.Color = Color.DeepSkyBlue;
                });
            }
        }
        dfs(entry.Find(), entry.Find().Nodes.MinBy(x => costModel[x]));
        return dotGraph;
    }

    internal static DotGraph DumpEgraphAsDot(EGraph eGraph, CostModel.EGraphCostModel costModel, EClass entry, string file)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        printer.AttachEGraphCost(costModel, entry);
        return printer.SaveToFile(file);
    }
}