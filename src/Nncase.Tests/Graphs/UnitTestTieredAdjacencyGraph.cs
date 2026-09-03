// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.Graphs;
using QuikGraph;
using Xunit;

namespace Nncase.Tests.Graphs;

public sealed class UnitTestTieredAdjacencyGraph
{
    [Fact]
    public void TestRemoveVertexRangeRebuildsCompleteHierarchy()
    {
        var root = new TieredAdjacencyGraph<string, Edge<string>>(
            new AdjacencyGraph<string, Edge<string>>(true));
        var first = root.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        var nested = first.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        var second = root.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();

        root.AddVerticesAndEdge(new Edge<string>("source", "removed_first"));
        root.AddVerticesAndEdge(new Edge<string>("removed_second", "removed_first"));
        root.AddVerticesAndEdge(new Edge<string>("removed_first", "sink"));
        root.AddVerticesAndEdge(new Edge<string>("source", "sink"));
        first.AddVertexRange(new[] { "source", "removed_first", "sink" });
        first.AddEdge(new Edge<string>("source", "removed_first"));
        first.AddEdge(new Edge<string>("removed_first", "sink"));
        nested.AddVertexRange(new[] { "removed_first", "sink" });
        nested.AddEdge(new Edge<string>("removed_first", "sink"));
        second.AddVertexRange(new[] { "removed_second", "sink" });
        second.AddEdge(new Edge<string>("removed_second", "sink"));

        var removed = root.RemoveVertexRange(new[] { "removed_first", "removed_second" });

        Assert.Equal(2, removed);
        Assert.Equal(new[] { "sink", "source" }, root.Vertices.OrderBy(vertex => vertex));
        Assert.Single(root.Edges);
        Assert.True(root.ContainsEdge("source", "sink"));
        Assert.Equal(new[] { "sink", "source" }, first.Vertices.OrderBy(vertex => vertex));
        Assert.Empty(first.Edges);
        Assert.Equal(new[] { "sink" }, nested.Vertices);
        Assert.Empty(nested.Edges);
        Assert.Equal(new[] { "sink" }, second.Vertices);
        Assert.Empty(second.Edges);
    }
}
