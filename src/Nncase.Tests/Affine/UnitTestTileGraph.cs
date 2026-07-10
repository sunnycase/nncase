// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Schedule.TileGraph;
using QuikGraph;
using QuikGraph.Graphviz;
using Xunit;

namespace Nncase.Tests.AffineTest;

[Collection(nameof(NotThreadSafeResourceCollection))]
[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTileGraph : TestClassBase
{
    public static readonly TheoryData<Func<Function>, int> BuildTileGraphDatas = new()
    {
        { FunctionSamples.GetMatmulExpMatmul, 0 },
        { FunctionSamples.GetMatmulBinaryBinary, 1 },
        { FunctionSamples.GetMulDivMulSub, 2 },
        { FunctionSamples.GetBinaryNeg, 3 },
    };

    public static readonly TheoryData<Func<Function>, (IntMergePoint, bool)[], Action<TieredTileGraph>, int, int> MergeTileGraphDatas = new()
    {
        { FunctionSamples.GetMatmulExpMatmul, new (IntMergePoint, bool)[] { (new(2, 1, 0), true), }, MergeTileGraphCheckerDefault, 1, 1 },
        { FunctionSamples.GetMatmulExpMatmul, new (IntMergePoint, bool)[] { (new(2, 1, 1), true), (new(2, 0, 1), true), (new(2, 0, 0), false), (new(1, 0, 0), true) }, MergeTileGraphChecker0, 2, 2 },
        { FunctionSamples.GetMatmulExpMatmul, new (IntMergePoint, bool)[] { (new(1, 0, 1), true), (new(2, 0, 1), false), (new(2, 1, 1), true), }, MergeTileGraphCheckerDefault, 2, 3 },
        { FunctionSamples.GetVectorizeMatmulExpMatmul, new (IntMergePoint, bool)[] { (new(2, 0, 1), true), (new(2, 1, 1), true), (new(2, 0, 0), true), (new(2, 1, 0), true), (new(3, 2, 1), true), (new(5, 4, 1), true) }, MergeTileGraphChecker2, 2, 4 },
        { FunctionSamples.GetDynamicVectorizedSwish, new (IntMergePoint, bool)[] { (new(2, 1, 0), true), (new(2, 0, 0), true) }, MergeTileGraphCheckerDefault, 1, 5 },
    };

    public static readonly TheoryData<Func<Function>, IntMergePoint[], Action<BaseExpr>, int, int> SolveTileGraphDatas = new()
    {
        { FunctionSamples.GetBinaryNeg, [], SolveTileGraphChecker0, 2, 0 },
        { FunctionSamples.GetMatmulExpMatmul, [new(1, 0, 1), new(1, 0, 0)], (_) => { }, 2, 1 },
        { FunctionSamples.GetMatmulExpMatmul, [new(2, 1, 1)], (_) => { }, 2, 2 },
        { FunctionSamples.GetMatmulExpMatmul, [new(1, 0, 1), new(2, 1, 1), new(1, 0, 0)], (_) => { }, 2, 3 },
        { FunctionSamples.GetAddBranchMerge, [new(1, 0, 1)], (_) => { }, 1, 4 },
        { FunctionSamples.GetUnaryCastTrans, [new(2, 1, 0), new(2, 0, 0)], (_) => { }, 1, 5 },
        { FunctionSamples.GetBinaryUnary, [new(1, 0, 0)], SolveBinaryUnaryChecker, 1, 6 },
        { FunctionSamples.GetAddBranchMerge, [new(3, 1, 0), new(3, 2, 0), new(3, 0, 0)], (_) => { }, 1, 7 },
        { FunctionSamples.GetDynamicVectorizedCastTranspose, [new(1, 0, 0)], (_) => { }, 1, 8 },

        // just for check single op tiling results
        // { FunctionSamples.Get1Matmul, [], (_) => { }, 5 },
        // { FunctionSamples.Get1Exp, [], (_) => { }, 6 },
    };

    public static readonly TheoryData<Func<Function>, int, int> MCTSDatas = new()
    {
        { FunctionSamples.GetMatmulExpMatmul, 1, 0 },
        { FunctionSamples.GetAddBranchMerge, 1, 1 },
        { FunctionSamples.GetQwen3Rope, 1, 2 },
    };

    public static readonly TheoryData<Func<Function>, IntMergePoint[], Action<BufferGraph>, int> BufferizeTileGraphDatas = new()
    {
        { FunctionSamples.GetMatmulExpMatmul, [new(1, 0, 1)], (bufGraph) => { Assert.Equal(4, bufGraph.Clusters.OfType<BufferGraph>().First().Edges.Count()); }, 0 },
    };

    public UnitTestTileGraph()
    {
        CompileOptions.TargetOptions = new Targets.NTTTargetOptions()
        {
            MemoryCapacities = new[] { 256 * 1024, 512 * 1024, int.MaxValue },
            MemoryBandWidths = new[] { 128, 64, 8 },
        };
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Tiling;
#endif
    }

    [Fact]
    public void TestReshapeAffineSelectionPreservesVectorLanes()
    {
        var vectorType = new VectorType(DataTypes.BFloat16, [8]);
        var input = new Var("input", new TensorType(vectorType, new[] { 20, 128 }));
        var reshape = IR.F.Tensors.Reshape(input, new Dimension[] { 20, 16, 8 });
        var function = new Function("main", Targets.CPUTarget.Kind, reshape, [input]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        var reshapeKernel = Assert.IsType<TIR.NTT.Reshape>(bodyCall.Target);
        Assert.Equal(vectorType, bodyCall.Arguments[0].CheckedDataType);
        Assert.Equal(vectorType, bodyCall.Arguments[1].CheckedDataType);
    }

    [Fact]
    public void TestBitcastAffineSelectionUsesDedicatedLaneReinterpretationKernel()
    {
        var input = new Var("input", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 20, 128 }));
        var bitcast = IR.F.Tensors.Bitcast(input, DataTypes.BFloat16);
        var function = new Function("main", Targets.CPUTarget.Kind, bitcast, [input]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        var bitcastKernel = Assert.IsType<TIR.NTT.Bitcast>(bodyCall.Target);
        Assert.Equal(new VectorType(DataTypes.BFloat16, [8]), bodyCall.Arguments[0].CheckedDataType);
        Assert.Equal(DataTypes.BFloat16, bodyCall.Arguments[1].CheckedDataType);
    }

    [Fact]
    public void TestPackedMatMulGluAffineSelection()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { 20, 512 }));
        var packedWeightType = new TensorType(new VectorType(DataTypes.BFloat16, [4, 8]), new[] { 24, 512 });
        var gateWeight = new Var("gate_weight", packedWeightType);
        var upWeight = new Var("up_weight", packedWeightType);
        var glu = IR.F.NTT.PackedMatMulGlu(
            input,
            gateWeight,
            upWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            IR.NN.GluType.SwiGLU,
            DataTypes.BFloat16);
        var function = new Function("main", Targets.CPUTarget.Kind, glu, [input, gateWeight, upWeight]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Equal(3, grid.Reads.Length);
        Assert.Equal(1, grid.Writes.Length);
        Assert.Equal(2, grid.AccessMaps[0].Domains.Length);
        var kRange = grid.AccessMaps[0].Results[^1];
        Assert.Equal(0, Assert.IsType<IR.Affine.AffineConstant>(kRange.Offset).Value);
        Assert.Equal(512, Assert.IsType<IR.Affine.AffineConstant>(kRange.Extent).Value);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        Assert.IsType<TIR.NTT.PackedMatMulGlu>(bodyCall.Target);
    }

    [Fact]
    public void TestPackedQKVParallelLinearAffineSelectionUsesSharedProjectionDomain()
    {
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { 20, 512 }));
        var packedDataType = new VectorType(DataTypes.BFloat16, [4, 8]);
        var qWeight = new Var("q_weight", new TensorType(packedDataType, new[] { 16, 512 }));
        var kWeight = new Var("k_weight", new TensorType(packedDataType, new[] { 8, 512 }));
        var vWeight = new Var("v_weight", new TensorType(packedDataType, new[] { 8, 512 }));
        var qkv = IR.F.NTT.PackedQKVParallelLinear(
            input,
            qWeight,
            kWeight,
            vWeight,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            None.Default,
            numHeads: 16,
            numKvHeads: 8,
            outDataType: DataTypes.BFloat16);
        var function = new Function("main", Targets.CPUTarget.Kind, qkv, [input, qWeight, kWeight, vWeight]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Equal(4, grid.Reads.Length);
        Assert.Equal(3, grid.Writes.Length);
        Assert.All(grid.AccessMaps.ToArray(), map => Assert.Equal(2, map.Domains.Length));

        var domainOffsets = new long[] { 3, 5 };
        var domainExtents = new long[] { 7, 11 };
        var qWeightRange = grid.AccessMaps[1].Results[0].Apply(domainOffsets, domainExtents);
        var kWeightRange = grid.AccessMaps[2].Results[0].Apply(domainOffsets, domainExtents);
        var vWeightRange = grid.AccessMaps[3].Results[0].Apply(domainOffsets, domainExtents);
        Assert.Equal(new ValueRange<long>(10, 22), qWeightRange);
        Assert.Equal(new ValueRange<long>(5, 11), kWeightRange);
        Assert.Equal(new ValueRange<long>(5, 11), vWeightRange);

        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        Assert.IsType<TIR.NTT.PackedQKVParallelLinear>(bodyCall.Target);
    }

    [Fact]
    public void TestUpdatePagedAttentionKVCacheAffineSelectionUsesObjectAsSsaOutput()
    {
        var slots = new Var("slots", new TensorType(DataTypes.BFloat16, new[] { 1, 1, 8 }));
        var objectType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var kvCache = new Var("kv_cache", objectType);
        var update = IR.F.NN.UpdatePagedAttentionKVCache(
            slots,
            kvCache,
            IR.NN.AttentionCacheKind.Key,
            0,
            [IR.NN.AttentionDimKind.Seq, IR.NN.AttentionDimKind.Head, IR.NN.AttentionDimKind.Dim]);
        var function = new Function("main", Targets.CPUTarget.Kind, update, [slots, kvCache]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Equal(2, grid.Reads.Length);
        Assert.Equal(1, grid.Writes.Length);
        Assert.Same(kvCache, grid.Reads[1]);
        Assert.Same(kvCache, grid.Writes[0]);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        Assert.IsType<TIR.NTT.UpdatePagedAttentionKVCache>(bodyCall.Target);
    }

    [Fact]
    public async Task TestAutoTilePreservesPassthroughObjectAtClusterBoundary()
    {
        var slots = new Var("slots", new TensorType(DataTypes.BFloat16, new[] { 1, 1, 8 }));
        var objectType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var kvCache = new Var("kv_cache", objectType);
        var passthrough = new Var("passthrough", objectType);
        var update = IR.F.NN.UpdatePagedAttentionKVCache(
            slots,
            kvCache,
            IR.NN.AttentionCacheKind.Key,
            0,
            [IR.NN.AttentionDimKind.Seq, IR.NN.AttentionDimKind.Head, IR.NN.AttentionDimKind.Dim]);
        var selectionInput = new Function("selection_input", Targets.CPUTarget.Kind, update, [slots, kvCache]);
        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(selectionInput, new()));
        var grid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var function = new Function("main", Targets.CPUTarget.Kind, new IR.Tuple(grid, passthrough), [slots, kvCache, passthrough]);

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(function, new()));

        var outputs = Assert.IsType<IR.Tuple>(tiled.Body);
        Assert.Same(passthrough, outputs.Fields[1]);
    }

    [Fact]
    public async Task TestAutoTilePreservesPartialResultBeforeBoxing()
    {
        var placement = new Placement(new[] { 4, 8 }, "yx", "bb");
        var inputType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new long[] { 20, 128 }),
            new SBP[] { SBP.S([0], 5), SBP.S([1], 16) },
            placement);
        var input = new Var("input", inputType);
        var stats = IR.F.NN.NormStats(1, input, useMean: false);
        var partialStatsType = Assert.IsType<DistributedType>(stats.CheckedType);
        var broadcastStatsType = partialStatsType with { Partial = null };
        var boxed = IR.F.Distributed.Boxing(stats, broadcastStatsType);
        var selectionInput = new Function("selection_input", Targets.CPUTarget.Kind, boxed, [input]);
        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(selectionInput, new()));
        var selectedBoxing = Assert.IsType<Call>(selected.Body);
        Assert.IsType<IR.Distributed.Boxing>(selectedBoxing.Target);
        var selectedGrid = Assert.IsType<IR.Affine.Grid>(selectedBoxing.Arguments[0]);
        Assert.Equal(partialStatsType.Partial, Assert.IsType<DistributedType>(selectedGrid.CheckedType).Partial);

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));

        var boxingCall = Assert.IsType<Call>(tiled.Body);
        Assert.True(
            boxingCall.Target is IR.Distributed.Boxing,
            $"AutoTile changed partial stats into {boxingCall.CheckedType} through {boxingCall.Target.GetType().Name}.\n{CompilerServices.Print(tiled)}");
        var tiledStatsType = Assert.IsType<DistributedType>(boxingCall.Arguments[0].CheckedType);
        Assert.Equal(partialStatsType.Partial, tiledStatsType.Partial);
    }

    [Fact]
    public void TestClusteredGraph()
    {
        string a = "a", b = "b", c = "c", d = "d", e = "e", f = "f";

        var g0 = new AdjacencyGraph<string, Edge<string>>();
        var g = new ClusteredAdjacencyGraph<string, Edge<string>>(g0);
        var g1 = g.AddCluster();
        var g2 = g.AddCluster();

        g1.AddVerticesAndEdge(new(e, f));
        g1.AddVerticesAndEdge(new(c, f));

        Assert.Equal(3, g1.VertexCount);

        g2.AddVerticesAndEdge(new(a, b));

        Assert.Equal(2, g2.VertexCount);

        Assert.Equal(5, g0.VertexCount);

        g.AddEdge(new(e, b));
        g.AddEdge(new(b, c));

        g.AddVertex(d);
        g.AddVerticesAndEdge(new(b, d));
        g.AddVerticesAndEdge(new(f, d));

#if DEBUG
        using (var file = Dumpper.OpenFile("g.dot"))
        {
            using var writer = new StreamWriter(file);
            writer.Write(g.ToGraphviz(algorithm =>
            {
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex;
                algorithm.FormatCluster += (_, args) => args.GraphFormat.Label = "f";
            }));
        }
#endif

        // build a graph for subgraphs.
        var cg = new AdjacencyGraph<ClusteredAdjacencyGraph<string, Edge<string>>, Edge<ClusteredAdjacencyGraph<string, Edge<string>>>>();
        foreach (var subGraph in g.Clusters.OfType<ClusteredAdjacencyGraph<string, Edge<string>>>())
        {
            cg.AddVertex(subGraph);
        }

        foreach (var edge in g.Edges)
        {
            foreach (var source in cg.Vertices)
            {
                foreach (var target in cg.Vertices.Where(v => v != source))
                {
                    if (source.ContainsVertex(edge.Source) && target.ContainsVertex(edge.Target))
                    {
                        cg.AddEdge(new(source, target));
                    }
                }
            }
        }

#if DEBUG
        using (var file = Dumpper.OpenFile("cg.dot"))
        {
            using var writer = new StreamWriter(file);
            writer.Write(cg.ToGraphviz(algorithm =>
            {
            }));
        }
#endif
    }

    [Fact]
    public void TestClusteredGraphAsBufferGraph()
    {
        var g = new AdjacencyGraph<string, Edge<string>>();
        var cg = new ClusteredAdjacencyGraph<string, Edge<string>>(g);
        var cg0 = cg.AddCluster();
        var cg0_0 = cg0.AddCluster();
        var cg0_1 = cg0.AddCluster();

        cg0_0.AddVerticesAndEdge(new("op0_in0", "op0_out"));
        cg0_1.AddVerticesAndEdge(new("op1_in0", "op1_out"));
        cg0_1.AddVerticesAndEdge(new("op1_in1", "op1_out"));
        cg0.AddEdge(new("op0_out", "op1_in0"));

        var cg1 = cg.AddCluster();
        var cg1_0 = cg1.AddCluster();
        cg1_0.AddVerticesAndEdge(new("op2_in0", "op2_out"));

        cg.AddEdge(new("op1_out", "op2_in0"));

        var nameMap = new Dictionary<IVertexAndEdgeListGraph<string, Edge<string>>, string>() {
            { cg0, "cg0" },
            { cg0_0, "cg0_0" },
            { cg0_1, "cg0_1" },
            { cg1, "cg1" },
            { cg1_0, "cg1_0" },
        };

#if DEBUG
        using (var file = Dumpper.OpenFile("g.dot"))
        {
            using var writer = new StreamWriter(file);
            writer.Write(cg.ToGraphviz(algorithm =>
            {
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex;
                algorithm.FormatCluster += (_, args) => args.GraphFormat.Label = nameMap[args.Cluster];
            }));
        }
#endif
    }

    [Fact]
    public void TestRemoveAndAddSubGraph()
    {
        AdjacencyGraph<string, Edge<string>> root = new();
        TieredAdjacencyGraph<string, Edge<string>> cg = new(root);
        var cg0 = cg.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        var cg00 = cg0.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        cg00.AddVertex("a");
        var cg1 = cg.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        var cg11 = cg1.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        cg11.AddVertex("b");
        root.AddEdge(new("a", "b"));

        var cnames = new Dictionary<IVertexAndEdgeListGraph<string, Edge<string>>, string>()
        {
            { cg, "cg" },
            { cg0, "cg0" },
            { cg00, "cg00" },
            { cg1, "cg1" },
            { cg11, "cg11" },
        };

#if DEBUG
        void Dump(TieredAdjacencyGraph<string, Edge<string>> graph, string name)
        {
            using (var file = Dumpper.OpenFile($"{name}.dot"))
            {
                using var writer = new StreamWriter(file);
                writer.Write(graph.ToGraphviz(algorithm =>
                {
                    algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex;
                    algorithm.FormatCluster += (_, args) => args.GraphFormat.Label = cnames[args.Cluster];
                }));
            }
        }

        Dump(cg, "cg");
#endif

        Assert.Equal(2, root.VertexCount);
        Assert.Equal(2, cg.VertexCount);
        Assert.Equal(1, cg0.VertexCount);
        Assert.Equal(1, cg1.VertexCount);

        void MergeSubGraph(TieredAdjacencyGraph<string, Edge<string>> source, TieredAdjacencyGraph<string, Edge<string>> target)
        {
            var parent = target.Parent!;

            // try move cg00 into cg1
            parent.RemoveCluster(source);
            foreach (var sourceChild in source.Clusters.OfType<TieredAdjacencyGraph<string, Edge<string>>>())
            {
                target.AddCluster(sourceChild);
            }

            target.AddVertexRange(source.Vertices);
        }

        MergeSubGraph(cg0, cg1);

#if DEBUG
        Dump(cg, "cg_merge_cg0_to_cg1");
#endif

        Assert.Equal(2, root.VertexCount); // keep the root vertex not change.
        Assert.Equal(2, cg.VertexCount);
        Assert.Equal(1, cg.ClustersCount);
        Assert.Equal(2, cg1.ClustersCount);
        Assert.Equal(2, cg1.VertexCount);
        Assert.Equal(1, root.EdgeCount);
        Assert.Equal(1, cg.EdgeCount);

        MergeSubGraph(cg11, cg00);

#if DEBUG
        Dump(cg, "cg_merge_cg11_to_cg00");
#endif

        Assert.Equal(2, root.VertexCount);
        Assert.Equal(2, cg.VertexCount);
        Assert.Equal(1, cg.ClustersCount);
        Assert.Equal(1, cg1.ClustersCount);
        Assert.Equal(0, cg00.ClustersCount);
        Assert.Equal(2, cg00.VertexCount);
        Assert.Equal(1, root.EdgeCount);
        Assert.Equal(1, cg.EdgeCount);
    }

    [Theory]
    [MemberData(nameof(BuildTileGraphDatas))]
    public void TestBuildTileGraph(Func<Function> functor, int count)
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var func = functor();
        var post = new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;
#if DEBUG
        Dumpper.DumpIR(post, $"post{count}");
#endif

        var graph = TieredTileGraphBuilder.Build(post, 2, out var exprMemo);

#if DEBUG
        graph.Dump($"g{count}");
#endif

        Assert.Equal(-1, graph.Level);
    }

    [Theory]
    [MemberData(nameof(MergeTileGraphDatas))]
    public void TestMergeTileGraph(Func<Function> functor, (IntMergePoint, bool)[] mergePoints, Action<TieredTileGraph> checker, int levelCount, int count)
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = (INTTTargetOptions)CompileOptions.TargetOptions;
        if (levelCount == 1)
        {
            targetOptions.MemoryCapacities = new[] { 256 * 1024, int.MaxValue };
            targetOptions.MemoryBandWidths = new[] { 128, 16 };
        }

        var func = functor();
        var post = new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;
#if DEBUG
        Dumpper.DumpIR(post, $"post{count}");
#endif

        var tileGraph = TieredTileGraphBuilder.Build(post, levelCount, out var exprMemo);
#if DEBUG
        tileGraph.Dump($"g{count}");
#endif

        for (int i = 0; i < mergePoints.Length; i++)
        {
            var (point, excepted) = mergePoints[i];
            Assert.Equal(excepted, tileGraph.Merge(new(tileGraph.Vertices.Skip(point.Consumer).First(), tileGraph.Vertices.Skip(point.Producer).First(), point.Level)));
            if (excepted)
            {
#if DEBUG
                tileGraph.Dump($"g{count}_m{i}");
#endif
            }
        }

        checker(tileGraph);
    }

    [Theory]
    [MemberData(nameof(SolveTileGraphDatas))]
    public void TestSolveTileGraph(Func<Function> functor, IntMergePoint[] mergePoints, Action<BaseExpr> action, int levelCount, int count)
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = (INTTTargetOptions)CompileOptions.TargetOptions;
        if (levelCount == 1)
        {
            targetOptions.MemoryCapacities = new[] { 256 * 1024, int.MaxValue };
            targetOptions.MemoryBandWidths = new[] { 128, 16 };
        }

        var func = functor();
        var pre = (Function)new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;
        using var dumpScope = new Diagnostics.DumpScope(count.ToString());
#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(pre, $"post{count}");
#endif
        var tileGraph = TieredTileGraphBuilder.Build(pre.Body, targetOptions.MemoryBandWidths.Length - 1, out var exprMemo);

        for (int i = 0; i < mergePoints.Length; i++)
        {
            var point = mergePoints[i];
            tileGraph.Merge(new(tileGraph.Vertices.Skip(point.Consumer).First(), tileGraph.Vertices.Skip(point.Producer).First(), point.Level));
        }
#if DEBUG
        tileGraph.Dump($"g{count}_m");
#endif

        var tiler = new Schedule.GraphTiler();

        var (argumentMemo, _) = tiler.SolveRootGraph(tileGraph, Targets.CPUTarget.Kind, targetOptions, Array.Empty<DimVar>());
        var replaces = new Dictionary<BaseExpr, BaseExpr>();
        foreach (var (bid, value) in argumentMemo)
        {
            // use bid to find the old expr.
            var oldExpr = bid.IsOutput ? bid.Node.Grid : bid.Node.Grid.GetArgument(bid.Index);
            if (!replaces.ContainsKey(oldExpr))
            {
                replaces.Add(oldExpr, value);
            }
        }

        var cloner = new ReplacingExprCloner(replaces);
        var post = cloner.Clone(pre, default);
#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(post, $"result{count}", flags: Diagnostics.PrinterFlags.Normal);
#endif
        action(post);
    }

    [Theory]
    [MemberData(nameof(MCTSDatas))]
    public void TestMCTS(Func<Function> functor, int levelCount, int count)
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = (INTTTargetOptions)CompileOptions.TargetOptions;
        if (levelCount == 1)
        {
            targetOptions.MemoryCapacities = new[] { 256 * 1024, int.MaxValue };
            targetOptions.MemoryBandWidths = new[] { 128, 16 };
        }

        var func = functor();
        var post = new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;

        using var dumpScope = new Diagnostics.DumpScope(count.ToString());
        var tileGraph = TieredTileGraphBuilder.Build(post, targetOptions.MemoryBandWidths.Length - 1, out var exprMemo);

        var tiler = new Schedule.GraphTiler();
        var state = new MCTState(tileGraph, "cpu", count.ToString(), tiler, targetOptions, Array.Empty<DimVar>());
        var rootNode = new MCTNode(state);
        var searcher = new MCTSearcher(60);
        searcher.Search(rootNode);
#if DEBUG
        rootNode.Dump("mct");
#endif
    }

    [Theory]
    [MemberData(nameof(BufferizeTileGraphDatas))]
    public void TestBufferizeTileGraph(Func<Function> functor, IntMergePoint[] mergePoints, Action<BufferGraph> action, int count)
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = (INTTTargetOptions)CompileOptions.TargetOptions;
        var func = functor();
        var post = new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;

        using var dumpScope = new Diagnostics.DumpScope(count.ToString());
        var tileGraph = TieredTileGraphBuilder.Build(post, targetOptions.MemoryBandWidths.Length - 1, out var exprMemo);

        for (int i = 0; i < mergePoints.Length; i++)
        {
            var point = mergePoints[i];
            tileGraph.Merge(new(tileGraph.Vertices.Skip(point.Consumer).First(), tileGraph.Vertices.Skip(point.Producer).First(), point.Level));
        }
#if DEBUG
        tileGraph.Dump($"g{count}_m");
#endif
        var bufferGraph = tileGraph.Bufferize();
        action(bufferGraph[tileGraph]);
    }

    [Fact]
    public void TestPrimTreeEqualityComparer()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var func = FunctionSamples.GetMulDivMulSub();
        var post = new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;
        var grid = (IR.Affine.Grid)((Function)post).Body;
        var rootGraph = TieredTileGraphBuilder.Build(grid, 2, out _);
#if DEBUG
        rootGraph.Dump($"g");
#endif
        var rootTree = TileNode.FromTileGraph(rootGraph, out var _);

        Assert.True(new ITreeNodeComparer().Equals(rootTree.Children[0], rootTree.Children[2]));

        var set = new HashSet<ITreeNode>(new ITreeNodeComparer());
        foreach (var item in rootTree.Children)
        {
            if (!set.Contains(item))
            {
                set.Add(item);
            }
        }

        Assert.Equal(4, rootTree.Children.Length);
        Assert.Equal(3, set.Count);
    }

    private static void SolveTileGraphCheckerDefault(Expr post)
    {
    }

    private static void SolveTileGraphChecker0(BaseExpr post)
    {
        Assert.IsType<IR.Function>(post);
        Assert.IsType<IR.Tuple>(((IR.Function)post).Body);
    }

    private static void MergeTileGraphCheckerDefault(TieredTileGraph tileGraph)
    {
    }

    private static void MergeTileGraphChecker0(TieredTileGraph tileGraph)
    {
        tileGraph.Walk(g =>
        {
            if (g is TieredTileGraph { Level: 0, OpId: 1 } g1)
            {
                Assert.Equal(2, g1.VertexCount);
                foreach (var op in g1.Vertices.Where(v => v.OpId == 0))
                {
                    Assert.Equal(1, op.DomainRelation.DomainOp);
                    Assert.Equal(0, op.DomainRelation.RangeOp);
                }
            }
        });
    }

    private static void MergeTileGraphChecker2(TieredTileGraph tileGraph)
    {
        // (new(2, 0, 2), true), (new(2, 1, 2), true), (new(2, 0, 1), true), (new(2, 1, 1), true), (new(3, 2, 2), true), (new(5, 4, 2), true)
        tileGraph.Walk(g =>
        {
            if (g is TieredTileGraph { Level: 1, OpId: 5 } g1)
            {
                Assert.Equal(2, g1.VertexCount);
                Assert.Equal(2, g1.ClustersCount);
                foreach (var item in g1.Clusters.OfType<TieredTileGraph>())
                {
                    Assert.Equal(5, item.DomainRelation.DomainOp);
                    Assert.Equal(item.OpId, item.DomainRelation.RangeOp);
                }
            }

            if (g is TieredTileGraph { Level: 1, OpId: 2 } g2)
            {
                Assert.Equal(3, g2.VertexCount);
                Assert.Equal(1, g2.ClustersCount);
            }

            if (g is TieredTileGraph { Level: 0, OpId: 2 } g3)
            {
                Assert.Equal(3, g3.VertexCount);
                Assert.Equal(0, g3.ClustersCount);
                foreach (var item in g3.Vertices)
                {
                    Assert.Equal(2, item.DomainRelation.DomainOp);
                    Assert.Equal(item.OpId, item.DomainRelation.RangeOp);
                }
            }
        });
    }

    private static void SolveBinaryUnaryChecker(BaseExpr post)
    {
        var exprs = ExprCollector.Collect(post);
        Assert.DoesNotContain(exprs, e => e is IR.Affine.Grid);
        var func = Assert.IsType<IR.Function>(post);
        Assert.IsType<IR.Tuple>(func.Body);
    }

    public sealed record IntMergePoint(int Consumer, int Producer, int Level)
    {
        public override string ToString() => $"merge({Consumer},{Producer},{Level})";
    }
}
