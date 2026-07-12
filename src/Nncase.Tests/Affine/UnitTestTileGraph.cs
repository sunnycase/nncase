// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Schedule.TileGraph;
using Nncase.TIR;
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
        CompileOptions.TargetOptions = new Targets.NTTTargetOptions { TargetMachineModel = CreateTestMachine(2) };
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
        var view = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Same(input, view.Accesses[0].Value);
        Assert.Equal(vectorType, view.CheckedDataType);
        Assert.Equal(input.CheckedShape.Rank, view.Accesses[0].AffineMap.Results.Length);
        Assert.Equal(view.CheckedShape.Rank, view.Accesses[1].AffineMap.Results.Length);
        Assert.IsType<TIR.NTT.Reshape>(Assert.IsType<Call>(view.Body[0]).Target);
    }

    [Fact]
    public void TestBitcastAffineSelectionUsesZeroCopyBufferAliasGrid()
    {
        var input = new Var("input", new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 20, 128 }));
        var bitcast = IR.F.Tensors.Bitcast(input, DataTypes.BFloat16);
        var function = new Function("main", Targets.CPUTarget.Kind, bitcast, [input]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var view = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Same(input, view.Accesses[0].Value);
        Assert.Equal(new VectorType(DataTypes.BFloat16, [8]), input.CheckedDataType);
        Assert.Equal(DataTypes.BFloat16, view.CheckedDataType);
        Assert.Equal(input.CheckedShape.Rank, view.Accesses[0].AffineMap.Results.Length);
        Assert.Equal(view.CheckedShape.Rank, view.Accesses[1].AffineMap.Results.Length);
        Assert.IsType<TIR.NTT.Bitcast>(Assert.IsType<Call>(view.Body[0]).Target);
    }

    [Fact]
    public void TestNestedPrimFunctionPropagatesBufferAliasSummary()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 4 }));
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Tensors.Reshape(input, new Dimension[] { 2, 8 }),
            [input]);
        var selected = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var view = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var nestedInput = new Var("nested_input", view.Accesses[0].Parameter.CheckedType);
        var nestedOutput = new Var("nested_output", view.Accesses[1].Parameter.CheckedType);
        var nested = new TIR.PrimFunction(
            "nested_view",
            Targets.CPUTarget.Kind,
            new TIR.Sequential(TIR.F.NTT.Reshape(nestedInput, nestedOutput)),
            new IVar[] { nestedInput, nestedOutput });
        var nestedView = view.With(
            body: new TIR.Sequential(
                TIR.T.Nop(),
                new Call(nested, view.Accesses[0].Parameter, view.Accesses[1].Parameter)));

        var graph = TieredTileGraphBuilder.Build(nestedView, 1, out _);

        Assert.True(Assert.Single(graph.Vertices).IsPureBufferView);
    }

    [Fact]
    public void TestSharedBufferViewFusionIsPerUse()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 4 }));
        var view = IR.F.Tensors.Reshape(input, new Dimension[] { 2, 8 });
        var lhs = IR.F.Math.Unary(UnaryOp.Neg, view);
        var rhs = IR.F.Math.Unary(UnaryOp.Abs, view);
        var function = new Function("main", Targets.CPUTarget.Kind, new IR.Tuple(lhs, rhs), [input]);
        var selected = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var graph = TieredTileGraphBuilder.Build(selected.Body, 1, out _);
        var viewNode = Assert.Single(graph.Vertices.Where(node => node.IsPureBufferView));
        var viewUses = graph.GetMergePoints()
            .Where(point => ReferenceEquals(point.Producer, viewNode))
            .ToArray();
        Assert.Equal(2, viewUses.Length);

        Assert.True(graph.Merge(viewUses[0]));
        Assert.Contains(viewNode, graph.Vertices);
        Assert.Equal(1, graph.OutDegree(viewNode));
        Assert.Equal(2, graph.Vertices.Count(node => node.IsPureBufferView));

        var remainingUse = Assert.Single(graph.GetMergePoints().Where(point => ReferenceEquals(point.Producer, viewNode)));
        Assert.True(graph.Merge(remainingUse));
        Assert.Contains(viewNode, graph.Vertices);
        Assert.Contains(
            graph.Clusters.OfType<TieredTileGraph>(),
            cluster => cluster.ContainsVertex(viewNode) && cluster.ContainsVertex(remainingUse.Consumer));
        Assert.Equal(2, graph.Vertices.Count(node => node.IsPureBufferView));
    }

    [Fact]
    public void TestSingletonReshapePreservesDistributedShardRegions()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var dataType = new VectorType(DataTypes.Float32, [2, 8]);
        var source = new DistributedType(
            new TensorType(dataType, new[] { 20, 8 }),
            new SBP[] { SBP.B, SBP.S([1], 1) },
            placement);
        var result = new DistributedType(
            new TensorType(dataType, new[] { 20, 1, 8 }),
            new SBP[] { SBP.B, SBP.B, SBP.S([1], 1) },
            placement);

        Assert.True(IR.Affine.BufferViewUtility.TryCreate(source, result, out var transform));
        Assert.Equal(2, transform.SourceMap.Domains.Length);
        Assert.Equal(2, transform.SourceMap.Results.Length);
        Assert.Equal(3, transform.ResultMap.Results.Length);
        Assert.Equal(0, Assert.IsType<IR.Affine.AffineConstant>(transform.ResultMap.Results[1].Offset).Value);
        Assert.Equal(1, Assert.IsType<IR.Affine.AffineConstant>(transform.ResultMap.Results[1].Extent).Value);
    }

    [Fact]
    public void TestPackedReshapePreservesDistributedShardRegions()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var dataType = new VectorType(DataTypes.BFloat16, [4, 8]);
        var source = new DistributedType(
            new TensorType(dataType, new[] { 20, 64 }),
            new SBP[] { SBP.B, SBP.S([1], 8) },
            placement);
        var result = new DistributedType(
            new TensorType(dataType, new[] { 20, 16, 4 }),
            new SBP[] { SBP.B, SBP.S([1], 2), SBP.B },
            placement);

        Assert.True(IR.Affine.BufferViewUtility.TryCreate(source, result, out _));
    }

    [Fact]
    public void TestVectorBitcastRejectsByteIncompatibleShardGranularity()
    {
        var placement = new Placement([8], "b", "b");
        var source = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 4, 8 }),
            new SBP[] { SBP.B, SBP.S([0], 1) },
            placement);
        var incompatibleResult = new DistributedType(
            new TensorType(DataTypes.BFloat16, new[] { 4, 64 }),
            new SBP[] { SBP.B, SBP.S([0], 1) },
            placement);
        var compatibleResult = new DistributedType(
            new TensorType(DataTypes.BFloat16, new[] { 4, 64 }),
            new SBP[] { SBP.B, SBP.S([0], 8) },
            placement);

        Assert.False(IR.Affine.BufferViewUtility.TryCreate(source, incompatibleResult, out _));
        Assert.True(IR.Affine.BufferViewUtility.TryCreate(source, compatibleResult, out _));
    }

    [Fact]
    public void TestBufferViewAllowsZeroStrideSingletonSuffix()
    {
        var sourceType = new TensorType(DataTypes.Float32, new[] { 4, 1, 8 });
        var resultType = new TensorType(DataTypes.Float32, new[] { 4, 8 });
        Assert.True(IR.Affine.BufferViewUtility.TryCreate(sourceType, resultType, out var transform));
        var source = T.CreateBuffer(sourceType, MemoryLocation.Data, out _)
            .With(strides: new Dimension[] { 8, 0, 1 });

        var strides = IR.Affine.BufferViewUtility.CreateBufferViewStrides(source, resultType, transform);

        Assert.Equal(new long[] { 8, 1 }, strides.Select(stride => stride.FixedValue));
    }

    [Fact]
    public void TestBufferViewAllowsZeroStrideDynamicDegenerateSuffix()
    {
        var sourceType = new TensorType(DataTypes.Float32, new[] { 4, 1 });
        var resultType = new TensorType(DataTypes.Float32, new[] { 4 });
        Assert.True(IR.Affine.BufferViewUtility.TryCreate(sourceType, resultType, out var transform));
        var tailExtent = new DimVar("tail_extent")
        {
            Metadata = new()
            {
                Range = new(0, 1),
            },
        };
        var source = T.CreateBuffer(sourceType, MemoryLocation.Data, out _)
            .With(dimensions: new Dimension[] { 4, tailExtent }, strides: new Dimension[] { 1, 0 });

        var strides = IR.Affine.BufferViewUtility.CreateBufferViewStrides(source, resultType, transform);

        Assert.Equal(new long[] { 1 }, strides.Select(stride => stride.FixedValue));
    }

    [Fact]
    public void TestBufferViewAllowsZeroStrideDistributedLocalSingleton()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var sourceType = new DistributedType(
            new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 4, 8 }),
            new SBP[] { SBP.S([0], 1), SBP.S([1], 1) },
            placement);
        var resultType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new[] { 4, 64 }),
            new SBP[] { SBP.S([0], 1), SBP.S([1], 8) },
            placement);
        Assert.True(IR.Affine.BufferViewUtility.TryCreate(sourceType, resultType, out var transform));
        var source = T.CreateBuffer(sourceType.TensorType, MemoryLocation.Input, out _, distributedType: sourceType);

        var strides = IR.Affine.BufferViewUtility.CreateBufferViewStrides(source, resultType.TensorType, transform);

        Assert.Equal(new long[] { 0, 1 }, strides.Select(stride => stride.FixedValue));
    }

    [Fact]
    public void TestBufferAliasGridDomainUsesLogicalShape()
    {
        var packedType = new VectorType(DataTypes.BFloat16, [4, 8]);
        var storageType = new TensorType(packedType, new[] { 1, 4748 });
        var input = new Var("input", storageType);
        var bitcast = IR.F.Tensors.Bitcast(input, DataTypes.BFloat16);
        var function = new Function("main", Targets.CPUTarget.Kind, bitcast, [input]);
        var selected = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(selected.Body);

        Assert.Equal(new long[] { 1, 4748 }, grid.DomainBounds.ToArray().Select(bound => bound.FixedValue));
        Assert.Equal(new long[] { 1, 151936 }, grid.CheckedShape.ToValueArray());
    }

    [Fact]
    public void TestGridDomainPreservesRuntimeLocalShardDimensionRange()
    {
        var sequenceLength = new DimVar("sequence_length")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };
        var placement = new Placement([4, 8], "yx", "bb");
        var distributedType = new DistributedType(
            new TensorType(DataTypes.BFloat16, new Dimension[] { sequenceLength, 8, 128 }),
            new SBP[] { SBP.S([0], Dimension.CeilDiv(sequenceLength, 4)), SBP.S([1], 1), SBP.B },
            placement);
        var input = new Var("input", distributedType);
        var output = new Var("output", distributedType);

        var grid = IR.F.Affine.Grid()
            .Domain(3, out _)
            .Read(input, IR.Affine.AffineMap.Identity(3), out _)
            .Write(output, IR.Affine.AffineMap.Identity(3), out _)
            .Body(T.Nop())
            .Build();

        var localSequenceLength = Assert.IsType<AsDim>(grid.DomainBounds[0]);
        Assert.Equal(0, localSequenceLength.Metadata.Range!.Value.Min);
        Assert.Equal(32, localSequenceLength.Metadata.Range!.Value.Max);
        Assert.Equal(1, grid.DomainBounds[1].FixedValue);
        Assert.Equal(128, grid.DomainBounds[2].FixedValue);
        Assert.Equal(new long[] { 32, 1, 128 }, CompilerServices.GetMaxShape(new RankedShape(grid.DomainBounds.ToArray())));

        var clonedGrid = grid.Clone();
        Assert.Equal(new long[] { 32, 1, 128 }, CompilerServices.GetMaxShape(new RankedShape(clonedGrid.DomainBounds.ToArray())));
    }

    [Fact]
    public async Task TestChainedBufferAliasGridsPreserveLogicalTypes()
    {
        var packedType = new VectorType(DataTypes.BFloat16, [4, 8]);
        var storageType = new TensorType(packedType, new[] { 2, 32 });
        var input = new Var("input", storageType);
        var reshaped = IR.F.Tensors.Reshape(input, new Dimension[] { 2, 8, 4 });
        var logical = IR.F.Tensors.Bitcast(reshaped, DataTypes.BFloat16);
        var function = new Function("main", Targets.CPUTarget.Kind, IR.F.Math.Unary(UnaryOp.Neg, logical), [input]);

        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var consumerGrid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var bitcastGrid = Assert.IsType<IR.Affine.Grid>(consumerGrid.Accesses[0].Value);
        var reshapeGrid = Assert.IsType<IR.Affine.Grid>(bitcastGrid.Accesses[0].Value);
        Assert.Equal(new long[] { 2, 8, 128 }, consumerGrid.CheckedShape.ToValueArray());
        Assert.Equal(new long[] { 2, 8, 128 }, bitcastGrid.CheckedShape.ToValueArray());
        Assert.Equal(new long[] { 2, 8, 4 }, reshapeGrid.CheckedShape.ToValueArray());
        Assert.Equal(new long[] { 2, 8, 128 }, consumerGrid.DomainBounds.ToArray().Select(bound => bound.FixedValue));

        var graph = TieredTileGraphBuilder.Build(selected.Body, 1, out _);
        Assert.Equal(3, graph.VertexCount);
        Assert.Equal(2, graph.Vertices.Count(node => node.IsPureBufferView));
    }

    [Fact]
    public void TestTileGraphIntersectsLogicalDomainWithBlockLocalStorageDomain()
    {
        var scalarType = new TensorType(DataTypes.BFloat16, new[] { 20, 1024 });
        var packedType = new TensorType(new VectorType(DataTypes.BFloat16, [8]), new[] { 20, 128 });
        var logicalInput = new Var("logical_input", scalarType);
        var logicalOutput = new Var("logical_output", packedType);
        var domains = IR.F.Affine.Domains(2);
        var inputMap = new IR.Affine.AffineMap(domains, default, new IR.Affine.AffineRange[]
        {
            new(domains[0].Offset, domains[0].Extent),
            new(domains[1].Offset * 8, domains[1].Extent * 8),
        });
        var logicalGrid = IR.F.Affine.Grid()
            .Domain(2, out _)
            .Read(logicalInput, inputMap, out var inputTile)
            .Write(logicalOutput, IR.Affine.AffineMap.Identity(2), out var outputTile)
            .Body(TIR.F.NTT.Pack(inputTile, outputTile, [8], [1]))
            .Build();
        Assert.Equal(new long[] { 20, 128 }, logicalGrid.DomainBounds.ToArray().Select(bound => bound.FixedValue));

        var placement = new Placement([4, 8], "yx", "bb");
        var physicalInput = new Var(
            "physical_input",
            new DistributedType(scalarType, new SBP[] { SBP.B, SBP.S([0, 1], 32) }, placement));
        var physicalOutput = new Var(
            "physical_output",
            new DistributedType(packedType, new SBP[] { SBP.B, SBP.S([0, 1], 4) }, placement));
        var accesses = logicalGrid.Accesses.ToArray();
        accesses[0] = accesses[0].With(value: physicalInput, buffer: IR.F.Buffer.BufferOf(physicalInput));
        accesses[1] = accesses[1].With(value: physicalOutput, buffer: IR.F.Buffer.BufferOf(physicalOutput));
        var distributedGrid = logicalGrid.With(accesses: accesses);

        var graph = TieredTileGraphBuilder.Build(distributedGrid, 1, out _);
        var tileGrid = Assert.Single(graph.Vertices);
        Assert.Equal(new long[] { 20, 4 }, tileGrid.DomainBounds);
        Assert.Equal(new long[] { 20, 4 }, tileGrid.DomainBoundExprs.Select(bound => bound.FixedValue));
    }

    [Fact]
    public void TestAffineDomainIntersectionSupportsOpaqueDynamicDimensions()
    {
        var runtimeExtent = new AsDim(new Var("runtime_extent", TensorType.Scalar(DataTypes.Int64)))
        {
            Metadata = new()
            {
                Range = new(1, 32),
            },
        };
        var domains = IR.F.Affine.Domains(1);
        var projectedMap = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(
                    new IR.Affine.AffineDivBinary(
                        IR.Affine.AffineDivBinaryOp.FloorDiv,
                        domains[0].Offset,
                        2),
                    domains[0].Extent),
            });

        var bounds = IR.Affine.AffineDomainInference.IntersectDomainBounds(
            new Dimension[] { runtimeExtent },
            new Shape[] { new RankedShape(new Dimension[] { 4 }) },
            new[] { projectedMap });

        Assert.Equal(new long[] { 8 }, CompilerServices.GetMaxShape(new RankedShape(bounds)));
    }

    [Fact]
    public void TestAffineDomainIntersectionPreservesUnevenLocalShardWithVectorView()
    {
        var placement = new Placement([4, 8], "yx", "bb");
        var localN = new AsDim(IR.F.Tensors.LocalShardDim(4748L, SBP.S([0, 1], 149), placement))
        {
            Metadata = new()
            {
                Range = new(0, 149),
            },
        };
        var domains = IR.F.Affine.Domains(3);
        var lhsStorageMap = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(domains[0].Offset, domains[0].Extent),
                new(
                    new IR.Affine.AffineDivBinary(
                        IR.Affine.AffineDivBinaryOp.FloorDiv,
                        domains[2].Offset,
                        8),
                    1),
            });
        var rhsStorageMap = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(domains[1].Offset, domains[1].Extent),
                new(domains[2].Offset, domains[2].Extent),
            });
        var outputStorageMap = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(domains[0].Offset, domains[0].Extent),
                new(domains[1].Offset, domains[1].Extent),
            });

        var bounds = IR.Affine.AffineDomainInference.IntersectDomainBounds(
            new Dimension[] { 1, localN, 1024 },
            new Shape[]
            {
                new RankedShape(new Dimension[] { 1, 128 }),
                new RankedShape(new Dimension[] { 4748, 1024 }),
                new RankedShape(new Dimension[] { 1, localN }),
            },
            new[] { lhsStorageMap, rhsStorageMap, outputStorageMap });

        Assert.Equal(new long[] { 1, 149, 1024 }, CompilerServices.GetMaxShape(new RankedShape(bounds)));
        var runtimeLocalN = Assert.IsType<AsDim>(bounds[1]);
        Assert.IsType<IR.Tensors.LocalShardDim>(((Call)runtimeLocalN.Dim).Target);
    }

    [Fact]
    public void TestPhysicalAccessMapRestrictsStaticFullAxisRange()
    {
        var domains = IR.F.Affine.Domains(1);
        var map = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(domains[0].Offset, domains[0].Extent),
                new(0, 128),
            });

        var restricted = IR.Affine.AffineUtility.RestrictAccessMapToShape(map, new long[] { 20, 4 });

        Assert.Equal(128, Assert.IsType<IR.Affine.AffineConstant>(map.Results[1].Extent).Value);
        Assert.Equal(4, Assert.IsType<IR.Affine.AffineConstant>(restricted.Results[1].Extent).Value);
    }

    [Fact]
    public async Task TestTileGraphFusesThroughRankChangingBufferAliasGrid()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var inputType = new TensorType(DataTypes.Float32, new[] { 4, 8 });
        var input = new Var("input", inputType);
        var producer = IR.F.Math.Unary(UnaryOp.Neg, input);
        var view = IR.F.Tensors.Reshape(producer, new Dimension[] { 4, 1, 8 });
        var consumer = IR.F.Math.Unary(UnaryOp.Abs, view);
        var function = new Function("main", Targets.CPUTarget.Kind, consumer, [input]);

        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var consumerGrid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var viewGrid = Assert.IsType<IR.Affine.Grid>(consumerGrid.Accesses[0].Value);
        Assert.IsType<IR.Affine.Grid>(viewGrid.Accesses[0].Value);
        Assert.Equal(new long[] { 4, 8 }, viewGrid.DomainBounds.ToArray().Select(bound => bound.FixedValue));

        var graph = TieredTileGraphBuilder.Build(selected.Body, 1, out _);
        Assert.Equal(3, graph.VertexCount);
        var viewNode = Assert.Single(graph.Vertices.Where(node => ReferenceEquals(node.Grid, viewGrid)));
        var consumerNode = Assert.Single(graph.Vertices.Where(node => ReferenceEquals(node.Grid, consumerGrid)));
        Assert.True(viewNode.IsPureBufferView, "Reshape Grid must be recognized as a pure buffer alias.");
        var viewUse = Assert.Single(graph.GetMergePoints().Where(point => ReferenceEquals(point.Producer, viewNode) && ReferenceEquals(point.Consumer, consumerNode)));
        Assert.True(graph.Merge(viewUse), "Reshape Grid must fuse into its consumer.");
        var targetOptions = Assert.IsAssignableFrom<INTTTargetOptions>(CompileOptions.TargetOptions);
        _ = new Schedule.GraphTiler().SolveRootGraph(
            graph,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));
        Assert.DoesNotContain(ExprCollector.Collect(tiled.Body), expression => expression is IR.Affine.Grid);
        var deviceFunctions = ExprCollector.Collect(tiled.Body)
            .OfType<PrimFunctionWrapper>()
            .Select(wrapper => wrapper.Target)
            .OfType<TIR.PrimFunction>()
            .Distinct((IEqualityComparer<TIR.PrimFunction>)ReferenceEqualityComparer.Instance)
            .ToArray();
        Assert.NotEmpty(deviceFunctions);
        Assert.DoesNotContain(
            deviceFunctions.SelectMany(function => ExprCollector.Collect(function.Body)),
            expression => expression is Call { Target: TIR.NTT.Reshape });
    }

    [Fact]
    public async Task TestProducerFusedIntoBufferAliasResolvesDefStorageView()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 8 }));
        var producer = IR.F.Math.Unary(UnaryOp.Neg, input);
        var view = IR.F.Tensors.Reshape(producer, new Dimension[] { 4, 1, 8 });
        var consumer = IR.F.Math.Unary(UnaryOp.Abs, view);
        var function = new Function("main", Targets.CPUTarget.Kind, consumer, [input]);
        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var consumerGrid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var viewGrid = Assert.IsType<IR.Affine.Grid>(consumerGrid.Accesses[0].Value);
        var producerGrid = Assert.IsType<IR.Affine.Grid>(viewGrid.Accesses[0].Value);
        var graph = TieredTileGraphBuilder.Build(selected.Body, 1, out _);
        var producerNode = Assert.Single(graph.Vertices.Where(node => ReferenceEquals(node.Grid, producerGrid)));
        var viewNode = Assert.Single(graph.Vertices.Where(node => ReferenceEquals(node.Grid, viewGrid)));
        var producerIntoView = Assert.Single(graph.GetMergePoints().Where(point =>
            ReferenceEquals(point.Producer, producerNode) &&
            ReferenceEquals(point.Consumer, viewNode)));
        Assert.True(graph.Merge(producerIntoView));

        var targetOptions = Assert.IsAssignableFrom<INTTTargetOptions>(CompileOptions.TargetOptions);
        _ = new Schedule.GraphTiler().SolveRootGraph(
            graph,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
    }

    [Fact]
    public async Task TestTileGraphFusesThroughDTypeChangingBufferAliasGrid()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var sourceDataType = new VectorType(DataTypes.BFloat16, [8]);
        var sourceType = new TensorType(sourceDataType, new[] { 1, 8 });
        var input = new Var("input", sourceType);
        var producer = IR.F.Math.Unary(UnaryOp.Neg, input);
        var logicalInput = IR.F.Tensors.Bitcast(producer, DataTypes.BFloat16);
        var consumer = IR.F.Math.Unary(UnaryOp.Abs, logicalInput);
        var function = new Function("main", Targets.CPUTarget.Kind, consumer, [input]);

        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var consumerGrid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var bitcastGrid = Assert.IsType<IR.Affine.Grid>(consumerGrid.Accesses[0].Value);
        Assert.IsType<IR.Affine.Grid>(bitcastGrid.Accesses[0].Value);
        var graph = TieredTileGraphBuilder.Build(selected.Body, 1, out _);
        var bitcastNode = Assert.Single(graph.Vertices.Where(node => ReferenceEquals(node.Grid, bitcastGrid)));
        Assert.True(bitcastNode.IsPureBufferView);

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));
        Assert.DoesNotContain(ExprCollector.Collect(tiled.Body), expression => expression is IR.Affine.Grid);
        var deviceFunctions = ExprCollector.Collect(tiled.Body)
            .OfType<PrimFunctionWrapper>()
            .Select(wrapper => wrapper.Target)
            .ToArray();
        Assert.NotEmpty(deviceFunctions);
        Assert.DoesNotContain(
            deviceFunctions.SelectMany(function => ExprCollector.Collect(function.Body)),
            expression => expression is Call { Target: TIR.NTT.Bitcast });
    }

    [Fact]
    public void TestTileGraphFusesThroughTupleGetItem()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { 4, 64 }));
        var packedDataType = new VectorType(DataTypes.BFloat16, [4, 8]);
        var qWeight = new Var("q_weight", new TensorType(packedDataType, new[] { 4, 64 }));
        var kWeight = new Var("k_weight", new TensorType(packedDataType, new[] { 2, 64 }));
        var vWeight = new Var("v_weight", new TensorType(packedDataType, new[] { 2, 64 }));
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
            numHeads: 4,
            numKvHeads: 2,
            outDataType: DataTypes.BFloat16);
        var key = qkv[1];
        var consumer = IR.F.Math.Unary(UnaryOp.Neg, key);
        var function = new Function("main", Targets.CPUTarget.Kind, new IR.Tuple(consumer, qkv[2]), [input, qWeight, kWeight, vWeight]);

        var selected = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var graph = TieredTileGraphBuilder.Build(selected.Body, 1, out _);
        Assert.Equal(2, graph.VertexCount);
        var edge = Assert.Single(graph.Edges);
        Assert.Equal(1, Nncase.Schedule.TileGraph.GraphExtensions.GetProducerOutputIndex(edge.Target.Grid.Accesses[edge.Tag].Value, edge.Source));
        Assert.True(graph.Merge(new MergePoint(edge.Target, edge.Source, 0, edge.Tag)));
    }

    [Fact]
    public async Task TestRootBufferAliasGridLowersToInputBackedPrimFunctionResult()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 4 }));
        var reshape = IR.F.Tensors.Reshape(input, new Dimension[] { 2, 8 });
        var function = new Function("main", Targets.CPUTarget.Kind, reshape, [input]);
        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));
        Assert.IsType<IR.Affine.Grid>(tiled.Body);
        Assert.DoesNotContain(ExprCollector.Collect(tiled.Body), expression => expression is PrimFunctionWrapper);

        var lowered = Assert.IsType<TIR.PrimFunction>(await new NTTTIRSelectionPass(CompileOptions).RunAsync(tiled, new()));
        var abi = lowered.GetAbiView();
        Assert.Empty(abi.OutputParameters);
        var result = Assert.Single(abi.Results);
        Assert.Same(Assert.Single(abi.Inputs), result.Storage);
        Assert.DoesNotContain(ExprCollector.Collect(lowered.Body).OfType<Call>(), call => call.Target is TIR.Memcopy);
    }

    [Fact]
    public async Task TestLiveOutBufferAliasDoesNotCreateEmptyPrimFunction()
    {
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = new(1, 16) } };
        var input = new Var("input", new TensorType(DataTypes.Float32, new Dimension[] { sequenceLength, 4 }));
        var producer = IR.F.Math.Unary(UnaryOp.Neg, input);
        var reshape = IR.F.Tensors.Reshape(producer, new Dimension[] { sequenceLength, 2, 2 });
        var function = new Function("main", Targets.CPUTarget.Kind, reshape, [input]);
        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var selectedView = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));

        var residualView = Assert.IsType<IR.Affine.Grid>(tiled.Body);
        Assert.Same(selectedView.CheckedShape[0], residualView.CheckedShape[0]);
        var wrapperCall = Assert.IsType<Call>(residualView.Accesses[0].Value);
        var wrapper = Assert.IsType<PrimFunctionWrapper>(wrapperCall.Target);
        Assert.NotEmpty(wrapper.Target.Body.Fields.ToArray());
        Assert.Single(ExprCollector.Collect(tiled.Body).OfType<Call>().Where(call => call.Target is PrimFunctionWrapper));

        var lowered = Assert.IsType<TIR.PrimFunction>(await new NTTTIRSelectionPass(CompileOptions).RunAsync(tiled, new()));
        Assert.DoesNotContain(ExprCollector.Collect(lowered), expression => expression is IR.Affine.Grid or PrimFunctionWrapper);
        var deviceCall = Assert.Single(ExprCollector.Collect(lowered.Body).OfType<Call>().Where(call => call.Target is TIR.PrimFunction));
        var deviceFunction = Assert.IsType<TIR.PrimFunction>(deviceCall.Target);
        Assert.Single(ExprCollector.Collect(deviceFunction.Body).OfType<Call>().Where(call => call.Target is TIR.NTT.Unary));
        Assert.DoesNotContain(ExprCollector.Collect(lowered.Body).OfType<Call>(), call => call.Target is TIR.Memcopy);
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
        Assert.Equal(3, grid.Accesses.ToArray().Count(access => access.IsRead));
        Assert.Equal(1, grid.Accesses.ToArray().Count(access => access.IsWrite));
        Assert.Equal(2, grid.Accesses[0].AffineMap.Domains.Length);
        var kRange = grid.Accesses[0].AffineMap.Results[^1];
        Assert.Equal(0, Assert.IsType<IR.Affine.AffineConstant>(kRange.Offset).Value);
        Assert.Equal(512, Assert.IsType<IR.Affine.AffineConstant>(kRange.Extent).Value);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        Assert.IsType<TIR.NTT.PackedMatMulGlu>(bodyCall.Target);
    }

    [Fact]
    public void TestPackedQKVParallelLinearAffineSelectionUsesSharedProjectionDomain()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
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
        Assert.Equal(4, grid.Accesses.ToArray().Count(access => access.IsRead));
        Assert.Equal(3, grid.Accesses.ToArray().Count(access => access.IsWrite));
        Assert.All(grid.Accesses.ToArray(), access => Assert.Equal(2, access.AffineMap.Domains.Length));

        var domainOffsets = new long[] { 3, 5 };
        var domainExtents = new long[] { 7, 11 };
        var qWeightRange = grid.Accesses[1].AffineMap.Results[0].Apply(domainOffsets, domainExtents);
        var kWeightRange = grid.Accesses[2].AffineMap.Results[0].Apply(domainOffsets, domainExtents);
        var vWeightRange = grid.Accesses[3].AffineMap.Results[0].Apply(domainOffsets, domainExtents);
        Assert.Equal(new ValueRange<long>(10, 22), qWeightRange);
        Assert.Equal(new ValueRange<long>(5, 11), kWeightRange);
        Assert.Equal(new ValueRange<long>(5, 11), vWeightRange);

        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        Assert.IsType<TIR.NTT.PackedQKVParallelLinear>(bodyCall.Target);

        var tileGraph = TieredTileGraphBuilder.Build(grid, 1, out _);
        var tileGrid = Assert.Single(tileGraph.Vertices);
        Assert.IsType<Schedule.MatrixTileWorkload>(tileGrid.GetTileWorkload());
        foreach (var outputAccessIndex in tileGrid.WriteAccessIndices)
        {
            Assert.Equal(MemoryEffect.Write, tileGrid.LocalAccessEffects[outputAccessIndex]);
        }
    }

    [Fact]
    public void TestMatMulLocalAccumulatorEffectComesFromParameterInfo()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var function = FunctionSamples.GetMatmul();
        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);
        var tileGraph = TieredTileGraphBuilder.Build(grid, 1, out _);
        var tileGrid = Assert.Single(tileGraph.Vertices);
        var outputAccessIndex = Assert.Single(tileGrid.WriteAccessIndices);

        Assert.Equal(IR.Affine.GridAccessMode.Write, tileGrid.Grid.Accesses[outputAccessIndex].AccessMode);
        Assert.Equal(MemoryEffect.ReadWrite, tileGrid.LocalAccessEffects[outputAccessIndex]);
        Assert.IsType<Schedule.MatrixTileWorkload>(tileGrid.GetTileWorkload());
    }

    [Fact]
    public void TestRoPEFullHeadAxisTilingPolicyIsStoredOnGrid()
    {
        var input = new Var("input", new TensorType(new VectorType(DataTypes.Float16, [8]), new[] { 20, 64, 16 }));
        var cos = new Var("cos", new TensorType(new VectorType(DataTypes.Float32, [8]), new[] { 20, 1, 16 }));
        var sin = new Var("sin", new TensorType(new VectorType(DataTypes.Float32, [8]), new[] { 20, 1, 16 }));
        var rope = IR.F.NTT.VectorizedRoPE(input, cos, sin);
        var function = new Function("main", Targets.CPUTarget.Kind, rope, [input, cos, sin]);
        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);

        Assert.Equal(3, grid.TileAxisPolicies.Count);
        Assert.Equal(IR.Affine.GridTileExtentKind.Search, grid.TileAxisPolicies[0].ExtentKind);
        Assert.Equal(IR.Affine.GridTileExtentKind.FullExtent, grid.TileAxisPolicies[1].ExtentKind);
        Assert.Equal(IR.Affine.GridTileExtentKind.Search, grid.TileAxisPolicies[2].ExtentKind);
        Assert.Equal(grid.TileAxisPolicies, grid.With().TileAxisPolicies);
    }

    [Fact]
    public void TestUpdatePagedAttentionKVCacheAffineSelectionUsesObjectAsSsaOutput()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
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
        Assert.Equal(2, grid.Accesses.Length);
        Assert.Equal(IR.Affine.GridAccessMode.Read, grid.Accesses[0].AccessMode);
        Assert.Equal(IR.Affine.GridBindingMode.Subview, grid.Accesses[0].BindingMode);
        Assert.Equal(IR.Affine.GridAccessMode.ReadWrite, grid.Accesses[1].AccessMode);
        Assert.Equal(IR.Affine.GridBindingMode.Root, grid.Accesses[1].BindingMode);
        Assert.False(grid.Accesses[1].IsAffine);
        Assert.Same(kvCache, grid.Accesses[1].Value);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        Assert.IsType<TIR.NTT.UpdatePagedAttentionKVCache>(bodyCall.Target);
        Assert.Equal(MemoryEffect.ChipWrite, TIR.NTT.UpdatePagedAttentionKVCache.KVCaches.MemoryEffect);

        var tileGraph = TieredTileGraphBuilder.Build(grid, 1, out _);
        var tileGrid = Assert.Single(tileGraph.Vertices);
        Assert.Equal(MemoryEffect.Read, tileGrid.LocalAccessEffects[0]);
        Assert.Equal(MemoryEffect.ChipWrite, tileGrid.LocalAccessEffects[1]);
    }

    [Fact]
    public void TestBoxingAffineSelectionTilesSourceAndKeepsDestinationRootBound()
    {
        var placement = new Placement([4], "b", "b");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 16 });
        var inputType = new DistributedType(tensorType, new SBP[] { SBP.S([0], 0), SBP.B }, placement);
        var outputType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([0], 1) }, placement);
        var input = new Var("input", inputType);
        var boxing = IR.F.Distributed.Boxing(input, outputType);
        var function = new Function("main", Targets.CPUTarget.Kind, boxing, [input]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);
        Assert.Equal(2, grid.Accesses.Length);
        Assert.Equal(IR.Affine.GridAccessMode.Read, grid.Accesses[0].AccessMode);
        Assert.Equal(IR.Affine.GridBindingMode.Subview, grid.Accesses[0].BindingMode);
        Assert.Equal(IR.Affine.GridDomainMode.Constraint, grid.Accesses[0].DomainMode);
        Assert.Equal(IR.Affine.GridAccessMode.Write, grid.Accesses[1].AccessMode);
        Assert.Equal(IR.Affine.GridBindingMode.Root, grid.Accesses[1].BindingMode);
        Assert.Equal(IR.Affine.GridDomainMode.Footprint, grid.Accesses[1].DomainMode);
        Assert.IsType<TIR.NTT.GatherReduceScatter>(Assert.IsType<Call>(Assert.Single(grid.Body.Fields.ToArray())).Target);
        Assert.Equal(MemoryEffect.ChipRead, TIR.NTT.GatherReduceScatter.Input.MemoryEffect);
        Assert.Equal(MemoryEffect.ChipWrite, TIR.NTT.GatherReduceScatter.Output.MemoryEffect);
    }

    [Fact]
    public void TestTensorToDistributedBoxingKeepsGlobalSourceRootBound()
    {
        var placement = new Placement([4], "b", "b");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 16 });
        var outputType = new DistributedType(tensorType, new SBP[] { SBP.S([0], 8), SBP.B }, placement);
        var input = new Var("input", tensorType);
        var boxing = IR.F.Distributed.Boxing(input, outputType);
        var function = new Function("main", Targets.CPUTarget.Kind, boxing, [input]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);

        Assert.Equal(IR.Affine.GridBindingMode.Root, grid.Accesses[0].BindingMode);
        Assert.Equal(IR.Affine.GridDomainMode.Footprint, grid.Accesses[0].DomainMode);
        Assert.Equal(IR.Affine.GridBindingMode.Root, grid.Accesses[1].BindingMode);
        Assert.Equal(IR.Affine.GridDomainMode.Constraint, grid.Accesses[1].DomainMode);
        Assert.IsType<TIR.NTT.TensorLoad>(Assert.IsType<Call>(Assert.Single(grid.Body.Fields.ToArray())).Target);
    }

    [Fact]
    public async Task TestAutoTileDoesNotChargeCallerAllocatedRootsToLocalCapacity()
    {
        var placement = new Placement([4], "b", "b");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 4096 });
        var outputType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.B }, placement);
        var input = new Var("input", tensorType);
        var boxing = IR.F.Distributed.Boxing(input, outputType);
        var selectionInput = new Function("main", Targets.CPUTarget.Kind, boxing, [input]);
        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions).RunAsync(selectionInput, new()));

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));

        var call = Assert.IsType<Call>(tiled.Body);
        var primFunction = Assert.IsType<PrimFunctionWrapper>(call.Target).Target;
        Assert.DoesNotContain(
            ExprCollector.Collect(primFunction.Body).OfType<PhysicalBuffer>(),
            buffer => buffer.Location == MemoryLocation.Cache && buffer.Size.FixedValue > 0);
    }

    [Fact]
    public void TestPartialBoxingReadsAcrossChipAtSubviewGranularity()
    {
        var placement = new Placement([4], "b", "b");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 16 });
        var inputType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([0], 1) }, placement, SBP.P([0], ReduceOp.Sum));
        var outputType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([0], 1) }, placement);
        var input = new Var("input", inputType);
        var boxing = IR.F.Distributed.Boxing(input, outputType);
        var function = new Function("main", Targets.CPUTarget.Kind, boxing, [input]);

        var post = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(post.Body);

        Assert.Equal(IR.Affine.GridBindingMode.Subview, grid.Accesses[0].BindingMode);
        Assert.Equal(IR.Affine.GridBindingMode.Root, grid.Accesses[1].BindingMode);
        Assert.Equal(MemoryEffect.ChipRead, TIR.NTT.GatherReduceScatter.Input.MemoryEffect);
        Assert.Equal(MemoryEffect.ChipWrite, TIR.NTT.GatherReduceScatter.Output.MemoryEffect);
    }

    [Fact]
    public void TestChipVisibleRootWriteIsTileFusionBoundary()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var placement = new Placement([4], "b", "b");
        var tensorType = new TensorType(DataTypes.Float32, new[] { 32, 16 });
        var inputType = new DistributedType(tensorType, new SBP[] { SBP.S([0], 0), SBP.B }, placement);
        var intermediateType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.B }, placement);
        var outputType = new DistributedType(tensorType, new SBP[] { SBP.B, SBP.S([0], 1) }, placement);
        var input = new Var("input", inputType);
        var intermediate = IR.F.Distributed.Boxing(input, intermediateType);
        var output = IR.F.Distributed.Boxing(intermediate, outputType);
        var function = new Function("main", Targets.CPUTarget.Kind, output, [input]);
        var selected = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);

        var graph = TieredTileGraphBuilder.Build(selected.Body, 2, out _);

        Assert.Equal(2, graph.VertexCount);
        Assert.Single(graph.Edges);
        Assert.Empty(graph.GetMergePoints());
        var edge = Assert.Single(graph.Edges);
        Assert.False(graph.Merge(new MergePoint(edge.Target, edge.Source, 1, edge.Tag)));
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
        var selectedTransfer = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var selectedGrid = Assert.IsType<IR.Affine.Grid>(selectedTransfer.Accesses[0].Value);
        Assert.Equal(partialStatsType.Partial, Assert.IsType<DistributedType>(selectedTransfer.Accesses[0].Value.CheckedType).Partial);
        Assert.Equal(partialStatsType.Partial, Assert.IsType<DistributedType>(selectedGrid.CheckedType).Partial);

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));

        Assert.DoesNotContain(ExprCollector.Collect(tiled.Body).OfType<Call>(), call => call.Target is IR.Distributed.Boxing);
        var tiledCall = Assert.IsType<Call>(tiled.Body);
        var tiledFunction = Assert.IsType<PrimFunctionWrapper>(tiledCall.Target).Target;
        var reshard = Assert.Single(ExprCollector.Collect(tiledFunction.Body).OfType<Call>().Select(call => call.Target).OfType<TIR.NTT.GatherReduceScatter>());
        Assert.Equal(partialStatsType.Partial, reshard.InType.Partial);
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
        var targetOptions = (Targets.NTTTargetOptions)CompileOptions.TargetOptions;
        if (levelCount == 1)
        {
            targetOptions.TargetMachineModel = CreateTestMachine(1);
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
            Assert.Equal(excepted, tileGraph.Merge(ResolveMergePoint(tileGraph, point)));
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
        var targetOptions = (Targets.NTTTargetOptions)CompileOptions.TargetOptions;
        if (levelCount == 1)
        {
            targetOptions.TargetMachineModel = CreateTestMachine(1);
        }

        var func = functor();
        var pre = (Function)new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;
        using var dumpScope = new Diagnostics.DumpScope(count.ToString());
#if DEBUG
        Diagnostics.DumpScope.Current.DumpIR(pre, $"post{count}");
#endif
        var tileGraph = TieredTileGraphBuilder.Build(pre.Body, targetOptions.TargetMachineModel.TilingMemorySpaces.Length, out var exprMemo);

        for (int i = 0; i < mergePoints.Length; i++)
        {
            var point = mergePoints[i];
            tileGraph.Merge(ResolveMergePoint(tileGraph, point));
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
        var targetOptions = (Targets.NTTTargetOptions)CompileOptions.TargetOptions;
        if (levelCount == 1)
        {
            targetOptions.TargetMachineModel = CreateTestMachine(1);
        }

        var func = functor();
        var post = new NTTAffineSelectionPass(CompileOptions).RunAsync(func, new()).Result;

        using var dumpScope = new Diagnostics.DumpScope(count.ToString());
        var tileGraph = TieredTileGraphBuilder.Build(post, targetOptions.TargetMachineModel.TilingMemorySpaces.Length, out var exprMemo);

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
        var tileGraph = TieredTileGraphBuilder.Build(post, targetOptions.TargetMachineModel.TilingMemorySpaces.Length, out var exprMemo);

        for (int i = 0; i < mergePoints.Length; i++)
        {
            var point = mergePoints[i];
            tileGraph.Merge(ResolveMergePoint(tileGraph, point));
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

    private static TargetMachineModel CreateTestMachine(int levelCount)
    {
        if (levelCount is < 1 or > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(levelCount), levelCount, "TileGraph tests support one or two local memory levels.");
        }

        var root = new TargetMemorySpaceId("test.main-memory");
        var localSpaces = Enumerable.Range(0, levelCount)
            .Select(level => new TargetMemorySpaceSpec(
                new TargetMemorySpaceId($"test.cache.l{level}"),
                TargetMemorySpaceKind.Cache,
                MemorySharingScope.Block,
                new(MemoryLocation.Cache, level),
                level == 0 ? 256 * 1024 : 512 * 1024,
                level == 0 ? 128 : 64,
                level == 0 ? 128 : 64,
                4L << level,
                true,
                level,
                true,
                true,
                false,
                64))
            .ToArray();
        var rootSpace = new TargetMemorySpaceSpec(
            root,
            TargetMemorySpaceKind.Global,
            MemorySharingScope.Chip,
            null,
            int.MaxValue,
            levelCount == 1 ? 16 : 8,
            levelCount == 1 ? 16 : 8,
            120,
            false,
            -1,
            true,
            true,
            false,
            64);
        return new TargetMachineModel(
            $"tile-graph-test-{levelCount}",
            new(BlockExecutionKind.CpuCore, 1, 1, 1, 1.0, 512, 4),
            new(16, 16, ImmutableArray<MatrixComputePrimitiveSpec>.Empty),
            new(25, 25_000),
            localSpaces.Append(rootSpace),
            root,
            Enumerable.Range(0, localSpaces.Length).SelectMany(level =>
            {
                var localSpace = localSpaces[level];
                var parentSpace = level + 1 < localSpaces.Length ? localSpaces[level + 1] : rootSpace;
                return new[]
                {
                    new TargetMemoryTransferSpec(parentSpace.Id, localSpace.Id, Math.Min(parentSpace.ReadBytesPerCycle, localSpace.WriteBytesPerCycle), parentSpace.LatencyCycles),
                    new TargetMemoryTransferSpec(localSpace.Id, parentSpace.Id, Math.Min(localSpace.ReadBytesPerCycle, parentSpace.WriteBytesPerCycle), parentSpace.LatencyCycles),
                };
            }),
            new Dictionary<MemoryLocation, TargetMemorySpaceId>());
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

    private static MergePoint ResolveMergePoint(TieredTileGraph graph, IntMergePoint point)
    {
        var consumer = graph.Vertices.Skip(point.Consumer).First();
        var producer = graph.Vertices.Skip(point.Producer).First();
        var edge = graph.Edges.SingleOrDefault(candidate =>
            ReferenceEquals(candidate.Source, producer) &&
            ReferenceEquals(candidate.Target, consumer));
        return new MergePoint(consumer, producer, point.Level, edge?.Tag ?? -1);
    }

    public sealed record IntMergePoint(int Consumer, int Producer, int Level)
    {
        public override string ToString() => $"merge({Consumer},{Producer},{Level})";
    }
}
