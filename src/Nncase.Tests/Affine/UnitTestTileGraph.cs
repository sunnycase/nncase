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

    public static readonly TheoryData<Func<Function>, int> HierarchicalPlannerDatas = new()
    {
        { FunctionSamples.GetBinaryNeg, 0 },
        { FunctionSamples.GetMatmulExpMatmul, 1 },
        { FunctionSamples.GetAddBranchMerge, 2 },
        { FunctionSamples.GetQwen3Rope, 3 },
    };

    public UnitTestTileGraph()
    {
        CompileOptions.TargetOptions = new Targets.NTTTargetOptions { TargetMachineModel = CreateTestMachine(2) };
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Tiling;
#endif
    }

    [Fact]
    public void TestTileLifetimeUsesInclusiveExecutionPhases()
    {
        var inputLifetime = new TileLifetime(4, 5);
        var outputLifetime = new TileLifetime(5, 6);

        Assert.Equal(2, inputLifetime.PhaseCount);
        Assert.Equal(2, outputLifetime.PhaseCount);
        Assert.True(inputLifetime.Overlaps(outputLifetime));
        Assert.Equal(new TileLifetime(4, 6), inputLifetime.Union(outputLifetime));
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
    public void TestReduceAffineSelectionPlacesReductionAxesInnermost()
    {
        var input = new Var(
            "input",
            new TensorType(new VectorType(DataTypes.Float32, [1]), new[] { 3, 4 }));
        var reduce = IR.F.NTT.VectorizedReduce(
            input,
            ReduceOp.Sum,
            [0],
            0.0f,
            false,
            [1],
            new RankedShape(new Dimension[] { 0 }));
        var function = new Function("main", Targets.CPUTarget.Kind, reduce, [input]);

        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        Assert.Equal(IR.Affine.GridAxisKind.Parallel, grid.TileAxisPolicies[0].AxisKind);
        Assert.Equal(IR.Affine.GridAxisKind.Reduction, grid.TileAxisPolicies[1].AxisKind);

        var inputMap = grid.Accesses[0].AffineMap;
        Assert.Equal(1, Assert.IsType<IR.Affine.AffineDim>(inputMap.Results[0].Offset).Position);
        Assert.Equal(0, Assert.IsType<IR.Affine.AffineDim>(inputMap.Results[1].Offset).Position);
        var outputMap = grid.Accesses[1].AffineMap;
        Assert.Equal(0, Assert.IsType<IR.Affine.AffineDim>(outputMap.Results[0].Offset).Position);
    }

    [Fact]
    public void TestScalarReduceAffineSelectionExposesReductionAxis()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 513 }));
        var reduce = IR.F.Tensors.Reduce(ReduceOp.Mean, input, new[] { 1L }, 0.0f, false);
        var function = new Function("main", Targets.CPUTarget.Kind, reduce, [input]);

        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var grid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        Assert.Equal(IR.Affine.GridAxisKind.Parallel, grid.TileAxisPolicies[0].AxisKind);
        Assert.Equal(IR.Affine.GridAxisKind.Reduction, grid.TileAxisPolicies[1].AxisKind);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        var tirReduce = Assert.IsType<TIR.NTT.Reduce>(bodyCall.Target);
        Assert.Equal(ReduceOp.Mean, tirReduce.ReduceOp);
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
        using var ctx = IntegerSetLibrary.ctx.Create();
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 4 }));
        var view = IR.F.Tensors.Reshape(input, new Dimension[] { 2, 8 });
        var lhs = IR.F.Math.Unary(UnaryOp.Neg, view);
        var rhs = IR.F.Math.Unary(UnaryOp.Abs, view);
        var function = new Function("main", Targets.CPUTarget.Kind, new IR.Tuple(lhs, rhs), [input]);
        var selected = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var graph = TieredTileGraphBuilder.Build(selected.Body, 1, out _);
        var viewNode = Assert.Single(graph.Vertices.Where(node => node.IsPureBufferView));
        var region = TileRegion.Create(graph);
        var viewUses = region.Uses
            .Where(use => use.Id.ProducerOpId == viewNode.OpId)
            .ToArray();
        Assert.Equal(2, viewUses.Length);
        Assert.Equal(2, viewUses.Select(use => use.Id.ConsumerOpId).Distinct().Count());
        Assert.All(viewUses, use => Assert.Equal(MemoryAccessScope.Block, use.RequiredMemoryScope));

        var targetOptions = Assert.IsAssignableFrom<INTTTargetOptions>(CompileOptions.TargetOptions);
        var tiled = new Schedule.GraphTiler().Tile(
            selected.Body,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
        Assert.DoesNotContain(ExprCollector.Collect(tiled), expression => expression is IR.Affine.Grid);
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
    public void TestRectangularProjectionNormalizesComposedAxisScale()
    {
        var domains = IR.F.Affine.Domains(2);
        var map = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(domains[0].Offset, domains[0].Extent),
                new(
                    IR.F.Affine.FloorDiv(2 * domains[1].Offset, 2),
                    IR.F.Affine.FloorDiv(2 * domains[1].Extent, 2)),
                new(0, 64),
            });

        Assert.True(map.IsRectangularProjection(allowConstInResults: true));

        var ceilOffset = new IR.Affine.AffineDivBinary(
            IR.Affine.AffineDivBinaryOp.CeilDiv,
            2 * domains[1].Offset,
            2);
        var ceilExtent = new IR.Affine.AffineDivBinary(
            IR.Affine.AffineDivBinaryOp.CeilDiv,
            2 * domains[1].Extent,
            2);
        var ceilMap = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(domains[0].Offset, domains[0].Extent),
                new(
                    2 * ceilOffset,
                    2 * ceilExtent),
            });
        Assert.True(ceilMap.IsRectangularProjection(allowConstInResults: true));

        var nonEquivalentFloorOrder = new IR.Affine.AffineMap(
            domains,
            default,
            new IR.Affine.AffineRange[]
            {
                new(domains[0].Offset, domains[0].Extent),
                new(
                    2 * IR.F.Affine.FloorDiv(domains[1].Offset, 2),
                    domains[1].Extent),
            });
        Assert.False(nonEquivalentFloorOrder.IsRectangularProjection(allowConstInResults: true));
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
        Assert.True(viewNode.IsPureBufferView, "Reshape Grid must be recognized as a pure buffer alias.");

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
        var region = TileRegion.Create(graph);
        Assert.Contains(
            region.Uses,
            use => use.Id.ProducerOpId == producerNode.OpId &&
                use.Id.ConsumerOpId == viewNode.OpId &&
                use.RequiredMemoryScope == MemoryAccessScope.Block);

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));
        Assert.DoesNotContain(ExprCollector.Collect(tiled.Body), expression => expression is IR.Affine.Grid);
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
    public async Task TestInputBackedBufferAliasDoesNotCreateEmptyTileScope()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 8 }));
        var view = IR.F.Tensors.Reshape(input, new Dimension[] { 4, 1, 8 });
        var consumer = IR.F.Math.Unary(UnaryOp.Abs, view);
        var function = new Function("main", Targets.CPUTarget.Kind, consumer, [input]);

        var selected = Assert.IsType<Function>(
            await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var tiled = Assert.IsType<Function>(
            await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));
        var deviceBodies = ExprCollector.Collect(tiled.Body)
            .OfType<PrimFunctionWrapper>()
            .SelectMany(wrapper => ExprCollector.Collect(wrapper.Target.Body))
            .ToArray();

        Assert.NotEmpty(deviceBodies);
        Assert.DoesNotContain(deviceBodies.OfType<TIR.Let>(), let => let.Body.Count == 0);
        Assert.DoesNotContain(deviceBodies.OfType<TIR.For>(), loop => loop.Body.Count == 0);
        Assert.Single(deviceBodies.OfType<Call>().Where(call => call.Target is TIR.NTT.Unary));
    }

    [Fact]
    public void TestTileGraphTracksTupleGetItemUses()
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
        var queryConsumer = IR.F.Math.Unary(UnaryOp.Abs, qkv[0]);
        var keyConsumer = IR.F.Math.Unary(UnaryOp.Neg, qkv[1]);
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            new IR.Tuple(queryConsumer, keyConsumer, qkv[2]),
            [input, qWeight, kWeight, vWeight]);

        var selected = Assert.IsType<Function>(new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var targetOptions = Assert.IsAssignableFrom<INTTTargetOptions>(CompileOptions.TargetOptions);
        var graph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        Assert.Equal(3, graph.VertexCount);
        var producerOutputIndices = graph.Edges
            .Select(edge => Nncase.Schedule.TileGraph.GraphExtensions.GetProducerOutputIndex(
                edge.Target.Grid.Accesses[edge.Tag].Value,
                edge.Source))
            .Order()
            .ToArray();
        Assert.Equal(new[] { 0, 1 }, producerOutputIndices);
        var region = TileRegion.Create(graph);
        Assert.Equal(2, region.Uses.Length);
        Assert.All(region.Uses, use => Assert.Equal(MemoryAccessScope.Block, use.RequiredMemoryScope));

        var levelCount = targetOptions.TargetMachineModel.TilingMemorySpaces.Length;
        Assert.All(region.Uses, use => Assert.True(use.CanFuseAtLevel(levelCount - 1, levelCount)));

        var tiled = new Schedule.GraphTiler().Tile(
            selected.Body,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
        Assert.DoesNotContain(ExprCollector.Collect(tiled), expression => expression is IR.Affine.Grid);
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
    public async Task TestAutoTilePreservesDynamicDimensionParameterIdentity()
    {
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = new(1, 16) } };
        var input = new Var("input", new TensorType(DataTypes.Float32, new Dimension[] { sequenceLength, 4 }));
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Neg, input),
            new IVar[] { input, sequenceLength });
        var selected = Assert.IsType<Function>(
            await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var tiled = Assert.IsType<Function>(
            await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));

        var wrapper = Assert.Single(ExprCollector.Collect(tiled.Body).OfType<PrimFunctionWrapper>());
        var dimensionParameter = Assert.Single(wrapper.Target.Parameters.ToArray().OfType<DimVar>());
        Assert.Same(sequenceLength, dimensionParameter);
        var bodyDimensions = ExprCollector.Collect(wrapper.Target.Body)
            .OfType<DimVar>()
            .Where(dimVar => dimVar.Name == sequenceLength.Name)
            .ToArray();
        Assert.NotEmpty(bodyDimensions);
        Assert.All(bodyDimensions, dimension => Assert.Same(dimensionParameter, dimension));
    }

    [Fact]
    public async Task TestAutoTilePreservesDynamicDimensionCapturedByGridBody()
    {
        var layerId = new DimVar("layer_id") { Metadata = { Range = new(0, 31) } };
        var slots = new Var("slots", new TensorType(DataTypes.BFloat16, new[] { 1, 1, 8 }));
        var objectType = TensorType.Scalar(new ReferenceType(DataTypes.Int32));
        var kvCache = new Var("kv_cache", objectType);
        var update = IR.F.NN.UpdatePagedAttentionKVCache(
            slots,
            kvCache,
            layerId,
            IR.NN.AttentionCacheKind.Key,
            [IR.NN.AttentionDimKind.Seq, IR.NN.AttentionDimKind.Head, IR.NN.AttentionDimKind.Dim]);
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            update,
            new IVar[] { slots, kvCache, layerId });
        var selected = Assert.IsType<Function>(
            await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));
        var selectedGrid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var selectedUpdate = Assert.IsType<Call>(Assert.Single(selectedGrid.Body.Fields.ToArray()));
        Assert.Same(layerId, selectedUpdate[TIR.NTT.UpdatePagedAttentionKVCache.LayerId]);

        var tiled = Assert.IsType<Function>(
            await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));

        var wrapper = Assert.Single(ExprCollector.Collect(tiled.Body).OfType<PrimFunctionWrapper>());
        var dimensionParameter = Assert.Single(wrapper.Target.Parameters.ToArray().OfType<DimVar>());
        Assert.Same(layerId, dimensionParameter);
        var bodyUpdates = ExprCollector.Collect(wrapper.Target.Body)
            .OfType<Call>()
            .Where(call => call.Target is TIR.NTT.UpdatePagedAttentionKVCache)
            .ToArray();
        Assert.NotEmpty(bodyUpdates);
        Assert.All(
            bodyUpdates,
            bodyUpdate => Assert.Same(
                dimensionParameter,
                bodyUpdate[TIR.NTT.UpdatePagedAttentionKVCache.LayerId]));
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
        Assert.Equal(3, grid.Accesses[0].AffineMap.Domains.Length);
        Assert.Equal(IR.Affine.GridAxisKind.Reduction, grid.TileAxisPolicies[^1].AxisKind);
        var kRange = grid.Accesses[0].AffineMap.Results[^1];
        Assert.Equal(2, Assert.IsType<IR.Affine.AffineDim>(kRange.Offset).Position);
        Assert.Equal(2, Assert.IsType<IR.Affine.AffineExtent>(kRange.Extent).Position);
        Assert.Equal(1, grid.Body.Fields.Length);
        var bodyCall = Assert.IsType<Call>(grid.Body.Fields[0]);
        Assert.IsType<TIR.NTT.PackedMatMulGlu>(bodyCall.Target);
    }

    [Fact]
    public void TestHierarchicalPlannerTilesPrimePackedMatMulExtent()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = new Targets.NTTTargetOptions
        {
            TargetMachineModel = Targets.NTTTargetMachineCatalog.Resolve(Targets.NTTTargetMachineCatalog.Rtx5060Ti16Gb),
        };
        CompileOptions.TargetOptions = targetOptions;
        var input = new Var("input", new TensorType(DataTypes.BFloat16, new[] { 1, 1024 }));
        var packedType = new VectorType(DataTypes.BFloat16, [2, 16]);
        var weight = new Var("weight", new TensorType(packedType, new[] { 149, 1024 }));
        var matmul = IR.F.NTT.PackedMatMul(input, weight, outDataType: DataTypes.BFloat16);
        var function = new Function("main", Targets.CPUTarget.Kind, matmul, [input, weight]);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);

        var tiled = new Schedule.GraphTiler().Tile(
            selected.Body,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());

        Assert.DoesNotContain(ExprCollector.Collect(tiled), expression => expression is IR.Affine.Grid);
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
        Assert.All(grid.Accesses.ToArray(), access => Assert.Equal(3, access.AffineMap.Domains.Length));
        Assert.Equal(IR.Affine.GridAxisKind.Reduction, grid.TileAxisPolicies[^1].AxisKind);

        var domainOffsets = new long[] { 3, 5, 13 };
        var domainExtents = new long[] { 7, 11, 17 };
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
            Assert.Equal(MemoryEffect.ReductionWrite, tileGrid.LocalAccessEffects[outputAccessIndex]);
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
        Assert.Equal(MemoryEffect.ReductionReadWrite, tileGrid.LocalAccessEffects[outputAccessIndex]);
        Assert.Equal(IR.Affine.GridAxisKind.Reduction, grid.TileAxisPolicies[^1].AxisKind);
        var workload = Assert.IsType<Schedule.MatrixTileWorkload>(tileGrid.GetTileWorkload());
        var solver = new Google.OrTools.ConstraintSolver.Solver("matrix-accumulator-state");
        var localShapes = tileGrid.BufferShapes
            .Select(shape => shape.Select(extent => (Google.OrTools.ConstraintSolver.IntExpr)solver.MakeIntConst(extent)).ToArray())
            .ToArray();
        localShapes[outputAccessIndex][^2] = solver.MakeIntConst(1);
        localShapes[outputAccessIndex][^1] = solver.MakeIntConst(1);
        var stateBytes = workload.GetReductionStateBytes(
            localShapes,
            solver,
            new(tileGrid.Op, tileGrid.BufferShapes, tileGrid.BufferDataTypes),
            Targets.NTTTargetMachineCatalog.Resolve(Targets.NTTTargetMachineCatalog.Rtx5060Ti16Gb));
        Assert.Equal(16 * 16 * DataTypes.Float32.SizeInBytes, stateBytes.Var().Max());
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
        var reshardCall = Assert.IsType<Call>(Assert.Single(grid.Body.Fields.ToArray()));
        Assert.IsType<TIR.NTT.GatherReduceScatter>(reshardCall.Target);
        Assert.Equal(MemoryEffect.Read, TIR.NTT.GatherReduceScatter.Input.MemoryEffect);
        Assert.Equal(MemoryEffect.ChipWrite, TIR.NTT.GatherReduceScatter.Output.MemoryEffect);
        MemoryEffect? resolvedInputEffect = null;
        MemoryEffectUtility.VisitCallEffects(reshardCall, (_, parameter, effect) =>
        {
            if (ReferenceEquals(parameter, TIR.NTT.GatherReduceScatter.Input))
            {
                resolvedInputEffect = effect;
            }
        });
        Assert.Equal(MemoryEffect.Read, resolvedInputEffect);
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
        var reshardCall = Assert.IsType<Call>(Assert.Single(grid.Body.Fields.ToArray()));
        MemoryEffect? resolvedInputEffect = null;
        MemoryEffectUtility.VisitCallEffects(reshardCall, (_, parameter, effect) =>
        {
            if (ReferenceEquals(parameter, TIR.NTT.GatherReduceScatter.Input))
            {
                resolvedInputEffect = effect;
            }
        });
        Assert.Equal(MemoryEffect.ChipRead, resolvedInputEffect);
        Assert.Equal(MemoryEffect.ChipWrite, TIR.NTT.GatherReduceScatter.Output.MemoryEffect);
    }

    [Fact]
    public void TestChipVisibleRootWriteRequiresOutermostPhaseFusion()
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
        var use = Assert.Single(TileRegion.Create(graph).Uses);
        Assert.Equal(MemoryAccessScope.Chip, use.RequiredMemoryScope);
        Assert.False(use.CanFuseAtLevel(0, 2));
        Assert.True(use.CanFuseAtLevel(1, 2));
    }

    [Fact]
    public void TestRankChangingProducerReduceFusionPreservesIndependentL0Loops()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 8 }));
        var producer = IR.F.Math.Unary(UnaryOp.Neg, input);
        var stats = IR.F.NN.NormStats(1, producer, useMean: false);
        var function = new Function("main", Targets.CPUTarget.Kind, stats, [input]);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var statsGrid = Assert.IsType<IR.Affine.Grid>(selected.Body);
        var producerGrid = Assert.IsType<IR.Affine.Grid>(statsGrid.Accesses[0].Value);
        var targetOptions = (Targets.NTTTargetOptions)CompileOptions.TargetOptions;
        var graph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        var producerNode = Assert.Single(graph.Vertices.Where(node => ReferenceEquals(node.Grid, producerGrid)));
        var statsNode = Assert.Single(graph.Vertices.Where(node => ReferenceEquals(node.Grid, statsGrid)));
        var region = TileRegion.Create(graph);
        var use = Assert.Single(region.Uses.Where(candidate =>
            candidate.Id.ProducerOpId == producerNode.OpId &&
            candidate.Id.ConsumerOpId == statsNode.OpId));

        var plan = new HierarchicalTilePlanner(new Schedule.GraphTiler()).Plan(
            region,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());

        Assert.Equal(
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length - 1,
            plan.Structure.GetFusionLevel(use.Id));
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
        using var ctx = IntegerSetLibrary.ctx.Create();
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

        var targetOptions = (Targets.NTTTargetOptions)CompileOptions.TargetOptions;
        var graph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        var region = TileRegion.Create(graph);
        var chipUse = Assert.Single(region.Uses.Where(use => use.RequiredMemoryScope == MemoryAccessScope.Chip));
        var plan = new HierarchicalTilePlanner(new Schedule.GraphTiler()).Plan(
            region,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
        Assert.Equal(
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length - 1,
            plan.Structure.GetFusionLevel(chipUse.Id));
        var sequentialScopes = new List<TieredTileGraph>();
        plan.ScheduleGraph.Walk(node =>
        {
            if (node is TieredTileGraph { ScopeKind: TileScopeKind.Sequential } sequence)
            {
                sequentialScopes.Add(sequence);
            }
        });
        var sequentialScope = Assert.Single(sequentialScopes);
        Assert.Equal(targetOptions.TargetMachineModel.TilingMemorySpaces.Length - 1, sequentialScope.Level);
        Assert.Equal(0, sequentialScope.DomainRelation.Map.Domains.Length);
        Assert.Equal(0, sequentialScope.DomainRelation.Map.Results.Length);
        Assert.Empty(sequentialScope.DomainBoundExprs);
        Assert.Empty(sequentialScope.LoopOrder);

        var phases = sequentialScope.Clusters.OfType<TieredTileGraph>().ToArray();
        Assert.Equal(2, phases.Length);
        Assert.All(phases, phase =>
        {
            Assert.Equal(TileScopeKind.Iteration, phase.ScopeKind);
            Assert.Equal(sequentialScope.Level, phase.Level);
            Assert.Same(sequentialScope, phase.Parent);
            Assert.NotEmpty(phase.DomainBoundExprs);
        });
        var producerPhase = Assert.Single(phases.Where(phase =>
            phase.Vertices.Any(vertex => vertex.RegionOpId == chipUse.Id.ProducerOpId)));
        var consumerPhase = Assert.Single(phases.Where(phase =>
            phase.Vertices.Any(vertex => vertex.RegionOpId == chipUse.Id.ConsumerOpId)));
        Assert.NotSame(producerPhase, consumerPhase);
        var originalProducerScope = Assert.Single(graph.Clusters.OfType<TieredTileGraph>().Where(scope =>
            scope.Vertices.Any(vertex => vertex.RegionOpId == chipUse.Id.ProducerOpId)));
        var originalConsumerScope = Assert.Single(graph.Clusters.OfType<TieredTileGraph>().Where(scope =>
            scope.Vertices.Any(vertex => vertex.RegionOpId == chipUse.Id.ConsumerOpId)));
        Assert.Equal(originalProducerScope.DomainBoundExprs.ToArray(), producerPhase.DomainBoundExprs.ToArray());
        Assert.Equal(originalConsumerScope.DomainBoundExprs.ToArray(), consumerPhase.DomainBoundExprs.ToArray());

        var rootMaterialization = Assert.Single(plan.Materializations.OfType<TileRootMaterialization>());
        Assert.Equal(MemoryAccessScope.Chip, rootMaterialization.RequiredMemoryScope);
        Assert.Contains(chipUse.Id, rootMaterialization.Uses);

        var tiled = Assert.IsType<Function>(await new AutoTilePass(Targets.CPUTarget.Kind, CompileOptions).RunAsync(selected, new()));

        Assert.DoesNotContain(ExprCollector.Collect(tiled.Body).OfType<Call>(), call => call.Target is IR.Distributed.Boxing);
        Assert.Single(
            ExprCollector.Collect(tiled.Body)
                .OfType<PrimFunctionWrapper>()
                .Distinct((IEqualityComparer<PrimFunctionWrapper>)ReferenceEqualityComparer.Instance));
        var tiledCall = Assert.IsType<Call>(tiled.Body);
        var tiledFunction = Assert.IsType<PrimFunctionWrapper>(tiledCall.Target).Target;
        var calls = ExprCollector.Collect(tiledFunction.Body).OfType<Call>().ToArray();
        Assert.Single(calls.Where(call => call.Target is TIR.NTT.NormStats));
        var tileStore = Assert.Single(calls.Where(call => call.Target is TIR.TileStore));
        var reshardCall = Assert.Single(calls.Where(call => call.Target is TIR.NTT.GatherReduceScatter));
        var reshard = Assert.IsType<TIR.NTT.GatherReduceScatter>(reshardCall.Target);
        Assert.Equal(partialStatsType.Partial, reshard.InType.Partial);

        var rootParameter = Assert.Single(
            tiledFunction.Parameters.ToArray().OfType<BufferVar>().Where(parameter => parameter.Role == BufferVarRole.InOut));
        var letBindings = ExprCollector.Collect(tiledFunction.Body)
            .OfType<TIR.Let>()
            .ToDictionary(
                let => (BaseExpr)let.Var,
                let => let.Expression,
                (IEqualityComparer<BaseExpr>)ReferenceEqualityComparer.Instance);
        Assert.True(IsBackedBy(tileStore[TIR.TileStore.Dest], rootParameter));
        Assert.True(IsBackedBy(reshardCall[TIR.NTT.GatherReduceScatter.Input], rootParameter));
        Assert.DoesNotContain(
            ExprCollector.Collect(tileStore[TIR.TileStore.Dest]).OfType<PhysicalBuffer>(),
            buffer => buffer.Location == MemoryLocation.Shared);
        Assert.True(
            Array.IndexOf(calls, tileStore) < Array.IndexOf(calls, reshardCall),
            "The complete producer phase must store its root result before the chip-visible consumer phase starts.");

        bool IsBackedBy(BaseExpr expression, BaseExpr storage)
        {
            var pending = new Stack<BaseExpr>();
            var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            pending.Push(expression);
            while (pending.TryPop(out var current))
            {
                if (!visited.Add(current))
                {
                    continue;
                }

                var collected = ExprCollector.Collect(current);
                if (collected.Any(candidate => ReferenceEquals(candidate, storage)))
                {
                    return true;
                }

                foreach (var reference in collected.Where(letBindings.ContainsKey))
                {
                    pending.Push(letBindings[reference]);
                }
            }

            return false;
        }
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
    [MemberData(nameof(HierarchicalPlannerDatas))]
    public void TestHierarchicalPlanner(Func<Function> functor, int count)
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = (Targets.NTTTargetOptions)CompileOptions.TargetOptions;
        var function = functor();
        var selected = (Function)new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result;
        using var dumpScope = new Diagnostics.DumpScope($"planner_{count}");
        var tiled = new Schedule.GraphTiler().Tile(
            selected.Body,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());

        Assert.DoesNotContain(ExprCollector.Collect(tiled), expression => expression is IR.Affine.Grid);
    }

    [Fact]
    public void TestTileStructuralScheduleUsesStablePerUseAndScopeIdentity()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var selected = (Function)new NTTAffineSelectionPass(CompileOptions)
            .RunAsync(FunctionSamples.GetAddBranchMerge(), new()).Result;
        var targetOptions = (Targets.NTTTargetOptions)CompileOptions.TargetOptions;
        var graph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        var region = TileRegion.Create(graph);
        var schedule = TileStructuralSchedule.Create(region);

        Assert.NotEmpty(region.Uses);
        Assert.NotEmpty(region.Scopes);
        Assert.Equal(region.Uses.Length, region.Uses.Select(use => use.Id).Distinct().Count());
        Assert.Equal(region.Scopes.Length, region.Scopes.Select(scope => scope.Id).Distinct().Count());
        Assert.All(region.Uses, use => Assert.Equal(-1, schedule.GetFusionLevel(use.Id)));
        Assert.All(
            region.Scopes,
            scope => Assert.Equal(Enumerable.Range(0, scope.Rank), schedule.GetLoopOrder(scope.Id)));

        var first = region.Uses.First(use => use.RequiredMemoryScope != MemoryAccessScope.Chip);
        var fused = schedule.WithFusionLevel(first.Id, targetOptions.TargetMachineModel.TilingMemorySpaces.Length - 1);
        Assert.Equal(-1, schedule.GetFusionLevel(first.Id));
        Assert.NotEqual(schedule, fused);

        var reorderableScope = region.Scopes.FirstOrDefault(scope => scope.Rank > 1);
        if (reorderableScope is not null)
        {
            var reordered = schedule.WithLoopOrder(
                reorderableScope.Id,
                schedule.GetLoopOrder(reorderableScope.Id).Reverse());
            Assert.NotEqual(schedule, reordered);
        }
    }

    [Fact]
    public void TestLexicalLoopOrderControlsGeneratedTIRNesting()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = Assert.IsAssignableFrom<INTTTargetOptions>(CompileOptions.TargetOptions);
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 8, 16 }));
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Neg, input),
            [input]);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var rootGraph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        var outerScope = Assert.Single(rootGraph.Clusters.OfType<TieredTileGraph>());
        Assert.Equal(2, outerScope.DomainBoundExprs.Length);
        outerScope.SetLoopOrder([1, 0]);

        var (argumentMemo, _, _) = new Schedule.GraphTiler().SolveRootGraph(
            rootGraph,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
        var wrapper = Assert.IsType<PrimFunctionWrapper>(
            Assert.Single(argumentMemo.Values.OfType<Call>()).Target);
        var loops = ExprCollector.Collect(wrapper.Target.Body).OfType<TIR.For>().ToArray();
        var outerLevelMarker = $"__l{outerScope.Level}_";
        var d1 = Assert.Single(loops.Where(loop =>
            loop.LoopVar.Name.StartsWith("loop_", StringComparison.Ordinal) &&
            loop.LoopVar.Name.Contains(outerLevelMarker, StringComparison.Ordinal) &&
            loop.LoopVar.Name.EndsWith("__spatial_axis1", StringComparison.Ordinal)));
        var d0 = Assert.Single(loops.Where(loop =>
            loop.LoopVar.Name.StartsWith("loop_", StringComparison.Ordinal) &&
            loop.LoopVar.Name.Contains(outerLevelMarker, StringComparison.Ordinal) &&
            loop.LoopVar.Name.EndsWith("__spatial_axis0", StringComparison.Ordinal)));

        Assert.Contains(d0, ExprCollector.Collect(d1.Body));
        Assert.DoesNotContain(d1, ExprCollector.Collect(d0.Body));
    }

    [Fact]
    public async Task TestAutoTilingUsesSemanticFunctionBufferAndDimensionNames()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = new Targets.NTTTargetOptions
        {
            TargetMachineModel = Targets.NTTTargetMachineCatalog.Resolve(Targets.NTTTargetMachineCatalog.Rtx5060Ti16Gb),
        };
        CompileOptions.TargetOptions = targetOptions;
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = new(1, 16) } };
        var input = new Var("input", new TensorType(DataTypes.Float32, new Dimension[] { sequenceLength, 128 }));
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Abs, IR.F.Math.Unary(UnaryOp.Neg, input)),
            [input, sequenceLength]);
        var selected = Assert.IsType<Function>(
            await new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()));

        var tiled = new Schedule.GraphTiler().Tile(
            selected.Body,
            Targets.CPUTarget.Kind,
            targetOptions,
            [sequenceLength]);
        var wrappers = ExprCollector.Collect(tiled).OfType<PrimFunctionWrapper>().ToArray();
        Assert.NotEmpty(wrappers);
        Assert.All(wrappers, wrapper => Assert.StartsWith("device_", wrapper.Target.Name, StringComparison.Ordinal));
        Assert.Contains(wrappers, wrapper => wrapper.Target.Name.Contains("op0_unary", StringComparison.Ordinal));

        var scheduledBodies = wrappers.SelectMany(wrapper => ExprCollector.Collect(wrapper.Target.Body)).ToArray();
        var generatedDims = scheduledBodies.OfType<DimVar>().ToArray();
        Assert.DoesNotContain(generatedDims, dim => dim.Name.StartsWith("L0_d", StringComparison.Ordinal) || dim.Name.StartsWith("L1_d", StringComparison.Ordinal));
        Assert.Contains(generatedDims, dim =>
            dim.Name.StartsWith("loop_", StringComparison.Ordinal) &&
            dim.Name.Contains("__l0_gpu_shared__", StringComparison.Ordinal));
        Assert.Contains(generatedDims, dim =>
            dim.Name.StartsWith("loop_", StringComparison.Ordinal) &&
            dim.Name.Contains("__l1_gpu_block_global__", StringComparison.Ordinal));

        var buffers = scheduledBodies.OfType<TIR.Buffer>().ToArray();
        Assert.Contains(buffers, buffer =>
            buffer.Name.StartsWith("buffer_", StringComparison.Ordinal) &&
            buffer.Name.Contains("__l0_gpu_shared__at_", StringComparison.Ordinal));
    }

    [Fact]
    public void TestSpatialLoopInsideReductionScopeHasSingleTrip()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = Assert.IsAssignableFrom<INTTTargetOptions>(CompileOptions.TargetOptions);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(FunctionSamples.GetMatmul(), new()).Result);
        var rootGraph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        var outerScope = Assert.Single(rootGraph.Clusters.OfType<TieredTileGraph>());
        var innerScope = Assert.Single(outerScope.Clusters.OfType<TieredTileGraph>());
        Assert.Equal(3, innerScope.DomainBoundExprs.Length);

        // M, K, N is legal when the N loop does not multiplex one backend
        // accumulator across multiple output tiles.
        innerScope.SetLoopOrder([0, 2, 1]);
        var (argumentMemo, _, _) = new Schedule.GraphTiler().SolveRootGraph(
            rootGraph,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
        var wrapper = Assert.IsType<PrimFunctionWrapper>(
            Assert.Single(argumentMemo.Values.OfType<Call>()).Target);
        var loops = ExprCollector.Collect(wrapper.Target.Body).OfType<TIR.For>().ToArray();
        var innerLevelMarker = $"__l{innerScope.Level}_";
        var m = Assert.Single(loops.Where(loop =>
            loop.LoopVar.Name.Contains(innerLevelMarker, StringComparison.Ordinal) &&
            loop.LoopVar.Name.EndsWith("__spatial_axis0", StringComparison.Ordinal)));
        var k = Assert.Single(loops.Where(loop =>
            loop.LoopVar.Name.Contains(innerLevelMarker, StringComparison.Ordinal) &&
            loop.LoopVar.Name.EndsWith("__reduce_axis2", StringComparison.Ordinal)));
        var n = Assert.Single(loops.Where(loop =>
            loop.LoopVar.Name.Contains(innerLevelMarker, StringComparison.Ordinal) &&
            loop.LoopVar.Name.EndsWith("__spatial_axis1", StringComparison.Ordinal)));

        Assert.Equal(LoopMode.Reduction, k.Mode);
        Assert.Contains(k, ExprCollector.Collect(m.Body));
        Assert.Contains(n, ExprCollector.Collect(k.Body));
        Assert.True(n.Domain.Step.IsFixed);
        Assert.True(
            CompilerServices.GetMaxShape(new RankedShape([n.Domain.Stop]))[0] <= n.Domain.Step.FixedValue);
    }

    [Fact]
    public void TestDomainRelationToMapPreservesScaledRangeMultiplicity()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var domains = IR.F.Affine.Domains(1);
        var relation = new DomainRelation(
            0,
            1,
            new IR.Affine.AffineMap(
                domains,
                default,
                [new IR.Affine.AffineRange(domains[0].Offset * 4, domains[0].Extent * 4)]));
        var source = new IntegerSetLibrary.set(ctx, "{ [d0] : d0 = 2 }");

        var image = source.apply(relation.ToMap());

        Assert.Equal(8, image.dim_min_val(0).num_si());
        Assert.Equal(11, image.dim_max_val(0).num_si());
    }

    [Fact]
    public void TestAliasOnlyConnectionHasViewPlacementWithoutStorage()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var inputType = new TensorType(new VectorType(DataTypes.BFloat16, [4, 8]), new[] { 2, 32 });
        var input = new Var("input", inputType);
        var reshapedView = IR.F.Tensors.Reshape(input, new Dimension[] { 2, 8, 4 });
        var scalarView = IR.F.Tensors.Bitcast(reshapedView, DataTypes.BFloat16);
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Neg, scalarView),
            [input]);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var targetOptions = Assert.IsAssignableFrom<INTTTargetOptions>(CompileOptions.TargetOptions);
        var rootGraph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        var region = TileRegion.Create(rootGraph);
        var aliasUses = region.Uses.Where(use => use.IsAliasView).ToArray();
        Assert.NotEmpty(aliasUses);

        var executionPlan = new HierarchicalTilePlanner(new Schedule.GraphTiler()).Plan(
            region,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());

        var aliasMaterializations = executionPlan.Materializations
            .Where(materialization => materialization.Uses.Any(use => aliasUses.Any(alias => alias.Id == use)))
            .ToArray();
        Assert.NotEmpty(aliasMaterializations);
        Assert.All(aliasMaterializations, materialization => Assert.IsType<TileAliasMaterialization>(materialization));
        Assert.All(aliasMaterializations, materialization => Assert.True(materialization.CreationScope.Level >= 0));
    }

    [Fact]
    public void TestHierarchicalPlannerLowersOuterFusionIntoInnerSharedStorage()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = new Targets.NTTTargetOptions
        {
            TargetMachineModel = Targets.NTTTargetMachineCatalog.Resolve(Targets.NTTTargetMachineCatalog.Rtx5060Ti16Gb),
        };
        CompileOptions.TargetOptions = targetOptions;
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1024 }));
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Abs, IR.F.Math.Unary(UnaryOp.Neg, input)),
            [input]);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);

        var tiled = new Schedule.GraphTiler().Tile(
            selected.Body,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
#if DEBUG
        Dumpper.DumpIR(tiled, "shared_tiled");
#endif
        var buffers = ExprCollector.Collect(tiled)
            .OfType<PrimFunctionWrapper>()
            .SelectMany(wrapper => ExprCollector.Collect(wrapper.Target.Body))
            .OfType<PhysicalBuffer>()
            .Where(buffer => buffer.Size.FixedValue > 0)
            .ToArray();

        Assert.Contains(buffers, buffer => buffer.Location == MemoryLocation.Shared);
        var calls = ExprCollector.Collect(tiled)
            .OfType<PrimFunctionWrapper>()
            .SelectMany(wrapper => ExprCollector.Collect(wrapper.Target.Body))
            .OfType<Call>()
            .ToArray();
        Assert.Single(calls.Where(call => call.Target is TIR.TileLoad));
        Assert.Single(calls.Where(call => call.Target is TIR.TileStore));
    }

    [Fact]
    public void TestUnifiedPlacementPropagatesFusedUseAcrossHierarchy()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = new Targets.NTTTargetOptions
        {
            TargetMachineModel = Targets.NTTTargetMachineCatalog.Resolve(Targets.NTTTargetMachineCatalog.Rtx5060Ti16Gb),
        };
        CompileOptions.TargetOptions = targetOptions;

        // The full intermediate exceeds the compiler-managed shared-memory
        // budget, while a legal power-of-two tile fits.
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 131072 }));
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Abs, IR.F.Math.Unary(UnaryOp.Neg, input)),
            [input]);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);
        var rootGraph = TieredTileGraphBuilder.Build(
            selected.Body,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            out _);
        var executionPlan = new HierarchicalTilePlanner(new Schedule.GraphTiler()).Plan(
            TileRegion.Create(rootGraph),
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
        var bufferGraphMemo = executionPlan.ScheduleGraph.Bufferize();
        var ownedEdges = bufferGraphMemo
            .Where(item => item.Value.GetOwnedInterEdges().Any())
            .Select(item => item.Key.Level)
            .ToArray();
        Assert.Equal(new[] { 0 }, ownedEdges);

        var materialization = Assert.Single(executionPlan.Materializations);
        Assert.Equal(0, materialization.CreationScope.Level);
        Assert.True(materialization.LoopEntry >= 0);
        var storage = Assert.IsType<TileStorageMaterialization>(materialization);
        Assert.Equal(targetOptions.TargetMachineModel.TilingMemorySpaces[0].Id, storage.StorageSpace);

        _ = TileNode.FromTileGraph(executionPlan.ScheduleGraph, out var treeMemo);
        var component = Assert.Single(executionPlan.ScheduleGraph.Condense().Vertices);
        var componentTree = treeMemo[component];
        TreeSolverInitializer.Init(
            componentTree,
            bufferGraphMemo,
            targetOptions.TargetMachineModel.TilingMemorySpaces.Length,
            targetOptions,
            out _,
            out _,
            out var tileNodeMemo,
            out _);
        var intermediateLevels = tileNodeMemo
            .Where(item => item.Value.BufferInfoMap.Keys.Any(buffer =>
                buffer.IsOutput &&
                buffer.Node.OpId == 0))
            .Select(item => item.Key.Level)
            .Order()
            .ToArray();
        Assert.Equal(new[] { 0, 1 }, intermediateLevels);
    }

    [Fact]
    public void TestAutoTilingPublishesBlockLocalDataPoolUsage()
    {
        using var ctx = IntegerSetLibrary.ctx.Create();
        var targetOptions = new Targets.NTTTargetOptions
        {
            TargetMachineModel = CreateTestMachine(2, l0AllocationLimitBytes: 64, l0DirectAccess: true),
        };
        CompileOptions.TargetOptions = targetOptions;

        // L0 cannot hold one vector element. Direct-access root operands remain
        // legal, while the internal value must use the BlockLocalData arena.
        var vectorType = new VectorType(DataTypes.Float32, [32]);
        var input = new Var("input", new TensorType(vectorType, new[] { 1024 }));
        var function = new Function(
            "main",
            Targets.CPUTarget.Kind,
            IR.F.Math.Unary(UnaryOp.Abs, IR.F.Math.Unary(UnaryOp.Neg, input)),
            [input]);
        var selected = Assert.IsType<Function>(
            new NTTAffineSelectionPass(CompileOptions).RunAsync(function, new()).Result);

        var tiled = new Schedule.GraphTiler().Tile(
            selected.Body,
            Targets.CPUTarget.Kind,
            targetOptions,
            Array.Empty<DimVar>());
        var wrapper = Assert.Single(ExprCollector.Collect(tiled).OfType<PrimFunctionWrapper>());
        var blockLocalBuffers = ExprCollector.Collect(wrapper.Target.Body)
            .OfType<PhysicalBuffer>()
            .Where(buffer => buffer.Location == MemoryLocation.BlockLocalData && buffer.Size.FixedValue > 0)
            .ToArray();

        Assert.NotEmpty(blockLocalBuffers);
        Assert.True(wrapper.Target.SchedResult.IsScheduled);
        var blockLocalSpace = targetOptions.TargetMachineModel.TilingMemorySpaces[1];
        var requiredBytes = blockLocalBuffers.Max(buffer =>
            checked((long)Assert.IsType<TensorConst>(buffer.Start).Value.ToScalar<ulong>()) +
            buffer.Size.FixedValue);
        var expectedPoolBytes = targetOptions.TargetMachineModel.GetAllocationSizeBytes(
            blockLocalSpace,
            requiredBytes);
        Assert.True(requiredBytes >= 32L * sizeof(float));
        Assert.Equal((ulong)expectedPoolBytes, wrapper.Target.SchedResult.BlockLocalDataPoolSize);
        var resourceAlignment = targetOptions.TargetMachineModel.GetMemoryResource(
            blockLocalSpace).AllocationGranularityBytes;
        Assert.Equal(
            0UL,
            wrapper.Target.SchedResult.BlockLocalDataPoolSize % (ulong)resourceAlignment);
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

    private static TargetMachineModel CreateTestMachine(
        int levelCount,
        long l0AllocationLimitBytes = 256 * 1024,
        bool l0DirectAccess = false)
    {
        if (levelCount is < 1 or > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(levelCount), levelCount, "TileGraph tests support one or two local memory levels.");
        }

        var memoryResourceId = new TargetMemoryResourceId("test.main-memory");
        var memoryResource = new TargetMemoryResourceSpec(
            memoryResourceId,
            TargetMemorySpaceKind.Global,
            int.MaxValue,
            levelCount == 1 ? 16 : 8,
            levelCount == 1 ? 16 : 8,
            120,
            64);
        var resources = new List<TargetMemoryResourceSpec> { memoryResource };
        var localSpaces = Enumerable.Range(0, levelCount)
            .Select(level =>
            {
                var isBlockGlobal = level == levelCount - 1;
                var resourceId = isBlockGlobal || (level == 0 && l0DirectAccess)
                    ? memoryResourceId
                    : new TargetMemoryResourceId($"test.cache.l{level}");
                if (!isBlockGlobal && resourceId != memoryResourceId)
                {
                    resources.Add(new TargetMemoryResourceSpec(
                        resourceId,
                        TargetMemorySpaceKind.Cache,
                        256 * 1024,
                        128,
                        128,
                        4L << level,
                        64));
                }

                return new TargetMemorySpaceSpec(
                    new TargetMemorySpaceId(isBlockGlobal ? "test.block-local-main-memory" : $"test.cache.l{level}"),
                    resourceId,
                    MemorySharingScope.Block,
                    new(isBlockGlobal ? MemoryLocation.BlockLocalData : MemoryLocation.Cache, isBlockGlobal ? 0 : level),
                    isBlockGlobal ? 16 * 1024 * 1024 : level == 0 ? l0AllocationLimitBytes : 256 * 1024,
                    TargetMemoryAllocationSizePolicy.GranularityAligned,
                    true,
                    level,
                    true,
                    true,
                    false);
            })
            .ToArray();
        var root = new TargetMemorySpaceId("test.main-memory");
        var rootSpace = new TargetMemorySpaceSpec(
            root,
            memoryResourceId,
            MemorySharingScope.Chip,
            null,
            int.MaxValue,
            TargetMemoryAllocationSizePolicy.GranularityAligned,
            false,
            -1,
            true,
            true,
            false);
        return new TargetMachineModel(
            $"tile-graph-test-{levelCount}",
            new(BlockExecutionKind.CpuCore, 1, 1, 1, 1.0, 512, 4, 64 * 1024, 1, 1, 1),
            new(16, 16, ImmutableArray<MatrixComputePrimitiveSpec>.Empty),
            new(25, 25_000),
            resources,
            localSpaces.Append(rootSpace),
            root,
            Enumerable.Range(0, localSpaces.Length).SelectMany(level =>
            {
                var localSpace = localSpaces[level];
                var parentSpace = level + 1 < localSpaces.Length ? localSpaces[level + 1] : rootSpace;
                var localResource = resources.Single(resource => resource.Id == localSpace.ResourceId);
                var parentResource = resources.Single(resource => resource.Id == parentSpace.ResourceId);
                return new[]
                {
                    new TargetMemoryTransferSpec(
                        parentSpace.Id,
                        localSpace.Id,
                        Math.Min(parentResource.ReadBytesPerCycle, localResource.WriteBytesPerCycle),
                        parentSpace.ResourceId == localSpace.ResourceId ? 0 : parentResource.LatencyCycles,
                        parentSpace.ResourceId == localSpace.ResourceId ? TargetMemoryTransferMode.DirectAccess : TargetMemoryTransferMode.ExplicitCopy),
                    new TargetMemoryTransferSpec(
                        localSpace.Id,
                        parentSpace.Id,
                        Math.Min(localResource.ReadBytesPerCycle, parentResource.WriteBytesPerCycle),
                        parentSpace.ResourceId == localSpace.ResourceId ? 0 : parentResource.LatencyCycles,
                        parentSpace.ResourceId == localSpace.ResourceId ? TargetMemoryTransferMode.DirectAccess : TargetMemoryTransferMode.ExplicitCopy),
                };
            }),
            new Dictionary<MemoryLocation, TargetMemorySpaceId>
            {
                [MemoryLocation.BlockLocalData] = localSpaces[^1].Id,
            });
    }
}
