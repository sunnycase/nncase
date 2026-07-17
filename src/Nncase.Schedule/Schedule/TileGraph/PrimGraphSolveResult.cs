// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.Utilities;
using static Nncase.TIR.TIRExtensions;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// because a tile node may have multiple buffer, and each buffer may store at different level.
/// so we need a class to represent the buffer identity.
/// </summary>
public record class NodeWithBuffer(TileNode Node, BufferIdentity Id)
{
    public long MaxSize => Id.Access.Buffer.CheckedDataType is ReferenceType
        ? 0
        : TensorUtilities.GetProduct(Id.Node.BufferShapes[Id.Index].ToArray()) * Id.Node.GetBufferElemSize(Id.Index);
}

public record NodeWithBufferInfo(
    long Size,
    TileLifetime Lifetime,
    long[] Shape,
    long[] Strides,
    int AlignmentBytes,
    TargetStorageEncodingSelection? StorageEncoding)
{
    public ulong Offset { get; set; } = ulong.MaxValue;
}

/// <summary>
/// Physical arena usage produced by scheduling one AutoTiling memory space.
/// </summary>
/// <param name="MemorySpace">Target memory-space identity.</param>
/// <param name="Binding">TIR storage binding used by generated buffers.</param>
/// <param name="RequiredBytes">Highest scheduled buffer end before target allocation rounding.</param>
/// <param name="AllocationBytes">Arena size after applying the target allocation policy.</param>
/// <param name="Alignment">Required arena base alignment.</param>
public sealed record TileBufferPoolSchedule(
    TargetMemorySpaceId MemorySpace,
    TIRMemorySpaceBinding Binding,
    long RequiredBytes,
    long AllocationBytes,
    int Alignment);

/// <summary>
/// Complete physical buffer schedule for one tiled PrimFunction.
/// </summary>
public sealed record TileBufferScheduleResult(IReadOnlyList<TileBufferPoolSchedule> Pools);

/// <summary>
/// Hidden caller binding for one root materialization retained between
/// sequential phases of a scheduled region.
/// </summary>
public sealed record TileRootParameterBinding(BufferIdentity Source, BufferVar Parameter, Expr Argument);

/// <summary>
/// Represents the view information of a buffer.
/// </summary>
/// <param name="Parent">The parent view information.</param>
/// <param name="View">The view expression.</param>
/// <param name="ViewVar">The variable for the view expression. Root-bound values may be passed directly and do not need a view variable.</param>
/// <param name="Buffer">The buffer expression.</param>
/// <param name="GlobalOffsets">The global offsets for the buffer.</param>
/// <param name="LocalOffsets">The local offsets of parent for the buffer.</param>
/// <param name="Shape">The shape of the view.</param>
/// <param name="RequiresParentTransfer">Whether this view owns storage that requires an explicit load/store to its parent.</param>
internal sealed record ViewInfo(ViewInfo? Parent, Expr View, Var? ViewVar, Expr Buffer, RankedShape GlobalOffsets, RankedShape LocalOffsets, RankedShape Shape, bool RequiresParentTransfer)
{
    public Expr Value => ViewVar ?? View;
}

public sealed class TreeSolveResult : TreeSolverBase<long>, ITreeNodeVisitor<TreeSolveResult.Context, Unit>
{
    private readonly List<TileRootParameterBinding> _rootParameterBindings;
    private readonly Dictionary<ITileable, Dictionary<BufferIdentity, ViewInfo>> _viewInfoMemo;
    private readonly IReadOnlyDictionary<int, BlockMicroKernelSelection> _microKernelSelections;

    public TreeSolveResult(BufferGraph primBufferGraph, long objectiveValue, Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>> levelNodeBufferInfos, Dictionary<OpNode, OpNodeInfo<long>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<long>> levelBufferInfos, Dictionary<ITileable, DomainInfo<long>> domainInfos, IReadOnlyList<TileMaterialization> materializations, IReadOnlyDictionary<int, BlockMicroKernelSelection> microKernelSelections, INTTTargetOptions targetOptions, string moduleKind)
        : base(null!, primitiveBufferInfo, levelBufferInfos, domainInfos, targetOptions)
    {
        PrimBufferGraph = primBufferGraph;
        (Inputs, Outputs) = primBufferGraph.GetInputsOutputs(primBufferGraph.Parent as BufferGraph);
        OutputStorageBids = ResolveOutputStorageBids();
        InputOutputVars = new();
        OutputValues = new();
        _rootParameterBindings = new();
        var inOutInputs = OutputStorageBids.Values.Where(Inputs.Contains).ToHashSet();
        foreach (var bid in Inputs)
        {
            var expr = bid.Access.Buffer;
            var bufferType = GetBufferType(expr);
            var isInOut = inOutInputs.Contains(bid);
            InputOutputVars.Add(
                bid,
                new BufferVar(
                    TileSemanticNaming.GetBufferEndpointName(bid),
                    bufferType,
                    isInOut ? BufferVarRole.InOut : BufferVarRole.Input,
                    MemoryLocation.Input));
        }

        foreach (var bid in Outputs)
        {
            var storageBid = OutputStorageBids[bid];
            if (!InputOutputVars.TryGetValue(storageBid, out var storageVar))
            {
                var storageType = GetBufferType(storageBid.Access.Buffer);
                storageVar = new BufferVar(TileSemanticNaming.GetBufferEndpointName(storageBid), storageType, BufferVarRole.Output, MemoryLocation.Output);
                InputOutputVars.Add(storageBid, storageVar);
            }

            InputOutputVars.TryAdd(bid, storageVar);
            OutputValues.Add(bid, CreateLogicalResultValue(bid, storageBid, storageVar));
        }

        BindRootMaterializations();

        ObjectiveValue = objectiveValue;
        LevelNodeBufferInfos = levelNodeBufferInfos;
        Materializations = materializations;
        _microKernelSelections = microKernelSelections;
        ModuleKind = moduleKind;
        _viewInfoMemo = new();
    }

    public BufferGraph PrimBufferGraph { get; }

    public IReadOnlyList<TileMaterialization> Materializations { get; }

    public HashSet<BufferIdentity> Inputs { get; }

    public HashSet<BufferIdentity> Outputs { get; }

    public IReadOnlyDictionary<BufferIdentity, BufferIdentity> OutputStorageBids { get; }

    public Dictionary<BufferIdentity, IVar> InputOutputVars { get; }

    public Dictionary<BufferIdentity, Expr> OutputValues { get; }

    public IReadOnlyList<TileRootParameterBinding> RootParameterBindings => _rootParameterBindings;

    public IReadOnlyList<BufferVar> OutputParameters => Outputs
        .OrderBy(output => output.Node.OpId)
        .ThenBy(output => output.Index)
        .Select(output => OutputStorageBids[output])
        .Where(storage => !Inputs.Contains(storage))
        .Distinct()
        .Select(storage => (BufferVar)InputOutputVars[storage])
        .ToArray();

    public long ObjectiveValue { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>> LevelNodeBufferInfos { get; }

    public string ModuleKind { get; }

    private void BindRootMaterializations()
    {
        foreach (var group in PrimBufferGraph.GetOwnedRootMaterializationEdges()
                     .GroupBy(edge => edge.Source)
                     .OrderBy(group => group.Key.Node.RegionOpId)
                     .ThenBy(group => group.Key.OutputIndex))
        {
            var source = group.Key;
            IVar? storageVar = TryResolveExistingStorageVar(source);
            foreach (var edge in group)
            {
                var targetStorage = TryResolveExistingStorageVar(edge.Target);
                if (storageVar is not null && targetStorage is not null && !ReferenceEquals(storageVar, targetStorage))
                {
                    throw new InvalidOperationException(
                        $"Root materialization {source} resolves to conflicting caller buffers {storageVar} and {targetStorage}.");
                }

                storageVar ??= targetStorage;
            }

            if (storageVar is null)
            {
                var sourceType = GetBufferType(source.Access.Buffer);
                var location = GetCallerBufferLocation(source.Access.Buffer);
                var parameter = new BufferVar(
                    $"{TileSemanticNaming.GetBufferEndpointName(source)}_root",
                    sourceType,
                    BufferVarRole.InOut,
                    location);
                storageVar = parameter;
                _rootParameterBindings.Add(new TileRootParameterBinding(source, parameter, source.Access.Buffer));
            }

            BindEndpoint(source, storageVar);
            foreach (var edge in group)
            {
                var sourceType = GetBufferType(source.Access.Buffer);
                var targetType = GetBufferType(edge.Target.Access.Buffer);
                if (!Equals(sourceType, targetType))
                {
                    throw new InvalidOperationException(
                        $"Root materialization {source} -> {edge.Target} changes buffer type from {sourceType} to {targetType}. " +
                        "Logical view transformations must remain explicit grid operations.");
                }

                BindEndpoint(edge.Target, storageVar);
            }
        }

        IVar? TryResolveExistingStorageVar(BufferIdentity bid)
        {
            if (InputOutputVars.TryGetValue(bid, out var direct))
            {
                return direct;
            }

            return TryGetAliasReadBid(bid, out var aliasRead) && InputOutputVars.TryGetValue(aliasRead, out var aliasStorage)
                ? aliasStorage
                : null;
        }

        void BindEndpoint(BufferIdentity bid, IVar storageVar)
        {
            if (InputOutputVars.TryGetValue(bid, out var existing) && !ReferenceEquals(existing, storageVar))
            {
                throw new InvalidOperationException(
                    $"Root materialization endpoint {bid} is already bound to a different buffer {existing}.");
            }

            InputOutputVars[bid] = storageVar;
        }
    }

    private static MemoryLocation GetCallerBufferLocation(Expr buffer)
        => buffer switch
        {
            Call { Target: IR.Buffers.Uninitialized uninitialized } => uninitialized.MemoryLocation,
            TIR.Buffer tirBuffer => tirBuffer.MemSpan.Buffer.Location,
            BufferVar bufferVar => bufferVar.Location,
            _ => throw new InvalidOperationException(
                $"A root materialization requires caller-allocated storage, got {buffer.GetType().Name}: {buffer}."),
        };

    public RankedShape PartialShapeFromDomain(
        Isl.set parentDomain,
        DomainRelation domainRel,
        Isl.set tiledDomain,
        AffineMap access,
        IReadOnlyList<int> loopOrder,
        int loopEntry,
        Dictionary<string, Dimension> paramDimMap)
    {
        var domainRank = tiledDomain.dim(Isl.dim_type.set);
        var shapeRank = access.Results.Length;
        if (loopOrder.Count != domainRank || loopEntry < 0 || loopEntry > loopOrder.Count)
        {
            throw new ArgumentException(
                $"Partial buffer shape requires a rank-{domainRank} loop order and an entry in [0, {domainRank}], " +
                $"got order [{string.Join(", ", loopOrder)}] and entry {loopEntry}.");
        }

        var enteredAxes = new bool[domainRank];
        for (int position = 0; position < loopEntry; position++)
        {
            enteredAxes[loopOrder[position]] = true;
        }

        var (domainRelMinMpa, domainRelMaxMpa) = TilingUtilities.ToMinMaxMpa(domainRel.Map);
        var parentMaxMpa = parentDomain.max_multi_pw_aff();
        var parentMinMpa = parentDomain.min_multi_pw_aff();
        var currentMaxMpa = domainRelMaxMpa.pullback(parentMaxMpa);
        var currentMinMpa = domainRelMinMpa.pullback(parentMinMpa);
        var tiledMaxMpa = tiledDomain.max_multi_pw_aff();
        var tiledMinMpa = tiledDomain.min_multi_pw_aff();
        var (accessMinMpa, accessMaxMpa) = TilingUtilities.ToMinMaxMpa(access);

        for (int axis = 0; axis < domainRank; axis++)
        {
            if (!enteredAxes[axis])
            {
                tiledMaxMpa = tiledMaxMpa.set_at(axis, currentMaxMpa.at(axis));
                tiledMinMpa = tiledMinMpa.set_at(axis, currentMinMpa.at(axis));
            }
        }

        var bufferMaxMpa = accessMaxMpa.pullback(tiledMaxMpa.add_constant(1));
        var bufferMinMpa = accessMinMpa.pullback(tiledMinMpa);
        var bufferShapeMpa = bufferMaxMpa.sub(bufferMinMpa);
        var dimensions = new Dimension[shapeRank];
        for (int i = 0; i < bufferShapeMpa.size(); i++)
        {
            var accessMax = accessMaxMpa.at(i);
            if (accessMax.is_cst() && accessMax.max_val().num_si() == 0)
            {
                dimensions[i] = 1;
            }
            else
            {
                var pa = bufferShapeMpa.at(i);
                dimensions[i] = ISLUtility.ToDimension(pa, paramDimMap);
            }
        }

        return new RankedShape(dimensions);
    }

    public Unit Visit(TileNode value, Context context)
    {
        if (value.ScopeKind == TileScopeKind.Sequential)
        {
            return VisitSequentialScope(value, context);
        }

        if (value.Wrapped.IsPureBufferViewScope())
        {
            var selectedPlacements = TileNodeMemo[value].BufferInfoMap
                .SelectMany(pair => pair.Value.Places.SelectMany(
                    (places, loopEntry) => places.Select(
                        (selected, storageLevel) => (pair.Key, LoopEntry: loopEntry, StorageLevel: storageLevel, Selected: selected))))
                .Where(placement => placement.Selected != 0)
                .ToArray();
            if (selectedPlacements.Length != 0)
            {
                throw new InvalidOperationException(
                    $"Pure buffer-view scope {value} selected physical placements: " +
                    string.Join(", ", selectedPlacements.Select(
                        placement => $"{placement.Key}@entry{placement.LoopEntry}/L{placement.StorageLevel}")));
            }

            // Alias descriptors are created by the nearest executable owner
            // scope. A pure view has no iteration or storage semantics of its
            // own, so lowering a loop nest here would only create empty TIR.
            return default;
        }

        var (parentbuilder, parentOffsets, parentExtents) = context;
        {
            var newParentExtents = new Dimension[parentExtents.Length];
            for (int i = 0; i < parentExtents.Length; i++)
            {
                if (parentExtents[i] is AsDim { Dim: Call { Target: IR.Tensors.LocalShardDim } } localShardDim)
                {
                    var letDim = T.LetDim(
                        out var dimVar,
                        localShardDim,
                        TileSemanticNaming.GetLocalExtentName(value, i, TargetOptions.TargetMachineModel));
                    parentbuilder.Body(letDim);
                    parentbuilder = letDim;
                    dimVar.Metadata = new() { Range = localShardDim.Metadata.Range };
                    newParentExtents[i] = dimVar;
                }
                else
                {
                    newParentExtents[i] = parentExtents[i];
                }
            }

            parentExtents = newParentExtents;
        }

        // get current tile node's domain.
        // todo use domain map to introduce the new dimensions.
        var parentDomain = ISLUtility.ToParametricDomain(parentExtents, out var paramVarMap);
        var paramDimMap = paramVarMap.Select(p => (p.Key.Name, p.Value)).Concat(parentExtents.Select((d, i) => ($"d{i}", d))).ToDictionary();

        var loopBuilders = new ISequentialBuilder<TIR.For>[value.DomainRelation.Map.Results.Length];
        var loopVars = new DimVar[value.DomainRelation.Map.Results.Length];

        var nodeMemo = TileNodeMemo[value];
        var domainRank = value.DomainRelation.Map.Results.Length;
        var reductionAxes = ReductionAxisAnalysis.GetReductionAxes(value);
        var loopOrder = value.LoopOrder;

        // create tile map from tile vars
        Isl.map tilemap;
        {
            var dims = new List<string>();
            var outerDims = new List<string>();
            var innerDims = new List<string>();
            var constraints = new List<string>();
            for (int i = 0; i < value.DomainRelation.Map.Results.Length; i++)
            {
                var tilesize = nodeMemo.BackWardExtents[0][i] / TileableNodeMemo[value].TileVars[i];
                dims.Add($"d{i}");
                outerDims.Add($"d{i}_out");
                innerDims.Add($"d{i}_in");
                constraints.Add($"d{i}_out = {tilesize} * (d{i} // {tilesize}) and d{i}_in = d{i} - d{i}_out");
            }

            tilemap = constraints.Count == 0
                ? new Isl.map(Isl.ctx.Current, "{ [] -> [] }")
                : new Isl.map(Isl.ctx.Current, $"{{ [{string.Join(',', dims)}] -> [{string.Join(',', outerDims)},{string.Join(',', innerDims)}] : {string.Join(" and ", constraints)} }}");
        }

        var currentDomain = parentDomain.apply(value.DomainRelation.ToMap());
        var currentRanges = value.DomainRelation.Map.Apply(parentOffsets, parentExtents);
        var currentOffsets = currentRanges.Select(r => r.Start).ToArray();
        var currentExtents = currentRanges.Select(r => r.Stop).ToArray();
        var tiledParentDomain = tilemap.intersect_domain(currentDomain).range();
        var tiledChildDomain = tiledParentDomain.move_dims(Isl.dim_type.param, 0, Isl.dim_type.set, 0, (uint)domainRank);
        var childBoundsMpa = tiledChildDomain.max_multi_pw_aff().add_constant(1).sub(tiledChildDomain.min_multi_pw_aff());

        // from inner to outer
        var forwardExtents = new Dimension[domainRank];
        for (int i = value.DomainRelation.Map.Results.Length - 1; i >= 0; i--)
        {
            Dimension start = 0L;
            Dimension stop = currentExtents[i];
            Dimension stride = nodeMemo.BackWardExtents[0][i] / TileableNodeMemo[value].TileVars[i];
            DimVar loopVar;
            var loopName = TileSemanticNaming.GetLoopVariableName(
                value,
                i,
                reductionAxes[i] && value.Level == 0,
                TargetOptions.TargetMachineModel);
            loopBuilders[i] = reductionAxes[i] && value.Level == 0
                ? T.Reduction(out loopVar, (0L, stop, stride), loopName)
                : T.Serial(out loopVar, (0L, stop, stride), loopName);
            loopVar.Metadata.Range = new(0, nodeMemo.BackWardExtents[0][i]);
            loopVars[i] = loopVar;
            paramDimMap.Add($"d{i}_out", loopVar);
            {
                Dimension forwardExtent;
                var boundPa = childBoundsMpa.at(i);
                if (boundPa.is_cst())
                {
                    forwardExtent = boundPa.max_val().num_si();
                }
                else
                {
                    var build = new Isl.ast_build(Isl.ctx.Current);
                    var astExpr = build.expr_from(boundPa);
                    forwardExtent = ISLUtility.ToDimension(astExpr, paramDimMap);
                    forwardExtent.Metadata = new()
                    {
                        Range = new(boundPa.min_val().num_si(), boundPa.max_val().num_si()),
                    };
                }

                // ISL describes the affine-domain span, while the serial loop defines
                // the concrete tile geometry. Keep both constraints: a tile cannot
                // extend beyond its stride or the remaining loop domain.
                var remainingExtent = Dimension.Max(0L, stop - loopVar);
                forwardExtents[i] = Dimension.Min(forwardExtent, stride, remainingExtent);
            }
        }

        // forwardOffsets[0] means partentOffsets, forwardOffsets[i] means partentOffsets[0:i] + loop vars[0:i]
        var forwardOffsets = new Dimension[loopVars.Length + 1][];
        for (int entry = 0; entry < loopVars.Length + 1; entry++)
        {
            var offsets = forwardOffsets[entry] = currentOffsets.ToArray();

            for (int position = 0; position < entry; position++)
            {
                var axis = loopOrder[position];
                offsets[axis] += loopVars[axis];
            }
        }

        // var domainLetBuilders = Enumerable.Range(0, value.DimNames.Length).Select(i => new List<ISequentialBuilder<Expr>>()).ToArray();
        var cntBuilder = parentbuilder;
        var childBuilders = new List<ISequentialBuilder<Expr>>();
        for (int i = 0; i < value.Children.Length; i++)
        {
            var childBuilder = T.Sequential();
            childBuilders.Add(childBuilder);
        }

        for (int ci = 0; ci < loopVars.Length + 1; ci++)
        {
            foreach (var (bid, bufferInfo) in OrderBufferInfosForViewCreation(value, nodeMemo.BufferInfoMap))
            {
                var place = bufferInfo.Places[ci];
                for (int sl = 0; sl < place.Length; sl++)
                {
                    if (place[sl] != 1)
                    {
                        continue;
                    }

                    var partialShape = PartialShapeFromDomain(
                        parentDomain,
                        value.DomainRelation,
                        tiledChildDomain,
                        bufferInfo.Map,
                        loopOrder,
                        ci,
                        paramDimMap);

                    var viewInfo = GetViewInfo(sl, value, bid, bufferInfo.Map, forwardOffsets[ci], partialShape);
                    if (viewInfo.ViewVar is { } viewVar)
                    {
                        var letBuilder = T.Let(viewVar, viewInfo.View);
                        cntBuilder.Body(letBuilder);
                        cntBuilder = letBuilder;
                    }

                    // note when create loop is inner loop, the buffer load store should be instert by children's order.
                    {
                        var localBuilder = ci < loopVars.Length ? cntBuilder : childBuilders[FetchBidOwnerIndex(value, bid)];
                        if (viewInfo.RequiresParentTransfer && viewInfo.Parent is ViewInfo parentViewInfo)
                        {
                            var localEffect = bid.Node.LocalAccessEffects[bid.Index];
                            if (!bid.IsOutput && localEffect.Mode.HasFlag(MemoryAccessMode.Read))
                            {
                                localBuilder.Body(T.TileLoad(viewInfo.ViewVar!, IR.F.Buffer.BufferSubview(parentViewInfo.Value, viewInfo.LocalOffsets, viewInfo.Shape)));
                            }

                            if (bid.IsOutput && localEffect.Mode.HasFlag(MemoryAccessMode.Write))
                            {
                                localBuilder.Tail(T.TileStore(viewInfo.ViewVar!, IR.F.Buffer.BufferSubview(parentViewInfo.Value, viewInfo.LocalOffsets, viewInfo.Shape)));
                            }
                        }
                    }

                    if (!_viewInfoMemo.TryGetValue(value, out var subViewMap))
                    {
                        subViewMap = new();
                        _viewInfoMemo.Add(value, subViewMap);
                    }

                    subViewMap[bid] = viewInfo;
                }
            }

            if (ci < loopVars.Length)
            {
                var axis = loopOrder[ci];
                cntBuilder.Body(loopBuilders[axis]);
                cntBuilder = loopBuilders[axis];
            }
            else
            {
            }
        }

        for (int i = 0; i < value.Children.Length; i++)
        {
            var childBuilder = childBuilders[i];
            value.Children[i].Accept(this, new(childBuilder, forwardOffsets[^1], forwardExtents));
            cntBuilder.Body(childBuilder);
        }

        return default;
    }

    public Unit Visit(OpNode value, Context context)
    {
        var (parentbuilder, parentOffsets, parentExtents) = context;
        var parentDomain = ISLUtility.ToParametricDomain(parentExtents, out var paramVarMap);
        var paramDimMap = paramVarMap.Select(p => (p.Key.Name, p.Value)).Concat(parentExtents.Select((d, i) => ($"d{i}", d))).ToDictionary();

        var currentDomain = parentDomain.apply(value.DomainRelation.ToMap());
        var currentRanges = value.DomainRelation.Map.Apply(parentOffsets, parentExtents);
        var currentOffsets = currentRanges.Select(r => r.Start).ToArray();
        var currentExtents = currentRanges.Select(r => r.Stop).ToArray();
        var bufferViews = new Expr[value.BufferShapes.Length];
        var bodyVarReplaces = new Dictionary<BaseExpr, BaseExpr>();
        var opLoopOrder = Enumerable.Range(0, currentDomain.dim(Isl.dim_type.set)).ToArray();
        for (int i = 0; i < value.BufferShapes.Length; i++)
        {
            var access = value.Grid.Accesses[i];
            var endpoint = access.IsRead ? BufferEndpoint.Input : BufferEndpoint.Output;
            var bid = new BufferIdentity(value.Wrapped, i, endpoint);
            var shape = PartialShapeFromDomain(
                parentDomain,
                value.DomainRelation,
                currentDomain,
                value.AccessMaps[i],
                opLoopOrder,
                opLoopOrder.Length,
                paramDimMap);
            if (!TryGetParentViewInfo(value, bid, out var parentViewInfo))
            {
                throw new InvalidOperationException($"can't find parent view info for {bid} at OpNode {value}!");
            }

            var bufferOffsets = OpNodeMemo[value].Maps[i].Apply(currentOffsets, Enumerable.Repeat<Dimension>(0L, currentOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
            var offsets = new Dimension[bufferOffsets.Length];
            for (int j = 0; j < offsets.Length; j++)
            {
                var x = bufferOffsets[j] - parentViewInfo.GlobalOffsets[j];
                offsets[j] = x;
            }

            offsets = ISLUtility.RoundTrip(offsets);
            bufferViews[i] = access.BindingMode == GridBindingMode.Root
                ? parentViewInfo.Value
                : IR.F.Buffer.BufferSubview(parentViewInfo.Value, offsets, shape);

            bodyVarReplaces.Add(access.Parameter, bufferViews[i]);
        }

        var domain = new IR.Tuple(currentOffsets.Select(off => new IR.Tuple(IR.F.Shapes.AsTensor(off), (Expr)0L)).ToArray());
        bodyVarReplaces.Add(value.Grid.DomainParameter, domain);
        if (value.IsPureBufferView)
        {
            return default;
        }

        var bodyCloner = new ReplacingExprCloner(bodyVarReplaces)
        {
            // Grid parameters are local placeholders, while dimensions captured by
            // the body belong to the enclosing function. Rebuilding untouched
            // leaves would create disconnected DimVar identities.
            CloneUnmutated = false,
        };
        var nestBody = bodyCloner.Clone(value.Grid.Body, default);
        if (_microKernelSelections.TryGetValue(value.Wrapped.RegionOpId, out var microKernel))
        {
            var annotator = new BlockMicroKernelAnnotator(value.Op, microKernel);
            annotator.Visit(nestBody);
            if (annotator.MatchCount != 1)
            {
                throw new InvalidOperationException(
                    $"Expected one semantic call for block microkernel Op{value.Wrapped.RegionOpId} " +
                    $"({value.Op.GetType().Name}), found {annotator.MatchCount}.");
            }
        }

        parentbuilder.Body(nestBody);
        return default;
    }

    public TileBufferScheduleResult ScheduleBuffers()
    {
        var pools = new List<TileBufferPoolSchedule>();
        foreach (var (level, nodeBufferInfos) in LevelNodeBufferInfos.OrderBy(pair => pair.Key))
        {
            var memorySpace = TargetOptions.TargetMachineModel.TilingMemorySpaces[level];
            var binding = memorySpace.TIRBinding
                ?? throw new InvalidOperationException($"Target tiling memory space {memorySpace.Id} has no TIR binding.");
            var model = new CpModel();
            var rectangles = new Dictionary<NodeWithBuffer, (IntervalVar XInterval, IntervalVar YInterval)>();
            var memoryEnds = new List<LinearExpr>();
            int count = 0;
            var cons = model.AddNoOverlap2D();
            var maxAlign = 1L;
            foreach (var (key, info) in nodeBufferInfos)
            {
                if (info.Size > 0 && TileBufferAliasAnalysis.IsPureAliasEndpoint(key.Id))
                {
                    throw new InvalidOperationException(
                        $"Pure buffer alias endpoint {key.Id} at {key.Node} was assigned {info.Size} bytes in {memorySpace.Id}. " +
                        "Alias descriptors must not own physical storage.");
                }

                if (info.Size > 0)
                {
                    var x = model.NewFixedSizeIntervalVar(info.Lifetime.FirstPhase, info.Lifetime.PhaseCount, $"x{count}");
                    var ystart = model.NewIntVar(0, memorySpace.MaxAllocationBytesPerScope - info.Size, $"ystart{count}");
                    var align = info.AlignmentBytes;
                    if (ModuleKind == "xpu")
                    {
                        align = 128;
                    }

                    maxAlign = Math.Max(maxAlign, align);
                    model.AddModuloEquality(0, ystart, align);
                    var y = model.NewFixedSizeIntervalVar(ystart, info.Size, $"y{count}");
                    memoryEnds.Add(y.EndExpr());
                    cons.AddRectangle(x, y);
                    rectangles.Add(key, (x, y));
                    count++;
                }
            }

            if (rectangles.Count == 0)
            {
                pools.Add(new(
                    memorySpace.Id,
                    binding,
                    0,
                    0,
                    TargetOptions.TargetMachineModel.GetMemoryResource(memorySpace).AllocationGranularityBytes));
                continue;
            }

            var memoryPoolEnd = model.NewIntVar(0, memorySpace.MaxAllocationBytesPerScope, $"memory_pool_end_l{level}");
            model.AddMaxEquality(memoryPoolEnd, memoryEnds);
            model.Minimize(memoryPoolEnd);

            var solver = new CpSolver();
            var status = solver.Solve(model);
            if (status is not CpSolverStatus.Optimal)
            {
                throw new InvalidOperationException("can't schedule buffers!");
            }

            foreach (var (k, (_, y)) in rectangles)
            {
                nodeBufferInfos[k].Offset = (ulong)solver.Value(y.StartExpr());
            }

            VerifyPhysicalSchedule(memorySpace.Id, nodeBufferInfos);

            var requiredBytes = solver.Value(memoryPoolEnd);
            var allocationBytes = TargetOptions.TargetMachineModel.GetAllocationSizeBytes(memorySpace, requiredBytes);
            if (allocationBytes > memorySpace.MaxAllocationBytesPerScope)
            {
                throw new InvalidOperationException(
                    $"Scheduled {memorySpace.Id} arena requires {requiredBytes} bytes " +
                    $"({allocationBytes} after {memorySpace.AllocationSizePolicy} rounding), exceeding " +
                    $"the target limit {memorySpace.MaxAllocationBytesPerScope}.");
            }

            var resourceAlignment = TargetOptions.TargetMachineModel.GetMemoryResource(memorySpace).AllocationGranularityBytes;
            pools.Add(new(
                memorySpace.Id,
                binding,
                requiredBytes,
                allocationBytes,
                checked((int)Math.Max(maxAlign, resourceAlignment))));
        }

        return new(pools);
    }

    private Unit VisitSequentialScope(TileNode value, Context context)
    {
        if (context.ForwardOffsets.Length != 0 || context.ForwardExtents.Length != 0)
        {
            throw new InvalidOperationException(
                $"Sequential tile scope {value} received an iteration context. " +
                "Independent chip phases must be lowered as an outermost zero-dimensional scope.");
        }

        var nodeInfo = TileNodeMemo[value];
        if (nodeInfo.BufferInfoMap.Count != 0 ||
            nodeInfo.BackWardExtents.Length != 1 ||
            nodeInfo.BackWardExtents[0].Length != 0)
        {
            throw new InvalidOperationException(
                $"Sequential tile scope {value} must not own tile extents or physical buffer placements.");
        }

        foreach (var child in value.Children)
        {
            if (child is not TileNode phase ||
                phase.ScopeKind != TileScopeKind.Iteration ||
                phase.Level != value.Level)
            {
                throw new InvalidOperationException(
                    $"Sequential tile scope {value} requires independent iteration children at L{value.Level}, got {child}.");
            }

            var childBuilder = T.Sequential();
            var phaseOffsets = Enumerable.Repeat<Dimension>(0L, phase.DomainBoundExprs.Length).ToArray();
            ((ITreeNode)phase).Accept(this, new(childBuilder, phaseOffsets, phase.DomainBoundExprs.ToArray()));
            context.ParentBuilder.Body(childBuilder);
        }

        return default;
    }

    private static void VerifyPhysicalSchedule(
        TargetMemorySpaceId memorySpace,
        IReadOnlyDictionary<NodeWithBuffer, NodeWithBufferInfo> nodeBufferInfos)
    {
        var allocated = nodeBufferInfos
            .Where(pair => pair.Value.Size > 0)
            .OrderBy(pair => pair.Value.Offset)
            .ToArray();
        for (int i = 0; i < allocated.Length; i++)
        {
            var (leftBuffer, leftInfo) = allocated[i];
            if (leftInfo.Offset == ulong.MaxValue)
            {
                throw new InvalidOperationException(
                    $"Tile buffer {leftBuffer} has no scheduled offset in {memorySpace}.");
            }

            var leftEnd = checked(leftInfo.Offset + (ulong)leftInfo.Size);
            for (int j = i + 1; j < allocated.Length; j++)
            {
                var (rightBuffer, rightInfo) = allocated[j];
                if (rightInfo.Offset >= leftEnd)
                {
                    break;
                }

                if (leftInfo.Lifetime.Overlaps(rightInfo.Lifetime))
                {
                    var rightEnd = checked(rightInfo.Offset + (ulong)rightInfo.Size);
                    throw new InvalidOperationException(
                        $"Tile buffers {leftBuffer} [{leftInfo.Offset}, {leftEnd}) and " +
                        $"{rightBuffer} [{rightInfo.Offset}, {rightEnd}) overlap in {memorySpace} while both are live " +
                        $"during inclusive phases [{leftInfo.Lifetime.FirstPhase}, {leftInfo.Lifetime.LastPhase}] and " +
                        $"[{rightInfo.Lifetime.FirstPhase}, {rightInfo.Lifetime.LastPhase}].");
                }
            }
        }
    }

    private Expr CreateLogicalResultValue(BufferIdentity resultBid, BufferIdentity storageBid, IVar storageVar)
    {
        var resultType = GetBufferType(resultBid.Access.Buffer);
        var storageType = GetBufferType(storageBid.Access.Buffer);
        if (Equals(resultType, storageType))
        {
            return (Expr)storageVar;
        }

        if (storageVar is not BufferVar bufferVar ||
            !TryGetTensorType(storageType, out var storageTensorType, out var storageDistributedType) ||
            resultType is not (TensorType or DistributedType))
        {
            throw new InvalidOperationException(
                $"Logical result {resultBid} aliases storage {storageBid}, but their types are not compatible tensor buffers: result={resultType}, storage={storageType}.");
        }

        if (!BufferViewUtility.TryCreate(storageType, resultType, out var transform))
        {
            throw new InvalidOperationException(
                $"Logical result {resultBid} cannot alias storage {storageBid}: result={resultType}, storage={storageType}.");
        }

        var storageBuffer = T.AttachBuffer(
            bufferVar,
            storageTensorType,
            bufferVar.Location,
            0,
            out _,
            $"{storageBid}_result_storage",
            storageDistributedType);
        return BufferViewUtility.CreateLogicalBufferView(
            storageBuffer,
            resultType,
            transform,
            $"{resultBid}_result_view");

        static bool TryGetTensorType(IRType type, out TensorType tensorType, out DistributedType? distributedType)
        {
            switch (type)
            {
                case DistributedType distributed:
                    tensorType = distributed.TensorType;
                    distributedType = distributed;
                    return true;
                case TensorType tensor:
                    tensorType = tensor;
                    distributedType = null;
                    return true;
                default:
                    tensorType = null!;
                    distributedType = null;
                    return false;
            }
        }
    }

    private TensorType GetBufferTensorType(Expr expr)
    {
        TensorType GetTensorType(IRType type) => type switch
        {
            TensorType t => t,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
            _ => throw new NotSupportedException(),
        };

        return expr switch
        {
            IR.Buffers.BufferOf bufof => GetTensorType(bufof.Input.CheckedType),
            Call { Target: IR.Buffers.Uninitialized } c => GetTensorType(c.CheckedType),
            _ => GetTensorType(expr.CheckedType),
        };
    }

    private static IRType GetBufferType(Expr expr)
        => expr switch
        {
            IR.Buffers.BufferOf bufof => bufof.Input.CheckedType,
            Call { Target: IR.Buffers.Uninitialized } call => call.CheckedType,
            _ => expr.CheckedType,
        };

    private bool TryGetParentViewInfo(ITreeNode node, BufferIdentity bid, [MaybeNullWhen(false)] out ViewInfo parentViewInfo)
    {
        var cbid = bid;
        var parentNode = node.Parent;
        parentViewInfo = null;
        while (parentNode is TileNode parentTileNode && parentTileNode.OpId != -1)
        {
            if (!TileNodeMemo[parentTileNode].TryGetByChildBuffer(cbid, out var pbid))
            {
                return false;
            }

            if (_viewInfoMemo.TryGetValue(parentTileNode, out var viewMap) && viewMap.TryGetValue(pbid, out var viewInfo))
            {
                parentViewInfo = viewInfo;
                return true;
            }

            parentNode = parentTileNode.Parent;
            cbid = pbid;
        }

        return false;
    }

    private ViewInfo GetViewInfo(int storeLevel, TileNode node, BufferIdentity bid, AffineMap map, Dimension[] forwardOffsets, RankedShape shape)
    {
        if (bid.IsOutput && bid.Node.TryGetAliasSourceAccess(bid.Index, out var sourceAccessIndex))
        {
            return GetBufferAliasViewInfo(node, bid, sourceAccessIndex, map, forwardOffsets, shape);
        }

        if (bid.Access.BindingMode == GridBindingMode.Root)
        {
            return GetRootViewInfo(storeLevel, node, bid, map, forwardOffsets, shape);
        }

        var requiresExplicitTransfer =
            bid.Node.LocalAccessEffects[bid.Index].Scope != MemoryAccessScope.Chip &&
            TargetOptions.TargetMachineModel.RequiresExplicitTransfer(storeLevel);

        TIR.Buffer AllocateBuffer(TileNode tileNode, BufferIdentity bid)
        {
            var expr = bid.Access.Buffer;
            var distributedType = GetBufferType(expr) as DistributedType;
            var tensorType = GetBufferTensorType(expr);
            tensorType = new TensorType(tensorType.DType, shape); // according to subtensor shape.
            var info = LevelNodeBufferInfos[storeLevel][new NodeWithBuffer(tileNode, bid)];
            var alignment = info.AlignmentBytes;
            var strides = info.Strides.Select(i => (Dimension)i).ToArray(); // using fixed strides.
            var binding = TargetOptions.TargetMachineModel.TilingMemorySpaces[storeLevel].TIRBinding
                ?? throw new InvalidOperationException($"Target tiling memory level {storeLevel} has no TIR binding.");
            var physicalBuffer = new PhysicalBuffer(alignment, Tensor.FromPointer(info.Offset, tensorType.DType), info.Size, binding.Location, binding.Hierarchy);
            var bufferName = TileSemanticNaming.GetStorageBufferName(bid, tileNode, storeLevel, TargetOptions.TargetMachineModel);
            return new TIR.Buffer(
                bufferName,
                tensorType.DType,
                new MemSpan(physicalBuffer),
                shape.Dimensions.ToArray(),
                strides,
                distributedType,
                info.StorageEncoding);
        }

        Expr GetViewExpr(ViewInfo? parentInfo, Expr buffer, RankedShape forwardOffsets, RankedShape relatedOffsets, RankedShape shape, bool ownsStorage)
        {
            if (ownsStorage)
            {
                return buffer switch
                {
                    TIR.Buffer buf => IR.F.Buffer.AllocateBufferView(buf, forwardOffsets),
                    _ => throw new InvalidOperationException($"Owned AutoTiling storage must be a TIR buffer, got {buffer.GetType().Name}."),
                };
            }

            return parentInfo switch
            {
                null => buffer switch
                {
                    Expr ivar when ivar is IVar => IR.F.Buffer.BufferSubview(ivar, relatedOffsets, shape),
                    _ => throw new InvalidOperationException($"Direct-access AutoTiling storage must be an external buffer view, got {buffer.GetType().Name}."),
                },
                ViewInfo info => IR.F.Buffer.BufferSubview(info.Value, relatedOffsets, shape),
            };
        }

        var bufferOffsets = map.Apply(forwardOffsets, Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
        var offsets = new RankedShape(bufferOffsets);
        if (TryGetCurrentOrParentViewInfo(node, bid, out var parentViewInfo))
        {
            var viewOffset = new Dimension[bufferOffsets.Length];
            for (int j = 0; j < viewOffset.Length; j++)
            {
                var x = bufferOffsets[j] - parentViewInfo.GlobalOffsets[j];
                viewOffset[j] = x;
            }

            offsets = ISLUtility.RoundTrip(viewOffset);
            var requiresParentTransfer = requiresExplicitTransfer;
            var buffer = requiresParentTransfer ? AllocateBuffer(node, bid) : parentViewInfo.Buffer;
            var view = GetViewExpr(parentViewInfo, buffer, bufferOffsets, offsets, shape, requiresParentTransfer);
            var viewVar = new Var(TileSemanticNaming.GetViewName(bid, node, TargetOptions.TargetMachineModel), AnyType.Default);
            return new ViewInfo(parentViewInfo, view, viewVar, buffer, bufferOffsets, offsets, shape, requiresParentTransfer);
        }
        else
        {
            parentViewInfo = null;
            var fromExternal = InputOutputVars.ContainsKey(bid);
            var requiresParentTransfer = fromExternal && requiresExplicitTransfer;
            Expr buffer = requiresParentTransfer || !fromExternal
                ? AllocateBuffer(node, bid)
                : (Expr)InputOutputVars[bid];
            if (requiresParentTransfer)
            {
                parentViewInfo = new ViewInfo(
                    null,
                    (Expr)InputOutputVars[bid],
                    null,
                    (Expr)InputOutputVars[bid],
                    new RankedShape(bufferOffsets.Select(_ => (Dimension)0L).ToArray()),
                    new RankedShape(bufferOffsets.Select(_ => (Dimension)0L).ToArray()),
                    shape,
                    false);
            }

            var view = GetViewExpr(parentViewInfo, buffer, bufferOffsets, offsets, shape, requiresParentTransfer || !fromExternal);
            var viewVar = new Var(TileSemanticNaming.GetViewName(bid, node, TargetOptions.TargetMachineModel), AnyType.Default);
            return new ViewInfo(
                parentViewInfo,
                view,
                viewVar,
                buffer,
                bufferOffsets,
                fromExternal ? bufferOffsets : new RankedShape(bufferOffsets.Select(_ => (Dimension)0L).ToArray()),
                shape,
                requiresParentTransfer);
        }
    }

    private ViewInfo GetBufferAliasViewInfo(
        TileNode node,
        BufferIdentity resultBid,
        int sourceAccessIndex,
        AffineMap resultMap,
        Dimension[] forwardOffsets,
        RankedShape resultShape)
    {
        var sourceBid = new BufferIdentity(resultBid.Node, sourceAccessIndex, BufferEndpoint.Input);
        if (!TryGetCurrentOrParentViewInfo(node, sourceBid, out var sourceViewInfo))
        {
            throw new InvalidOperationException(
                $"Buffer alias {resultBid} cannot resolve source storage for {sourceBid} at {node}. " +
                DescribeViewResolution(node, sourceBid));
        }

        var sourceStorage = sourceViewInfo.Buffer switch
        {
            TIR.Buffer buffer => buffer,
            BufferVar sourceVar => AttachBuffer(
                sourceVar,
                GetBufferType(sourceBid.Access.Buffer),
                $"{TileSemanticNaming.GetBufferEndpointName(sourceBid)}_alias_source"),
            _ => throw new InvalidOperationException(
                $"Buffer alias source {sourceBid} must resolve to TIR.Buffer or BufferVar, got {sourceViewInfo.Buffer.GetType().Name}."),
        };
        var resultType = GetBufferType(resultBid.Access.Buffer);
        var resultTensorType = resultType switch
        {
            DistributedType distributed => distributed.TensorType,
            TensorType tensor => tensor,
            _ => throw new InvalidOperationException($"Buffer alias result {resultBid} must be tensor-like, got {resultType}."),
        };
        var localResultType = new TensorType(resultTensorType.DType, resultShape);
        var sourceLayout = sourceStorage.With(dimensions: sourceViewInfo.Shape.Dimensions.ToArray());
        var transform = new BufferViewTransform(
            resultBid.Node.Grid.Accesses[sourceAccessIndex].AffineMap,
            resultBid.Access.AffineMap,
            new IRArray<Dimension>(resultBid.Node.Grid.DomainBounds.ToArray()));
        var resultStrides = BufferViewUtility.CreateBufferViewStrides(sourceLayout, localResultType, transform);
        var physicalOffsets = GetPhysicalStorageOffsets(sourceViewInfo);
        if (physicalOffsets.Length != sourceStorage.Rank)
        {
            throw new InvalidOperationException(
                $"Buffer alias source rank mismatch for {resultBid}: offsets={physicalOffsets.Length}, storage={sourceStorage.Rank}.");
        }

        var byteOffset = (TensorUtilities.GetLinearOffset(sourceStorage.Strides, physicalOffsets) * sourceStorage.ElemType.SizeInBytes).Simplify();
        var byteSize = BufferViewUtility.GetByteSpanSize(resultShape.Dimensions, resultStrides, resultTensorType.DType.SizeInBytes);
        var resultOffsets = resultMap.Apply(
            forwardOffsets,
            Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(range => range.Start).ToArray();
        ViewInfo? parentResultView = null;
        var localOffsets = resultOffsets.Select(_ => (Dimension)0L).ToArray();
        if (TryGetParentViewInfo(node, resultBid, out parentResultView))
        {
            localOffsets = resultOffsets.Zip(parentResultView.GlobalOffsets.Dimensions.ToArray())
                .Select(pair => (pair.First - pair.Second).Simplify())
                .ToArray();
        }

        var aliasBuffer = T.CreateBufferView(
            sourceStorage,
            resultTensorType.DType,
            resultShape.Dimensions,
            resultStrides,
            byteOffset,
            byteSize,
            resultType as DistributedType,
            TileSemanticNaming.GetViewName(resultBid, node, TargetOptions.TargetMachineModel, "alias_buffer"));
        var view = IR.F.Buffer.AllocateBufferView(aliasBuffer, new RankedShape(resultOffsets));
        var viewVar = new Var(TileSemanticNaming.GetViewName(resultBid, node, TargetOptions.TargetMachineModel, "alias_view"), AnyType.Default);
        return new ViewInfo(
            parentResultView,
            view,
            viewVar,
            aliasBuffer,
            new RankedShape(resultOffsets),
            new RankedShape(localOffsets),
            resultShape,
            false);

        static TIR.Buffer AttachBuffer(BufferVar sourceVar, IRType sourceType, string name)
        {
            var (tensorType, distributedType) = sourceType switch
            {
                DistributedType distributed => (distributed.TensorType, distributed),
                TensorType tensor => (tensor, null),
                _ => throw new InvalidOperationException($"Buffer alias source must be tensor-like, got {sourceType}."),
            };
            return T.AttachBuffer(
                sourceVar,
                tensorType,
                sourceVar.Location,
                0,
                out _,
                name,
                distributedType);
        }
    }

    private ViewInfo GetRootViewInfo(int storeLevel, TileNode node, BufferIdentity bid, AffineMap map, Dimension[] forwardOffsets, RankedShape shape)
    {
        TIR.Buffer AllocateRootBuffer()
        {
            var expr = bid.Access.Buffer;
            if (expr.CheckedDataType is ReferenceType)
            {
                throw new InvalidOperationException($"Opaque root access {bid} must alias an input resource.");
            }

            var distributedType = GetBufferType(expr) as DistributedType;
            var tensorType = GetBufferTensorType(expr);
            tensorType = new TensorType(tensorType.DType, shape);
            var info = LevelNodeBufferInfos[storeLevel][new NodeWithBuffer(node, bid)];
            var alignment = info.AlignmentBytes;
            var strides = info.Strides.Select(i => (Dimension)i).ToArray();
            var binding = TargetOptions.TargetMachineModel.TilingMemorySpaces[storeLevel].TIRBinding
                ?? throw new InvalidOperationException($"Target tiling memory level {storeLevel} has no TIR binding.");
            var physicalBuffer = new PhysicalBuffer(alignment, Tensor.FromPointer(info.Offset, tensorType.DType), info.Size, binding.Location, binding.Hierarchy);
            var bufferName = TileSemanticNaming.GetStorageBufferName(bid, node, storeLevel, TargetOptions.TargetMachineModel);
            return new TIR.Buffer(
                bufferName,
                tensorType.DType,
                new MemSpan(physicalBuffer),
                shape.Dimensions.ToArray(),
                strides,
                distributedType,
                info.StorageEncoding);
        }

        var bufferOffsets = map.Apply(forwardOffsets, Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
        var offsets = new RankedShape(bufferOffsets.Select(_ => (Dimension)0L).ToArray());
        var fromExternal = InputOutputVars.ContainsKey(bid);
        ViewInfo? parentViewInfo = null;
        Expr buffer;
        Var? viewVar = null;
        Expr view;

        if (fromExternal)
        {
            buffer = (Expr)InputOutputVars[bid];
            view = buffer;
        }
        else if (TryGetAliasSource(node, bid, out var aliasSource))
        {
            buffer = aliasSource;
            view = aliasSource;
        }
        else if (TryGetParentViewInfo(node, bid, out parentViewInfo))
        {
            buffer = parentViewInfo.Value;
            view = buffer;
        }
        else
        {
            buffer = AllocateRootBuffer();
            view = IR.F.Buffer.AllocateBufferView((TIR.Buffer)buffer, new RankedShape(bufferOffsets));
            viewVar = new Var(TileSemanticNaming.GetViewName(bid, node, TargetOptions.TargetMachineModel), AnyType.Default);
        }

        return new ViewInfo(parentViewInfo, view, viewVar, buffer, bufferOffsets, offsets, shape, false);
    }

    private static Dimension[] GetPhysicalStorageOffsets(ViewInfo viewInfo)
    {
        var offsets = viewInfo.LocalOffsets.Dimensions.ToArray();
        if (viewInfo.Parent is not { } parent || !ReferenceEquals(viewInfo.Buffer, parent.Buffer))
        {
            return offsets;
        }

        var parentOffsets = GetPhysicalStorageOffsets(parent);
        if (parentOffsets.Length != offsets.Length)
        {
            throw new InvalidOperationException(
                $"ViewInfo physical offset rank mismatch: parent={parentOffsets.Length}, child={offsets.Length}.");
        }

        return parentOffsets.Zip(offsets).Select(pair => (pair.First + pair.Second).Simplify()).ToArray();
    }

    private bool TryGetAliasSource(TileNode node, BufferIdentity bid, [MaybeNullWhen(false)] out Expr source)
    {
        source = null;
        if (!TryGetAliasReadBid(bid, out var inputBid))
        {
            return false;
        }

        if (TryGetCurrentOrParentViewInfo(node, inputBid, out var readViewInfo))
        {
            source = readViewInfo.Value;
            return true;
        }

        return false;
    }

    private Dictionary<BufferIdentity, BufferIdentity> ResolveOutputStorageBids()
    {
        var storages = new Dictionary<BufferIdentity, BufferIdentity>();
        var producerByRead = new Dictionary<BufferIdentity, BufferIdentity>();
        foreach (var edge in PrimBufferGraph.Edges.Where(edge =>
                     edge.Tag is BufferEdgeKind.Inter or BufferEdgeKind.RootMaterialization))
        {
            if (!producerByRead.TryAdd(edge.Target, edge.Source))
            {
                throw new InvalidOperationException($"SSA read {edge.Target} has multiple producers.");
            }
        }

        foreach (var output in Outputs)
        {
            storages.Add(output, ResolveStorageBid(output, producerByRead, new HashSet<BufferIdentity>()));
        }

        return storages;
    }

    private BufferIdentity ResolveStorageBid(
        BufferIdentity bid,
        IReadOnlyDictionary<BufferIdentity, BufferIdentity> producerByRead,
        HashSet<BufferIdentity> visited)
    {
        if (!visited.Add(bid))
        {
            throw new InvalidOperationException($"Grid access alias cycle detected at {bid}.");
        }

        if (Inputs.Contains(bid))
        {
            return bid;
        }

        if (!TryGetAliasReadBid(bid, out var read))
        {
            return bid;
        }

        if (Inputs.Contains(read))
        {
            return read;
        }

        if (!producerByRead.TryGetValue(read, out var producer))
        {
            throw new InvalidOperationException($"Alias result {bid} has no storage producer for source read {read}.");
        }

        return ResolveStorageBid(producer, producerByRead, visited);
    }

    private static bool TryGetAliasReadBid(BufferIdentity bid, [MaybeNullWhen(false)] out BufferIdentity read)
    {
        read = null;
        if (!bid.IsOutput)
        {
            return false;
        }

        if (bid.Node.TryGetAliasSourceAccess(bid.Index, out var sourceAccessIndex))
        {
            read = new BufferIdentity(bid.Node, sourceAccessIndex, BufferEndpoint.Input);
            return true;
        }

        if (bid.Access.AccessMode == GridAccessMode.ReadWrite)
        {
            read = new BufferIdentity(bid.Node, bid.Index, BufferEndpoint.Input);
            return true;
        }

        return false;
    }

    private bool TryGetCurrentOrParentViewInfo(TileNode node, BufferIdentity bid, [MaybeNullWhen(false)] out ViewInfo viewInfo)
    {
        var storageBid = GetCurrentStorageBid(node, bid);
        if (_viewInfoMemo.TryGetValue(node, out var viewMap) && viewMap.TryGetValue(storageBid, out viewInfo))
        {
            return true;
        }

        return TryGetParentViewInfo(node, storageBid, out viewInfo);
    }

    private BufferIdentity GetCurrentStorageBid(TileNode node, BufferIdentity bid)
        => TileNodeMemo[node].DefUseMap.TryGetByValue(bid, out var producerBid)
            ? producerBid
            : bid;

    private string DescribeViewResolution(TileNode node, BufferIdentity bid)
    {
        var levels = new List<string>();
        ITreeNode? current = node;
        var currentBid = bid;
        while (current is TileNode tileNode && TileNodeMemo.TryGetValue(tileNode, out var nodeInfo))
        {
            var storageBid = nodeInfo.DefUseMap.TryGetByValue(currentBid, out var producerBid)
                ? producerBid
                : currentBid;
            var views = _viewInfoMemo.TryGetValue(tileNode, out var viewMap)
                ? string.Join(",", viewMap.Keys)
                : "<none>";
            levels.Add(
                $"{tileNode}: requested={currentBid}, storage={storageBid}, " +
                $"buffers=[{string.Join(",", nodeInfo.BufferInfoMap.Keys)}], views=[{views}]");
            currentBid = storageBid;
            current = tileNode.Parent;
        }

        if (current is TileNode root)
        {
            levels.Add($"{root}: root sentinel, requested={currentBid}");
        }

        return $"Resolution path: {string.Join("; ", levels)}; ABI=[{string.Join(",", InputOutputVars.Keys)}].";
    }

    private IEnumerable<KeyValuePair<BufferIdentity, TileNodeBufferInfo<long>>> OrderBufferInfosForViewCreation(
        TileNode node,
        Dictionary<BufferIdentity, TileNodeBufferInfo<long>> bufferInfoMap)
    {
        var stableOrder = bufferInfoMap.Keys
            .OrderBy(pair => pair.IsOutput ? 1 : 0)
            .ThenBy(pair => pair.Node.OpId)
            .ThenBy(pair => pair.Index)
            .ToArray();
        var states = new Dictionary<BufferIdentity, int>();
        var result = new List<KeyValuePair<BufferIdentity, TileNodeBufferInfo<long>>>(bufferInfoMap.Count);

        foreach (var bid in stableOrder)
        {
            Visit(bid);
        }

        return result;

        void Visit(BufferIdentity bid)
        {
            if (states.TryGetValue(bid, out var state))
            {
                if (state == 1)
                {
                    throw new InvalidOperationException($"Buffer alias dependency cycle detected at {bid} in {node}.");
                }

                return;
            }

            states.Add(bid, 1);
            if (TileNodeMemo[node].DefUseMap.TryGetByValue(bid, out var definition) &&
                definition != bid &&
                bufferInfoMap.ContainsKey(definition))
            {
                Visit(definition);
            }

            if (TryGetAliasReadBid(bid, out var sourceRead))
            {
                var sourceStorage = GetCurrentStorageBid(node, sourceRead);
                if (sourceStorage != bid && bufferInfoMap.ContainsKey(sourceStorage))
                {
                    Visit(sourceStorage);
                }
            }

            states[bid] = 2;
            result.Add(new(bid, bufferInfoMap[bid]));
        }
    }

    private int FetchBidOwnerIndex(TileNode node, BufferIdentity bid)
    {
        for (int i = 0; i < node.Children.Length; i++)
        {
            var child = node.Children[i];
            if (child is TileNode tilenode)
            {
                if (TileNodeMemo[tilenode].BufferInfoMap.ContainsKey(bid))
                {
                    return i;
                }
            }
            else if (child is OpNode opnode)
            {
                if (bid.Node.OpId == opnode.OpId)
                {
                    return i;
                }
            }
        }

        return -1;
    }

    public sealed record Context(ISequentialBuilder<Expr> ParentBuilder, Dimension[] ForwardOffsets, Dimension[] ForwardExtents)
    {
    }

    private sealed class BlockMicroKernelAnnotator : ExprWalker
    {
        private readonly Op _op;
        private readonly BlockMicroKernelSelection _selection;

        public BlockMicroKernelAnnotator(Op op, BlockMicroKernelSelection selection)
        {
            _op = op;
            _selection = selection;
        }

        public int MatchCount { get; private set; }

        protected override Unit VisitLeafCall(Call expr)
        {
            if (ReferenceEquals(expr.Target, _op))
            {
                expr.Metadata.BlockMicroKernel = _selection;
                MatchCount++;
            }

            return default;
        }
    }
}
