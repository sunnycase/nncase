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
    public long MaxSize => IsObjectBuffer(Id.Node.Grid.Buffers[Id.Index])
        ? 0
        : TensorUtilities.GetProduct(Id.Node.BufferShapes[Id.Index].ToArray()) * Id.Node.GetBufferElemSize(Id.Index);

    private static bool IsObjectBuffer(Expr expr) => expr.CheckedDataType is ReferenceType;
}

public record NodeWithBufferInfo(long Size, Tuple<int, int> Liveness, long[] Shape, long[] Strides)
{
    public ulong Offset { get; set; } = ulong.MaxValue;
}

/// <summary>
/// Represents the view information of a buffer.
/// </summary>
/// <param name="Parent">The parent view information.</param>
/// <param name="View">The view expression.</param>
/// <param name="ViewVar">The variable for the view expression. Object buffers are passed through by reference and do not need a view variable.</param>
/// <param name="Buffer">The buffer expression.</param>
/// <param name="GlobalOffsets">The global offsets for the buffer.</param>
/// <param name="LocalOffsets">The local offsets of parent for the buffer.</param>
/// <param name="Shape">The shape of the view.</param>
internal sealed record ViewInfo(ViewInfo? Parent, Expr View, Var? ViewVar, Expr Buffer, RankedShape GlobalOffsets, RankedShape LocalOffsets, RankedShape Shape)
{
}

public sealed class TreeSolveResult : TreeSolverBase<long>, ITreeNodeVisitor<TreeSolveResult.Context, Unit>
{
    private readonly Dictionary<ITileable, Dictionary<BufferIdentity, ViewInfo>> _viewInfoMemo;

    public TreeSolveResult(BufferGraph primBufferGraph, long objectiveValue, Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>> levelNodeBufferInfos, Dictionary<OpNode, OpNodeInfo<long>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<long>> levelBufferInfos, Dictionary<ITileable, DomainInfo<long>> domainInfos, INTTTargetOptions targetOptions, string moduleKind)
        : base(null!, primitiveBufferInfo, levelBufferInfos, domainInfos, targetOptions)
    {
        PrimBufferGraph = primBufferGraph;
        (Inputs, Outputs) = primBufferGraph.GetInputsOutputs(primBufferGraph.Parent as BufferGraph);
        ObjectOutputAliases = ResolveObjectOutputAliases();
        InputOutputVars = new();
        var inOutInputs = ObjectOutputAliases.Values.ToHashSet();
        foreach (var bid in Inputs)
        {
            var expr = bid.Node.Grid.Buffers[bid.Index];
            var bufferType = GetBufferType(expr);
            var isInOut = inOutInputs.Contains(bid);
            InputOutputVars.Add(
                bid,
                new BufferVar(
                    $"{bid}",
                    bufferType,
                    isInOut ? BufferVarRole.InOut : BufferVarRole.Input,
                    MemoryLocation.Input));
        }

        foreach (var bid in Outputs)
        {
            if (ObjectOutputAliases.TryGetValue(bid, out var inputBid))
            {
                InputOutputVars.Add(bid, InputOutputVars[inputBid]);
                continue;
            }

            var expr = bid.Node.Grid.Buffers[bid.Index];
            var bufferType = GetBufferType(expr);
            InputOutputVars.Add(bid, new BufferVar($"{bid}", bufferType, BufferVarRole.Output, MemoryLocation.Output));
        }

        ObjectiveValue = objectiveValue;
        LevelNodeBufferInfos = levelNodeBufferInfos;
        ModuleKind = moduleKind;
        _viewInfoMemo = new();
    }

    public BufferGraph PrimBufferGraph { get; }

    public HashSet<BufferIdentity> Inputs { get; }

    public HashSet<BufferIdentity> Outputs { get; }

    public IReadOnlyDictionary<BufferIdentity, BufferIdentity> ObjectOutputAliases { get; }

    public Dictionary<BufferIdentity, IVar> InputOutputVars { get; }

    public long ObjectiveValue { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>> LevelNodeBufferInfos { get; }

    public string ModuleKind { get; }

    public RankedShape PartialShapeFromDomain(Isl.set parentDomain, DomainRelation domainRel, Isl.set tiledDomain, AffineMap access, uint dim, Dictionary<string, Dimension> paramDimMap)
    {
        var domainRank = tiledDomain.dim(Isl.dim_type.set);
        var shapeRank = access.Results.Length;

        var (domainRelMinMpa, domainRelMaxMpa) = TilingUtilities.ToMinMaxMpa(domainRel.Map);
        var parentMaxMpa = parentDomain.max_multi_pw_aff();
        var parentMinMpa = parentDomain.min_multi_pw_aff();
        var currentMaxMpa = domainRelMaxMpa.pullback(parentMaxMpa);
        var currentMinMpa = domainRelMinMpa.pullback(parentMinMpa);
        var tiledMaxMpa = tiledDomain.max_multi_pw_aff();
        var tiledMinMpa = tiledDomain.min_multi_pw_aff();
        var (accessMinMpa, accessMaxMpa) = TilingUtilities.ToMinMaxMpa(access);

        for (int i = (int)dim; i < domainRank; i++)
        {
            tiledMaxMpa = tiledMaxMpa.set_at(i, currentMaxMpa.at(i));
            tiledMinMpa = tiledMinMpa.set_at(i, currentMinMpa.at(i));
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
        var (parentbuilder, parentOffsets, parentExtents) = context;
        {
            var newParentExtents = new Dimension[parentExtents.Length];
            for (int i = 0; i < parentExtents.Length; i++)
            {
                if (parentExtents[i] is AsDim { Dim: Call { Target: IR.Tensors.LocalShardDim } } localShardDim)
                {
                    var letDim = T.LetDim(out var dimVar, localShardDim, $"L{value.Level}_d{i}");
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

            tilemap = new Isl.map(Isl.ctx.Current, $"{{ [{string.Join(',', dims)}] -> [{string.Join(',', outerDims)},{string.Join(',', innerDims)}] : {string.Join(" and ", constraints)} }}");
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
            loopBuilders[i] = T.Serial(out var loopVar, (0L, stop, stride), $"d{i}_Op{value.OpId}_L{value.Level}");
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
        for (int i = 0; i < loopVars.Length + 1; i++)
        {
            var offsets = forwardOffsets[i] = currentOffsets.ToArray();

            for (int j = 0; j < i; j++)
            {
                offsets[j] += loopVars[j];
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
            foreach (var (bid, bufferInfo) in OrderBufferInfosForViewCreation(nodeMemo.BufferInfoMap))
            {
                var place = bufferInfo.Places[ci];
                var expr = bid.Node.Grid.Buffers[bid.Index];
                for (int sl = 0; sl < place.Length; sl++)
                {
                    if (place[sl] != 1)
                    {
                        continue;
                    }

                    var kernelInfo = bid.Node.GetKernelInfo(TargetOptions);
                    var partialShape = PartialShapeFromDomain(parentDomain, value.DomainRelation, tiledChildDomain, bufferInfo.Map, (uint)ci, paramDimMap);

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
                        if (!IsObjectBuffer(expr) && !TargetOptions.UnifiedMemoryArch && viewInfo.Parent is ViewInfo parentViewInfo)
                        {
                            if (kernelInfo.BufferInfos[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Read))
                            {
                                localBuilder.Body(T.Memcopy(viewInfo.ViewVar!, IR.F.Buffer.BufferSubview(parentViewInfo.ViewVar ?? parentViewInfo.Buffer, viewInfo.LocalOffsets, viewInfo.Shape)));
                            }

                            if (kernelInfo.BufferInfos[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Write))
                            {
                                localBuilder.Tail(T.Memcopy(IR.F.Buffer.BufferSubview(parentViewInfo.ViewVar ?? parentViewInfo.Buffer, viewInfo.LocalOffsets, viewInfo.Shape), viewInfo.ViewVar!));
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
                cntBuilder.Body(loopBuilders[ci]);
                cntBuilder = loopBuilders[ci];
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
        var kernelInfo = value.GetKernelInfo(TargetOptions);

        var bufferParentViewInfos = new ViewInfo[value.BufferShapes.Length];
        var bufferViews = new Expr[value.BufferShapes.Length];
        for (int i = 0; i < value.BufferShapes.Length; i++)
        {
            var bid = new BufferIdentity(value.Wrapped, i);
            var shape = PartialShapeFromDomain(parentDomain, value.DomainRelation, currentDomain, value.Grid.AccessMaps[i], (uint)currentDomain.dim(Isl.dim_type.set), paramDimMap);
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
            bufferParentViewInfos[i] = parentViewInfo;
            bufferViews[i] = IsObjectBuffer(value.Grid.Buffers[i])
                ? parentViewInfo.ViewVar ?? parentViewInfo.Buffer
                : IR.F.Buffer.BufferSubview(parentViewInfo.ViewVar!, offsets, shape);
        }

        var bodyVarReplaces = new Dictionary<BaseExpr, BaseExpr>();
        for (int i = 0; i < value.Grid.BodyParameters.Length; i++)
        {
            bodyVarReplaces.Add(value.Grid.BodyParameters[i], bufferViews[i]);
        }

        var domain = new IR.Tuple(currentOffsets.Select(off => new IR.Tuple(IR.F.Shapes.AsTensor(off), (Expr)0L)).ToArray());
        bodyVarReplaces.Add(value.Grid.DomainParameter, domain);
        var nestBody = new ReplacingExprCloner(bodyVarReplaces).Clone(value.Grid.Body, default);
        parentbuilder.Body(nestBody);
        return default;
    }

    public long ScheduleBuffers()
    {
        var maxAlign = 0L;
        foreach (var (level, nodeBufferInfos) in LevelNodeBufferInfos)
        {
            var model = new CpModel();
            var rectangles = new Dictionary<NodeWithBuffer, (IntervalVar XInterval, IntervalVar YInterval)>();
            int count = 0;
            var cons = model.AddNoOverlap2D();
            foreach (var (key, info) in nodeBufferInfos)
            {
                if (info.Size > 0)
                {
                    var x = model.NewFixedSizeIntervalVar(info.Liveness.Item1, info.Liveness.Item2 - info.Liveness.Item1, $"x{count}");
                    var ystart = model.NewIntVar(0, TargetOptions.MemoryCapacities[level] - info.Size, $"ystart{count}");
                    var align = key.Id.Node.GetBufferElemSize(key.Id.Index);
                    if (ModuleKind == "xpu")
                    {
                        align = 128;
                    }

                    maxAlign = Math.Max(maxAlign, align);
                    model.AddModuloEquality(0, ystart, align);
                    var y = model.NewFixedSizeIntervalVar(ystart, info.Size, $"y{count}");
                    cons.AddRectangle(x, y);
                    rectangles.Add(key, (x, y));
                    count++;
                }
            }

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
        }

        return maxAlign;
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
            var pbid = TileNodeMemo[parentTileNode].GetByChildBuffer(cbid);
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
        if (IsObjectBuffer(bid.Node.Grid.Buffers[bid.Index]))
        {
            return GetObjectViewInfo(node, bid, map, forwardOffsets, shape);
        }

        TIR.Buffer AllocateBuffer(TileNode tileNode, BufferIdentity bid)
        {
            var expr = bid.Node.Grid.Buffers[bid.Index];
            var distributedType = GetBufferType(expr) as DistributedType;
            var tensorType = GetBufferTensorType(expr);
            tensorType = new TensorType(tensorType.DType, shape); // according to subtensor shape.
            var info = LevelNodeBufferInfos[storeLevel][new NodeWithBuffer(tileNode, bid)];
            var alignment = tensorType.DType.SizeInBytes;
            var strides = info.Strides.Select(i => (Dimension)i).ToArray(); // using fixed strides.
            var physicalBuffer = new PhysicalBuffer(alignment, Tensor.FromPointer(info.Offset, tensorType.DType), info.Size, MemoryLocation.Cache, storeLevel);
            return new TIR.Buffer($"{bid}", tensorType.DType, new MemSpan(physicalBuffer), shape.Dimensions.ToArray(), strides, distributedType);
        }

        Expr GetViewExpr(ViewInfo? parentInfo, Expr buffer, RankedShape forwardOffsets, RankedShape relatedOffsets, RankedShape shape)
        {
            return parentInfo switch
            {
                null => buffer switch
                {
                    TIR.Buffer buf => IR.F.Buffer.AllocateBufferView(buf),
                    Expr ivar when ivar is IVar => TargetOptions.UnifiedMemoryArch switch
                    {
                        true => IR.F.Buffer.BufferSubview(ivar, relatedOffsets, shape),
                        false => ivar,
                    },
                    _ => throw new NotSupportedException(),
                },
                ViewInfo info => TargetOptions.UnifiedMemoryArch switch
                {
                    true => IR.F.Buffer.BufferSubview(info.ViewVar!, relatedOffsets, shape),
                    false => buffer switch
                    {
                        TIR.Buffer buf => IR.F.Buffer.AllocateBufferView(buf),
                        _ => throw new NotSupportedException(),
                    },
                },
            };
        }

        var bufferOffsets = map.Apply(forwardOffsets, Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
        var offsets = new RankedShape(bufferOffsets);
        if (TryGetParentViewInfo(node, bid, out var parentViewInfo))
        {
            var viewOffset = new Dimension[bufferOffsets.Length];
            for (int j = 0; j < viewOffset.Length; j++)
            {
                var x = bufferOffsets[j] - parentViewInfo.GlobalOffsets[j];
                viewOffset[j] = x;
            }

            offsets = ISLUtility.RoundTrip(viewOffset);
            Expr buffer;
            if (TargetOptions.UnifiedMemoryArch)
            {
                buffer = parentViewInfo.Buffer;
            }
            else
            {
                buffer = AllocateBuffer(node, bid);
            }

            var view = GetViewExpr(parentViewInfo, buffer, bufferOffsets, offsets, shape);
            var viewVar = new Var($"{bid}_L{node.Level}", AnyType.Default);
            return new ViewInfo(parentViewInfo, view, viewVar, buffer, bufferOffsets, offsets, shape);
        }
        else
        {
            parentViewInfo = null;
            Expr buffer = null!;
            var fromExternal = Inputs.Contains(bid) || Outputs.Contains(bid);

            if (TargetOptions.UnifiedMemoryArch && fromExternal)
            {
                buffer = (Expr)InputOutputVars[bid];
            }
            else
            {
                buffer = AllocateBuffer(node, bid);
            }

            if (!TargetOptions.UnifiedMemoryArch && fromExternal)
            {
                parentViewInfo = new ViewInfo(null, null!, null!, (Expr)InputOutputVars[bid], new RankedShape(bufferOffsets.Select(i => 0).ToArray()), new RankedShape(bufferOffsets.Select(i => 0).ToArray()), shape);
            }

            var view = GetViewExpr(null, buffer, bufferOffsets, offsets, shape);
            var viewVar = new Var($"{bid}_L{node.Level}", AnyType.Default);
            return new ViewInfo(parentViewInfo, view, viewVar, buffer, bufferOffsets, fromExternal ? bufferOffsets : new RankedShape(bufferOffsets.Select(i => 0).ToArray()), shape);
        }
    }

    private ViewInfo GetObjectViewInfo(TileNode node, BufferIdentity bid, AffineMap map, Dimension[] forwardOffsets, RankedShape shape)
    {
        if (map.Results.Length != 0 || shape.Rank != 0)
        {
            throw new InvalidOperationException($"Object buffer {bid} must be passed by reference without subview. Access rank: {map.Results.Length}, shape rank: {shape.Rank}.");
        }

        var bufferOffsets = map.Apply(forwardOffsets, Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
        var offsets = new RankedShape(bufferOffsets.Select(_ => (Dimension)0L).ToArray());
        var fromExternal = Inputs.Contains(bid) || Outputs.Contains(bid);
        ViewInfo? parentViewInfo = null;
        Expr buffer;

        if (fromExternal)
        {
            buffer = (Expr)InputOutputVars[bid];
        }
        else if (TryGetObjectAliasSource(node, bid, out var aliasSource))
        {
            buffer = aliasSource;
        }
        else if (TryGetParentViewInfo(node, bid, out parentViewInfo))
        {
            buffer = parentViewInfo.ViewVar ?? parentViewInfo.Buffer;
        }
        else
        {
            throw new InvalidOperationException($"can't find object alias source for {bid} at TileNode {node}.");
        }

        return new ViewInfo(parentViewInfo, buffer, null, buffer, bufferOffsets, offsets, shape);
    }

    private bool TryGetObjectAliasSource(TileNode node, BufferIdentity bid, [MaybeNullWhen(false)] out Expr source)
    {
        source = null;
        if (!bid.IsOutput)
        {
            return false;
        }

        var argument = bid.Node.Grid.GetArgument(bid.Index);
        for (int i = 0; i < bid.Node.Grid.Reads.Length; i++)
        {
            if (!ReferenceEquals(bid.Node.Grid.Reads[i], argument))
            {
                continue;
            }

            var readBid = new BufferIdentity(bid.Node, i);
            if (TryGetCurrentOrParentViewInfo(node, readBid, out var readViewInfo))
            {
                source = readViewInfo.ViewVar ?? readViewInfo.Buffer;
                return true;
            }
        }

        return false;
    }

    private Dictionary<BufferIdentity, BufferIdentity> ResolveObjectOutputAliases()
    {
        var aliases = new Dictionary<BufferIdentity, BufferIdentity>();
        var producerByRead = new Dictionary<BufferIdentity, BufferIdentity>();
        foreach (var edge in PrimBufferGraph.Edges.Where(edge => edge.Tag == BufferEdgeKind.Inter))
        {
            if (!IsObjectBuffer(edge.Target.Node.Grid.Buffers[edge.Target.Index]))
            {
                continue;
            }

            if (!producerByRead.TryAdd(edge.Target, edge.Source))
            {
                throw new InvalidOperationException($"Object SSA read {edge.Target} has multiple producers.");
            }
        }

        foreach (var output in Outputs)
        {
            if (!IsObjectBuffer(output.Node.Grid.Buffers[output.Index]))
            {
                continue;
            }

            if (TryResolveObjectAliasInput(output, producerByRead, new HashSet<BufferIdentity>(), out var input))
            {
                aliases.Add(output, input);
            }
        }

        return aliases;
    }

    private bool TryResolveObjectAliasInput(
        BufferIdentity bid,
        IReadOnlyDictionary<BufferIdentity, BufferIdentity> producerByRead,
        HashSet<BufferIdentity> visited,
        [MaybeNullWhen(false)] out BufferIdentity input)
    {
        input = null;
        if (!visited.Add(bid))
        {
            throw new InvalidOperationException($"Object alias cycle detected at {bid}.");
        }

        if (Inputs.Contains(bid))
        {
            input = bid;
            return true;
        }

        if (!bid.IsOutput || !IsObjectBuffer(bid.Node.Grid.Buffers[bid.Index]))
        {
            return false;
        }

        var outputArgument = bid.Node.Grid.GetArgument(bid.Index);
        var aliasReads = Enumerable.Range(0, bid.Node.Grid.Reads.Length)
            .Where(index => ReferenceEquals(bid.Node.Grid.Reads[index], outputArgument))
            .Select(index => new BufferIdentity(bid.Node, index))
            .ToArray();
        if (aliasReads.Length == 0)
        {
            return false;
        }

        BufferIdentity? resolvedInput = null;
        foreach (var read in aliasReads)
        {
            if (!TryResolveObjectReadInput(read, producerByRead, new HashSet<BufferIdentity>(visited), out var candidate))
            {
                return false;
            }

            if (resolvedInput is not null && resolvedInput != candidate)
            {
                throw new InvalidOperationException($"Object output {bid} aliases multiple function inputs: {resolvedInput} and {candidate}.");
            }

            resolvedInput = candidate;
        }

        input = resolvedInput;
        return input is not null;
    }

    private bool TryResolveObjectReadInput(
        BufferIdentity read,
        IReadOnlyDictionary<BufferIdentity, BufferIdentity> producerByRead,
        HashSet<BufferIdentity> visited,
        [MaybeNullWhen(false)] out BufferIdentity input)
    {
        if (Inputs.Contains(read))
        {
            input = read;
            return true;
        }

        if (!producerByRead.TryGetValue(read, out var producer))
        {
            input = null;
            return false;
        }

        return TryResolveObjectAliasInput(producer, producerByRead, visited, out input);
    }

    private bool TryGetCurrentOrParentViewInfo(TileNode node, BufferIdentity bid, [MaybeNullWhen(false)] out ViewInfo viewInfo)
    {
        if (_viewInfoMemo.TryGetValue(node, out var viewMap) && viewMap.TryGetValue(bid, out viewInfo))
        {
            return true;
        }

        return TryGetParentViewInfo(node, bid, out viewInfo);
    }

    private static IEnumerable<KeyValuePair<BufferIdentity, TileNodeBufferInfo<long>>> OrderBufferInfosForViewCreation(Dictionary<BufferIdentity, TileNodeBufferInfo<long>> bufferInfoMap)
    {
        return bufferInfoMap
            .OrderBy(pair => pair.Key.IsOutput ? 1 : 0)
            .ThenBy(pair => pair.Key.Node.OpId)
            .ThenBy(pair => pair.Key.Index);
    }

    private static bool IsObjectBuffer(Expr expr) => expr.CheckedDataType is ReferenceType;

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
}
