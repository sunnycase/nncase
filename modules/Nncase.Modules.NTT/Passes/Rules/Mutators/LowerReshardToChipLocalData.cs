// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Mutators;

/// <summary>
/// Lowers copy-like intra-chip reshard into aliases over a shared chip-local mutable buffer.
/// </summary>
public sealed class LowerReshardToChipLocalData : ExprRewriter
{
    protected override BaseExpr RewriteLeafPrimFunction(PrimFunction expr)
    {
        var reshardCalls = new HashSet<Call>(
            ExprCollector.Collect(expr.Body)
                .OfType<Call>()
                .Where(call => call.Target is TIR.NTT.GatherReduceScatter gatherReduceScatter &&
                    TryGetCopyLikeReshard(call, gatherReduceScatter, out _, out _)),
            ReferenceEqualityComparer.Instance);
        if (reshardCalls.Count == 0)
        {
            return expr;
        }

        var physicalBufferSets = new PhysicalBufferUnion();
        foreach (var call in reshardCalls)
        {
            var gatherReduceScatter = (TIR.NTT.GatherReduceScatter)call.Target;
            if (!TryGetCopyLikeReshard(call, gatherReduceScatter, out var input, out var output))
            {
                throw new InvalidOperationException("A previously classified copy-like reshard changed during analysis.");
            }

            physicalBufferSets.Union(input.MemSpan.Buffer, output.MemSpan.Buffer, gatherReduceScatter.OutType.TensorType);
        }

        var physicalBufferMap = physicalBufferSets.BuildChipLocalBufferMap();
        var withoutReshard = (Sequential)new RemoveCopyLikeReshards(reshardCalls).Rewrite(expr.Body);
        var rewrittenBody = (Sequential)new ChipLocalBufferViewRewriter(physicalBufferMap).Rewrite(withoutReshard);
        return expr.With(body: rewrittenBody);
    }

    private static bool TryGetCopyLikeReshard(
        Call call,
        TIR.NTT.GatherReduceScatter gatherReduceScatter,
        out TIR.Buffer input,
        out TIR.Buffer output)
    {
        input = null!;
        output = null!;
        if (call.Arguments.Length < 2 ||
            call.Arguments[0] is not TIR.Buffer inputBuffer ||
            call.Arguments[1] is not TIR.Buffer outputBuffer)
        {
            return false;
        }

        if (inputBuffer.MemSpan.Buffer.Location is not (MemoryLocation.Data or MemoryLocation.ChipLocalData) ||
            outputBuffer.MemSpan.Buffer.Location is not (MemoryLocation.Data or MemoryLocation.ChipLocalData))
        {
            return false;
        }

        var inType = gatherReduceScatter.InType;
        var outType = gatherReduceScatter.OutType;
        if (HasPartial(inType) ||
            HasPartial(outType) ||
            inType.TensorType != outType.TensorType ||
            inType.Placement != outType.Placement ||
            !IsSplitOrBroadcast(inType) ||
            !IsSplitOrBroadcast(outType))
        {
            return false;
        }

        input = inputBuffer;
        output = outputBuffer;
        return true;
    }

    private static bool HasPartial(DistributedType distributedType)
        => distributedType.Partial is not null || distributedType.AxisPolicies.Any(policy => policy is SBPPartial);

    private static bool IsSplitOrBroadcast(DistributedType distributedType)
        => distributedType.AxisPolicies.All(policy => policy is SBPSplit or SBPBroadCast);

    private sealed class RemoveCopyLikeReshards : ExprRewriter
    {
        private readonly IReadOnlySet<Call> _calls;

        public RemoveCopyLikeReshards(IReadOnlySet<Call> calls)
        {
            _calls = calls;
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
            => _calls.Contains(expr) ? T.Nop() : expr;
    }

    private readonly record struct ComponentInfo(int Alignment, Dimension SizeBytes)
    {
        public static ComponentInfo From(TensorType tensorType)
        {
            var (sizeBytes, _) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(tensorType, null);
            return new ComponentInfo(tensorType.DType.SizeInBytes, sizeBytes);
        }

        public ComponentInfo Merge(ComponentInfo other)
            => new(Math.Max(Alignment, other.Alignment), Dimension.Max(SizeBytes, other.SizeBytes));
    }

    private sealed class PhysicalBufferUnion
    {
        private readonly Dictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer> _parents = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<TIR.PhysicalBuffer, ComponentInfo> _infos = new(ReferenceEqualityComparer.Instance);

        public void Union(TIR.PhysicalBuffer lhs, TIR.PhysicalBuffer rhs, TensorType tensorType)
        {
            Add(lhs, tensorType);
            Add(rhs, tensorType);

            var lhsRoot = Find(lhs);
            var rhsRoot = Find(rhs);
            if (ReferenceEquals(lhsRoot, rhsRoot))
            {
                _infos[lhsRoot] = _infos[lhsRoot].Merge(ComponentInfo.From(tensorType));
                return;
            }

            _parents[rhsRoot] = lhsRoot;
            _infos[lhsRoot] = _infos[lhsRoot].Merge(_infos[rhsRoot]).Merge(ComponentInfo.From(tensorType));
            _infos.Remove(rhsRoot);
        }

        public IReadOnlyDictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer> BuildChipLocalBufferMap()
        {
            var chipLocalBuffers = new Dictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer>(ReferenceEqualityComparer.Instance);
            var result = new Dictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer>(ReferenceEqualityComparer.Instance);
            foreach (var buffer in _parents.Keys)
            {
                var root = Find(buffer);
                if (!chipLocalBuffers.TryGetValue(root, out var chipLocalBuffer))
                {
                    var info = _infos[root];
                    chipLocalBuffer = new TIR.PhysicalBuffer(info.Alignment, info.SizeBytes, MemoryLocation.ChipLocalData);
                    chipLocalBuffers.Add(root, chipLocalBuffer);
                }

                result.Add(buffer, chipLocalBuffer);
            }

            return result;
        }

        private void Add(TIR.PhysicalBuffer buffer, TensorType tensorType)
        {
            if (_parents.ContainsKey(buffer))
            {
                var root = Find(buffer);
                _infos[root] = _infos[root].Merge(ComponentInfo.From(tensorType));
                return;
            }

            _parents.Add(buffer, buffer);
            _infos.Add(buffer, ComponentInfo.From(tensorType));
        }

        private TIR.PhysicalBuffer Find(TIR.PhysicalBuffer buffer)
        {
            var parent = _parents[buffer];
            if (ReferenceEquals(parent, buffer))
            {
                return buffer;
            }

            var root = Find(parent);
            _parents[buffer] = root;
            return root;
        }
    }

    private sealed class ChipLocalBufferViewRewriter : ExprRewriter
    {
        private const string ShardCoordDimPrefix = "__shard_coord_";

        private readonly IReadOnlyDictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer> _physicalBufferMap;
        private readonly Dictionary<TIR.Buffer, TIR.Buffer> _bufferMemo = new(ReferenceEqualityComparer.Instance);

        public ChipLocalBufferViewRewriter(IReadOnlyDictionary<TIR.PhysicalBuffer, TIR.PhysicalBuffer> physicalBufferMap)
        {
            _physicalBufferMap = physicalBufferMap;
        }

        protected override BaseExpr RewriteLeafBuffer(TIR.Buffer expr)
        {
            if (!_physicalBufferMap.TryGetValue(expr.MemSpan.Buffer, out var chipLocalBuffer))
            {
                return expr;
            }

            if (_bufferMemo.TryGetValue(expr, out var rewritten))
            {
                return rewritten;
            }

            rewritten = CreateChipLocalView(expr, chipLocalBuffer);
            _bufferMemo.Add(expr, rewritten);
            return rewritten;
        }

        private static TIR.Buffer CreateChipLocalView(TIR.Buffer source, TIR.PhysicalBuffer chipLocalBuffer)
        {
            if (source.DistributedType is not { } distributedType)
            {
                return source.With(memSpan: source.MemSpan.With(buffer: chipLocalBuffer));
            }

            var tensorType = distributedType.TensorType;
            var (_, globalStrides) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(tensorType, null);
            var shardIndex = Enumerable.Range(0, distributedType.Placement.Rank)
                .Select(axis => (Dimension)new DimVar($"{ShardCoordDimPrefix}{axis}"))
                .ToArray();
            var (localOffset, localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
            var localElementOffset = (Dimension)0;
            for (var axis = 0; axis < localOffset.Length; axis++)
            {
                localElementOffset += localOffset[axis] * globalStrides[axis];
            }

            var spanStart = source.MemSpan.Start + (localElementOffset * source.ElemType.SizeInBytes);
            return source.With(
                memSpan: new MemSpan(chipLocalBuffer, spanStart, chipLocalBuffer.Size),
                dimensions: localShape,
                strides: globalStrides);
        }
    }
}
