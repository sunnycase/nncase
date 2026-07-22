// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Exact lexical placement of one TileGraph buffer view.
/// </summary>
public readonly record struct TileBufferPlacement(
    TileNode Node,
    BufferIdentity Buffer,
    int LoopEntry,
    int StorageLevel);

/// <summary>
/// Exact source selected for a parent-to-local transfer.
/// </summary>
public abstract record SelectedTileBufferSource;

/// <summary>
/// A source view created by another solved TileGraph placement.
/// </summary>
public sealed record SelectedTileBufferPlacementSource(TileBufferPlacement Placement)
    : SelectedTileBufferSource;

/// <summary>
/// A source bound directly by the caller ABI.
/// </summary>
public sealed record SelectedTileBufferRootSource(BufferIdentity Buffer)
    : SelectedTileBufferSource;

/// <summary>
/// Placement ordering shared by symbolic source selection and TIR construction.
/// </summary>
internal static class TileBufferPlacementUtility
{
    public static BufferIdentity GetCurrentStorageBuffer<T>(TileNode node, BufferIdentity buffer, IReadOnlyDictionary<TileNode, TileNodeInfo<T>> nodeInfos)
        => GetCurrentStorageBuffer(buffer, nodeInfos[node]);

    public static BufferIdentity GetCurrentStorageBuffer<T>(BufferIdentity buffer, TileNodeInfo<T> nodeInfo)
        => nodeInfo.DefUseMap.TryGetByValue(buffer, out var producer)
            ? producer
            : buffer;

    public static ImmutableArray<BufferIdentity> OrderBuffersForViewCreation<T>(
        TileNode node,
        TileNodeInfo<T> nodeInfo)
    {
        var stableOrder = nodeInfo.BufferInfoMap.Keys
            .OrderBy(buffer => buffer.IsOutput ? 1 : 0)
            .ThenBy(buffer => buffer.Node.OpId)
            .ThenBy(buffer => buffer.Index)
            .ToArray();
        var states = new Dictionary<BufferIdentity, int>();
        var result = ImmutableArray.CreateBuilder<BufferIdentity>(nodeInfo.BufferInfoMap.Count);

        foreach (var buffer in stableOrder)
        {
            Visit(buffer);
        }

        return result.MoveToImmutable();

        void Visit(BufferIdentity buffer)
        {
            if (states.TryGetValue(buffer, out var state))
            {
                if (state == 1)
                {
                    throw new InvalidOperationException(
                        $"Buffer alias dependency cycle detected at {buffer} in {node}.");
                }

                return;
            }

            states.Add(buffer, 1);
            if (nodeInfo.DefUseMap.TryGetByValue(buffer, out var definition) &&
                definition != buffer &&
                nodeInfo.BufferInfoMap.ContainsKey(definition))
            {
                Visit(definition);
            }

            if (TryGetAliasReadBuffer(buffer, out var sourceRead))
            {
                var sourceStorage = GetCurrentStorageBuffer(sourceRead, nodeInfo);
                if (sourceStorage != buffer && nodeInfo.BufferInfoMap.ContainsKey(sourceStorage))
                {
                    Visit(sourceStorage);
                }
            }

            states[buffer] = 2;
            result.Add(buffer);
        }
    }

    public static VisibleTileBufferPlacements EnumerateVisiblePlacementsBefore<T>(
        TileBufferPlacement destination,
        IReadOnlyDictionary<TileNode, TileNodeInfo<T>> nodeInfos)
    {
        var placements = ImmutableArray.CreateBuilder<TileBufferPlacement>();
        var currentNode = destination.Node;
        var currentBuffer = GetCurrentStorageBuffer(currentNode, destination.Buffer, nodeInfos);
        AddCurrentNodePlacements(currentNode, currentBuffer, destination, nodeInfos[currentNode], placements);

        for (var parent = currentNode.Parent; parent is TileNode parentNode && parentNode.OpId != -1; parent = parentNode.Parent)
        {
            var parentInfo = nodeInfos[parentNode];
            if (!parentInfo.TryGetByChildBuffer(currentBuffer, out var parentBuffer))
            {
                break;
            }

            AddAllPlacementsInReverseCreationOrder(parentNode, parentBuffer, parentInfo, placements);
            currentBuffer = parentBuffer;
        }

        return new(placements.ToImmutable(), currentBuffer);
    }

    public static bool TryGetAliasReadBuffer(BufferIdentity buffer, out BufferIdentity read)
    {
        read = null!;
        if (!buffer.IsOutput)
        {
            return false;
        }

        if (buffer.Node.TryGetAliasSourceAccess(buffer.Index, out var sourceAccessIndex))
        {
            read = new BufferIdentity(buffer.Node, sourceAccessIndex, BufferEndpoint.Input);
            return true;
        }

        if (buffer.Access.AccessMode == GridAccessMode.ReadWrite)
        {
            read = new BufferIdentity(buffer.Node, buffer.Index, BufferEndpoint.Input);
            return true;
        }

        return false;
    }

    private static void AddCurrentNodePlacements<T>(
        TileNode node,
        BufferIdentity sourceBuffer,
        TileBufferPlacement destination,
        TileNodeInfo<T> nodeInfo,
        ImmutableArray<TileBufferPlacement>.Builder result)
    {
        if (!nodeInfo.BufferInfoMap.TryGetValue(sourceBuffer, out var sourceInfo))
        {
            return;
        }

        var order = OrderBuffersForViewCreation(node, nodeInfo);
        var sourceOrder = order.IndexOf(sourceBuffer);
        var destinationOrder = order.IndexOf(destination.Buffer);
        if (sourceOrder < 0 || destinationOrder < 0)
        {
            throw new InvalidOperationException(
                $"Cannot order source {sourceBuffer} before destination {destination.Buffer} in {node}.");
        }

        for (var loopEntry = destination.LoopEntry; loopEntry >= 0; loopEntry--)
        {
            var maximumStorageLevel = sourceInfo.Places[loopEntry].Length - 1;
            if (loopEntry == destination.LoopEntry)
            {
                maximumStorageLevel = sourceBuffer == destination.Buffer
                    ? destination.StorageLevel - 1
                    : sourceOrder < destinationOrder
                        ? maximumStorageLevel
                        : -1;
            }

            for (var storageLevel = maximumStorageLevel; storageLevel >= 0; storageLevel--)
            {
                result.Add(new(node, sourceBuffer, loopEntry, storageLevel));
            }
        }
    }

    private static void AddAllPlacementsInReverseCreationOrder<T>(
        TileNode node,
        BufferIdentity buffer,
        TileNodeInfo<T> nodeInfo,
        ImmutableArray<TileBufferPlacement>.Builder result)
    {
        if (!nodeInfo.BufferInfoMap.TryGetValue(buffer, out var bufferInfo))
        {
            return;
        }

        for (var loopEntry = bufferInfo.Places.Length - 1; loopEntry >= 0; loopEntry--)
        {
            for (var storageLevel = bufferInfo.Places[loopEntry].Length - 1; storageLevel >= 0; storageLevel--)
            {
                result.Add(new(node, buffer, loopEntry, storageLevel));
            }
        }
    }
}

internal sealed record VisibleTileBufferPlacements(
    ImmutableArray<TileBufferPlacement> Placements,
    BufferIdentity RootEndpoint);
