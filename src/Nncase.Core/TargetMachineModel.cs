// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.TIR;

namespace Nncase;

/// <summary>
/// Execution model used by one physical block.
/// </summary>
public enum BlockExecutionKind
{
    /// <summary>
    /// One logical block is executed by one core-bound CPU worker.
    /// </summary>
    CpuCore,

    /// <summary>
    /// One logical block is executed by one persistent GPU CTA.
    /// </summary>
    PersistentGpuBlock,
}

/// <summary>
/// Physical sharing scope of a memory space.
/// </summary>
public enum MemorySharingScope
{
    Block,
    Die,
    Chip,
}

/// <summary>
/// Physical kind of a target memory space.
/// </summary>
public enum TargetMemorySpaceKind
{
    Cache,
    Shared,
    Global,
}

/// <summary>
/// Size constraint imposed by the lowering used to allocate one memory arena.
/// </summary>
public enum TargetMemoryAllocationSizePolicy
{
    /// <summary>
    /// Round the requested arena size to the resource allocation granularity.
    /// </summary>
    GranularityAligned,

    /// <summary>
    /// Round the whole arena to a power of two. Triton tensor-backed local
    /// allocations currently impose this constraint on every shape dimension.
    /// </summary>
    PowerOfTwo,
}

/// <summary>
/// Semantics of an adjacent memory-hierarchy transfer.
/// </summary>
public enum TargetMemoryTransferMode
{
    /// <summary>
    /// The child scope directly addresses the same physical storage as its
    /// parent. Lowering creates a logical view and does not emit a copy.
    /// </summary>
    DirectAccess,

    /// <summary>
    /// The child scope owns distinct physical storage. Lowering must allocate
    /// a local tile and emit explicit load/store TIR at the boundary.
    /// </summary>
    ExplicitCopy,
}

/// <summary>
/// Target options exposing one fully-resolved machine model.
/// </summary>
public interface ITargetMachineModelProvider
{
    TargetMachineModel TargetMachineModel { get; }
}

/// <summary>
/// Stable identity of one target memory space.
/// </summary>
/// <param name="Value">Target-defined identity.</param>
public readonly record struct TargetMemorySpaceId(string Value)
{
    public override string ToString() => Value;
}

/// <summary>
/// Stable identity of one physical memory resource. Multiple logical storage
/// spaces may share a resource and therefore contend for the same bandwidth.
/// </summary>
/// <param name="Value">Target-defined identity.</param>
public readonly record struct TargetMemoryResourceId(string Value)
{
    public override string ToString() => Value;
}

/// <summary>
/// Stable identity of a backend-private block resource. These resources are
/// visible to scheduling but never become TIR memory locations.
/// </summary>
public readonly record struct TargetPrivateResourceId(string Value)
{
    public override string ToString() => Value;
}

public enum TargetPrivateResourceUnit
{
    Bytes,
    Register32,
}

/// <summary>
/// A block-scoped resource owned by backend microkernel lowering.
/// </summary>
public sealed record TargetPrivateResourceSpec
{
    public TargetPrivateResourceSpec(
        TargetPrivateResourceId id,
        TargetPrivateResourceUnit unit,
        long capacityUnits,
        int allocationGranularityUnits = 1,
        TargetMemoryResourceId? backingMemoryResource = null)
    {
        if (string.IsNullOrWhiteSpace(id.Value))
        {
            throw new ArgumentException("Target-private resource identity must not be empty.", nameof(id));
        }

        if (capacityUnits <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityUnits), capacityUnits, "Target-private resource capacity must be positive.");
        }

        if (allocationGranularityUnits <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(allocationGranularityUnits), allocationGranularityUnits, "Target-private resource allocation granularity must be positive.");
        }

        if (backingMemoryResource is not null && unit != TargetPrivateResourceUnit.Bytes)
        {
            throw new ArgumentException("A target-private resource backed by a memory resource must use byte units.", nameof(unit));
        }

        Id = id;
        Unit = unit;
        CapacityUnits = capacityUnits;
        AllocationGranularityUnits = allocationGranularityUnits;
        BackingMemoryResource = backingMemoryResource;
    }

    public TargetPrivateResourceId Id { get; }

    public TargetPrivateResourceUnit Unit { get; }

    public long CapacityUnits { get; }

    public int AllocationGranularityUnits { get; }

    public TargetMemoryResourceId? BackingMemoryResource { get; }
}

/// <summary>
/// Binds a physical target memory space to the existing TIR buffer representation.
/// </summary>
/// <param name="Location">TIR storage class.</param>
/// <param name="Hierarchy">Storage level within that class.</param>
public readonly record struct TIRMemorySpaceBinding(MemoryLocation Location, int Hierarchy = 0);

/// <summary>
/// One physical memory space visible to a block.
/// </summary>
public sealed record TargetMemoryResourceSpec
{
    public TargetMemoryResourceSpec(
        TargetMemoryResourceId id,
        TargetMemorySpaceKind kind,
        long capacityBytes,
        long readBytesPerCycle,
        long writeBytesPerCycle,
        long latencyCycles,
        int allocationGranularityBytes = 1)
    {
        if (string.IsNullOrWhiteSpace(id.Value))
        {
            throw new ArgumentException("Target memory space identity must not be empty.", nameof(id));
        }

        if (capacityBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityBytes), capacityBytes, "Target memory capacity must be positive.");
        }

        if (readBytesPerCycle <= 0 || writeBytesPerCycle <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(readBytesPerCycle), "Target memory bandwidth must be positive.");
        }

        if (latencyCycles < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(latencyCycles), latencyCycles, "Target memory latency must not be negative.");
        }

        if (allocationGranularityBytes <= 0 || !System.Numerics.BitOperations.IsPow2((uint)allocationGranularityBytes))
        {
            throw new ArgumentOutOfRangeException(nameof(allocationGranularityBytes), allocationGranularityBytes, "Allocation granularity must be a positive power of two.");
        }

        Id = id;
        Kind = kind;
        CapacityBytes = capacityBytes;
        ReadBytesPerCycle = readBytesPerCycle;
        WriteBytesPerCycle = writeBytesPerCycle;
        LatencyCycles = latencyCycles;
        AllocationGranularityBytes = allocationGranularityBytes;
    }

    public TargetMemoryResourceId Id { get; }

    public TargetMemorySpaceKind Kind { get; }

    public long CapacityBytes { get; }

    public long ReadBytesPerCycle { get; }

    public long WriteBytesPerCycle { get; }

    public long LatencyCycles { get; }

    public int AllocationGranularityBytes { get; }
}

/// <summary>
/// One logical storage space used by scheduling and TIR lowering.
/// </summary>
public sealed record TargetMemorySpaceSpec
{
    public TargetMemorySpaceSpec(
        TargetMemorySpaceId id,
        TargetMemoryResourceId resourceId,
        MemorySharingScope scope,
        TIRMemorySpaceBinding? tirBinding,
        long maxAllocationBytesPerScope,
        TargetMemoryAllocationSizePolicy allocationSizePolicy,
        bool isTilingCandidate,
        int tilingLevel,
        bool isAddressable,
        bool supportsDynamicIndexing,
        bool requiresExplicitSynchronization)
    {
        if (string.IsNullOrWhiteSpace(id.Value))
        {
            throw new ArgumentException("Target memory space identity must not be empty.", nameof(id));
        }

        if (string.IsNullOrWhiteSpace(resourceId.Value))
        {
            throw new ArgumentException("Target memory resource identity must not be empty.", nameof(resourceId));
        }

        if (maxAllocationBytesPerScope <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxAllocationBytesPerScope), maxAllocationBytesPerScope, "Storage allocation limit must be positive.");
        }

        if (isTilingCandidate && tirBinding is null)
        {
            throw new ArgumentException($"Tiling memory space {id} must have a TIR binding.", nameof(tirBinding));
        }

        if ((isTilingCandidate && tilingLevel < 0) || (!isTilingCandidate && tilingLevel != -1))
        {
            throw new ArgumentOutOfRangeException(nameof(tilingLevel), tilingLevel, "Tiling candidates require a non-negative level; fixed spaces require level -1.");
        }

        Id = id;
        ResourceId = resourceId;
        Scope = scope;
        TIRBinding = tirBinding;
        MaxAllocationBytesPerScope = maxAllocationBytesPerScope;
        AllocationSizePolicy = allocationSizePolicy;
        IsTilingCandidate = isTilingCandidate;
        TilingLevel = tilingLevel;
        IsAddressable = isAddressable;
        SupportsDynamicIndexing = supportsDynamicIndexing;
        RequiresExplicitSynchronization = requiresExplicitSynchronization;
    }

    public TargetMemorySpaceId Id { get; }

    public TargetMemoryResourceId ResourceId { get; }

    public MemorySharingScope Scope { get; }

    public TIRMemorySpaceBinding? TIRBinding { get; }

    public long MaxAllocationBytesPerScope { get; }

    public TargetMemoryAllocationSizePolicy AllocationSizePolicy { get; }

    public bool IsTilingCandidate { get; }

    public int TilingLevel { get; }

    public bool IsAddressable { get; }

    public bool SupportsDynamicIndexing { get; }

    public bool RequiresExplicitSynchronization { get; }
}

/// <summary>
/// Directed transfer path between two physical memory spaces.
/// </summary>
public sealed record TargetMemoryTransferSpec(
    TargetMemorySpaceId Source,
    TargetMemorySpaceId Destination,
    long BytesPerCycle,
    long LatencyCycles,
    TargetMemoryTransferMode Mode,
    bool RequiresSynchronization = false);

/// <summary>
/// Fixed execution resources for one block.
/// </summary>
public sealed record BlockExecutionSpec(
    BlockExecutionKind Kind,
    int ComputeUnitCount,
    int WorkersPerBlock,
    int WorkerWidth,
    double ClockRateGHz,
    int VectorWidthBits,
    int MatMulNr)
{
    public int ThreadsPerBlock => checked(WorkersPerBlock * WorkerWidth);
}

/// <summary>
/// Matrix instruction primitive exposed by a target.
/// </summary>
public sealed record MatrixComputePrimitiveSpec(
    string Name,
    int M,
    int N,
    int K,
    double InstructionsPerCyclePerBlock,
    ImmutableArray<DataType> SupportedOperandDataTypes,
    bool IsSupported = true)
{
    public bool Supports(DataType lhs, DataType rhs)
    {
        lhs = GetScalarDataType(lhs);
        rhs = GetScalarDataType(rhs);
        return IsSupported
            && lhs == rhs
            && SupportedOperandDataTypes.Contains(lhs);
    }

    private static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vectorType
            ? GetScalarDataType(vectorType.ElemType)
            : dataType;
}

/// <summary>
/// Per-block compute throughput.
/// </summary>
public sealed record BlockComputeSpec(
    double ElementwiseElementsPerCycle,
    double SimtFmaPerCycle,
    ImmutableArray<MatrixComputePrimitiveSpec> MatrixPrimitives);

/// <summary>
/// Target synchronization cost in target clock cycles.
/// </summary>
public sealed record TargetSynchronizationSpec(long BlockCycles, long GridCycles);

/// <summary>
/// Immutable, resolved hardware model shared by target cost models and AutoTiling.
/// </summary>
public sealed class TargetMachineModel
{
    private readonly ImmutableDictionary<TargetMemoryResourceId, TargetMemoryResourceSpec> _memoryResources;
    private readonly ImmutableDictionary<TargetPrivateResourceId, TargetPrivateResourceSpec> _privateResources;
    private readonly ImmutableDictionary<TargetMemorySpaceId, TargetMemorySpaceSpec> _memorySpaces;
    private readonly ImmutableDictionary<MemoryLocation, TargetMemorySpaceId> _fixedBindings;
    private readonly ImmutableDictionary<(TargetMemorySpaceId Source, TargetMemorySpaceId Destination), TargetMemoryTransferSpec> _transfers;

    public TargetMachineModel(
        string id,
        BlockExecutionSpec execution,
        BlockComputeSpec compute,
        TargetSynchronizationSpec synchronization,
        IEnumerable<TargetPrivateResourceSpec> privateResources,
        IEnumerable<TargetMemoryResourceSpec> memoryResources,
        IEnumerable<TargetMemorySpaceSpec> memorySpaces,
        TargetMemorySpaceId rootMemorySpace,
        IEnumerable<TargetMemoryTransferSpec> transfers,
        IReadOnlyDictionary<MemoryLocation, TargetMemorySpaceId> fixedBindings)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            throw new ArgumentException("Target machine identity must not be empty.", nameof(id));
        }

        if (execution.ComputeUnitCount <= 0 || execution.WorkersPerBlock <= 0 || execution.WorkerWidth <= 0 || !double.IsFinite(execution.ClockRateGHz) || execution.ClockRateGHz <= 0)
        {
            throw new ArgumentException("Target block execution resources must be positive.", nameof(execution));
        }

        if (execution.VectorWidthBits <= 0 || execution.MatMulNr <= 0)
        {
            throw new ArgumentException("Target vector width and MatMul Nr must be positive.", nameof(execution));
        }

        if (!double.IsFinite(compute.ElementwiseElementsPerCycle) || compute.ElementwiseElementsPerCycle <= 0 ||
            !double.IsFinite(compute.SimtFmaPerCycle) || compute.SimtFmaPerCycle <= 0)
        {
            throw new ArgumentException("Target compute throughput must be positive.", nameof(compute));
        }

        foreach (var primitive in compute.MatrixPrimitives)
        {
            if (string.IsNullOrWhiteSpace(primitive.Name) || primitive.M <= 0 || primitive.N <= 0 || primitive.K <= 0 ||
                !double.IsFinite(primitive.InstructionsPerCyclePerBlock) || primitive.InstructionsPerCyclePerBlock <= 0)
            {
                throw new ArgumentException("Target matrix primitives require a name, positive dimensions, and finite positive throughput.", nameof(compute));
            }

            if (primitive.SupportedOperandDataTypes.IsDefaultOrEmpty)
            {
                throw new ArgumentException($"Target matrix primitive {primitive.Name} must declare at least one operand data type.", nameof(compute));
            }
        }

        if (synchronization.BlockCycles < 0 || synchronization.GridCycles < 0)
        {
            throw new ArgumentException("Target synchronization cost must not be negative.", nameof(synchronization));
        }

        Id = id;
        Execution = execution;
        Compute = compute;
        Synchronization = synchronization;
        var privateResourceArray = privateResources.ToImmutableArray();
        if (privateResourceArray.IsDefaultOrEmpty)
        {
            throw new ArgumentException($"Target {id} must declare at least one target-private resource.", nameof(privateResources));
        }

        var duplicatePrivateResource = privateResourceArray
            .GroupBy(resource => resource.Id)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicatePrivateResource is not null)
        {
            throw new ArgumentException($"Target {id} declares duplicate target-private resource {duplicatePrivateResource.Key}.", nameof(privateResources));
        }

        _privateResources = privateResourceArray.ToImmutableDictionary(resource => resource.Id);
        var memoryResourceArray = memoryResources.ToImmutableArray();
        if (memoryResourceArray.IsDefaultOrEmpty)
        {
            throw new ArgumentException($"Target {id} must declare at least one physical memory resource.", nameof(memoryResources));
        }

        var duplicateMemoryResource = memoryResourceArray
            .GroupBy(resource => resource.Id)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateMemoryResource is not null)
        {
            throw new ArgumentException($"Target {id} declares duplicate memory resource {duplicateMemoryResource.Key}.", nameof(memoryResources));
        }

        _memoryResources = memoryResourceArray.ToImmutableDictionary(resource => resource.Id);
        foreach (var privateResource in privateResourceArray)
        {
            if (privateResource.BackingMemoryResource is { } backing && !_memoryResources.ContainsKey(backing))
            {
                throw new ArgumentException($"Target-private resource {privateResource.Id} references undeclared memory resource {backing}.", nameof(privateResources));
            }
        }

        var memorySpaceArray = memorySpaces.ToImmutableArray();
        if (memorySpaceArray.IsDefaultOrEmpty)
        {
            throw new ArgumentException($"Target {id} must declare at least one memory space.", nameof(memorySpaces));
        }

        var duplicateMemorySpace = memorySpaceArray
            .GroupBy(space => space.Id)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateMemorySpace is not null)
        {
            throw new ArgumentException($"Target {id} declares duplicate memory space {duplicateMemorySpace.Key}.", nameof(memorySpaces));
        }

        _memorySpaces = memorySpaceArray.ToImmutableDictionary(space => space.Id);
        foreach (var memorySpace in memorySpaceArray)
        {
            if (!_memoryResources.ContainsKey(memorySpace.ResourceId))
            {
                throw new ArgumentException($"Memory space {memorySpace.Id} references undeclared resource {memorySpace.ResourceId}.", nameof(memorySpaces));
            }

            if (memorySpace.MaxAllocationBytesPerScope > _memoryResources[memorySpace.ResourceId].CapacityBytes)
            {
                throw new ArgumentException($"Memory space {memorySpace.Id} allocation limit exceeds resource {memorySpace.ResourceId} capacity.", nameof(memorySpaces));
            }
        }

        if (!_memorySpaces.ContainsKey(rootMemorySpace))
        {
            throw new ArgumentException($"Root memory space {rootMemorySpace} is not declared by target {id}.", nameof(rootMemorySpace));
        }

        if (_memorySpaces[rootMemorySpace].IsTilingCandidate)
        {
            throw new ArgumentException($"Root memory space {rootMemorySpace} cannot be a tile-local placement candidate.", nameof(rootMemorySpace));
        }

        if (_memorySpaces[rootMemorySpace].Scope != MemorySharingScope.Chip || !_memorySpaces[rootMemorySpace].IsAddressable)
        {
            throw new ArgumentException($"Root memory space {rootMemorySpace} must be chip-scoped and addressable.", nameof(rootMemorySpace));
        }

        RootMemorySpace = rootMemorySpace;
        Transfers = transfers.ToImmutableArray();
        foreach (var transfer in Transfers)
        {
            if (!_memorySpaces.ContainsKey(transfer.Source) || !_memorySpaces.ContainsKey(transfer.Destination))
            {
                throw new ArgumentException($"Transfer {transfer.Source}->{transfer.Destination} references an undeclared memory space.", nameof(transfers));
            }

            if (transfer.BytesPerCycle <= 0 || transfer.LatencyCycles < 0)
            {
                throw new ArgumentException($"Transfer {transfer.Source}->{transfer.Destination} has invalid bandwidth or latency.", nameof(transfers));
            }
        }

        var duplicateTransfer = Transfers
            .GroupBy(transfer => (transfer.Source, transfer.Destination))
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateTransfer is not null)
        {
            throw new ArgumentException($"Target {id} declares duplicate transfer {duplicateTransfer.Key.Source}->{duplicateTransfer.Key.Destination}.", nameof(transfers));
        }

        _transfers = Transfers.ToImmutableDictionary(transfer => (transfer.Source, transfer.Destination));

        _fixedBindings = fixedBindings.ToImmutableDictionary();
        foreach (var (location, memorySpace) in _fixedBindings)
        {
            if (!_memorySpaces.TryGetValue(memorySpace, out var space))
            {
                throw new ArgumentException($"TIR location {location} is bound to undeclared memory space {memorySpace}.", nameof(fixedBindings));
            }
        }

        TilingMemorySpaces = _memorySpaces.Values
            .Where(space => space.IsTilingCandidate)
            .OrderBy(space => space.TilingLevel)
            .ToImmutableArray();
        if (TilingMemorySpaces.Length == 0)
        {
            throw new ArgumentException($"Target {id} must expose at least one tile-local memory space.", nameof(memorySpaces));
        }

        foreach (var memorySpace in TilingMemorySpaces)
        {
            if (memorySpace.Scope != MemorySharingScope.Block)
            {
                throw new ArgumentException($"AutoTiling memory space {memorySpace.Id} must be block-scoped; chip/die placement belongs to AutoDistributed and TIR selection.", nameof(memorySpaces));
            }

            var outerLevel = memorySpace.TilingLevel + 1;
            var parentMemorySpace = outerLevel < TilingMemorySpaces.Length
                ? TilingMemorySpaces[outerLevel].Id
                : rootMemorySpace;
            if (!_transfers.TryGetValue((parentMemorySpace, memorySpace.Id), out var loadTransfer) ||
                !_transfers.TryGetValue((memorySpace.Id, parentMemorySpace), out var storeTransfer))
            {
                throw new ArgumentException($"AutoTiling memory space {memorySpace.Id} requires transfer paths to and from its parent memory {parentMemorySpace}.", nameof(transfers));
            }

            if (loadTransfer.Mode != storeTransfer.Mode)
            {
                throw new ArgumentException($"AutoTiling transfer {parentMemorySpace}<->{memorySpace.Id} must use one symmetric transfer mode.", nameof(transfers));
            }

            if (loadTransfer.Mode == TargetMemoryTransferMode.DirectAccess &&
                _memorySpaces[parentMemorySpace].ResourceId != memorySpace.ResourceId)
            {
                throw new ArgumentException($"Direct-access transfer {parentMemorySpace}<->{memorySpace.Id} must reference one physical memory resource.", nameof(transfers));
            }
        }

        var duplicateBindings = TilingMemorySpaces
            .GroupBy(space => space.TIRBinding)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateBindings is not null)
        {
            throw new ArgumentException($"Target {id} has duplicate TIR tiling binding {duplicateBindings.Key}.", nameof(memorySpaces));
        }

        var duplicateLevels = TilingMemorySpaces
            .GroupBy(space => space.TilingLevel)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateLevels is not null)
        {
            throw new ArgumentException($"Target {id} has duplicate tiling level {duplicateLevels.Key}.", nameof(memorySpaces));
        }

        if (!TilingMemorySpaces.Select(space => space.TilingLevel).SequenceEqual(Enumerable.Range(0, TilingMemorySpaces.Length)))
        {
            throw new ArgumentException($"Target {id} tiling levels must be dense and start at zero.", nameof(memorySpaces));
        }
    }

    public string Id { get; }

    public BlockExecutionSpec Execution { get; }

    public BlockComputeSpec Compute { get; }

    public TargetSynchronizationSpec Synchronization { get; }

    public TargetMemorySpaceId RootMemorySpace { get; }

    public ImmutableArray<TargetMemorySpaceSpec> TilingMemorySpaces { get; }

    public ImmutableArray<TargetMemoryTransferSpec> Transfers { get; }

    public IReadOnlyDictionary<TargetMemoryResourceId, TargetMemoryResourceSpec> MemoryResources => _memoryResources;

    public IReadOnlyDictionary<TargetPrivateResourceId, TargetPrivateResourceSpec> PrivateResources => _privateResources;

    public IReadOnlyDictionary<TargetMemorySpaceId, TargetMemorySpaceSpec> MemorySpaces => _memorySpaces;

    public TargetMemoryResourceSpec GetMemoryResource(TargetMemoryResourceId id)
        => _memoryResources.TryGetValue(id, out var resource)
            ? resource
            : throw new KeyNotFoundException($"Target {Id} does not define memory resource {id}.");

    public TargetPrivateResourceSpec GetPrivateResource(TargetPrivateResourceId id)
        => _privateResources.TryGetValue(id, out var resource)
            ? resource
            : throw new KeyNotFoundException($"Target {Id} does not define target-private resource {id}.");

    public TargetMemoryResourceSpec GetMemoryResource(TargetMemorySpaceSpec space)
        => GetMemoryResource(space.ResourceId);

    public TargetMemorySpaceSpec GetMemorySpace(TargetMemorySpaceId id)
        => _memorySpaces.TryGetValue(id, out var space)
            ? space
            : throw new KeyNotFoundException($"Target {Id} does not define memory space {id}.");

    public TargetMemorySpaceSpec GetFixedMemorySpace(MemoryLocation location)
        => _fixedBindings.TryGetValue(location, out var id)
            ? GetMemorySpace(id)
            : throw new KeyNotFoundException($"Target {Id} does not bind TIR memory location {location} to a physical memory space.");

    public TargetMemoryTransferSpec GetTransfer(TargetMemorySpaceId source, TargetMemorySpaceId destination)
        => _transfers.TryGetValue((source, destination), out var transfer)
            ? transfer
            : throw new KeyNotFoundException($"Target {Id} does not define memory transfer {source}->{destination}.");

    public TargetMemorySpaceSpec GetTilingParentMemorySpace(int tilingLevel)
    {
        if ((uint)tilingLevel >= (uint)TilingMemorySpaces.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(tilingLevel), tilingLevel, $"Target {Id} has {TilingMemorySpaces.Length} tiling levels.");
        }

        return tilingLevel + 1 < TilingMemorySpaces.Length
            ? TilingMemorySpaces[tilingLevel + 1]
            : GetMemorySpace(RootMemorySpace);
    }

    public bool RequiresExplicitTransfer(int tilingLevel)
    {
        var local = TilingMemorySpaces[tilingLevel];
        var parent = GetTilingParentMemorySpace(tilingLevel);
        return GetTransfer(parent.Id, local.Id).Mode == TargetMemoryTransferMode.ExplicitCopy;
    }

    public long GetAllocationSizeBytes(TargetMemorySpaceSpec space, long requestedBytes)
    {
        if (requestedBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(requestedBytes), requestedBytes, "Allocation size must not be negative.");
        }

        if (requestedBytes == 0)
        {
            return 0;
        }

        var granularity = GetMemoryResource(space).AllocationGranularityBytes;
        return space.AllocationSizePolicy switch
        {
            TargetMemoryAllocationSizePolicy.GranularityAligned => AlignUp(requestedBytes, granularity),
            TargetMemoryAllocationSizePolicy.PowerOfTwo => RoundUpPowerOfTwo(Math.Max(requestedBytes, granularity)),
            _ => throw new ArgumentOutOfRangeException(nameof(space), space.AllocationSizePolicy, "Unknown memory allocation size policy."),
        };
    }

    public long GetMaximumUsableAllocationBytes(TargetMemorySpaceSpec space)
    {
        var granularity = GetMemoryResource(space).AllocationGranularityBytes;
        return space.AllocationSizePolicy switch
        {
            TargetMemoryAllocationSizePolicy.GranularityAligned =>
                (space.MaxAllocationBytesPerScope / granularity) * granularity,
            TargetMemoryAllocationSizePolicy.PowerOfTwo =>
                checked((long)(1UL << System.Numerics.BitOperations.Log2((ulong)space.MaxAllocationBytesPerScope))),
            _ => throw new ArgumentOutOfRangeException(nameof(space), space.AllocationSizePolicy, "Unknown memory allocation size policy."),
        };
    }

    private static long AlignUp(long value, long alignment)
        => checked(((value + alignment - 1) / alignment) * alignment);

    private static long RoundUpPowerOfTwo(long value)
    {
        var rounded = System.Numerics.BitOperations.RoundUpToPowerOf2((ulong)value);
        if (rounded == 0 || rounded > long.MaxValue)
        {
            throw new OverflowException($"Allocation size {value} cannot be represented as a power of two.");
        }

        return checked((long)rounded);
    }
}
