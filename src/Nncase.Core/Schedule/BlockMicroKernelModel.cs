// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;

namespace Nncase.Schedule;

/// <summary>
/// Target-owned model that maps one semantic block tile to legal backend
/// microkernel implementations. The model stops at block scope: warp, thread,
/// register layout, instruction selection, and accumulator lowering remain
/// backend-private. Compiler-visible loop scheduling is modeled independently
/// from these implementation candidates.
/// </summary>
public interface IBlockMicroKernelModelProvider
{
    IReadOnlyList<BlockMicroKernelCandidate> GetCandidates(BlockMicroKernelModelContext context);
}

/// <summary>
/// Symbolic information available while AutoTiling builds its global model.
/// </summary>
public sealed record BlockMicroKernelModelContext(
    Op Op,
    TileWorkload Workload,
    TileWorkloadContext WorkloadContext,
    IntExpr[][] LocalBufferShapes,
    IntExpr[][] FullBufferShapes,
    IntExpr BaseComputeCycles,
    TargetMachineModel Machine,
    Solver Solver)
{
    /// <summary>
    /// Gets the number of concurrently placed blocks that contend for
    /// chip-scoped services. Candidate-local compute and block-scoped memory
    /// remain per block; only a service backed by a chip-shared resource may
    /// use this scale.
    /// </summary>
    public long ChipActiveBlockCount { get; init; } = 1;
}

/// <summary>
/// One target-private resource consumption term.
/// </summary>
public sealed record BlockMicroKernelResourceUsage(
    TargetPrivateResourceId Resource,
    IntExpr Units);

/// <summary>
/// Execution-time traffic produced by one block microkernel after backend-local
/// reuse. This traffic is separate from TIR buffer materialization and transfer
/// traffic.
/// </summary>
/// <param name="BufferAccessIndex">
/// The semantic Grid access whose default execution traffic is replaced, or
/// <see langword="null"/> for backend-private traffic such as a shared-memory
/// accumulator.
/// </param>
/// <param name="Resource">The physical memory resource serving the access.</param>
/// <param name="Mode">Exactly one of <see cref="MemoryAccessMode.Read"/> or <see cref="MemoryAccessMode.Write"/>.</param>
/// <param name="Bytes">Total bytes accessed while evaluating the complete local shard.</param>
public sealed record BlockMicroKernelMemoryAccess(
    int? BufferAccessIndex,
    TargetMemoryResourceId Resource,
    MemoryAccessMode Mode,
    IntExpr Bytes);

/// <summary>
/// One numeric parameter selected together with a microkernel candidate.
/// </summary>
public sealed record BlockMicroKernelParameter(string Name, IntExpr Value);

/// <summary>
/// Physical storage encodings accepted by one semantic buffer access of a
/// block microkernel. The requirement refers to an explicit TIR
/// materialization; backend-private accumulator/register layouts are not
/// represented here.
/// </summary>
public sealed record BlockMicroKernelBufferEncodingRequirement(
    int BufferAccessIndex,
    TargetMemorySpaceId MemorySpace,
    ImmutableArray<TargetStorageEncodingId> AcceptedEncodings);

/// <summary>
/// A legal backend implementation for one semantic block tile.
/// </summary>
public sealed record BlockMicroKernelCandidate(
    string Family,
    string Variant,
    IntExpr IsLegal,
    BlockMicroKernelExecutionCost ExecutionCost,
    ImmutableArray<BlockMicroKernelResourceUsage> Resources,
    ImmutableArray<BlockMicroKernelMemoryAccess> MemoryAccesses,
    ImmutableArray<BlockMicroKernelParameter> Parameters)
{
    /// <summary>
    /// Gets the target-owned deterministic preference among candidates with
    /// identical predicted end-to-end latency. Lower values are preferred.
    /// This value is never interpreted as cycles and cannot outweigh a
    /// one-cycle improvement in the primary objective.
    /// </summary>
    public int SelectionPriority { get; init; }

    public ImmutableArray<BlockMicroKernelBufferEncodingRequirement> BufferEncodingRequirements { get; init; } = [];
}

/// <summary>
/// Concrete candidate selected by AutoTiling and carried through TIR to
/// backend code generation.
/// </summary>
public sealed record BlockMicroKernelSelection(
    string Family,
    string Variant,
    long RegionCycles,
    ImmutableDictionary<string, long> Resources,
    ImmutableDictionary<string, long> Parameters);
