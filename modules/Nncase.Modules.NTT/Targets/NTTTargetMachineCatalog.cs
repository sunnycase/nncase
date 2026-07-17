// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.TIR;

namespace Nncase.Targets;

/// <summary>
/// Canonical, reproducible target machine profiles for NTT-family targets.
/// </summary>
public static class NTTTargetMachineCatalog
{
    public const string CpuGeneric = "cpu-generic-avx512";
    public const string CudaGeneric = "cuda-generic";
    public const string XpuGeneric = "xpu-generic";
    public const string Rtx5060Ti16Gb = "nvidia-rtx5060-ti-16gb";
    public const string H800Sxm80Gb = "nvidia-h800-sxm-80gb";

    public static readonly TargetPrivateResourceId CpuBackendPrivateBytes = new("cpu.backend-private-bytes");
    public static readonly TargetPrivateResourceId XpuBackendPrivateBytes = new("xpu.backend-private-bytes");
    public static readonly TargetPrivateResourceId GpuRegisterFile = new("gpu.register-file-r32");
    public static readonly TargetPrivateResourceId GpuBackendSharedMemory = new("gpu.backend-shared-memory");

    private static readonly ImmutableArray<DataType> _tensorCoreOperandDataTypes =
    [
        DataTypes.Float16,
        DataTypes.BFloat16,
        DataTypes.Float8E4M3,
        DataTypes.Float8E5M2,
        DataTypes.Int8,
        DataTypes.Float32,
    ];

    private static readonly IReadOnlyDictionary<string, Func<TargetMachineModel>> _factories =
        new Dictionary<string, Func<TargetMachineModel>>(StringComparer.OrdinalIgnoreCase)
        {
            [CpuGeneric] = CreateCpuGeneric,
            [CudaGeneric] = CreateCudaGeneric,
            [XpuGeneric] = CreateXpuGeneric,
            [Rtx5060Ti16Gb] = CreateRtx5060Ti16Gb,
            [H800Sxm80Gb] = CreateH800Sxm80Gb,
            ["rtx5060"] = CreateRtx5060Ti16Gb,
            ["h800"] = CreateH800Sxm80Gb,
        };

    public static TargetMachineModel Resolve(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Target machine name must not be empty.", nameof(name));
        }

        return _factories.TryGetValue(name, out var factory)
            ? factory()
            : throw new NotSupportedException($"Unknown NTT target machine '{name}'. Available models: {string.Join(", ", _factories.Keys.Order(StringComparer.OrdinalIgnoreCase))}.");
    }

    private static TargetMachineModel CreateCpuGeneric()
    {
        const long blockLocalWorkspaceBytes = 64L * 1024 * 1024;
        var cacheResource = new TargetMemoryResourceId("cpu.cache");
        var memoryResource = new TargetMemoryResourceId("cpu.main-memory");
        var cache = new TargetMemorySpaceId("cpu.cache.l0");
        var blockGlobal = new TargetMemorySpaceId("cpu.block-local-main-memory");
        var root = new TargetMemorySpaceId("cpu.main-memory");
        return new TargetMachineModel(
            CpuGeneric,
            new(BlockExecutionKind.CpuCore, Math.Max(1, Environment.ProcessorCount), 1, 1, 3.0, 512, 4),
            new(16, 16, ImmutableArray<MatrixComputePrimitiveSpec>.Empty),
            new(25, 25_000),
            [new(CpuBackendPrivateBytes, TargetPrivateResourceUnit.Bytes, 64 * 1024, 4)],
            [
                new(cacheResource, TargetMemorySpaceKind.Cache, 512 * 1024, 64, 64, 4, 64),
                new(memoryResource, TargetMemorySpaceKind.Global, int.MaxValue, 8, 8, 120, 64),
            ],
            [
                new(cache, cacheResource, MemorySharingScope.Block, new(MemoryLocation.Cache, 0), 512 * 1024, TargetMemoryAllocationSizePolicy.GranularityAligned, true, 0, true, true, false),
                new(blockGlobal, memoryResource, MemorySharingScope.Block, new(MemoryLocation.BlockLocalData), blockLocalWorkspaceBytes, TargetMemoryAllocationSizePolicy.GranularityAligned, true, 1, true, true, false),
                new(root, memoryResource, MemorySharingScope.Chip, null, int.MaxValue, TargetMemoryAllocationSizePolicy.GranularityAligned, false, -1, true, true, false),
            ],
            root,
            [
                new(root, blockGlobal, 8, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, root, 8, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, cache, 8, 120, TargetMemoryTransferMode.ExplicitCopy),
                new(cache, blockGlobal, 8, 120, TargetMemoryTransferMode.ExplicitCopy),
            ],
            CreateBindings(root, blockGlobal));
    }

    private static TargetMachineModel CreateCudaGeneric()
    {
        const long blockLocalWorkspaceBytes = 64L * 1024 * 1024;
        const long sharedCapacityBytes = 512L * 1024;
        var sharedResource = new TargetMemoryResourceId("cuda.shared-memory");
        var globalResource = new TargetMemoryResourceId("cuda.global");
        var shared = new TargetMemorySpaceId("cuda.shared");
        var blockGlobal = new TargetMemorySpaceId("cuda.block-global");
        var root = new TargetMemorySpaceId("cuda.global");
        return new TargetMachineModel(
            CudaGeneric,
            new(BlockExecutionKind.PersistentGpuBlock, 108, 8, 32, 1.4, 128, 4),
            new(128, 64, ImmutableArray.Create(new MatrixComputePrimitiveSpec("mma", 16, 8, 16, 4, _tensorCoreOperandDataTypes))),
            new(25, 2200),
            [
                new(GpuRegisterFile, TargetPrivateResourceUnit.Register32, 255L * 8 * 32, 8 * 32),
                new(GpuBackendSharedMemory, TargetPrivateResourceUnit.Bytes, sharedCapacityBytes, 16, sharedResource),
            ],
            [
                new(sharedResource, TargetMemorySpaceKind.Shared, sharedCapacityBytes, 512, 512, 20, 16),
                new(globalResource, TargetMemorySpaceKind.Global, int.MaxValue, 1110, 1110, 300, 128),
            ],
            [
                new(shared, sharedResource, MemorySharingScope.Block, new(MemoryLocation.Shared), GetCompilerManagedSharedAllocationLimit(sharedCapacityBytes), TargetMemoryAllocationSizePolicy.PowerOfTwo, true, 0, true, true, true),
                new(blockGlobal, globalResource, MemorySharingScope.Block, new(MemoryLocation.BlockLocalData), blockLocalWorkspaceBytes, TargetMemoryAllocationSizePolicy.GranularityAligned, true, 1, true, true, true),
                new(root, globalResource, MemorySharingScope.Chip, null, int.MaxValue, TargetMemoryAllocationSizePolicy.GranularityAligned, false, -1, true, true, false),
            ],
            root,
            [
                new(root, blockGlobal, 1110, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, root, 1110, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, shared, 512, 300, TargetMemoryTransferMode.ExplicitCopy),
                new(shared, blockGlobal, 512, 300, TargetMemoryTransferMode.ExplicitCopy),
            ],
            CreateBindings(root, blockGlobal));
    }

    private static TargetMachineModel CreateXpuGeneric()
    {
        const long blockLocalWorkspaceBytes = 16L * 1024 * 1024;
        var sramResource = new TargetMemoryResourceId("xpu.sram");
        var memoryResource = new TargetMemoryResourceId("xpu.main-memory");
        var cache = new TargetMemorySpaceId("xpu.sram");
        var blockGlobal = new TargetMemorySpaceId("xpu.block-local-main-memory");
        var root = new TargetMemorySpaceId("xpu.main-memory");
        return new TargetMachineModel(
            XpuGeneric,
            new(BlockExecutionKind.CpuCore, 64, 1, 1, 1.0, 128, 4),
            new(16, 16, ImmutableArray<MatrixComputePrimitiveSpec>.Empty),
            new(25, 25_000),
            [new(XpuBackendPrivateBytes, TargetPrivateResourceUnit.Bytes, 64 * 1024, 4)],
            [
                new(sramResource, TargetMemorySpaceKind.Cache, 256 * 1024, 64, 64, 4, 128),
                new(memoryResource, TargetMemorySpaceKind.Global, 144 * 1024 * 1024, 32, 32, 120, 128),
            ],
            [
                new(cache, sramResource, MemorySharingScope.Block, new(MemoryLocation.Cache, 0), 256 * 1024, TargetMemoryAllocationSizePolicy.GranularityAligned, true, 0, true, true, false),
                new(blockGlobal, memoryResource, MemorySharingScope.Block, new(MemoryLocation.BlockLocalData), blockLocalWorkspaceBytes, TargetMemoryAllocationSizePolicy.GranularityAligned, true, 1, true, true, false),
                new(root, memoryResource, MemorySharingScope.Chip, null, 144 * 1024 * 1024, TargetMemoryAllocationSizePolicy.GranularityAligned, false, -1, true, true, false),
            ],
            root,
            [
                new(root, blockGlobal, 32, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, root, 32, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, cache, 32, 120, TargetMemoryTransferMode.ExplicitCopy),
                new(cache, blockGlobal, 32, 120, TargetMemoryTransferMode.ExplicitCopy),
            ],
            CreateBindings(root, blockGlobal));
    }

    private static TargetMachineModel CreateRtx5060Ti16Gb()
        => CreatePersistentNvidiaGpu(
            Rtx5060Ti16Gb,
            computeUnits: 36,
            clockRateGHz: 2.57,
            globalCapacityBytes: 16L * 1024 * 1024 * 1024,
            chipGlobalBytesPerCycle: 174,
            sharedCapacityBytes: 101_376,
            sharedBytesPerCycle: 512,
            supportsWgmma: false);

    private static TargetMachineModel CreateH800Sxm80Gb()
        => CreatePersistentNvidiaGpu(
            H800Sxm80Gb,
            computeUnits: 114,
            clockRateGHz: 1.755,
            globalCapacityBytes: 80L * 1024 * 1024 * 1024,
            chipGlobalBytesPerCycle: 1908,
            sharedCapacityBytes: 227 * 1024,
            sharedBytesPerCycle: 1024,
            supportsWgmma: true);

    private static TargetMachineModel CreatePersistentNvidiaGpu(
        string id,
        int computeUnits,
        double clockRateGHz,
        long globalCapacityBytes,
        long chipGlobalBytesPerCycle,
        long sharedCapacityBytes,
        long sharedBytesPerCycle,
        bool supportsWgmma)
    {
        const long blockLocalWorkspaceBytes = 64L * 1024 * 1024;
        var sharedResource = new TargetMemoryResourceId("gpu.shared-memory");
        var globalResource = new TargetMemoryResourceId("gpu.global-memory");
        var shared = new TargetMemorySpaceId("gpu.shared");
        var blockGlobal = new TargetMemorySpaceId("gpu.block-global");
        var root = new TargetMemorySpaceId("gpu.global");
        var globalSharedBytesPerCycle = Math.Min(chipGlobalBytesPerCycle, sharedBytesPerCycle);
        var compilerManagedSharedBytes = GetCompilerManagedSharedAllocationLimit(sharedCapacityBytes);
        var matrixPrimitives = ImmutableArray.Create(
            new MatrixComputePrimitiveSpec("mma", 16, 8, 16, 4, _tensorCoreOperandDataTypes),
            new MatrixComputePrimitiveSpec("wgmma", 64, 8, 16, 8, _tensorCoreOperandDataTypes, supportsWgmma));

        return new TargetMachineModel(
            id,
            new(BlockExecutionKind.PersistentGpuBlock, computeUnits, 8, 32, clockRateGHz, 128, 4),
            new(128, 64, matrixPrimitives),
            new(25, 2200),
            [
                new(GpuRegisterFile, TargetPrivateResourceUnit.Register32, 255L * 8 * 32, 8 * 32),
                new(GpuBackendSharedMemory, TargetPrivateResourceUnit.Bytes, sharedCapacityBytes, 16, sharedResource),
            ],
            [
                new(sharedResource, TargetMemorySpaceKind.Shared, sharedCapacityBytes, sharedBytesPerCycle, sharedBytesPerCycle, 20, 16),
                new(globalResource, TargetMemorySpaceKind.Global, globalCapacityBytes, chipGlobalBytesPerCycle, chipGlobalBytesPerCycle, 300, 128),
            ],
            [
                new(shared, sharedResource, MemorySharingScope.Block, new(MemoryLocation.Shared), compilerManagedSharedBytes, TargetMemoryAllocationSizePolicy.PowerOfTwo, true, 0, true, true, true),
                new(blockGlobal, globalResource, MemorySharingScope.Block, new(MemoryLocation.BlockLocalData), blockLocalWorkspaceBytes, TargetMemoryAllocationSizePolicy.GranularityAligned, true, 1, true, true, true),
                new(root, globalResource, MemorySharingScope.Chip, null, globalCapacityBytes, TargetMemoryAllocationSizePolicy.GranularityAligned, false, -1, true, true, false),
            ],
            root,
            [
                new(root, blockGlobal, chipGlobalBytesPerCycle, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, root, chipGlobalBytesPerCycle, 0, TargetMemoryTransferMode.DirectAccess),
                new(blockGlobal, shared, globalSharedBytesPerCycle, 300, TargetMemoryTransferMode.ExplicitCopy),
                new(shared, blockGlobal, globalSharedBytesPerCycle, 300, TargetMemoryTransferMode.ExplicitCopy),
            ],
            CreateBindings(root, blockGlobal));
    }

    private static long GetCompilerManagedSharedAllocationLimit(long physicalCapacityBytes)
    {
        // Keep one third of physical shared memory available for backend-private
        // state. GraphTiler separately constrains the compiler arena and every
        // backed private resource against the full physical capacity.
        var budget = checked((physicalCapacityBytes * 2) / 3);
        if (budget <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(physicalCapacityBytes), physicalCapacityBytes, "Physical shared-memory capacity is too small.");
        }

        return checked((long)(1UL << System.Numerics.BitOperations.Log2((ulong)budget)));
    }

    private static IReadOnlyDictionary<MemoryLocation, TargetMemorySpaceId> CreateBindings(TargetMemorySpaceId root, TargetMemorySpaceId blockGlobal)
        => new Dictionary<MemoryLocation, TargetMemorySpaceId>
        {
            [MemoryLocation.Input] = root,
            [MemoryLocation.Output] = root,
            [MemoryLocation.Rdata] = root,
            [MemoryLocation.ChipLocalRdata] = root,
            [MemoryLocation.BlockLocalRdata] = blockGlobal,
            [MemoryLocation.Data] = root,
            [MemoryLocation.ChipLocalData] = root,
            [MemoryLocation.BlockLocalData] = blockGlobal,
            [MemoryLocation.PrivateBase] = root,
        };
}
