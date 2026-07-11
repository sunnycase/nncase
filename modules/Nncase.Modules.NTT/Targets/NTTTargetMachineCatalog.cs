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
        var cache = new TargetMemorySpaceId("cpu.cache.l0");
        var root = new TargetMemorySpaceId("cpu.main-memory");
        return new TargetMachineModel(
            CpuGeneric,
            new(BlockExecutionKind.CpuCore, Math.Max(1, Environment.ProcessorCount), 1, 1, 3.0, 512, 4),
            new(16, 16, ImmutableArray<MatrixComputePrimitiveSpec>.Empty),
            new(25, 25_000),
            [
                new(cache, TargetMemorySpaceKind.Cache, MemorySharingScope.Block, new(MemoryLocation.Cache, 0), 512 * 1024, 64, 64, 4, true, 0, true, true, false, 64),
                new(root, TargetMemorySpaceKind.Global, MemorySharingScope.Chip, null, int.MaxValue, 8, 8, 120, false, -1, true, true, false, 64),
            ],
            root,
            [
                new(root, cache, 8, 120),
                new(cache, root, 8, 120),
            ],
            CreateRootBindings(root));
    }

    private static TargetMachineModel CreateCudaGeneric()
    {
        var cache = new TargetMemorySpaceId("cuda.block-cache");
        var root = new TargetMemorySpaceId("cuda.global");
        return new TargetMachineModel(
            CudaGeneric,
            new(BlockExecutionKind.PersistentGpuBlock, 108, 8, 32, 1.4, 128, 4),
            new(128, 64, ImmutableArray.Create(new MatrixComputePrimitiveSpec("mma", 16, 8, 16, 4, _tensorCoreOperandDataTypes))),
            new(25, 2200),
            [
                new(cache, TargetMemorySpaceKind.Cache, MemorySharingScope.Block, new(MemoryLocation.Cache, 0), 512 * 1024, 512, 512, 20, true, 0, true, true, true, 16),
                new(root, TargetMemorySpaceKind.Global, MemorySharingScope.Chip, null, int.MaxValue, 1110, 1110, 300, false, -1, true, true, false, 128),
            ],
            root,
            [
                new(root, cache, 512, 300),
                new(cache, root, 512, 300),
            ],
            CreateRootBindings(root));
    }

    private static TargetMachineModel CreateXpuGeneric()
    {
        var cache = new TargetMemorySpaceId("xpu.sram");
        var root = new TargetMemorySpaceId("xpu.main-memory");
        return new TargetMachineModel(
            XpuGeneric,
            new(BlockExecutionKind.CpuCore, 64, 1, 1, 1.0, 128, 4),
            new(16, 16, ImmutableArray<MatrixComputePrimitiveSpec>.Empty),
            new(25, 25_000),
            [
                new(cache, TargetMemorySpaceKind.Cache, MemorySharingScope.Block, new(MemoryLocation.Cache, 0), 256 * 1024, 64, 64, 4, true, 0, true, true, false, 128),
                new(root, TargetMemorySpaceKind.Global, MemorySharingScope.Chip, null, 144 * 1024 * 1024, 32, 32, 120, false, -1, true, true, false, 128),
            ],
            root,
            [
                new(root, cache, 32, 120),
                new(cache, root, 32, 120),
            ],
            CreateRootBindings(root));
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
        var register = new TargetMemorySpaceId("gpu.register");
        var shared = new TargetMemorySpaceId("gpu.shared");
        var root = new TargetMemorySpaceId("gpu.global");
        var globalSharedTransferBytesPerCycle = Math.Min(chipGlobalBytesPerCycle, sharedBytesPerCycle);
        var matrixPrimitives = ImmutableArray.Create(
            new MatrixComputePrimitiveSpec("mma", 16, 8, 16, 4, _tensorCoreOperandDataTypes),
            new MatrixComputePrimitiveSpec("wgmma", 64, 8, 16, 8, _tensorCoreOperandDataTypes, supportsWgmma));

        // Register storage is an SSA resource in Triton, so only its physical budget is modeled here.
        return new TargetMachineModel(
            id,
            new(BlockExecutionKind.PersistentGpuBlock, computeUnits, 8, 32, clockRateGHz, 128, 4),
            new(128, 64, matrixPrimitives),
            new(25, 2200),
            [
                new(register, TargetMemorySpaceKind.Register, MemorySharingScope.Block, new(MemoryLocation.Register), 256 * 1024, 4096, 4096, 1, false, -1, false, false, false, 4),
                new(shared, TargetMemorySpaceKind.Shared, MemorySharingScope.Block, new(MemoryLocation.Shared), sharedCapacityBytes, sharedBytesPerCycle, sharedBytesPerCycle, 20, true, 0, true, true, true, 16),
                new(root, TargetMemorySpaceKind.Global, MemorySharingScope.Chip, null, globalCapacityBytes, chipGlobalBytesPerCycle, chipGlobalBytesPerCycle, 300, false, -1, true, true, false, 128),
            ],
            root,
            [
                new(root, register, chipGlobalBytesPerCycle, 300),
                new(register, root, chipGlobalBytesPerCycle, 300),
                new(root, shared, globalSharedTransferBytesPerCycle, 300),
                new(shared, root, globalSharedTransferBytesPerCycle, 300),
                new(shared, register, sharedBytesPerCycle, 20),
                new(register, shared, sharedBytesPerCycle, 20),
            ],
            CreateRootBindings(root));
    }

    private static IReadOnlyDictionary<MemoryLocation, TargetMemorySpaceId> CreateRootBindings(TargetMemorySpaceId root)
        => new Dictionary<MemoryLocation, TargetMemorySpaceId>
        {
            [MemoryLocation.Input] = root,
            [MemoryLocation.Output] = root,
            [MemoryLocation.Rdata] = root,
            [MemoryLocation.ChipLocalRdata] = root,
            [MemoryLocation.BlockLocalRdata] = root,
            [MemoryLocation.Data] = root,
            [MemoryLocation.ChipLocalData] = root,
            [MemoryLocation.BlockLocalData] = root,
            [MemoryLocation.PrivateBase] = root,
        };
}
